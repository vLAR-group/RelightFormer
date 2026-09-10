import json
import math
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import kiui
import glob
from itertools import product

from torch.utils.data import Dataset

from .utils import (
    resize_tensor, generate_view_pairs, IndexDecomposer,
    apply_rotation_to_views, read_hdr
)
from diffsynth.pipelines.relightformer import camera_ray, equirectangular_ray

IS_ADDITION_ROTATION_FOR_ARGUMENT = True
IS_CROPPING_FOR_ARGUMENT = True
N_LIGHTINGS = 16  # MAX at 16
N_VIEWS = 16
DEPTH_SCALE_PNG = 1
MAX_EXAMPLES = 100_000_000


class LavalObjaverseDataset(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 object_split: str = 'training', 
                 lighting_split: str = 'training', 
                 view_split: str = 'training', 
                 source_view_num: int = 4,
                 target_view_num: int = 16, 
                 black_background: bool = True, 
                 resolution: int | Tuple[int, int] = 256, 
                 seed: int = 180,
                 is_train: bool = False,
                 **kwargs):
        super().__init__()
        
        self.data_dir = data_dir
        self.object_split = object_split
        self.lighting_split = lighting_split
        self.view_split = view_split
        self.source_view_num = source_view_num
        self.target_view_num = target_view_num
        self.black_background = black_background
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)
        self.seed = seed
        self.is_train = is_train
        self.scale = kwargs.pop("scale", 1.0)
        self.background = torch.tensor([0.0, 0.0, 0.0]) if black_background else torch.tensor([1.0, 1.0, 1.0])
        self.is_ablation = kwargs.pop("ablation", False)
        
        assert source_view_num >= 0, "source_view_num must be >= 0"
        assert target_view_num >= 0, "target_view_num must be >= 0"
        assert object_split in ['training', 'testing', 'validation']
        assert lighting_split in ['training', 'testing', 'validation']
        assert view_split in ['training', 'testing', 'validation']

        # Load objects
        if object_split == "training":
            self.objects = []
            json_files = glob.glob(os.path.join(self.data_dir, "info/objaverse", "training_subsets/subset_*.json"))
            for json_file in json_files:
                subset = os.path.splitext(os.path.basename(json_file))[0]
                if self.is_ablation and subset not in ["subset_0"]:
                    continue
                with open(json_file, 'r') as f:
                    content = json.load(f)
                for obj_uid in content:
                    self.objects.append(f"{subset}/{obj_uid}")
        else: 
            object_info_file = os.path.join(self.data_dir, "info/objaverse", f"full_{object_split}_objects.json")
            with open(object_info_file, 'r') as f:
                self.objects = json.load(f)

        self.lighting_index_mapping = list(product(range(0, N_LIGHTINGS), repeat=2))
        
        self.novel_view = kwargs.pop("novel_view", False)
        self.same_view = kwargs.pop("same_view", False)

        valid_view_num = source_view_num if (not is_train and self.same_view) else N_VIEWS
        self.view_index_mapping = generate_view_pairs(
            views=range(0, valid_view_num), 
            source_view_num=source_view_num, 
            target_view_num=target_view_num, 
            novel_view=self.novel_view, 
            same_view=self.same_view
        )
        
        self.decomposer = IndexDecomposer([
            self.length_of_objects, 
            self.length_of_lighting_mapping, 
            self.length_of_view_mapping
        ])
        self._rotation_call_count = 0

    @property
    def length_of_objects(self): return len(self.objects)
    
    @property
    def length_of_lighting_mapping(self): return len(self.lighting_index_mapping)
    
    @property
    def length_of_view_mapping(self): return len(self.view_index_mapping)
    
    def __len__(self): return MAX_EXAMPLES

    def __getitem__(self, idx: int):
        if self.is_train:
            object_index = random.randrange(self.length_of_objects)
            lighting_index = random.randrange(self.length_of_lighting_mapping)
            view_index = random.randrange(self.length_of_view_mapping)
        else:
            object_index, lighting_index, view_index = self.decomposer(idx)
        
        try:  
            return self._fetch_one_pair(
                object_name=self.objects[object_index],
                lighting_index_mapping=self.lighting_index_mapping[lighting_index],
                view_index_mapping=self.view_index_mapping[view_index]
            )
        except Exception as e:
            # Fallback to a random valid index to prevent training stalls
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def _fetch_one_pair(self, object_name: str, 
                        lighting_index_mapping: Optional[Tuple] = None, 
                        view_index_mapping: Optional[Tuple] = None,
                        source_lighting_name: Optional[str] = None,
                        target_lighting_name: Optional[str] = None,
                        source_view_name: Optional[list] = None,
                        target_view_name: Optional[list] = None,
                        view_crop_mapping: Optional[dict] = None,
                        **kwargs):
        
        rendered_path = os.path.join(self.data_dir, 'rendered', self.object_split, object_name)
        info_path = os.path.join(rendered_path, 'info.json')
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Required file not found: {info_path}")
        
        with open(info_path, 'r') as f:
            info = json.load(f)
            sensor_size = info['basic']['sensor_size']
            focal = info['basic']['focal']
            fov = 2 * math.atan(sensor_size[0] / (2 * focal))
            
            # Fallback to mapping if explicit names aren't provided (Eval dataset provides them directly)
            if source_lighting_name is None:
                lightings = info['basic']["lighting"][self.lighting_split]
                src_l_idx, tgt_l_idx = lighting_index_mapping
                source_lighting_name = lightings[src_l_idx]
                target_lighting_name = lightings[tgt_l_idx]
                
            if source_view_name is None:
                views = info['basic']["view"][self.view_split]
                src_v_idx, tgt_v_idx = view_index_mapping
                source_view_name = [views[i] for i in src_v_idx]
                target_view_name = [views[i] for i in tgt_v_idx]

        # ==========================================================
        # Early check availablity
        # ==========================================================
        missing_paths = []

        def check_png_exists(r_path: str, v_name: str, l_name: str) -> Optional[str]:
            v_clean = v_name.split('.')[0]
            l_clean = l_name.replace("/", "_").split('.')[0]
            image_file_name = f"{v_clean}&{l_clean}_image.png"
            
            full_path = os.path.join(r_path, image_file_name)
            if not os.path.exists(full_path):
                return full_path
            return None

        # 检查所有 source images
        for v in source_view_name:
            missing = check_png_exists(rendered_path, v, source_lighting_name)
            if missing:
                missing_paths.append(missing)

        # 检查所有 target images
        for v in target_view_name:
            missing = check_png_exists(rendered_path, v, target_lighting_name)
            if missing:
                missing_paths.append(missing)

        # 如果有任何缺失，立即抛出包含详细路径的错误
        if missing_paths:
            error_msg = f"Missing required .png files for object '{object_name}':\n"
            error_msg += "\n".join(f"  - {p}" for p in missing_paths)
            raise FileNotFoundError(error_msg)
        # ==========================================================

        def _get_crop_ratio(view_name: str) -> float:
            # if not IS_CROPPING_FOR_ARGUMENT:
                # return 1.0
            if view_crop_mapping is not None and view_name in view_crop_mapping:
                return view_crop_mapping[view_name]
            return torch.empty(1).uniform_(0.4, 1.0).item()

        def _fetch_lightings(lighting_name: str, addition_rotations=None):
            path = os.path.join(self.data_dir, 'laval/preprocessed', lighting_name)
            ldr, log, rays = self.read_environment(path, addition_rotations)
            lightings = torch.stack([ldr, log], dim=0)  # [2, 3, H, W]
            rays = rays.unsqueeze(1)  # [6, 1, H, W]
            return lightings, rays

        def _fetch_images(lighting_name: str, view_name_list: list):
            images, masks, Ks = [], [], []
            for view_name in view_name_list:
                view_name_clean = view_name.split('.')[0]
                lighting_name_clean = lighting_name.replace("/", "_").split('.')[0]
                image_file_name = f"{view_name_clean}&{lighting_name_clean}"
                
                image, mask = self.read_masked_image(rendered_path, image_file_name)
                
                _, H_orig, W_orig = image.shape
                crop_ratio = _get_crop_ratio(view_name)
                if IS_CROPPING_FOR_ARGUMENT:
                    H_crop = int(H_orig * crop_ratio)
                    W_crop = int(W_orig * crop_ratio)
                    top = (H_orig - H_crop) // 2
                    left = (W_orig - W_crop) // 2
                    
                    image = image[:, top:top+H_crop, left:left+W_crop]
                    mask = mask[:, top:top+H_crop, left:left+W_crop]

                # Resize using the extracted utility
                image = resize_tensor(image, self.resolution)
                mask = resize_tensor(mask, self.resolution, mode='nearest')

                W, H = self.resolution
                f_effective = (W / 2.0) / math.tan(fov / 2.0) / crop_ratio
                cx = (W - 1.0) / 2.0
                cy = (H - 1.0) / 2.0
                
                K = torch.tensor([
                    [f_effective, 0,           cx],
                    [0,           f_effective, cy],
                    [0,           0,           1]
                ], dtype=torch.float32)

                images.append(image)
                masks.append(mask)
                Ks.append(K)

            stacked_masks = torch.stack(masks, dim=0)
            return (
                torch.stack(images, dim=0),
                stacked_masks,
                torch.stack(Ks, dim=0)
            )

        def _fetch_view(view_name_list: list):
            blender_to_cv = torch.tensor([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, 0],
                [0,  0,  0, 1]
            ], dtype=torch.float32)

            views = []
            for view in view_name_list:
                for item in info["images"]:
                    if item['view'] == view:
                        c2w = torch.tensor(item['transform'], dtype=torch.float32)
                        views.append(c2w @ blender_to_cv)
                        break
            return torch.stack(views)

        addition_rotation = torch.tensor(
            self.get_random_rotation()[0].as_matrix(), dtype=torch.float32
        ) if (IS_ADDITION_ROTATION_FOR_ARGUMENT and self.is_train) else torch.eye(3, dtype=torch.float32)
        
        src_imgs, src_mask, src_Ks = _fetch_images(source_lighting_name, source_view_name)
        tgt_imgs, tgt_mask, tgt_Ks = _fetch_images(target_lighting_name, target_view_name)
        
        src_lighting, lighting_rays = _fetch_lightings(source_lighting_name, addition_rotation)
        tgt_lighting, _ = _fetch_lightings(target_lighting_name, addition_rotation)
        
        src_view = _fetch_view(source_view_name)
        tgt_view = _fetch_view(target_view_name)

        src_view = apply_rotation_to_views(src_view, addition_rotation)
        tgt_view = apply_rotation_to_views(tgt_view, addition_rotation)

        src_rays = camera_ray(src_view.unsqueeze(0), 
                              src_Ks.unsqueeze(0), 
                              H=self.resolution[1], 
                              W=self.resolution[0]).squeeze(0)
        tgt_rays = camera_ray(tgt_view.unsqueeze(0), 
                              tgt_Ks.unsqueeze(0), 
                              H=self.resolution[1], 
                              W=self.resolution[0]).squeeze(0)

        return {
            "source_lighting": src_lighting,
            "target_lighting": tgt_lighting,
            "source_images": src_imgs,
            "target_images": tgt_imgs,
            "source_rays": src_rays,
            "target_rays": tgt_rays,
            "lighting_rays": lighting_rays,
            "source_view": src_view,
            "target_view": tgt_view,
            "source_mask": src_mask,
            "target_mask": tgt_mask,
            "source_Ks": src_Ks,
            "target_Ks": tgt_Ks,
            "addition_rotation": addition_rotation,
        }

    def read_environment(self, path: str, addition_rotation=None):
        M_ldr, M_log = 16, 10_000
        raw = read_hdr(path, self.resolution)
        
        ldr = torch.from_numpy(raw / (1.0 + raw) * (1.0 + raw / M_ldr**2)).float()
        log = torch.from_numpy(np.log(1.0 + raw) / np.log(1.0 + M_log)).float()
        
        ldr = resize_tensor(ldr.permute(2, 0, 1).contiguous(), self.resolution)
        log = resize_tensor(log.permute(2, 0, 1).contiguous(), self.resolution)

        rays = equirectangular_ray(self.resolution[0], addition_rotation)
        return ldr, log, rays

    def read_masked_image(self, data_path: str, name: str):
        img_path = os.path.join(data_path, f"{name}_image.png")
        rgba = kiui.read_image(img_path, mode='tensor', order='RGBA')
        if rgba is None:
            raise FileNotFoundError(f"Not found: {img_path}")
            
        rgb = rgba[..., :3]
        mask = rgba[..., 3].unsqueeze(-1)
        
        image = rgb * mask + self.background * (1.0 - mask)
        return image.permute(2, 0, 1).contiguous(), mask.permute(2, 0, 1).contiguous()

    def get_random_rotation(self, batch_size: int = 1):
        if hasattr(self, 'seed') and self.seed is not None:
            current_seed = self.seed + self._rotation_call_count
            original_state = np.random.get_state()
            np.random.seed(current_seed)
            random_rot = Rotation.random(batch_size)
            np.random.set_state(original_state)
        else:
            random_rot = Rotation.random(batch_size)
        
        self._rotation_call_count += 1
        return random_rot


class LavalObjaverseEvalDataset(LavalObjaverseDataset):
    def __init__(self, 
                 data_dir: str, 
                 pair_info: str, 
                 black_background: bool = True, 
                 resolution: Tuple[int, int] = (256, 256), 
                 seed: int = 180,
                 **kwargs):
        
        object_split = kwargs.pop("object_split", "testing")
        
        # Initialize parent with evaluation-specific defaults
        super().__init__(
            data_dir=data_dir,
            object_split=object_split,
            lighting_split="testing",
            view_split="testing",
            source_view_num=16,
            target_view_num=16,
            black_background=black_background,
            resolution=resolution,
            seed=seed,
            is_train=False,
            **kwargs
        )
        
        self.pair_info = pair_info
        with open(self.pair_info, 'r') as f:
            self.data_pairs = json.load(f)

    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int):
        data_pair = self.data_pairs[idx]
        object_name = data_pair["object"]
        source_lighting = data_pair["source_lighting"]
        target_lighting = data_pair["target_lighting"]
        
        if "view" in data_pair:
            views = data_pair["view"]
        elif "source_view" in data_pair and "target_view" in data_pair:
            views = (data_pair["source_view"], data_pair["target_view"])
        else:
            # 如果兩者都沒有，拋出明確的錯誤方便你檢查 JSON 檔案
            raise KeyError(f"Missing 'view' or 'source_view'/'target_view' in data pair: {data_pair}")
        crop_ratios = data_pair.get("crop_ratio", [])
        
        if isinstance(views, tuple):
            source_view_name, target_view_name = views
        else:
            source_view_name = target_view_name = views
            
        # Build explicit crop mapping for evaluation reproducibility
        view_crop_mapping = {}
        
        for i, view in enumerate(source_view_name):
            view_crop_mapping[view] = crop_ratios[i] if len(crop_ratios) > 0 else 1.0

        try:
            item = self._fetch_one_pair(
                object_name=object_name,
                source_lighting_name=source_lighting,
                target_lighting_name=target_lighting,
                source_view_name=source_view_name,
                target_view_name=target_view_name,
                view_crop_mapping=view_crop_mapping
            )
            item["idx"] = idx
            return item
        except Exception as e:
            print(f"Fetch Error:\n Object: {object_name}\n Lighting: {target_lighting}\n Views: {views}\n Error: {e}")
            # Safe modulo fallback to prevent IndexError
            return self.__getitem__((idx + 1) % len(self))