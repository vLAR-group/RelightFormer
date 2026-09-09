import os
import json
import torch
import logging
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from collections import defaultdict
import kiui
from .utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StanfordORB_Eval")
FOCAL_DEFAULT = 50
SENSOR_SIZE = (36, 36)
LIGHTING_SCALE = 10

def meta_to_str(meta):
    res = "" 
    res += meta['obj_id']
    res += "-" + meta['source']
    res += "-" + meta['target']
    return res


class StanfordORBEval(Dataset):
    def __init__(self, data_dir, resolution=(512,512), black_background=True, scale=1.0, gaussian=False):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.black_background = black_background
        self.background = torch.tensor([0.0, 0.0, 0.0]).view(3, 1, 1) if black_background else torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
        self.scale = scale
        self.gaussian = gaussian

        self.blender_to_cv = torch.tensor([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=torch.float32)

        self.ldr_root = self.data_dir / "blender_LDR"
        self.gt_root = self.data_dir / "ground_truth"

        # Group scenes by Object ID
        obj_to_scenes = defaultdict(list)
        for d in self.ldr_root.iterdir():
            if d.is_dir() and "_scene" in d.name:
                obj_id = d.name.split("_scene")[0]
                obj_to_scenes[obj_id].append(d.name)

        self.eval_list = []
        for obj_id, scenes in obj_to_scenes.items():
            scenes = sorted(scenes)
            for src_scene in scenes:
                scene_path = self.ldr_root / src_scene
                json_path = scene_path / f"transforms_novel.json"
                with open(json_path, 'r') as f:
                    frames = json.load(f)["frames"]
                    target_scenes = []
                    for frame in frames:
                        if frame["scene_name"] not in target_scenes:
                            target_scenes.append(frame["scene_name"])
                for tgt_scene in target_scenes:
                    self.eval_list.append({
                        "obj_id": obj_id,
                        "source": src_scene,
                        "target": tgt_scene
                    })

    def __len__(self):
        return len(self.eval_list)
    
    def _load_lighting(self, capture):
        # load pose of the lighting
        pose_info_path = (self.ldr_root / capture / "transforms_test.json").resolve()
    
        # Open the file first, THEN load the JSON
        with open(pose_info_path, 'r') as f:
            pose_data = json.load(f)
            frame = pose_data["frames"][0]
            # TODO： should we rotated here?
        transforms = torch.tensor(frame["transform_matrix"], dtype=torch.float32) @ self.blender_to_cv
        lighting_path = self.gt_root / capture / "env_map" / Path(frame["file_path"].split('/')[-1]).with_suffix('.exr')
        
        # follow LuxDiT
        M_ldr = 16
        M_log = 10_000

        raw = read_hdr(lighting_path, self.resolution)
        # raw = raw * LIGHTING_SCALE
        ldr = raw / (1.0 + raw) * (1.0 + raw / M_ldr**2)
        log = np.log(1.0 + raw) / np.log(1.0 + M_log)

        # Convert to tensors and resize
        raw = torch.from_numpy(raw).float()
        ldr = torch.from_numpy(ldr).float()
        log = torch.from_numpy(log).float()
        
        raw = resize_tensor(raw.permute(2,0,1).contiguous(), self.resolution)
        raw = raw.unsqueeze(0)
        # ldr = resize_tensor(ldr.permute(2,0,1).contiguous(), self.resolution)
        # log = resize_tensor(log.permute(2,0,1).contiguous(), self.resolution)
        
        # rays = mercator2ray(self.resolution[0], self.resolution[1], transforms[:3, :3]).permute(2,0,1).contiguous()

        # lightings = torch.stack([ldr, log], dim=0)
        # rays = rays.unsqueeze(1)# [6, 1, H, W]
        return None, raw, None

    def _load_scene_data(self, capture, split='train'):
        scene_path = self.ldr_root / capture
        json_path = scene_path / f"transforms_{split}.json"
        
        with open(json_path, 'r') as f:
            meta = json.load(f)
        
        W, H = self.resolution
        

        # fx = 50 * 512 / 36
        # fy = fx

        # cx = (512 - 1) / 2.0
        # cy = cx

        # sensor_size = SENSOR_SIZE

        images = []
        masks = []
        frames_path = []
        Ts = []
        Ks = []

        for frame in meta['frames']:
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            c2w[:3, 3] = c2w[:3, 3] * self.scale
            Ts.append(c2w @ self.blender_to_cv)
            image_path = scene_path / Path(frame['file_path']).with_suffix('.png')
            image = kiui.read_image(str(image_path), mode='tensor', order='RGB')
            image = resize_tensor(image.permute(2,0,1).contiguous(), self.resolution)
            images.append(image)

            mask_path = scene_path / Path(frame['file_path'].replace(f"{split}", f"{split}_mask")).with_suffix('.png')
            mask = kiui.read_image(str(mask_path), mode='tensor', order='RGB').unsqueeze(-1)
            mask = resize_tensor(mask.permute(2,0,1).contiguous(), self.resolution, mode='nearest')
            masks.append(mask)

            frames_path.append(str(image_path.resolve()).replace("blender_LDR", "blender_HDR").replace(".png", ".exr"))

            _, W, H = image.size()
            fov_x = meta.get("camera_angle_x", None)
    
            fx = W / (2.0 * np.tan(fov_x / 2.0))
            fy = fx  # Assuming square pixels
            cx = (W - 1) / 2.0
            cy = (H - 1) / 2.0

            intrinsic = torch.tensor([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ], dtype=torch.float32)
            Ks.append(intrinsic)


        images = torch.stack(images)
        masks = torch.stack(masks)
        Ts = torch.stack(Ts)
        Ks = torch.stack(Ks)


        images_masked = images * masks + self.background * (1.0 - masks)
        depths = None

        # rays = camera2ray(Ts, Ks, H=self.resolution[1], W=self.resolution[0])

        return images_masked, images, masks, depths, Ts, None, Ks, frames_path
    
    def _load_novel_scene_data(self, capture, novel_capture):
        scene_path = self.ldr_root / capture
        json_path = scene_path / f"transforms_novel.json"

        novel_scene_path = self.ldr_root / novel_capture
        
        with open(json_path, 'r') as f:
            meta = json.load(f)
        
        W, H = self.resolution

        images = []
        masks = []
        depths = []
        frames_path = []
        Ts = []
        Ks = []
        for frame in meta['frames']:
            if frame["scene_name"] != novel_capture:
                continue
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            Ts.append(c2w @ self.blender_to_cv)
            image_path = novel_scene_path / Path(frame['file_path']).with_suffix('.png')
            image = kiui.read_image(str(image_path), mode='tensor', order='RGB')
            image = resize_tensor(image.permute(2,0,1).contiguous(), self.resolution)
            images.append(image)

            mask_path = novel_scene_path / Path(frame['file_path'].replace(f"test", f"test_mask")).with_suffix('.png')
            mask = kiui.read_image(str(mask_path), mode='tensor', order='RGB').unsqueeze(-1)
            mask = resize_tensor(mask.permute(2,0,1).contiguous(), self.resolution, mode='nearest')
            masks.append(mask)

            frames_path.append(str(image_path.resolve()).replace("blender_LDR", "blender_HDR").replace(".png", ".exr"))

            _, W, H = image.size()
            fov_x = frame.get("camera_angle_x", None)
    
            fx = W / (2.0 * np.tan(fov_x / 2.0))
            fy = fx  # Assuming square pixels
            cx = (W - 1) / 2.0
            cy = (H - 1) / 2.0

            intrinsic = torch.tensor([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ], dtype=torch.float32)
            Ks.append(intrinsic)

        images = torch.stack(images)
        masks = torch.stack(masks)
        Ts = torch.stack(Ts)
        # depths = torch.stack(depths)
        Ks = torch.stack(Ks)

        images_masked = images * masks + self.background * (1.0 - masks)
        # depths = depths * masks
        depths = None

        # rays = camera2ray(Ts, Ks, H=self.resolution[1], W=self.resolution[0])

        return images_masked, images, masks, depths, Ts, None, Ks, frames_path
    
    def _load_gaussian_scene_data(self, capture, novel_capture):
        scene_path = self.ldr_root / capture
        json_path = scene_path / f"transforms_novel.json"

        novel_scene_path = self.ldr_root / novel_capture
        gaussian_scene_path = scene_path / 'gaussian' / 'renders' / novel_capture
        
        with open(json_path, 'r') as f:
            meta = json.load(f)
        
        W, H = self.resolution

        images = []
        masks = []
        depths = []
        frames_path = []
        Ts = []
        Ks = []
        for frame in meta['frames']:
            if frame["scene_name"] != novel_capture:
                continue
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            Ts.append(c2w @ self.blender_to_cv)
            image_path = gaussian_scene_path / Path(frame['file_path']).with_suffix('.png')
            # print(image_path)
            image = kiui.read_image(str(image_path), mode='tensor', order='RGB')
            image = resize_tensor(image.permute(2,0,1).contiguous(), self.resolution)
            images.append(image)

            mask_path = novel_scene_path / Path(frame['file_path'].replace(f"test", f"test_mask")).with_suffix('.png')
            mask = kiui.read_image(str(mask_path), mode='tensor', order='RGB').unsqueeze(-1)
            mask = resize_tensor(mask.permute(2,0,1).contiguous(), self.resolution, mode='nearest')
            masks.append(mask)

            frames_path.append(str(image_path.resolve()))

            _, W, H = image.size()
            fov_x = frame.get("camera_angle_x", None)
    
            fx = W / (2.0 * np.tan(fov_x / 2.0))
            fy = fx  # Assuming square pixels
            cx = (W - 1) / 2.0
            cy = (H - 1) / 2.0

            intrinsic = torch.tensor([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ], dtype=torch.float32)
            Ks.append(intrinsic)

        images = torch.stack(images)
        masks = torch.stack(masks)
        Ts = torch.stack(Ts)
        # depths = torch.stack(depths)
        Ks = torch.stack(Ks)

        images_masked = images * masks + self.background * (1.0 - masks)
        # depths = depths * masks
        depths = None

        # rays = camera2ray(Ts, Ks, H=self.resolution[1], W=self.resolution[0])

        return images_masked, images, masks, depths, Ts, None, Ks, frames_path

    def _fetch_one_pair(self, item):
        _, source_lighting, _ = self._load_lighting(item['source'])
        _, target_lighting, lighting_rays = self._load_lighting(item['target'])

        if self.gaussian:
            source_images, \
            source_images_unmask, \
            source_mask, \
            source_depths, \
            source_view, source_rays, source_intrinsic, \
            source_frame_path \
                = self._load_gaussian_scene_data(item['source'], item['target'])
        else:
            source_images, \
            source_images_unmask, \
            source_mask, \
            source_depths, \
            source_view, source_rays, source_intrinsic, \
            source_frame_path \
                = self._load_scene_data(item['source'])
        
        target_images, \
        target_images_unmask, \
        target_mask, \
        target_depths, \
        target_view, target_rays, target_intrinsic, \
        target_frame_path \
            = self._load_novel_scene_data(item['source'], item['target'])

        # target_images, \
        # target_images_unmask, \
        # target_mask, \
        # target_depths, \
        # target_view, target_rays, target_intrinsic, \
        # target_frame_path \
        #     = self._load_scene_data(item['target'])
        
        # TODO:
        # target_images, target_images_unmask = source_images, source_images_unmask
        # target_mask, target_depths = source_mask, source_depths
        # target_view, target_intrinsic = source_view, source_intrinsic
        # target_frame_path = source_frame_path

        all_views = view_preprocess(2.0, torch.cat([source_view, target_view], dim=0))
        source_view, target_view = all_views[:source_view.size()[0]], all_views[source_view.size()[0]:]

        return {
                "source_lighting": source_lighting,
                "target_lighting": target_lighting,

                "source_images": source_images,
                "target_images": target_images,

                "source_images_unmask": source_images_unmask,
                "target_images_unmask": target_images_unmask,

                # "source_rays": source_rays,
                # "target_rays": target_rays,
                # "lighting_rays": lighting_rays,

                "source_view": source_view,
                "target_view": target_view,

                # "source_depths": source_depths,
                # "target_depths": target_depths,

                "source_mask": source_mask,
                "target_mask": target_mask,

                "source_Ks": source_intrinsic,
                "target_Ks": target_intrinsic,

                "meta": meta_to_str(item),
                
                "source_frame_path": source_frame_path,
                "target_frame_path": target_frame_path
            } 

    def __getitem__(self, idx):
        # try:
        item = self.eval_list[idx]
        
        data_pair = self._fetch_one_pair(item)
        data_pair['idx'] = idx

        return data_pair
        # except Exception as e:
        #     logger.error(f"Error index {idx}: {e}")
        #     return self.__getitem__((idx + 1) % len(self.eval_list))

def view_preprocess(scale, view):
    # 2. Normalize translations so the average length equals self.scale
    translations = view[:, :3, 3]  # Shape: (N, 3)
    avg_len = torch.norm(translations, dim=1).mean()  # Scalar
    
    # Global scaling factor (+1e-8 prevents division by zero)
    scale_factor = scale / (avg_len + 1e-8)
    
    # Apply scaling
    view[:, :3, 3] = translations * scale_factor

    return view

if __name__ == "__main__":
    path = "/media/HDD2/hejun/Stanford-ORB"
    dataset = StanfordORBEval(path)