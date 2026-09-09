import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

logger = logging.getLogger(__name__)


def fov2K(fov: float, width: int, height: int) -> torch.Tensor:
    """
    Convert horizontal field of view to camera intrinsics matrix K.
    
    Args:
        fov: Horizontal field of view in radians.
        width: Image width in pixels.
        height: Image height in pixels.
        
    Returns:
        3x3 Camera intrinsic matrix.
    """
    f_x = 0.5 * width / np.tan(0.5 * fov)
    f_y = f_x  # Assuming square pixels
    c_x = 0.5 * width
    c_y = 0.5 * height
    
    return torch.tensor([
        [f_x, 0.0, c_x],
        [0.0, f_y, c_y],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)


class TensoIREvalDataset(Dataset):
    """Evaluation dataset for TensorIR relighting tasks."""
    
    def __init__(self, 
                 data_dir: str, 
                 pair_info: SystemError, 
                 black_background: bool = True, 
                 resolution: Tuple[int, int] = (512, 512), 
                 lighting_resolution: Tuple[int, int] = (256, 256),
                 seed: int = 180,
                 **kwargs):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.pair_info = Path(pair_info)
        self.black_background = black_background
        self.resolution = resolution  # (H, W) for images
        self.lighting_resolution = lighting_resolution  # (H, W) for environment maps
        self.seed = seed
        self.scale = kwargs.pop("scale", 2.0)
        
        bg_color = [0.0, 0.0, 0.0] if black_background else [1.0, 1.0, 1.0]
        self.background = torch.tensor(bg_color, dtype=torch.float32).view(3, 1, 1)
        
        # Load data pairs
        if not self.pair_info.exists():
            logger.warning(f"Pair info file not found at {self.pair_info}. Dataset will be empty.")
            self.data_pairs: List[Dict[str, Any]] = []
        else:
            with open(self.pair_info, 'r') as f:
                self.data_pairs = json.load(f)

        # Transformation for images and masks
        self.transform = T.Compose([
            T.Resize(self.resolution, interpolation=Image.BICUBIC),
            T.ToTensor(),
        ])

    def view_preprocess(self, view: torch.Tensor) -> torch.Tensor:
        """
        Convert Blender camera coordinates to OpenCV and normalize translation scale.
        
        Args:
            view: Camera-to-world matrices of shape (N, 4, 4).
            
        Returns:
            Processed camera-to-world matrices of shape (N, 4, 4).
        """
        # 1. Blender (-Z forward, +Y up) -> OpenCV (+Z forward, -Y down)
        blender_to_cv = torch.tensor([
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0]
        ], dtype=view.dtype, device=view.device)
        
        view = view @ blender_to_cv
        
        # 2. Normalize translations so the average distance from origin equals self.scale
        translations = view[:, :3, 3]  # Shape: (N, 3)
        avg_len = torch.norm(translations, dim=1).mean()  # Scalar
        
        scale_factor = self.scale / (avg_len + 1e-8)
        view = view.clone()  # Avoid in-place modification of potentially shared tensors
        view[:, :3, 3] = translations * scale_factor
    
        return view

    def read_masked_image(self, img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads an RGBA image, resizes it, applies background color, and returns (image, mask).
        """
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = Image.open(img_path).convert('RGBA')
        img_tensor = self.transform(img)  # Shape: (4, H, W)
        
        image = img_tensor[:3, :, :]
        mask = img_tensor[3:4, :, :]
        
        # Composite with background
        image = image * mask + self.background * (1.0 - mask)
        
        return image, mask

    def read_lighting(self, light_path: Path) -> torch.Tensor:
        """
        Reads HDR lighting environment map and resizes it to lighting_resolution.
        """
        if not light_path.exists():
            logger.warning(f"HDR environment map not found: {light_path}. Returning zeros.")
            return torch.zeros((3, self.lighting_resolution[0], self.lighting_resolution[1]), dtype=torch.float32)
        
        # cv2.IMREAD_ANYDEPTH is required for .hdr files
        env_map = cv2.imread(str(light_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        
        if env_map is None:
            logger.warning(f"Failed to read HDR file with OpenCV: {light_path}. Returning zeros.")
            return torch.zeros((3, self.lighting_resolution[0], self.lighting_resolution[1]), dtype=torch.float32)
        
        # Resize to lighting_resolution (OpenCV expects (W, H))
        env_map = cv2.resize(
            env_map, 
            (self.lighting_resolution[1], self.lighting_resolution[0]), 
            interpolation=cv2.INTER_LINEAR
        )
        # BGR to RGB, then (H, W, C) -> (C, H, W)
        env_map = cv2.cvtColor(env_map, cv2.COLOR_BGR2RGB)
        env_map = torch.from_numpy(env_map).permute(2, 0, 1)
        # env_map = rescale_lighting_map_torch(env_map, target_mean=0.565, target_std=10.94)

        return env_map.unsqueeze(0)

    def _fetch_one_pair(self, object_name: str, source_lighting_name: str, 
                        target_lighting_name: str, selected_views: List[int]) -> Dict[str, torch.Tensor]:
        
        transform_path = self.data_dir / "data" / object_name / "transforms_test.json"
        if not transform_path.exists():
            raise FileNotFoundError(f"Transform file not found: {transform_path}")
            
        with open(transform_path, 'r') as f:
            info = json.load(f)
            
        camera_angle_x = info["camera_angle_x"]
        frames = info["frames"]

        # Calculate K using image width (horizontal FOV)
        K = fov2K(camera_angle_x, width=self.resolution[1], height=self.resolution[0])
        
        frame_dict = {
            frame["file_path"]: torch.tensor(frame["transform_matrix"], dtype=torch.float32) 
            for frame in frames
        }

        source_images, source_masks = [], []
        target_images, target_masks = [], []
        source_views, target_views = [], []

        for view_id in selected_views:
            view_dir = self.data_dir / "data" / object_name / f"test_{view_id}"
            src_c2w_key = f"./test_{view_id}/rgba"
            
            # --- Source ---
            src_img_path = view_dir / f"rgba_{source_lighting_name}.png"
            src_img, src_mask = self.read_masked_image(src_img_path)
            source_images.append(src_img)
            source_masks.append(src_mask)
            if src_c2w_key in frame_dict:
                source_views.append(frame_dict[src_c2w_key])
            else:
                raise KeyError(f"Camera pose {src_c2w_key} not found in transforms_test.json")

            # --- Target ---
            tgt_img_path = view_dir / f"rgba_{target_lighting_name}.png"
            tgt_img, tgt_mask = self.read_masked_image(tgt_img_path)
            target_images.append(tgt_img)
            target_masks.append(tgt_mask)
            # Target uses the same camera pose as source for relighting evaluation
            target_views.append(frame_dict[src_c2w_key])

        # Stack tensors
        source_images = torch.stack(source_images, dim=0)
        source_masks = torch.stack(source_masks, dim=0)
        target_images = torch.stack(target_images, dim=0)
        target_masks = torch.stack(target_masks, dim=0)
        
        source_views = torch.stack(source_views, dim=0)
        target_views = torch.stack(target_views, dim=0)

        # Preprocess views (coordinate conversion + scaling)
        all_views = self.view_preprocess(torch.cat([source_views, target_views], dim=0))
        num_src_views = source_views.size(0)
        source_views = all_views[:num_src_views]
        target_views = all_views[num_src_views:]

        # Intrinsics are shared between source and target for the same view
        K_expanded = K.unsqueeze(0).repeat(len(selected_views), 1, 1)
        source_Ks = target_Ks = K_expanded

        # Load HDR lighting
        env_map_dir = self.data_dir / "Environment_Maps" / "high_res_envmaps_2k"
        raw_source_lighting = self.read_lighting(env_map_dir / f"{source_lighting_name}.hdr")
        raw_target_lighting = self.read_lighting(env_map_dir / f"{target_lighting_name}.hdr")

        return {
            "source_lighting": raw_source_lighting,
            "target_lighting": raw_target_lighting,
            "source_images": source_images,
            "source_mask": source_masks,
            "target_images": target_images,
            "target_mask": target_masks,
            "source_view": source_views,
            "target_view": target_views,
            "source_Ks": source_Ks,
            "target_Ks": target_Ks,
        }

    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
            
        data_pair = self.data_pairs[idx]
        object_name = data_pair["object"]
        source_lighting = data_pair["source_lighting"]
        target_lighting = data_pair["target_lighting"]
        views = data_pair["view"]

        try:
            item = self._fetch_one_pair(object_name, source_lighting, target_lighting, views)
            item["idx"] = idx
            item["object"] = object_name
            return item
            
        except Exception as e:
            logger.error(f"Fetch Error at index {idx} (Object: {object_name}): {e}")
            
            # Safe fallback: try the next index, but prevent infinite recursion 
            # if the entire dataset is corrupted by checking if we've wrapped around.
            next_idx = (idx + 1) % len(self)
            if next_idx == 0 and idx != 0:
                raise RuntimeError(f"Failed to fetch any valid item after cycling through dataset. Last error: {e}")
            
            return self.__getitem__(next_idx)

def rescale_lighting_map_torch(lighting_map, target_mean, target_std):
    """
    使用 PyTorch 调整 lighting map 的像素模长。
    
    参数:
    lighting_map: torch.Tensor, 形状为 (C, H, W) 或 (B, C, H, W)
    target_mean: float, 目标均值
    target_std: float, 目标标准差
    """
    # 1. 计算每个像素的 norm 
    # 假设 lighting_map 形状为 (C, H, W)，在通道维度(C)计算L2范数
    # 如果输入带 Batch 维 (B, C, H, W)，dim 应设为 1
    channel_dim = 0 if lighting_map.ndim == 3 else 1
    
    pixel_norms = torch.norm(lighting_map, p=2, dim=channel_dim, keepdim=True)
    
    # 2. 计算当前统计量
    current_mean = torch.mean(pixel_norms)
    current_std = torch.std(pixel_norms)
    
    # 3. 避免除以 0
    eps = 1e-8
    current_std = torch.clamp(current_std, min=eps)
    
    # 4. 计算目标新模长并截断负值
    scaled_norms = (pixel_norms - current_mean) * (target_std / current_std) + target_mean
    scaled_norms = torch.clamp(scaled_norms, min=0.0)
    
    # 5. 计算缩放因子并乘回原图
    scale_factor = scaled_norms / (pixel_norms + eps)
    rescaled_map = lighting_map * scale_factor
    
    return rescaled_map