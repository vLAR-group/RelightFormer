import os
import numpy as np
import torch
import torch.nn.functional as F
import glob
from itertools import combinations, product
import cv2

def resize_tensor(x: torch.Tensor, size: tuple, mode: str = 'bilinear') -> torch.Tensor:
    """
    Resize a tensor, automatically handling CHW or HWC formats.
    """
    assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"
    
    if x.dim() == 2:
        x = x.unsqueeze(0)
        is_chw = True
    elif x.dim() == 3:
        if x.shape[0] in [1, 3]:
            is_chw = True
        elif x.shape[2] in [1, 3]:
            is_chw = False
        else:
            is_chw = True  # Default fallback
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

    if (is_chw and x.shape[1:] == size) or (not is_chw and x.shape[:2] == size):
        return x

    if not is_chw:
        x = x.permute(2, 0, 1)  # HWC -> CHW

    if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        x = F.interpolate(x.unsqueeze(0), size=size, mode=mode, align_corners=False).squeeze(0)
    else:
        x = F.interpolate(x.unsqueeze(0), size=size, mode=mode).squeeze(0)

    if not is_chw:
        x = x.permute(1, 2, 0)  # CHW -> HWC
    
    return x

def generate_view_pairs(views: list, source_view_num: int, target_view_num: int, 
                        novel_view: bool = False, same_view: bool = False) -> list:
    """Generate source-target view pairs with overlap constraints."""
    if novel_view and same_view:
        raise ValueError("Cannot set both novel_view=True and same_view=True")
    
    pairs = []
    if novel_view:
        for source_combo in combinations(views, source_view_num):
            remaining = [v for v in views if v not in source_combo]
            if len(remaining) >= target_view_num:
                for target_combo in combinations(remaining, target_view_num):
                    pairs.append((source_combo, target_combo))
    elif same_view:
        if source_view_num != target_view_num:
            raise ValueError("same_view=True requires source_view_num == target_view_num")
        for combo in combinations(views, source_view_num):
            pairs.append((combo, combo))
    else:
        for pair in product(combinations(views, source_view_num), combinations(views, target_view_num)):
            pairs.append(pair)
    
    return pairs

class IndexDecomposer:
    """Decomposes a flat index into multi-dimensional indices based on group lengths."""
    def __init__(self, group_lengths: list):
        self.group_lengths = group_lengths

    def __call__(self, idx: int) -> list:
        return self._recursive_decompose(idx, len(self.group_lengths) - 1)

    def _recursive_decompose(self, idx: int, group_index: int) -> list:
        if group_index < 0:
            return []
        current_length = self.group_lengths[group_index]
        current_index = idx % current_length
        next_indices = self._recursive_decompose(idx // current_length, group_index - 1)
        return next_indices + [current_index]
    
def apply_rotation_to_views(views, rotation_matrix):
    """
    Apply a rotation matrix to a batch of camera view matrices.
    Rotates both the rotation (R) and translation (T) components.

    Args:
        views: torch.Tensor of shape (N, 4, 4) - batch of camera pose matrices
        rotation_matrix: torch.Tensor of shape (3, 3) - rotation matrix to apply

    Returns:
        torch.Tensor of shape (N, 4, 4) - rotated view matrices
        [R_add | 0 ][R_original | t_original] = [R_add*R_original | R_add*t_original]
        [   0  | 1 ][   0       |     1     ] = [   0             |           1     ]
        
    """
    if rotation_matrix is None:
        return views
    
    # Extract rotation part (N, 3, 3) and translation part (N, 3, 1)
    R_original = views[:, :3, :3]  # (N, 3, 3)
    t_original = views[:, :3, 3:4]  # (N, 3, 1)
    
    # Apply the rotation to the original rotation: R_new = R_rotation @ R_original
    R_rotated = torch.matmul(rotation_matrix.unsqueeze(0), R_original)  # (1, 3, 3) @ (N, 3, 3) -> (N, 3, 3)
    
    # Apply the rotation to the original translation: t_new = R_rotation @ t_original
    t_rotated = torch.matmul(rotation_matrix.unsqueeze(0), t_original)  # (1, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
    
    # Reconstruct the rotated pose matrix
    views_rotated = torch.cat([
        torch.cat([R_rotated, t_rotated], dim=2),  # (N, 3, 4)
        views[:, 3:, :]  # (N, 1, 4) - keep the last row [0, 0, 0, 1]
    ], dim=1)  # (N, 4, 4)
    
    return views_rotated

def read_hdr(path, size):
    """
    Reads an HDR map from disk (.hdr or .exr).
    Prioritizes OpenCV/imageio for stability, falling back to OpenEXR only if necessary.
    """
    path = str(path) # Ensure it's a string for library compatibility
    if not os.path.exists(path):
        raise FileNotFoundError(f"HDR file not found: {path}")

    img = None
    
    # --- Method 1: OpenCV (Fastest & Most Stable for DataLoaders) ---
    # Note: Ensure 'OPENCV_IO_ENABLE_OPENEXR=1' is set in your bashrc or env
    try:
        # IMREAD_ANYCOLOR | IMREAD_ANYDEPTH is vital for HDR/EXR float values
        img_cv2 = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img_cv2 is not None:
            # OpenCV loads as BGR; convert to RGB
            if len(img_cv2.shape) == 3 and img_cv2.shape[2] >= 3:
                img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            else:
                img = img_cv2
    except Exception:
        pass

    # --- Method 2: imageio (Excellent fallback for EXR/HDR) ---
    if img is None:
        try:
            import imageio
            # imageio v3+ handles EXR well via the freeimage or pyexr plugin
            img = imageio.imread(path)
        except Exception:
            pass

    # --- Method 3: OpenEXR (Last resort, handled carefully) ---
    if img is None and path.lower().endswith('.exr'):
        try:
            import OpenEXR
            import Imath
            exr_file = OpenEXR.InputFile(path)
            header = exr_file.header()
            dw = header['dataWindow']
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = ['R', 'G', 'B']
            # Only read channels that actually exist in the file
            available = list(header['channels'].keys())
            to_read = [c for c in channels if c in available]
            
            channel_data = exr_file.channels(to_read, pt)
            exr_file.close() # CRITICAL: Close immediately to prevent segfaults
            
            decoded = [np.frombuffer(c, dtype=np.float32).reshape(h, w) for c in channel_data]
            img = np.stack(decoded, axis=-1)
        except Exception as e:
            raise RuntimeError(f"All HDR load methods failed for {path}. Error: {e}")

    if img is None:
        raise RuntimeError(f"Failed to load HDR image at {path}")

    # --- Post-processing ---
    img = img.astype(np.float32)

    # Standardize to 3 channels (RGB)
    if img.ndim == 2: # Gray
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1: # Gray with channel dim
        img = np.tile(img, (1, 1, 3))
    elif img.shape[-1] > 3: # Remove Alpha
        img = img[:, :, :3]

    # Resize
    if size is not None:
        # cv2.resize expects (width, height)
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    return img