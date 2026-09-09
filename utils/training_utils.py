import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import kornia

import torch
from functools import partial

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from diffsynth.utils.camera import batch_cam2pose_embedding
import socket
import torch.nn.functional as F
from itertools import product


device = 'cuda'
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_input.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- diffusers
inference: true
---
    """
    model_card = f"""
# zero123-{repo_id}

These are zero123 weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def calculate_metrics(outputs, labels, masks=None, average=False):
    """
    Calculate PSNR, SSIM, LPIPS per sample (average over frames S).
    
    Args:
        outputs: (B, S, C, H, W) or (B, C, H, W)
        labels:  (B, S, C, H, W) or (B, C, H, W)
        masks:   (B, S, 1, H, W) or (B, S, H, W) or None
        average: If True, return scalar averages. If False, return lists per sample.

    Returns:
        If average=False: psnr_list, ssim_list, lpips_list (lists of length B)
        If average=True: mean_psnr, mean_ssim, mean_lpips (scalars)
    """
    device = outputs.device
    labels = labels.to(device)
    # Ensure 5D: (B, S, C, H, W)
    global ssim, lpips  # If they are global instances
    ssim = ssim.to(outputs.device)
    lpips = lpips.to(outputs.device)
    if outputs.ndim == 4:
        outputs = outputs.unsqueeze(1)  # (B, 1, C, H, W)
    if labels.ndim == 4:
        labels = labels.unsqueeze(1)

    B, S, C, H, W = outputs.shape
    assert labels.shape == (B, S, C, H, W)

    # Handle masks
    if masks is not None:
        if masks.ndim == 4:
            masks = masks.unsqueeze(2)  # (B, S, 1, H, W)
        if masks.shape[2] == 1:  # (B, S, 1, H, W)
            masks = masks.expand(-1, -1, C, -1, -1)  # (B, S, C, H, W)
        masks = masks.float()
        outputs = outputs * masks
        labels = labels * masks

    # Clamp to [0, 1]
    outputs = torch.clamp(outputs, 0.0, 1.0)
    labels = torch.clamp(labels, 0.0, 1.0)

    # ================== PSNR per sample (avg over S) ==================
    if masks is not None:
        diff_sq = (outputs - labels) ** 2
        mse = (diff_sq * masks).sum(dim=(2, 3, 4)) / (masks.sum(dim=(2, 3, 4)) + 1e-8)  # (B, S)
    else:
        mse = F.mse_loss(outputs, labels, reduction='none').mean(dim=(2, 3, 4))  # (B, S)
    
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))  # (B, S)
    psnr_per_sample = psnr.mean(dim=1)  # (B,) - avg over S
    psnr_list = psnr_per_sample.cpu().numpy().tolist()  # [val_b0, val_b1, ...]

    # ================== SSIM per sample (avg over S) ==================
    ssim_list = []
    for b in range(B):
        ssim_vals = []
        for s in range(S):
            pred_frame = outputs[b, s].unsqueeze(0)  # (1, C, H, W)
            gt_frame = labels[b, s].unsqueeze(0)     # (1, C, H, W)
            
            if masks is not None:
                mask_frame = masks[b, s].unsqueeze(0)  # (1, C, H, W)
                pred_frame = pred_frame * mask_frame
                gt_frame = gt_frame * mask_frame
            
            # Ensure ssim is called with tensors on the same device
            # ssim_val = ssim(pred_frame, gt_frame)  # Assuming ssim is imported from somewhere
            ssim_val = ssim(pred_frame, gt_frame)
            ssim_vals.append(ssim_val.item())
        
        # Average over S for this sample
        ssim_list.append(np.mean(ssim_vals))

    # ================== LPIPS per sample (avg over S) ==================
    lpips_list = []
    # LPIPS expects [-1, 1]
    outputs_lpips = outputs * 2.0 - 1.0  # [-1, 1]
    labels_lpips = labels * 2.0 - 1.0    # [-1, 1]

    for b in range(B):
        lpips_vals = []
        for s in range(S):
            pred_frame = outputs_lpips[b, s].unsqueeze(0)  # (1, C, H, W)
            gt_frame = labels_lpips[b, s].unsqueeze(0)     # (1, C, H, W)
            
            if masks is not None:
                mask_frame = masks[b, s].unsqueeze(0)  # (1, C, H, W)
                pred_frame = pred_frame * mask_frame
                gt_frame = gt_frame * mask_frame
            
            # Ensure lpips is called with tensors on the same device
            # lpips_val = lpips(pred_frame, gt_frame)  # Assuming lpips is imported from somewhere
            lpips_val = lpips(pred_frame, gt_frame)
            lpips_vals.append(lpips_val.item())
        
        # Average over S for this sample
        lpips_list.append(np.mean(lpips_vals))

    # ================== Return based on average flag ==================
    if average:
        mean_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0.0
        mean_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0.0
        mean_lpips = sum(lpips_list) / len(lpips_list) if lpips_list else 0.0
        return mean_psnr, mean_ssim, mean_lpips
    else:
        return psnr_list, ssim_list, lpips_list

def create_log_images(
    source,                # (B, F_src, 3, H, W)
    target,                # (B, F_tgt, 3, H, W)
    pred,                  # (B, F_tgt, 3, H, W)
    lighting_log=None,     # (B, L, 3, H, W) or None
    pred_depth=None,       # (B, F, 1, H, W) or None
    gt_depth=None,         # (B, F, 1, H, W) or None
    pred_mask=None,        # (B, F, 1, H, W) or None
    gt_mask=None,          # (B, F, 1, H, W) or None
    label_height=50,
    font_size=36,
    resize_size=(256, 256)  # 🔑 新增參數：統一縮放尺寸
):
    """
    創建用於訓練監控的網格圖。
    所有輸入先統一 Resize 到 resize_size，再進行水平拼接與垂直堆疊。
    """
    B = source.size(0)
    
    # ==================== 0. 統一 Resize 預處理 ====================
    def safe_resize(tensor, size, mode='bilinear'):
        """安全 Resize：處理 5D Tensor (B, F, C, H, W) -> (B, F, C, size[0], size[1])"""
        if tensor is None:
            return None
        B_in, F_in, C_in, H_in, W_in = tensor.shape
        if H_in == size[0] and W_in == size[1]:
            return tensor  # 已是目標尺寸，跳過
        # Flatten -> Interpolate -> Reshape
        tensor_flat = tensor.reshape(B_in * F_in, C_in, H_in, W_in)
        tensor_resized = F.interpolate(tensor_flat, size=size, mode=mode)
        return tensor_resized.reshape(B_in, F_in, C_in, size[0], size[1])
    
    # RGB 圖像使用 bilinear，單通道 Depth/Mask 使用 nearest 避免插值偽影
    source = safe_resize(source, resize_size, mode='bilinear')
    target = safe_resize(target, resize_size, mode='bilinear')
    pred   = safe_resize(pred, resize_size, mode='bilinear')
    if lighting_log is not None:
        lighting_log = safe_resize(lighting_log, resize_size, mode='bilinear')
    if pred_depth is not None:
        pred_depth = safe_resize(pred_depth, resize_size, mode='nearest')
    if gt_depth is not None:
        gt_depth = safe_resize(gt_depth, resize_size, mode='nearest')
    if pred_mask is not None:
        pred_mask = safe_resize(pred_mask, resize_size, mode='nearest')
    if gt_mask is not None:
        gt_mask = safe_resize(gt_mask, resize_size, mode='nearest')
    
    # 更新 H, W 為 resize 後的尺寸
    H, W = resize_size
    
    # ==================== 1. 統一幀數基準 ====================
    F_src = source.size(1)
    F_tgt = target.size(1)
    F_lit = lighting_log.size(1) if lighting_log is not None else 0
    F_pd  = pred_depth.size(1) if pred_depth is not None else 0
    F_pm  = pred_mask.size(1) if pred_mask is not None else 0
    
    F_max = max(F_src, F_tgt, F_lit, F_pd, F_pm)
    
    # ==================== 2. 輔助函數 ====================
    def stack_frames_horizontally_rgb(tensor, target_frames):
        """將幀水平堆疊，不足 target_frames 的部分補黑幀"""
        B_in, F, C, H_in, W_in = tensor.shape
        img = torch.clamp(tensor * 255.0, 0, 255).byte()
        img = img.permute(0, 3, 1, 4, 2)  # (B, H, F, W, 3)
        
        if F < target_frames:
            pad_size = target_frames - F
            black_frames = torch.zeros(B_in, H_in, pad_size, W_in, 3, dtype=torch.uint8, device=img.device)
            img = torch.cat([img, black_frames], dim=2)
        
        final_f = img.shape[2]
        img = img.reshape(B_in, H_in, final_f * W_in, 3)
        return img.cpu().numpy()
    
    def process_to_rgb(tensor, target_frames, is_mask=False):
        """處理 Depth 或 Mask (B, F, 1, H, W) 為 RGB 格式並補齊"""
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(2)
        
        B_in, F, C, H_in, W_in = tensor.shape
        
        if not is_mask:
            # 深度圖歸一化 (per-sample)
            t_flat = tensor.view(B_in, F, -1)
            min_v = t_flat.min(dim=-1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)
            max_v = t_flat.max(dim=-1, keepdim=True).values.unsqueeze(-1).unsqueeze(-1)
            tensor = (tensor - min_v) / (max_v - min_v + 1e-8)
            
        rgb = tensor.float().expand(-1, -1, 3, -1, -1)
        return stack_frames_horizontally_rgb(rgb, target_frames)

    # ==================== 3. 處理所有圖像行 ====================
    source_img = stack_frames_horizontally_rgb(source, F_max)
    target_img = stack_frames_horizontally_rgb(target, F_max)
    pred_img   = stack_frames_horizontally_rgb(pred, F_max)
    
    pred_rows = [
        ("Source", [source_img[b] for b in range(B)]),
        ("Pred",   [pred_img[b] for b in range(B)]),
    ]
    gt_rows = [
        ("GT",     [target_img[b] for b in range(B)]),
    ]
    
    if lighting_log is not None:
        lighting_img = stack_frames_horizontally_rgb(lighting_log, F_max)
        pred_rows.append(("Lighting", [lighting_img[b] for b in range(B)]))
    
    if pred_depth is not None and gt_depth is not None:
        p_depth = process_to_rgb(pred_depth, F_max)
        g_depth = process_to_rgb(gt_depth, F_max)
        pred_rows.append(("Depth Pred", [p_depth[b] for b in range(B)]))
        gt_rows.append(("Depth GT",     [g_depth[b] for b in range(B)]))
        
    if pred_mask is not None:
        p_mask = process_to_rgb(pred_mask, F_max, is_mask=True)
        pred_rows.append(("Pred Mask", [p_mask[b] for b in range(B)]))
    if gt_mask is not None:
        g_mask = process_to_rgb(gt_mask, F_max, is_mask=True)
        gt_rows.append(("GT Mask", [g_mask[b] for b in range(B)]))

    # ==================== 4. 拼接與渲染 ====================
    all_rows_data = pred_rows + gt_rows
    
    row_widths = [sum(img.shape[1] for img in row_imgs) for _, row_imgs in all_rows_data]
    total_width = max(row_widths)
    
    each_img_h = all_rows_data[0][1][0].shape[0]
    num_rows = len(all_rows_data)
    total_height = num_rows * (each_img_h + (label_height if font_size > 0 else 0))
    
    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    current_y = 0
    for label, img_list in all_rows_data:
        if font_size > 0:
            draw.rectangle([0, current_y, total_width, current_y + label_height], fill=(200, 200, 200))
            draw.text((10, current_y + (label_height - font_size)//2), label, fill=(0, 0, 0), font=font)
            current_y += label_height
        
        row_combined = np.concatenate(img_list, axis=1)
        canvas.paste(Image.fromarray(row_combined), (0, current_y))
        current_y += each_img_h
        
    return canvas


validation_batch = 1
max_validation_length = 1024
mini_validation_length = 64
def dataloader_maker(data_dict, mode='full'):
    results = {}
    torch_dataloader = partial(
        torch.utils.data.DataLoader,
        shuffle=False,
        batch_size=validation_batch,
        num_workers=0,
    )
    for key, value in data_dict.items():
        if mode == 'full':
            if len(value) > max_validation_length:
                indices = np.round(np.linspace(0, len(value) - 1, max_validation_length)).astype(int)
                value = torch.utils.data.Subset(value, indices.tolist())
            key = mode + '_' + key
            results[key] = torch_dataloader(value)
        elif mode == 'mini':
            if len(value) > mini_validation_length:
                indices = np.round(np.linspace(0, len(value) - 1, mini_validation_length)).astype(int)
                value = torch.utils.data.Subset(value, indices.tolist())
            key = mode + '_' + key
            results[key] = torch_dataloader(value)
        else:
            raise ValueError(f'Invalid mode {mode}')
    return results

def get_validation_datasets(dataset_cls, args):
    datasets = {}
    
    # # Various Training Dataset
    N_list = [4, 16]
    M_list = [16]
    for n, m in product(N_list, M_list):
        if n > m:
            continue
        key = f"Training_{n}_to_{m}"
        datasets[key] = dataset_cls(
            args.dataset_path,
            source_view_num=n,
            target_view_num=m, 
            is_train=True,
            ablation=args.ablation
        )
    # Novel View in Training
    # N_list = [8]
    # M_list = [8]
    # for n, m in product(N_list, M_list):
    #     if n != m:
    #         continue
    #     key = f"Training_Novel_{n}_to_{m}"
    #     datasets[key] = dataset_cls(
    #         args.dataset_path,
    #         source_view_num=n,
    #         target_view_num=m, 
    #         is_train=False,
    #         novel_view=True,
    #         ablation=args.ablation
    #     )

    # Same View in Training
    # N_list = [1, 8]
    # M_list = [1, 8]
    # for n, m in product(N_list, M_list):
    #     if n != m:
    #         continue
    #     key = f"Training_Same_{n}_to_{m}"
    #     datasets[key] = dataset_cls(
    #         args.dataset_path,
    #         source_view_num=n,
    #         target_view_num=m, 
    #         is_train=False,
    #         same_view=True,
    #         ablation=args.ablation
    #     )
    
    # Novel View in Validation
    # N_list = [8]
    # M_list = [8]
    # for n, m in product(N_list, M_list):
    #     key = f"Validation_Novel_{n}_to_{m}"
    #     datasets[key] = dataset_cls(
    #         args.dataset_path,
    #         source_view_num=n,
    #         target_view_num=m,
    #         object_split = "testing",
    #         lighting_split = "testing",
    #         view_split = "testing",
    #         is_train=False,
    #         novel_view=True
    #     )

    # Same View in Validation
    # N_list = [1, 8]
    # M_list = [1, 8]
    # for n, m in product(N_list, M_list):
    #     if n != m:
    #         continue
    #     key = f"Validation_Same_{n}_to_{m}"
    #     datasets[key] = dataset_cls(
    #         args.dataset_path,
    #         source_view_num=n,
    #         target_view_num=m,
    #         object_split = "testing",
    #         lighting_split = "testing",
    #         view_split = "testing",
    #         is_train=False,
    #         same_view=True
    #     )
    
    return datasets

def get_training_dataset(dataset_cls, args):
    """
    Create a super-large dataset by merging multiple N-to-M configurations.
    """
    datasets = {}
    # N to M configurations
    max_view = args.max_view
    if args.training_in_same_view:
        N_list = M_list = range(1, max_view+1)
    else:
        N_list = range(1, max_view+1)
        M_list = [max_view]
    
    # Generate all N-to-M combinations
    for n, m in product(N_list, M_list):
        if args.training_in_same_view and n != m:
            continue
        key = f"{n}_to_{m}"
        datasets[key] = dataset_cls(
            args.dataset_path,
            source_view_num=n,
            target_view_num=m, 
            is_train=True,
            same_view=args.training_in_same_view,
            ablation=args.ablation,
            resolution=(args.resolution, args.resolution)
        )
        # print(f"Created {key}: {len(datasets[key]):,} samples")
    
    return datasets

def print_model_parameter(m):
    for name, param in m.named_parameters():
        if param.grad is not None:  # Check if the parameter has a gradient
            print(f"Parameter: {name} - Grad dtype: {param.grad.dtype}, Grad shape: {param.grad.shape}")
        else:
            print(f"Parameter: {name} - No gradient")


def manage_checkpoints(checkpoint_dir):
    """
    Manage checkpoint files by deleting older ones, keeping only the latest 3.

    Parameters:
    checkpoint_dir: Directory where checkpoint files are stored.
    """
    # List all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    
    # Sort files by creation time (oldest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))

    # If there are more than 3 checkpoints, delete the oldest ones
    while len(checkpoint_files) > 3:
        oldest_checkpoint = checkpoint_files.pop(0)  # Get the oldest checkpoint
        os.remove(os.path.join(checkpoint_dir, oldest_checkpoint))
        print(f"Deleted old checkpoint: {oldest_checkpoint}")


@torch.no_grad()
def data_preprocess(batch, pipe, vae_dtype, tiled, tile_size, tile_stride,
                    default_prompt="边缘清晰，黑色背景，高质量图片，objaverse数据集，纹理清晰",
                    vae_rescale=True):
    data = {}
    tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

    if "source_lighting" in batch.keys():
        video = batch['source_lighting'].to(vae_dtype)
        data['source_lighting'] = \
            pipe.encode_video_in_frame(video, rescale=vae_rescale, **tiler_kwargs).contiguous()

    if "target_lighting" in batch.keys():
        video = batch['target_lighting'].to(vae_dtype)
        data['target_lighting'] = \
            pipe.encode_video_in_frame(video, rescale=vae_rescale, **tiler_kwargs).contiguous()

    if 'source_images' in batch.keys():
        pipe.load_models_to_device(['vae'])
        video = batch['source_images'].to(vae_dtype)
        data['source_images'] = \
            pipe.encode_video_in_frame(video, rescale=vae_rescale, **tiler_kwargs).contiguous()
    
    if 'target_images' in batch.keys():
        pipe.load_models_to_device(['vae'])
        video = batch['target_images'].to(vae_dtype)
        data['target_images'] = \
            pipe.encode_video_in_frame(video, rescale=vae_rescale, **tiler_kwargs).contiguous()
        
    if 'source_rays' in batch.keys():
        data["source_rays"] = batch["source_rays"]

    if 'target_rays' in batch.keys():
        data["target_rays"] = batch["target_rays"]

    if 'lighting_rays' in batch.keys():
        data["lighting_rays"] = batch["lighting_rays"]
    
    if 'source_view' in batch.keys():
        data['source_view'] = batch['source_view']

    if 'target_view' in batch.keys():
        data['target_view'] = batch['target_view']

    if 'intrinsic_matrix' in batch.keys():
        data['intrinsic_matrix'] = batch['intrinsic_matrix']

    if 'source_depths' in batch.keys():
        data['source_depths'] = batch['source_depths']

    if 'target_depths' in batch.keys():
        data['target_depths'] = batch['target_depths']

    if 'source_mask' in batch.keys():
        data['source_mask'] = batch['source_mask']

    if 'target_mask' in batch.keys():
        data['target_mask'] = batch['target_mask']

    if 'source_Ks' in batch.keys():
        data['source_Ks'] = batch['source_Ks']

    if 'target_Ks' in batch.keys():
        data['target_Ks'] = batch['target_Ks']

    return data

def get_unused_port() -> int:
    """Dynamically find an unused port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind to localhost with port 0 → OS assigns random unused port
        s.bind(('localhost', 0))
        # Get the assigned port
        _, port = s.getsockname()
    return port


def combine_dataloaders(dataloaders, seed=12):
    iterators = [iter(dl) for dl in dataloaders]
    return CombinedIterator(iterators, seed=seed)

import random
class CombinedIterator:
    """
    An iterator that combines multiple iterators, randomly selecting one
    at each step to yield an item from, until all are exhausted.
    """

    def __init__(self, iterators, seed=12):
        """
        Initializes the CombinedIterator.

        Args:
            iterators (list of iterator objects): The list of iterators to combine.
            seed (int, optional): Seed for the random number generator used
                                  to select iterators. Defaults to None.
        """
        if not iterators:
            raise ValueError("Iterators list cannot be empty.")

        self.iterators = list(iterators) # Store a copy
        self.rng = random.Random(seed) # Random number generator
        # Keep track of which iterators are still active
        self.active_mask = [True] * len(self.iterators)

    def __iter__(self):
        """
        Returns itself as an iterator.
        """
        # Reset the active mask when a new iteration starts
        self.active_mask = [True] * len(self.iterators)
        return self

    def __next__(self):
        """
        Random iterator and yields the next item from it.
        Raises StopIteration when all iterators are exhausted.
        """
        # Find indices of active iterators
        active_indices = [i for i, active in enumerate(self.active_mask) if active]

        if not active_indices:
            # All iterators are exhausted
            raise StopIteration

        # Randomly select an index from the active ones
        chosen_idx = self.rng.choice(active_indices)

        try:
            # Get the next item from the chosen iterator
            item = next(self.iterators[chosen_idx])
            return item
        except StopIteration:
            # The chosen iterator is now exhausted.
            # Mark it as inactive.
            self.active_mask[chosen_idx] = False
            # Recursively call __next__ to try another active iterator
            return self.__next__()


def split_loss(diff, source_views, target_views, tol=1e-6):
    """
    Splits loss based on whether the target camera pose exists in the source camera pool.
    diff shape: (B, C, F, H, W)
    """
    device = diff.device
    B, C, F, H, W = diff.shape
    f_src = source_views.shape[1]

    # Safety check: ensure frame counts align
    assert diff.shape[2] == target_views.shape[1], \
        f"Frame count mismatch: diff F={diff.shape[2]}, target_views F={target_views.shape[1]}"

    assert source_views.shape[2:] == target_views.shape[2:], (
        f"Camera matrix shape mismatch: source={source_views.shape[2:]}, target={target_views.shape[2:]}. "
        f"Ensure both are 4x4, both are 3x3, or slice them to match before calling."
    )
    mat_size = source_views.shape[2] * source_views.shape[3]

    # 1. Per-view loss: average over C, H, W -> [B, F]
    # FIXED: dims (1, 3, 4) instead of (2, 3, 4) to match (B, C, F, H, W)
    loss_per_view = diff.abs().mean(dim=(1, 3, 4))

    # 2. Flatten matrices: [B, N_views, mat_size]
    target_flat = target_views.reshape(B, F, mat_size)
    source_flat = source_views.reshape(B, f_src, mat_size)

    # 3. Vectorized comparison
    matches = torch.isclose(
        target_flat.unsqueeze(2),  # [B, F, 1, mat_size]
        source_flat.unsqueeze(1),  # [B, 1, f_src, mat_size]
        atol=tol
    )
    is_seen = matches.all(dim=-1).any(dim=2)  # [B, F]

    # 4. Split & average
    seen_losses = loss_per_view[is_seen]
    novel_losses = loss_per_view[~is_seen]

    mean_seen = seen_losses.mean() if seen_losses.numel() > 0 else torch.tensor(0.0, device=device)
    mean_novel = novel_losses.mean() if novel_losses.numel() > 0 else torch.tensor(0.0, device=device)

    return mean_seen, mean_novel

def complie_model(models):
    compiled_models = []
    for m in models:
        compiled_models.append(
            torch.compile(
                m,
                mode="default",  # ❗ Must be None when using options in PyTorch 2.10
                dynamic=True,
            )
        )
    return compiled_models