import torch
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MetricCalculator:
    def __init__(self, device, depth_tolerance=0.25):
        self.device = device
        self.depth_tolerance = depth_tolerance
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=False).to(device).eval()

    @staticmethod
    def _align_scale(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """閉式最優 scale: s = <pred, gt> / <pred, pred>"""
        scale = (pred * gt).sum() / ((pred * pred).sum() + 1e-8)
        return pred * scale

    def _resize_if_spatial_mismatch(self, pred: torch.Tensor, target: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
        if pred.shape[-2:] == target.shape[-2:]:
            return pred
        target_h, target_w = target.shape[-2], target.shape[-1]
        original_dtype = pred.dtype
        is_5d = pred.ndim == 5
        if is_5d:
            B, F, C, H, W = pred.shape
            pred = pred.reshape(B * F, C, H, W)
        needs_cast = not pred.is_floating_point()
        if needs_cast:
            pred = pred.float()
        pred = torch.nn.functional.interpolate(pred, size=(target_h, target_w), mode=mode)
        if original_dtype == torch.bool:
            pred = (pred > 0.5).bool()
        elif needs_cast:
            pred = pred.to(original_dtype)
        if is_5d:
            pred = pred.view(B, F, pred.shape[1], target_h, target_w)
        return pred

    def compute_iou(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_bin = (pred > 0.5).float()
        gt_bin = (gt > 0.5).float()
        intersection = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
        union = pred_bin.sum(dim=(1, 2, 3)) + gt_bin.sum(dim=(1, 2, 3)) - intersection
        return (intersection + 1e-8) / (union + 1e-8)

    def _compute_frame_metrics(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None) -> dict:
        """計算單一 Frame 的 PSNR / sPSNR / SSIM / LPIPS (expects input in [0, 1])"""
        res = {}
        if mask is not None:
            if mask.ndim == 2: mask = mask.unsqueeze(0)
            mask = mask.float()
            if mask.shape[0] == 1 and pred.shape[0] != 1:
                mask = mask.expand_as(pred)
            
            valid_count = mask.clamp(min=0.0).sum()
            if valid_count > 1e-4:
                mse = (((pred - gt) * mask) ** 2).sum() / valid_count
                res["psnr"] = 10 * torch.log10(1.0 / (mse + 1e-8)).item()
                
                pred_v, gt_v = pred[mask > 0.5], gt[mask > 0.5]
                if pred_v.numel() > 0:
                    pred_s = self._align_scale(pred_v, gt_v)
                    res["spsnr"] = 10 * torch.log10(1.0 / (torch.nn.functional.mse_loss(pred_s, gt_v) + 1e-8)).item()
                else:
                    res["spsnr"] = 0.0
                pred_m, gt_m = pred * mask, gt * mask
            else:
                res["psnr"] = res["spsnr"] = 0.0
                pred_m, gt_m = pred, gt
        else:
            mse = torch.nn.functional.mse_loss(pred, gt)
            res["psnr"] = 10 * torch.log10(1.0 / (mse + 1e-8)).item()
            pred_s = self._align_scale(pred, gt)
            res["spsnr"] = 10 * torch.log10(1.0 / (torch.nn.functional.mse_loss(pred_s, gt) + 1e-8)).item()
            pred_m, gt_m = pred, gt

        res["ssim"] = self.ssim(pred_m.unsqueeze(0), gt_m.unsqueeze(0)).item()
        # LPIPS expects [-1, 1], so we map [0, 1] back to [-1, 1]
        res["lpips"] = self.lpips(
            pred_m.unsqueeze(0) * 2.0 - 1.0, 
            gt_m.unsqueeze(0) * 2.0 - 1.0
        ).item()
        return res

    @torch.no_grad()
    def __call__(self, outputs, labels, mask_pred=None, mask_gt=None, 
                 depth_pred=None, depth_gt=None, average=False):
        outputs = outputs.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).to(self.device).clamp(0.0, 1.0)
        labels = labels.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).to(self.device).clamp(0.0, 1.0)
        
        if outputs.ndim == 4: outputs = outputs.unsqueeze(1)
        if labels.ndim == 4: labels = labels.unsqueeze(1)

        def _prep5d(t):
            if t is None: return None
            t = t.to(self.device)
            # Auto-detect & convert if mask/depth accidentally leaks [-1, 1]
            if t.min() < -0.1: 
                t = (t + 1.0) / 2.0
            if t.ndim == 4: t = t.unsqueeze(2)  # (B, F, H, W) -> (B, F, 1, H, W)
            return t

        mask_pred = _prep5d(mask_pred)
        mask_gt = _prep5d(mask_gt)
        depth_pred = _prep5d(depth_pred)
        depth_gt = _prep5d(depth_gt)

        # Spatial alignment
        if outputs.shape[-2:] != labels.shape[-2:]:
            outputs = self._resize_if_spatial_mismatch(outputs, labels, mode='bilinear')

        has_mask = mask_gt is not None
        mask_for_img = None
        if has_mask:
            mask_for_img = self._resize_if_spatial_mismatch(mask_gt, outputs, mode='nearest')
            mask_for_img = (mask_for_img > 0.5).float()
            if mask_pred is not None and mask_pred.shape[-2:] != mask_gt.shape[-2:]:
                mask_pred = self._resize_if_spatial_mismatch(mask_pred, mask_gt, mode='nearest')

        has_depth = depth_gt is not None
        if has_depth and depth_pred is not None:
            if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                depth_pred = self._resize_if_spatial_mismatch(depth_pred, depth_gt, mode='bilinear')
                if has_mask and mask_gt.shape[-2:] != depth_gt.shape[-2:]:
                    mask_gt = self._resize_if_spatial_mismatch(mask_gt, depth_gt, mode='nearest')

        B, F, C, H, W = outputs.shape
        
        # 11 metrics in strict order
        METRIC_KEYS = [
            "psnr", "spsnr", "ssim", "lpips",                    # Unmasked
            "psnr_mask", "spsnr_mask", "ssim_mask", "lpips_mask", # Masked
            "depth_acc", "depth_mse", "mask_iou"                  # Depth + IoU
        ]
        res_data = {k: [] for k in METRIC_KEYS}

        for b in range(B):
            batch_metrics = {k: [] for k in METRIC_KEYS}
            for f in range(F):
                pred_f, gt_f = outputs[b, f], labels[b, f]

                # Unmasked metrics
                unmasked = self._compute_frame_metrics(pred_f, gt_f, mask=None)
                for k in ["psnr", "spsnr", "ssim", "lpips"]:
                    batch_metrics[k].append(unmasked[k])

                # Masked metrics
                if has_mask and mask_for_img is not None:
                    masked = self._compute_frame_metrics(pred_f, gt_f, mask=mask_for_img[b, f])
                    for k in ["psnr_mask", "spsnr_mask", "ssim_mask", "lpips_mask"]:
                        batch_metrics[k].append(masked[k.replace("_mask", "")])
                else:
                    for k in ["psnr_mask", "spsnr_mask", "ssim_mask", "lpips_mask"]:
                        batch_metrics[k].append(None)

                # Mask IoU
                if has_mask and mask_pred is not None:
                    iou_val = self.compute_iou(
                        mask_pred[b, f].unsqueeze(0), 
                        mask_gt[b, f].unsqueeze(0)
                    ).mean().item()
                    batch_metrics["mask_iou"].append(iou_val)
                else:
                    batch_metrics["mask_iou"].append(None)

                # Depth Metrics
                if has_depth and depth_pred is not None:
                    d_p, d_g = depth_pred[b, f], depth_gt[b, f]
                    valid_mask = (mask_gt[b, f] > 0.5) if has_mask else torch.ones_like(d_p, dtype=torch.bool)
                    if valid_mask.shape != d_p.shape:
                        valid_mask = valid_mask.expand_as(d_p)
                    d_p_v, d_g_v = d_p[valid_mask], d_g[valid_mask]
                    if d_p_v.numel() > 0:
                        d_p_aligned = self._align_scale(d_p_v, d_g_v)
                        batch_metrics["depth_mse"].append(torch.nn.functional.mse_loss(d_p_aligned, d_g_v).item())
                        ratio = torch.maximum(
                            d_p_aligned / (d_g_v + 1e-8), 
                            d_g_v / (d_p_aligned + 1e-8)
                        )
                        batch_metrics["depth_acc"].append(
                            (ratio < (1.0 + self.depth_tolerance)).float().mean().item()
                        )
                    else:
                        batch_metrics["depth_mse"].append(None)
                        batch_metrics["depth_acc"].append(None)
                else:
                    batch_metrics["depth_mse"].append(None)
                    batch_metrics["depth_acc"].append(None)

            # Aggregate within batch
            for k in METRIC_KEYS:
                valid_vals = [v for v in batch_metrics[k] if v is not None]
                res_data[k].append(float(np.mean(valid_vals)) if valid_vals else None)

        # Global aggregation
        final_results = []
        for k in METRIC_KEYS:
            valid_vals = [v for v in res_data[k] if v is not None]
            final_results.append(
                float(np.mean(valid_vals)) if average and valid_vals else res_data[k]
            )
        return tuple(final_results)

            
def resize_5d(tensor, size=(256, 256), mode='bilinear', align_corners=False):
    B, N, C, H, W = tensor.shape
    # 1. Flatten (B, N) -> single batch dim for 2D interpolation
    tensor = tensor.reshape(B * N, C, H, W)
    # 2. Interpolate
    tensor = torch.nn.functional.interpolate(tensor, size=size, mode=mode, align_corners=align_corners)
    # 3. Restore 5D shape
    return tensor.reshape(B, N, C, size[0], size[1])