import json
import logging
import os
import shutil
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers.utils import is_wandb_available
from tqdm.auto import tqdm
from torchvision.io import ImageReadMode, read_image
from torchvision.utils import save_image

from diffsynth import RelightFormerPipeline
from datasets.LavalObjaverseDataset import LavalObjaverseEvalDataset as LODEvalDataset
from utils.metrics import MetricCalculator, resize_5d
from utils.args import read_yaml_to_namespce

# -----------------------------------------------------------------------------
# Environment & Warning Configuration
# -----------------------------------------------------------------------------
os.environ['HF_HOME'] = './hf_cache'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ['WANDB_CONFIG_DIR'] = f"/tmp/.config-{os.environ.get('USER', 'user')}"

NCCL_TIMEOUT = 360000

# Suppress known benign warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings("ignore", message="cc_projection/diffusion_pytorch_model.safetensors not found")
warnings.filterwarnings("ignore", message="The config attributes {'cc_projection':.*")
warnings.filterwarnings("ignore", message=".*is_pinned.*device.*", category=DeprecationWarning, module="torch")
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = get_logger(__name__)

def generate_sequence(N, R):
    visited = [False] * (N + 1)
    result = []
    
    for start in range(R):
        for current in range(start, N + 1, R):
            if not visited[current]:
                visited[current] = True
                result.append(current)
                
    return result


def load_pred_image(
    image_path: Path,
    target_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load a saved prediction as an RGB tensor in [0, 1]."""
    image = read_image(str(image_path), mode=ImageReadMode.RGB)
    image = image.to(device=device, dtype=torch.float32).div_(255.0)

    if image.shape[-2:] != target_size:
        image = F.interpolate(
            image.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image.to(dtype=dtype)


# -----------------------------------------------------------------------------
# Validation & Logging
# -----------------------------------------------------------------------------
@torch.no_grad()
def log_validation(
    validation_dataloader: torch.utils.data.DataLoader,
    pipe: RelightFormerPipeline,
    args: Any,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    split: str = "testing",
    cur_step: int = 0,
) -> Dict[str, Any]:
    
    cfg = args.config
    device = accelerator.device
    proc_id = accelerator.process_index
    logger.info(f"🚀 Running {split} validation at step {cur_step} (Process {proc_id})...")
    metric_calculator = MetricCalculator(device=device, depth_tolerance=0.1)
    res_json_path = (Path(args.output_dir) / f"results_rank_{proc_id}.json").resolve()
    
    # 1. Resume logic
    if getattr(args, "skip_exist", False) and res_json_path.exists():
        with open(res_json_path, 'r') as f:
            evaluation_results = json.load(f)
        # Ensure data_pair is a dict for O(1) lookups
        if isinstance(evaluation_results.get("data_pair"), list):
            evaluation_results["data_pair"] = {str(item["sample_idx"]): item for item in evaluation_results["data_pair"]}
        logger.info(f"📦 Resumed from {res_json_path}, already processed {len(evaluation_results['data_pair'])} samples.")
    else:
        evaluation_results = {"average": {}, "data_pair": {}}

    # 2. Initialize Pipeline
    # 3. Load metadata (optional, for saving paths)
    data_pairs = None
    if getattr(args, "pair_info", None) and os.path.isfile(args.pair_info):
        try:
            with open(args.pair_info, 'r') as f:
                data_pairs = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load pair info from {args.pair_info}: {e}")

    # 4. Evaluation Loop
    bar = tqdm(enumerate(validation_dataloader), desc=f"{split}-Rank{proc_id}", total=len(validation_dataloader), disable=not accelerator.is_local_main_process)
    
    for valid_step, batch in bar:
        if batch is None:
            continue

        inference_size = (cfg.resolution, cfg.resolution)

        # Always save GT and reference before resume/skip filtering. This also
        # covers samples already recorded in JSON or backed by cached predictions.
        source_to_save = batch["source_images"]
        target_to_save = batch["target_images"]
        if source_to_save.shape[-2:] != inference_size:
            source_to_save = resize_5d(source_to_save, size=inference_size)
        if target_to_save.shape[-2:] != inference_size:
            target_to_save = resize_5d(target_to_save, size=inference_size)

        for batch_pos, sample_idx_value in enumerate(batch["idx"].tolist()):
            sample_idx = int(sample_idx_value)
            if isinstance(data_pairs, list):
                meta = data_pairs[sample_idx] if 0 <= sample_idx < len(data_pairs) else None
            else:
                meta = data_pairs.get(str(sample_idx)) if isinstance(data_pairs, dict) else None

            subfolder_name = (
                str(batch["meta"][batch_pos]) if "meta" in batch else str(sample_idx)
            )
            num_save_views = target_to_save.shape[1]

            for v in range(num_save_views):
                view_name = (
                    meta["view"][v].split(".")[0]
                    if meta and "view" in meta
                    else f"view_{v}"
                )
                out_dir = (
                    Path(args.output_dir) / subfolder_name / view_name
                ).resolve()
                out_dir.mkdir(parents=True, exist_ok=True)
                save_image(
                    target_to_save[batch_pos, v],
                    out_dir / "gt_relight.png",
                    normalize=False,
                )
                save_image(
                    source_to_save[batch_pos, v],
                    out_dir / "ref_relight.png",
                    normalize=False,
                )

                save_image(
                    batch['source_mask'][batch_pos, v],
                    out_dir / "mask.png",
                    normalize=False,
                )

        # Filter out already processed samples if skip_exist is True
        all_indices = batch['idx'].tolist()
        keep_mask = [
            not (getattr(args, "skip_exist", False) and str(idx_val) in evaluation_results["data_pair"])
            for idx_val in all_indices
        ]
        
        if not any(keep_mask):
            continue
            
        keep_mask_tensor = torch.tensor(keep_mask, device=device, dtype=torch.bool)
        
        # Apply mask to batch
        idx = batch['idx'][keep_mask_tensor]
        B = idx.size(0)
        
        source = batch['source_images'][keep_mask_tensor].to(device, dtype=weight_dtype)
        target = batch['target_images'][keep_mask_tensor].to(device, dtype=weight_dtype)
        mask = batch['source_mask'][keep_mask_tensor].to(device, dtype=weight_dtype)
        lighting = batch['target_lighting'][keep_mask_tensor].to(device, dtype=weight_dtype)
        source_view = batch["source_view"][keep_mask_tensor].to(device, dtype=weight_dtype)
        target_view = batch["target_view"][keep_mask_tensor].to(device, dtype=weight_dtype)
        source_Ks = batch["source_Ks"][keep_mask_tensor].to(device, dtype=weight_dtype)
        target_Ks = batch["target_Ks"][keep_mask_tensor].to(device, dtype=weight_dtype)
        # Keep pipeline inputs, cached predictions, labels, and masks at one size.
        if source.shape[-2:] != inference_size:
            source = resize_5d(source, size=inference_size)
            lighting = resize_5d(lighting, size=inference_size)
        if target.shape[-2:] != inference_size:
            target = resize_5d(target, size=inference_size)
        if mask.shape[-2:] != inference_size:
            mask = resize_5d(mask, size=inference_size)

        # Keep the original batch positions so metadata paths remain correct
        # after filtering samples that already exist in results_rank_*.json.
        kept_batch_positions = torch.nonzero(
            keep_mask_tensor, as_tuple=False
        ).flatten().tolist()

        num_views = target.shape[1]
        sample_metas = []
        sample_output_dirs = []

        # Resolve every expected pred_relight.png path before inference.
        for b in range(B):
            sample_idx = int(idx[b].item())

            if isinstance(data_pairs, list):
                meta = data_pairs[sample_idx] if 0 <= sample_idx < len(data_pairs) else None
            else:
                meta = data_pairs.get(str(sample_idx)) if isinstance(data_pairs, dict) else None

            original_batch_pos = kept_batch_positions[b]
            if "meta" in batch:
                subfolder_name = str(batch["meta"][original_batch_pos])
            else:
                subfolder_name = str(sample_idx)

            output_dirs = []
            for v in range(num_views):
                view_name = (
                    meta["view"][v].split(".")[0]
                    if meta and "view" in meta
                    else f"view_{v}"
                )
                output_dirs.append(
                    (Path(args.output_dir) / subfolder_name / view_name).resolve()
                )

            sample_metas.append(meta)
            sample_output_dirs.append(output_dirs)

        # 5. Load existing predictions and infer only missing samples.
        skip_exist = getattr(args, "skip_exist", False)
        output_by_sample = [None] * B
        loaded_from_disk = [False] * B
        infer_positions = []

        for b, output_dirs in enumerate(sample_output_dirs):
            pred_paths = [out_dir / "pred_relight.png" for out_dir in output_dirs]

            # A multi-view sample is reusable only when every expected view exists.
            if skip_exist and all(pred_path.is_file() for pred_path in pred_paths):
                output_by_sample[b] = torch.stack(
                    [
                        load_pred_image(
                            pred_path,
                            target_size=inference_size,
                            device=device,
                            dtype=weight_dtype,
                        )
                        for pred_path in pred_paths
                    ],
                    dim=0,
                )
                loaded_from_disk[b] = True
                logger.info(
                    f"Loaded existing pred_relight.png files for sample "
                    f"{int(idx[b].item())}; pipeline inference skipped."
                )
            else:
                infer_positions.append(b)

        if infer_positions:
            infer_index = torch.tensor(
                infer_positions, device=device, dtype=torch.long
            )

            with torch.autocast("cuda", dtype=weight_dtype):
                generated_output = pipe(
                    source=source[infer_index],
                    lighting=lighting[infer_index],
                    source_view=source_view[infer_index],
                    target_view=target_view[infer_index],
                    source_Ks=source_Ks[infer_index],
                    target_Ks=target_Ks[infer_index],
                    cfg_scale=args.guidance_scale,
                    denoising_strength=args.denoising_strength,
                    seed=cfg.seed,
                )

            for generated_pos, batch_pos in enumerate(infer_positions):
                output_by_sample[batch_pos] = generated_output[generated_pos].to(device=device, dtype=weight_dtype)

        output = torch.stack(output_by_sample, dim=0)

        # 6. Metric Calculation
        res = metric_calculator(
                outputs=output, 
                labels=target, 
                mask_gt=mask, 
                average=False)
        
        # Unpack 11 metrics
        (b_psnr, b_spsnr, b_ssim, b_lpips, 
         b_psnr_mask, b_spsnr_mask, b_ssim_mask, b_lpips_mask,
         b_d_acc, b_d_mse, b_m_iou) = res

        # 7. Per-sample Storage & Logging
        for b in range(B):
            sample_idx = int(idx[b].item())
            
            # Metadata and output directories were resolved before inference.
            meta = sample_metas[b]
            num_views = output.shape[1]

            current_eval = {
                "sample_idx": sample_idx,
                "object": meta.get("object") if meta else None,
                "psnr": float(b_psnr[b]), 
                "spsnr": float(b_spsnr[b]), 
                "ssim": float(b_ssim[b]), 
                "lpips": float(b_lpips[b]),
                "psnr_mask": float(b_psnr_mask[b]) if b_psnr_mask[b] is not None else None,
                "spsnr_mask": float(b_spsnr_mask[b]) if b_spsnr_mask[b] is not None else None,
                "ssim_mask": float(b_ssim_mask[b]) if b_ssim_mask[b] is not None else None,
                "lpips_mask": float(b_lpips_mask[b]) if b_lpips_mask[b] is not None else None,
                "pred_image": [], 
                "gt_image": []
            }

            # Save generated files per view. Existing predictions remain untouched.
            for v in range(num_views):
                out_dir = sample_output_dirs[b][v]
                out_dir.mkdir(parents=True, exist_ok=True)

                pred_path = out_dir / "pred_relight.png"
                if not loaded_from_disk[b]:
                    save_image(output[b, v], pred_path, normalize=False)
                current_eval["pred_image"].append(str(pred_path))
                
                # GT and reference were saved before skip/resume filtering.
                # save_image(lighting[b, 0], out_dir / "target_lighting.png", normalize=False)

                if 'target_frame_path' in batch:
                    if args.dataset in ["Stanford-ORB", "Stanford-ORBGS"]:
                        current_eval["gt_image"].append(str(batch['target_frame_path'][v][b]))
                    else:
                        current_eval["gt_image"].append(str(batch['target_frame_path'][keep_mask_tensor][v][b]))

            # 8. Real-time Aggregation
            evaluation_results["data_pair"][str(sample_idx)] = current_eval
            all_samples = list(evaluation_results["data_pair"].values())
            
            keys_to_avg = ["psnr", "spsnr", "ssim", "lpips", "psnr_mask", "spsnr_mask", "ssim_mask", "lpips_mask"]
            evaluation_results["average"] = {
                k: float(np.mean([s[k] for s in all_samples if s.get(k) is not None]))
                for k in keys_to_avg if any(s.get(k) is not None for s in all_samples)
            }

        # Update progress bar
        avg = evaluation_results["average"]
        bar.set_postfix({
            "PSNR": f"{avg.get('psnr', 0):.2f}",
            "PSNR_m": f"{avg.get('psnr_mask', 0):.2f}" if avg.get('psnr_mask') is not None else "N/A",
            "SSIM": f"{avg.get('ssim', 0):.3f}",
            "LPIPS": f"{avg.get('lpips', 0):.3f}"
        })
        
        # Incremental save
        with open(res_json_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)

        # Optional: Clear cache periodically to prevent OOM on long evals
        torch.cuda.empty_cache()

    logger.info(f"✅ Validation Rank {proc_id} Finished. Results at {res_json_path}")
    return evaluation_results


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main(args: Any):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout= timedelta(seconds=NCCL_TIMEOUT))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # 1. Model Initialization
    pipe = RelightFormerPipeline.from_pretrained(args.from_pretrained, revision=cfg.revision, torch_dtype=getattr(torch, cfg.torch_dtype))
    pipe.to(accelerator.device)
    pipe.requires_grad_(False)
    pipe.eval()
    pipe.load_prompt_dict(path=cfg.prompt_path)

    def print_model_info(model: torch.nn.Module):
        if accelerator.is_main_process and model is not None:
            logger.info("=" * 40)
            logger.info(f"Model: {type(model).__name__}")
            logger.info(f"Learnable params (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}")
            logger.info(f"Non-learnable params (M): {sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6:.2f}")
            logger.info(f"Total params (M): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}")
            logger.info(f"Model size (MB): {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024:.2f}")
            logger.info("=" * 40)

    print_model_info(pipe.dit)
    print_model_info(pipe.vae)
    # 2. Dataset Preparation
    dataset = LODEvalDataset(
        args.dataset_path,
        args.pair_info,
        resolution=(512, 512)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=0 # Increased from 0 for better performance
    )
    
    pipe.dit, dataloader = accelerator.prepare(pipe.dit, dataloader)

    # 3. Precision Setup
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if pipe.vae is not None:
        pipe.vae.to(dtype=torch.float16) # VAE is typically safe at fp16 for inference

    # 5. Run Evaluation
    _ = log_validation(
        validation_dataloader=dataloader,
        pipe=pipe,
        args=args,
        accelerator=accelerator,
        weight_dtype=weight_dtype,
        cur_step=0
    )

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    torch.manual_seed(6)
    np.random.seed(66)

    parser = argparse.ArgumentParser(description="RelightFormer Evaluation Script")
    parser.add_argument("--from_pretrained", type=str, default='vLAR/LavalObjaverseDataset/checkpoints')
    parser.add_argument("--revision", type=str, default="main", help="Specify a revision name to load from the pretrained directory.")
    parser.add_argument("--output_dir", type=str, default='./output/')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default="./laval-objaverse-dataset")
    parser.add_argument("--pair_info", type=str, default='./laval-objaverse-dataset/paris/16_to_16_mapping_pairs.json')
    parser.add_argument("--skip_exist", action='store_true', help="Skip samples already present in results JSON")
    parser.add_argument("--save_gt", action='store_true')
    parser.add_argument("--save_ref", action='store_true')
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--denoising_strength", type=float, default=1.0)

    cli_args = parser.parse_args()
    
    # Load config from the pretrained directory

    # Construct clean output directory name
    cli_args.output_dir = os.path.join(cli_args.output_dir, ckpt_str)
    os.makedirs(cli_args.output_dir, exist_ok=True)

    main(cli_args)