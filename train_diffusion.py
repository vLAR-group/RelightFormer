import logging
import math
import os
import shutil
import warnings
from datetime import timedelta
from pathlib import Path

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
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available, is_xformers_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm

# Local imports
from datasets.LavalObjaverseDataset import LavalObjaverseDataset as Dataset
from datasets.LavalObjaverseDataset import LavalObjaverseEvalDataset as LODEvalDataset
from datasets.TensoIR import TensoIREvalDataset as TensoEvalDataset

from utils.metrics import MetricCalculator, resize_5d
from utils.args import read_yaml_to_namespce
from utils.training_utils import (
    combine_dataloaders,
    data_preprocess,
    dataloader_maker,
    get_training_dataset,
    get_validation_datasets,
    save_model_card,
    split_loss,
    create_log_images
)

# Pipeline imports
from diffsynth.pipelines import RelightFormerPipeline

# -----------------------------------------------------------------------------
# Environment & Warning Configuration
# -----------------------------------------------------------------------------
os.environ['HF_HOME'] = './hf_cache'
# os.environ['NCCL_P2P_DISABLE'] = '1' 
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["FLASH_ATTENTION_FORCE_V3"] = "1"

if is_wandb_available():
    import wandb
    os.environ['WANDB_CONFIG_DIR'] = f"/tmp/.config-{os.environ.get('USER', 'user')}"

NCCL_TIMEOUT = 360_000
NUM_VALIDATION = 256

# Suppress known benign warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings("ignore", message="cc_projection/diffusion_pytorch_model.safetensors not found")
warnings.filterwarnings("ignore", message="The config attributes {'cc_projection':.*")
warnings.filterwarnings("ignore", message=".*is_pinned.*device.*", category=DeprecationWarning, module="torch")
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Validation & Logging
# -----------------------------------------------------------------------------
@torch.no_grad()
def log_validation(validation_dataloader, vae, dit, args, accelerator, weight_dtype, split="val", cur_step=0):
    logger.info(f"Running {split} validation...")
    
    metric_calculator = MetricCalculator(device=accelerator.device)
    
    # Setup pipeline for evaluation
    pipe = RelightFormerPipeline(device=accelerator.device, torch_dtype=torch.bfloat16)
    pipe.load_prompt_dict(path=args.prompt_path)
    pipe.vae = vae.eval()
    pipe.dit = dit.eval()

    length = len(validation_dataloader)
    log_image_num = 16
    log_image_rate = max(1, length // log_image_num)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    
    image_logs = []
    measurements = {
        "image": {"psnr": [], "spsnr": [], "ssim": [], "lpips": [], "psnr_mask": [], "spsnr_mask": [], "ssim_mask": [], "lpips_mask": []},
        "mask": {"iou": []}
    }

    for valid_step, batch in tqdm(enumerate(validation_dataloader), desc=split, total=length, disable=not accelerator.is_local_main_process):
        # Move to device
        source = batch['source_images'].to(accelerator.device, dtype=weight_dtype)
        source_mask = batch['source_mask'].to(accelerator.device, dtype=weight_dtype)
        target = batch['target_images'].to(accelerator.device, dtype=weight_dtype)
        source_lighting = batch["source_lighting"].to(accelerator.device, dtype=weight_dtype)
        lighting = batch['target_lighting'].to(accelerator.device, dtype=weight_dtype)
        source_view = batch["source_view"].to(accelerator.device, dtype=weight_dtype)
        target_view = batch["target_view"].to(accelerator.device, dtype=weight_dtype)
        source_Ks = batch["source_Ks"].to(accelerator.device, dtype=weight_dtype)
        target_Ks = batch["target_Ks"].to(accelerator.device, dtype=weight_dtype)

        if args.resolution != source.shape[2]:
            source = resize_5d(source, size=(args.resolution, args.resolution))
            lighting = resize_5d(lighting, size=(args.resolution, args.resolution))

        with torch.autocast("cuda", dtype=weight_dtype):
            output = pipe(
                source=source,
                lighting=lighting,
                source_view=source_view,
                target_view=target_view,
                source_Ks=source_Ks,
                target_Ks=target_Ks,
                cfg_scale=args.guidance_scale,
                seed=args.seed,
                generator=generator
            )

        # Calculate metrics
        res = metric_calculator(
            outputs=output,            
            labels=target,
            mask_gt=source_mask,
            average=False
        )

        # Unpack metrics (ensure this matches MetricCalculator output order)
        (b_psnr, b_spsnr, b_ssim, b_lpips, 
         b_psnr_mask, b_spsnr_mask, b_ssim_mask, b_lpips_mask,
         b_d_acc, b_d_mse, b_m_iou) = res

        # Store metrics
        measurements["image"]["psnr"].extend(b_psnr)
        measurements["image"]["spsnr"].extend(b_spsnr)  
        measurements["image"]["ssim"].extend(b_ssim)
        measurements["image"]["lpips"].extend(b_lpips)

        measurements["image"]["psnr_mask"].extend([v for v in b_psnr_mask if v is not None])
        measurements["image"]["spsnr_mask"].extend([v for v in b_spsnr_mask if v is not None])
        measurements["image"]["ssim_mask"].extend([v for v in b_ssim_mask if v is not None])
        measurements["image"]["lpips_mask"].extend([v for v in b_lpips_mask if v is not None])

        # Log images periodically
        if valid_step % log_image_rate == 0:
            caption_lines = []
            for j in range(output.size(0)):
                p_m = f"{b_psnr_mask[j]:.2f}" if b_psnr_mask[j] is not None else "N/A"
                ss_m = f"{b_ssim_mask[j]:.4f}" if b_ssim_mask[j] is not None else "N/A"
                l_m = f"{b_lpips_mask[j]:.4f}" if b_lpips_mask[j] is not None else "N/A"
                
                caption_lines.append(
                    f"Sample {j}: PSNR={b_psnr[j]:.2f}/{p_m}, SSIM={b_ssim[j]:.4f}/{ss_m}, LPIPS={b_lpips[j]:.4f}/{l_m}"
                )
            
            lighting_log = torch.cat([resize_5d(source_lighting), resize_5d(lighting)], dim=1)
            result_img = create_log_images(source=source, target=target, pred=output, lighting_log=lighting_log)

            image_logs.append({"result": result_img, "caption": "\n".join(caption_lines)})

    # Aggregate metrics
    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    val_metrics = {
        f"PSNR/{split}": safe_mean(measurements["image"]["psnr"]),
        f"sPSNR/{split}": safe_mean(measurements["image"]["spsnr"]),
        f"SSIM/{split}": safe_mean(measurements["image"]["ssim"]),
        f"LPIPS/{split}": safe_mean(measurements["image"]["lpips"]),
        f"PSNR_mask/{split}": safe_mean(measurements["image"]["psnr_mask"]),
        f"sPSNR_mask/{split}": safe_mean(measurements["image"]["spsnr_mask"]),
        f"SSIM_mask/{split}": safe_mean(measurements["image"]["ssim_mask"]),
        f"LPIPS_mask/{split}": safe_mean(measurements["image"]["lpips_mask"]),
    }

    # Log to trackers
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb" and image_logs:
                formatted_images = [wandb.Image(log["result"], caption=log["caption"]) for log in image_logs]
                tracker.log({split: formatted_images}, step=cur_step)

    # Restore training mode
    dit.train()
    return image_logs, val_metrics


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
def main(args):
    # 1. Accelerator & Logging Setup
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        static_graph=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=64,
    )
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=NCCL_TIMEOUT))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
        # diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        # diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token).repo_id

    # 2. Model Initialization & Freezing
    pipe = RelightFormerPipeline.init_from_wan(
        wan_path=args.wan_path,
        channel_merge=getattr(args, 'channel', False),
        rope=args.rope
        )
    pipe.to(accelerator.device)
    pipe.load_prompt_dict(path=args.prompt_path)
    pipe.requires_grad_(False)
    pipe.eval()
    
    # Unfreeze specific modules
    dit_model = pipe.denoising_model()
    dit_model.train()
    
    for name, module in dit_model.named_modules():
        for param in module.parameters():
            param.requires_grad = True

    training_params = list(filter(lambda p: p.requires_grad, dit_model.parameters()))

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
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            dit_model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Install it to use memory-efficient attention.")

    if args.gradient_checkpointing:
        dit_model.enable_gradient_checkpointing()

    # Ensure DiT is in float32 for stable training
    if accelerator.unwrap_model(dit_model).dtype != torch.float32:
        raise ValueError("DiT must be in float32 precision when starting training, even if using mixed precision.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 3. Optimizer & Scheduler
    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.training_batch_size * accelerator.num_processes

    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Install bitsandbytes to use 8-bit Adam: `pip install bitsandbytes`")

    optimizer = optimizer_class(
        [{"params": training_params, "lr": args.learning_rate}],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=float(args.adam_epsilon)
    )

    # 4. Datasets & Dataloaders
    train_dataset_dictionary = get_training_dataset(Dataset, args)
    train_dataloaders = []
    num_examples = 0
    train_dataloader_length = 0

    for train_dataset in train_dataset_dictionary.values():
        dl = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        train_dataloaders.append(dl)
        train_dataloader_length += len(dl)
        num_examples += len(train_dataset)

    valid_dataset_dictionary = get_validation_datasets(Dataset, args)
    dataloaders_for_log = dataloader_maker(valid_dataset_dictionary, mode='mini')
    
    # Main evaluation dataset
    main_eval_dataset = LODEvalDataset(
        args.dataset_path,
        args.dataset_path +'/experimental_pair/32_to_32_mapping_pairs.json',
        resolution=(512, 512)
    )
    indices = np.round(np.linspace(0, len(main_eval_dataset) - 1, NUM_VALIDATION)).astype(int)
    main_eval_dataset = torch.utils.data.Subset(main_eval_dataset, indices.tolist())
    main_evaluation_dataloader = torch.utils.data.DataLoader(main_eval_dataset, 
                                                             shuffle=False, 
                                                             batch_size=1, 
                                                             num_workers=0)

    # Scheduler math
    num_update_steps_per_epoch = math.ceil(train_dataloader_length / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * accelerator.num_processes * args.gradient_accumulation_steps,
    )

    # 5. Accelerator Preparation
    models_to_prepare = [dit_model, lr_scheduler] + train_dataloaders
    prepared = accelerator.prepare(*models_to_prepare)
    dit_model = prepared[0]
    lr_scheduler = prepared[1]
    train_dataloaders = prepared[2:]

    # Cast VAE to lower precision for inference
    vae_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
    if pipe.vae is not None:
        pipe.vae.to(dtype=vae_dtype)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 6. Tracking & Logging Setup
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        last_three_components = args.output_dir.split(os.sep)[-3:]
        output_basename = '_'.join(last_three_components)
        accelerator.init_trackers(
            args.tracker_project_name, 
            config=tracker_config, 
            init_kwargs={"wandb": {"name": output_basename}}
        )

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples:,}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    # 7. Resume from Checkpoint
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None

        if path is None:
            logger.info("Checkpoint does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            strict_load = False if args.ablation else True
            accelerator.load_state(os.path.join(args.output_dir, path), strict=strict_load)
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    # 8. Training Loop
    progress_bar = tqdm(
        range(initial_global_step, args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    tiler_kwargs = {"tiled": args.tiled, "tile_size": (32, 32), "tile_stride": (16, 16)}
    
    for epoch in range(first_epoch, args.num_train_epochs):
        dit_model.train()
        loss_epoch = 0.0
        num_train_elems = 0

        data_iter = combine_dataloaders(train_dataloaders)
        
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            with accelerator.accumulate(dit_model):
                # Preprocess and move to device
                batch = data_preprocess(batch, pipe, vae_dtype, **tiler_kwargs)
                
                source = batch["source_images"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                target = batch["target_images"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                lighting = batch["target_lighting"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                source_rays = batch["source_rays"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                target_rays = batch["target_rays"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                lighting_rays = batch["lighting_rays"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                source_view = batch["source_view"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                target_view = batch["target_view"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                source_Ks = batch["source_Ks"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                target_Ks = batch["target_Ks"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)

                batch_size, frame_size, C, H, W = source.size()

                source = source.permute(0, 2, 1, 3, 4).contiguous()
                target = target.permute(0, 2, 1, 3, 4).contiguous()

                lighting = lighting.permute(0, 2, 1, 3, 4).contiguous()

                noise = torch.randn_like(target)
                pipe.scheduler.set_timesteps(1000, training=True)
                timestep_id = torch.randint(0, pipe.scheduler.num_train_timesteps, (batch_size,))
                timestep = pipe.scheduler.timesteps[timestep_id].to(device=accelerator.device, dtype=torch.float32)
                
                noisy_latents = pipe.scheduler.add_noise(target, noise, timestep)
                tgt_latent_len = noisy_latents.shape[2]
                
                rays = torch.cat([target_rays, source_rays], dim=1).permute(0, 2, 1, 3, 4).contiguous()
                Ts = torch.cat([target_view, source_view], dim=1)
                Ks = torch.cat([target_Ks, source_Ks], dim=1)

                # Classifier-Free Guidance Dropout
                cfg_mask = (torch.rand(batch_size, device=accelerator.device) >= args.conditioning_dropout_prob).to(weight_dtype)
                lighting = cfg_mask.view(batch_size, 1, 1, 1, 1) * lighting
                lighting_rays = cfg_mask.view(batch_size, 1, 1, 1, 1) * lighting_rays

                noisy_latents = torch.cat((noisy_latents, source), dim=2)
                training_target = pipe.scheduler.training_target(target, noise, timestep)

                condition = {
                    "rays": rays,
                    "lighting": lighting,
                    "lighting_rays": lighting_rays,
                    "Ts": Ts,
                    "Ks": Ks,
                }
                extra_input = {}

                # Forward pass
                noise_pred = dit_model(
                    noisy_latents, 
                    timestep=timestep, 
                    **condition, 
                    **extra_input,
                    use_gradient_checkpointing=args.use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload
                )

                # Loss computation
                weights = pipe.scheduler.training_weight(timestep).view(batch_size, 1, 1, 1, 1)
                diff = weights * F.mse_loss(noise_pred[:, :, :tgt_latent_len].float(), training_target[:, :, :tgt_latent_len].float(), reduction='none')
                
                loss_dit_in_seen_view, loss_dit_in_novel_view = split_loss(diff, source_view, target_view)
                loss_dit = loss_dit_in_seen_view if args.training_in_same_view else (loss_dit_in_seen_view + loss_dit_in_novel_view)
                loss = loss_dit

                # Backward pass
                accelerator.backward(loss)
                grad_norm = torch.nn.utils.clip_grad_norm_(training_params, max_norm=float('inf'))
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(training_params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Logging
            loss_epoch += loss.detach().item()
            num_train_elems += 1

            logs = {
                "loss": loss.detach().item(),
                "loss_dit": loss_dit.detach().item(),
                "loss_dit_in_seen_view": loss_dit_in_seen_view.detach().item(),
                "loss_dit_in_novel_view": loss_dit_in_novel_view.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0], 
                "loss_epoch": loss_epoch / num_train_elems,
                "epoch": epoch,
                "grad_norm": grad_norm.detach().item(),
            }
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Checkpointing & Validation
            if accelerator.sync_gradients:
                step_log = {}
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 and global_step > 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = sorted([d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")], key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for removing_checkpoint in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                                    logger.info(f"Removed old checkpoint: {removing_checkpoint}")

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    def run_validation(dataloaders_dict):
                        for dl_name, dl in dataloaders_dict.items():
                            if dl is None:
                                continue
                            try:
                                _, temp_log = log_validation(
                                    dl, pipe.vae, 
                                    accelerator.unwrap_model(dit_model),
                                    args, accelerator, weight_dtype, split=dl_name, cur_step=global_step
                                )
                                step_log.update(temp_log)
                            except Exception as e:
                                logger.error(f"Validation failed on {dl_name}: {e}")
                    # Validation
                    if global_step == 100 or global_step % args.mini_validation_steps == 0 or global_step % args.validation_steps == 0:
                        if global_step % args.validation_steps == 0:
                            run_validation({"main_evaluation": main_evaluation_dataloader})
                        run_validation(dataloaders_for_log)
                        if step_log:
                            accelerator.log(step_log, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # 9. Final Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved final state to {save_path}")

        pipe.dit = accelerator.unwrap_model(dit_model)
        pipe.save_pretrained(args.output_dir)
        logger.info(f"Saved pipeline to {args.output_dir}")

        if args.push_to_hub:
            save_model_card(repo_id, image_logs=None, base_model=args.model_key, repo_folder=args.output_dir)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir, commit_message="End of training", ignore_patterns=["step_*", "epoch_*"])

    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    
    # Set base seeds
    torch.manual_seed(6)
    np.random.seed(66)

    parser = argparse.ArgumentParser(description="Training configuration parser")
    parser.add_argument("--config", type=str, default='./config.yaml')
    cli_args = parser.parse_args()
    
    args = read_yaml_to_namespce(cli_args.config)
    
    # Construct output directory name based on ablation flags
    if args.ablation:
        flags = []
        if args.training_in_same_view: flags.append('-s')
        if getattr(args, 'channel', False): flags.append('-h')
        if args.rope == 'WAN': flags.append('-wan')
        
        max_view = int(getattr(args, 'max_view', 16))
        if max_view != 16:
            flags.append(f"-v{max_view}")
            
        ablation_suffix = "_" + "_".join(flags) if flags else ""
        name = os.path.join(ablation_suffix.lstrip('_'))
    else:
        name = 'relightformer'
    
    args.output_dir = os.path.join(args.output_dir, name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Backup config
    shutil.copy(cli_args.config, os.path.join(args.output_dir, 'config.yaml'))

    main(args)