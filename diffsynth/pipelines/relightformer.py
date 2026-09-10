import torch.nn as nn
from ..models import ModelManager
from ..models.relightformer_dit import RelightFormerModel
from ..models.wan_video_vae import WanVideoVAE
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from typing import Optional
from pathlib import Path
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from tqdm import tqdm

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.relightformer_dit import RMSNorm, sinusoidal_embedding_1d, LightingPatchifier, RaysEmbedding, ChannelMergeModule, LightingAttention
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from contextlib import redirect_stdout
from ..rope import RopeDotProductAttention
def suppress_output(func):
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                return func(*args, **kwargs)
    return wrapper

DIVISION_FACTOR = 16
class RelightFormerPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit: RelightFormerModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['dit', 'vae']

    @classmethod
    def init_from_wan(cls, wan_path: str, 
                      channel_merge=False, rope="PRoPE", 
                      device='cuda', torch_dtype=torch.float16):
        """Initialize the model from a pre-trained Wan model checkpoint."""
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device)
        dit_path = f"{wan_path}/diffusion_pytorch_model.safetensors"
        vae_path = f"{wan_path}/Wan2.1_VAE.pth"

        model_manager.load_models([dit_path, vae_path])
        
        # Initialize from Wan Video Model
        wan = cls.from_model_manager(model_manager, device=device, torch_dtype=torch_dtype)
        pipe = RelightFormerPipeline(device=device, torch_dtype=torch_dtype)
        
        pipe.vae = wan.vae
        pipe.dit = RelightFormerModel.init_from_wan_model(wan.dit, 
                                                          division_factor=DIVISION_FACTOR, 
                                                          rope=rope)
        pipe.scheduler = wan.scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        if channel_merge:
            print("This is ablation version with channel-wise merging of lighting and video tokens.")
            pipe.dit.channel_wise_merger = ChannelMergeModule(c1=16, c2=2 * 16)  # 16 = vae dim
            pipe.dit.channel_wise_merger.xavier_init()
            for block in pipe.dit.blocks:
                block.lighting_attn = None
        
        return pipe
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "vLAR/RelightFormer",
        revision: str = "main",
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
        hub_revision: str = "main",
        resolution: int = 256,
    ):
        """Load RelightFormer directly from its released checkpoint.

        ``revision`` selects the weight file (``main-model.safetensors`` or
        ``post-model.safetensors``). ``hub_revision`` independently selects the
        Hugging Face Git branch/tag/commit and normally remains ``main``.

        Unlike the old implementation, this method does not load the checkpoint as
        ``wan_video_dit`` and does not call ``init_from_wan_model``. It constructs
        the final RelightFormer architecture and streams every trained tensor from
        the checkpoint directly into that model.
        """
        if revision not in {"main", "post"}:
            raise ValueError(
                f"Unsupported revision {revision!r}; expected 'main' or 'post'."
            )
        if resolution != 256:
            raise ValueError(
                "The released RelightFormer checkpoints use 256x256 inference."
            )

        model_source = Path(pretrained_model_name_or_path).expanduser()
        checkpoint_filename = f"{revision}-model.safetensors"
        vae_filename = "Wan2.1_VAE.pth"

        if model_source.is_dir():
            local_root = model_source.resolve()
            checkpoint_candidates = (
                local_root / checkpoint_filename,
                local_root / "checkpoints" / revision / checkpoint_filename,
            )
            vae_candidates = (
                local_root / vae_filename,
                local_root / "checkpoints" / "vae" / vae_filename,
            )
            checkpoint_path = next(
                (path for path in checkpoint_candidates if path.is_file()), None
            )
            vae_path = next((path for path in vae_candidates if path.is_file()), None)

            if checkpoint_path is None:
                checked = "\n  - ".join(str(path) for path in checkpoint_candidates)
                raise FileNotFoundError(
                    f"RelightFormer checkpoint not found. Checked:\n  - {checked}"
                )
            if vae_path is None:
                checked = "\n  - ".join(str(path) for path in vae_candidates)
                raise FileNotFoundError(
                    f"Wan VAE checkpoint not found. Checked:\n  - {checked}"
                )
        else:
            print(
                f"Downloading {checkpoint_filename} from "
                f"{pretrained_model_name_or_path}@{hub_revision} ..."
            )
            checkpoint_path = Path(
                hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=checkpoint_filename,
                    cache_dir=cache_dir,
                    revision=hub_revision,
                )
            )

            print(
                f"Downloading {vae_filename} from "
                f"{pretrained_model_name_or_path}@{hub_revision} ..."
            )
            vae_path = Path(
                hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=vae_filename,
                    cache_dir=cache_dir,
                    revision=hub_revision,
                )
            )

        device = torch.device(device)

        # Construct the final RelightFormer architecture on the meta device so no
        # temporary randomly initialized 1.3B-parameter model consumes CPU/GPU RAM.
        with init_empty_weights():
            relightformer = RelightFormerModel(
                dim=1536,
                in_dim=16,
                ffn_dim=8960,
                out_dim=16,
                freq_dim=256,
                eps=1e-6,
                patch_size=(1, 2, 2),
                num_heads=12,
                num_layers=30,
                has_image_input=False,
            )

            # These final RelightFormer modules must exist before loading so their
            # trained checkpoint tensors are restored rather than discarded.
            for block in relightformer.blocks:
                block.lighting_attn = LightingAttention(
                    block.self_attn.dim,
                    block.self_attn.head_dim,
                    eps=relightformer.eps,
                )
                block.self_attn.attn = RopeDotProductAttention(
                    block.self_attn.head_dim,
                    patches_x=resolution // 16,
                    patches_y=resolution // 16,
                    image_width=resolution,
                    image_height=resolution,
                    rope="PRoPE",
                )

        # Verify an exact architecture/checkpoint match, then stream one tensor at a
        # time to the target device. There is no intermediate Wan DiT model.
        expected_keys = set(relightformer.state_dict().keys())
        with safe_open(
            str(checkpoint_path), framework="pt", device="cpu"
        ) as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
            missing_keys = sorted(expected_keys - checkpoint_keys)
            unexpected_keys = sorted(checkpoint_keys - expected_keys)

            if missing_keys or unexpected_keys:
                messages = []
                if missing_keys:
                    messages.append(f"missing keys: {missing_keys[:20]}")
                if unexpected_keys:
                    messages.append(f"unexpected keys: {unexpected_keys[:20]}")
                raise RuntimeError(
                    "Checkpoint does not exactly match RelightFormer: "
                    + "; ".join(messages)
                )

            for tensor_name in tqdm(
                checkpoint.keys(),
                desc=f"Loading {revision} checkpoint",
                dynamic_ncols=True,
            ):
                tensor = checkpoint.get_tensor(tensor_name)
                set_module_tensor_to_device(
                    relightformer,
                    tensor_name,
                    device,
                    value=tensor,
                    dtype=torch_dtype if tensor.is_floating_point() else None,
                )

        relightformer.requires_grad_(False)
        relightformer.eval()

        # Only the VAE uses ModelManager; the RelightFormer checkpoint does not.
        vae_dtype = torch.float32 if torch_dtype == torch.float32 else torch.float16
        model_manager = ModelManager(torch_dtype=vae_dtype, device=device)
        model_manager.load_models([str(vae_path)])
        vae = model_manager.fetch_model("wan_video_vae")
        if vae is None:
            raise RuntimeError(f"Unable to load Wan VAE checkpoint: {vae_path}")
        vae.requires_grad_(False)
        vae.eval()

        pipe = cls(device=device, torch_dtype=torch_dtype)
        pipe.dit = relightformer
        pipe.vae = vae
        pipe.scheduler = FlowMatchScheduler(
            shift=5,
            sigma_min=0.0,
            extra_one_step=True,
        )
        pipe.scheduler.set_timesteps(1000, training=True)
        pipe.requires_grad_(False)
        pipe.eval()
        return pipe
    
    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = RelightFormerPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    

    def denoising_model(self):
        return self.dit

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    def lighting_tunemap(self, raw):
        assert raw.ndim == 5 and raw.size()[1] == 1, f"Not support lighting input size {raw.size()}, expect [B 2 C H W] or [B 1 C H W]" 

        M_ldr = 16
        M_log = 10_000

        ldr = raw / (1.0 + raw) * (1.0 + raw / M_ldr**2)
        log = torch.log(1.0 + raw) / torch.log(torch.tensor(1.0 + M_log, device=raw.device, dtype=raw.dtype))

        return torch.cat([ldr, log], dim=1)

    @torch.no_grad()
    def encode_video_in_frame(self, video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16), rescale=True):
        # input/output [B, F, C, H, W]
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        b, f, c, h, w = video.size()
        video = video.reshape(-1, c, 1, h, w).contiguous()
        if rescale:
            video = video * 2.0 - 1.0
        latents = self.encode_video(video, **tiler_kwargs)
        _, c, _, h, w = latents.size()
        latents = latents.reshape(b, f, c, h, w)
        return latents
    
    @torch.no_grad()
    def decode_video_in_frame(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16), rescale=True):
        # input/output [B, F, C, H, W]
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        b, f, c, h, w = latents.size()
        latents = latents.reshape(-1, c, 1, h, w).contiguous()
        video = self.decode_video(latents, **tiler_kwargs)
        _, c, _, h, w = video.size()
        video = video.reshape(b, f, c, h, w)
        if rescale:
            video = 0.5 * (video + 1.0)
        return video

    def load_prompt_dict(self, path):
        self.prompt_dict = torch.load(path)
        return self

    def get_prompt_emb(self, prompt, batch_size, positive=True):
        word = "positive" if positive else "negative"
        emb = self.prompt_dict[word][prompt]
        return torch.cat([emb]*batch_size, dim=0)
    
    @property
    def training_prompt_emb(self):
        return self.prompt_dict["positive"]["边缘清晰，黑色背景，高质量图片，objaverse数据集，纹理清晰"]

    @torch.no_grad()
    def __call__(
        self,
        source, # B, F, C, H, W
        lighting, # B, N, C, H, W
        source_view,
        target_view,
        source_Ks,
        target_Ks,
        lighting_rays=None,
        target=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        cfg_scale=3.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(32, 32),
        tile_stride=(16, 16),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        **kwargs
    ):
        B, F, C, H, W = source.size()
        source_rays = camera_ray(source_view, source_Ks).to(dtype=self.torch_dtype, device=self.device)
        target_rays = camera_ray(target_view, target_Ks).to(dtype=self.torch_dtype, device=self.device)
        
        if lighting_rays is None: 
            lighting_rays = equirectangular_ray().to(dtype=self.torch_dtype, device=self.device)
            lighting_rays = lighting_rays.unsqueeze(1)
            lighting_rays = lighting_rays.unsqueeze(0).expand(B, -1, -1, -1, -1) 
        batch_size, source_num_frames, _, _, _ = source_rays.size()
        target_num_frames = target_rays.size()[1]

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        # Encode source video
        self.load_models_to_device(['vae'])
        source = source.to(dtype=self.torch_dtype, device=self.device)
        source_latents = self.encode_video_in_frame(source, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        source_latents = rearrange(source_latents, 'b f c h w -> b c f h w').contiguous()

        if lighting.size()[1] == 2: # lighting has been tunemapped
            pass
        else:
            lighting = self.lighting_tunemap(lighting)

        lighting_latents = self.encode_video_in_frame(lighting, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        lighting_latents = rearrange(lighting_latents, 'b f c h w -> b c f h w').contiguous()

        # Initialize noise
        noise_shape = (*source_latents.size()[:2], target_num_frames, *source_latents.size()[3:])
        noise = self.generate_noise(noise_shape, seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)

        if target is None:
            latents = noise
        else:
            target = target.to(dtype=self.torch_dtype, device=self.device)
            target_latents = self.encode_video_in_frame(target, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            target_latents = rearrange(target_latents, 'b f c h w -> b c f h w').contiguous()
            latents = self.scheduler.add_noise(target_latents, noise, self.scheduler.timesteps[0])
        
        # rays
        rays = torch.cat([target_rays, source_rays], dim=1)
        rays = rearrange(rays, 'b f c h w -> b c f h w').contiguous()

        # Cameras:
        Ts = torch.cat([target_view, source_view], dim=1)
        Ks = torch.cat([target_Ks, source_Ks], dim=1)
        
        # Extra input
        extra_input = self.prepare_extra_input(latents) # actually nothing
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device(["dit"])
        tgt_latent_length = latents.shape[2]

        for progress_id, timestep in enumerate(self.scheduler.timesteps):
        # for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # x = torch.cat([latents]*1, dim=0)
            x = latents
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            # Inference
            condition = {"rays": rays,
                        "lighting": lighting_latents,
                        "lighting_rays": lighting_rays,
                        "Ts": Ts,
                        "Ks": Ks,}
            
            noise_pred_posi = model_fn_wan_video(self.dit, torch.cat([x, source_latents], dim=2), 
                                            timestep=timestep, 
                                            **condition, 
                                            **extra_input, 
                                            **tea_cache_posi)[:,:,:tgt_latent_length,...]
            if cfg_scale != 1.0:
                condition = {"rays": rays,
                        "lighting": torch.zeros_like(lighting_latents),
                        "lighting_rays": torch.zeros_like(lighting_rays),
                        "Ts": Ts,
                        "Ks": Ks,}
                noise_pred_nega = model_fn_wan_video(self.dit, torch.cat([x, source_latents], dim=2),
                                            timestep=timestep, 
                                            **condition, 
                                            **extra_input, 
                                            **tea_cache_nega)[:,:,:tgt_latent_length,...]
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            # Scheduler
            latents = self.scheduler.step(noise_pred, 
                                          self.scheduler.timesteps[progress_id], 
                                          latents)

        # Decode
        latents = rearrange(latents, 'b c f h w -> b f c h w').contiguous()
        self.load_models_to_device(['vae'])
        frames = self.decode_video_in_frame(latents, **tiler_kwargs)

        return frames


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: RelightFormerModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states

def model_fn_wan_video(
    dit: RelightFormerModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    lighting: torch.Tensor,
    rays: torch.Tensor,
    lighting_rays: torch.Tensor,
    Ts: torch.Tensor,
    Ks: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    x = x + dit.rays_embedding(rays)
    if lighting_rays is not None and lighting is not None:
        lighting = lighting + dit.rays_embedding(lighting_rays)

    if hasattr(dit, 'channel_wise_merger') and lighting is not None:
        B, _, _, H, W = x.shape
        lighting = lighting.view(B, -1, 1, H, W)
        x = dit.channel_wise_merger(x, lighting)
        lighting = None  # merged into x, no separate lighting tokens
    if lighting is not None:
        lighting = dit.lighting_patchifier(lighting)
    
    x, (f, h, w) = dit.patchify(x)

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    tea_cache=None
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod) # TODO: is it right?
    else:
        tea_cache_update = False
    
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        # blocks
        for block in dit.blocks:
            x = block(x, t_mod, freqs, lighting, Ts, Ks)
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x

def camera_ray(Ts, Ks, H=256, W=256, device='cpu'):
    """
    T K -> Rays (Plücker Coordinates: m, d)。
    
    Args:
        Ts: [B, F, 4, 4] 
        Ks: [B, F, 3, 3]
    """
    B, F, _, _ = Ts.shape
    device = Ts.device

    # Flatten B and F to process all views at once
    Ts = Ts.reshape(-1, 4, 4) # [B*F, 4, 4]
    Ks = Ks.reshape(-1, 3, 3) # [B*F, 3, 3]

    N = Ts.shape[0]
    # 1. 創建像素坐標網格 (Pixel Grid)
    # y 對應 H, x 對應 W
    y_range = torch.arange(H, dtype=torch.float32, device=device)
    x_range = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij') # [H, W]

    # 2. 提取內參參數
    # K 矩陣格式: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    fx = Ks[:, 0, 0].view(N, 1, 1)
    fy = Ks[:, 1, 1].view(N, 1, 1)
    cx = Ks[:, 0, 2].view(N, 1, 1)
    cy = Ks[:, 1, 2].view(N, 1, 1)

    # 3. 將像素坐標轉換為相機坐標系下的方向 (Camera Space)
    # 根據公式: x_cam = (x_pixel - cx) / fx, y_cam = (y_pixel - cy) / fy
    # z 方向在相機空間通常定義為 1
    x_cam = (x_grid.unsqueeze(0) - cx) / fx
    y_cam = (y_grid.unsqueeze(0) - cy) / fy
    z_cam = torch.ones_like(x_cam)

    directions_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1) # [N, H, W, 3]

    # 4. 歸一化方向向量 (Normalize)
    directions_cam = directions_cam / torch.norm(directions_cam, dim=-1, keepdim=True)
    # 5. 轉換到世界坐標系 (World Space)
    R = Ts[:, :3, :3]  # [N, 3, 3]
    t = Ts[:, :3, 3]   # [N, 3] (Camera Origin)

    # 使用矩陣乘法旋轉方向向量: [N, H*W, 3] @ [N, 3, 3]^T
    directions_world = torch.matmul(directions_cam.view(N, -1, 3), R.transpose(-2, -1))
    directions_world = directions_world.view(N, H, W, 3)

    # 6. 計算普呂克坐標之矩 (Plücker Momentum: m = o x d)
    camera_pos = t.view(N, 1, 1, 3).expand(-1, H, W, -1)
    ray_momentum = torch.cross(camera_pos, directions_world, dim=-1)

    # 7. 拼接並調整維度
    # 返回 [N, 6, H, W] -> (mx, my, mz, dx, dy, dz)
    rays = torch.cat([ray_momentum, directions_world], dim=-1).permute(0, 3, 1, 2) # [N, 6, H, W]
    rays = rays.view(B, F, 6, H, W) # [B, F, 6, H, W]
    return rays

def equirectangular_ray(
    resolution=256,
    addition_rotation=None,
    device="cpu",
):
    """
#     Generate rays for equirectangular projection, where rays go from pi to -pi horizontally
#     and +pi/2 to -pi/2 vertically.
#     - H: height of the image
#     - W: width of the image
#     """
    H = W = resolution
    dtype = torch.float32

    lat_step = torch.pi / H
    lng_step = 2 * torch.pi / W

    # Pixel-center latitude: top -> bottom, +pi/2 -> -pi/2.
    lat = torch.linspace(
        torch.pi / 2 - 0.5 * lat_step,
        -torch.pi / 2 + 0.5 * lat_step,
        H,
        device=device,
        dtype=dtype,
    )

    lon = torch.linspace(
        torch.pi - 0.5 * lng_step,
        -torch.pi + 0.5 * lng_step,
        W,
        device=device,
        dtype=dtype,
    )

    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")  # [H, W]

    cos_lat = torch.cos(lat_grid)

    # Original spherical / z-up convention:
    # lon = 0 -> +x (center of image)
    # lon = +pi/2 -> +y (left of image)
    # top -> +z (top of image)
    ray_dirs = torch.stack(
        [
            cos_lat * torch.cos(lon_grid),
            cos_lat * torch.sin(lon_grid),
            torch.sin(lat_grid),
        ],
        dim=-1,
    )  # [H, W, 3]

    # Optional active rotation. Row-vector version: d' = d @ R.T.
    if addition_rotation is not None:
        addition_rotation = addition_rotation.to(device=device, dtype=dtype)
        ray_dirs = ray_dirs.reshape(-1, 3) @ addition_rotation.T
        ray_dirs = ray_dirs.reshape(H, W, 3)

    # Forward-facing camera / x-right, y-down, z-forward convention (OpenCV):
    _to_opencv = torch.tensor(
        [
            [0.0,  1.0,  0.0],
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0],
        ],
        device=device,
        dtype=dtype,
    )
    ray_dirs = ray_dirs.reshape(-1, 3) @ _to_opencv.T
    ray_dirs = ray_dirs.reshape(H, W, 3)
    # Make sure directions are unit length after all transforms.
    ray_dirs = ray_dirs / torch.linalg.norm(ray_dirs, dim=-1, keepdim=True).clamp_min(1e-8)

    # Environment rays have zero origin here, so Plücker moment m = o x d = 0.
    ray_momentum = torch.zeros_like(ray_dirs)

    # Match camera2ray format: [m, d], then [N, 6, H, W].
    rays = torch.cat([ray_momentum, ray_dirs], dim=-1)  # [H, W, 6]
    rays = rays.permute(2, 0, 1)           # [6, H, W]
    return rays
