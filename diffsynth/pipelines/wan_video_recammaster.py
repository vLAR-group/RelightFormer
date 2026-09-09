from ..models import ModelManager
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
from ..utils.camera import batch_cam2pose_embedding
import json

from contextlib import redirect_stdout
def suppress_output(func):
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                return func(*args, **kwargs)
    return wrapper

class WanVideoReCamMasterPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16


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
        pipe = WanVideoReCamMasterPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    

    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}


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
    
    def encode_video_in_frame(self, video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        # input/output [B, F, C, H, W]
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        b, f, c, h, w = video.size()
        # video = video * 2.0 - 1.0
        video = video.reshape(-1, c, 1, h, w).contiguous()
        latents = self.encode_video(video, **tiler_kwargs)
        _, c, _, h, w = latents.size()
        latents = latents.reshape(b, f, c, h, w)
        return latents
    

    def decode_video_in_frame(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        # input/output [B, F, C, H, W]
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        b, f, c, h, w = latents.size()
        latents = latents.reshape(-1, c, 1, h, w).contiguous()
        video = self.decode_video(latents, **tiler_kwargs)
        _, c, _, h, w = video.size()
        video = video.reshape(b, f, c, h, w)
        # video = ( video + 1.0 ) / .0
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
        target_camera, # B, F, C=12
        lighting, # B, C, H, W
        target = None,  # B, F, C, H, W
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        prompt="边缘清晰，黑色背景，高质量图片，objaverse数据集，纹理清晰",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_video=True,
        encode_lighting=False
    ):
        # Parameter check
        batch_size, num_frames, _, _, _ = source.size()
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Encode source video (recammaster)
        if input_video:
            source_latents = source
        else:
            self.load_models_to_device(['vae'])
            source = source.to(dtype=self.torch_dtype, device=self.device)
            source_latents = self.encode_video_in_frame(source, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
        source_latents = rearrange(source_latents, 'b f c h w -> b c f h w').contiguous()

        if encode_lighting:
            self.load_models_to_device(['vae'])
            lighting = lighting.to(dtype=self.torch_dtype, device=self.device)
            B, C, H, W = lighting.size()
            N = C // 3
            lighting = 2 * lighting - 1
            lighting = lighting.view(-1, N, 3, H, W)
            lighting = self.encode_video_in_frame(lighting, **tiler_kwargs)
            lighting = lighting.reshape(B, -1, *lighting.size()[3:]).contiguous()
        
        # Initialize noise
        noise = self.generate_noise(source_latents.shape, seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if target is not None:
            if input_video:
                self.load_models_to_device(['vae'])
                # input_video = self.preprocess_images(input_video)
                target = target.to(dtype=self.torch_dtype, device=self.device)
                latents = self.encode_video_in_frame(target, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            else:
                latents = target
            latents = rearrange(latents, 'b f c h w -> b c f h w').contiguous()
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise

        # Process target camera (recammaster)
        cam_emb = batch_cam2pose_embedding(target_camera, original_coordinate="opencv").to(dtype=self.torch_dtype, device=self.device)
        posi_prompt = self.get_prompt_emb(prompt, batch_size, positive=True).to(dtype=self.torch_dtype, 
                                                                device=self.device)
        nega_prompt = self.get_prompt_emb(negative_prompt, batch_size, positive=False).to(dtype=self.torch_dtype, 
                                                                device=self.device)

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
            condition = {"cam_emb": cam_emb,
                        "context": posi_prompt,
                        "lighting": lighting}
            
            noise_pred_posi = model_fn_wan_video(self.dit, torch.cat([x, source_latents], dim=2), 
                                            timestep=timestep, **condition, 
                                            **extra_input, **tea_cache_posi)[:,:,:tgt_latent_length,...]
            if cfg_scale != 1.0:
                condition = {"cam_emb": cam_emb,
                        "context": posi_prompt,
                        "lighting": torch.zeros_like(lighting)}
                noise_pred_nega = model_fn_wan_video(self.dit, torch.cat([x, source_latents], dim=2),
                                            timestep=timestep, **condition, 
                                            **extra_input, **tea_cache_nega)[:,:,:tgt_latent_length,...]
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
        # frames = self.tensor2video(frames[0])

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

    def check(self, dit: WanModel, x, t_mod):
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
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    cam_emb: torch.Tensor,
    context: torch.Tensor,
    lighting: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    **kwargs,
):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    lighting, lighting_uv = dit.lighting_embedding(lighting)
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
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
            x = block(x, context, cam_emb, t_mod, freqs, lighting, lighting_uv)
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
