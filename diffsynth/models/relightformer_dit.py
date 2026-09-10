import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffsynth import ModelManager

from ..rope import RopeDotProductAttention
from .utils import Cached, hash_state_dict_keys

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode: bool = False) -> torch.Tensor:
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor, n: int = 10000) -> torch.Tensor:
    sinusoid = torch.outer(
        position.type(torch.float64), 
        torch.pow(n, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def sinusoidal_embedding_2d(dim: int, u_position: torch.Tensor, v_position: torch.Tensor, n: int = 10000):
    u_freqs = sinusoidal_embedding_1d(dim // 2, u_position, n)
    v_freqs = sinusoidal_embedding_1d(dim - dim // 2, v_position, n)
    return u_freqs, v_freqs


def rope_embedding_1d(dim: int, position: torch.Tensor, n: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (n ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(position, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def rope_embedding_2d(dim: int, u_position: torch.Tensor, v_position: torch.Tensor, n: int = 10000):
    u_freqs = rope_embedding_1d(dim - (dim // 2), u_position, n)
    v_freqs = rope_embedding_1d(dim // 2, v_position, n)
    return u_freqs, v_freqs


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def precompute_freqs_cis_2d(dim: int, end: int = 1024, theta: float = 10000.0):
    h_freqs_cis = precompute_freqs_cis(dim - (dim // 2), end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 2, end, theta)
    return h_freqs_cis, w_freqs_cis


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class RMSNorm2d(nn.Module):
    """RMSNorm for 4D tensors (B, C, H, W). Normalizes across spatial dimensions (H, W) per channel."""
    def __init__(self, num_channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=[2, 3], keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight[None, :, None, None]


class AttentionModule(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)


class ChannelMergeModule(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.linear = nn.Linear(c1 + c2, c1)

    def xavier_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, _, F1, _, _ = x.shape
        B, _, F2, _, _ = y.shape
        
        if F2 == 1 and F1 > 1:
            y = y.expand(-1, -1, F1, -1, -1)  # Broadcast y to match x's frame dimension
            
        cat = torch.cat([x, y], dim=1)  # (B, C1+C2, F, H, W)
        cat = cat.permute(0, 2, 3, 4, 1)  # (B, F, H, W, C1+C2)
        out = self.linear(cat)  # (B, F, H, W, C1)
        return out.permute(0, 4, 1, 2, 3)  # (B, C1, F, H, W)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, Ts: Optional[torch.Tensor] = None, Ks: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        if isinstance(self.attn, AttentionModule):
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
            x = self.attn(q, k, v)
        else:
            x = self.attn(q, k, v, viewmats=Ts, Ks=Ks, is_c2w=True)

        return self.o(x)


class LightingAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = AttentionModule(num_heads)

    def xavier_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        attn_out = self.attn(q, k, v)
        return self.o(attn_out)


class LightingPatchifier(nn.Module):
    """Multi-stage Convolutional Network for Lighting Embedding."""
    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-4, patch_size: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm1 = RMSNorm2d(64, eps=eps)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.norm2 = RMSNorm2d(32, eps=eps)

        self.conv3 = nn.Conv2d(32, out_dim, kernel_size=3, padding=1, bias=False)
        self.norm3 = RMSNorm2d(out_dim, eps=eps)

        self.act = nn.LeakyReLU(inplace=True)

    def xavier_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:  # (B, F, in_dim, H, W)
            x = rearrange(x, 'b f c h w -> b (f c) h w')
        
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.act(self.norm3(self.conv3(x)))

        return rearrange(x, 'b c h w -> b (h w) c')


class RaysEmbedding(nn.Module):
    """
    Shallow Conv2D-based network for ray/light embedding.
    Input:  (B, C, F, 256, 256) or (B, C, F, 512, 512)
    Output: (B, out_dim, F, 32, 32)
    
    Optimizations:
    1. Activation: SiLU --> Sine (with configurable omega).
    2. Initialization: Replaced Xavier with SIREN-specific initialization 
       to prevent gradient explosion caused by high-frequency sine waves.
    """
    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-4, omega: float = 30.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.omega = omega
        
        # Block 1: 256 → 128 (or 512 → 256)
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=2, padding=1, bias=True)
        
        # Block 2: 128 → 64 (or 256 → 128)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)

        # Block 3: 64 → 64 (or 128 → 128)
        self.conv3 = nn.Conv2d(128, out_dim, kernel_size=3, stride=1, padding=1, bias=True)
        
        # Block 4: Pooling to reach final 32x32 resolution
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)
    
    def _apply_siren_init(self):
        """
        Apply SIREN (Sinusoidal Representation Networks) initialization.
        - First layer: weights ~ U(-1/fan_in, 1/fan_in)
        - Subsequent layers: weights ~ U(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)
        - Biases: initialized to 0
        """
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # Calculate fan_in (number of input connections: in_channels * kernel_h * kernel_w)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                
                # Determine if it's the first layer
                is_first_layer = (name == 'conv1')
                
                if is_first_layer:
                    w_std = 1.0 / fan_in
                else:
                    w_std = math.sqrt(6.0 / fan_in) / self.omega
                
                # Initialize weights with uniform distribution
                nn.init.uniform_(m.weight, -w_std, w_std)
                
                # Initialize biases to 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
            elif hasattr(m, "weight") and "Norm" in m.__class__.__name__:
                # Keep Norm layers initialized to identity scaling
                nn.init.constant_(m.weight, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def siren_init(self):
        """
        Kept for backward compatibility with external training scripts.
        Internally, it now applies the SIREN initialization which is required 
        for the Sine activation function to prevent gradient explosion.
        """
        self._apply_siren_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: (B, C, F, 256, 256) or (B, C, F, 512, 512)
        Output:
            x: (B, out_dim, F, 32, 32)
        """
        B, C, F, H, W = x.shape
        assert (H == 256 and W == 256) or (H == 512 and W == 512), f"Expected (256, 256) or (512, 512), got ({H},{W})"

        # Merge batch and frame dims: (B*F, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, F, C, H, W)
        x = x.view(B * F, C, H, W)
        
        # Block 1
        x = self.act(self.conv1(x))

        # Block 2
        x = self.act(self.conv2(x))

        # Block 3
        x = self.act(self.conv3(x))

        # Block 4: pooling (H/4 → H/8)
        x = self.pool(x)

        # Reshape back: (B, out_dim, F, H // 8, W // 8)
        x = x.view(B, F, self.out_dim, H // 8, W // 8)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        return x


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), 
            nn.GELU(approximate='tanh'), 
            nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.has_text_cross_attn = False

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor, freqs: torch.Tensor, 
                lighting: Optional[torch.Tensor], Ts: Optional[torch.Tensor] = None, 
                Ks: Optional[torch.Tensor] = None, addition_rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        if lighting is None:
            z = self.self_attn(input_x, freqs, Ts, Ks)
        elif hasattr(self, "lighting_attn"):
            z = self.self_attn(input_x, freqs, Ts, Ks) + self.lighting_attn(input_x, lighting)
        else:
            B, _, _, _ = Ts.shape
            padding_T = torch.eye(4, device=Ts.device, dtype=Ts.dtype).view(1, 1, 4, 4).expand(B, 1, 4, 4)
            if addition_rotation is not None:
                padding_T = padding_T.clone()
                padding_T[:, 0, :3, :3] = addition_rotation
            Ts = torch.cat([Ts, padding_T], dim=1)
            
            padding_K = torch.eye(3, device=Ts.device, dtype=Ts.dtype).view(1, 1, 3, 3).expand(B, 1, 3, 3)
            Ks = torch.cat([Ks, padding_K], dim=1)

            z = self.self_attn(torch.cat([input_x, lighting], dim=1), freqs, Ts, Ks)
            z = z[:, :input_x.size(1), :]

        # FIX: 'self.projector' was undefined in original code. 
        # Since self_attn already applies the output projection (self.o), we use 'z' directly.
        x = x + gate_msa * z
        
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        
        return x.contiguous() if not x.is_contiguous() else x


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if t_mod.ndim == 2:
            t_mod = t_mod.unsqueeze(1)  # [B, 1, dim]
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class RelightFormerModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        # text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
    ):
        super().__init__()
        self._supports_gradient_checkpointing = True
        self.gradient_checkpointing = True
        
        self.dim = dim
        self.freq_dim = freq_dim
        self.eps = eps
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.has_image_input = has_image_input

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self.rays_embedding = RaysEmbedding(6, in_dim)
        self.lighting_patchifier = LightingPatchifier(2 * in_dim, dim)
    
    @property
    def patch_downsample_ratio(self) -> int:
        return self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        lighting: torch.Tensor,
        rays: torch.Tensor,
        lighting_rays: torch.Tensor,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = True,
        **kwargs,
    ):
        target_frame_size = kwargs.pop('tgt_latent_len', 0)
        Ts = kwargs.get('Ts', None)
        Ks = kwargs.get('Ks', None)
        addition_rotation = kwargs.get('addition_rotation', None)

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        x = x + self.rays_embedding(rays)
        if lighting_rays is not None and lighting is not None:
            lighting = lighting + self.rays_embedding(lighting_rays)

        if hasattr(self, 'channel_wise_merger') and lighting is not None:
            # print("WARNING: This version is for ablation, lighting is merged into x and will not be used separately.")
            B, _, _, H, W = x.shape
            lighting = lighting.view(B, -1, 1, H, W)
            x = self.channel_wise_merger(x, lighting)
            lighting = None  # merged into x, no separate lighting tokens
        if lighting is not None:
            lighting = self.lighting_patchifier(lighting)
            
        x, (f, h, w) = self.patchify(x)
        
        # modification: no positional embedding for frames
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs, **inner_kwargs):
                return module(*inputs, **inner_kwargs)
            return custom_forward

        for _, block in enumerate(self.blocks):
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block), 
                            x, t_mod, freqs, lighting, 
                            Ts=Ts, Ks=Ks, addition_rotation=addition_rotation,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, t_mod, freqs, lighting, 
                        Ts=Ts, Ks=Ks, addition_rotation=addition_rotation,
                        use_reentrant=False,
                    )
            else:
                x = block(x, t_mod, freqs, lighting, Ts=Ts, Ks=Ks, addition_rotation=addition_rotation)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))

        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    @classmethod
    def init_from_wan_model(cls, wan, division_factor, rope="PRoPE", resolution=256, **kwargs):
        model = cls(
            wan.dim,
            wan.patch_embedding.weight.shape[1],
            wan.blocks[0].ffn_dim,
            wan.head.out_dim,
            wan.time_embedding[0].weight.shape[1],
            wan.eps,
            wan.patch_size,
            wan.num_heads,
            wan.num_layers,
            wan.has_image_input,
        )
        print*()
        model.load_state_dict(wan.state_dict(), strict=False)
        model.lighting_patchifier.xavier_init()
        model.rays_embedding.siren_init()

        if rope != "PRoPE":
            print(f"This is a abltion version, the rope is not 'PRoPE' but {rope}.")

        for block in model.blocks:
            block.lighting_attn = LightingAttention(block.self_attn.dim, block.self_attn.head_dim, eps=model.eps)
            block.lighting_attn.xavier_init()
            block.self_attn.attn = RopeDotProductAttention(block.self_attn.head_dim, 
                                                        patches_x=resolution // division_factor,
                                                        patches_y=resolution // division_factor,
                                                        image_height=resolution,
                                                        image_width=resolution,
                                                        rope=rope)
        return model

class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict: dict):
        rename_dict = {
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        
        num_blocks = 29
        for i in range(num_blocks):
            rename_dict[f"blocks.{i}.attn1.norm_k.weight"] = f"blocks.{i}.self_attn.norm_k.weight"
            rename_dict[f"blocks.{i}.attn1.norm_q.weight"] = f"blocks.{i}.self_attn.norm_q.weight"
            rename_dict[f"blocks.{i}.attn1.to_k.bias"] = f"blocks.{i}.self_attn.k.bias"
            rename_dict[f"blocks.{i}.attn1.to_k.weight"] = f"blocks.{i}.self_attn.k.weight"
            rename_dict[f"blocks.{i}.attn1.to_out.0.bias"] = f"blocks.{i}.self_attn.o.bias"
            rename_dict[f"blocks.{i}.attn1.to_out.0.weight"] = f"blocks.{i}.self_attn.o.weight"
            rename_dict[f"blocks.{i}.attn1.to_q.bias"] = f"blocks.{i}.self_attn.q.bias"
            rename_dict[f"blocks.{i}.attn1.to_q.weight"] = f"blocks.{i}.self_attn.q.weight"
            rename_dict[f"blocks.{i}.attn1.to_v.bias"] = f"blocks.{i}.self_attn.v.bias"
            rename_dict[f"blocks.{i}.attn1.to_v.weight"] = f"blocks.{i}.self_attn.v.weight"

            rename_dict[f"blocks.{i}.ffn.net.0.proj.bias"] = f"blocks.{i}.ffn.0.bias"
            rename_dict[f"blocks.{i}.ffn.net.0.proj.weight"] = f"blocks.{i}.ffn.0.weight"
            rename_dict[f"blocks.{i}.ffn.net.2.bias"] = f"blocks.{i}.ffn.2.bias"
            rename_dict[f"blocks.{i}.ffn.net.2.weight"] = f"blocks.{i}.ffn.2.weight"

            rename_dict[f"blocks.{i}.scale_shift_table"] = f"blocks.{i}.modulation"

        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
                    
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
            
        return state_dict_, config