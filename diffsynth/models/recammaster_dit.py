import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys, Cached
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
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position, n=10000):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        n, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def sinusoidal_embedding_2d(dim, u_position, v_position, n=10000):
    u_freqs = sinusoidal_embedding_1d(dim // 2, u_position, n)
    v_freqs = sinusoidal_embedding_1d(dim - dim // 2, v_position, n)
    return u_freqs, v_freqs


def rope_embedding_1d(dim, position, n=10000.0):
    freqs = 1.0 / (n ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(position, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_embedding_2d(dim, u_position, v_position, n=10000):
    u_freqs = rope_embedding_1d(dim - (dim // 2), u_position, n)
    v_freqs = rope_embedding_1d(dim // 2, v_position, n)
    return u_freqs, v_freqs


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis_2d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    h_freqs_cis = precompute_freqs_cis(dim - (dim // 2), end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 2, end, theta)
    return h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
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

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
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
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)
    
class LightingAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, 
                 k_from = "rgb", albedo_mask = False):
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

        assert k_from in ["rgb", "uv", "uv+rgb", "uv*rgb", "rope"], f"Invalid k_from [{k_from}]"
        self.k_from = k_from
        self.albedo_mask = albedo_mask
        if self.albedo_mask:
            self.a = nn.Linear(dim, dim)
            self.norm_a = RMSNorm(dim, eps=eps)
            self.activate_a = nn.Sigmoid()


    def xavier_init(self):
        # Apply Xavier initialization to linear layers
        for module in self.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # or use nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # Initialize biases to zero

    def zero_init(self):
        # Apply Xavier initialization to linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # Initialize biases to zero

    def forward(self, x: torch.Tensor, y: torch.Tensor, uv: torch.Tensor):
        B, _, C = x.size()
        x = x.reshape(B, -1, C)
        q = self.norm_q(self.q(x))

        if self.k_from == "rgb":
            k = self.norm_k(self.k(y))
        elif self.k_from == "uv":
            k = self.norm_k(self.k(uv))
        elif self.k_from == "uv+rgb":
            k = self.norm_k(self.k(uv+y))
        elif self.k_from == "uv*rgb":
            k = self.norm_k(self.k(uv*y))
        elif self.k_from == "rope":
            uv = rearrange(uv, "b l c -> b l 1 c")
            k = self.norm_k(self.k(rope_apply(y, uv, self.num_heads)))
        else:
            raise ValueError(f"Unsupport k_from: {self.k_from}")

        v = self.v(y)


        if self.albedo_mask:
            a = self.activate_a(self.norm_a(self.a(x)))

        x = self.attn(q, k, v)
        x = x.reshape(B, _, C)

        if self.albedo_mask:
            x = x * a
        return self.o(x)
    
class Lighting_Embedding(nn.Module):
    '''
    Convolution Network Based Lighting Embedding
    '''
    def __init__(self, in_dim:int, out_dim:int, num_heads:int, 
                 patch_size:int=2, num_register_tokens=0, uv=None):
        super().__init__()
        layers = [nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(out_dim), 
                  nn.SiLU()]
        layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        self.down_sampler = nn.Sequential(*layers)
        self.patchifier = nn.Conv2d(out_dim, out_dim, kernel_size=patch_size, 
                                    stride=patch_size, padding=0, bias=False)
        self.num_register_tokens = num_register_tokens
        if num_register_tokens <=0:
            self.register_tokens = None
        else:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, out_dim))

        assert uv in [None, "sin", "rope"], f"Invalid positional [{uv}]"
        self.uv = uv

        self.cached = Cached()

    def xavier_init(self):
        # Apply Xavier initialization to convolutional layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  # Initialize biases to zero
        if self.register_tokens is None:
            pass
        else:
             nn.init.constant_(self.register_tokens, 0)

    def get_uv_emb(self, x):
        B, C, H, W = x.size()
        cached_key = f"{H},{W}"
        # Check if we can reuse cached embeddings
        if self.cached.is_cached(cached_key):
            uv_emb = self.cached.retrieve(cached_key).to(x.device, dtype=x.dtype)
        else: # not cached
            if self.uv is None:
                return None
            elif self.uv == 'sin':
                u, v = sinusoidal_embedding_2d(dim=self.out_dim,  # for all heads
                        u_position=torch.linspace(torch.pi / 2.0, -1.0 * torch.pi / 2.0, H),
                        v_position=torch.linspace(-1.0 * torch.pi, torch.pi, W),
                        n=200.0)
                uv_emb = torch.cat([
                    u.view(1, H, 1, -1).expand(1, H, W, -1),
                    v.view(1, 1, W, -1).expand(1, H, W, -1),
                ], dim=-1).reshape(1, H, W, -1).to(x.device)  # [B==1, H, W, dim]
            else:
                u, v = rope_embedding_2d(dim=self.head_dim,  # for each head
                        u_position=torch.linspace(torch.pi / 2.0, -1.0 * torch.pi / 2.0, H),
                        v_position=torch.linspace(-1.0 * torch.pi, torch.pi, W),
                        n=200.0)
                # u, v = precompute_freqs_cis_2d(dim=self.head_dim, theta=200.0)
                u, v = u[:H], v[:W]
                uv_emb = torch.cat([
                    u.view(1, H, 1, -1).expand(1, H, W, -1),
                    v.view(1, 1, W, -1).expand(1, H, W, -1),
                ], dim=-1).reshape(1, H, W, -1).to(x.device)  # [B==1, H, W, dim]
            uv_emb = rearrange(uv_emb, "b h w c -> b c h w").to(x.device, dtype=x.dtype)
            # Cache the computed embeddings and dimensions
            self.cached.store(key=cached_key, value=uv_emb)
        uv_emb = uv_emb.repeat(B, 1, 1, 1)
        return uv_emb

    def forward(self, x):
        B, C, H, W = x.size()
        # x = x.reshape(-1, C, H, W)
        x = self.down_sampler(x)

        x = self.patchifier(x)
        x = x.reshape(B, *x.size()[1:])
        uv = self.get_uv_emb(x) # [B, C, H, W] [16, 1536, 64, 64] or [16, 128, 64, 64]

        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if uv is not None:
            uv = rearrange(uv, 'b c h w -> b (h w) c').contiguous()
        if self.register_tokens is None:
            pass
        else:
            register_tokens = torch.cat([self.register_tokens]*B, dim=0)
            x = torch.cat([x, register_tokens], dim=1)
            if uv is not None:
                uv = torch.cat([uv, torch.zeros(B, self.num_register_tokens, uv.size()[-1]).to(x.device, dtype=x.dtype)], dim=1)

        return x, uv


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, 
                 num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # In addition
        self.cam_encoder = nn.Linear(12, dim)
        self.projector = nn.Linear(dim, dim)

    def forward(self, x, context, cam_emb, t_mod, freqs, lighting, lighting_uv):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        # encode camera
        cam_emb = self.cam_encoder(cam_emb)

        frame_size = cam_emb.size()[1]
        cam_emb = cam_emb.repeat(1, 2, 1)
        cam_emb = cam_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16, 1) # TODO: Auto
        cam_emb = rearrange(cam_emb, 'b f h w d -> b (f h w) d')
        input_x = input_x + cam_emb
        x = x + gate_msa * self.projector(self.self_attn(input_x, freqs) 
                                            + self.lighting_attn(input_x, lighting, lighting_uv))
        
        x = x + self.cross_attn(self.norm3(x), context)

        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class ReCamMasterModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.eps = eps
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.has_image_input = has_image_input

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)  # clip_feature_dim = 1280

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                cam_emb: torch.Tensor,
                context: torch.Tensor,
                lighting: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = True,
                use_gradient_checkpointing_offload: bool = True,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        lighting, lighting_uv = self.lighting_embedding(lighting)
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, cam_emb, t_mod, freqs, lighting, lighting_uv,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, cam_emb, t_mod, freqs, lighting, lighting_uv,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, cam_emb, t_mod, freqs, lighting, lighting_uv)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter(): # TODO: Unfinish WanModelStateConverter
        return WanModelStateDictConverter()
    
    @classmethod
    def init_from_WanModel(self, wan):
        return ReCamMasterModel(wan.dim,
                                wan.patch_embedding.weight.shape[1],
                                wan.blocks[0].ffn_dim,
                                wan.head.out_dim,
                                wan.text_embedding[0].weight.shape[1],
                                wan.time_embedding[0].weight.shape[1],
                                wan.eps,
                                wan.patch_size,
                                wan.num_heads,
                                wan.num_layers,
                                wan.has_image_input,
                                )

    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = { # Source : Target
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
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
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        return state_dict, config
