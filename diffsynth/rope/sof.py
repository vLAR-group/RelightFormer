# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# How to use PRoPE attention for self-attention:
# 
# 1. Easiest way (fast):
#    attn = PropeDotProductAttention(...)
#    o = attn(q, k, v, viewmats, Ks)
#
# 2. More flexible way (fast):
#    attn = PropeDotProductAttention(...)
#    attn._precompute_and_cache_apply_fns(viewmats, Ks)
#    q = attn._apply_to_q(q)
#    k = attn._apply_to_k(k)
#    v = attn._apply_to_v(v)
#    o = F.scaled_dot_product_attention(q, k, v, **kwargs)
#    o = attn._apply_to_o(o)
# 
# 3. The most flexible way (but slower because repeated computation of RoPE coefficients):
#    o = rope_dot_product_attention(q, k, v, ...)
# 
# How to use PRoPE attention for cross-attention:
# 
#    attn_src = PropeDotProductAttention(...)
#    attn_tgt = PropeDotProductAttention(...)
#    attn_src._precompute_and_cache_apply_fns(viewmats_src, Ks_src)
#    attn_tgt._precompute_and_cache_apply_fns(viewmats_tgt, Ks_tgt)
#    q_src = attn_src._apply_to_q(q_src)
#    k_tgt = attn_tgt._apply_to_k(k_tgt)
#    v_tgt = attn_tgt._apply_to_v(v_tgt)
#    o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt, **kwargs)
#    o_src = attn_src._apply_to_o(o_src)

from functools import partial
from typing import Callable, Optional, Tuple, List, Union

import torch
import torch.nn.functional as F

from .group import se3_to_so4, scaled_se3_to_so4
from einops import rearrange
from .prope import _invert_K, _invert_SE3, _lift_K

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

from .cache import SOFCache
_SOF_CACHE = SOFCache(max_entries=32)
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, compatibility_mode=False, **kwargs):
    if compatibility_mode:
        x = F.scaled_dot_product_attention(q, k, v)
    elif FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_func(q, k, v)
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_func(q, k, v)
    elif SAGE_ATTN_AVAILABLE:
        x = sageattn(q, k, v)
    else:
        x = F.scaled_dot_product_attention(q, k, v)
    return x

class RopeDotProductAttention(torch.nn.Module):
    """ RoPE attention with precomputed RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int = 16,
        patches_y: int = 16,
        image_width: int = 256,
        image_height: int = 256,
        freq_base: float = 10000.0,
        rope: str = 'WAN'
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x # How many patches wide is each image? 
        self.patches_y = patches_y # How many patches high is each image?
        self.image_width = image_width
        self.image_height = image_height
        self.rope = rope

        freqs_cis_x = _precompute_freqs_cis_complex(
            torch.tile(torch.arange(patches_x), (patches_y,)),
            theta=freq_base,
            dim=_block_dim(rope, head_dim, 'H'),
        )
        freqs_cis_y = _precompute_freqs_cis_complex(
            torch.repeat_interleave(torch.arange(patches_y), patches_x),
            theta=freq_base,
            dim=_block_dim(rope, head_dim, 'W'),
        )
        
        # Do not save coeffs to checkpoint as `cameras` might change during testing.
        self.register_buffer("freqs_cis_x", freqs_cis_x, persistent=False)
        self.register_buffer("freqs_cis_y", freqs_cis_y, persistent=False)

    # override load_state_dict to not load coeffs if they exist (for backward compatibility)
    def load_state_dict(self, state_dict, strict=True):
        # remove coeffs from state_dict
        state_dict.pop("freqs_cis_x", None)
        state_dict.pop("freqs_cis_y", None)
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4) camera<=world
        Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
        **kwargs,
    ) -> torch.Tensor:
        q = rearrange(q, "b s (n d) -> b n s d", d=self.head_dim)
        k = rearrange(k, "b s (n d) -> b n s d", d=self.head_dim)
        v = rearrange(v, "b s (n d) -> b n s d", d=self.head_dim)
        x = rope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            freqs_cis_x=self.freqs_cis_x,
            freqs_cis_y=self.freqs_cis_y,
            rope=self.rope,
            **kwargs,
        )
        x = rearrange(x, "b n s d -> b s (n d)")
        return x

    def _precompute_and_cache_apply_fns(
        self, viewmats: torch.Tensor, Ks: Optional[torch.Tensor]
    ):
        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        self.cameras = cameras

        self.apply_fn_q, 
        self.apply_fn_k, 
        self.apply_fn_k, 
        self.apply_fn_o \
            = _prepare_apply_fns(
                head_dim=self.head_dim,
                viewmats=viewmats,
                Ks=Ks,
                patches_x=self.patches_x,
                patches_y=self.patches_y,
                image_width=self.image_width,
                image_height=self.image_height,
                coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
                coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
            )

    def _apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = q.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert q.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_q is not None
        return self.apply_fn_q(q)

    def _apply_to_k(self, k: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = k.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert k.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_k is not None
        return self.apply_fn_k(k)
    
    def _apply_to_v(self, v: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = v.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert v.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_v is not None
        return self.apply_fn_v(v)

    def _apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = o.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert o.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_o is not None
        return self.apply_fn_o(o)


def rope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    freqs_cis_x: Optional[torch.Tensor] = None,
    freqs_cis_y: Optional[torch.Tensor] = None,
    rope: Optional[str] = None,
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3), f"Unsupport Ks shape: {Ks.shape}"
    assert seqlen == cameras * patches_x * patches_y

    apply_fn_q, apply_fn_k, apply_fn_v, apply_fn_o = _prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        freqs_cis_x=freqs_cis_x,
        freqs_cis_y=freqs_cis_y,
        rope=rope,
        **kwargs
    )

    # TODO: Apply Attn Mask
    out = flash_attention(
        q=apply_fn_q(q),
        k=apply_fn_k(k),
        v=apply_fn_v(v),
        **kwargs,
    )
    out = apply_fn_o(out)
    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out


def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    freqs_cis_x: Optional[torch.Tensor] = None,
    freqs_cis_y: Optional[torch.Tensor] = None,
    rope: Optional[str] = None,
    **kwargs
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape
    freq_base = kwargs.pop('freq_base', 10000.0)
    if kwargs.pop("is_c2w", False):
        viewmats = _invert_SE3(viewmats)
        T_w2c = _invert_SE3(viewmats)
        T_c2w = viewmats
    else:
        T_w2c = viewmats
        T_c2w = _invert_SE3(T_w2c)

    # Normalize camera intrinsics.
    if rope == 'PRoPE':
        if Ks is not None:
            Ks_norm = torch.zeros_like(Ks)
            Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
            Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
            Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
            Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
            Ks_norm[..., 2, 2] = 1.0
            del Ks

            # Compute the camera projection matrices we use in PRoPE.
            # - K is an `image<-camera` transform.
            # - viewmats is a `camera<-world` transform.
            # - P = lift(K) @ viewmats is an `image<-world` transform.
            P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
            P_T = P.transpose(-1, -2)
            P_inv = torch.einsum(
                "...ij,...jk->...ik",
                T_c2w,
                _lift_K(_invert_K(Ks_norm)),
            )
        else:
            # GTA formula. P is `camera<-world` transform.
            P = viewmats
            P_T = P.transpose(-1, -2)
            P_inv = T_c2w

        assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Precompute cos/sin terms for RoPE. We use tiles/repeats for 'row-major'
    # broadcasting.
    if freqs_cis_x is None:
        freqs_cis_x = _precompute_freqs_cis_complex(
            torch.tile(torch.arange(patches_x), (patches_y,)),
            theta=freq_base,
            dim=_block_dim(rope, head_dim, 'H'),
        )
    if freqs_cis_y is None:
        freqs_cis_y = _precompute_freqs_cis_complex(
            torch.repeat_interleave(torch.arange(patches_y), patches_x),
            theta=freq_base,
            dim=_block_dim(rope, head_dim, 'W'),
        )

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0

    if rope == None:
        apply_fn_q = _identity
        apply_fn_k = _identity
        apply_fn_v = _identity
        apply_fn_o = _identity
    elif rope == 'SO4-QK':
        transforms = [
            (partial(_apply_tiled_projmat, matrix=se3_to_so4(T_c2w)), head_dim // 2),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x), head_dim // 4),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y), head_dim // 4),
        ]
        apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
        apply_fn_k = partial(_apply_block_diagonal, func_size_pairs=transforms_k)
        apply_fn_v = _identity
        apply_fn_o = _identity
    elif rope == 'SO4-QK-MS':
        transforms = [
            (partial(_apply_tiled_sof, SE3=T_c2w), head_dim // 2),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x), head_dim // 4),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y), head_dim // 4),
        ]
        apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms)
        apply_fn_k = partial(_apply_block_diagonal, func_size_pairs=transforms)
        apply_fn_v = _identity
        apply_fn_o = _identity
    elif rope == 'WAN':
        freqs_cis_t = kwargs.pop('freqs_cis_t', None)
        if freqs_cis_t is None:
            freqs_cis_t = _precompute_freqs_cis_complex(
                    torch.repeat_interleave(
                        torch.arange(cameras, device=device),
                        patches_x * patches_y
                        ),  # shape: (cameras * patches_y * patches_x,)
                    theta=freq_base,
                    dim=_block_dim(rope, head_dim, 'F')
                )

        transforms = [
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_t), head_dim - 2 * (head_dim // 3)),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x), head_dim // 3),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y), head_dim // 3),
        ]
        apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms)
        apply_fn_k = apply_fn_q
        apply_fn_v = _identity
        apply_fn_o = _identity
    elif rope == 'CaPE':
        transforms_q = [
            (partial(_apply_tiled_projmat, matrix=T_w2c.transpose(-1, -2)), head_dim // 2),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x), head_dim // 4),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y), head_dim // 4),
        ]
        transforms_k = [
            (partial(_apply_tiled_projmat, matrix=T_c2w), head_dim // 2),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x), head_dim // 4),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y), head_dim // 4),
        ]
        apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
        apply_fn_k = partial(_apply_block_diagonal, func_size_pairs=transforms_k)
        apply_fn_v = _identity
        apply_fn_o = _identity
    elif rope == 'GTA':
        transforms_q = [
            (partial(_apply_tiled_projmat, matrix=T_w2c.transpose(-1, -2)), head_dim // 2),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x), head_dim // 4),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y), head_dim // 4),
        ]
        transforms_k = [
            (partial(_apply_tiled_projmat, matrix=T_c2w), head_dim // 2),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x), head_dim // 4),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y), head_dim // 4),
        ]
        transforms_o = [
            (partial(_apply_tiled_projmat, matrix=T_w2c), head_dim // 2),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x, inverse=True), head_dim // 4),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y, inverse=True), head_dim // 4),
        ]
        apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
        apply_fn_k = partial(_apply_block_diagonal, func_size_pairs=transforms_k)
        apply_fn_v = apply_fn_k
        apply_fn_o = _identity
    elif rope == 'PRoPE':
        transforms_q = [
            (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x), head_dim // 4),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y), head_dim // 4),
        ]
        transforms_k = [
            (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x), head_dim // 4),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y), head_dim // 4),
        ]
        transforms_o = [
            (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_x, inverse=True), head_dim // 4),
            (partial(_rope_apply_complex, freqs_cis=freqs_cis_y, inverse=True), head_dim // 4),
        ]

        apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
        apply_fn_k = partial(_apply_block_diagonal, func_size_pairs=transforms_k)
        apply_fn_v = apply_fn_k
        apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    else:
        raise NotImplemented(f"Not Implement rope type {rope}")
    return apply_fn_q, apply_fn_k, apply_fn_v, apply_fn_o

def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply projection matrix to features."""
    # - seqlen => (cameras, patches_x * patches_y)
    # - feat_dim => (feat_dim // 4, 4)
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)

def _apply_tiled_sof(
    feats: torch.Tensor,
    SE3: torch.Tensor,  # (batch, cameras, 4, 4)
) -> torch.Tensor:
    B, H, L, F = feats.shape
    assert SE3.ndim == 4 and SE3    .shape[0] == B
    C, D = SE3.shape[1], SE3.shape[-1]
    assert L % C == 0 and F % D == 0
    P, G = L // C, F // D

    # Generate ALL matrices at once: (B, C, P, D, D)
    # import time
    # start_time = time.perf_counter()

    cached = _SOF_CACHE.get(SE3, P)
    if cached is not None:
        # cache_hit_time = time.perf_counter() - start_time
        # print(f"Using SOF cache | Time: {cache_hit_time*1000:.3f} ms")
        M_all = cached.to(device=feats.device)
    else:
        # Time the expensive computation
        # compute_start = time.perf_counter()
        n = torch.arange(P, dtype=feats.dtype, device=feats.device) / P   # shape (P,)
        M_all = scaled_se3_to_so4(SE3, n)
        # compute_time = time.perf_counter() - compute_start

        # Time the put operation
        # put_start = time.perf_counter()
        _SOF_CACHE.put(SE3, P, M_all.cpu())
        # put_time = time.perf_counter() - put_start

        # total_miss_time = time.perf_counter() - start_time
        # print(
            # f"SOF cache MISS | "
            # f"Compute: {compute_time*1000:.3f} ms | "
            # f"Put: {put_time*1000:.3f} ms | "
            # f"Total: {total_miss_time*1000:.3f} ms"
        # )
    
    # print(M_all)
    # print(torch.isnan(M_all).any())

    # Reshape features: (B, H, C, P, G, D)
    feats_reshaped = feats.view(B, H, C, P, G, D)

    # One einsum to rule them all
    output = torch.einsum("bhcpgd,bcpdj->bhcpgj", feats_reshaped, M_all)

    return output.reshape(B, H, L, F)

def _precompute_freqs_cis_complex(
    positions: torch.Tensor,  # (seqlen,)
    theta: float = 10000.0,
    dim: int = 64,
) -> torch.Tensor:
    """
    Returns complex freqs of shape (1, 1, seqlen, dim // 2)
    Compatible with view_as_complex apply.
    """
    assert dim % 2 == 0
    device = positions.device
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))  # (dim//2,)
    angles = positions[:, None] * freqs[None, :]  # (seqlen, dim//2)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # (seqlen, dim//2)
    return freqs_cis[None, None, :, :]  # (1, 1, seqlen, dim//2)

def _rope_apply_complex(
    feats: torch.Tensor,        # (B, H, L, D)
    freqs_cis: torch.Tensor,    # (L_c, D//2) or (1, 1, L_c, D//2)
    inverse: bool = False
) -> torch.Tensor:
    B, H, L, D = feats.shape
    assert D % 2 == 0, f"Feature dim must be even, got {D}"

    # Normalize freqs_cis to (1, 1, L_c, half_d)
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis[None, None, :, :]  # (1, 1, L_c, half_d)
    elif freqs_cis.ndim != 4:
        raise ValueError(f"freqs_cis must be 2D or 4D, got {freqs_cis.shape}")

    L_c = freqs_cis.shape[2]

    if L_c > L:
        freqs_cis = freqs_cis[:, :, :L, :]
    # Ensure L is divisible by L_c for clean chunking (optional but clean)
    elif L % L_c != 0:
        # Optionally pad feats (not shown), or raise error
        raise ValueError(f"L={L} must be divisible by L_c={L_c} for chunked RoPE")

    # Reshape feats into chunks of length L_c
    F = L // L_c
    # (B, H, L, D) -> (B, H, n_chunks, L_c, D)
    feats_chunked = feats.reshape(B, H, F, L_c, D)

    # View as complex: (B, H, n_chunks, L_c, half_d, 2) -> (B, H, n_chunks, L_c, half_d)
    x_complex = torch.view_as_complex(feats_chunked.float().reshape(B, H, F, L_c, -1, 2))

    # freqs_cis: (1, 1, L_c, half_d) → broadcasts over B, H, n_chunks
    if inverse:
        freqs_cis = freqs_cis.conj()
    x_rotated = x_complex * freqs_cis  # Broadcasting: no repeat needed!

    # Convert back and flatten
    x_out = torch.view_as_real(x_rotated).reshape(B, H, L, D)

    return x_out.type_as(feats)  # restore original dtype

def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _block_dim(rope, feat_dim, info_encode='F'):
    assert info_encode in "FHW"
    if rope == None:
        dim = feat_dim
    elif rope == 'WAN':
        if info_encode == 'F':
            dim = feat_dim - 2 * (feat_dim // 3)
        else:
            dim = feat_dim // 3
    elif rope == 'HW':
        if info_encode == 'F':
            raise ValueError("WAN never encode Frame information")
        elif info_encode in 'HW':
            dim = feat_dim // 2
    elif rope in ['SO4-QK', 'SO4-QK-MS', 'CaPE', 'GTA', 'PRoPE']:
        if info_encode == 'F':
            raise ValueError("WAN never encode Frame information")
        else:
            dim = feat_dim // 4

    return dim