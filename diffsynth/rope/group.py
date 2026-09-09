import torch
from typing import Tuple

import torch

from typing import Optional, Tuple
from .cache import SE3ToSO4ElementCache

CACHE_SIZE = 16
BASE = 10_000

def rotvec_to_so3(v: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation vectors to SO(3) matrices.
    Args:
        v: (..., 3)
    Returns:
        R: (..., 3, 3)
    """
    angle = torch.norm(v, dim=-1, keepdim=True)
    small = angle.squeeze(-1) < 1e-8
    angle_safe = torch.where(small.unsqueeze(-1), torch.ones_like(angle), angle)
    axis = v / angle_safe

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1)
    ], dim=-2)

    sin_a = torch.sin(angle).squeeze(-1)
    cos_a = torch.cos(angle).squeeze(-1)
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand(*v.shape[:-1], 3, 3)

    R_out = (
        I +
        sin_a.unsqueeze(-1).unsqueeze(-1) * K +
        (1.0 - cos_a).unsqueeze(-1).unsqueeze(-1) * (K @ K)
    )
    R_out = torch.where(small.unsqueeze(-1).unsqueeze(-1), I, R_out)
    return R_out

def so3_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert SO(3) matrix to unit quaternion (w, x, y, z).
    Assumes R is in SO(3). If not, call project_to_so3 first.
    """

    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    tr = m00 + m11 + m22

    # Pre-allocate
    quat = torch.empty((*R.shape[:-2], 4), dtype=R.dtype, device=R.device)

    # Case 1: tr > 0 → qw largest
    cond0 = tr > 0
    if cond0.any():
        S = torch.sqrt(tr[cond0] + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (m21[cond0] - m12[cond0]) / S
        qy = (m02[cond0] - m20[cond0]) / S
        qz = (m10[cond0] - m01[cond0]) / S
        quat[cond0] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Case 2: m00 is max diagonal → qx largest
    cond1 = (~cond0) & (m00 >= m11) & (m00 >= m22)
    if cond1.any():
        S = torch.sqrt(1.0 + m00[cond1] - m11[cond1] - m22[cond1]) * 2
        qw = (m21[cond1] - m12[cond1]) / S
        qx = 0.25 * S
        qy = (m01[cond1] + m10[cond1]) / S
        qz = (m02[cond1] + m20[cond1]) / S
        quat[cond1] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Case 3: m11 is max → qy largest
    cond2 = (~cond0) & ~cond1 & (m11 >= m22)
    if cond2.any():
        S = torch.sqrt(1.0 + m11[cond2] - m00[cond2] - m22[cond2]) * 2
        qw = (m02[cond2] - m20[cond2]) / S
        qx = (m01[cond2] + m10[cond2]) / S
        qy = 0.25 * S
        qz = (m12[cond2] + m21[cond2]) / S
        quat[cond2] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Case 4: m22 is max → qz largest
    cond3 = (~cond0) & ~cond1 & ~cond2
    if cond3.any():
        S = torch.sqrt(1.0 + m22[cond3] - m00[cond3] - m11[cond3]) * 2
        qw = (m10[cond3] - m01[cond3]) / S
        qx = (m02[cond3] + m20[cond3]) / S
        qy = (m12[cond3] + m21[cond3]) / S
        qz = 0.25 * S
        quat[cond3] = torch.stack([qw, qx, qy, qz], dim=-1)

    # Normalize (for residual error)
    quat = quat / torch.norm(quat, dim=-1, keepdim=True).clamp(min=1e-8)
    return quat


def quat_left_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack([w, -x, -y, -z], dim=-1)
    row1 = torch.stack([x,  w, -z,  y], dim=-1)
    row2 = torch.stack([y,  z,  w, -x], dim=-1)
    row3 = torch.stack([z, -y,  x,  w], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


def quat_right_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack([w, -x, -y, -z], dim=-1)
    row1 = torch.stack([x,  w,  z, -y], dim=-1)
    row2 = torch.stack([y, -z,  w,  x], dim=-1)
    row3 = torch.stack([z,  y, -x,  w], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)

def process_translation(t_vec: torch.Tensor) -> torch.Tensor:
    """
    Scale translation and reduce its norm modulo 2π to the range [-π, π].
    
    Args:
        t_vec: (..., 3)
        t_scale: float
    
    Returns:
        tau: (..., 3) with ||tau|| ∈ [0, π], direction preserved for shortest arc
    """
    norm = torch.norm(t_vec, dim=-1, keepdim=True)
    direction = torch.where(norm < 1e-8,
                           torch.zeros_like(t_vec),
                           t_vec / norm)

    # Reduce angle modulo 2π to [0, 2π)
    norm_mod = torch.remainder(norm, 2 * torch.pi)  # [0, 2π)

    # Map to [-π, π)
    norm_centered = torch.where(norm_mod > torch.pi,
                                norm_mod - 2 * torch.pi,
                                norm_mod)  # now in [-π, π)

    # Take absolute value for shortest rotation (optional: keep sign if needed)
    # But for rotation vector, direction + magnitude <= π is enough
    # We keep the sign to allow "negative" rotations (though axis flips)
    # However, standard practice is to use positive norm and adjust axis

    # To ensure ||tau|| <= π, we take absolute value and flip axis if negative
    tau = direction * norm_centered
    return tau

def se3_to_so4_via_quat(se3_matrices: torch.Tensor) -> torch.Tensor:
    """
    Map SE(3) matrices to SO(4) matrices (non-group-preserving).
    
    Steps:
      1. Extract R and t from SE(3)
      2. Process t → tau (scaled + clamped to ||tau|| ≤ π)
      3. Build R1 = SO3(omega), R2 = SO3(tau)
      4. Convert to quaternions q1, q2
      5. Construct SO(4) = L(q1) @ R(q2^*)

    Args:
        se3_matrices: (..., 4, 4)
        t_scale: scaling factor for translation
    
    Returns:
        SO4 matrices: (..., 4, 4)
    """
    R_mat = se3_matrices[..., :3, :3]      # (..., 3, 3)
    t_vec = se3_matrices[..., :3, 3]       # (..., 3)

    # omega = rotation_matrix_to_rotvec(R_mat)
    tau = process_translation(t_vec)

    # R1 = rotvec_to_so3(omega)
    R1 = R_mat
    R2 = rotvec_to_so3(tau)

    q1 = so3_to_quaternion(R1)
    q2 = so3_to_quaternion(R2)
    q2_conj = torch.stack([q2[..., 0], -q2[..., 1], -q2[..., 2], -q2[..., 3]], dim=-1)

    L_q1 = quat_left_matrix(q1)
    R_q2_conj = quat_right_matrix(q2_conj)
    if torch.isnan(L_q1).any() or torch.isnan(R_q2_conj).any():
        print("NaN detected in quaternion matrices.")
        print("q1:", q1)
        print("q2_conj:", q2_conj)
        print("L_q1:", L_q1)
        print("R_q2_conj:", R_q2_conj) 
        print("R1:", R1)
        print("R2:", R2)
        print("tau:", tau)
        raise ValueError("NaN detected in quaternion matrices.")
    return torch.matmul(L_q1, R_q2_conj)


# Global cache instance
def scaled_se3_to_so4(X_se3: torch.Tensor, n: torch = 0.0) -> torch.Tensor:
    """
    Compute outer product over C and S: scale each (B,C) SE(3) by all S scales.
    
    Args:
        X_se3: (B, C, 4, 4)
        n: (S,) — scale exponents
    
    Returns:
        SO(4) matrices: (B, C, S, 4, 4)
    """
    B, C = X_se3.shape[:2]
    S = n.shape[0]

    # Expand X to (B, C, 1, 4, 4) → will broadcast over S
    X_exp = X_se3.unsqueeze(2)  # (B, C, 1, 4, 4)

    # Extract translation: (B, C, 1, 3, 1)
    trans = X_exp[..., :3, 3:4]  # (B, C, 1, 3, 1)

    # Scale factors: (S,) → (1, 1, S, 1, 1)
    scale_factors = (BASE ** (-n)).view(1, 1, -1, 1, 1)  # (1, 1, S, 1, 1)

    # Apply scaling via broadcasting → (B, C, S, 3, 1)
    trans_scaled = trans * scale_factors

    # Build scaled SE(3) matrices
    X_scaled = torch.eye(4, device=X_se3.device, dtype=X_se3.dtype).repeat(B, C, S, 1, 1)
    X_scaled[..., :3, :3] = X_exp[..., :3, :3].expand(B, C, S, 3, 3)  # copy rotation
    X_scaled[..., :3, 3:4] = trans_scaled  # assign scaled translation

    # Map to SO(4)
    return se3_to_so4(X_scaled)  # (B, C, S, 4, 4)

# Global cache instance
# _SE3_SO4_ELEM_CACHE = SE3ToSO4ElementCache(max_size=1024, eps=1e-5)

def se3_to_so4(X_se3: torch.Tensor) -> torch.Tensor:
    return se3_to_so4_via_quat(X_se3)  # [B, C, 6]

