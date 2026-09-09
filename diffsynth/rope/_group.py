import torch
from typing import Tuple

BASE = 10_000

# ----------------------------
# 1. SE(3) ↔ se(3) (6D twist) — Multi-Camera Aware
# ----------------------------

def se3_to_se3vec(X_se3: torch.Tensor, debug: bool = False) -> torch.Tensor:
    r"""
    Convert SE(3) transformation matrices to 6D Lie algebra twist vectors ξ = [ρ, φ] ∈ ℝ⁶.

    The SE(3) matrix is:
        X = [ R   t ] ∈ SE(3)
            [ 0ᵀ  1 ]
    where R ∈ SO(3), t ∈ ℝ³.

    The logarithmic map log: SE(3) → se(3) is:
        ξ = [ρ, φ] = [V⁻¹(φ) t, log_SO(3)(R)]

    ────────────────────────────────────────────────────────────────
    MATHEMATICAL BACKGROUND: SO(3) LOGARITHM MAP

    For R ∈ SO(3), there exists a unique rotation vector φ ∈ ℝ³ such that:
        R = exp(φ̂),   where φ̂ ∈ so(3) is skew-symmetric.

    The rotation angle is θ = ‖φ‖ ∈ [0, π], and the axis is a = φ / θ (if θ > 0).

    The standard formula for φ is:
        φ̂ = (θ / (2 sinθ)) (R − Rᵀ),    for θ ∈ (0, π)

    However, this formula has two singularities:

    1. **θ → 0**: sinθ → 0, but (R − Rᵀ) → 0 faster.
       - Use Taylor expansion: φ̂ ≈ ½(R − Rᵀ)

    2. **θ → π**: sinθ → 0, and (R − Rᵀ) → 0, but the ratio becomes unstable
       due to floating-point noise in non-exact SO(3) matrices.
       - At θ = π, R is symmetric (R = Rᵀ), so (R − Rᵀ) = 0 exactly.
       - But for θ ≈ π, numerical errors make (R − Rᵀ) ≠ 0, while sinθ ≈ 0,
         causing catastrophic amplification: φ ~ θ/(2 sinθ) * noise → ∞.

    To resolve θ ≈ π, we use the **eigenvector method**:

        When θ = π, R has eigenvalue +1 with eigenvector v (the rotation axis),
        and R = I + 2(vvᵀ − I) ⇒ diag(R) = 2v² − 1 ⇒ v_i = ±√((R_ii + 1)/2)

        A robust approximation (avoiding square roots and sign ambiguity) is:
            v ∝ [R₀₀ + 1, R₁₁ + 1, R₂₂ + 1]ᵀ

        Then φ = π · v / ‖v‖.

    This method is **division-free** and stable near θ = π.

    ────────────────────────────────────────────────────────────────
    LEFT JACOBIAN INVERSE V⁻¹(φ)

    The translation part is ρ = V⁻¹(φ) t, where:
        V(φ) = I + (1 − cosθ)/θ² φ̂ + (θ − sinθ)/θ³ φ̂²

    Its inverse admits the closed form:
        V⁻¹(φ) = I − ½ φ̂ + c₂(θ) φ̂²

    with:
                     ⎧ 1/12                      if θ → 0
            c₂(θ) = ⎨ (1/θ²) − (1 + cosθ)/(2θ sinθ)   if θ ∈ (0, π)
                     ⎩ 1/π²                      if θ → π

    We handle all three regimes with thresholding.

    ────────────────────────────────────────────────────────────────
    DTYPE HANDLING NOTE:

    Input may be in bfloat16 (common in mixed-precision training).
    Trigonometric and division operations are computed in float32 for numerical stability,
    then cast back to input dtype to avoid "dtype mismatch" errors during indexing.

    ────────────────────────────────────────────────────────────────

    Args:
        X_se3: [B, C, 4, 4] SE(3) matrices (assumed to have last row = [0,0,0,1])
        debug: bool, enable diagnostic prints

    Returns:
        xi: [B, C, 6] twist vectors [ρ_x, ρ_y, ρ_z, φ_x, φ_y, φ_z]
    """
    device = X_se3.device
    dtype = X_se3.dtype  # Preserve input dtype (e.g., bfloat16)

    B, C = X_se3.shape[:2]

    # Extract rotation and translation components
    R = X_se3[..., :3, :3]  # [B, C, 3, 3]
    t = X_se3[..., :3, 3]   # [B, C, 3]

    # ───────────────────────────────────────────────
    # STEP 1: Compute rotation angle θ from trace
    # tr(R) = 1 + 2 cosθ  ⇒  cosθ = (tr(R) - 1)/2
    # θ ∈ [0, π]
    # ───────────────────────────────────────────────
    traces = R.diagonal(dim1=-2, dim2=-1).sum(-1)      # [B, C]
    cos_theta = (traces - 1.0) / 2.0                   # May promote to float32

    # Clamp in float32 to avoid NaN, then cast back to original dtype
    cos_theta_f32 = torch.clamp(cos_theta.float(), -1.0, 1.0)
    cos_theta = cos_theta_f32.to(dtype)

    theta = torch.acos(cos_theta).unsqueeze(-1)        # [B, C, 1]
    sin_theta = torch.sin(theta)

    # Predefine mathematical constants in correct dtype
    pi_val = torch.tensor(torch.pi, device=device, dtype=dtype)

    # Initialize output rotation vector in input dtype
    phi = torch.zeros(B, C, 3, device=device, dtype=dtype)

    # ───────────────────────────────────────────────
    # STEP 2: Compute φ via hybrid method
    # ───────────────────────────────────────────────
    # Thresholds for regime switching
    THETA_NEAR_ZERO = 1e-8      # radians
    THETA_NEAR_PI   = 1e-3      # radians (≈ 0.057° from π)

    # Boolean masks for batch elements
    theta_squeezed = theta.squeeze(-1)                 # [B, C]
    mask_near_pi = theta_squeezed > (pi_val - THETA_NEAR_PI)  # [B, C]
    mask_standard = ~mask_near_pi

    # ───────────────────────────────────────────────
    # CASE A: Standard regime (θ ∈ [0, π - δ])
    # Use: φ̂ = (θ / (2 sinθ)) (R - Rᵀ)
    # ───────────────────────────────────────────────
    if torch.any(mask_standard):
        R_std = R[mask_standard]                       # [N, 3, 3]
        theta_std = theta[mask_standard]               # [N, 1]
        sin_theta_std = sin_theta[mask_standard]       # [N, 1]

        # Compute scale_skew in float32 for stability, then cast
        theta_f32 = theta_std.float()
        sin_theta_f32 = torch.sin(theta_f32)
        scale_skew_f32 = torch.where(
            theta_f32 > THETA_NEAR_ZERO,
            theta_f32 / (2.0 * sin_theta_f32),
            torch.full_like(theta_f32, 0.5)
        )
        scale_skew = scale_skew_f32.to(dtype)

        phi_skew_std = scale_skew.unsqueeze(-1) * (R_std - R_std.transpose(-1, -2))
        phi_std = torch.stack([
            phi_skew_std[:, 2, 1],  # φ_x = φ̂[2,1]
            phi_skew_std[:, 0, 2],  # φ_y = φ̂[0,2]
            phi_skew_std[:, 1, 0],  # φ_z = φ̂[1,0]
        ], dim=-1)  # [N, 3]

        phi[mask_standard] = phi_std  # Same dtype → no error

    # ───────────────────────────────────────────────
    # CASE B: Near θ = π (θ ∈ (π - δ, π])
    # Use eigenvector method: φ = π · v, where Rv = v
    # Derivation:
    #   For θ = π, R = I + 2(vvᵀ - I) ⇒ R_ii = 2v_i² - 1 ⇒ v_i² = (R_ii + 1)/2
    #   Thus, v ∝ [R₀₀ + 1, R₁₁ + 1, R₂₂ + 1]ᵀ
    #   This avoids division and is robust to noise.
    # ───────────────────────────────────────────────
    if torch.any(mask_near_pi):
        R_pi = R[mask_near_pi]                         # [M, 3, 3]

        # Compute unnormalized axis: v ∝ diag(R) + 1
        v_unnorm = torch.diagonal(R_pi, dim1=-2, dim2=-1) + 1.0  # [M, 3]

        # Handle degenerate case (should not occur for valid R near π)
        v_norm = torch.norm(v_unnorm, dim=-1, keepdim=True)      # [M, 1]
        v_unnorm = torch.where(
            v_norm < 1e-8,
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            v_unnorm
        )
                # Recompute norm after replacement (in dtype)
        v_norm_final = torch.norm(v_unnorm.float(), dim=-1, keepdim=True).to(dtype)
        v = v_unnorm / v_norm_final  # [M, 3] in correct dtype
        # Set φ = π · v (since ‖φ‖ = θ ≈ π)
        phi_pi = pi_val * v
        phi[mask_near_pi] = phi_pi

    # ───────────────────────────────────────────────
    # STEP 3: Compute V⁻¹(φ) and ρ = V⁻¹ t
    # V⁻¹ = I - ½ φ̂ + c₂(θ) φ̂²
    # ───────────────────────────────────────────────
    # Reconstruct full skew-symmetric matrix φ̂ from φ (in correct dtype)
    phi_skew_full = torch.zeros(B, C, 3, 3, device=device, dtype=dtype)
    phi_skew_full[:, :, 0, 1] = -phi[:, :, 2]
    phi_skew_full[:, :, 0, 2] =  phi[:, :, 1]
    phi_skew_full[:, :, 1, 0] =  phi[:, :, 2]
    phi_skew_full[:, :, 1, 2] = -phi[:, :, 0]
    phi_skew_full[:, :, 2, 0] = -phi[:, :, 1]
    phi_skew_full[:, :, 2, 1] =  phi[:, :, 0]

    # Identity matrix batch
    I = torch.eye(3, device=device, dtype=dtype).repeat(B, C, 1, 1)

    # Compute φ̂²
    phi_skew_sq = torch.matmul(phi_skew_full, phi_skew_full)

    # Safe theta for division (avoid div-by-zero at θ=0)
    theta_safe = torch.where(theta > THETA_NEAR_ZERO, theta, torch.ones_like(theta))

    # Compute c₂(θ) with triple regime handling — use float32 internally
    theta_f32 = theta_safe.float()
    sin_theta_f32 = torch.sin(theta_f32)
    cos_theta_f32 = torch.cos(theta_f32)
    c2_default_f32 = (1.0 / theta_f32**2) - (1.0 + cos_theta_f32) / (2.0 * theta_f32 * sin_theta_f32)
    c2_default = c2_default_f32.to(dtype)

    c2_near_zero = torch.full_like(theta, 1.0 / 12.0, dtype=dtype)          # Taylor at θ=0
    c2_near_pi = torch.full_like(theta, 1.0 / (pi_val ** 2), dtype=dtype)   # Limit as θ→π

    mask_zero = theta < THETA_NEAR_ZERO
    mask_pi = torch.abs(theta - pi_val) < THETA_NEAR_PI

    c2 = torch.where(
        mask_zero,
        c2_near_zero,
        torch.where(mask_pi, c2_near_pi, c2_default)
    )  # [B, C, 1]

    # Assemble V⁻¹
    V_inv = I - 0.5 * phi_skew_full + c2.unsqueeze(-1) * phi_skew_sq

    # Compute ρ = V⁻¹ t
    rho = torch.matmul(V_inv, t.unsqueeze(-1)).squeeze(-1)  # [B, C, 3]

    if debug:
        max_phi = phi.abs().max().item()
        max_rho = rho.abs().max().item()
        if max_phi > 10.0 or max_rho > 10.0:
            print(f"⚠️ Large values detected:")
            print(f"   max|phi| = {max_phi:.2f} (expected ≤ {pi_val.item():.2f})")
            print(f"   max|rho| = {max_rho:.2f}")
            print(f"   theta range: [{theta.min().item():.4f}, {theta.max().item():.4f}]")

    # ───────────────────────────────────────────────
    # FINAL OUTPUT: ξ = [ρ, φ] ∈ ℝ⁶
    # ───────────────────────────────────────────────
    return torch.cat([rho, phi], dim=-1)  # [B, C, 6]

def se3vec_to_se3(xi: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D twist vectors to SE(3) matrices.
    
    Args:
        xi: [B, C, 6] twist vectors [rho, phi]
        
    Returns:
        X_se3: [B, C, 4, 4] SE(3) matrices
    """
    B, C = xi.shape[:2]
    device, dtype = xi.device, xi.dtype
    
    rho = xi[..., :3]      # [B, C, 3]
    phi = xi[..., 3:]      # [B, C, 3]

    # Build φ̂ (skew-symmetric)
    phi_skew = torch.zeros(B, C, 3, 3, device=device, dtype=dtype)
    phi_skew[:, :, 0, 1] = -phi[:, :, 2]
    phi_skew[:, :, 0, 2] =  phi[:, :, 1]
    phi_skew[:, :, 1, 0] =  phi[:, :, 2]
    phi_skew[:, :, 1, 2] = -phi[:, :, 0]
    phi_skew[:, :, 2, 0] = -phi[:, :, 1]
    phi_skew[:, :, 2, 1] =  phi[:, :, 0]

    # Compute θ = ||phi||
    theta = torch.norm(phi, dim=-1, keepdim=True)  # [B, C, 1]
    theta_safe = torch.where(theta > 1e-6, theta, torch.ones_like(theta))

    # Rodrigues' formula
    I = torch.eye(3, device=device, dtype=dtype).repeat(B, C, 1, 1)  # [B, C, 3, 3]
    K = phi_skew / theta_safe.unsqueeze(-1)  # [B, C, 3, 3]
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    R = I + sin_t.unsqueeze(-1) * K + (1 - cos_t).unsqueeze(-1) * torch.matmul(K, K)  # [B, C, 3, 3]

    # Left Jacobian V
    theta2 = theta_safe ** 2
    A = torch.where(theta > 1e-6, sin_t / theta, torch.ones_like(theta))
    B_coeff = torch.where(theta > 1e-6, (1 - cos_t) / theta2, torch.full_like(theta, 0.5))
    C_coeff = torch.where(theta > 1e-6, (1 - A) / theta2, torch.full_like(theta, 1.0 / 6.0))

    V = (A.unsqueeze(-1) * I +
         B_coeff.unsqueeze(-1) * phi_skew +
         C_coeff.unsqueeze(-1) * torch.matmul(phi_skew, phi_skew))  # [B, C, 3, 3]

    t = torch.matmul(V, rho.unsqueeze(-1)).squeeze(-1)  # [B, C, 3]

    # Assemble SE(3)
    X_se3 = torch.eye(4, device=device, dtype=dtype).repeat(B, C, 1, 1)
    X_se3[:, :, :3, :3] = R
    X_se3[:, :, :3, 3] = t
    return X_se3


# ----------------------------
# 2. SO(4) ↔ so(4) — Multi-Camera Aware
# ----------------------------

def so4_to_so4vec(X_so4: torch.Tensor) -> torch.Tensor:
    """
    Convert SO(4) matrices to 6D Lie algebra vectors.
    Convention: [Ω_01, Ω_02, Ω_03, Ω_12, Ω_13, Ω_23]
    
    Args:
        X_so4: [B, C, 4, 4] SO(4) matrices
        
    Returns:
        omega: [B, C, 6]
    """
    Omega = torch.linalg.logm(X_so4)  # [B, C, 4, 4]
    omega = torch.stack([
        Omega[:, :, 0, 1],
        Omega[:, :, 0, 2],
        Omega[:, :, 0, 3],
        Omega[:, :, 1, 2],
        Omega[:, :, 1, 3],
        Omega[:, :, 2, 3]
    ], dim=-1)  # [B, C, 6]
    return omega


def so4vec_to_so4(omega: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D Lie algebra vectors to SO(4) matrices.
    Handles bfloat16 safely by computing exp in float32.
    """
    B, C = omega.shape[:2]
    device = omega.device
    dtype = omega.dtype  # e.g., bfloat16

    # Build Omega in original dtype
    Omega = torch.zeros(B, C, 4, 4, device=device, dtype=dtype)
    Omega[:, :, 0, 1] =  omega[:, :, 0]
    Omega[:, :, 0, 2] =  omega[:, :, 1]
    Omega[:, :, 0, 3] =  omega[:, :, 2]
    Omega[:, :, 1, 2] =  omega[:, :, 3]
    Omega[:, :, 1, 3] =  omega[:, :, 4]
    Omega[:, :, 2, 3] =  omega[:, :, 5]

    Omega[:, :, 1, 0] = -omega[:, :, 0]
    Omega[:, :, 2, 0] = -omega[:, :, 1]
    Omega[:, :, 3, 0] = -omega[:, :, 2]
    Omega[:, :, 2, 1] = -omega[:, :, 3]
    Omega[:, :, 3, 1] = -omega[:, :, 4]
    Omega[:, :, 3, 2] = -omega[:, :, 5]

    # 🔑 CRITICAL: Compute matrix exponential in float32
    Omega_f32 = Omega.float()
    X_so4_f32 = torch.linalg.matrix_exp(Omega_f32)

    # Cast back to original dtype
    X_so4 = X_so4_f32.to(dtype)

    return X_so4

# ----------------------------
# 3. SE(3) ↔ SO(4) via embedding — Multi-Camera
# ----------------------------

def se3_to_so4_via_vec(X_se3: torch.Tensor) -> torch.Tensor:
    xi = se3_to_se3vec(X_se3)  # [B, C, 6]
    rho, phi = xi[..., :3], xi[..., 3:]
    omega = torch.cat([rho, phi], dim=-1)  # [B, C, 6]
    return so4vec_to_so4(omega)


def so4_to_se3_via_vec(X_so4: torch.Tensor) -> torch.Tensor:
    omega = so4_to_so4vec(X_so4)  # [B, C, 6]
    xi = omega  # [B, C, 6]
    return se3vec_to_se3(xi)


def scaled_se3_to_so4_via_vec(X_se3: torch.Tensor, n: float = 0.0) -> torch.Tensor:
    xi = se3_to_se3vec(X_se3)  # [B, C, 6]
    rho, phi = xi[..., :3], xi[..., 3:]
    rho = rho / (BASE ** n)
    omega = torch.cat([rho, phi], dim=-1)  # [B, C, 6]
    omega = normalize_so4_lie_algebra(omega)
    ri = so4vec_to_so4(omega)
    return ri

def normalize_so4_lie_algebra(omega: torch.Tensor) -> torch.Tensor:
    r"""
    Normalize an element of the Lie algebra 𝔰𝔬(4) to its canonical representative
    within the principal domain of the exponential map.

    ────────────────────────────────────────────────────────────────
    MATHEMATICAL BACKGROUND

    The Lie algebra 𝔰𝔬(4) consists of 4×4 real skew-symmetric matrices.
    It is 6-dimensional, and any Ω ∈ 𝔰𝔬(4) can be written as:

        Ω = ∑_{0≤i<j≤3} ω_{ij} (E_{ij} - E_{ji})

    where E_{ij} is the matrix with 1 at (i,j) and 0 elsewhere.

    A fundamental result in 4D geometry is the **isomorphism**:
        𝔰𝔬(4) ≅ 𝔰𝔬(3) ⊕ 𝔰𝔬(3)

    This arises from the Hodge star operator on ∧²ℝ⁴, which splits bivectors into
    self-dual (SD) and anti-self-dual (ASD) parts.

    Concretely, define:
        a = ½ [ ω₂₃ + ω₀₁,  ω₃₁ + ω₀₂,  ω₁₂ + ω₀₃ ]ᵀ   ∈ ℝ³  (self-dual)
        b = ½ [ ω₂₃ − ω₀₁,  ω₃₁ − ω₀₂,  ω₁₂ − ω₀₃ ]ᵀ   ∈ ℝ³  (anti-self-dual)

    Then (a, b) ∈ 𝔰𝔬(3) ⊕ 𝔰𝔬(3), and the original Ω is recovered via:
        ω₀₁ = a₁ − b₁,    ω₂₃ = a₁ + b₁
        ω₀₂ = a₂ − b₂,    ω₃₁ = a₂ + b₂  ⇒ ω₁₃ = −ω₃₁
        ω₀₃ = a₃ − b₃,    ω₁₂ = a₃ + b₃

    The exponential map exp: 𝔰𝔬(4) → SO(4) acts as:
        exp(Ω) = (exp(â), exp(b̂))  under the double cover Spin(4) ≅ SU(2)×SU(2)

    The rotation angles in the two orthogonal 2-planes are:
        α = ‖a‖ + ‖b‖,     β = |‖a‖ − ‖b‖|

    Since SO(4) is compact, exp is periodic with period 2π in α and β.
    Thus, the principal domain is α, β ∈ [−π, π] (or [0, 2π)).

    This function maps any ω ∈ ℝ⁶ to an equivalent ω̃ such that the resulting
    (α, β) lie in [−π, π], ensuring numerical stability and uniqueness.

    ────────────────────────────────────────────────────────────────

    Args:
        omega: [..., 6] — coordinates [ω₀₁, ω₀₂, ω₀₃, ω₁₂, ω₁₃, ω₂₃]

    Returns:
        omega_norm: [..., 6] — normalized coefficients representing the same SO(4) rotation
    """
    device, dtype = omega.device, omega.dtype
    shape_prefix = omega.shape[:-1]
    omega = omega.view(-1, 6)  # Flatten batch dimensions
    B = omega.shape[0]

    # ───────────────────────────────────────────────
    # Step 1: Reconstruct skew-symmetric matrix Ω
    # Ω_{ij} = ω_{ij} for i < j, Ω_{ji} = -ω_{ij}
    # ───────────────────────────────────────────────
    Omega = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    Omega[:, 0, 1] =  omega[:, 0]   # ω₀₁
    Omega[:, 0, 2] =  omega[:, 1]   # ω₀₂
    Omega[:, 0, 3] =  omega[:, 2]   # ω₀₃
    Omega[:, 1, 2] =  omega[:, 3]   # ω₁₂
    Omega[:, 1, 3] =  omega[:, 4]   # ω₁₃
    Omega[:, 2, 3] =  omega[:, 5]   # ω₂₃

    Omega[:, 1, 0] = -omega[:, 0]
    Omega[:, 2, 0] = -omega[:, 1]
    Omega[:, 3, 0] = -omega[:, 2]
    Omega[:, 2, 1] = -omega[:, 3]
    Omega[:, 3, 1] = -omega[:, 4]
    Omega[:, 3, 2] = -omega[:, 5]

    # ───────────────────────────────────────────────
    # Step 2: Decompose into self-dual (a) and anti-self-dual (b) parts
    #
    # The Hodge star *: ∧²ℝ⁴ → ∧²ℝ⁴ satisfies *² = I, so eigenvalues ±1.
    # Self-dual: *η = η, Anti-self-dual: *η = −η.
    #
    # In coordinates, this gives:
    #   a_k = ½ (ω_{ij} + ½ ε_{ijkl} ω_{kl})  → simplified to:
    #   a = ½ [ω₂₃ + ω₀₁, ω₃₁ + ω₀₂, ω₁₂ + ω₀₃]
    #   b = ½ [ω₂₃ − ω₀₁, ω₃₁ − ω₀₂, ω₁₂ − ω₀₃]
    #
    # Note: ω₃₁ = −ω₁₃, etc., due to skew-symmetry.
    # ───────────────────────────────────────────────
    w01, w02, w03, w12, w13, w23 = omega.unbind(dim=-1)
    w31 = -w13  # because Ω_{31} = -Ω_{13}
    # w32 = -w23  # not needed
    # w21 = -w12  # not needed

    # Self-dual part a ∈ ℝ³
    a1 = 0.5 * (w23 + w01)
    a2 = 0.5 * (w31 + w02)  # = 0.5*(-w13 + w02)
    a3 = 0.5 * (w12 + w03)
    a = torch.stack([a1, a2, a3], dim=-1)  # [B, 3]

    # Anti-self-dual part b ∈ ℝ³
    b1 = 0.5 * (w23 - w01)
    b2 = 0.5 * (w31 - w02)
    b3 = 0.5 * (w12 - w03)
    b = torch.stack([b1, b2, b3], dim=-1)  # [B, 3]

    # Compute norms: these correspond to rotation magnitudes in each su(2) factor
    norm_a = torch.norm(a, dim=-1, keepdim=True)  # [B, 1] = ‖a‖
    norm_b = torch.norm(b, dim=-1, keepdim=True)  # [B, 1] = ‖b‖

    # ───────────────────────────────────────────────
    # Step 3: Compute canonical rotation angles (α, β)
    #
    # Any rotation in SO(4) is conjugate to a double rotation:
    #   R = R(α) ⊕ R(β)
    # where R(θ) is a 2D rotation by θ.
    #
    # The angles are related to (a,b) by:
    #   α = ‖a‖ + ‖b‖,    β = |‖a‖ − ‖b‖|
    #
    # This ensures α ≥ β ≥ 0, and covers all possible SO(4) rotations.
    # ───────────────────────────────────────────────
    alpha = norm_a + norm_b          # [B, 1]
    beta  = torch.abs(norm_a - norm_b)  # [B, 1]

    # ───────────────────────────────────────────────
    # Step 4: Wrap angles to principal domain [−π, π]
    #
    # Since exp(Ω) is periodic with period 2π in both α and β,
    # we reduce them modulo 2π to the symmetric interval [−π, π].
    # This ensures numerical stability and uniqueness.
    # ───────────────────────────────────────────────
    alpha_wrapped = (alpha + torch.pi) % (2 * torch.pi) - torch.pi
    beta_wrapped  = (beta  + torch.pi) % (2 * torch.pi) - torch.pi

    # ───────────────────────────────────────────────
    # Step 5: Reconstruct normalized (a_new, b_new)
    #
    # We want new vectors a_new, b_new such that:
    #   ‖a_new‖ + ‖b_new‖ = α_wrapped
    #   |‖a_new‖ − ‖b_new‖| = β_wrapped
    #
    # Solving this system gives:
    #   ‖a_new‖ = (α_wrapped + β_wrapped) / 2
    #   ‖b_new‖ = (α_wrapped − β_wrapped) / 2
    #
    # We preserve the original directions of a and b (to maintain rotation planes).
    # ───────────────────────────────────────────────
    # Handle zero-norm cases to avoid division by zero
    a_dir = torch.where(norm_a > 1e-8, a / norm_a, torch.zeros_like(a))
    b_dir = torch.where(norm_b > 1e-8, b / norm_b, torch.zeros_like(b))

    # Solve for new norms
    norm_a_new = (alpha_wrapped + beta_wrapped) / 2
    norm_b_new = (alpha_wrapped - beta_wrapped) / 2

    # Ensure non-negativity (numerical safety)
    norm_a_new = torch.clamp(norm_a_new, min=0.0)
    norm_b_new = torch.clamp(norm_b_new, min=0.0)

    # Reconstruct vectors with new norms but same directions
    a_new = norm_a_new * a_dir
    b_new = norm_b_new * b_dir

    # ───────────────────────────────────────────────
    # Step 6: Map back to ω coordinates
    #
    # Invert the SD/ASD decomposition:
    #   ω₀₁ = a₁ − b₁,    ω₂₃ = a₁ + b₁
    #   ω₀₂ = a₂ − b₂,    ω₃₁ = a₂ + b₂  ⇒ ω₁₃ = −ω₃₁ = −(a₂ + b₂)
    #   ω₀₃ = a₃ − b₃,    ω₁₂ = a₃ + b₃
    # ───────────────────────────────────────────────
    w01_new = a_new[:, 0] - b_new[:, 0]
    w02_new = a_new[:, 1] - b_new[:, 1]
    w03_new = a_new[:, 2] - b_new[:, 2]
    w12_new = a_new[:, 2] + b_new[:, 2]
    w13_new = -(a_new[:, 1] + b_new[:, 1])  # because ω₁₃ = -ω₃₁
    w23_new = a_new[:, 0] + b_new[:, 0]

    omega_norm = torch.stack([
        w01_new, w02_new, w03_new,
        w12_new, w13_new, w23_new
    ], dim=-1)

    return omega_norm.view(*shape_prefix, 6)

def normalize_so4_lie_algebra_simple(omega: torch.Tensor, max_norm: float = torch.pi) -> torch.Tensor:
    """
    Simple normalization: scale ω to have norm ≤ max_norm.
    Preserves direction, bounds magnitude.
    """
    norm = torch.norm(omega, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return omega * scale