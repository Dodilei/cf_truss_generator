import numpy as np

K_BUCK = 0.8  # efetivo flambagem (Euler)


def _get_ud_strengths(ud_mat, fallback_Xt=1.5e9, fallback_Xc=1.2e9):
    Xt = None
    Xc = None
    for k in ["Xt", "X_t", "sigma1t", "S1t", "X1t"]:
        if hasattr(ud_mat, k):
            Xt = float(getattr(ud_mat, k))
            break
    for k in ["Xc", "X_c", "sigma1c", "S1c", "X1c"]:
        if hasattr(ud_mat, k):
            Xc = float(getattr(ud_mat, k))
            break
    if Xt is None or not np.isfinite(Xt) or Xt <= 0:
        Xt = float(fallback_Xt)
    if Xc is None or not np.isfinite(Xc) or Xc <= 0:
        Xc = float(fallback_Xc)
    return Xt, Xc


# =========================
# FAIL: TSAI-WU (UNIAXIAL) + BUCKLING
# =========================
def sf_tsaiwu_uniaxial(axial_stresses, ud_mat):
    """
    Tsai-Wu uniaxial (σ1=axial_stress, σ2=τ=0):
      FI = F1*σ + F11*σ² <= 1
    Resolve para k>0 em FI(kσ)=1.
    """
    Xt, Xc = _get_ud_strengths(ud_mat)
    stresses = np.asarray(axial_stresses, dtype=float)
    sf_tsaiwu = np.full_like(stresses, np.inf, dtype=float)

    F1 = 1.0 / Xt - 1.0 / Xc
    F11 = 1.0 / (Xt * Xc)

    # Standard mask for non-zero stresses
    mask_active = np.abs(stresses) > 1e-12
    s_active = stresses[mask_active]

    # Quadratic equation coefficients: quad_coeff*k² + lin_coeff*k + const_coeff = 0
    quad_coeff = F11 * s_active**2
    lin_coeff = F1 * s_active
    const_coeff = -1.0

    # Calculate discriminant for robust quadratic roots
    discriminant = lin_coeff**2 - 4.0 * quad_coeff * const_coeff
    discriminant[discriminant < 0.0] = 0.0
    sqrt_disc = np.sqrt(discriminant)

    sf_root1 = np.full_like(s_active, np.inf, dtype=float)
    sf_root2 = np.full_like(s_active, np.inf, dtype=float)

    mask_quad = np.abs(quad_coeff) > 1e-18
    sf_root1[mask_quad] = (-lin_coeff[mask_quad] + sqrt_disc[mask_quad]) / (
        2.0 * quad_coeff[mask_quad]
    )
    sf_root2[mask_quad] = (-lin_coeff[mask_quad] - sqrt_disc[mask_quad]) / (
        2.0 * quad_coeff[mask_quad]
    )

    # Fallback to linear solution if quad_coeff ~ 0 (e.g., if F11 is zero): lin_coeff*k = 1
    mask_linear = ~mask_quad
    lin_coeff_pure = lin_coeff[mask_linear]
    sf_linear = np.full_like(lin_coeff_pure, np.inf, dtype=float)
    mask_valid_lin = np.abs(lin_coeff_pure) > 1e-18
    sf_linear[mask_valid_lin] = 1.0 / lin_coeff_pure[mask_valid_lin]
    sf_root1[mask_linear] = sf_linear

    # Select the valid positive safety factor
    sf_final = np.maximum(sf_root1, sf_root2)
    sf_final[(~np.isfinite(sf_final)) | (sf_final <= 0.0)] = np.inf

    sf_tsaiwu[mask_active] = sf_final
    return sf_tsaiwu


def sf_buckling_column(truss, axial_forces, E_eff):
    element_I = truss.element_I
    element_length = truss.element_length

    with np.errstate(divide="ignore", invalid="ignore"):
        sf_buck = (
            -K_BUCK
            * (np.pi**2 * E_eff * element_I)
            / (element_length**2 * axial_forces)
        )

    sf_buck[~np.isfinite(sf_buck)] = np.inf
    sf_buck[sf_buck < 0] = np.inf

    return sf_buck
