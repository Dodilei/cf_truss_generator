# joint_analyzer.py
import numpy as np


def element_outer_diameters(element_kind, diam_o_family):
    ek = np.asarray(element_kind, dtype=int)
    do = np.zeros_like(ek, dtype=float)
    do[ek == 0] = float(diam_o_family[0])
    do[ek == 1] = float(diam_o_family[1])
    do[ek == 2] = float(diam_o_family[2])
    return do


def check_bonded_joints(
    axial_forces,
    element_kind,
    diam_o_family,
    overlap_ratio=1.5,
    tau_allow=30e6,
):
    """
    Junta colada (cisalhamento médio na cola):
      tau = |F| / (pi*D*l), com l = overlap_ratio*D
      SF  = tau_allow / tau

    Retorna: dict com SF_min, tau_max, SF_por_membro, tau_por_membro
    """
    F = np.asarray(axial_forces, dtype=float)
    D = element_outer_diameters(element_kind, diam_o_family)

    overlap_ratio = float(overlap_ratio)
    tau_allow = float(tau_allow)

    l = overlap_ratio * D
    A_bond = np.pi * D * l  # área de colagem (cilindro)

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.abs(F) / A_bond
        sf = tau_allow / tau

    tau[~np.isfinite(tau)] = 0.0
    sf[~np.isfinite(sf)] = np.inf

    return dict(
        sf_min=float(np.min(sf)),
        tau_max=float(np.max(tau)),
        sf_member=sf,
        tau_member=tau,
        overlap_length=l,
        area_bond=A_bond,
        tau_allow=tau_allow,
        overlap_ratio=overlap_ratio,
    )


def estimate_joint_mass_sleeves(
    element_kind,
    diam_o_family,
    overlap_ratio=1.5,
    sleeve_thickness=0.5e-3,
    rho_sleeve=2700.0,
    ends_per_member=2.0,
    mass_factor=1.0,
):
    """
    Estima massa adicionada por "mangas" (sleeves) em cada extremidade do membro.
    Modelo: cilindro oco de espessura sleeve_thickness, comprimento l=overlap_ratio*D,
    com diâmetro interno ~ D (tubo) e diâmetro externo ~ D + 2*t.

    Retorna massa total (kg).
    """
    D = element_outer_diameters(element_kind, diam_o_family)

    overlap_ratio = float(overlap_ratio)
    t = float(sleeve_thickness)
    rho = float(rho_sleeve)
    ends = float(ends_per_member)
    mf = float(mass_factor)

    l = overlap_ratio * D
    Di = D
    Do = D + 2.0 * t

    A_shell = 0.25 * np.pi * (Do**2 - Di**2)
    V = A_shell * l

    m = mf * ends * rho * V
    m[~np.isfinite(m)] = 0.0
    return float(np.sum(m))
