from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


# -------------------------
# Material UD (lamina)
# -------------------------
@dataclass(frozen=True)
class UDMaterial:
    # Propriedades elásticas (lamina ortotrópica) - plano de tensão
    E1: float   # Pa (direção fibra)
    E2: float   # Pa (transversal)
    G12: float  # Pa (cisalhamento 1-2)
    nu12: float # (-)

    # Opcionais (pra usar depois em falha / densidade)
    rho: float = 0.0
    Xt: float = np.inf  # Pa
    Xc: float = np.inf  # Pa
    Yt: float = np.inf  # Pa
    Yc: float = np.inf  # Pa
    S12: float = np.inf # Pa

    @property
    def nu21(self) -> float:
        # nu21 = nu12 * E2/E1
        return float(self.nu12) * float(self.E2) / float(self.E1)


# -------------------------
# CLT core
# -------------------------
def lamina_Q(mat: UDMaterial) -> np.ndarray:
    """
    Reduced stiffness matrix Q (3x3) para lamina ortotrópica (plano de tensão):
    [Q11 Q12  0 ]
    [Q12 Q22  0 ]
    [ 0   0  Q66]
    """
    E1 = float(mat.E1)
    E2 = float(mat.E2)
    nu12 = float(mat.nu12)
    nu21 = float(mat.nu21)
    G12 = float(mat.G12)

    den = 1.0 - nu12 * nu21
    if den <= 0:
        raise ValueError("Material inválido: 1 - nu12*nu21 <= 0")

    Q11 = E1 / den
    Q22 = E2 / den
    Q12 = nu12 * E2 / den
    Q66 = G12

    Q = np.array([[Q11, Q12, 0.0],
                  [Q12, Q22, 0.0],
                  [0.0, 0.0, Q66]], dtype=float)
    return Q


def Qbar_from_Q(Q: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Transformação de Q -> Qbar para ângulo theta (graus), CLT padrão.
    Retorna Qbar (3x3).
    """
    th = np.deg2rad(float(theta_deg))
    c = np.cos(th)
    s = np.sin(th)

    Q11, Q12, Q22, Q66 = float(Q[0, 0]), float(Q[0, 1]), float(Q[1, 1]), float(Q[2, 2])

    c2 = c * c
    s2 = s * s
    c4 = c2 * c2
    s4 = s2 * s2
    s2c2 = s2 * c2

    Qbar11 = Q11 * c4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * s4
    Qbar22 = Q11 * s4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * c4
    Qbar12 = (Q11 + Q22 - 4.0 * Q66) * s2c2 + Q12 * (s4 + c4)

    Qbar16 = (Q11 - Q12 - 2.0 * Q66) * (c * c2 * s) - (Q22 - Q12 - 2.0 * Q66) * (c * s2 * s)
    Qbar26 = (Q11 - Q12 - 2.0 * Q66) * (c * s2 * s) - (Q22 - Q12 - 2.0 * Q66) * (c * c2 * s)

    Qbar66 = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2c2 + Q66 * (s4 + c4)

    Qbar = np.array([[Qbar11, Qbar12, Qbar16],
                     [Qbar12, Qbar22, Qbar26],
                     [Qbar16, Qbar26, Qbar66]], dtype=float)
    return Qbar


def laminate_ABD(
    angles_deg: Iterable[float],
    t_plies: Iterable[float],
    mat: UDMaterial,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Monta A, B, D de um laminado geral:
    N = A*eps0 + B*kappa
    M = B*eps0 + D*kappa

    Retorna: (A, B, D, h_total, z_coords)
    z_coords tem tamanho (nplies+1), com z0=-h/2 ... zn=+h/2
    """
    angles = np.array(list(angles_deg), dtype=float)
    t = np.array(list(t_plies), dtype=float)

    if angles.size == 0:
        raise ValueError("Laminado inválido: sem plies.")
    if angles.size != t.size:
        raise ValueError("Laminado inválido: angles e t_plies com tamanhos diferentes.")
    if np.any(t <= 0):
        raise ValueError("Laminado inválido: espessura de ply <= 0.")

    h = float(np.sum(t))
    z = np.empty(angles.size + 1, dtype=float)
    z[0] = -0.5 * h
    for k in range(angles.size):
        z[k + 1] = z[k] + t[k]

    Q = lamina_Q(mat)
    A = np.zeros((3, 3), dtype=float)
    B = np.zeros((3, 3), dtype=float)
    D = np.zeros((3, 3), dtype=float)

    for k in range(angles.size):
        Qbar = Qbar_from_Q(Q, float(angles[k]))
        zk0 = float(z[k])
        zk1 = float(z[k + 1])

        A += Qbar * (zk1 - zk0)
        B += 0.5 * Qbar * (zk1**2 - zk0**2)
        D += (1.0 / 3.0) * Qbar * (zk1**3 - zk0**3)

    return A, B, D, h, z


def effective_inplane_props_from_A(A: np.ndarray, h: float) -> Dict[str, float]:
    """
    Propriedades efetivas in-plane aproximadas via A:
      a = inv(A)
      Ex  = 1/(a11*h)
      Ey  = 1/(a22*h)
      Gxy = 1/(a66*h)
      nu_xy = -a12/a11
      nu_yx = -a12/a22
    """
    h = float(h)
    if h <= 0:
        raise ValueError("h <= 0")

    try:
        a = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return dict(Ex=np.nan, Ey=np.nan, Gxy=np.nan, nu_xy=np.nan, nu_yx=np.nan)

    a11 = float(a[0, 0])
    a22 = float(a[1, 1])
    a12 = float(a[0, 1])
    a66 = float(a[2, 2])

    Ex = 1.0 / (a11 * h) if a11 != 0 else np.nan
    Ey = 1.0 / (a22 * h) if a22 != 0 else np.nan
    Gxy = 1.0 / (a66 * h) if a66 != 0 else np.nan
    nu_xy = -a12 / a11 if a11 != 0 else np.nan
    nu_yx = -a12 / a22 if a22 != 0 else np.nan

    return dict(Ex=float(Ex), Ey=float(Ey), Gxy=float(Gxy), nu_xy=float(nu_xy), nu_yx=float(nu_yx))


# -------------------------
# Layup builders
# -------------------------
def build_layup_from_angles(
    angles_deg: Iterable[float],
    t_ply: float,
) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.array(list(angles_deg), dtype=float)
    t_ply = float(t_ply)
    if angles.size == 0:
        raise ValueError("angles vazio.")
    if t_ply <= 0:
        raise ValueError("t_ply <= 0.")
    t = np.full(angles.size, t_ply, dtype=float)
    return angles, t


def build_symmetric_layup(
    half_angles_deg: Iterable[float],
    t_ply: float,
) -> Tuple[np.ndarray, np.ndarray]:
    half = np.array(list(half_angles_deg), dtype=float)
    if half.size == 0:
        raise ValueError("half_angles vazio.")
    full = np.concatenate([half, half[::-1]], axis=0)
    return build_layup_from_angles(full, t_ply)


def build_layup_from_fractions(
    fractions: Dict[Union[int, str], float],
    t_total: float,
    t_ply: float,
    symmetric: bool = True,
    order: Tuple[float, ...] = (0.0, 45.0, -45.0, 90.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    fractions: ex {0:0.5, 45:0.25, -45:0.25, 90:0.0}
    t_total: espessura total do laminado
    t_ply: espessura por ply (discretização)
    symmetric=True -> empilha metade e espelha (B ~ 0)
    """
    t_total = float(t_total)
    t_ply = float(t_ply)

    if t_total <= 0 or t_ply <= 0:
        raise ValueError("t_total e t_ply devem ser > 0.")

    # número de plies (força par se simétrico)
    n = int(np.round(t_total / t_ply))
    n = max(n, 2)
    if symmetric and (n % 2 != 0):
        n += 1

    # normaliza frações
    fr = {float(k): float(v) for k, v in fractions.items()}
    for ang in order:
        fr.setdefault(float(ang), 0.0)

    total_fr = sum(max(v, 0.0) for v in fr.values())
    if total_fr <= 0:
        raise ValueError("fractions inválidas (soma <= 0).")

    fr = {k: max(v, 0.0) / total_fr for k, v in fr.items()}

    if symmetric:
        n_half = n // 2
        target = {ang: int(np.round(fr[ang] * n_half)) for ang in order}

        # ajusta soma para bater n_half
        s = sum(target.values())
        if s != n_half:
            # corrige no ângulo dominante
            dom = max(order, key=lambda a: fr[a])
            target[dom] += (n_half - s)

        # monta half-stack com ordem estável (você pode trocar depois)
        half = []
        # tenta balancear +45/-45 quando ambos existem
        n0 = target.get(0.0, 0)
        n90 = target.get(90.0, 0)
        n45 = target.get(45.0, 0)
        nm45 = target.get(-45.0, 0)

        half += [0.0] * n0

        # intercala pares ±45
        n_pairs = min(n45, nm45)
        for _ in range(n_pairs):
            half += [45.0, -45.0]
        half += [45.0] * (n45 - n_pairs)
        half += [-45.0] * (nm45 - n_pairs)

        half += [90.0] * n90

        # se sobrou/desandou por arredondamento, corrige com 0°
        while len(half) < n_half:
            half.append(0.0)
        half = half[:n_half]

        return build_symmetric_layup(half, t_ply)
    else:
        target = {ang: int(np.round(fr[ang] * n)) for ang in order}
        s = sum(target.values())
        if s != n:
            dom = max(order, key=lambda a: fr[a])
            target[dom] += (n - s)

        stack = []
        for ang in order:
            stack += [float(ang)] * int(target[ang])

        while len(stack) < n:
            stack.append(0.0)
        stack = stack[:n]

        return build_layup_from_angles(stack, t_ply)


# -------------------------
# High-level API
# -------------------------
def laminate_effective_props(
    mat: UDMaterial,
    angles_deg: Iterable[float],
    t_plies: Iterable[float],
) -> Dict[str, float]:
    A, B, D, h, z = laminate_ABD(angles_deg, t_plies, mat)
    props = effective_inplane_props_from_A(A, h)
    props.update(dict(h=float(h)))
    # B/D ficam disponíveis se você quiser usar depois
    props["_A"] = A
    props["_B"] = B
    props["_D"] = D
    props["_z"] = z
    return props


def tube_EG_from_layup_spec(
    layup_spec: Dict,
    mat: UDMaterial,
) -> Dict[str, float]:
    """
    layup_spec pode ser:
      - {"angles_deg":[...], "t_plies":[...]}
      - {"angles_deg":[...], "t_ply":..., "symmetric":False/True}  (todas iguais)
      - {"fractions":{0:...,45:...,-45:...,90:...}, "t_total":..., "t_ply":..., "symmetric":True}
    Retorna dict com Ex, Ey, Gxy, nu_xy, nu_yx, h + matrizes internas.
    """
    if "fractions" in layup_spec:
        angles, t = build_layup_from_fractions(
            fractions=layup_spec["fractions"],
            t_total=layup_spec["t_total"],
            t_ply=layup_spec["t_ply"],
            symmetric=bool(layup_spec.get("symmetric", True)),
            order=tuple(layup_spec.get("order", (0.0, 45.0, -45.0, 90.0))),
        )
        return laminate_effective_props(mat, angles, t)

    if "angles_deg" in layup_spec and "t_plies" in layup_spec:
        return laminate_effective_props(mat, layup_spec["angles_deg"], layup_spec["t_plies"])

    if "angles_deg" in layup_spec and "t_ply" in layup_spec:
        angles = np.array(layup_spec["angles_deg"], dtype=float)
        t_ply = float(layup_spec["t_ply"])
        if bool(layup_spec.get("symmetric", False)):
            # interpreta angles_deg como half-stack se symmetric=True
            angles, t = build_symmetric_layup(angles, t_ply)
        else:
            angles, t = build_layup_from_angles(angles, t_ply)
        return laminate_effective_props(mat, angles, t)

    raise ValueError("layup_spec inválido. Veja tube_EG_from_layup_spec doc.")


def default_carbon_epoxy_ud() -> UDMaterial:
    """
    Default “típico” (ajuste depois com dados reais do seu prepreg/tecido).
    """
    return UDMaterial(
        E1=135e9,
        E2=10e9,
        G12=5e9,
        nu12=0.30,
        rho=1600.0,
        Xt=1500e6,
        Xc=1200e6,
        Yt=40e6,
        Yc=200e6,
        S12=70e6,
    )


if __name__ == "__main__":
    # teste rápido
    mat = default_carbon_epoxy_ud()

    layup = {
        "fractions": {0: 0.50, 45: 0.25, -45: 0.25, 90: 0.0},
        "t_total": 1.0e-3,
        "t_ply": 0.125e-3,
        "symmetric": True,
    }

    props = tube_EG_from_layup_spec(layup, mat)
    print("=== Effective laminate props ===")
    for k in ["Ex", "Ey", "Gxy", "nu_xy", "nu_yx", "h"]:
        print(f"{k:>6s}: {props[k]:.6e}")
