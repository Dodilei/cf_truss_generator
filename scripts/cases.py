from core.fem import PostProcessor
from core.fem import FEMSystem
from core.truss import TrussSystem

import numpy as np


LOAD_CASES = ["debug"]  # ["pullup_limit", "pullup_ultimate", "lateral_gust", "torsion"]



def create_load_case(case_name, zdiv, load_mag_base):
    """
    dof: 0=X, 1=Y, 2=Z
    Top nodes indices: top0..top0+3
    """
    top0 = 4 * (zdiv - 1)
    n0 = top0 + 0
    n1 = top0 + 1
    n2 = top0 + 2
    n3 = top0 + 3

    L = load_mag_base

    if case_name == "debug":
        Ftot = 1.0 * L
        load_nodes = np.array([[n0, 1], [n3, 1]])
        load_magnitudes = np.array([+0.5 * Ftot, +0.5 * Ftot])

    elif case_name == "pullup_limit":
        Ftot = 6.0 * L
        load_nodes = np.array([[n0, 1], [n3, 1]])
        load_magnitudes = np.array([+0.5 * Ftot, +0.5 * Ftot])

    elif case_name == "pullup_ultimate":
        Ftot = 1.5 * (6.0 * L)
        load_nodes = np.array([[n0, 1], [n3, 1]])
        load_magnitudes = np.array([+0.5 * Ftot, +0.5 * Ftot])

    elif case_name == "lateral_gust":
        Fx = 0.5 * L  # 30N se L=60
        load_nodes = np.array([[n0, 0], [n3, 0]])
        load_magnitudes = np.array([-Fx, -Fx])

    elif case_name == "torsion":
        Fx = 0.5 * L
        load_nodes = np.array([[n0, 0], [n3, 0], [n1, 0], [n2, 0]])
        load_magnitudes = np.array([-Fx, -Fx, +Fx, +Fx])

    else:
        raise ValueError(
            f"Load case desconhecido: {case_name}, selecione algum de: {LOAD_CASES}"
        )

    return load_nodes, load_magnitudes



def solve_case(
    system,
    case_name,
    load_mag,
):
    system.assemble_F_global(*create_load_case(case_name, system.geom.zdiv, load_mag))
    system.apply_constraints()
    system.solve_fem()


def solve_design(
    base_scale,
    taper_ratio,
    zexp,
    diam_ext,
    dd_vals,
    total_length,
    rho,
    E_eff,
    load_mag,
    return_per_case=False,
):
    try:
        truss_system = TrussSystem(
            base_scale,
            taper_ratio,
            zexp,
            diam_ext,
            dd_vals,
            total_length,
            rho,
        )
    except AssertionError:
        return None, None

    fem_system = FEMSystem(truss_system, E_eff)
    postprocessor = PostProcessor(fem_system)
    fem_system.assemble_K_global()

    for cname in LOAD_CASES:
        solve_case(
            fem_system,
            cname,
            load_mag,
        )

        postprocessor.update_metrics(case_name=cname)

    if return_per_case:
        return {
            "critical_metrics": postprocessor.metrics,
            **postprocessor.metrics_per_case,
        }, fem_system

    return postprocessor.metrics, fem_system
