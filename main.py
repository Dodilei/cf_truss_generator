# truss_framework.py
from core.fem import PostProcessor
from scripts.cases import solve_case
from scripts.cases import create_load_case
from scripts.cases import solve_design
from utils.visual import plot_deformed_truss_overlaid
from utils.visual import plot_stress_and_buckling
from utils.visual import plot_structure_with_constraints_and_loads
import numpy as np
import matplotlib.pyplot as plt
import materials.composite_engine as comp

from optimizer.pso import PSO
from utils.visual import (
    plot_single_convergence,
)

from scripts.run_article import (
    run_article_mode,
)

from utils.reporting import (
    print_sections,
    node_incident_force_stats,
    print_nodes_table,
)


LAYUP_SPEC = {
    "fractions": {0: 0.50, 45: 0.25, -45: 0.25, 90: 0.0},
    "t_total": 1.0e-3,
    "t_ply": 0.125e-3,
    "symmetric": True,
}


np.seterr(divide="ignore", invalid="ignore")
np.set_printoptions(suppress=False)


RUN_PSO = True
ARTICLE_MODE = True
N_RUNS = 5
SEED_SINGLE = 42
SHOW_ARTICLE_PLOTS = False
VISUALIZE_CRIT = False

MANUAL_BEST_POSITION = np.array([])  # 9 vars se RUN_PSO=False


RHO = 2000.0  # kg/m3 (membros)
TOTAL_LENGTH = 0.85  # m
LOAD_MAG = 60.0  # N (base)


UD = comp.default_carbon_epoxy_ud()

LAM_PROPS = comp.tube_EG_from_layup_spec(LAYUP_SPEC, UD)
E_LAM = LAM_PROPS["Ex"]
G_LAM = LAM_PROPS["Gxy"]

E_EFF = E_LAM

# Set random seed
np.random.seed(SEED_SINGLE)


bounds = [
    (0.02, 0.15),  # base_scale
    (0.2, 1.0),  # taper_ratio
    (0.5, 2.0),  # zexp
    (1e-3, 8e-3),  # D1
    (1e-3, 8e-3),  # D2
    (1e-3, 8e-3),  # D3
    (0.5e-3, 8e-3),  # dd1
    (0.5e-3, 8e-3),  # dd2
    (0.5e-3, 8e-3),  # dd3
]
var_names = ["base_scale", "taper_ratio", "zexp", "D1", "D2", "D3", "dd1", "dd2", "dd3"]
dimensions = len(bounds)


def objective_function(X):

    p1, p2, p3, p4, p5, p6, p7, p8, p9 = X
    metrics, fem_system = solve_design(
        base_scale=p1,
        taper_ratio=p2,
        zexp=p3,
        diam_ext=[p4, p5, p6],
        dd_vals=[p7, p8, p9],
        total_length=TOTAL_LENGTH,
        rho=RHO,
        E_eff=E_EFF,
        load_mag=LOAD_MAG,
    )

    if metrics is None:
        return np.inf

    sf_tol = 1.05

    feasible = (
        (metrics["min_sf_tsaiwu"] >= sf_tol)
        and (metrics["min_sf_buckling"] >= sf_tol)
        and (metrics["min_sf_joint_shear"] >= sf_tol)
    )
    if not feasible:
        return np.inf

    return np.exp(metrics["max_def"]) * fem_system.total_mass


def translate_params(p1, p2, p3, p4, p5, p6, p7, p8, p9):
    z = (np.arange(5) / 4.0) ** p3
    do = [p4, p5, p6]
    di = [p4 - p7, p5 - p8, p6 - p9]
    return (
        p1,  # base_scale
        p2,  # taper_ratio
        z,  # z_spacings
        do,  # diam_o
        di,  # diam_i
    )


def main():
    best_position = None
    best_value = None

    if RUN_PSO:
        if ARTICLE_MODE:
            pso_kwargs = dict(
                num_particles=10,
                max_iterations=30,
                w_inertia=0.9,
                w_min=0.4,
                inertia_scheme="nonlinear",
                c1_cogn=1.4,
                c2_soc=1.8,
            )

            best_position, best_value = run_article_mode(
                N_RUNS,
                objective_function,
                dimensions,
                bounds,
                pso_kwargs,
                var_names,
                verbose=True,
                plot=SHOW_ARTICLE_PLOTS,
            )

        else:
            pso = PSO(
                objective_function,
                dimensions,
                bounds,
                num_particles=10,
                max_iterations=100,
                w_inertia=0.9,
                w_min=0.4,
                inertia_scheme="nonlinear",
                c1_cogn=1.4,
                c2_soc=1.8,
            )
            best_position, best_value = pso.optimize(verbose=True)

            print("\nOptimization Complete!")
            print(f"Best found position: {best_position}")
            print(f"Best objective function value: {best_value:.6e}")
            plot_single_convergence(pso)
            plt.show()
    else:
        best_position = np.array(MANUAL_BEST_POSITION)
        best_value = np.nan

    if best_position is None or len(best_position) != 9:
        raise ValueError(
            "best_position inválido. Defina RUN_PSO=True ou cole MANUAL_BEST_POSITION com 9 valores."
        )

    print("\n===== BEST_POSITION (para colar no visual se quiser) =====")
    print(best_position)
    if np.isfinite(best_value):
        print(f"best_value = {best_value:.6e}")
    print("=========================================================\n")

    print(f"Translated parameters: {translate_params(*best_position)}")

    p1, p2, p3, p4, p5, p6, p7, p8, p9 = best_position

    metrics, fem_system = solve_design(
        base_scale=p1,
        taper_ratio=p2,
        zexp=p3,
        diam_ext=[p4, p5, p6],
        dd_vals=[p7, p8, p9],
        total_length=TOTAL_LENGTH,
        rho=RHO,
        E_eff=E_EFF,
        load_mag=LOAD_MAG,
        return_per_case=True,
    )
    if metrics is None:
        raise ValueError("Melhor solução retornou None.")

    base_scale, taper_ratio, z_spacing, diam_o, diam_i = translate_params(
        p1, p2, p3, p4, p5, p6, p7, p8, p9
    )

    print("\n===== FABRICÁVEL (SEÇÕES) =====")
    print_sections(diam_o, diam_i)

    total_len_sum = np.sum(fem_system.truss.element_length)
    print(f"Total length (sum members) {total_len_sum:.2f} m")
    print(f"Mass (members only): {1000 * fem_system.truss.total_mass_members:.1f} g")
    print(f"Mass (joints est.): {1000 * fem_system.truss.total_mass_joints:.1f} g")
    print(f"Total mass: {1000 * fem_system.truss.total_mass:.1f} g")

    print("\n===== LOAD CASES (ENVELOPE) — RESULTADOS =====")
    for cname, metrics_case in metrics.items():
        if cname == "critical_metrics":
            continue
        print(
            f"{cname:>14s} | "
            f"max_def={1000 * metrics_case['max_def']:.3e} mm | "
            f"minSF_TW={metrics_case['min_sf_tsaiwu']:.3e} | "
            f"minSF_b={metrics_case['min_sf_buckling']:.3e} | "
            f"SF_joint={metrics_case['min_sf_joint_shear']:.3e}"
        )
    print("=============================================")

    crit_metrics = metrics["critical_metrics"]
    print(f"\nMinimum Tsai-Wu SF (envelope): {crit_metrics['min_sf_tsaiwu']:.3e}")
    print(f"Minimum Buckling SF (envelope): {crit_metrics['min_sf_buckling']:.3e}")
    print(f"Minimum Joint SF (envelope): {crit_metrics['min_sf_joint_shear']:.3e}")
    print(f"Maximum deflection (envelope): {1000 * crit_metrics['max_def']:.3e} mm")

    if VISUALIZE_CRIT:
        print()
        crit_case_name = metrics["critical_metrics"]["critical_case"]
        print(f"\n===== CRITICAL CASE ({crit_case_name}) =====")

        nodes = fem_system.geom.nodes
        elements = fem_system.geom.elements
        fixed_nodes = fem_system.fixed_nodes

        load_nodes, load_magnitudes = create_load_case(
            crit_case_name, fem_system.geom.zdiv, LOAD_MAG
        )

        solve_case(
            fem_system,
            crit_case_name,
            LOAD_MAG,
        )

        postprocessor = PostProcessor(fem_system)
        case_metrics = postprocessor.update_metrics()

        print(
            f"\nMinimum Tsai-Wu SF ({crit_case_name}): {case_metrics['min_sf_tsaiwu']:.3e}"
        )
        print(
            f"Minimum Buckling SF ({crit_case_name}): {case_metrics['min_sf_buckling']:.3e}"
        )
        print(
            f"Minimum Joint SF ({crit_case_name}): {case_metrics['min_sf_joint_shear']:.3e}"
        )
        print(
            f"Maximum deflection ({crit_case_name}): {1000 * case_metrics['max_def']:.3e} mm"
        )

        plot_structure_with_constraints_and_loads(
            nodes,
            elements,
            fixed_nodes,
            load_nodes,
            load_magnitudes,
            title="3D Truss Structure with Constraints and Applied Loads",
        )
        plot_stress_and_buckling(
            nodes,
            elements,
            postprocessor.axial_stresses,
            postprocessor.sf_buckling,
            title_left=f"Axial Stress ({crit_case_name})",
            title_right=f"Buckling SF ({crit_case_name})",
        )

        print_nodes_table(
            nodes,
            elements,
            fem_system.geom.node_degree,
            postprocessor.axial_forces,
            postprocessor.node_deflection,
            postprocessor.node_reaction,
            top_n=10,
        )

        # Nova visual: sobreposta deformada (Walbrun-style)
        plot_deformed_truss_overlaid(
            nodes,
            elements,
            fem_system.U_global,
            postprocessor.axial_stresses,
            scale_factor=10.0,  # Ajuste se def real for muito pequena
            title=f"Truss Deformation Overlay ({crit_case_name})",
        )


if __name__ == "__main__":
    main()
