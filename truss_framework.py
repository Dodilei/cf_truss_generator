# truss_framework.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib

import composite_engine as comp
import joint_analyzer as ja

from PSO import (
    PSO,
    plot_single_convergence,
    run_article_mode,
    plot_convergence_band,
    plot_diversity_band,
    plot_parallel_coordinates_best,
    plot_boxplots_best,
    print_article_summary,
)

# =========================
# COMPOSITE / LAMINATE
# =========================
UD = comp.default_carbon_epoxy_ud()

LAYUP_SPEC = {
    "fractions": {0: 0.50, 45: 0.25, -45: 0.25, 90: 0.0},
    "t_total": 1.0e-3,
    "t_ply": 0.125e-3,
    "symmetric": True,
}

LAM_PROPS = comp.tube_EG_from_layup_spec(LAYUP_SPEC, UD)
E_LAM = float(LAM_PROPS["Ex"])
G_LAM = float(LAM_PROPS["Gxy"])

# =========================
# NUMPY SETTINGS
# =========================
np.seterr(divide="ignore", invalid="ignore")
np.set_printoptions(suppress=False)

# =========================
# RUN CONFIG
# =========================
RUN_PSO = True
ARTICLE_MODE = True
N_RUNS = 5
SEED_SINGLE = 42
SHOW_ARTICLE_PLOTS = True
VISUALIZE_BEST = True

MANUAL_BEST_POSITION = np.array([], dtype=float)  # 9 vars se RUN_PSO=False

# =========================
# CONSTANTES / MATERIAIS
# =========================
rho = 2000.0          # kg/m3 (membros)
K_eff = 0.8           # efetivo flambagem (Euler)
total_length = 0.85   # m
load_mag = 60.0       # N (base)

# joints (1ª ordem)
OVERLAP_RATIO = 1.5
TAU_ALLOW = 30e6
SLEEVE_THICKNESS = 0.5e-3
RHO_SLEEVE = 2700.0
ENDS_PER_MEMBER = 2.0
JOINT_MASS_FACTOR = 1.0

# =========================
# BASE GEOMETRY (square)
# =========================
square = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
    ],
    dtype=float,
) - np.array([0.5, 0.5, 0.0], dtype=float)

square_conn = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
    ],
    dtype=int,
)

z_conn = np.arange(8).reshape(2, -1).T.astype(int)     # [0,4],[1,5],[2,6],[3,7]
diag_conn = z_conn.copy()
diag_conn[:, 0] = (diag_conn[:, 0] + 1) % 4             # [(1,4),(2,5),(3,6),(0,7)]

z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
ksign = np.array([[1, -1], [-1, 1]], dtype=float).repeat(3, axis=0).repeat(3, axis=1)

# =========================
# HELPERS
# =========================
def get_cmap_safe(name="managua", fallback="viridis"):
    try:
        return matplotlib.colormaps.get_cmap(name)
    except Exception:
        return matplotlib.colormaps.get_cmap(fallback)

def set_axes_equal_3d(ax, nodes):
    x = nodes[:, 0]; y = nodes[:, 1]; z = nodes[:, 2]
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    mid_x = 0.5 * (x.max() + x.min())
    mid_y = 0.5 * (y.max() + y.min())
    mid_z = 0.5 * (z.max() + z.min())
    max_range = max(x_range, y_range, z_range, 1e-12)
    ax.set_xlim(mid_x - 0.5 * max_range, mid_x + 0.5 * max_range)
    ax.set_ylim(mid_y - 0.5 * max_range, mid_y + 0.5 * max_range)
    ax.set_zlim(mid_z - 0.5 * max_range, mid_z + 0.5 * max_range)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

def make_fabricable(diam_o, diam_i):
    diam_o = np.asarray(diam_o, dtype=float).copy()
    diam_i = np.asarray(diam_i, dtype=float).copy()

    thickness = np.round(4000.0 * (diam_o - diam_i) / 2.0) / 4000.0
    diam_o = np.ceil(2000.0 * diam_o) / 2000.0
    diam_i = diam_o - 2.0 * thickness

    diam_i[diam_i < 0] = 0.0
    mask_bad = diam_i >= diam_o
    if np.any(mask_bad):
        diam_i[mask_bad] = np.maximum(diam_o[mask_bad] - 1e-6, 0.0)

    diam_i[diam_o <= 0.0015] = 0.0
    return diam_o, diam_i

def print_sections(diam_o, diam_i):
    tube_position = ["Secondary element", "Primary element", "Diagonal element"]
    for desc, do, di in zip(tube_position, diam_o, diam_i):
        if di > 0.0:
            print(f"{desc} = Tube, diameter: {1000*do:.2f} mm | thickness: {500*(do-di):.2f} mm")
        else:
            print(f"{desc} = Rod, diameter: {1000*do:.2f} mm")
    print()

def _get_ud_strengths(ud_mat, fallback_Xt=1.5e9, fallback_Xc=1.2e9):
    Xt = None; Xc = None
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
# LOAD CASES (ENVELOPE)
# =========================
def get_load_case(case_name, zdiv, load_mag_base):
    """
    dof: 0=X, 1=Y, 2=Z
    Top nodes indices: top0..top0+3
    """
    top0 = 4 * (zdiv - 1)
    n0 = top0 + 0
    n1 = top0 + 1
    n2 = top0 + 2
    n3 = top0 + 3

    L = float(load_mag_base)

    if case_name == "pullup_limit":
        Ftot = 6.0 * L
        load_nodes = np.array([[n0, 1], [n3, 1]], dtype=int)
        load_magnitudes = np.array([+0.5 * Ftot, +0.5 * Ftot], dtype=float)
        return load_nodes, load_magnitudes

    if case_name == "pullup_ultimate":
        Ftot = 1.5 * (6.0 * L)
        load_nodes = np.array([[n0, 1], [n3, 1]], dtype=int)
        load_magnitudes = np.array([+0.5 * Ftot, +0.5 * Ftot], dtype=float)
        return load_nodes, load_magnitudes

    if case_name == "lateral_gust":
        Fx = 0.5 * L  # 30N se L=60
        load_nodes = np.array([[n0, 0], [n3, 0]], dtype=int)
        load_magnitudes = np.array([-Fx, -Fx], dtype=float)
        return load_nodes, load_magnitudes

    if case_name == "torsion":
        Fx = 0.5 * L
        load_nodes = np.array([[n0, 0], [n3, 0], [n1, 0], [n2, 0]], dtype=int)
        load_magnitudes = np.array([-Fx, -Fx, +Fx, +Fx], dtype=float)
        return load_nodes, load_magnitudes

    raise ValueError(f"Load case desconhecido: {case_name}")

LOAD_CASES = ["pullup_limit", "pullup_ultimate", "lateral_gust", "torsion"]

# =========================
# GEOMETRY
# =========================
def generate_geometry(base_scale, taper_ratio, zdist, total_length_):
    base_scale = float(base_scale)
    taper_ratio = float(taper_ratio)
    total_length_ = float(total_length_)

    nodes = []
    elements = []
    element_kind = []

    zdiv = len(zdist)
    sfunc = base_scale * ((1.0 - zdist) + zdist * taper_ratio)

    for i in range(zdiv):
        sq0 = square * sfunc[i] + zdist[i] * z_axis * total_length_
        nodes.append(sq0)

        sq_conn0 = square_conn + i * 4
        elements.append(sq_conn0)
        element_kind.append(0 * np.ones(4, dtype=int))

        if i < zdiv - 1:
            z_conn0 = z_conn + i * 4
            diag_conn0 = diag_conn + i * 4
            elements.append(z_conn0)
            element_kind.append(1 * np.ones(4, dtype=int))
            elements.append(diag_conn0)
            element_kind.append(2 * np.ones(4, dtype=int))

    nodes = np.concatenate(nodes, axis=0)
    elements = np.concatenate(elements, axis=0).astype(int)
    element_kind = np.concatenate(element_kind, axis=0).astype(int)
    return nodes, elements, element_kind

# =========================
# FEM
# =========================
def build_system(nodes, elements, element_kind, diam_o, diam_i, rho_member, load_nodes, load_magnitudes):
    num_nodes = nodes.shape[0]
    n_dofs = 3 * num_nodes

    dofs = (3 * elements.reshape(-1, 2, 1) +
            np.array([0, 1, 2]).reshape(1, 1, 3)).reshape(-1, 6)

    element_diam_ext = np.zeros_like(element_kind, dtype=float)
    element_diam_int = np.zeros_like(element_kind, dtype=float)

    element_diam_ext[element_kind == 0] = diam_o[0]
    element_diam_ext[element_kind == 1] = diam_o[1]
    element_diam_ext[element_kind == 2] = diam_o[2]

    element_diam_int[element_kind == 0] = diam_i[0]
    element_diam_int[element_kind == 1] = diam_i[1]
    element_diam_int[element_kind == 2] = diam_i[2]

    element_area = 0.25 * np.pi * (element_diam_ext**2 - element_diam_int**2)
    element_I = (1.0 / 64.0) * np.pi * (element_diam_ext**4 - element_diam_int**4)

    dxyz = (nodes[elements] * np.array([1.0, -1.0]).reshape(1, 2, 1)).sum(axis=1)
    element_length = np.sqrt(np.sum(dxyz**2, axis=1))

    total_mass_members = float((element_length * element_area * float(rho_member)).sum())

    fixed_nodes = np.array([0, 1, 2, 3], dtype=int)
    constraints = np.ones((len(fixed_nodes), 3), dtype=bool)

    return dict(
        dofs=dofs,
        n_dofs=n_dofs,
        element_area=element_area,
        element_I=element_I,
        element_length=element_length,
        dxyz=dxyz,
        constraints=constraints,
        fixed_nodes=fixed_nodes,
        load_nodes=np.asarray(load_nodes, dtype=int),
        load_magnitudes=np.asarray(load_magnitudes, dtype=float),
        total_mass_members=total_mass_members,
    )

def build_K_global(dofs, n_dofs, element_area, element_length, dxyz, E_):
    E_ = float(E_)
    k_block = (element_area.reshape(-1, 1, 1) * E_) * \
              (dxyz.reshape(-1, 3, 1) * dxyz.reshape(-1, 1, 3)) / \
              (element_length.reshape(-1, 1, 1) ** 3)
    k = (np.tile(k_block, (1, 2, 2)) * ksign)

    n_el = len(element_area)
    K_stack = np.zeros((n_el, n_dofs, n_dofs), dtype=float)

    dofs_xy = np.outer(dofs, np.ones((1, 6))).reshape(*dofs.shape, 6).astype(int)
    idx_el = np.outer(np.arange(n_el), np.ones((1, 6, 6))).reshape(-1, 6, 6).astype(int)

    K_stack[idx_el, dofs_xy, dofs_xy.transpose((0, 2, 1))] = k
    return K_stack.sum(axis=0)

def build_F_global(load_nodes, load_magnitudes, n_dofs):
    F_global = np.zeros(int(n_dofs), dtype=float)
    if len(load_nodes):
        F_global[3 * load_nodes[:, 0] + load_nodes[:, 1]] = load_magnitudes
    return F_global

def apply_constraints(K_global, F_global, constraints, fixed_nodes, n_dofs):
    constrained_dofs = (3 * fixed_nodes.reshape(-1, 1) +
                        np.array([0, 1, 2]).reshape(1, 3))[constraints].astype(int)

    unconstrained_mask = np.ones(int(n_dofs), dtype=bool)
    unconstrained_mask[constrained_dofs] = False
    unconstrained = np.arange(int(n_dofs))[unconstrained_mask]

    K_reduced = K_global[np.ix_(unconstrained, unconstrained)]
    F_reduced = F_global[unconstrained]
    return K_reduced, F_reduced, unconstrained

def solve_fem(K_reduced, F_reduced, unconstrained_dofs, n_dofs):
    U_reduced = np.linalg.solve(K_reduced, F_reduced)
    U_global = np.zeros(int(n_dofs), dtype=float)
    U_global[np.asarray(unconstrained_dofs, dtype=int)] = U_reduced
    return U_global

def structural_analysis(system, E_eff):
    K_global = build_K_global(
        system["dofs"],
        system["n_dofs"],
        system["element_area"],
        system["element_length"],
        system["dxyz"],
        float(E_eff),
    )
    F_global = build_F_global(system["load_nodes"], system["load_magnitudes"], system["n_dofs"])
    K_reduced, F_reduced, unconstrained = apply_constraints(
        K_global, F_global, system["constraints"], system["fixed_nodes"], system["n_dofs"]
    )
    U_global = solve_fem(K_reduced, F_reduced, unconstrained, system["n_dofs"])
    return U_global, K_global, F_global

# =========================
# POST: FORCES / STRESS / DEFLECTION
# =========================
def member_quantities(U_global, dofs, dxyz, element_length, element_area, E_eff):
    E_eff = float(E_eff)
    Ue = U_global[dofs].reshape(-1, 2, 3)
    du = (Ue * np.array([1.0, -1.0]).reshape(1, 2, 1)).sum(axis=1)

    # dL_num = (u0-u1)·(x0-x1)
    dL_num = np.sum(du * dxyz, axis=1)

    strain = dL_num / (element_length**2)
    axial_forces = E_eff * element_area * strain
    axial_stresses = axial_forces / element_area

    node_deflection = np.sqrt((U_global.reshape(-1, 3) ** 2).sum(axis=1))
    max_def = float(np.max(node_deflection))

    return dL_num, strain, axial_forces, axial_stresses, node_deflection, max_def

# =========================
# FAIL: TSAI-WU (UNIAXIAL) + BUCKLING
# =========================
def tsai_wu_sf_uniaxial(axial_stresses, ud_mat):
    """
    Tsai-Wu uniaxial (σ1=axial_stress, σ2=τ=0):
      FI = F1*σ + F11*σ² <= 1
    Resolve para k>0 em FI(kσ)=1.
    """
    Xt, Xc = _get_ud_strengths(ud_mat)
    stresses = np.asarray(axial_stresses, dtype=float)
    sfs = np.full_like(stresses, np.inf, dtype=float)

    F1 = 1.0 / Xt - 1.0 / Xc
    F11 = 1.0 / (Xt * Xc)

    mask = np.abs(stresses) > 1e-12
    s = stresses[mask]

    a = F11 * s**2
    b = F1 * s
    c = -1.0

    # quadratic (robusto) + fallback linear
    disc = b**2 - 4.0 * a * c
    disc[disc < 0.0] = 0.0
    sqrt_disc = np.sqrt(disc)

    k1 = np.full_like(s, np.inf, dtype=float)
    k2 = np.full_like(s, np.inf, dtype=float)

    mask_a = np.abs(a) > 1e-18
    k1[mask_a] = (-b[mask_a] + sqrt_disc[mask_a]) / (2.0 * a[mask_a])
    k2[mask_a] = (-b[mask_a] - sqrt_disc[mask_a]) / (2.0 * a[mask_a])

    # linear if a ~ 0: b*k = 1
    mask_lin = ~mask_a
    b_lin = b[mask_lin]
    k_lin = np.full_like(b_lin, np.inf, dtype=float)
    mask_b = np.abs(b_lin) > 1e-18
    k_lin[mask_b] = 1.0 / b_lin[mask_b]
    k1[mask_lin] = k_lin

    k = np.maximum(k1, k2)
    k[(~np.isfinite(k)) | (k <= 0.0)] = np.inf

    sfs[mask] = k
    return sfs

def safety_factors(axial_stresses, element_I, element_area, dL_num, K_eff_, ud_mat):
    """
    Retorna:
      tsai_wu_sf (por barra), min_tsai_wu_sf
      buckling_sf (por barra), min_buckling_sf
    Observação: buckling usando dL_num (numerador) -> forma equivalente ao Pcr/|F|.
    """
    K_eff_ = float(K_eff_)

    tw_sf = tsai_wu_sf_uniaxial(axial_stresses, ud_mat)
    min_tw = float(np.min(tw_sf))

    with np.errstate(divide="ignore", invalid="ignore"):
        SF_B = -(np.pi**2 * element_I) / (K_eff_**2 * element_area * dL_num)
    SF_B[~np.isfinite(SF_B)] = np.inf
    SF_B[SF_B < 0] = np.inf
    buckling_sf = SF_B
    min_b = float(np.min(buckling_sf))

    return tw_sf, buckling_sf, min_tw, min_b

# =========================
# NODES (1ª ORDEM): REACTIONS + LOADS + CONNECTIVITY
# =========================
def node_degree(num_nodes, elements):
    deg = np.zeros(int(num_nodes), dtype=int)
    for (a, b) in elements.astype(int):
        deg[a] += 1
        deg[b] += 1
    return deg

def node_incident_force_stats(num_nodes, elements, axial_forces):
    n = int(num_nodes)
    sum_abs = np.zeros(n, dtype=float)
    max_abs = np.zeros(n, dtype=float)
    for i, (a, b) in enumerate(elements.astype(int)):
        f = float(abs(axial_forces[i]))
        sum_abs[a] += f
        sum_abs[b] += f
        if f > max_abs[a]:
            max_abs[a] = f
        if f > max_abs[b]:
            max_abs[b] = f
    return sum_abs, max_abs

def node_resultants(system, U_global, K_global, F_global):
    n = int(system["n_dofs"] // 3)
    fixed = set([int(i) for i in system["fixed_nodes"]])

    # reactions (only meaningful on constrained dofs)
    R_global = K_global @ U_global - F_global

    # applied loads per node
    Fnode = F_global.reshape(-1, 3)

    # resultant used in table:
    # - fixed node: reaction vector
    # - free node: applied load vector
    Rnode = np.zeros((n, 3), dtype=float)
    for i in range(n):
        if i in fixed:
            Rnode[i, :] = R_global[3*i:3*i+3]
        else:
            Rnode[i, :] = Fnode[i, :]
    return Rnode

def print_nodes_table(case_name, nodes, elements, axial_forces, node_deflection, system, U_global, K_global, F_global, top_n=10):
    Rnode = node_resultants(system, U_global, K_global, F_global)
    Rmag = np.sqrt((Rnode**2).sum(axis=1))

    deg = node_degree(nodes.shape[0], elements)
    sum_abs, max_abs = node_incident_force_stats(nodes.shape[0], elements, axial_forces)

    order = np.argsort(-Rmag)
    print("\n===== NODES (1ª ORDEM) — RESULTANTE DE FORÇA =====")
    print(f"Case: {case_name}")
    print("rank | node | |R| (N) | Rx (N) | Ry (N) | Rz (N) | deg | sum|F| (N) | max|F| (N) | defl (mm)")
    shown = 0
    for idx in order:
        if shown >= int(top_n):
            break
        i = int(idx)
        r = float(Rmag[i])
        if r <= 0 and np.isfinite(r):
            continue
        rx, ry, rz = Rnode[i, :]
        print(
            f"{shown+1:>4d} | {i:>4d} | {r:>7.2f} | {rx:>7.2f} | {ry:>7.2f} | {rz:>7.2f} |"
            f" {int(deg[i]):>3d} | {sum_abs[i]:>10.2f} | {max_abs[i]:>10.2f} | {1000.0*node_deflection[i]:>8.3f}"
        )
        shown += 1

# =========================
# SOLVE DESIGN (ENVELOPE)
# =========================
def solve_one_case(
    base_scale, taper_ratio, zexp, D_ext, dd_vals,
    case_name,
    total_length_=total_length, rho_=rho, load_mag_=load_mag,
    E_eff_=E_LAM, K_eff_=K_eff,
):
    zdist_local = (np.arange(5) / 4.0) ** float(zexp)

    diam_o_local = np.array(D_ext, dtype=float)
    diam_i_local = diam_o_local - np.array(dd_vals, dtype=float)

    if np.any(diam_o_local <= 0) or np.any(diam_i_local < 0) or np.any(diam_i_local >= diam_o_local):
        return None

    diam_o_f, diam_i_f = make_fabricable(diam_o_local, diam_i_local)
    if np.any(diam_o_f <= 0) or np.any(diam_i_f < 0) or np.any(diam_i_f >= diam_o_f):
        return None

    nodes, elements, element_kind = generate_geometry(
        float(base_scale), float(taper_ratio), zdist_local, float(total_length_)
    )

    load_nodes, load_magnitudes = get_load_case(str(case_name), len(zdist_local), float(load_mag_))

    system = build_system(
        nodes, elements, element_kind,
        diam_o_f, diam_i_f,
        float(rho_),
        load_nodes, load_magnitudes,
    )

    U_global, K_global, F_global = structural_analysis(system, float(E_eff_))

    dL_num, strain, axial_forces, axial_stresses, node_deflection, max_def = member_quantities(
        U_global, system["dofs"], system["dxyz"], system["element_length"], system["element_area"], float(E_eff_)
    )

    tw_sf, buckling_sf, min_tw, min_b = safety_factors(
        axial_stresses, system["element_I"], system["element_area"], dL_num, float(K_eff_), UD
    )

    joints = ja.check_bonded_joints(
        axial_forces=axial_forces,
        element_kind=element_kind,
        diam_o_family=diam_o_f,
        overlap_ratio=float(OVERLAP_RATIO),
        tau_allow=float(TAU_ALLOW),
    )

    return dict(
        case_name=str(case_name),
        nodes=nodes,
        elements=elements,
        element_kind=element_kind,
        system=system,
        U_global=U_global,
        K_global=K_global,
        F_global=F_global,
        dL_num=dL_num,
        strain=strain,
        axial_forces=axial_forces,
        axial_stresses=axial_stresses,
        node_deflection=node_deflection,
        max_def=float(max_def),
        tsai_wu_sf=tw_sf,
        buckling_sf=buckling_sf,
        min_tsai_wu_sf=float(min_tw),
        min_buckling_sf=float(min_b),
        joints=joints,
        diam_o=diam_o_f,
        diam_i=diam_i_f,
        zdist=zdist_local,
        total_mass_members=float(system["total_mass_members"]),
    )

def solve_design(
    base_scale, taper_ratio, zexp, D_ext, dd_vals,
    total_length_=total_length, rho_=rho, load_mag_=load_mag,
    E_eff_=E_LAM, K_eff_=K_eff,
):
    per_case = {}
    min_tw_env = np.inf
    min_b_env = np.inf
    min_joint_env = np.inf
    max_def_env = 0.0

    crit_case = None
    crit_metric = np.inf
    crit_def = -np.inf

    common = None

    for cname in LOAD_CASES:
        out = solve_one_case(
            base_scale, taper_ratio, zexp, D_ext, dd_vals,
            cname,
            total_length_=total_length_,
            rho_=rho_,
            load_mag_=load_mag_,
            E_eff_=E_eff_,
            K_eff_=K_eff_,
        )
        if out is None:
            return None

        per_case[cname] = out
        common = out  # geometry/sections are same across cases

        min_tw_env = min(min_tw_env, float(out["min_tsai_wu_sf"]))
        min_b_env = min(min_b_env, float(out["min_buckling_sf"]))
        min_joint_env = min(min_joint_env, float(out["joints"]["sf_min"]))
        max_def_env = max(max_def_env, float(out["max_def"]))

        metric = min(float(out["min_tsai_wu_sf"]), float(out["min_buckling_sf"]), float(out["joints"]["sf_min"]))
        if (metric < crit_metric - 1e-12) or (abs(metric - crit_metric) <= 1e-12 and out["max_def"] > crit_def):
            crit_metric = metric
            crit_def = float(out["max_def"])
            crit_case = str(cname)

    mass_joints = ja.estimate_joint_mass_sleeves(
        element_kind=common["element_kind"],
        diam_o_family=common["diam_o"],
        overlap_ratio=float(OVERLAP_RATIO),
        sleeve_thickness=float(SLEEVE_THICKNESS),
        rho_sleeve=float(RHO_SLEEVE),
        ends_per_member=float(ENDS_PER_MEMBER),
        mass_factor=float(JOINT_MASS_FACTOR),
    )

    total_mass = float(common["total_mass_members"] + mass_joints)

    return dict(
        base_scale=float(base_scale),
        taper_ratio=float(taper_ratio),
        zexp=float(zexp),
        total_length=float(total_length_),
        load_mag=float(load_mag_),
        E_eff=float(E_eff_),
        K_eff=float(K_eff_),
        diam_o=common["diam_o"],
        diam_i=common["diam_i"],
        zdist=common["zdist"],
        per_case=per_case,
        crit_case=str(crit_case),
        min_tsai_wu_sf_env=float(min_tw_env),
        min_buckling_sf_env=float(min_b_env),
        min_joint_sf_env=float(min_joint_env),
        max_def_env=float(max_def_env),
        total_mass_members=float(common["total_mass_members"]),
        mass_joints=float(mass_joints),
        total_mass=float(total_mass),
        nodes=common["nodes"],
        elements=common["elements"],
        element_kind=common["element_kind"],
    )

# =========================
# OBJECTIVE
# =========================
def objective_function(X):
    try:
        p1, p2, p3, p4, p5, p6, p7, p8, p9 = map(float, X)
        out = solve_design(
            base_scale=p1,
            taper_ratio=p2,
            zexp=p3,
            D_ext=[p4, p5, p6],
            dd_vals=[p7, p8, p9],
        )
        if out is None:
            return np.inf

        tol = 1e-9
        feasible = (
            (out["min_tsai_wu_sf_env"] >= 1.0 - tol) and
            (out["min_buckling_sf_env"] >= 1.0 - tol) and
            (out["min_joint_sf_env"] >= 1.0 - tol)
        )
        if not feasible:
            return np.inf

        return float(np.exp(out["max_def_env"]) * out["total_mass"])
    except Exception:
        return np.inf

def translate_params(p1, p2, p3, p4, p5, p6, p7, p8, p9):
    z = (np.arange(5) / 4.0) ** float(p3)
    do = [float(p4), float(p5), float(p6)]
    di = [float(p4-p7), float(p5-p8), float(p6-p9)]
    return (
        float(p1),                              # base_scale
        float(p2),                              # taper_ratio
        z,                                      # z_spacings
        float(total_length),                    # total_length
        do,                                     # diam_o
        di,                                     # diam_i
        float(rho),
        float(load_mag),
        float(E_LAM),
        float(K_eff),
        float(TAU_ALLOW),
        float(OVERLAP_RATIO),
    )

# =========================
# BOUNDS
# =========================
bounds = [
    (0.02, 0.15),     # base_scale
    (0.2, 1.0),       # taper_ratio
    (0.5, 2.0),       # zexp
    (1e-3, 8e-3),     # D1
    (1e-3, 8e-3),     # D2
    (1e-3, 8e-3),     # D3
    (0.5e-3, 8e-3),   # dd1
    (0.5e-3, 8e-3),   # dd2
    (0.5e-3, 8e-3),   # dd3
]
var_names = ["base_scale","taper_ratio","zexp","D1","D2","D3","dd1","dd2","dd3"]
dimensions = len(bounds)

# =========================
# PLOTS
# =========================
def plot_structure_with_constraints_and_loads(nodes, elements, fixed_nodes, load_nodes, load_magnitudes, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="blue", s=50, label="Nodes")

    for element in elements:
        n1, n2 = int(element[0]), int(element[1])
        pA = nodes[n1]; pB = nodes[n2]
        ax.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], "k-", linewidth=2, alpha=0.7)

    for node_idx in fixed_nodes:
        p = nodes[int(node_idx)]
        ax.scatter(p[0], p[1], p[2], color="red", marker="s", s=100, label=f"Fixed Node {int(node_idx)}")

    max_load = float(np.max(np.abs(load_magnitudes))) if len(load_magnitudes) else 0.0
    arrow_scale = 0.05 / max_load if max_load != 0 else 0.1

    for (node_idx, dof_type), magnitude in zip(load_nodes, load_magnitudes):
        node_idx = int(node_idx); dof_type = int(dof_type)
        p = nodes[node_idx]
        u, v, w = 0.0, 0.0, 0.0
        if dof_type == 0: u = float(magnitude) * arrow_scale
        if dof_type == 1: v = float(magnitude) * arrow_scale
        if dof_type == 2: w = float(magnitude) * arrow_scale
        ax.quiver(p[0], p[1], p[2], u, v, w,
                  color="orange", length=1, arrow_length_ratio=0.25,
                  label=f"Load Node {node_idx} ({magnitude:.1f}N dof={dof_type})")

    ax.set_xlabel("X-coordinate (m)")
    ax.set_ylabel("Y-coordinate (m)")
    ax.set_zlabel("Z-coordinate (m)")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(lbl)] for lbl in unique_labels]
    ax.legend(unique_handles, unique_labels)

    set_axes_equal_3d(ax, nodes)
    ax.view_init(elev=20, azim=-50)
    ax.grid(True)
    plt.show()

def plot_stress_and_buckling(nodes, elements, axial_stresses, buckling_sf, title_left, title_right):
    cmap_stress = get_cmap_safe("managua", "viridis")
    cmap_sf = get_cmap_safe("managua", "viridis")

    stresses = np.asarray(axial_stresses, dtype=float)
    smax = float(np.max(np.abs(stresses))) if len(stresses) else 1.0
    vmin, vmax = -smax, +smax
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_stress = (stresses - vmin) / (vmax - vmin)
    norm_stress[~np.isfinite(norm_stress)] = 0.5
    norm_stress = np.clip(norm_stress, 0.0, 1.0)

    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    # (a) stress
    ax1.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="black", s=20)
    for i, element in enumerate(elements):
        n1, n2 = int(element[0]), int(element[1])
        pA = nodes[n1]; pB = nodes[n2]
        ax1.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], color=cmap_stress(float(norm_stress[i])), linewidth=2.5)
    ax1.set_xlabel("X-coordinate (m)")
    ax1.set_ylabel("Y-coordinate (m)")
    ax1.set_zlabel("Z-coordinate (m)")
    ax1.set_title(title_left)
    set_axes_equal_3d(ax1, nodes)
    ax1.view_init(elev=20, azim=-50)
    ax1.grid(True)

    sm1 = cm.ScalarMappable(cmap=cmap_stress)
    sm1.set_array([vmin, vmax])
    cbar1 = fig.colorbar(sm1, ax=ax1, pad=0.1, shrink=0.9, location="bottom")
    cbar1.set_label("Axial Stress (Pa)")

    # (b) buckling
    ax2.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="black", s=20)
    for i, element in enumerate(elements):
        n1, n2 = int(element[0]), int(element[1])
        pA = nodes[n1]; pB = nodes[n2]
        sf = float(buckling_sf[i])

        if sf < 1.0:
            color = "orangered"
        elif sf > 3.0:
            color = "gray"
        else:
            t = (sf - 1.0) / (3.0 - 1.0)
            t = float(np.clip(t, 0.0, 1.0))
            color = cmap_sf(t)

        ax2.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], color=color, linewidth=2.5)

    ax2.set_xlabel("X-coordinate (m)")
    ax2.set_ylabel("Y-coordinate (m)")
    ax2.set_zlabel("Z-coordinate (m)")
    ax2.set_title(title_right)
    set_axes_equal_3d(ax2, nodes)
    ax2.view_init(elev=20, azim=-50)
    ax2.grid(True)

    sm2 = cm.ScalarMappable(cmap=cmap_sf)
    sm2.set_array([1.0, 3.0])
    cbar2 = fig.colorbar(sm2, ax=ax2, pad=0.1, shrink=0.9, location="bottom")
    cbar2.set_label("Buckling Safety Factor (1 <= SF <= 3)")

    custom_lines = [
        Line2D([0], [0], color="orangered", lw=4, label="SF < 1 (Critical)"),
        Line2D([0], [0], color="gray", lw=4, label="SF > 3 (Safe)"),
    ]
    ax2.legend(custom_lines, [cl.get_label() for cl in custom_lines])

    plt.show()

# =========================
# MAIN
# =========================
def main():
    best_position = None
    best_value = None

    if RUN_PSO:
        if ARTICLE_MODE:
            seeds = list(range(int(N_RUNS)))
            pso_kwargs = dict(
                num_particles=100,
                max_iterations=500,
                w=0.9, w_min=0.4, inertia_scheme="nonlinear",
                c1=1.4, c2=1.8
            )

            results = run_article_mode(int(N_RUNS), seeds, objective_function, dimensions, bounds, pso_kwargs)
            print_article_summary(results, var_names)

            vals = np.array(results["best_values"], dtype=float)
            idx = int(np.nanargmin(vals))

            best_position = np.array(results["best_positions"][idx], dtype=float)
            best_value = float(vals[idx])

            if SHOW_ARTICLE_PLOTS:
                plot_convergence_band(results)
                plot_diversity_band(results)
                plot_parallel_coordinates_best(results, bounds, var_names)
                plot_boxplots_best(results, bounds, var_names, zoom_quantiles=(5, 95), pad_frac=0.25)
                plt.show()
        else:
            np.random.seed(int(SEED_SINGLE))
            pso = PSO(
                objective_function, dimensions, bounds,
                num_particles=100,
                max_iterations=500,
                w=0.9, w_min=0.4, inertia_scheme="nonlinear",
                c1=1.4, c2=1.8
            )
            best_position, best_value = pso.optimize()
            best_position = np.array(best_position, dtype=float)
            best_value = float(best_value)

            print("\nOptimization Complete!")
            print(f"Best found position: {best_position}")
            print(f"Best objective function value: {best_value:.6e}")
            plot_single_convergence(pso)
            plt.show()
    else:
        best_position = np.array(MANUAL_BEST_POSITION, dtype=float)
        best_value = np.nan

    if best_position is None or len(best_position) != 9:
        raise ValueError("best_position inválido. Defina RUN_PSO=True ou cole MANUAL_BEST_POSITION com 9 valores.")

    print("\n===== BEST_POSITION (para colar no visual se quiser) =====")
    print(best_position)
    if np.isfinite(best_value):
        print(f"best_value = {best_value:.6e}")
    print("=========================================================\n")

    print(f"Translated parameters: {translate_params(*best_position)}")

    p1, p2, p3, p4, p5, p6, p7, p8, p9 = best_position
    out = solve_design(
        base_scale=p1,
        taper_ratio=p2,
        zexp=p3,
        D_ext=[p4, p5, p6],
        dd_vals=[p7, p8, p9],
    )
    if out is None:
        raise ValueError("Melhor solução retornou out=None.")

    print("\n===== FABRICÁVEL (SEÇÕES) =====")
    print_sections(out["diam_o"], out["diam_i"])

    total_len_sum = float(np.sum(out["per_case"][LOAD_CASES[0]]["system"]["element_length"]))
    print(f"Total length (sum members) {total_len_sum:.2f} m")
    print(f"Mass (members only): {1000*out['total_mass_members']:.1f} g")
    print(f"Mass (joints est.): {1000*out['mass_joints']:.1f} g")
    print(f"Total mass: {1000*out['total_mass']:.1f} g")

    print("\n===== LOAD CASES (ENVELOPE) — RESULTADOS =====")
    for cname in LOAD_CASES:
        c = out["per_case"][cname]
        print(
            f"{cname:>14s} | "
            f"max_def={1000*c['max_def']:.3e} mm | "
            f"minSF_TW={c['min_tsai_wu_sf']:.3e} | "
            f"minSF_b={c['min_buckling_sf']:.3e} | "
            f"SF_joint={c['joints']['sf_min']:.3e}"
        )
    print("=============================================")

    print(f"\nMinimum Tsai-Wu SF (envelope): {out['min_tsai_wu_sf_env']:.3e}")
    print(f"Minimum Buckling SF (envelope): {out['min_buckling_sf_env']:.3e}")
    print(f"Minimum Joint SF (envelope): {out['min_joint_sf_env']:.3e}")
    print(f"Maximum deflection (envelope): {1000*out['max_def_env']:.3e} mm")

    crit = out["crit_case"]
    print(f"\n[VISUAL] Caso crítico: {crit}")

    if VISUALIZE_BEST:
        c = out["per_case"][crit]
        nodes = c["nodes"]
        elements = c["elements"]
        fixed_nodes = c["system"]["fixed_nodes"]
        load_nodes = c["system"]["load_nodes"]
        load_magnitudes = c["system"]["load_magnitudes"]

        plot_structure_with_constraints_and_loads(
            nodes, elements, fixed_nodes, load_nodes, load_magnitudes,
            title="3D Truss Structure with Constraints and Applied Loads"
        )
        plot_stress_and_buckling(
            nodes, elements, c["axial_stresses"], c["buckling_sf"],
            title_left=f"Axial Stress ({crit})",
            title_right=f"Buckling SF ({crit})",
        )

        print("\n===== JOINTS (1ª ORDEM) =====")
        j = c["joints"]
        print(f"SF_min (bond shear): {j['sf_min']:.3e}")
        print(f"tau_max (bond shear): {j['tau_max']:.3e} Pa")
        print(f"tau_allow: {j['tau_allow']:.3e} Pa | overlap_ratio: {j['overlap_ratio']:.2f}")
        print("============================\n")

        print_nodes_table(
            crit,
            nodes,
            elements,
            c["axial_forces"],
            c["node_deflection"],
            c["system"],
            c["U_global"],
            c["K_global"],
            c["F_global"],
            top_n=10,
        )

    # Nova visual: sobreposta deformada (Walbrun-style)
    plot_deformed_truss_overlaid(
    nodes, elements, c["U_global"], c["axial_stresses"], 
    scale_factor=10.0,  # Ajuste se def real for muito pequena
    title=f"Truss Deformation Overlay ({crit})"
)
    
def plot_deformed_truss_overlaid(nodes, elements, U_global, axial_stresses, scale_factor=10.0, title="Truss: Undeformed (Dashed) vs Deformed (Scaled 10x, Colored by Stress)"):
    """
    Visualização 3D sobreposta: estrutura original (cinza tracejado) + deformada (sólida, colorida por axial_stresses).
    Inspirado na Fig 16 de Walbrun et al.: amplifica U_global por scale_factor pra destacar defor. pequenas.
    Args:
        nodes: (N,3) array original.
        elements: (M,2) conectividades.
        U_global: (3N,) deslocamentos do EF.
        axial_stresses: (M,) stresses axiais por element (pra color).
        scale_factor: Multiplicador pra deformada (ex: 10x, como Walbrun).
    """
    # Nodes deformados: U reshape + scale
    U_reshaped = U_global.reshape(-1, 3) * scale_factor
    nodes_def = nodes + U_reshaped

    # Cmap pra stress (simétrico ±max)
    stresses = np.asarray(axial_stresses, dtype=float)
    smax = float(np.max(np.abs(stresses)))
    vmin, vmax = -smax, smax
    norm_stress = np.clip((stresses - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap_stress = get_cmap_safe("viridis", "coolwarm")  # Coolwarm pra comp(azul)-trac(laranja)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Original: cinza tracejado (alpha baixo)
    for elem in elements:
        n1, n2 = int(elem[0]), int(elem[1])
        pA = nodes[n1]; pB = nodes[n2]
        ax.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], 
                color="lightgray", linestyle="--", linewidth=1.5, alpha=0.6, label="Undeformed" if elem[0]==elements[0][0] else "")

    # Nodes originais (pequenos, cinza)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="lightgray", s=20, alpha=0.7)

    # Deformada: sólida, colorida por stress
    for i, elem in enumerate(elements):
        n1, n2 = int(elem[0]), int(elem[1])
        pA_def = nodes_def[n1]; pB_def = nodes_def[n2]
        color = cmap_stress(norm_stress[i])
        ax.plot([pA_def[0], pB_def[0]], [pA_def[1], pB_def[1]], [pA_def[2], pB_def[2]], 
                color=color, linewidth=3, label="Deformed (Scaled)" if i==0 else "")

    # Nodes deformados (coloridos por def mag, s maior)
    def_mag = np.linalg.norm(U_reshaped, axis=1)
    ax.scatter(nodes_def[:, 0], nodes_def[:, 1], nodes_def[:, 2], 
               c=def_mag, s=50, cmap="plasma", alpha=0.8)  # Plasma pra mag def

    # Labels/Title
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"{title}\n(Deformation scaled by {scale_factor}x; max def ~{np.max(def_mag):.1f} m)")
    ax.legend()

    # Equal axes + view (como seus plots)
    set_axes_equal_3d(ax, nodes)
    ax.view_init(elev=20, azim=-50)
    ax.grid(True, alpha=0.3)

    # Colorbar pra stress
    sm = cm.ScalarMappable(cmap=cmap_stress)
    sm.set_array([vmin, vmax])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label("Axial Stress (Pa)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
