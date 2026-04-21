import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib

import numpy as np


def show():
    plt.show()


# =========================
# HELPERS
# =========================
def get_cmap_safe(name="managua", fallback="viridis"):
    try:
        return matplotlib.colormaps.get_cmap(name)
    except Exception:
        return matplotlib.colormaps.get_cmap(fallback)


def set_axes_equal_3d(ax, nodes):
    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]
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


def plot_deformed_truss_overlaid(
    nodes,
    elements,
    U_global,
    axial_stresses,
    scale_factor=10.0,
    title="Truss: Undeformed (Dashed) vs Deformed (Scaled 10x, Colored by Stress)",
):
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
    cmap_stress = get_cmap_safe(
        "viridis", "coolwarm"
    )  # Coolwarm pra comp(azul)-trac(laranja)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Original: cinza tracejado (alpha baixo)
    for elem in elements:
        n1, n2 = int(elem[0]), int(elem[1])
        pA = nodes[n1]
        pB = nodes[n2]
        ax.plot(
            [pA[0], pB[0]],
            [pA[1], pB[1]],
            [pA[2], pB[2]],
            color="lightgray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            label="Undeformed" if elem[0] == elements[0][0] else "",
        )

    # Nodes originais (pequenos, cinza)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="lightgray", s=20, alpha=0.7)

    # Deformada: sólida, colorida por stress
    for i, elem in enumerate(elements):
        n1, n2 = int(elem[0]), int(elem[1])
        pA_def = nodes_def[n1]
        pB_def = nodes_def[n2]
        color = cmap_stress(norm_stress[i])
        ax.plot(
            [pA_def[0], pB_def[0]],
            [pA_def[1], pB_def[1]],
            [pA_def[2], pB_def[2]],
            color=color,
            linewidth=3,
            label="Deformed (Scaled)" if i == 0 else "",
        )

    # Nodes deformados (coloridos por def mag, s maior)
    def_mag = np.linalg.norm(U_reshaped, axis=1)
    ax.scatter(
        nodes_def[:, 0],
        nodes_def[:, 1],
        nodes_def[:, 2],
        c=def_mag,
        s=50,
        cmap="plasma",
        alpha=0.8,
    )  # Plasma pra mag def

    # Labels/Title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"{title}\n(Deformation scaled by {scale_factor}x; max def ~{np.max(def_mag):.1f} m)"
    )
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


# =========================
# PLOTS
# =========================
def plot_structure_with_constraints_and_loads(
    nodes, elements, fixed_nodes, load_nodes, load_magnitudes, title
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="blue", s=50, label="Nodes")

    for element in elements:
        n1, n2 = int(element[0]), int(element[1])
        pA = nodes[n1]
        pB = nodes[n2]
        ax.plot(
            [pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], "k-", linewidth=2, alpha=0.7
        )

    for node_idx in fixed_nodes:
        p = nodes[int(node_idx)]
        ax.scatter(
            p[0],
            p[1],
            p[2],
            color="red",
            marker="s",
            s=100,
            label=f"Fixed Node {int(node_idx)}",
        )

    max_load = float(np.max(np.abs(load_magnitudes))) if len(load_magnitudes) else 0.0
    arrow_scale = 0.05 / max_load if max_load != 0 else 0.1

    for (node_idx, dof_type), magnitude in zip(load_nodes, load_magnitudes):
        node_idx = int(node_idx)
        dof_type = int(dof_type)
        p = nodes[node_idx]
        u, v, w = 0.0, 0.0, 0.0
        if dof_type == 0:
            u = float(magnitude) * arrow_scale
        if dof_type == 1:
            v = float(magnitude) * arrow_scale
        if dof_type == 2:
            w = float(magnitude) * arrow_scale
        ax.quiver(
            p[0],
            p[1],
            p[2],
            u,
            v,
            w,
            color="orange",
            length=1,
            arrow_length_ratio=0.25,
            label=f"Load Node {node_idx} ({magnitude:.1f}N dof={dof_type})",
        )

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


def plot_stress_and_buckling(
    nodes, elements, axial_stresses, buckling_sf, title_left, title_right
):
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
        pA = nodes[n1]
        pB = nodes[n2]
        ax1.plot(
            [pA[0], pB[0]],
            [pA[1], pB[1]],
            [pA[2], pB[2]],
            color=cmap_stress(float(norm_stress[i])),
            linewidth=2.5,
        )
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
        pA = nodes[n1]
        pB = nodes[n2]
        sf = float(buckling_sf[i])

        if sf < 1.0:
            color = "orangered"
        elif sf > 3.0:
            color = "gray"
        else:
            t = (sf - 1.0) / (3.0 - 1.0)
            t = float(np.clip(t, 0.0, 1.0))
            color = cmap_sf(t)

        ax2.plot(
            [pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], color=color, linewidth=2.5
        )

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


def _nan_quantile(A, q):
    A2 = np.array(A, dtype=float, copy=True)
    A2[~np.isfinite(A2)] = np.nan
    return np.nanquantile(A2, q, axis=0)


def plot_convergence_band(best_value_history):
    H = np.array(best_value_history, dtype=float)
    it = np.arange(1, H.shape[1] + 1)

    med = _nan_quantile(H, 0.50)
    p25 = _nan_quantile(H, 0.25)
    p75 = _nan_quantile(H, 0.75)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(it, med, linewidth=2.2, label="Mediana (gbest)")
    ax.fill_between(it, p25, p75, alpha=0.25, label="P25–P75")
    ax.set_yscale("log")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Função objetivo (gbest)")
    ax.set_title("Convergência do PSO (N execuções)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_diversity_band(diversity_history):
    H = np.array(diversity_history, dtype=float)
    it = np.arange(1, H.shape[1] + 1)

    med = np.median(H, axis=0)
    p25 = np.quantile(H, 0.25, axis=0)
    p75 = np.quantile(H, 0.75, axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(it, med, linewidth=2.2, label="Mediana (diversidade)")
    ax.fill_between(it, p25, p75, alpha=0.25, label="P25–P75")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Diversidade (distância média ao centróide)")
    ax.set_title("Diversidade do enxame (N execuções)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_parallel_coordinates_best(best_positions, best_values, bounds, var_names):
    X = np.asarray(best_positions, dtype=float)  # (N, D)
    y = np.asarray(best_values, dtype=float)

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    span = ub - lb
    span[span == 0] = 1.0
    Xn = (X - lb) / span

    best_idx = int(np.nanargmin(y))
    dims = X.shape[1]
    xax = np.arange(dims)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i in range(Xn.shape[0]):
        ax.plot(xax, Xn[i, :], color="lightgray", alpha=0.45)

    ax.plot(
        xax,
        Xn[best_idx, :],
        color="red",
        linewidth=2.8,
        label="Melhor execução", # seed is meaningless
    )

    ax.set_xticks(xax)
    ax.set_xticklabels(var_names, rotation=45, ha="right")
    ax.set_xlim(0, dims - 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor normalizado")
    ax.set_title("Coordenadas paralelas (best_positions)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_boxplots_best(
    best_positions, best_values, bounds, var_names, zoom_quantiles=(5, 95), pad_frac=0.25
):
    X = np.asarray(best_positions, dtype=float)  # (N, D)
    y = np.asarray(best_values, dtype=float)
    best_idx = int(np.nanargmin(y))

    dims = X.shape[1]
    ncols = 3
    nrows = int(np.ceil(dims / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    rng = np.random.default_rng(123)  # jitter estável

    for d in range(dims):
        ax = axes[d]
        data = X[:, d]

        ax.boxplot(data, vert=True, widths=0.4, showfliers=True)

        jitter = rng.uniform(-0.075, 0.075, size=len(data))
        ax.scatter(1 + jitter, data, s=18, alpha=0.60)

        ax.scatter(
            [1],
            [X[best_idx, d]],
            s=110,
            marker="*",
            edgecolor="k",
            label="Melhor execução",
        )

        ax.set_title(var_names[d] if d < len(var_names) else f"var_{d}")
        ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)

        if zoom_quantiles is not None:
            ql, qh = zoom_quantiles
            allv = np.concatenate([data, np.array([X[best_idx, d]], dtype=float)])
            allv = allv[np.isfinite(allv)]
            if allv.size == 0:
                ax.set_ylim(bounds[d][0], bounds[d][1])
            else:
                lo = np.percentile(allv, ql)
                hi = np.percentile(allv, qh)
                span = hi - lo
                if not np.isfinite(span) or span == 0:
                    center = np.median(allv)
                    pad = max(abs(center) * 0.05, 1e-6)
                    ax.set_ylim(center - pad, center + pad)
                else:
                    pad = max(span * pad_frac, 1e-6)
                    ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.set_ylim(bounds[d][0], bounds[d][1])

        vmax = np.nanmax(np.abs(data))
        if np.isfinite(vmax) and vmax < 1e-2:
            ax.ticklabel_format(
                axis="y", style="sci", scilimits=(-3, 3), useMathText=True
            )

    for k in range(dims, len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle(
        "Distribuição dos parâmetros finais (best_positions) – N execuções", fontsize=14
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# =========================
# PLOTS — SINGLE RUN
# =========================
def plot_single_convergence(pso):
    it = np.arange(1, len(pso.gbest_value_history) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(it, pso.gbest_value_history, label="Melhor global", linewidth=2)
    plt.plot(
        it,
        pso.mean_value_history,
        label="Média do enxame",
        linewidth=1.5,
        linestyle="--",
    )
    plt.yscale("log")
    plt.xlabel("Iteração")
    plt.ylabel("Função objetivo")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.title("Convergência do PSO (1 execução)")
    plt.tight_layout()
