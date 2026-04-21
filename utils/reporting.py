import numpy as np


def print_sections(diam_o, diam_i):
    tube_position = ["Secondary element", "Primary element", "Diagonal element"]
    for desc, diam_ext, diam_int in zip(tube_position, diam_o, diam_i):
        if diam_int > 0.0:
            print(
                f"{desc} = Tube, diameter: {1000 * diam_ext:.2f} mm | thickness: {500 * (diam_ext - diam_int):.2f} mm"
            )
        else:
            print(f"{desc} = Rod, diameter: {1000 * diam_ext:.2f} mm")
    print()


def node_incident_force_stats(num_nodes, elements, axial_forces):
    n = num_nodes
    sum_abs = np.zeros(n)
    max_abs = np.zeros(n)
    for i, (a, b) in enumerate(elements):
        f = abs(axial_forces[i])
        sum_abs[a] += f
        sum_abs[b] += f
        if f > max_abs[a]:
            max_abs[a] = f
        if f > max_abs[b]:
            max_abs[b] = f
    return sum_abs, max_abs


def print_nodes_table(
    nodes,
    elements,
    node_deg,
    axial_forces,
    node_deflection,
    node_reactions,
    top_n=10,
):
    Rnode = node_reactions
    Rmag = np.sqrt((Rnode**2).sum(axis=1))

    sum_abs, max_abs = node_incident_force_stats(nodes.shape[0], elements, axial_forces)

    order = np.argsort(-Rmag)
    print("\n===== NODES (1ª ORDEM) — RESULTANTE DE FORÇA =====")
    print(
        "rank | node | |R| (N) | Rx (N) | Ry (N) | Rz (N) | deg | sum|F| (N) | max|F| (N) | defl (mm)"
    )
    shown = 0
    for idx in order:
        if shown >= top_n:
            break
        i = idx
        r = Rmag[i]
        if r <= 0 and np.isfinite(r):
            continue
        rx_react, ry_react, rz_react = Rnode[i, :]
        print(
            f"{shown + 1:>4d} | {i:>4d} | {r:>7.2f} | {rx_react:>7.2f} | {ry_react:>7.2f} | {rz_react:>7.2f} |"
            f" {node_deg[i]:>3d} | {sum_abs[i]:>10.2f} | {max_abs[i]:>10.2f} | {1000.0 * node_deflection[i]:>8.3f}"
        )
        shown += 1
