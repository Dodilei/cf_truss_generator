# node_analyzer.py
import numpy as np


def element_unit_vectors(nodes, elements):
    """
    Retorna:
      u: (n_el, 3) vetores unitários do nó n1 -> n2
      L: (n_el,) comprimentos
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)

    p1 = nodes[elements[:, 0]]
    p2 = nodes[elements[:, 1]]
    d = p2 - p1
    L = np.linalg.norm(d, axis=1)

    u = np.zeros_like(d, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = d / L[:, None]

    u[~np.isfinite(u)] = 0.0
    L[~np.isfinite(L)] = 0.0
    return u, L


def nodal_resultant_forces(nodes, elements, axial_forces):
    """
    Converte forças axiais de barras em resultantes por nó (equilíbrio 1ª ordem).

    Convenção:
      - axial_forces[i] > 0 (tração): barra "puxa" os nós um em direção ao outro
      - axial_forces[i] < 0 (compressão): barra "empurra" os nós para fora

    Implementação:
      u = unit(n1->n2)
      f_on_n1 += F * u
      f_on_n2 += -F * u
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    F = np.asarray(axial_forces, dtype=float)

    n_nodes = nodes.shape[0]
    f_node = np.zeros((n_nodes, 3), dtype=float)
    degree = np.zeros(n_nodes, dtype=int)
    sum_abs = np.zeros(n_nodes, dtype=float)
    max_abs = np.zeros(n_nodes, dtype=float)

    u, _ = element_unit_vectors(nodes, elements)

    n1 = elements[:, 0]
    n2 = elements[:, 1]

    f1 = F[:, None] * u
    f2 = -F[:, None] * u

    # acumula
    np.add.at(f_node, n1, f1)
    np.add.at(f_node, n2, f2)

    # métricas auxiliares
    np.add.at(degree, n1, 1)
    np.add.at(degree, n2, 1)

    absF = np.abs(F)
    np.add.at(sum_abs, n1, absF)
    np.add.at(sum_abs, n2, absF)

    np.maximum.at(max_abs, n1, absF)
    np.maximum.at(max_abs, n2, absF)

    res = np.linalg.norm(f_node, axis=1)

    return dict(
        f_node=f_node,          # (n_nodes,3)
        resultant=res,          # (n_nodes,)
        degree=degree,          # (n_nodes,)
        sum_abs_member=sum_abs, # (n_nodes,)
        max_abs_member=max_abs, # (n_nodes,)
    )


def top_nodes_by_metric(metric_array, top_k=10):
    metric_array = np.asarray(metric_array, dtype=float)
    top_k = int(top_k)
    if top_k <= 0:
        return np.array([], dtype=int)

    idx = np.argsort(metric_array)[::-1]
    return idx[:top_k].astype(int)
