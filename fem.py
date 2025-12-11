import numpy as np

num_dims = 3


def build_fem_system(
    nodes, elements, element_kind, diam_o, diam_i, rho, load_mag, z_spacings
):
    num_nodes = nodes.shape[0]
    n_dofs = num_dims * num_nodes

    # Generate dof-index array
    # -- each node in each element has 3 (num_dims) dofs
    dofs = (
        num_dims * elements.reshape(-1, 2, 1) + np.arange(num_dims).reshape(-1, 1, 1).T
    ).reshape(-1, 6)

    element_diam_ext = np.zeros_like(element_kind, dtype=float)
    element_diam_ext[element_kind == 0] = diam_o[0]
    element_diam_ext[element_kind == 1] = diam_o[1]
    element_diam_ext[element_kind == 2] = diam_o[2]

    element_diam_int = np.zeros_like(element_kind, dtype=float)
    element_diam_int[element_kind == 0] = diam_i[0]
    element_diam_int[element_kind == 1] = diam_i[1]
    element_diam_int[element_kind == 2] = diam_i[2]

    element_area = 0.25 * np.pi * (element_diam_ext**2 - element_diam_int**2)

    element_I = (1.0 / 64.0) * np.pi * (element_diam_ext**4 - element_diam_int**4)

    # Define elements in vector notation (node1, node2) -> (dx, dy, dz)
    dxyz = (nodes[elements] * (np.array([1, -1]).reshape(1, 2, 1))).sum(axis=1)

    element_length = np.sqrt(np.sum(dxyz**2, axis=1))
    total_mass = (element_length * element_area * rho).sum()

    # All base nodes fixed
    fixed_nodes = np.array([0, 1, 2, 3])
    constraints = np.array([[True, True, True] for fn in fixed_nodes])

    # Two tip nodes loaded
    load_nodes = np.array(
        [
            [len(z_spacings) * 4 - 1, 1],
            [len(z_spacings) * 4 - 4, 1],
        ]
    ).astype(int)

    load_magnitudes = np.array([load_mag / 2, load_mag / 2])

    return (
        dofs,
        n_dofs,
        element_area,
        element_I,
        element_length,
        dxyz,
        constraints,
        fixed_nodes,
        load_nodes,
        load_magnitudes,
        total_mass,
    )


# Local stiffness matrix pattern
ksign = np.array([[1, -1], [-1, 1]]).repeat(num_dims, axis=0).repeat(num_dims, axis=1)


def build_K_global(
    dofs,
    n_dofs,
    element_area,
    element_length,
    dxyz,
    E,
):
    k_block = (
        (element_area.reshape(-1, 1, 1) * E)
        * (dxyz.reshape(-1, num_dims, 1) * dxyz.reshape(-1, 1, num_dims))
        / (element_length**3).reshape(-1, 1, 1)
    )
    k = np.tile(k_block, (1, 2, 2)) * ksign

    n_el = len(element_area)
    K_global = np.zeros((n_el, n_dofs, n_dofs), dtype=float)

    dofs_xy = (
        np.outer(dofs, np.ones((1, num_dims * 2)))
        .reshape(*dofs.shape, num_dims * 2)
        .astype(int)
    )
    idx_el = (
        np.outer(np.arange(n_el), np.ones((1, num_dims * 2, num_dims * 2)))
        .reshape(-1, num_dims * 2, num_dims * 2)
        .astype(int)
    )
    K_global[idx_el, dofs_xy, dofs_xy.transpose((0, 2, 1))] = k
    K_global = K_global.sum(axis=0)

    return K_global


def build_F_global(load_nodes, load_magnitudes, n_dofs):
    F_global = np.zeros(n_dofs, dtype=float)

    F_global[num_dims * load_nodes[:, 0] + load_nodes[:, 1]] = load_magnitudes

    return F_global


def apply_constraints(K_global, F_global, constraints, fixed_nodes, n_dofs):
    constrained_dofs = (
        num_dims * fixed_nodes.reshape(-1, 1)
        + np.outer(np.arange(num_dims), np.ones(len(fixed_nodes))).T
    )[constraints].astype(int)

    unconstrained_dofs = np.ones(n_dofs)
    unconstrained_dofs[constrained_dofs] = 0
    unconstrained_dofs = np.arange(n_dofs)[unconstrained_dofs.astype(bool)]

    K_reduced = K_global[np.ix_(unconstrained_dofs, unconstrained_dofs)]

    F_reduced = F_global[unconstrained_dofs]

    return K_reduced, F_reduced, unconstrained_dofs


def solve_fem(K_reduced, F_reduced, unconstrained_dofs, n_dofs):
    if np.linalg.det(K_reduced) == 0:
        print(
            "Warning: K_reduced is singular. The system may not have a unique solution or may be ill-conditioned."
        )
        raise ValueError("K_reduced is singular")

    U_reduced = np.linalg.solve(K_reduced, F_reduced)

    U_global = np.zeros(n_dofs, dtype=float)
    U_global[unconstrained_dofs] = U_reduced

    return U_global


def element_strain(U_global, dxyz, element_length, dofs):
    return (
        (
            U_global[dofs].reshape(-1, 2, num_dims) * np.array([1, -1]).reshape(1, 2, 1)
        ).sum(axis=-2)
        * dxyz
    ).sum(axis=-1) / element_length**2


def calculate_SF(
    dofs, U_global, dxyz, element_length, element_area, element_I, E, K_eff, lrt, lrc
):
    strain = element_strain(U_global, dxyz, element_length, dofs)

    axial_forces = E * element_area * strain
    axial_stresses = axial_forces / (element_area)

    buckling_sf = -(np.pi**2 * element_I * E) / (
        axial_forces * (K_eff * element_length) ** 2
    )

    # Find maximum tensile stress
    tension_sf = lrt / axial_stresses

    # Find maximum compressive stress (most negative value)
    compression_sf = -lrc / axial_stresses

    buckling_sf[buckling_sf < 0] = np.inf
    tension_sf[tension_sf < 0] = np.inf
    compression_sf[compression_sf < 0] = np.inf

    # Maximum deflection
    node_deflection = np.abs((U_global.reshape(-1, 3) ** 2).sum(axis=1))

    return tension_sf, compression_sf, buckling_sf, node_deflection
