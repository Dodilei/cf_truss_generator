import numpy as np

SECTION_NUMBER = 5

UNIT_Z = np.array([0.0, 0.0, 1.0], dtype=float)


class SquareProfile:
    vertices = 4

    unit_coords = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=float,
    ) - np.array([0.5, 0.5, 0.0], dtype=float)

    node_conn = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ],
        dtype=int,
    )

    z_conn = np.arange(8).reshape(2, -1).T.astype(int)
    diag_conn = z_conn.copy()
    diag_conn[:, 0] = (diag_conn[:, 0] + 1) % 4


class Geometry:
    def __init__(
        self,
        base_scale,
        taper_ratio,
        zexp,
        total_length,
        profile=SquareProfile,
        zdiv=SECTION_NUMBER,
    ):

        self.base_scale = base_scale
        self.taper_ratio = taper_ratio
        self.zexp = zexp
        self.total_length = total_length
        self.profile = profile

        zdist = (np.arange(zdiv) / (zdiv - 1)) ** float(zexp)

        nodes = []
        elements = []
        element_kind = []

        profile_scale = base_scale * ((1.0 - zdist) + zdist * taper_ratio)

        for i in range(zdiv):
            profile_nodes = (
                profile.unit_coords * profile_scale[i]
                + zdist[i] * UNIT_Z * total_length
            )
            nodes.append(profile_nodes)

            profile_conn = profile.node_conn + i * profile.vertices
            elements.append(profile_conn)
            element_kind.append(0 * np.ones(profile.vertices, dtype=int))

            if i < zdiv - 1:
                z_conn = profile.z_conn + i * profile.vertices
                elements.append(z_conn)
                element_kind.append(1 * np.ones(profile.vertices, dtype=int))

                diag_conn = profile.diag_conn + i * profile.vertices
                elements.append(diag_conn)
                element_kind.append(2 * np.ones(profile.vertices, dtype=int))

        nodes = np.concatenate(nodes, axis=0)
        elements = np.concatenate(elements, axis=0).astype(int)
        element_kind = np.concatenate(element_kind, axis=0).astype(int)

        self.zdiv = zdiv
        self.nodes = nodes
        self.elements = elements
        self.element_kind = element_kind

        deg = np.zeros(self.nodes.shape[0], dtype=int)
        for a, b in self.elements:
            deg[a] += 1
            deg[b] += 1

        self.node_degree = deg
