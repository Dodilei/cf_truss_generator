import numpy as np

# Define coordinates for points in unit square (plane XY)
square_nodes0 = np.array(
    [
        [0, 0, 0],  # lower left
        [0, 1, 0],  # upper left
        [1, 1, 0],  # upper right
        [1, 0, 0],  # lower right
    ]
) - np.array([0.5, 0.5, 0.0])

# Define node connections (elements) for square
square_elements0 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

# Define straigth (z-axis) node connections between two z-spaced squares
z_elements0 = np.arange(8).reshape(2, -1).T
# Define diagnonal node connections between two z-spaced squares
diag_elements0 = (z_elements0 + np.array([1, 0])) % np.array([4, np.inf])

# Z-axis unit vector
z_hat = np.array([0, 0, 1])


def square_truss_geometry(base_scale, taper_ratio, z_spacings, total_length):
    nodes = []
    elements = []
    element_kind = []

    z_sections = len(z_spacings)
    sq_scales = (1 - z_spacings) * base_scale + z_spacings * base_scale * taper_ratio

    for i in range(z_sections):
        square_nodes = (
            square_nodes0 * sq_scales[i] + z_spacings[i] * z_hat * total_length
        )
        nodes.append(square_nodes)

        square_elements = square_elements0 + i * 4
        elements.append(square_elements)
        element_kind.append(0 * np.ones(4))

        if i < z_sections - 1:
            z_elements = z_elements0 + i * 4
            diag_elements = diag_elements0 + i * 4
            elements.append(z_elements)
            element_kind.append(1 * np.ones(4))
            elements.append(diag_elements)
            element_kind.append(2 * np.ones(4))

    nodes = np.concatenate(nodes)
    elements = np.concatenate(elements).astype(int)
    element_kind = np.concatenate(element_kind).astype(int)

    return nodes, elements, element_kind
