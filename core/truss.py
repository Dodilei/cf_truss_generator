import numpy as np
from core.geometry import Geometry


# TODO FIX internal diameter is not monotonical
def make_fabricable(diam_o_init, diam_i_init):

    # snap to 0.5mm steps
    thickness = np.round(2000.0 * (diam_o_init - diam_i_init)) / 2000.0 / 2.0
    diam_o = np.ceil(2000.0 * diam_o_init) / 2000.0

    assert np.all(thickness > 0.0)

    diam_i = diam_o - 2.0 * thickness
    diam_i[diam_i < 0] = 0.0

    return diam_o, diam_i


def parse_diams(diam_ext, widths):
    diam_o_init = np.array(diam_ext)
    diam_i_init = diam_o_init - np.array(widths)

    diam_i_init[diam_i_init < 0] = 0.0

    assert np.all(diam_o_init > 0.0)
    assert np.all(diam_o_init > diam_i_init)

    diam_o_f, diam_i_f = make_fabricable(diam_o_init, diam_i_init)

    return diam_o_f, diam_i_f


class TrussSystem:
    # JOINT PROPERTIES
    OVERLAP_RATIO = 1.5
    SLEEVE_THICKNESS = 0.5e-3
    RHO_SLEEVE = 2700.0
    JOINT_MASS_FACTOR = 1.0

    # TODO check this value
    TAU_ALLOW = 100.0e6

    # TODO this is not constant
    ENDS_PER_MEMBER = 2.0

    def __init__(
        self,
        base_scale,
        taper_ratio,
        zexp,
        diam_ext,
        dd_vals,
        total_length,
        rho,
    ):

        self.rho = rho

        diam_o_f, diam_i_f = parse_diams(diam_ext, dd_vals)

        self.geom = Geometry(base_scale, taper_ratio, zexp, total_length)

        self.num_nodes = self.geom.nodes.shape[0]

        self.fixed_nodes = np.array([0, 1, 2, 3], dtype=int)
        self.constraints = np.ones((len(self.fixed_nodes), 3), dtype=bool)

        self.set_element_diameter(diam_o_f, diam_i_f)
        self.calc_element_properties()
        self.calc_joint_properties()

        self.total_mass = self.total_mass_members + self.total_mass_joints

    def set_element_diameter(
        self,
        diam_o,
        diam_i,
    ):
        element_kind = self.geom.element_kind

        element_diam_ext = np.zeros_like(element_kind, dtype=float)
        element_diam_int = np.zeros_like(element_kind, dtype=float)

        element_diam_ext[element_kind == 0] = diam_o[0]
        element_diam_ext[element_kind == 1] = diam_o[1]
        element_diam_ext[element_kind == 2] = diam_o[2]

        element_diam_int[element_kind == 0] = diam_i[0]
        element_diam_int[element_kind == 1] = diam_i[1]
        element_diam_int[element_kind == 2] = diam_i[2]

        self.element_diam_ext = element_diam_ext
        self.element_diam_int = element_diam_int

    def calc_element_properties(self):
        self.element_area = (
            0.25 * np.pi * (self.element_diam_ext**2 - self.element_diam_int**2)
        )
        self.element_I = (
            (1.0 / 64.0) * np.pi * (self.element_diam_ext**4 - self.element_diam_int**4)
        )

        self.dxyz = (
            self.geom.nodes[self.geom.elements] * np.array([1.0, -1.0]).reshape(1, 2, 1)
        ).sum(axis=1)
        self.element_length = np.sqrt(np.sum(self.dxyz**2, axis=1))

        self.total_mass_members = (self.element_length * self.element_area * self.rho).sum()

    def calc_joint_properties(self):
        element_diam_ext = self.element_diam_ext

        bond_length = self.OVERLAP_RATIO * element_diam_ext
        inner_diam = element_diam_ext
        outer_diam = element_diam_ext + 2.0 * self.SLEEVE_THICKNESS

        cross_section_area = 0.25 * np.pi * (outer_diam**2 - inner_diam**2)
        volume_sleeve = cross_section_area * bond_length

        mass_sleeve = (
            self.JOINT_MASS_FACTOR
            * self.ENDS_PER_MEMBER
            * self.RHO_SLEEVE
            * volume_sleeve
        )
        mass_sleeve[~np.isfinite(mass_sleeve)] = 0.0

        self.bond_length = bond_length
        self.bond_area = np.pi * element_diam_ext * bond_length

        self.total_mass_joints = mass_sleeve.sum()
