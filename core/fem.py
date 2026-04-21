from materials.composite_engine import default_carbon_epoxy_ud
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import materials.failure_criteria as fc

ksign = np.array([[1, -1], [-1, 1]]).repeat(3, axis=0).repeat(3, axis=1)


class FEMSystem:
    # TODO this should be an argument
    ud_mat = default_carbon_epoxy_ud

    def __init__(self, truss_system, E_eff):
        self.truss = truss_system
        self.E_eff = E_eff

        self.n_dofs = 3 * self.truss.num_nodes

        self.dofs = (
            3 * self.geom.elements.reshape(-1, 2, 1)
            + np.array([0, 1, 2]).reshape(1, 1, 3)
        ).reshape(-1, 6)

        self.assemble_K_global()

        self.F_global = None
        self.K_reduced = None
        self.F_reduced = None
        self.unconstrained = None
        self.U_global = None

    def __getattr__(self, attr):
        return getattr(self.truss, attr)

    # PREPROCESSING

    def assemble_K_global(self):

        element_area = self.truss.element_area
        element_length = self.truss.element_length
        dxyz = self.truss.dxyz
        n_dofs = self.n_dofs
        dofs = self.dofs

        k_block = (
            (element_area.reshape(-1, 1, 1) * self.E_eff)
            * (dxyz.reshape(-1, 3, 1) * dxyz.reshape(-1, 1, 3))
            / (element_length.reshape(-1, 1, 1) ** 3)
        )
        k = np.tile(k_block, (1, 2, 2)) * ksign

        # Vectorized generation of row and column indices for sparse assembly
        dofs_indices = np.outer(dofs, np.ones(6)).reshape(-1, 6, 6).astype(int)
        rows = dofs_indices.flatten()
        cols = dofs_indices.transpose((0, 2, 1)).flatten()
        data = k.flatten()

        self.K_global = coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()

    def assemble_F_global(self, load_nodes, load_magnitudes):
        F_global = np.zeros(self.n_dofs)
        if len(load_nodes):
            F_global[3 * load_nodes[:, 0] + load_nodes[:, 1]] = load_magnitudes

        self.F_global = F_global

    def apply_constraints(self):
        if self.F_global is None:
            raise ValueError(
                "F_global não foi definido. Chame assemble_F_global primeiro."
            )

        n_dofs = self.n_dofs
        fixed_nodes = self.fixed_nodes
        constraints = self.truss.constraints

        constrained_dofs = (
            3 * fixed_nodes.reshape(-1, 1) + np.array([0, 1, 2]).reshape(1, 3)
        )[constraints].astype(int)

        unconstrained_mask = np.ones(n_dofs, dtype=bool)
        unconstrained_mask[constrained_dofs] = False
        unconstrained = np.arange(n_dofs)[unconstrained_mask]

        self.K_reduced = self.K_global[unconstrained, :][:, unconstrained]
        self.F_reduced = self.F_global[unconstrained]
        self.unconstrained = unconstrained

    # SOLVER

    def solve_fem(self):
        if (
            self.K_reduced is None
            or self.F_reduced is None
            or self.unconstrained is None
        ):
            raise ValueError(
                "Sistema não foi corretamente inicializado. Garanta que K_red, F_red e unconstrained foram definidos."
            )

        U_reduced = spsolve(self.K_reduced, self.F_reduced)
        U_global = np.zeros(self.n_dofs)
        U_global[self.unconstrained] = U_reduced

        self.U_global = U_global
        return U_global


class PostProcessor:
    def __init__(self, fem_system):
        self.fem = fem_system

        self.safety_factors = {}
        self.metrics_per_case = {}
        self.metrics = {}

    def __getattr__(self, attr):
        return getattr(self.fem, attr)

    def update_metrics(self, case_name=None):
        self.postprocess_all()
        self.calculate_sf_all()

        sf_mins = {}
        for key, value in self.safety_factors.items():
            sf_mins["min_" + key] = np.min(value)

        max_def = np.max(self.node_deflection)

        current_metrics = {
            **sf_mins,
            "max_def": max_def,
        }

        critical_sf_value = np.inf
        critical_sf_name = None

        for key, value in current_metrics.items():
            if key == "max_def":
                continue
            if value < critical_sf_value:
                critical_sf_value = value
                critical_sf_name = key

        current_metrics["critical_metric"] = critical_sf_name

        if case_name:
            self.metrics_per_case[case_name] = current_metrics

            for key, value in current_metrics.items():
                if key not in self.metrics:
                    self.metrics[key] = value

                if key == "max_def":
                    self.metrics[key] = max(self.metrics[key], value)
                else:
                    self.metrics[key] = min(self.metrics[key], value)

            if crit_case := self.metrics.get("critical_case") is None:
                self.metrics["critical_case"] = case_name
            else:
                if (
                    self.metrics_per_case[crit_case]["critical_metric"]
                    > critical_sf_value
                ):
                    self.metrics["critical_case"] = case_name

        else:
            self.metrics = current_metrics

        return current_metrics

    def postprocess_all(self):
        self.postprocess_elements()
        self.postprocess_nodes()
        self.postprocess_joints()

    def calculate_sf_all(self):
        self.calculate_sf_elements()
        self.calculate_sf_joints()

    def postprocess_elements(self):
        dxyz = self.truss.dxyz
        element_length = self.truss.element_length
        element_area = self.truss.element_area

        Ue = self.U_global[self.dofs].reshape(-1, 2, 3)
        du_diff = (Ue * np.array([1.0, -1.0]).reshape(1, 2, 1)).sum(axis=1)

        strain = np.sum(du_diff * dxyz, axis=1) / (element_length**2)
        axial_stresses = self.E_eff * strain
        axial_forces = element_area * axial_stresses

        self.strain = strain
        self.axial_forces = axial_forces
        self.axial_stresses = axial_stresses

        return strain, axial_forces, axial_stresses

    def postprocess_nodes(self):
        node_deflection = np.sqrt((self.U_global.reshape(-1, 3) ** 2).sum(axis=1))

        self.node_deflection = node_deflection

        fixed = self.fixed_nodes

        R_global = self.K_global @ self.U_global - self.F_global

        F_node = self.F_global.reshape(-1, 3)

        R_node = np.copy(F_node)
        R_node[fixed] = R_global.reshape(-1, 3)[fixed]

        self.node_deflection = node_deflection
        self.node_reaction = R_node

        return node_deflection, R_node

    def postprocess_joints(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            shear_stress = np.abs(self.axial_forces) / self.bond_area

        self.joint_shear_stress = shear_stress

        return shear_stress

    # TODO modularize ALL sfs, names only on other module
    def calculate_sf_elements(self):
        sf_tsaiwu = fc.sf_tsaiwu_uniaxial(self.axial_stresses, self.ud_mat)
        sf_buck = fc.sf_buckling_column(self.truss, self.axial_forces, self.E_eff)

        self.sf_tsaiwu = sf_tsaiwu
        self.sf_buckling = sf_buck

        sf_dict = {
            "sf_tsaiwu": sf_tsaiwu,
            "sf_buckling": sf_buck,
        }

        self.safety_factors.update(sf_dict)

        return sf_dict

    def calculate_sf_joints(self):

        sf_joint = self.TAU_ALLOW / self.joint_shear_stress

        sf_joint[~np.isfinite(sf_joint)] = np.inf

        self.sf_joint_shear = sf_joint

        sf_dict = {"sf_joint_shear": sf_joint}

        self.safety_factors.update(sf_dict)

        return sf_dict
