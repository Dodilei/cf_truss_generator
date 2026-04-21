import numpy as np
from tqdm import tqdm


class Particle:
    def __init__(self, dimensions, low_bounds, up_bounds, vmin, vmax):
        self.position = np.random.uniform(low_bounds, up_bounds, size=dimensions)
        self.velocity = np.random.uniform(vmin, vmax, size=dimensions)
        self.current_value = np.inf
        self.pbest_position = self.position.copy()
        self.pbest_value = np.inf


class PSO:
    def __init__(
        self,
        objective_function,
        dimensions,
        bounds,
        num_particles=100,
        max_iterations=500,
        w_inertia=0.9,
        w_min=0.4,
        inertia_scheme="nonlinear",
        c1_cogn=1.4,
        c2_soc=1.8,
        initialize=True,
    ):

        self.objective_function = objective_function
        self.dimensions = dimensions
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations

        self.w_initial = w_inertia
        self.w_min = w_min
        self.inertia_scheme = inertia_scheme

        self.c1_cogn = c1_cogn
        self.c2_soc = c2_soc

        self.low_bounds = np.array([b[0] for b in bounds])
        self.up_bounds = np.array([b[1] for b in bounds])

        vmax = (self.up_bounds - self.low_bounds) * 0.1
        vmin = -vmax
        self.vmax = vmax
        self.vmin = vmin

        self.gbest_position = np.zeros(self.dimensions)
        self.gbest_value = np.inf

        self.gbest_value_history = np.zeros(self.max_iterations)
        self.mean_value_history = np.zeros(self.max_iterations)
        self.diversity_history = np.zeros(self.max_iterations)

        if initialize:
            self.initialize()

    def initialize(self, seed=None):
        if seed:
            np.random.seed(seed)
        self.particles = [
            Particle(self.dimensions, self.low_bounds, self.up_bounds, self.vmin, self.vmax)
            for _ in range(self.num_particles)
        ]

        self._evaluate_swarm()

    def _current_inertia(self, it):
        if self.inertia_scheme == "constant" or self.max_iterations <= 1:
            return self.w_initial
        t = it / (self.max_iterations - 1)
        if self.inertia_scheme == "linear":
            return self.w_initial - (self.w_initial - self.w_min) * t
        if self.inertia_scheme == "nonlinear":
            tau = 1.0 - t
            return self.w_min + (self.w_initial - self.w_min) * (tau**2)
        return self.w_initial

    def _evaluate_swarm(self):
        values = np.empty(self.num_particles, dtype=float)
        for i, p in enumerate(self.particles):
            v = self.objective_function(p.position)
            p.current_value = v
            values[i] = v
            if v < p.pbest_value:
                p.pbest_value = v
                p.pbest_position = p.position.copy()
            if v < self.gbest_value:
                self.gbest_value = v
                self.gbest_position = p.position.copy()
        return values

    def optimize(self, verbose=False):
        iterator = range(self.max_iterations)
        if verbose:
            print("\nStarting PSO with parameters:")
            print(f"  Number of particles: {self.num_particles}")
            print(f"  Number of iterations: {self.max_iterations}")
            print(f"  Cognitive coefficient: {self.c1_cogn}")
            print(f"  Social coefficient: {self.c2_soc}")
            print()
            iterator = tqdm(iterator, desc="PSO Progress", unit="it", colour="BLUE")

        for it in iterator:
            w_curr = self._current_inertia(it)
            for p in self.particles:
                r1 = np.random.rand(self.dimensions)
                r2 = np.random.rand(self.dimensions)
                cognitive = self.c1_cogn * r1 * (p.pbest_position - p.position)
                social = self.c2_soc * r2 * (self.gbest_position - p.position)
                p.velocity = w_curr * p.velocity + cognitive + social
                p.velocity = np.clip(p.velocity, self.vmin, self.vmax)
                p.position = np.clip(p.position + p.velocity, self.low_bounds, self.up_bounds)

            values = self._evaluate_swarm()

            self.mean_value_history[it] = np.mean(values)
            self.gbest_value_history[it] = self.gbest_value

            pos = np.array([pp.position for pp in self.particles])
            centroid = pos.mean(axis=0)
            diversity = np.mean(np.linalg.norm(pos - centroid, axis=1))
            self.diversity_history[it] = diversity

            if verbose:
                iterator.set_postfix(
                    {"best": f"{self.gbest_value:.4e}", "div": f"{diversity:.2e}"}
                )

        return self.gbest_position, self.gbest_value


class PSOEnsemble:
    def __init__(
        self,
        objective_function,
        dimensions,
        bounds,
        num_particles=100,
        max_iterations=500,
        w_inertia=0.9,
        w_min=0.4,
        inertia_scheme="nonlinear",
        c1_cogn=1.4,
        c2_soc=1.8,
        n_runs=1,
    ):

        self.pso_solver = PSO(
            objective_function,
            dimensions,
            bounds,
            num_particles,
            max_iterations,
            w_inertia,
            w_min,
            inertia_scheme,
            c1_cogn,
            c2_soc,
            initialize=False,
        )

        self.n_runs = n_runs
        self.dimensions = dimensions

        self.gbest_position = np.zeros(self.dimensions)
        self.gbest_value = np.inf

        self.rbest_position = np.zeros((n_runs, self.dimensions))
        self.rbest_value = np.zeros(n_runs)

        self.rbest_value_history = np.zeros((n_runs, max_iterations))
        self.mean_value_history = np.zeros((n_runs, max_iterations))
        self.diversity_history = np.zeros((n_runs, max_iterations))

    def optimize(self, verbose=False):
        for i in range(self.n_runs):
            self.pso_solver.initialize(seed=i)
            self.pso_solver.optimize(verbose=verbose)

            print(self.pso_solver.gbest_value_history.shape)
            print(self.pso_solver.mean_value_history.shape)
            print(self.pso_solver.diversity_history.shape)

            self.rbest_position[i] = self.pso_solver.gbest_position
            self.rbest_value[i] = self.pso_solver.gbest_value
            self.rbest_value_history[i] = self.pso_solver.gbest_value_history
            self.mean_value_history[i] = self.pso_solver.mean_value_history
            self.diversity_history[i] = self.pso_solver.diversity_history

            if self.pso_solver.gbest_value < self.gbest_value:
                self.gbest_value = self.pso_solver.gbest_value
                self.gbest_position = self.pso_solver.gbest_position

        return self.gbest_position, self.gbest_value
