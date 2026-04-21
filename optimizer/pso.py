# pso_engine.py  (PSO — separado do framework)
import numpy as np
from tqdm import tqdm


class Particle:
    def __init__(self, dimensions, lb, ub, vmin, vmax):
        self.position = np.random.uniform(lb, ub, size=dimensions)
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
        w=0.9,
        w_min=0.4,
        inertia_scheme="nonlinear",
        c1=1.4,
        c2=1.8,
        initialize=True,
    ):

        self.objective_function = objective_function
        self.dimensions = int(dimensions)
        self.bounds = bounds
        self.num_particles = int(num_particles)
        self.max_iterations = int(max_iterations)

        self.w_initial = float(w)
        self.w_min = float(w_min)
        self.inertia_scheme = str(inertia_scheme)

        self.c1 = float(c1)
        self.c2 = float(c2)

        self.lb = np.array([b[0] for b in bounds], dtype=float)
        self.ub = np.array([b[1] for b in bounds], dtype=float)

        vmax = (self.ub - self.lb) * 0.1
        vmin = -vmax
        self.vmax = vmax
        self.vmin = vmin

        self.gbest_position = np.zeros(self.dimensions, dtype=float)
        self.gbest_value = np.inf

        self.gbest_value_history = np.zeros(self.max_iterations, dtype=float)
        self.mean_value_history = np.zeros(self.max_iterations, dtype=float)
        self.diversity_history = np.zeros(self.max_iterations, dtype=float)

        if initialize:
            self.initialize()

    def initialize(self, seed=None):
        if seed:
            np.random.seed(seed)
        self.particles = [
            Particle(self.dimensions, self.lb, self.ub, self.vmin, self.vmax)
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
            print(f"  Cognitive coefficient: {self.c1}")
            print(f"  Social coefficient: {self.c2}")
            print()
            iterator = tqdm(iterator, desc="PSO Progress", unit="it", colour="BLUE")

        for it in iterator:
            w_curr = self._current_inertia(it)
            for p in self.particles:
                r1 = np.random.rand(self.dimensions)
                r2 = np.random.rand(self.dimensions)
                cognitive = self.c1 * r1 * (p.pbest_position - p.position)
                social = self.c2 * r2 * (self.gbest_position - p.position)
                p.velocity = w_curr * p.velocity + cognitive + social
                p.velocity = np.clip(p.velocity, self.vmin, self.vmax)
                p.position = np.clip(p.position + p.velocity, self.lb, self.ub)

            values = self._evaluate_swarm()

            self.mean_value_history[it] = float(np.mean(values))
            self.gbest_value_history[it] = float(self.gbest_value)

            pos = np.array([pp.position for pp in self.particles], dtype=float)
            centroid = pos.mean(axis=0)
            diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1)))
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
        w=0.9,
        w_min=0.4,
        inertia_scheme="nonlinear",
        c1=1.4,
        c2=1.8,
        n_runs=1,
    ):

        self.pso_solver = PSO(
            objective_function,
            dimensions,
            bounds,
            num_particles,
            max_iterations,
            w,
            w_min,
            inertia_scheme,
            c1,
            c2,
            initialize=False,
        )

        self.n_runs = n_runs
        self.dimensions = dimensions

        self.gbest_position = np.zeros((self.dimensions), dtype=float)
        self.gbest_value = np.inf

        self.rbest_position = np.zeros((n_runs, self.dimensions), dtype=float)
        self.rbest_value = np.zeros(n_runs, dtype=float)

        self.rbest_value_history = np.zeros((n_runs, max_iterations), dtype=float)
        self.mean_value_history = np.zeros((n_runs, max_iterations), dtype=float)
        self.diversity_history = np.zeros((n_runs, max_iterations), dtype=float)

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
