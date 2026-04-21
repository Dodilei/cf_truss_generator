import numpy as np
from tqdm import tqdm
import concurrent.futures


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
        n_workers=None,
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
        self.n_workers = n_workers

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
        if seed is not None:
            np.random.seed(seed)

        self.positions = np.random.uniform(
            self.low_bounds, self.up_bounds, size=(self.num_particles, self.dimensions)
        )
        self.velocities = np.random.uniform(
            self.vmin, self.vmax, size=(self.num_particles, self.dimensions)
        )
        self.pbest_positions = self.positions.copy()
        self.pbest_values = np.full(self.num_particles, np.inf)
        self.current_values = np.full(self.num_particles, np.inf)

        # One-off evaluation for initialization without an external executor context
        if self.n_workers is not None and self.n_workers > 1 or self.n_workers is None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                self._evaluate_swarm(executor)
        else:
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

    def _evaluate_swarm(self, executor=None):
        if executor is not None:
            results = list(executor.map(self.objective_function, self.positions))
        else:
            results = [self.objective_function(pos) for pos in self.positions]

        for i, v in enumerate(results):
            self.current_values[i] = v

            if v < self.pbest_values[i]:
                self.pbest_values[i] = v
                self.pbest_positions[i] = self.positions[i].copy()

            if v < self.gbest_value:
                self.gbest_value = v
                self.gbest_position = self.positions[i].copy()

        return self.current_values

    def optimize(self, verbose=False):
        iterator = range(self.max_iterations)
        if verbose:
            print("\nStarting PSO with parameters:")
            print(f"  Number of particles: {self.num_particles}")
            print(f"  Number of iterations: {self.max_iterations}")
            print(f"  Cognitive coefficient: {self.c1_cogn}")
            print(f"  Social coefficient: {self.c2_soc}")
            print(f"  Workers (threads): {self.n_workers if self.n_workers else 'Auto'}")
            print()
            iterator = tqdm(iterator, desc="PSO Progress", unit="it", colour="BLUE")

        if self.n_workers is not None and self.n_workers <= 1:
            # Run sequentially
            for it in iterator:
                self._optimize_step(it, iterator, verbose)
        else:
            # Run with thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                for it in iterator:
                    self._optimize_step(it, iterator, verbose, executor)
            
        return self.gbest_position, self.gbest_value

    def _optimize_step(self, it, iterator, verbose, executor=None):
        w_curr = self._current_inertia(it)

        r1 = np.random.rand(self.num_particles, self.dimensions)
        r2 = np.random.rand(self.num_particles, self.dimensions)

        cognitive = self.c1_cogn * r1 * (self.pbest_positions - self.positions)
        social = self.c2_soc * r2 * (self.gbest_position - self.positions)

        self.velocities = w_curr * self.velocities + cognitive + social
        self.velocities = np.clip(self.velocities, self.vmin, self.vmax)
        self.positions = np.clip(
            self.positions + self.velocities, self.low_bounds, self.up_bounds
        )

        values = self._evaluate_swarm(executor)

        self.mean_value_history[it] = np.mean(values)
        self.gbest_value_history[it] = self.gbest_value

        centroid = self.positions.mean(axis=0)
        diversity = np.mean(np.linalg.norm(self.positions - centroid, axis=1))
        self.diversity_history[it] = diversity

        if verbose:
            iterator.set_postfix(
                {"best": f"{self.gbest_value:.4e}", "div": f"{diversity:.2e}"}
            )


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
        n_workers=None,
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
            n_workers=n_workers,
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

            self.rbest_position[i] = self.pso_solver.gbest_position
            self.rbest_value[i] = self.pso_solver.gbest_value
            self.rbest_value_history[i] = self.pso_solver.gbest_value_history
            self.mean_value_history[i] = self.pso_solver.mean_value_history
            self.diversity_history[i] = self.pso_solver.diversity_history

            if self.pso_solver.gbest_value < self.gbest_value:
                self.gbest_value = self.pso_solver.gbest_value
                self.gbest_position = self.pso_solver.gbest_position

        return self.gbest_position, self.gbest_value
