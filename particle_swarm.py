import numpy as np


class Particle:
    def __init__(self, dimensions, bounds):
        """
        Initializes a single particle in the swarm.

        Args:
            dimensions (int): The number of dimensions for the search space.
            bounds (list of tuples): A list of (min, max) for each dimension.
        """
        self.position = np.array(
            [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)]
        )
        self.velocity = np.array(
            [
                np.random.uniform(
                    -abs(bounds[i][1] - bounds[i][0]), abs(bounds[i][1] - bounds[i][0])
                )
                for i in range(dimensions)
            ]
        )

        # Personal best position found by this particle
        self.pbest_position = np.copy(self.position)
        # Best objective function value found by this particle
        self.pbest_value = float("inf")

    def update_pbest(self, objective_function):
        """
        Updates the particle's personal best position and value.
        """
        current_value = objective_function(self.position)
        if current_value < self.pbest_value:
            self.pbest_value = current_value
            self.pbest_position = np.copy(self.position)


class PSO:
    def __init__(
        self,
        objective_function,
        dimensions,
        bounds,
        num_particles=30,
        max_iterations=100,
        w=0.5,
        c1=1.5,
        c2=1.5,
    ):
        """
        Initializes the Particle Swarm Optimization algorithm.

        Args:
            objective_function (callable): The function to be minimized.
            dimensions (int): The number of dimensions for the search space.
            bounds (list of tuples): A list of (min, max) for each dimension.
            num_particles (int): The number of particles in the swarm.
            max_iterations (int): The maximum number of iterations for the optimization.
            w (float): Inertia weight. Controls the impact of the previous velocity.
                       A higher 'w' encourages global exploration, while a lower 'w'
                       encourages local exploitation.
            c1 (float): Cognitive coefficient (or personal acceleration coefficient).
                        Controls the influence of the particle's personal best position.
                        Higher 'c1' makes particles more independent and prone to explore.
            c2 (float): Social coefficient (or global acceleration coefficient).
                        Controls the influence of the global best position found by the swarm.
                        Higher 'c2' makes particles more collaborative and prone to exploit.
        """
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.particles = [Particle(dimensions, bounds) for _ in range(num_particles)]

        # Global best position found by the entire swarm
        self.gbest_position = np.zeros(dimensions)
        # Best objective function value found by the entire swarm
        self.gbest_value = float("inf")

        # History for visualization
        self.position_history = []
        self.gbest_history = []

        self._update_gbest()

    def _update_gbest(self):
        """
        Updates the global best position and value across all particles.
        """
        for particle in self.particles:
            particle.update_pbest(self.objective_function)
            if particle.pbest_value < self.gbest_value:
                self.gbest_value = particle.pbest_value
                self.gbest_position = np.copy(particle.pbest_position)

    def optimize(self):
        """
        Runs the PSO optimization process for a specified number of iterations.
        """
        for iteration in range(self.max_iterations):
            current_positions = []
            for i, particle in enumerate(self.particles):
                # Store current position for visualization
                current_positions.append(np.copy(particle.position))

                # 1. Update Velocity
                # Equation: v_new = w * v_old + c1 * r1 * (pbest - current_pos) + c2 * r2 * (gbest - current_pos)
                # r1, r2 are random numbers between 0 and 1, introducing stochasticity.
                r1 = np.random.rand(self.dimensions)
                r2 = np.random.rand(self.dimensions)

                # Cognitive component: particle's tendency to return to its own best position.
                cognitive_component = (
                    self.c1 * r1 * (particle.pbest_position - particle.position)
                )

                # Social component: particle's tendency to move towards the swarm's best position.
                social_component = (
                    self.c2 * r2 * (self.gbest_position - particle.position)
                )

                # Inertia component: maintains a portion of the particle's previous velocity,
                # helping to explore new areas and avoid local minima.
                particle.velocity = (
                    self.w * particle.velocity + cognitive_component + social_component
                )

                # Optional: Velocity clamping to prevent particles from flying out of control.
                # Typically set as a percentage of the search space range.
                max_velocity = (
                    np.array([b[1] for b in self.bounds])
                    - np.array([b[0] for b in self.bounds])
                ) * 0.1
                min_velocity = -max_velocity
                particle.velocity = np.clip(
                    particle.velocity, min_velocity, max_velocity
                )

                # 2. Update Position
                # Equation: pos_new = pos_old + v_new
                particle.position += particle.velocity

                # Optional: Position clamping to keep particles within the defined search bounds.
                for d in range(self.dimensions):
                    particle.position[d] = np.clip(
                        particle.position[d], self.bounds[d][0], self.bounds[d][1]
                    )

            # After all particles have moved, update the global best position
            self._update_gbest()

            # Store history for visualization
            self.position_history.append(np.array(current_positions))
            self.gbest_history.append(np.copy(self.gbest_position))

            # print(f"Iteration {iteration+1}/{self.max_iterations}, Global Best Value: {self.gbest_value:.4f}, Global Best Position: {self.gbest_position}")

        return self.gbest_position, self.gbest_value
