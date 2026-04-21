# Carbon Fiber Truss Generator

This repository contains a Python-based generative design pipeline for evaluating and optimizing carbon fiber composite truss structures. The implementation couples a customized Finite Element Method (FEM) solver with a Particle Swarm Optimization (PSO) ensemble to find optimal geometric configurations that minimize mass while satisfying structural integrity and safety constraints.

---

## Technical Architecture

### Vectorized Particle Swarm Optimization
The optimization layer utilizes a multi-threaded, parallelized Particle Swarm Optimization (PSO) algorithm mapped to evaluate structural constraints across parameter arrays. The search space is explored iteratively:
- **Vectorized Position Updates**: The velocity and position vectors of the particle swarm are handled via batched NumPy arrays to minimize Python's interpreter overhead.
- **Ensemble Evaluation**: To improve computational throughput, objective functions are evaluated concurrently across thread pools via `concurrent.futures`, allowing bulk processing of the design space.

### Finite Element Method and Composite Mechanics
The structural evaluations rely on an internal 3D Frame element formulation, specialized for Carbon Fiber Reinforced Polymers (CFRP):
- **Sparse Matrix Formulation**: The global stiffness matrix assembly utilizes `scipy.sparse` (COO/CSR formats) for scalable memory footprints and fast linear system resolutions via `scipy.sparse.linalg`.
- **Macromechanics Failure Criteria**: The stress tensors within each member are evaluated against the **Tsai-Wu failure criterion** and local Euler buckling margins, factoring in customized orthotropic properties ($E_x, G_{xy}$) derived from local laminated layup specifications.

---

## Implementation Details

### Core Modules
* **core/fem.py**: Implements the structural node assembly, sparse stiffness matrix compilation, application of boundary conditions, and direct linear system solution. It also includes post-processing logic to derive axial stresses, nodal reactions, and local deflections.
* **optimizer/pso.py**: Contains the PSO orchestration logic, enabling adaptive convergence based on configurable cognitive and social coefficients alongside nonlinear inertia weight adjustments.
* **materials/composite_engine.py**: Provides analytical models to derive equivalent elastic properties from unidirectional (UD) tape data and layered stacking sequences (e.g., $0/90/\pm45$).
* **utils/visual.py**: Provides visualization tools to plot convergence metrics, stress distributions, geometric parameters, and overlaid deformed structures resulting from evaluated load cases.

### Execution Flow
The pipeline integration manages the following calculation cycle:
1. **Parameter Translation**: PSO particle vectors are converted into physical attributes such as geometric node spacing algorithms, primary stringer diameters, and secondary diagonal tube thicknesses.
2. **Evaluation Mesh**: A parameter-driven mesh topology is constructed containing 3D node coordinates and connectivity, defining continuous truss layout mappings subject to base constraints and dynamic structural load forces.
3. **Fitness Resolution**: Structural violations are tracked using multi-variate metrics: constraints involving buckling safety factors, Tsai-Wu SF, and joint-shear margins act as feasibility boundaries. If a design configuration meets threshold minimums, it is scored to optimize and minimize the total structure mass and tip deflection penalties.

---

## Performance Considerations

The framework is optimized for bulk numerical throughput on multi-core CPU environments. By migrating from dense generic matrix constructs to a dedicated sparse linear algebra backend, the underlying FEM equation solver minimizes memory footprints and scaling bottlenecks. The decoupled parallel architecture enables isolated structural instances to be evaluated concurrently over all logical processors during the swarm progression, ensuring rapid layout comparisons on consumer hardware.