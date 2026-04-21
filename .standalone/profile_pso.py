import cProfile
import pstats
import os
import sys
import numpy as np

# Add the project root to sys.path to ensure imports work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import objective_function, bounds, dimensions, N_RUNS
from optimizer.pso import PSOEnsemble

def main():
    # Parameters from main.py
    pso_kwargs = dict(
        num_particles=100,
        max_iterations=100,
        w=0.9,
        w_min=0.4,
        inertia_scheme="nonlinear",
        c1=1.4,
        c2=1.8,
    )
    
    n_runs = int(N_RUNS)
    
    print(f"Starting Profile for PSOEnsemble with {n_runs} runs...")
    print(f"Parameters: {pso_kwargs}")
    
    # Initialize Ensemble
    ensemble = PSOEnsemble(
        objective_function,
        dimensions,
        bounds,
        **pso_kwargs,
        n_runs=n_runs
    )
    
    # Setup profiling
    profiler = cProfile.Profile()
    
    print("Optimization started (this may take a while)...")
    profiler.enable()
    
    try:
        best_pos, best_val = ensemble.optimize(verbose=True)
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user.")
    finally:
        profiler.disable()
        print("\nOptimization finished. Analyzing results...")
        
        # Ensure .results directory exists
        results_dir = os.path.join(project_root, ".results")
        os.makedirs(results_dir, exist_ok=True)
        
        stats_file = os.path.join(results_dir, "pso_profile.stats")
        
        # Save stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative').dump_stats(stats_file)
        
        # Print summary
        print("\n" + "="*50)
        print("PROFILING SUMMARY (Top 20 by Cumulative Time)")
        print("="*50)
        stats.sort_stats('cumulative').print_stats(20)
        print("="*50)
        print(f"Full stats saved to: {stats_file}")

if __name__ == "__main__":
    main()
