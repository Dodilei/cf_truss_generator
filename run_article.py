from optimizer.pso import PSOEnsemble
import numpy as np

from visual import (
    plot_convergence_band,
    plot_diversity_band,
    plot_parallel_coordinates_best,
    plot_boxplots_best,
    show,
)


# =========================
# ARTICLE MODE — RUN + PLOTS
# =========================
def run_article_mode(
    n_runs,
    objective_function,
    dimensions,
    bounds,
    pso_kwargs,
    var_names,
    verbose=False,
    plot=False,
):

    pso_ensemble = PSOEnsemble(
        objective_function,
        dimensions,
        bounds,
        **pso_kwargs,
        n_runs=n_runs,
    )

    best_position, best_value = pso_ensemble.optimize(verbose=verbose)

    vals = np.array(pso_ensemble.rbest_value, dtype=float, copy=True)
    vals[~np.isfinite(vals)] = np.nan

    if verbose:
        print_article_summary(vals, pso_ensemble.gbest_position, var_names)

    if plot:
        plot_convergence_band(pso_ensemble.rbest_value_history)
        plot_diversity_band(pso_ensemble.diversity_history)
        plot_parallel_coordinates_best(
            pso_ensemble.rbest_position, pso_ensemble.rbest_value, bounds, var_names
        )
        plot_boxplots_best(
            pso_ensemble.rbest_position,
            pso_ensemble.rbest_value,
            bounds,
            var_names,
            zoom_quantiles=(5, 95),
            pad_frac=0.25,
        )
        show()

    return pso_ensemble.rbest_position, pso_ensemble.rbest_value


def print_article_summary(vals, best_position, var_names):

    best = float(np.nanmin(vals))
    worst = float(np.nanmax(vals))
    mean = float(np.nanmean(vals))
    std = float(np.nanstd(vals))
    med = float(np.nanmedian(vals))

    print("\n===================== SUMMARY (ARTICLE MODE) =====================")
    print(f"N runs: {len(vals)}")
    print(f"Best  : {best:.6e}")
    print(f"Median: {med:.6e}")
    print(f"Mean  : {mean:.6e}")
    print(f"Std   : {std:.6e}")
    print(f"Worst : {worst:.6e}")

    print("Best position:")
    for name, v in zip(var_names, best_position):
        print(f"  {name:>18s} = {float(v):.6e}")
    print("==================================================================\n")
