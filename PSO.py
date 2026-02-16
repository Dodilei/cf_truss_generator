# pso_engine.py  (PSO — separado do framework)
import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, dimensions, lb, ub, vmin, vmax):
        self.position = np.random.uniform(lb, ub, size=dimensions)
        self.velocity = np.random.uniform(vmin, vmax, size=dimensions)
        self.current_value = np.inf
        self.pbest_position = self.position.copy()
        self.pbest_value = np.inf

class PSO:
    def __init__(self, objective_function, dimensions, bounds,
                 num_particles=100, max_iterations=500,
                 w=0.9, w_min=0.4, inertia_scheme="nonlinear",
                 c1=1.4, c2=1.8):

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

        self.particles = [Particle(self.dimensions, self.lb, self.ub, self.vmin, self.vmax)
                          for _ in range(self.num_particles)]

        self.gbest_position = np.zeros(self.dimensions, dtype=float)
        self.gbest_value = np.inf

        self.gbest_value_history = []
        self.mean_value_history = []
        self.diversity_history = []

        _ = self._evaluate_swarm()

    def _current_inertia(self, it):
        if self.inertia_scheme == "constant" or self.max_iterations <= 1:
            return self.w_initial
        t = it / (self.max_iterations - 1)
        if self.inertia_scheme == "linear":
            return self.w_initial - (self.w_initial - self.w_min) * t
        if self.inertia_scheme == "nonlinear":
            tau = 1.0 - t
            return self.w_min + (self.w_initial - self.w_min) * (tau ** 2)
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

    def optimize(self):
        for it in range(self.max_iterations):
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

            self.mean_value_history.append(float(np.mean(values)))
            self.gbest_value_history.append(float(self.gbest_value))

            pos = np.array([pp.position for pp in self.particles], dtype=float)
            centroid = pos.mean(axis=0)
            diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1)))
            self.diversity_history.append(diversity)

        return self.gbest_position, float(self.gbest_value)


# =========================
# PLOTS — SINGLE RUN
# =========================
def plot_single_convergence(pso):
    it = np.arange(1, len(pso.gbest_value_history) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(it, pso.gbest_value_history, label="Melhor global", linewidth=2)
    plt.plot(it, pso.mean_value_history, label="Média do enxame", linewidth=1.5, linestyle="--")
    plt.yscale("log")
    plt.xlabel("Iteração")
    plt.ylabel("Função objetivo")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.title("Convergência do PSO (1 execução)")
    plt.tight_layout()


# =========================
# ARTICLE MODE — RUN + PLOTS
# =========================
def run_article_mode(N_runs, seeds, objective_function, dimensions, bounds, pso_kwargs):
    best_values = np.empty(N_runs, dtype=float)
    best_positions = np.empty((N_runs, dimensions), dtype=float)

    gbest_hist = np.empty((N_runs, pso_kwargs["max_iterations"]), dtype=float)
    div_hist = np.empty((N_runs, pso_kwargs["max_iterations"]), dtype=float)

    for i in range(N_runs):
        np.random.seed(int(seeds[i]))

        pso = PSO(objective_function, dimensions, bounds, **pso_kwargs)
        bp, bv = pso.optimize()

        best_positions[i, :] = bp
        best_values[i] = bv

        gbest_hist[i, :] = np.array(pso.gbest_value_history, dtype=float)
        div_hist[i, :] = np.array(pso.diversity_history, dtype=float)

        print(f"Run {i+1:02d}/{N_runs} | seed={seeds[i]} | best={bv:.6e}")

    return {
        "seeds": np.array(seeds, dtype=int),
        "best_values": best_values,
        "best_positions": best_positions,
        "gbest_history": gbest_hist,
        "diversity_history": div_hist,
        "pso_kwargs": dict(pso_kwargs),
    }

def _nan_quantile(A, q):
    A2 = np.array(A, dtype=float, copy=True)
    A2[~np.isfinite(A2)] = np.nan
    return np.nanquantile(A2, q, axis=0)

def plot_convergence_band(results):
    H = np.array(results["gbest_history"], dtype=float)
    it = np.arange(1, H.shape[1] + 1)

    med = _nan_quantile(H, 0.50)
    p25 = _nan_quantile(H, 0.25)
    p75 = _nan_quantile(H, 0.75)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(it, med, linewidth=2.2, label="Mediana (gbest)")
    ax.fill_between(it, p25, p75, alpha=0.25, label="P25–P75")
    ax.set_yscale("log")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Função objetivo (gbest)")
    ax.set_title("Convergência do PSO (N execuções)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_diversity_band(results):
    H = np.array(results["diversity_history"], dtype=float)
    it = np.arange(1, H.shape[1] + 1)

    med = np.median(H, axis=0)
    p25 = np.quantile(H, 0.25, axis=0)
    p75 = np.quantile(H, 0.75, axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(it, med, linewidth=2.2, label="Mediana (diversidade)")
    ax.fill_between(it, p25, p75, alpha=0.25, label="P25–P75")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Diversidade (distância média ao centróide)")
    ax.set_title("Diversidade do enxame (N execuções)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_parallel_coordinates_best(results, bounds, var_names):
    X = np.asarray(results["best_positions"], dtype=float)  # (N, D)
    y = np.asarray(results["best_values"], dtype=float)

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    span = ub - lb
    span[span == 0] = 1.0
    Xn = (X - lb) / span

    best_idx = int(np.nanargmin(y))
    dims = X.shape[1]
    xax = np.arange(dims)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i in range(Xn.shape[0]):
        ax.plot(xax, Xn[i, :], color="lightgray", alpha=0.45)

    ax.plot(xax, Xn[best_idx, :], color="red", linewidth=2.8,
            label=f"Melhor execução (seed={results['seeds'][best_idx]})")

    ax.set_xticks(xax)
    ax.set_xticklabels(var_names, rotation=45, ha="right")
    ax.set_xlim(0, dims - 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor normalizado")
    ax.set_title("Coordenadas paralelas (best_positions)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_boxplots_best(results, bounds, var_names, zoom_quantiles=(5, 95), pad_frac=0.25):
    X = np.asarray(results["best_positions"], dtype=float)  # (N, D)
    y = np.asarray(results["best_values"], dtype=float)
    best_idx = int(np.nanargmin(y))

    dims = X.shape[1]
    ncols = 3
    nrows = int(np.ceil(dims / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    rng = np.random.default_rng(123)  # jitter estável

    for d in range(dims):
        ax = axes[d]
        data = X[:, d]

        ax.boxplot(data, vert=True, widths=0.4, showfliers=True)

        jitter = rng.uniform(-0.075, 0.075, size=len(data))
        ax.scatter(1 + jitter, data, s=18, alpha=0.60)

        ax.scatter([1], [X[best_idx, d]], s=110, marker="*", edgecolor="k",
                   label="Melhor execução")

        ax.set_title(var_names[d] if d < len(var_names) else f"var_{d}")
        ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)

        if zoom_quantiles is not None:
            ql, qh = zoom_quantiles
            allv = np.concatenate([data, np.array([X[best_idx, d]], dtype=float)])
            allv = allv[np.isfinite(allv)]
            if allv.size == 0:
                ax.set_ylim(bounds[d][0], bounds[d][1])
            else:
                lo = np.percentile(allv, ql)
                hi = np.percentile(allv, qh)
                span = hi - lo
                if not np.isfinite(span) or span == 0:
                    center = np.median(allv)
                    pad = max(abs(center) * 0.05, 1e-6)
                    ax.set_ylim(center - pad, center + pad)
                else:
                    pad = max(span * pad_frac, 1e-6)
                    ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.set_ylim(bounds[d][0], bounds[d][1])

        vmax = np.nanmax(np.abs(data))
        if np.isfinite(vmax) and vmax < 1e-2:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)

    for k in range(dims, len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle("Distribuição dos parâmetros finais (best_positions) – N execuções", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def print_article_summary(results, var_names):
    vals = np.array(results["best_values"], dtype=float, copy=True)
    vals[~np.isfinite(vals)] = np.nan

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

    best_idx = int(np.nanargmin(vals))
    print(f"\nBest seed: {int(results['seeds'][best_idx])}")
    print("Best position:")
    for name, v in zip(var_names, results["best_positions"][best_idx]):
        print(f"  {name:>18s} = {float(v):.6e}")
    print("==================================================================\n")
