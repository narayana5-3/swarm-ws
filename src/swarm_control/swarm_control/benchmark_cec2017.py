"""
CEC2017 benchmark comparison: this project's NOA-derived optimizer
(noa_optimizer.py -- the exact same core used by path_planning.py) vs. GWO
and PSO baselines (baseline_optimizers.py), on real CEC2017 test functions
via the `opfunu` library.

This matches the base paper's own validation methodology (Luo et al. 2026,
Section 5.2): before applying the optimizer to the actual application
(path planning), validate it on the standard CEC2017 suite against known
baselines, with multiple independent trials and significance testing.

Requirements:
    pip install opfunu "setuptools<81"

    (opfunu's CEC2017 module imports pkg_resources, which recent setuptools
    versions no longer bundle by default -- pin setuptools<81 or you'll get
    a ModuleNotFoundError. This is a real dependency fragility worth knowing
    about, not a bug in this script; opfunu may fix it in a future release.)

Usage:
    python benchmark_cec2017.py --functions F12017,F32017,F92017 \
        --dim 10 --n_trials 10 --max_iters 150 --out_dir cec_results
"""

import os
import json
import argparse
import numpy as np

from .noa_optimizer import NOAOptimizer, NOAOptimizerConfig
from .baseline_optimizers import GWOOptimizer, PSOOptimizer

DEFAULT_FUNCTIONS = ["F12017", "F32017", "F52017", "F92017", "F132017"]


def get_function(name, ndim):
    import opfunu.cec_based.cec2017 as cec2017
    cls = getattr(cec2017, name)
    return cls(ndim=ndim)


def run_single_trial(optimizer_name, func, pop_size, max_iters, seed):
    rng = np.random.default_rng(seed)
    lo, hi, dim = func.lb, func.ub, func.ndim

    if optimizer_name == "noa":
        config = NOAOptimizerConfig(pop_size=pop_size, max_iters=max_iters)
        opt = NOAOptimizer(func.evaluate, lo, hi, dim, config=config, rng=rng)
    elif optimizer_name == "gwo":
        opt = GWOOptimizer(func.evaluate, lo, hi, dim, pop_size=pop_size,
                            max_iters=max_iters, rng=rng)
    elif optimizer_name == "pso":
        opt = PSOOptimizer(func.evaluate, lo, hi, dim, pop_size=pop_size,
                            max_iters=max_iters, rng=rng)
    else:
        raise ValueError(optimizer_name)

    _, best_f = opt.run(max_iters=max_iters)
    error = best_f - func.f_global  # distance from known theoretical optimum
    return float(best_f), float(error), opt.history


def run_benchmark(function_names, dim=10, pop_size=40, max_iters=150,
                   n_trials=10, out_dir="cec_results", base_seed=0):
    os.makedirs(out_dir, exist_ok=True)
    optimizer_names = ["noa", "gwo", "pso"]
    results = {}

    for fname in function_names:
        print(f"\n{'=' * 70}\n{fname} (dim={dim})\n{'=' * 70}")
        func = get_function(fname, dim)
        results[fname] = {opt_name: {"errors": [], "histories": []} for opt_name in optimizer_names}

        for opt_name in optimizer_names:
            for trial in range(n_trials):
                seed = base_seed + trial
                best_f, error, history = run_single_trial(opt_name, func, pop_size, max_iters, seed)
                results[fname][opt_name]["errors"].append(error)
                results[fname][opt_name]["histories"].append(history)
            mean_err = np.mean(results[fname][opt_name]["errors"])
            std_err = np.std(results[fname][opt_name]["errors"])
            print(f"  {opt_name:5s}  mean_error={mean_err:.6g}  std={std_err:.6g}  "
                  f"({n_trials} trials)")

    _summarize(results, function_names, out_dir, n_trials)
    return results


def _summarize(results, function_names, out_dir, n_trials):
    from scipy.stats import ranksums

    print(f"\n{'=' * 70}\nSUMMARY TABLE (mean error from known global optimum)\n{'=' * 70}")
    header = "function".ljust(12) + "".join(name.ljust(18) for name in ["noa", "gwo", "pso"])
    print(header)

    summary = {}
    for fname in function_names:
        row = fname.ljust(12)
        summary[fname] = {}
        for opt_name in ["noa", "gwo", "pso"]:
            errors = results[fname][opt_name]["errors"]
            mean_err = float(np.mean(errors))
            summary[fname][opt_name] = {"mean_error": mean_err, "std_error": float(np.std(errors)),
                                          "errors": errors}
            row += f"{mean_err:.4g}".ljust(18)
        print(row)

    print(f"\n{'=' * 70}\nSIGNIFICANCE (Wilcoxon rank-sum, noa vs each baseline)\n{'=' * 70}")
    significance = {}
    wins = {"noa": 0, "gwo": 0, "pso": 0, "tie": 0}
    for fname in function_names:
        noa_errors = summary[fname]["noa"]["errors"]
        best_baseline = min(["gwo", "pso"], key=lambda o: summary[fname][o]["mean_error"])
        baseline_errors = summary[fname][best_baseline]["errors"]

        stat, pvalue = ranksums(noa_errors, baseline_errors)
        significant = pvalue < 0.05
        noa_better = summary[fname]["noa"]["mean_error"] < summary[fname][best_baseline]["mean_error"]

        significance[fname] = {"vs": best_baseline, "pvalue": float(pvalue),
                                "significant": bool(significant), "noa_better": bool(noa_better)}
        if significant and noa_better:
            wins["noa"] += 1
        elif significant:
            wins[best_baseline] += 1
        else:
            wins["tie"] += 1

        marker = ("*** noa significantly better" if (significant and noa_better) else
                   "*** baseline significantly better" if significant else "not significant")
        print(f"{fname} (noa vs best-baseline={best_baseline}): p={pvalue:.4g}  [{marker}]")

    print(f"\nWin count: noa={wins['noa']}, baseline_better={wins['gwo']+wins['pso']}, "
          f"not-significant={wins['tie']} (out of {len(function_names)} functions)")

    out_path = os.path.join(out_dir, "cec2017_summary.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "significance": significance,
                    "win_count": wins, "n_trials": n_trials}, f, indent=2)
    print(f"\nFull results saved -> {out_path}")

    _plot_convergence(results, function_names, out_dir)
    _plot_ranking(summary, function_names, out_dir)


def _plot_convergence(results, function_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(function_names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    colors = {"noa": "#d62728", "gwo": "#2ca02c", "pso": "#1f77b4"}
    for idx, fname in enumerate(function_names):
        ax = axes[idx // ncols][idx % ncols]
        for opt_name, color in colors.items():
            histories = results[fname][opt_name]["histories"]
            mean_history = np.mean(histories, axis=0)
            ax.plot(mean_history, label=opt_name, color=color)
        ax.set_yscale("log")
        ax.set_title(fname)
        ax.set_xlabel("iteration")
        ax.set_ylabel("best cost (log scale)")
        ax.legend()
        ax.grid(alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "convergence_curves.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Convergence curves saved -> {plot_path}")
    plt.close(fig)


def _plot_ranking(summary, function_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranks = {"noa": [], "gwo": [], "pso": []}
    for fname in function_names:
        errs = {o: summary[fname][o]["mean_error"] for o in ranks}
        order = sorted(errs, key=errs.get)
        for rank, name in enumerate(order, start=1):
            ranks[name].append(rank)

    avg_ranks = {name: np.mean(r) for name, r in ranks.items()}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(avg_ranks.keys(), avg_ranks.values(), color=["#d62728", "#2ca02c", "#1f77b4"])
    ax.set_ylabel("Average rank (1=best)")
    ax.set_title(f"Average rank across {len(function_names)} CEC2017 functions")
    ax.invert_yaxis()
    plot_path = os.path.join(out_dir, "ranking_chart.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Ranking chart saved -> {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions", type=str, default=",".join(DEFAULT_FUNCTIONS),
                         help="Comma-separated CEC2017 function names (opfunu naming, "
                              "e.g. F12017,F32017). Paper uses all F1-F30; a representative "
                              "subset is faster to run and still meaningful.")
    parser.add_argument("--dim", type=int, default=10,
                         help="Problem dimension. Paper uses 50; lower here for speed.")
    parser.add_argument("--pop_size", type=int, default=40)
    parser.add_argument("--max_iters", type=int, default=150)
    parser.add_argument("--n_trials", type=int, default=10,
                         help="Paper uses 30 independent runs per function.")
    parser.add_argument("--out_dir", type=str, default="cec_results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    function_names = [f.strip() for f in args.functions.split(",")]
    run_benchmark(function_names, dim=args.dim, pop_size=args.pop_size,
                  max_iters=args.max_iters, n_trials=args.n_trials,
                  out_dir=args.out_dir, base_seed=args.seed)
