"""
Statistical comparison: our CBBA-based risk-prioritized allocator
(cbba_task_allocation.py, reused unchanged -- not reimplemented here) vs.
two standard task-allocation baselines from the literature, run on REAL
captured risk grids from this project's own verified runs (demo_dryrun/,
verify_run/, live_run/), not synthetic data.

Baselines, both capacity-capped at MAX_BUNDLE_SIZE per agent (the same cap
CBBA itself uses) so the comparison is fair -- no baseline gets an
advantage from being allowed to hoard more tasks:
  - greedy_allocate: repeatedly assigns the globally closest (unclaimed
    task, agent-with-spare-capacity) pair. A common, simple baseline in
    task-allocation literature -- locally optimal per single assignment,
    no lookahead, no explicit risk-value weighting in the choice itself.
  - round_robin_allocate: sorts tasks by risk value descending, assigns in
    rotating agent order. Ignores agent position/travel cost entirely --
    the "no routing intelligence at all" baseline.

Multiple trials per real risk grid (varying agent start positions via
seeded RNG) -> Wilcoxon rank-sum on total risk-value captured, matching
the statistical-rigor convention already used elsewhere in this repo
(damage_detection/compare_algorithms.py, swarm_control/benchmark_cec2017.py).

Metrics (identical scoring function applied to every allocator's output,
so the comparison is apples-to-apples):
  - total risk value captured (higher = better -- did the swarm actually
    go after the real damage that matters most)
  - load balance: std of per-agent captured value (lower = better -- an
    even workload split, not one agent doing everything)
  - travel efficiency: total travel distance per unit value captured
    (lower = better)

Usage (run from the repo root, ~/swarm_ws, so the default relative run-dir
names resolve):
    python3 -m cbba_allocator.compare_allocators --n_trials 15 \
        --out_dir cbba_comparison_results
"""

import os
import glob
import json
import argparse
import numpy as np

from .cbba_task_allocation import (
    CBBAAgent, extract_tasks_from_risk_grid, run_sequential_auction,
    MAX_BUNDLE_SIZE, ENV_MIN, ENV_MAX,
)

DEFAULT_RUN_DIRS = ["demo_dryrun", "verify_run", "live_run"]
ALGO_NAMES = ["cbba", "greedy", "round_robin"]


# ---------------------------------------------------------------------------
# Real data loading
# ---------------------------------------------------------------------------

def _load_combined_risk_grid(run_dir):
    """Same combination pattern cbba_node.py uses live: element-wise max
    across each agent's fused risk_score grid for this run. Defensively
    skips any individual file that fails to load (e.g. a 0-byte snapshot
    left behind by a run torn down mid-write) rather than aborting the
    whole comparison -- same defensive stance cbba_node.py's own
    _load_combined_risk_grid takes against transient read races."""
    risk_dir = os.path.join(run_dir, "damage_probability")
    paths = sorted(glob.glob(os.path.join(risk_dir, "*_risk_score.npy")))
    grids = []
    for p in paths:
        try:
            grids.append(np.load(p))
        except (EOFError, ValueError, OSError) as exc:
            print(f"[compare_allocators] skipping unreadable {p}: {exc}")
    if not grids:
        return None
    return np.maximum.reduce(grids)


def _random_start_positions(agent_names, rng):
    xy = rng.uniform(ENV_MIN[:2], ENV_MAX[:2], size=(len(agent_names), 2))
    # z restricted to the structure's real operating depth band (matches
    # where actual detections land in this project's captured runs, e.g.
    # prognosis.json positions cluster around z=-10..-30), not the full
    # water-column ENV_MIN/ENV_MAX z-range which would be an unrealistic
    # start position for an inspection sweep already underway.
    z = rng.uniform(-25.0, -10.0, size=len(agent_names))
    return {name: np.array([xy[i, 0], xy[i, 1], z[i]]) for i, name in enumerate(agent_names)}


# ---------------------------------------------------------------------------
# Baseline allocators
# ---------------------------------------------------------------------------

def greedy_allocate(agent_positions, tasks):
    """Repeatedly assigns the globally closest (task, agent-with-spare-
    capacity) pair until no capacity or tasks remain."""
    agent_names = list(agent_positions.keys())
    positions = {name: np.array(agent_positions[name], dtype=np.float64) for name in agent_names}
    assigned = {name: [] for name in agent_names}
    remaining = {t["task_id"]: t for t in tasks}

    while remaining:
        bidders = [n for n in agent_names if len(assigned[n]) < MAX_BUNDLE_SIZE]
        if not bidders:
            break
        best = None  # (dist, agent_name, task_id)
        for name in bidders:
            for tid, t in remaining.items():
                dist = float(np.linalg.norm(t["position"] - positions[name]))
                if best is None or dist < best[0]:
                    best = (dist, name, tid)
        _, name, tid = best
        assigned[name].append(tid)
        positions[name] = remaining[tid]["position"]  # agent "moves" there for its next leg
        del remaining[tid]
    return assigned


def round_robin_allocate(agent_positions, tasks):
    """Sorts tasks by value descending, assigns in rotating agent order --
    ignores position/travel cost entirely."""
    agent_names = list(agent_positions.keys())
    assigned = {name: [] for name in agent_names}
    ordered = sorted(tasks, key=lambda t: t["value"], reverse=True)
    idx = 0
    for t in ordered:
        for _ in range(len(agent_names)):
            name = agent_names[idx % len(agent_names)]
            idx += 1
            if len(assigned[name]) < MAX_BUNDLE_SIZE:
                assigned[name].append(t["task_id"])
                break
    return assigned


def _cbba_allocate(agent_positions, tasks):
    agents = {name: CBBAAgent(name, pos) for name, pos in agent_positions.items()}
    run_sequential_auction(agents, tasks)
    return {name: list(agent.path) for name, agent in agents.items()}


# ---------------------------------------------------------------------------
# Scoring (identical for every allocator)
# ---------------------------------------------------------------------------

def _score_assignment(agent_positions, assigned, tasks_by_id):
    total_value = 0.0
    total_distance = 0.0
    per_agent_value = []
    for name, task_ids in assigned.items():
        pos = np.array(agent_positions[name], dtype=np.float64)
        agent_value = 0.0
        for tid in task_ids:
            t = tasks_by_id[tid]
            total_distance += float(np.linalg.norm(t["position"] - pos))
            pos = t["position"]
            agent_value += t["value"]
        total_value += agent_value
        per_agent_value.append(agent_value)
    load_std = float(np.std(per_agent_value)) if per_agent_value else 0.0
    travel_per_value = total_distance / total_value if total_value > 1e-9 else None
    return {"total_value_captured": total_value, "load_balance_std": load_std,
            "travel_per_value": travel_per_value}


def run_single_trial(tasks, tasks_by_id, num_auvs, seed):
    agent_names = [f"auv{i}" for i in range(num_auvs)]
    rng = np.random.default_rng(seed)
    agent_positions = _random_start_positions(agent_names, rng)

    return {
        "cbba": _score_assignment(agent_positions, _cbba_allocate(agent_positions, tasks), tasks_by_id),
        "greedy": _score_assignment(agent_positions, greedy_allocate(agent_positions, tasks), tasks_by_id),
        "round_robin": _score_assignment(agent_positions, round_robin_allocate(agent_positions, tasks), tasks_by_id),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_comparison(run_dirs, num_auvs=3, n_trials=15, out_dir="cbba_comparison_results", base_seed=0):
    os.makedirs(out_dir, exist_ok=True)
    metrics = {a: {"total_value_captured": [], "load_balance_std": [], "travel_per_value": []}
               for a in ALGO_NAMES}

    n_ran = 0
    for run_dir in run_dirs:
        risk_grid = _load_combined_risk_grid(run_dir)
        if risk_grid is None:
            print(f"[compare_allocators] no *_risk_score.npy in {run_dir}/damage_probability, skipping")
            continue

        tasks = extract_tasks_from_risk_grid(risk_grid)
        if not tasks:
            print(f"[compare_allocators] no tasks above threshold in {run_dir}, skipping")
            continue
        tasks_by_id = {t["task_id"]: t for t in tasks}

        print(f"\n{'=' * 70}\n{run_dir}: {len(tasks)} real tasks extracted, "
              f"{n_trials} trials (varying agent start positions)\n{'=' * 70}")
        for trial in range(n_trials):
            seed = base_seed + trial
            result = run_single_trial(tasks, tasks_by_id, num_auvs, seed)
            n_ran += 1
            for a in ALGO_NAMES:
                for k in metrics[a]:
                    v = result[a][k]
                    if v is not None:
                        metrics[a][k].append(v)

    if n_ran == 0:
        raise RuntimeError(
            "No valid trials ran -- check --run_dirs point at real run directories "
            "containing damage_probability/*_risk_score.npy files.")

    _summarize_and_test(metrics, out_dir, n_ran)
    return metrics


def _summarize_and_test(metrics, out_dir, n_trials):
    from scipy.stats import ranksums

    print(f"\n{'=' * 70}\nSUMMARY ({n_trials} trials total across all run dirs)\n{'=' * 70}")
    summary = {}
    for a in ALGO_NAMES:
        tv = metrics[a]["total_value_captured"]
        lb = metrics[a]["load_balance_std"]
        tpv = metrics[a]["travel_per_value"]
        summary[a] = {
            "mean_total_value_captured": float(np.mean(tv)),
            "std_total_value_captured": float(np.std(tv)),
            "mean_load_balance_std": float(np.mean(lb)),
            "mean_travel_per_value": float(np.mean(tpv)) if tpv else None,
            "all_total_value_captured": tv,
        }
        s = summary[a]
        eff_str = f"{s['mean_travel_per_value']:.3f}" if s["mean_travel_per_value"] is not None else "n/a"
        print(f"{a:12s}  value_captured = {s['mean_total_value_captured']:7.2f} +/- "
              f"{s['std_total_value_captured']:.2f}   load_balance_std = "
              f"{s['mean_load_balance_std']:6.2f}   travel/value = {eff_str}")

    print(f"\n{'=' * 70}\nSIGNIFICANCE (Wilcoxon rank-sum on total value captured, "
          f"cbba vs. each baseline)\n{'=' * 70}")
    pairwise = {}
    for baseline in ["greedy", "round_robin"]:
        stat, pvalue = ranksums(summary["cbba"]["all_total_value_captured"],
                                 summary[baseline]["all_total_value_captured"])
        significant = pvalue < 0.05
        cbba_better = (summary["cbba"]["mean_total_value_captured"] >
                        summary[baseline]["mean_total_value_captured"])
        pairwise[f"cbba_vs_{baseline}"] = {
            "statistic": float(stat), "pvalue": float(pvalue),
            "significant_at_0.05": bool(significant), "cbba_better": bool(cbba_better),
        }
        marker = ("*** CBBA SIGNIFICANTLY BETTER" if significant and cbba_better else
                   "*** baseline significantly better" if significant else "not significant")
        print(f"cbba vs {baseline}: p={pvalue:.4g}  [{marker}]")

    best = max(ALGO_NAMES, key=lambda a: summary[a]["mean_total_value_captured"])
    print(f"\nBest mean value captured: {best} ({summary[best]['mean_total_value_captured']:.2f})")
    print("NOTE: check the pairwise significance above before claiming a definitive win -- "
          "a higher mean with a not-significant p-value against the runner-up means the "
          "difference could be noise, not a real effect.")

    out_path = os.path.join(out_dir, "comparison_summary.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "pairwise_significance": pairwise,
                    "n_trials": n_trials}, f, indent=2)
    print(f"\nFull results saved -> {out_path}")

    _plot_comparison(summary, out_dir)


def _plot_comparison(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {"cbba": "#023047", "greedy": "#219ebc", "round_robin": "#ffb703"}
    colors = [palette[a] for a in ALGO_NAMES]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    data = [summary[a]["all_total_value_captured"] for a in ALGO_NAMES]
    bp = axes[0].boxplot(data, labels=ALGO_NAMES, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    axes[0].set_ylabel("Total risk value captured")
    axes[0].set_title("Value captured (higher = better)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(ALGO_NAMES, [summary[a]["mean_load_balance_std"] for a in ALGO_NAMES], color=colors)
    axes[1].set_ylabel("Std of per-agent value captured")
    axes[1].set_title("Workload balance (lower = better)")
    axes[1].grid(axis="y", alpha=0.3)

    eff = [summary[a]["mean_travel_per_value"] or 0 for a in ALGO_NAMES]
    axes[2].bar(ALGO_NAMES, eff, color=colors)
    axes[2].set_ylabel("Travel distance per unit value captured")
    axes[2].set_title("Travel efficiency (lower = better)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(f"CBBA vs. baseline task allocators -- {len(data[0])} trials each, "
                 f"real captured risk grids")
    plot_path = os.path.join(out_dir, "comparison_chart.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Comparison chart saved -> {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", type=str, default=",".join(DEFAULT_RUN_DIRS),
                         help="Comma-separated real captured run directories (relative to cwd, "
                              "or absolute) to source risk grids from.")
    parser.add_argument("--num_auvs", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=15,
                         help="Independent trials per run dir (varying agent start positions).")
    parser.add_argument("--out_dir", type=str, default="cbba_comparison_results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_comparison(args.run_dirs.split(","), num_auvs=args.num_auvs, n_trials=args.n_trials,
                   out_dir=args.out_dir, base_seed=args.seed)
