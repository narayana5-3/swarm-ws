"""
Iterative / repeated task reallocation -- the "adaptive" half of "adaptive
task reallocation" that cbba_task_allocation.py's single-shot allocation
doesn't cover on its own.

Motivation: a single CBBA call assigns each agent up to MAX_BUNDLE_SIZE
tasks and stops -- on real data this covered only 16 of 292 detected
high-risk points in one round (see cbba_task_allocation.py's README
section). This module repeats the allocation across multiple rounds:
after each round, agents' positions advance to wherever their last
assigned task was (simulating that they physically completed that batch),
completed tasks are removed from the pool, and CBBA runs again on what's
left -- until the whole backlog is covered or a round cap is hit.

This reuses the EXACT SAME task-extraction and auction logic as
cbba_task_allocation.py (imported directly, not reimplemented), so a
single-round call through this module produces identical results to
calling cbba_task_allocation.py directly -- this is genuinely the same
algorithm run repeatedly, not a different one.

Design note: the risk grid itself is NOT re-sensed between rounds here --
this offline analysis pipeline doesn't have a live re-sensing loop (that
would need another real HoloOcean sweep between each round). What this
does model is the swarm working through its already-detected backlog of
risk points across multiple dispatch cycles, which is itself a real and
useful capability distinct from one-shot allocation. Extending this to
re-sense between rounds (rerun occupancy_mapping.py + 
damage_probability_mapping.py against fresh sensor data after each
physical sweep) is the natural next step once you're running iterative
rounds against a live sim rather than a fixed logged run.

Run:
    python iterative_task_allocation.py --run_dir logs/run_<ts> --max_rounds 10
"""

import os
import json
import pickle
import argparse
import numpy as np

from cbba_task_allocation import (CBBAAgent, extract_tasks_from_risk_grid,
                                    run_sequential_auction, get_last_positions,
                                    MAX_BUNDLE_SIZE)
from occupancy_mapping.occupancy_mapping import find_latest_run_dir


def run_iterative_allocation(run_dir, damage_probability_dir=None, max_rounds=10,
                              risk_capture_threshold=0.99):
    import glob
    damage_probability_dir = damage_probability_dir or os.path.join(run_dir, "damage_probability")
    risk_paths = sorted(glob.glob(os.path.join(damage_probability_dir, "*_risk_score.npy")))
    if not risk_paths:
        raise FileNotFoundError(
            f"No *_risk_score.npy found in {damage_probability_dir} -- "
            f"run damage_probability_mapping.py first.")

    agent_names = [os.path.basename(p).replace("_risk_score.npy", "") for p in risk_paths]
    risk_grid = np.load(risk_paths[0])

    all_tasks = extract_tasks_from_risk_grid(risk_grid)
    total_task_value = sum(t["value"] for t in all_tasks)
    print(f"[iterative] {len(all_tasks)} total tasks extracted, "
          f"total value = {total_task_value:.1f}")
    if not all_tasks:
        print("[iterative] No tasks above threshold -- nothing to allocate.")
        return None

    positions = get_last_positions(run_dir, agent_names)
    remaining_tasks = {t["task_id"]: t for t in all_tasks}

    round_history = []
    cumulative_value = 0.0

    for round_i in range(max_rounds):
        if not remaining_tasks:
            print(f"\n[iterative] all tasks covered after {round_i} round(s).")
            break

        print(f"\n{'=' * 70}\nROUND {round_i + 1}/{max_rounds}  "
              f"({len(remaining_tasks)} tasks remaining, "
              f"{cumulative_value:.1f}/{total_task_value:.1f} value captured so far)\n{'=' * 70}")

        agents = {name: CBBAAgent(name, positions[name]) for name in agent_names}
        tasks_this_round = list(remaining_tasks.values())
        agents = run_sequential_auction(agents, tasks_this_round)

        round_assigned = {}
        round_value = 0.0
        for name, agent in agents.items():
            if not agent.bundle:
                continue
            round_assigned[name] = list(agent.bundle)
            for task_id in agent.bundle:
                round_value += remaining_tasks[task_id]["value"]
            # advance this agent's position to its last task this round --
            # simulates having physically completed the sweep
            positions[name] = remaining_tasks[agent.bundle[-1]]["position"]

        for name, task_ids in round_assigned.items():
            for task_id in task_ids:
                del remaining_tasks[task_id]

        cumulative_value += round_value
        n_assigned_this_round = sum(len(v) for v in round_assigned.values())
        print(f"[iterative] round {round_i + 1}: {n_assigned_this_round} tasks assigned, "
              f"{round_value:.1f} value captured this round")

        round_history.append({
            "round": round_i + 1,
            "assigned": round_assigned,
            "n_assigned": n_assigned_this_round,
            "round_value": round_value,
            "cumulative_value": cumulative_value,
            "tasks_remaining": len(remaining_tasks),
            "agent_positions_after": {k: list(v) for k, v in positions.items()},
        })

        if cumulative_value / total_task_value >= risk_capture_threshold:
            print(f"\n[iterative] {risk_capture_threshold*100:.0f}% of total risk value "
                  f"captured after {round_i + 1} round(s) -- stopping early.")
            break

        if n_assigned_this_round == 0:
            print(f"\n[iterative] WARNING: round {round_i + 1} assigned nothing "
                  f"(swarm at capacity: {len(agent_names) * MAX_BUNDLE_SIZE} slots/round) "
                  f"but {len(remaining_tasks)} tasks remain -- stopping to avoid an "
                  f"infinite no-progress loop. This shouldn't happen since a fresh "
                  f"CBBAAgent is created each round (fresh capacity); if you see this, "
                  f"something's wrong with round setup, not swarm capacity.")
            break

    else:
        print(f"\n[iterative] hit max_rounds ({max_rounds}) with "
              f"{len(remaining_tasks)} tasks still unassigned "
              f"({cumulative_value:.1f}/{total_task_value:.1f} value captured).")

    out_dir = os.path.join(run_dir, "task_allocation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "iterative_allocation_history.pkl"), "wb") as f:
        pickle.dump({"round_history": round_history, "total_task_value": total_task_value,
                      "all_tasks": all_tasks}, f)

    summary = {
        "n_rounds_run": len(round_history),
        "total_tasks": len(all_tasks),
        "total_task_value": total_task_value,
        "final_cumulative_value": cumulative_value,
        "fraction_value_captured": cumulative_value / total_task_value if total_task_value > 0 else 0,
        "tasks_still_remaining": len(remaining_tasks),
    }
    with open(os.path.join(out_dir, "iterative_allocation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    _plot_progress(round_history, total_task_value, out_dir)
    return round_history


def _plot_progress(round_history, total_task_value, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not round_history:
        return

    rounds = [r["round"] for r in round_history]
    cumulative = [r["cumulative_value"] for r in round_history]
    remaining = [r["tasks_remaining"] for r in round_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rounds, cumulative, "o-", color="#2ca02c")
    ax1.axhline(total_task_value, color="gray", linestyle="--", label="total available value")
    ax1.set_xlabel("dispatch round")
    ax1.set_ylabel("cumulative risk value captured")
    ax1.set_title("Coverage progress over dispatch rounds")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(rounds, remaining, color="#d62728")
    ax2.set_xlabel("dispatch round")
    ax2.set_ylabel("tasks remaining after this round")
    ax2.set_title("Backlog remaining")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "iterative_allocation_progress.png")
    fig.savefig(plot_path, dpi=150)
    print(f"[iterative] progress plot saved -> {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--damage_probability_dir", type=str, default=None)
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--risk_capture_threshold", type=float, default=0.99,
                         help="Stop early once this fraction of total detected "
                              "risk value has been captured.")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    print(f"Using run_dir: {run_dir}")
    run_iterative_allocation(run_dir, args.damage_probability_dir,
                              max_rounds=args.max_rounds,
                              risk_capture_threshold=args.risk_capture_threshold)
