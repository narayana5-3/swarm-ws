"""
Risk-prioritized adaptive task reallocation for the AUV swarm.

Assigns inspection tasks to swarm agents based on the risk_score grids from
damage_probability_mapping.py (damage probability x uncertainty, masked to
detected structure), using a decentralized, acoustic-range-limited
consensus scheme in the spirit of CBBA (Choi, Brunet & How, 2009 --
"Consensus-Based Decentralized Auctions for Robust Task Allocation").

Consistent with the rest of this pipeline: agents only exchange bid
information with peers within ACOUSTIC_MAX_RANGE, not an all-to-all
broadcast -- the same acoustic-mesh communication model used for occupancy
and damage belief fusion.

Pipeline:
  1. Extract discrete task points from the risk_score grid via local-maxima
     peak detection (NOT connected-component blob merging -- that was
     tried first and collapsed the entire swept structure into one giant
     "task" on real data, since the risk-masked region is topologically
     one continuous band; see extract_tasks_from_risk_grid()'s docstring
     for the full story).
  2. Tasks are auctioned ONE AT A TIME: every agent with spare capacity
     bids on every remaining task (risk value minus marginal travel cost
     to insert it into their current path); the swarm-wide highest bidder
     for the single best (task, agent) pair is confirmed via proper
     distributed max-consensus over the acoustic graph. That task is
     removed from the pool and the process repeats.
  3. Outputs final per-agent task assignment + ordered visiting path,
     ready to become the next waypoint plan for a subsequent sim run.

Design notes / simplifications (documented, not hidden):
  - DEVIATION FROM THE ORIGINAL CBBA PAPER: the original algorithm has
    agents speculatively build multi-task BUNDLES up front and RELEASE
    tasks when outbid, using per-task timestamps to resolve staleness.
    An earlier version of this file implemented that directly and testing
    surfaced repeated, subtle bugs: released claims aren't automatically
    known to be void by agents who'd previously heard about them, so tasks
    could end up permanently "believed claimed but held by no one." Fully
    solving this needs the paper's complete timestamp bookkeeping. Given
    the actual goal here (a robust, working risk-prioritized allocator,
    not a literal reproduction of one paper's exact mechanism), this was
    replaced with the simpler sequential single-task auction above:
    nothing is ever assigned speculatively, so nothing ever needs to be
    released, eliminating the whole bug class. Still decentralized,
    acoustic-range-limited, and consensus-based -- just sequenced
    differently for robustness.
  - Exact bid ties do occur legitimately here (a task inserted between two
    already-shared bundled tasks has a marginal cost independent of which
    agent computes it) -- handled via a tiny deterministic per-agent
    perturbation (TIE_BREAK_EPSILON) so ties essentially never occur in
    practice, plus a deterministic name-based tie-break rule as a
    defensive fallback.
  - Travel cost is straight-line distance through the bundle's task
    sequence (no obstacle-aware path planning at this stage) -- fine
    given task points are already filtered to be near open, swept
    structure, not inside solid geometry.
  - Agent "current position" for planning is each agent's LAST logged
    position in the sensor log (i.e. where the previous sweep left off).

Run (after damage_probability_mapping.py has produced risk_score.npy files):
    python cbba_task_allocation.py --run_dir logs/run_<ts>
"""

import os
import glob
import pickle
import hashlib
import argparse
import numpy as np

from occupancy_mapping.occupancy_mapping import ENV_MIN, ENV_MAX, VOXEL_SIZE, ACOUSTIC_MAX_RANGE, find_latest_run_dir

MAX_BUNDLE_SIZE = 4          # max tasks per agent per allocation round
RISK_THRESHOLD = 0.05        # min risk_score to be considered a task at all
MAX_CONSENSUS_ROUNDS = 20
TRAVEL_COST_WEIGHT = 0.02    # risk lost per meter of travel (tuned against
                               # TASK_VALUE_SCALE below, not raw risk-mass sums)
TASK_VALUE_SCALE = 100.0     # tasks are normalized so the highest-value task
                               # equals this, regardless of a run's absolute
                               # risk-mass scale -- keeps TRAVEL_COST_WEIGHT's
                               # effect consistent across different runs/sweeps
MIN_TASK_SEPARATION = 3.0    # meters -- merge task clusters closer than this
TIE_BREAK_EPSILON = 1e-4     # tiny deterministic per-agent bid perturbation --
                               # eliminates exact ties at the source (which
                               # occur legitimately when a task is inserted
                               # between two already-shared bundled tasks,
                               # making raw insertion cost agent-position-
                               # independent) rather than relying solely on
                               # downstream tie-break bookkeeping during
                               # consensus, which is fragile under repeated
                               # cascading releases in the same round.


# ---------------------------------------------------------------------------
# Task extraction: cluster high-risk voxels into discrete task points
# ---------------------------------------------------------------------------

def extract_tasks_from_risk_grid(risk_grid, env_min=ENV_MIN, voxel_size=VOXEL_SIZE):
    """
    Local-maxima peak detection, NOT connected-component blob merging.

    Connected-component labeling was tried first and failed on real data: the
    risk-masked region near a swept structure is topologically ONE continuous
    band (it wraps the whole inspected surface), so labeling merged nearly
    the entire structure into a single giant "task" (36,847 voxels, value
    dwarfing everything else) plus a few stray 2-3 voxel specks -- meaning
    there was effectively only one real task to bid on, explaining why one
    agent swept everything. Local-maxima detection instead finds distinct
    high-risk POINTS within the field (with a minimum separation), which is
    what "distinct inspection targets" actually means, regardless of whether
    the underlying risk region is one contiguous band or several separate ones.
    """
    from scipy.ndimage import maximum_filter

    footprint_size = max(3, int(round(MIN_TASK_SEPARATION / voxel_size)))
    if footprint_size % 2 == 0:
        footprint_size += 1  # odd size -> symmetric neighborhood around each peak

    local_max = maximum_filter(risk_grid, size=footprint_size, mode="constant")
    peak_mask = (risk_grid == local_max) & (risk_grid > RISK_THRESHOLD)
    peak_idx = np.argwhere(peak_mask)

    half = footprint_size // 2
    tasks = []
    for idx in peak_idx:
        i, j, k = idx
        # task "value" = risk mass in a local neighborhood around the peak,
        # NOT the entire connected region -- keeps values representative of
        # a local inspection target's size, not the whole structure's extent.
        window = risk_grid[max(0, i - half):i + half + 1,
                            max(0, j - half):j + half + 1,
                            max(0, k - half):k + half + 1]
        value = float(window.sum())
        world_pos = env_min + (np.array([i, j, k]) + 0.5) * voxel_size
        tasks.append({"position": world_pos, "value": value, "n_voxels": int(window.size)})

    if not tasks:
        return []

    # merge tasks that are suspiciously close (avoids redundant near-duplicate
    # inspection targets from a single physically-contiguous defect)
    tasks = _merge_close_tasks(tasks, MIN_TASK_SEPARATION)

    # Normalize values to a fixed range (0-100) BEFORE bidding. Without this,
    # task value scale depends on how much risk mass a given run happens to
    # accumulate (a longer sweep or bigger structure -> bigger raw sums) --
    # and if that scale dwarfs TRAVEL_COST_WEIGHT's effect, position stops
    # mattering at all and whichever agent has even a microscopic advantage
    # can end up winning every task (confirmed on a real run: one agent
    # captured 100% of tasks when raw values averaged ~4600 against a travel
    # cost of a few units per meter). Normalizing keeps travel cost's
    # influence consistent across runs regardless of absolute risk-mass scale.
    if tasks:
        max_value = max(t["value"] for t in tasks)
        if max_value > 1e-6:
            for t in tasks:
                t["value"] = t["value"] / max_value * TASK_VALUE_SCALE

    for i, t in enumerate(tasks):
        t["task_id"] = f"task_{i:03d}"

    return tasks


def _merge_close_tasks(tasks, min_sep):
    """
    Deduplicates nearby peak detections by keeping only the STRONGEST
    (highest-value) peak in each close-together group, discarding the
    rest as redundant. This is deliberately NOT a sum -- unlike merging
    genuinely separate defect regions, nearby peaks from local-maxima
    detection often come from the same broad, gently-varying hotspot (a
    "plateau" where many adjacent voxels tie for the local max), and
    summing their values would double/triple/N-count the same underlying
    risk mass, recreating the exact mega-value inflation problem this
    module was rewritten to avoid in the first place.
    """
    merged = []
    used = [False] * len(tasks)
    for i, t in enumerate(tasks):
        if used[i]:
            continue
        group = [t]
        used[i] = True
        for j in range(i + 1, len(tasks)):
            if used[j]:
                continue
            if np.linalg.norm(t["position"] - tasks[j]["position"]) < min_sep:
                group.append(tasks[j])
                used[j] = True
        strongest = max(group, key=lambda g: g["value"])
        merged.append(dict(strongest))
    return merged


# ---------------------------------------------------------------------------
# CBBA agent state
# ---------------------------------------------------------------------------

class CBBAAgent:
    def __init__(self, name, position, max_bundle_size=MAX_BUNDLE_SIZE):
        self.name = name
        self.position = np.array(position, dtype=np.float64)
        self.max_bundle_size = max_bundle_size

        # deterministic, stable per-agent perturbation to break exact bid
        # ties at the source -- same agent always gets the same tiny offset
        # regardless of run order, different agents get different offsets
        # (collision probability negligible for any reasonable swarm size)
        h = int(hashlib.md5(name.encode()).hexdigest(), 16)
        self.tie_break_offset = (h % 10_000) / 10_000 * TIE_BREAK_EPSILON

        self.bundle = []   # ordered list of CONFIRMED task_ids (visiting order)
        self.path = []     # same as bundle -- kept separate name for clarity
                            # in cost calculations

    def path_positions(self, tasks_by_id):
        return [self.position] + [tasks_by_id[tid]["position"] for tid in self.path]

    def marginal_cost_of_insertion(self, task, tasks_by_id):
        """Best (task_value - travel_cost) achievable by inserting `task`
        at the best position in this agent's CONFIRMED path. Returns
        (best_score, best_insert_index). Only ever called for tasks not
        already in this agent's bundle and while it has spare capacity --
        the caller (run_sequential_auction) enforces both."""
        positions = self.path_positions(tasks_by_id)
        best_score = -np.inf
        best_index = None

        for insert_at in range(1, len(positions) + 1):
            prev_pos = positions[insert_at - 1]
            next_pos = positions[insert_at] if insert_at < len(positions) else None

            added_dist = np.linalg.norm(task["position"] - prev_pos)
            removed_dist = 0.0
            if next_pos is not None:
                added_dist += np.linalg.norm(next_pos - task["position"])
                removed_dist = np.linalg.norm(next_pos - prev_pos)

            travel_cost = (added_dist - removed_dist) * TRAVEL_COST_WEIGHT
            score = task["value"] - travel_cost

            if score > best_score:
                best_score = score
                best_index = insert_at - 1  # index into self.path to insert before

        return best_score + self.tie_break_offset, best_index

    def confirm_task(self, task_id, index):
        self.bundle.insert(index, task_id)
        self.path = list(self.bundle)


# ---------------------------------------------------------------------------
# Sequential single-task auction via distributed max-consensus
# ---------------------------------------------------------------------------
#
# Simpler and more robust than the original CBBA paper's bundle+release
# design for this use case: rather than agents speculatively building
# multi-task bundles and later needing to RELEASE tasks when outbid (which
# requires precise stale-claim propagation across the swarm to avoid
# orphaned tasks -- easy to get subtly wrong, as testing here surfaced),
# tasks are auctioned ONE AT A TIME. For each auction round, every agent
# with spare capacity bids on every remaining task; the swarm-wide highest
# bidder for the SINGLE best (task, agent) pair is confirmed via proper
# distributed max-consensus (monotonic, so no stale-claim bugs are
# possible -- there is nothing to release, since nothing is assigned
# speculatively). That one task is removed from the pool and the process
# repeats. Still decentralized and acoustic-range-limited, still
# consensus-based, just sequenced differently than the original paper for
# robustness. Documented here as a deliberate deviation, not an oversight.

def _bid_beats(their_bid, their_agent, my_bid, my_agent):
    """Deterministic comparison: strictly higher bid wins; exact ties
    (rare now that TIE_BREAK_EPSILON perturbs bids at the source, but kept
    as a defensive fallback) are broken by agent name so every agent
    resolves the same tie identically."""
    if their_bid > my_bid:
        return True
    if their_bid == my_bid and my_agent is None:
        return True
    if their_bid == my_bid and my_agent is not None and their_agent < my_agent:
        return True
    return False


def _max_consensus_round(agents, local_best, acoustic_range):
    """One round of max-consensus: each agent adopts the best (bid, agent)
    pair it hears from any neighbor within range, using the SAME
    deterministic tie-break as bidding itself. Returns whether anything
    changed (for convergence detection)."""
    incoming = {name: dict(local_best[name]) for name in agents}

    for sender_name, sender in agents.items():
        for receiver_name, receiver in agents.items():
            if sender_name == receiver_name:
                continue
            if np.linalg.norm(sender.position - receiver.position) > acoustic_range:
                continue
            for task_id, (bid, agent_name) in local_best[sender_name].items():
                current = incoming[receiver_name].get(task_id)
                if current is None or _bid_beats(bid, agent_name, current[0], current[1]):
                    incoming[receiver_name][task_id] = (bid, agent_name)

    changed = any(incoming[name] != local_best[name] for name in agents)
    return incoming, changed


def run_sequential_auction(agents, tasks, acoustic_range=ACOUSTIC_MAX_RANGE,
                            max_consensus_rounds=MAX_CONSENSUS_ROUNDS):
    tasks_by_id = {t["task_id"]: t for t in tasks}
    remaining_task_ids = set(tasks_by_id.keys())
    agent_names = list(agents.keys())

    n_auctions = 0
    while remaining_task_ids:
        bidders_left = [name for name in agent_names
                         if len(agents[name].bundle) < agents[name].max_bundle_size]
        if not bidders_left:
            break  # swarm is fully at capacity

        # each agent computes its own best (task, bid, index) among remaining tasks
        local_best = {name: {} for name in agent_names}
        agent_best_index = {}
        for name in bidders_left:
            agent = agents[name]
            for task_id in remaining_task_ids:
                score, index = agent.marginal_cost_of_insertion(tasks_by_id[task_id], tasks_by_id)
                local_best[name][task_id] = (score, name)
                agent_best_index[(name, task_id)] = index

        # distributed max-consensus over the acoustic graph until stable
        # (or a generous round cap -- covers worst-case graph diameter)
        for _ in range(max_consensus_rounds):
            local_best, changed = _max_consensus_round(agents, local_best, acoustic_range)
            if not changed:
                break

        # every agent now agrees on the swarm-wide best (task, bid, agent) for
        # each task; find the single overall best pair to confirm this round
        global_best_task, global_best_bid, global_best_agent = None, -np.inf, None
        for name in agent_names:
            for task_id, (bid, agent_name) in local_best[name].items():
                if _bid_beats(bid, agent_name, global_best_bid, global_best_agent):
                    global_best_task, global_best_bid, global_best_agent = task_id, bid, agent_name

        if global_best_task is None:
            break  # no remaining bids at all (shouldn't happen if bidders_left non-empty)

        winner = agents[global_best_agent]
        index = agent_best_index[(global_best_agent, global_best_task)]
        winner.confirm_task(global_best_task, index)
        remaining_task_ids.discard(global_best_task)
        n_auctions += 1

    print(f"[cbba] {n_auctions} sequential single-task auctions completed")
    return agents


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def get_last_positions(run_dir, agent_names):
    positions = {}
    for name in agent_names:
        path = os.path.join(run_dir, f"{name}_sensor_log.pkl")
        with open(path, "rb") as f:
            records = pickle.load(f)
        for r in reversed(records):
            if r.get("location") is not None:
                positions[name] = r["location"]
                break
    return positions


def allocate(run_dir, damage_probability_dir=None):
    damage_probability_dir = damage_probability_dir or os.path.join(run_dir, "damage_probability")
    risk_paths = sorted(glob.glob(os.path.join(damage_probability_dir, "*_risk_score.npy")))
    if not risk_paths:
        raise FileNotFoundError(
            f"No *_risk_score.npy found in {damage_probability_dir} -- "
            f"run damage_probability_mapping.py first.")

    agent_names = [os.path.basename(p).replace("_risk_score.npy", "") for p in risk_paths]

    # risk grids should already agree closely across agents post-fusion;
    # use the first agent's as the shared task-generation basis
    risk_grid = np.load(risk_paths[0])
    tasks = extract_tasks_from_risk_grid(risk_grid)
    print(f"[cbba] extracted {len(tasks)} candidate tasks from risk grid "
          f"(threshold={RISK_THRESHOLD})")
    if not tasks:
        print("[cbba] No tasks above threshold -- nothing to allocate. "
              "Consider lowering RISK_THRESHOLD or running a longer sweep "
              "to build up more risk signal.")
        return None

    positions = get_last_positions(run_dir, agent_names)
    agents = {name: CBBAAgent(name, positions[name]) for name in agent_names}

    agents = run_sequential_auction(agents, tasks)

    tasks_by_id = {t["task_id"]: t for t in tasks}
    results = {}
    for name, agent in agents.items():
        assigned = [{"task_id": tid, "position": tasks_by_id[tid]["position"].tolist(),
                     "value": tasks_by_id[tid]["value"]} for tid in agent.path]
        total_value = sum(t["value"] for t in assigned)
        results[name] = {"start_position": agent.position.tolist(),
                          "assigned_tasks": assigned, "total_value_captured": total_value}
        print(f"[cbba] {name}: {len(assigned)} tasks, "
              f"total risk value captured = {total_value:.3f}")

    unassigned = [tid for tid in tasks_by_id
                  if not any(tid in a.bundle for a in agents.values())]
    if unassigned:
        print(f"[cbba] {len(unassigned)} tasks unassigned (exceeded swarm's "
              f"total bundle capacity of {len(agents) * MAX_BUNDLE_SIZE})")

    out_dir = os.path.join(run_dir, "task_allocation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cbba_assignment.pkl"), "wb") as f:
        pickle.dump({"agents": results, "all_tasks": tasks,
                     "unassigned_task_ids": unassigned}, f)
    print(f"[cbba] assignment saved -> {out_dir}/cbba_assignment.pkl")

    _plot_assignment(results, tasks, unassigned, out_dir)
    return results


def _plot_assignment(results, tasks, unassigned, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10.colors

    unassigned_set = set(unassigned)
    for t in tasks:
        marker = "x" if t["task_id"] in unassigned_set else "o"
        ax.scatter(t["position"][0], t["position"][1], s=40 + t["value"] * 20,
                   c="gray" if t["task_id"] in unassigned_set else "lightgray",
                   marker=marker, edgecolors="black", linewidths=0.5, zorder=2)

    for i, (name, res) in enumerate(results.items()):
        color = colors[i % len(colors)]
        start = res["start_position"]
        path_x = [start[0]] + [t["position"][0] for t in res["assigned_tasks"]]
        path_y = [start[1]] + [t["position"][1] for t in res["assigned_tasks"]]
        ax.plot(path_x, path_y, "-", color=color, linewidth=1.5, zorder=1,
                label=f"{name} ({len(res['assigned_tasks'])} tasks)")
        ax.scatter([start[0]], [start[1]], s=100, c=[color], marker="^",
                   edgecolors="black", zorder=3)
        for t in res["assigned_tasks"]:
            ax.scatter(t["position"][0], t["position"][1], s=40 + t["value"] * 20,
                       c=[color], edgecolors="black", linewidths=0.8, zorder=3)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("CBBA task allocation (triangle=start, circle=assigned task, "
                 "x=unassigned, size~risk value)")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    plot_path = os.path.join(out_dir, "cbba_assignment.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[cbba] assignment plot saved -> {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None,
                         help="Path to a logs/run_<ts> directory. Defaults to most recent.")
    parser.add_argument("--damage_probability_dir", type=str, default=None,
                         help="Directory with *_risk_score.npy files. "
                              "Defaults to <run_dir>/damage_probability.")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    print(f"Using run_dir: {run_dir}")
    allocate(run_dir, args.damage_probability_dir)
