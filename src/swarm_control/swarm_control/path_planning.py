"""
Risk-aware AUV path planning, derived from and modifying:

Luo, Liu, Wu, Xia, Filaretov, Yukhimets, Zuev & Atanazarovich, "Multi-AUV
path planning based on a multi-strategy fused improved nutcracker
optimization algorithm" (INOA-SQP), Ocean Engineering 364 (2026) 127024.

WHAT'S KEPT FROM THE ORIGINAL PAPER (same core mechanism):
  - Nutcracker Optimization Algorithm (NOA) as the base metaheuristic:
    foraging (global search), storage/caching (local refinement via
    remembered reference points), recovery (greedy retention).
  - Dynamic Hierarchical Acceptance (DHA): a simulated-annealing-style
    Metropolis criterion where each individual's "temperature" is set by
    its EMA-smoothed fitness rank, not a single global temperature -- lets
    top performers exploit while lower-ranked individuals keep exploring.
  - Chaotic Dynamic Lens Opposition-Based Learning (CDL-LOBL): a Tent-map-
    modulated opposition search that kicks stagnant individuals to the
    mirrored region of the search space, escaping local optima.
  - SQP local refinement (via scipy's SLSQP, a sequential-quadratic-
    programming-family solver) applied to the elite individuals when the
    global best stagnates -- not every individual every iteration, since
    that's the paper's own identified computational bottleneck.
  - Cubic-spline path smoothing + post-smoothing feasibility verification
    with a rollback mechanism if smoothing reintroduces a collision.

WHAT'S GENUINELY MODIFIED (the actual contribution here, not a relabel):
  1. Obstacle field = this project's REAL swarm-sensed occupancy grid (from
     occupancy_mapping.py, built from actual sonar returns), not the
     paper's synthetic sphere/cuboid/seamount models. The swarm plans
     around what it actually detected, not assumed geometry.
  2. Cost function is risk/uncertainty-aware: paths get a bonus for passing
     near cells in the damage_probability_mapping.py risk-score grid --
     incidental inspection coverage on the way to a CBBA-assigned target,
     something the original planner has no concept of.
  3. No ocean-current term (the paper's Section 4.1.7) -- HoloOcean's
     current simulation isn't part of this project's sensor pipeline, so
     that cost term would be unused/fabricated. Dropped rather than faked.
  4. Addresses the paper's own explicitly-stated limitation ("real-time
     response capability in highly dynamic environments still needs to be
     improved," Section 6) by defaulting SQP's trigger threshold higher and
     its per-call iteration cap lower than the paper's own settings --
     deliberately trading a little final-polish accuracy for planning
     speed, appropriate for a resource-constrained swarm replanning
     periodically rather than a one-shot offline global plan.

Run (plans a path between two explicit points, using a run's real
occupancy + risk grids as the environment):
    python path_planning.py --run_dir logs/run_<ts> \
        --start 0,0,0 --goal 20,10,-5

Or, plan every leg of every agent's actual CBBA-assigned route:
    python path_planning.py --run_dir logs/run_<ts> --plan_cbba_routes
"""

import os
import glob
import pickle
import argparse
import numpy as np

from occupancy_mapping.occupancy_mapping import VoxelGrid, ENV_MIN, ENV_MAX, VOXEL_SIZE, find_latest_run_dir

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_WAYPOINTS = 7          # matches paper's num_inter_pts convention (5 intermediate + start/goal)
POP_SIZE = 40               # paper uses 100; scaled down for CPU-bound replanning speed
MAX_ITERS = 150             # paper uses 200 for benchmarks; kept lower per the real-time framing above
DELTA = 0.05                # foraging exploration probability control (paper's eq. 12)
PA2 = 0.2                   # cache/recovery stage switch probability (paper's eq. 26)

# DHA (Dynamic Hierarchical Acceptance) params -- paper Section 3.2.3
EMA_ALPHA_BASE = 0.7
EMA_C_ALPHA = 0.4
TEMP_COOLING_RATE = 0.97
TEMP_C1 = 0.5
TEMP_C2 = 1.5
TEMP_GAMMA = 1.5
ELITE_RATIO = 0.15

# CDL-LOBL (Chaotic Dynamic Lens Opposition-Based Learning) -- paper Section 3.2.4
TENT_ALPHA = 0.7
LOBL_K_MAX = 2.5
LOBL_K_MIN = 1.0
CHAOS_SIGMA = 0.3
STAGNATION_ROUNDS_FOR_LOBL = 5

# SQP triggering -- paper Section 3.2.2, thresholds adjusted per the
# real-time-focused modification described in the module docstring
SQP_STAGNATION_THRESHOLD = 15    # paper uses 5-20 dynamic; fixed lower-effort value here
SQP_ELITE_COUNT = 2               # paper uses 3
SQP_MAX_ITERS = 15                # cap scipy SLSQP's own iteration budget per call

# Cost function weights
W_LENGTH = 1.0
W_COLLISION = 400.0
W_SMOOTHNESS = 8.0
W_RISK_BONUS = 25.0        # subtracted from cost -- reward passing near risk

OCCUPANCY_COLLISION_THRESHOLD = 0.5   # P(occupied) above this counts as solid
BOUNDARY_MARGIN = 2.0                  # meters of slack inside env bounds


# ---------------------------------------------------------------------------
# Environment: real occupancy + risk grids as the obstacle/reward field
# ---------------------------------------------------------------------------

class PlanningEnvironment:
    def __init__(self, run_dir, agent_name=None):
        occ_paths = sorted(glob.glob(os.path.join(run_dir, "occupancy", "*_fused_logodds.npy")))
        if not occ_paths:
            raise FileNotFoundError(
                f"No occupancy grids in {run_dir}/occupancy -- run occupancy_mapping.py first.")
        occ_path = occ_paths[0] if agent_name is None else \
            os.path.join(run_dir, "occupancy", f"{agent_name}_fused_logodds.npy")
        occ_log_odds = np.load(occ_path)
        self.occ_prob = 1.0 / (1.0 + np.exp(-occ_log_odds))

        risk_paths = sorted(glob.glob(os.path.join(run_dir, "damage_probability", "*_risk_score.npy")))
        if risk_paths:
            risk_path = risk_paths[0] if agent_name is None else \
                os.path.join(run_dir, "damage_probability", f"{agent_name}_risk_score.npy")
            self.risk = np.load(risk_path) if os.path.exists(risk_path) else np.load(risk_paths[0])
        else:
            print("[path_planning] WARNING: no risk_score grids found -- "
                  "planning without risk-bonus term. Run damage_probability_mapping.py "
                  "first for the full risk-aware behavior.")
            self.risk = np.zeros_like(self.occ_prob)

        self.grid = VoxelGrid()  # just for world_to_index geometry, not used for storage here
        self.env_min = ENV_MIN.astype(np.float64)
        self.env_max = ENV_MAX.astype(np.float64)

    def occupancy_at(self, point):
        idx = self.grid.world_to_index(point)
        if not self.grid.in_bounds(idx):
            return 1.0  # out of known bounds -- treat as unsafe, not free
        i, j, k = idx
        return float(self.occ_prob[i, j, k])

    def risk_at(self, point):
        idx = self.grid.world_to_index(point)
        if not self.grid.in_bounds(idx):
            return 0.0
        i, j, k = idx
        return float(self.risk[i, j, k])


# ---------------------------------------------------------------------------
# Path representation and cost function
# ---------------------------------------------------------------------------
#
# Decision vector = flattened (x,y,z) of the NUM_WAYPOINTS-2 intermediate
# waypoints (start and goal are fixed). Matches the paper's P_k =
# {P_k,1,...,P_k,n} representation (Section 2.1 / Fig. 1).

def decision_vector_bounds(start, goal, env, margin=15.0):
    """Search space for intermediate waypoints: a padded box around the
    straight line between start and goal, clipped to the environment
    bounds. Keeps the search space from being the whole (huge) map."""
    lo = np.minimum(start, goal) - margin
    hi = np.maximum(start, goal) + margin
    lo = np.maximum(lo, env.env_min + BOUNDARY_MARGIN)
    hi = np.minimum(hi, env.env_max - BOUNDARY_MARGIN)
    return lo, hi


def build_full_path(decision_vector, start, goal, num_intermediate):
    intermediate = decision_vector.reshape(num_intermediate, 3)
    return np.vstack([start, intermediate, goal])


COLLISION_CHECK_SPACING = 0.2  # meters between collision-check samples --
                                 # must be smaller than VOXEL_SIZE (0.5m) or
                                 # a thin obstacle can be stepped over between
                                 # samples entirely (this happened: a fixed
                                 # 25-point resample missed a 1.5m-thick wall
                                 # on a long, winding path where point spacing
                                 # exceeded the wall's thickness)


def path_cost(decision_vector, start, goal, num_intermediate, env, resample_n=None):
    path = build_full_path(decision_vector, start, goal, num_intermediate)

    # path length
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

    # densely resample along the polyline for collision/risk checks -- point
    # SPACING (not a fixed count) must stay below COLLISION_CHECK_SPACING,
    # or a thin obstacle can be stepped over between samples on a long path
    if resample_n is None:
        resample_n = max(25, int(length / COLLISION_CHECK_SPACING))
    dense_pts = _resample_polyline(path, resample_n)

    collision = 0.0
    risk_bonus = 0.0
    for pt in dense_pts:
        p_occ = env.occupancy_at(pt)
        if p_occ > OCCUPANCY_COLLISION_THRESHOLD:
            collision += (p_occ - OCCUPANCY_COLLISION_THRESHOLD) ** 2
        risk_bonus += env.risk_at(pt)
    risk_bonus /= len(dense_pts)

    # smoothness: sum of squared turning angles at each intermediate waypoint
    smoothness = 0.0
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        smoothness += angle ** 2

    # boundary penalty
    boundary_penalty = 0.0
    for pt in dense_pts:
        below = np.maximum(0, (env.env_min + BOUNDARY_MARGIN) - pt)
        above = np.maximum(0, pt - (env.env_max - BOUNDARY_MARGIN))
        boundary_penalty += np.sum(below ** 2) + np.sum(above ** 2)

    cost = (W_LENGTH * length + W_COLLISION * collision + W_SMOOTHNESS * smoothness
            - W_RISK_BONUS * risk_bonus + 50.0 * boundary_penalty)
    return cost


def _resample_polyline(path, n):
    seg_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = seg_lengths.sum()
    if total < 1e-6:
        return np.tile(path[0], (n, 1))
    targets = np.linspace(0, total, n)
    cum = np.concatenate([[0], np.cumsum(seg_lengths)])
    out = np.zeros((n, 3))
    for i, t in enumerate(targets):
        seg_idx = np.searchsorted(cum, t, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(seg_lengths) - 1)
        seg_t = (t - cum[seg_idx]) / max(seg_lengths[seg_idx], 1e-6)
        out[i] = path[seg_idx] + seg_t * (path[seg_idx + 1] - path[seg_idx])
    return out


class NOAPlanner:
    """Thin wrapper around the generic NOAOptimizer (noa_optimizer.py) --
    provides the path-specific cost function and hybrid initialization,
    delegates the actual DHA/CDL-LOBL/SQP search loop to the shared core so
    this planner and the CEC2017 benchmark runner provably use the exact
    same algorithm."""

    def __init__(self, start, goal, env, num_intermediate=NUM_WAYPOINTS - 2,
                 pop_size=POP_SIZE, rng=None):
        from .noa_optimizer import NOAOptimizer, NOAOptimizerConfig

        self.start = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.env = env
        self.num_intermediate = num_intermediate
        self.dim = num_intermediate * 3
        self.pop_size = pop_size
        self.rng = rng or np.random.default_rng(0)

        self.lo, self.hi = decision_vector_bounds(self.start, self.goal, env)
        lo_flat = np.tile(self.lo, num_intermediate)
        hi_flat = np.tile(self.hi, num_intermediate)

        config = NOAOptimizerConfig(
            pop_size=pop_size, max_iters=MAX_ITERS,
            ema_alpha_base=EMA_ALPHA_BASE, ema_c_alpha=EMA_C_ALPHA,
            temp_cooling_rate=TEMP_COOLING_RATE, temp_c1=TEMP_C1, temp_c2=TEMP_C2,
            temp_gamma=TEMP_GAMMA, elite_ratio=ELITE_RATIO,
            tent_alpha=TENT_ALPHA, lobl_k_max=LOBL_K_MAX, lobl_k_min=LOBL_K_MIN,
            chaos_sigma=CHAOS_SIGMA, stagnation_rounds_for_lobl=STAGNATION_ROUNDS_FOR_LOBL,
            sqp_stagnation_threshold=SQP_STAGNATION_THRESHOLD,
            sqp_elite_count=SQP_ELITE_COUNT, sqp_max_iters=SQP_MAX_ITERS)

        self._optimizer = NOAOptimizer(
            cost_fn=self._cost, lo=lo_flat, hi=hi_flat, dim=self.dim,
            config=config, rng=self.rng, init_population=self._hybrid_init_fn)

    def _cost(self, x):
        return path_cost(x, self.start, self.goal, self.num_intermediate, self.env)

    def _hybrid_init_fn(self, rng, pop_size, dim, lo, hi):
        """Straight-line-guided initialization with smooth perturbation --
        simplified from the paper's A*-guided-path + Perlin-noise scheme
        (full A* on a dense voxel grid for every init individual is
        expensive; a straight-line baseline + smooth low-frequency
        perturbation gives comparable diversity-with-structure at much
        lower cost, appropriate for the real-time-oriented framing here)."""
        pop = np.zeros((pop_size, dim))
        base_points = np.linspace(0, 1, self.num_intermediate + 2)[1:-1]
        baseline = np.array([self.start + t * (self.goal - self.start) for t in base_points])

        for i in range(pop_size):
            noise_scale = rng.uniform(0.3, 1.0) * np.linalg.norm(self.hi - self.lo) * 0.15
            perturb = rng.normal(0, noise_scale, size=(self.num_intermediate, 3))
            candidate = baseline + perturb
            candidate = np.clip(candidate, self.lo, self.hi)
            pop[i] = candidate.flatten()
        return pop

    def run(self, max_iters=MAX_ITERS, verbose=True):
        return self._optimizer.run(max_iters=max_iters, verbose=verbose, verbose_every=20)



# ---------------------------------------------------------------------------
# Post-processing: cubic spline smoothing + feasibility verification/rollback
# (paper Section 4.2 / 4.2.1, simplified to 3D directly rather than the
# paper's XY-then-arc-length-elevation two-pass scheme)
# ---------------------------------------------------------------------------

def _max_penetration(path, env, n_samples=None):
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    if n_samples is None:
        n_samples = max(30, int(length / COLLISION_CHECK_SPACING))
    pts = _resample_polyline(path, n_samples)
    max_pen = 0.0
    for pt in pts:
        p_occ = env.occupancy_at(pt)
        if p_occ > OCCUPANCY_COLLISION_THRESHOLD:
            max_pen = max(max_pen, p_occ - OCCUPANCY_COLLISION_THRESHOLD)
    return max_pen


def smooth_and_verify(path, env, n_samples=None):
    from scipy.interpolate import CubicSpline

    t = np.linspace(0, 1, len(path))
    n_spline_pts = n_samples or max(60, int(
        np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) / COLLISION_CHECK_SPACING))
    t_dense = np.linspace(0, 1, n_spline_pts)
    splines = [CubicSpline(t, path[:, dim]) for dim in range(3)]
    smoothed = np.stack([s(t_dense) for s in splines], axis=1)

    smoothed_penetration = _max_penetration(smoothed, env)
    if smoothed_penetration <= 0.05:
        return smoothed, True

    # smoothing collided -- check whether the RAW waypoint path is actually
    # safe before falling back to it (don't just assume it is: the
    # optimizer's own collision check could have missed a thin obstacle too
    # if this function is ever called on an unverified path)
    raw_penetration = _max_penetration(path, env)
    if raw_penetration <= 0.05:
        print(f"[path_planning] smoothing reintroduced a collision "
              f"(penetration={smoothed_penetration:.3f}) -- rolling back to "
              f"the unsmoothed path, which IS verified collision-free.")
        return path, False

    print(f"[path_planning] WARNING: neither the smoothed path "
          f"(penetration={smoothed_penetration:.3f}) nor the raw waypoint "
          f"path (penetration={raw_penetration:.3f}) is collision-free. "
          f"The optimizer did not find a safe route -- do not use this path "
          f"as-is. Consider more iterations, a larger population, or "
          f"checking whether a feasible route exists at all given the "
          f"obstacle layout.")
    return path, False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def plan_path(start, goal, env, pop_size=POP_SIZE, max_iters=MAX_ITERS, seed=0):
    rng = np.random.default_rng(seed)
    planner = NOAPlanner(start, goal, env, pop_size=pop_size, rng=rng)
    best_vec, best_cost = planner.run(max_iters=max_iters)
    raw_path = build_full_path(best_vec, planner.start, planner.goal, planner.num_intermediate)
    smoothed_path, was_smoothed = smooth_and_verify(raw_path, env)
    is_collision_free = _max_penetration(smoothed_path, env) <= 0.05
    if not is_collision_free:
        print("[path_planning] RESULT IS NOT COLLISION-FREE -- check "
              "is_collision_free before using this path.")
    return {
        "raw_waypoints": raw_path,
        "final_path": smoothed_path,
        "was_smoothed": was_smoothed,
        "is_collision_free": is_collision_free,
        "cost": float(best_cost),
    }


def plan_cbba_routes(run_dir, pop_size=POP_SIZE, max_iters=MAX_ITERS):
    cbba_path = os.path.join(run_dir, "task_allocation", "cbba_assignment.pkl")
    if not os.path.exists(cbba_path):
        raise FileNotFoundError(f"No {cbba_path} -- run cbba_task_allocation.py first.")
    with open(cbba_path, "rb") as f:
        cbba = pickle.load(f)

    out_dir = os.path.join(run_dir, "path_planning")
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    for agent_name, info in cbba["agents"].items():
        env = PlanningEnvironment(run_dir, agent_name=agent_name)
        legs = [info["start_position"]] + [t["position"] for t in info["assigned_tasks"]]
        if len(legs) < 2:
            print(f"[path_planning] {agent_name}: no assigned tasks, skipping.")
            continue

        print(f"\n{'=' * 60}\n{agent_name}: planning {len(legs) - 1} leg(s)\n{'=' * 60}")
        agent_legs = []
        for i in range(len(legs) - 1):
            print(f"  leg {i + 1}/{len(legs) - 1}: {np.round(legs[i], 2)} -> {np.round(legs[i + 1], 2)}")
            result = plan_path(legs[i], legs[i + 1], env, pop_size=pop_size, max_iters=max_iters, seed=i)
            agent_legs.append(result)

        all_results[agent_name] = agent_legs
        with open(os.path.join(out_dir, f"{agent_name}_paths.pkl"), "wb") as f:
            pickle.dump(agent_legs, f)

    _plot_routes(all_results, out_dir)
    print(f"\n[path_planning] all agent routes saved -> {out_dir}")
    return all_results


def _plot_routes(all_results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10.colors
    for i, (agent_name, legs) in enumerate(all_results.items()):
        color = colors[i % len(colors)]
        for leg in legs:
            path = leg["final_path"]
            ax.plot(path[:, 0], path[:, 1], "-", color=color, linewidth=1.5)
        if legs:
            ax.scatter(*legs[0]["final_path"][0, :2], marker="^", s=100,
                       color=color, edgecolors="black", label=agent_name, zorder=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Planned routes (risk-aware NOA-SQP-derived planner)")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    plot_path = os.path.join(out_dir, "planned_routes.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[path_planning] route plot saved -> {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--start", type=str, default=None, help="x,y,z")
    parser.add_argument("--goal", type=str, default=None, help="x,y,z")
    parser.add_argument("--plan_cbba_routes", action="store_true",
                         help="Plan every leg of every agent's actual CBBA-assigned "
                              "route instead of a single explicit start/goal pair.")
    parser.add_argument("--pop_size", type=int, default=POP_SIZE)
    parser.add_argument("--max_iters", type=int, default=MAX_ITERS)
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    print(f"Using run_dir: {run_dir}")

    if args.plan_cbba_routes:
        plan_cbba_routes(run_dir, pop_size=args.pop_size, max_iters=args.max_iters)
    else:
        if not args.start or not args.goal:
            raise ValueError("Provide --start and --goal, or use --plan_cbba_routes.")
        start = np.array([float(v) for v in args.start.split(",")])
        goal = np.array([float(v) for v in args.goal.split(",")])
        env = PlanningEnvironment(run_dir)
        result = plan_path(start, goal, env, pop_size=args.pop_size, max_iters=args.max_iters)
        print(f"\nFinal cost: {result['cost']:.3f}, smoothed: {result['was_smoothed']}")
        out_dir = os.path.join(run_dir, "path_planning")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "single_path.pkl"), "wb") as f:
            pickle.dump(result, f)
        print(f"Saved -> {out_dir}/single_path.pkl")
