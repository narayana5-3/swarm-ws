"""
Shared damage-probability map + uncertainty heatmap.

Builds on occupancy_mapping.py's voxel grid and distributed delta-compressed
fusion mechanism, but tracks a SEPARATE "damage" log-odds grid derived from
each frame's AI crack-detection output (damage_detection/infer.py's
damage_inference.pkl), rather than raw sonar occupancy. Fused across the
swarm the same acoustic-mesh delta-compressed way as Phase 2.

Two outputs:
  - damage_probability_map: per-voxel P(damage present), fused across agents.
  - uncertainty_heatmap: per-voxel uncertainty, based on OBSERVATION COUNT
    (not just cross-agent disagreement). Voxels swept many times by sonar/
    camera have low uncertainty; voxels rarely or never observed have high
    uncertainty. This is the actionable signal for the next phase (CBBA
    task reallocation): agents should be drawn toward high-uncertainty,
    non-trivial-damage-probability regions -- a direct, literal
    implementation of "risk-prioritized" exploration.

Design notes / simplifications (documented, not hidden):
  - Cameras don't give per-pixel depth/range the way sonar does, so exact
    pixel-to-voxel projection isn't available. Approximation used here:
    map each camera image COLUMN to the nearest sonar AZIMUTH ray at the
    same tick+pose (both are forward-facing, roughly co-located), using
    the max damage probability in that column as the value for the voxel
    that sonar ray identified. This is a coarse but documented and
    reasonable first-pass association; a calibrated camera/sonar extrinsic
    + per-pixel depth estimate would improve precision later.
  - Damage log-odds update per observation is a standard Bayesian-style
    update: logit(reported probability), clamped, same clamping/threshold
    machinery as the occupancy grid for consistency.
  - Observation count is tracked per voxel per agent, then summed across
    the swarm during fusion (via the same delta-broadcast channel) --
    giving a genuinely swarm-wide coverage measure, not just a per-agent one.

Run (after occupancy_mapping.py AND damage_detection/infer.py have both
been run for the same log run):
    python damage_probability_mapping.py --run_dir logs/run_<ts> \
        --damage_inference_dir damage_detection/damage_inference
"""

import os
import glob
import pickle
import argparse
import numpy as np

from occupancy_mapping.occupancy_mapping import (VoxelGrid, rotation_matrix_rpy, ENV_MIN, ENV_MAX,
                                VOXEL_SIZE, SONAR_AZIMUTH_FOV_DEG, SONAR_RANGE_MIN,
                                SONAR_RANGE_MAX, LOG_ODDS_CLAMP,
                                BROADCAST_INTERVAL_TICKS, ACOUSTIC_MAX_RANGE,
                                DELTA_THRESHOLD, find_latest_run_dir, atomic_save_npy)

CAMERA_FOV_DEG = 90.0   # HoloOcean RGBCamera default FOV assumption -- verify
                        # against your actual camera config if precise
                        # camera/sonar alignment matters later.

DAMAGE_LOG_ODDS_SCALE = 2.0   # scales how strongly a single detection shifts
                               # belief; tune based on classifier confidence.
UNCERTAINTY_TAU = 3.0         # decay constant for count -> uncertainty mapping
STRUCTURE_PROXIMITY_VOXELS = 6  # ~3m at 0.5m voxels -- how far from detected
                                  # structure to still count risk (avoids
                                  # diluting the map with open-water baseline)


# ---------------------------------------------------------------------------
# Per-agent damage mapper (parallel structure to occupancy_mapping's
# AgentLocalMapper, but for damage probability + observation count)
# ---------------------------------------------------------------------------

class AgentDamageMapper:
    def __init__(self, name):
        self.name = name
        self.damage_grid = VoxelGrid(env_min=ENV_MIN, env_max=ENV_MAX, voxel_size=VOXEL_SIZE)
        self.count_grid = np.zeros(tuple(self.damage_grid.dims), dtype=np.int32)
        self.last_shared_damage = {}
        self.last_shared_count = {}
        self.received_damage = {}   # peer_name -> {voxel: log_odds}
        self.received_count = {}    # peer_name -> {voxel: count}

    def process_frame(self, location, rotation, sonar_array, damage_prob_map):
        """
        location/rotation: this agent's pose at the tick the frame was
        captured. sonar_array: same-tick sonar frame (for ray geometry).
        damage_prob_map: HxW array from infer.py's output for this tick.
        """
        if sonar_array is None or damage_prob_map is None:
            return

        sonar_array = np.asarray(sonar_array)
        if sonar_array.ndim != 2:
            return
        n_range_bins, n_azimuth_bins = sonar_array.shape

        damage_prob_map = np.asarray(damage_prob_map)
        img_h, img_w = damage_prob_map.shape[:2]

        R = rotation_matrix_rpy(*rotation)
        sensor_pos = np.array(location, dtype=np.float64)

        ranges = np.linspace(SONAR_RANGE_MIN, SONAR_RANGE_MAX, n_range_bins)
        sonar_azimuths = np.linspace(-SONAR_AZIMUTH_FOV_DEG / 2, SONAR_AZIMUTH_FOV_DEG / 2,
                                      n_azimuth_bins)

        col_max = sonar_array.max(axis=0, keepdims=True)
        col_max[col_max < 1e-6] = 1.0
        norm = sonar_array / col_max

        # max damage probability per camera image column
        col_damage = damage_prob_map.max(axis=0) if damage_prob_map.ndim == 2 \
            else damage_prob_map.max(axis=(0, 2))

        for az_col in range(n_azimuth_bins):
            az_deg = sonar_azimuths[az_col]
            if abs(az_deg) > CAMERA_FOV_DEG / 2:
                continue  # outside camera's field of view, no damage info available

            # map this sonar azimuth to the corresponding camera image column
            frac = (az_deg + CAMERA_FOV_DEG / 2) / CAMERA_FOV_DEG
            img_col = int(np.clip(frac * img_w, 0, img_w - 1))
            damage_p = float(col_damage[img_col])

            above = np.where(norm[:, az_col] > 0.3)[0]
            if len(above) == 0:
                continue  # no sonar-confirmed surface at this azimuth, skip
            r_hit = ranges[above[0]]
            az_rad = np.radians(az_deg)
            hit_local = np.array([r_hit * np.cos(az_rad), r_hit * np.sin(az_rad), 0.0])
            hit_world = sensor_pos + R @ hit_local

            idx = self.damage_grid.world_to_index(hit_world)
            if not self.damage_grid.in_bounds(idx):
                continue

            damage_p = np.clip(damage_p, 1e-4, 1 - 1e-4)
            log_odds_update = DAMAGE_LOG_ODDS_SCALE * np.log(damage_p / (1 - damage_p))
            self.damage_grid.add_log_odds(idx, log_odds_update)
            i, j, k = idx
            self.count_grid[i, j, k] += 1

    def compute_delta(self):
        damage_delta = {}
        nz = np.argwhere(np.abs(self.damage_grid.log_odds) > 1e-6)
        for idx in nz:
            key = tuple(idx)
            current = float(self.damage_grid.log_odds[key])
            previous = self.last_shared_damage.get(key, 0.0)
            if abs(current - previous) >= DELTA_THRESHOLD:
                damage_delta[key] = current

        count_delta = {}
        nz_count = np.argwhere(self.count_grid > 0)
        for idx in nz_count:
            key = tuple(idx)
            current = int(self.count_grid[key])
            previous = self.last_shared_count.get(key, 0)
            if current != previous:
                count_delta[key] = current

        return damage_delta, count_delta

    def commit_broadcast(self, damage_delta, count_delta):
        self.last_shared_damage.update(damage_delta)
        self.last_shared_count.update(count_delta)

    def receive(self, sender_name, damage_delta, count_delta):
        self.received_damage.setdefault(sender_name, {}).update(damage_delta)
        self.received_count.setdefault(sender_name, {}).update(count_delta)

    def fused_damage_log_odds(self):
        fused = self.damage_grid.log_odds.copy()
        for peer_map in self.received_damage.values():
            for (i, j, k), val in peer_map.items():
                fused[i, j, k] = np.clip(fused[i, j, k] + val, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)
        return fused

    def fused_observation_count(self):
        """Swarm-wide observation count: own count + latest known count from
        every peer (peers' counts are their own local totals, taken as
        latest-known snapshots -- consistent with the channel-filter scheme
        used for log-odds, avoids double counting across re-broadcasts)."""
        fused = self.count_grid.copy().astype(np.int64)
        for peer_map in self.received_count.values():
            for (i, j, k), val in peer_map.items():
                fused[i, j, k] = max(fused[i, j, k], val)  # take max, not sum,
                # since peer's reported count already includes their own history
        return fused


# ---------------------------------------------------------------------------
# Distributed fusion driver
# ---------------------------------------------------------------------------

def run_damage_fusion(run_dir, damage_inference_dir):
    log_files = sorted(glob.glob(os.path.join(run_dir, "*_sensor_log.pkl")))
    if not log_files:
        raise FileNotFoundError(f"No *_sensor_log.pkl files found in {run_dir}")

    agent_names = [os.path.basename(p).replace("_sensor_log.pkl", "") for p in log_files]

    sensor_logs = {}
    damage_logs = {}
    for name, path in zip(agent_names, log_files):
        with open(path, "rb") as f:
            sensor_logs[name] = pickle.load(f)

        damage_path = os.path.join(damage_inference_dir, name, "damage_inference.pkl")
        if os.path.exists(damage_path):
            with open(damage_path, "rb") as f:
                damage_records = pickle.load(f)
            # index by tick for fast lookup during replay
            damage_logs[name] = {r["tick"]: r for r in damage_records}
        else:
            print(f"[damage_fusion] WARNING: no damage_inference.pkl found for "
                  f"{name} at {damage_path} -- run infer.py for this agent first. "
                  f"Skipping this agent's damage contribution.")
            damage_logs[name] = {}

    mappers = {name: AgentDamageMapper(name) for name in agent_names}
    max_len = max(len(records) for records in sensor_logs.values())

    print(f"Replaying {max_len} ticks across {len(agent_names)} agents for "
          f"damage-probability fusion...")

    for t in range(max_len):
        positions = {}
        for name in agent_names:
            records = sensor_logs[name]
            if t >= len(records):
                continue
            rec = records[t]
            tick = rec.get("tick")
            loc, rot, sonar = rec.get("location"), rec.get("rotation"), rec.get("sonar")
            if loc is not None:
                positions[name] = np.array(loc)

            damage_rec = damage_logs[name].get(tick)
            if damage_rec is not None and loc is not None and rot is not None:
                mappers[name].process_frame(loc, rot, sonar, damage_rec["damage_prob_map"])

        if t > 0 and t % BROADCAST_INTERVAL_TICKS == 0:
            deltas = {name: mappers[name].compute_delta() for name in agent_names}
            for sender in agent_names:
                damage_delta, count_delta = deltas[sender]
                if not damage_delta and not count_delta:
                    continue
                for receiver in agent_names:
                    if receiver == sender:
                        continue
                    if sender not in positions or receiver not in positions:
                        continue
                    dist = np.linalg.norm(positions[sender] - positions[receiver])
                    if dist <= ACOUSTIC_MAX_RANGE:
                        mappers[receiver].receive(sender, damage_delta, count_delta)
                mappers[sender].commit_broadcast(damage_delta, count_delta)

    print("Damage-probability replay complete.")
    return mappers, agent_names


# ---------------------------------------------------------------------------
# Derive final maps + visualize
# ---------------------------------------------------------------------------

def count_to_uncertainty(count_grid, tau=UNCERTAINTY_TAU):
    """High uncertainty where observation count is low/zero, decaying toward
    0 as count grows. Simple, standard proxy for epistemic/coverage
    uncertainty (used widely in active-SLAM / next-best-view planning)."""
    return np.exp(-count_grid.astype(np.float64) / tau)


def evaluate_and_save(mappers, agent_names, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_ref = mappers[agent_names[0]].damage_grid
    damage_probs = {name: 1.0 / (1.0 + np.exp(-mappers[name].fused_damage_log_odds()))
                     for name in agent_names}
    uncertainty = {name: count_to_uncertainty(mappers[name].fused_observation_count())
                   for name in agent_names}

    out_dir = os.path.join(run_dir, "damage_probability")
    os.makedirs(out_dir, exist_ok=True)
    for name in agent_names:
        atomic_save_npy(os.path.join(out_dir, f"{name}_damage_prob.npy"), damage_probs[name])
        atomic_save_npy(os.path.join(out_dir, f"{name}_uncertainty.npy"), uncertainty[name])
    print(f"[damage_probability] fused grids saved -> {out_dir}")

    fig, axes = plt.subplots(2, len(agent_names), figsize=(5 * len(agent_names), 10))
    if len(agent_names) == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(agent_names):
        top_damage = np.max(damage_probs[name], axis=2).T
        im0 = axes[0, col].imshow(top_damage, origin="lower", cmap="inferno", vmin=0, vmax=1,
                                   extent=[grid_ref.env_min[0], grid_ref.env_max[0],
                                           grid_ref.env_min[1], grid_ref.env_max[1]])
        axes[0, col].set_title(f"{name} damage probability")
        axes[0, col].set_xlabel("x (m)")
        axes[0, col].set_ylabel("y (m)")

        top_uncertainty = np.max(uncertainty[name], axis=2).T
        im1 = axes[1, col].imshow(top_uncertainty, origin="lower", cmap="cividis", vmin=0, vmax=1,
                                   extent=[grid_ref.env_min[0], grid_ref.env_max[0],
                                           grid_ref.env_min[1], grid_ref.env_max[1]])
        axes[1, col].set_title(f"{name} uncertainty (unswept=bright)")
        axes[1, col].set_xlabel("x (m)")
        axes[1, col].set_ylabel("y (m)")

    fig.colorbar(im0, ax=axes[0, :], shrink=0.8, label="P(damage)")
    fig.colorbar(im1, ax=axes[1, :], shrink=0.8, label="uncertainty")
    plot_path = os.path.join(out_dir, "damage_and_uncertainty_topdown.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[damage_probability] plot saved -> {plot_path}")
    plt.close(fig)

    # Risk score = damage_probability * uncertainty -- directly usable as the
    # CBBA task-reallocation priority signal in the next phase.
    #
    # IMPORTANT: raw damage_prob * uncertainty gives every UNOBSERVED voxel a
    # uniform baseline risk (~0.5 * 1.0 = 0.5), including huge stretches of
    # open water nowhere near the actual structure -- diluting the signal
    # from real detections. Masked here to only count risk within
    # STRUCTURE_PROXIMITY_VOXELS of a voxel the occupancy grid actually
    # thinks is occupied, so the risk map reflects "near the dam/pipe and
    # under-inspected or damaged," not "anywhere in open water we haven't
    # been."
    risk = {}
    for name in agent_names:
        occ_path = os.path.join(run_dir, "occupancy", f"{name}_fused_logodds.npy")
        if os.path.exists(occ_path):
            occ_log_odds = np.load(occ_path)
            occ_prob = 1.0 / (1.0 + np.exp(-occ_log_odds))
            structure_mask = _dilate_mask(occ_prob > 0.6, STRUCTURE_PROXIMITY_VOXELS)
            risk[name] = damage_probs[name] * uncertainty[name] * structure_mask
        else:
            print(f"[damage_probability] WARNING: no occupancy grid found at "
                  f"{occ_path} -- run occupancy_mapping.py first for a masked, "
                  f"meaningful risk score. Falling back to unmasked (diluted) risk.")
            risk[name] = damage_probs[name] * uncertainty[name]

        atomic_save_npy(os.path.join(out_dir, f"{name}_risk_score.npy"), risk[name])
    print(f"[damage_probability] risk_score grids (damage_prob * uncertainty, "
          f"masked to near detected structure) saved -> {out_dir} "
          f"(ready for CBBA task reallocation)")


def _dilate_mask(binary_mask, n_voxels):
    """Simple binary dilation by n_voxels using scipy if available, else a
    manual box-max fallback (avoids a hard scipy dependency)."""
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(binary_mask, iterations=n_voxels)
    except ImportError:
        result = binary_mask.copy()
        for _ in range(n_voxels):
            padded = np.pad(result, 1, mode="constant", constant_values=False)
            result = (padded[:-2, 1:-1, 1:-1] | padded[2:, 1:-1, 1:-1] |
                      padded[1:-1, :-2, 1:-1] | padded[1:-1, 2:, 1:-1] |
                      padded[1:-1, 1:-1, :-2] | padded[1:-1, 1:-1, 2:] | result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None,
                         help="Path to a logs/run_<ts> directory. Defaults to most recent.")
    parser.add_argument("--damage_inference_dir", type=str,
                         default="damage_detection/damage_inference",
                         help="Directory containing <agent>/damage_inference.pkl "
                              "outputs from damage_detection/infer.py.")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    print(f"Using run_dir: {run_dir}")

    mappers, agent_names = run_damage_fusion(run_dir, args.damage_inference_dir)
    evaluate_and_save(mappers, agent_names, run_dir)
