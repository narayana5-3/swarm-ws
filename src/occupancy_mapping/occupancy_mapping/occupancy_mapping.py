"""
Phase 2: Shared occupancy map via simulated distributed belief fusion.

Each agent maintains its OWN local log-odds occupancy grid, built from its
own sonar returns + pose. Periodically (simulating a bandwidth-limited
acoustic channel), each agent broadcasts only the voxels whose log-odds have
changed meaningfully since its last broadcast -- a delta/compressed update,
not the full grid. Other agents within acoustic range receive it and fold it
into their view of the shared map.

This directly operationalizes the "compressed acoustic-mesh belief sharing"
layer of the swarm architecture, and gives you real metrics for the paper:
  - map agreement across agents over time (convergence)
  - communication volume: compressed delta size vs. full-grid-every-broadcast

Design notes / simplifications (documented, not hidden):
  - Occupancy grid is occupied/free geometry ONLY. Sonar intensity is NOT
    folded in here -- that's reserved for the Phase 3 AI damage-detection
    layer, which produces a separate damage-probability map on top of this
    completed structural map. Keeping these layers separate matches your
    project's own checklist (occupancy map vs. damage probability map vs.
    uncertainty heatmap as three distinct deliverables).
  - Ray casting is done in a single horizontal plane through the sonar's
    azimuth sweep (ignoring the sonar's ~20 degree elevation spread). Good
    enough for a first structural map; extend to full elevation binning
    later if you need vertical resolution on the damage layer.
  - Uses each agent's own local sonar frame's first above-threshold return
    per azimuth column as "the hit." Multiple returns (multipath) are
    ignored at this stage.
  - Fusion avoids double-counting via a channel-filter-style scheme: each
    agent tracks the LATEST known log-odds contribution from every other
    agent (not an accumulating stream), and its fused view is the sum of
    its own log-odds plus the latest known contribution from each peer.

Run (after swarm_sim.py has produced a logs/run_<ts>/ directory):
    python occupancy_mapping.py --run_dir logs/run_<ts>
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np


def atomic_save_npy(path, array):
    """np.save() writes in place and isn't atomic -- fine for the original
    offline batch scripts (write once, read once, both after the process
    exits), but the live ROS2 pipeline has readers (cbba_allocator_node,
    swarm_control) polling these exact files while occupancy_mapping_node/
    belief_fusion_node periodically re-snapshot them, and a reader can catch
    a partially-written file mid-save. Confirmed live: cbba_allocator_node
    crashed with "cannot reshape array of size 5586928 into shape
    (200,200,140)" -- a torn read of a file belief_fusion_node was still
    writing. Fixed by writing to a temp file in the same directory (so the
    following os.replace is on the same filesystem, hence atomic on POSIX)
    and renaming into place -- a reader always sees either the complete old
    file or the complete new one, never a partial write."""
    tmp_path = path + f".tmp{os.getpid()}"
    np.save(tmp_path, array)
    # np.save appends .npy if the given path doesn't already end with it
    if not tmp_path.endswith(".npy"):
        tmp_path += ".npy"
    os.replace(tmp_path, path)

# ---------------------------------------------------------------------------
# Config -- keep in sync with swarm_sim.py's ENV_MIN/ENV_MAX/OCTREE_MIN if you
# change those; the map only covers the region the swarm actually flew in.
# ---------------------------------------------------------------------------

ENV_MIN = np.array([-50, -50, -50])
ENV_MAX = np.array([50, 50, 20])
VOXEL_SIZE = 0.5  # meters/voxel. Coarser than sonar octree_min (0.25) is fine
                   # for a structural occupancy map; refine later if needed.

SONAR_INTENSITY_THRESHOLD = 0.3   # normalize-and-threshold to call a bin "a hit"
SONAR_AZIMUTH_FOV_DEG = 120       # must match swarm_sim.py's sonar config
SONAR_RANGE_MIN = 1
SONAR_RANGE_MAX = 40               # must match swarm_sim.py's SONAR_RANGE_MAX

LOG_ODDS_FREE = -0.4
LOG_ODDS_OCC = 0.85
LOG_ODDS_CLAMP = 10.0

BROADCAST_INTERVAL_TICKS = 30     # how often agents exchange compressed updates
ACOUSTIC_MAX_RANGE = 60.0         # must match AcousticBeaconSensor MaxDistance
DELTA_THRESHOLD = 0.05            # min log-odds change to be worth broadcasting


# ---------------------------------------------------------------------------
# Voxel grid
# ---------------------------------------------------------------------------

class VoxelGrid:
    """Dense log-odds occupancy grid over a fixed world-space bounding box."""

    def __init__(self, env_min=ENV_MIN, env_max=ENV_MAX, voxel_size=VOXEL_SIZE):
        self.env_min = np.array(env_min, dtype=np.float64)
        self.env_max = np.array(env_max, dtype=np.float64)
        self.voxel_size = voxel_size
        self.dims = np.ceil((self.env_max - self.env_min) / voxel_size).astype(int)
        self.log_odds = np.zeros(tuple(self.dims), dtype=np.float32)

    def world_to_index(self, point):
        idx = np.floor((np.array(point) - self.env_min) / self.voxel_size).astype(int)
        return idx

    def in_bounds(self, idx):
        return np.all(idx >= 0) and np.all(idx < self.dims)

    def add_log_odds(self, idx, delta):
        if self.in_bounds(idx):
            i, j, k = idx
            self.log_odds[i, j, k] = np.clip(
                self.log_odds[i, j, k] + delta, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)

    def probability_grid(self):
        return 1.0 / (1.0 + np.exp(-self.log_odds))


# ---------------------------------------------------------------------------
# Ray casting: mark free space along the ray, occupied at the hit
# ---------------------------------------------------------------------------

def trace_ray_update(grid, start_world, end_world, hit=True):
    """
    Simple fixed-step ray march (not a true DDA, but plenty accurate at this
    voxel resolution and much simpler to read/debug/extend).
    """
    start = np.array(start_world, dtype=np.float64)
    end = np.array(end_world, dtype=np.float64)
    length = np.linalg.norm(end - start)
    if length < 1e-6:
        return
    direction = (end - start) / length
    step = grid.voxel_size * 0.5
    n_steps = max(1, int(length / step))

    for s in range(n_steps):
        point = start + direction * (s * step)
        idx = grid.world_to_index(point)
        grid.add_log_odds(idx, LOG_ODDS_FREE)

    if hit:
        idx = grid.world_to_index(end)
        grid.add_log_odds(idx, LOG_ODDS_OCC)


# ---------------------------------------------------------------------------
# Sonar frame -> world-space hits
# ---------------------------------------------------------------------------

def rotation_matrix_rpy(roll_deg, pitch_deg, yaw_deg):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matching HoloOcean's documented
    fixed-axis XYZ rotation convention."""
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def sonar_hits_world(sonar_array, location, rotation):
    """
    sonar_array: 2D array, assumed [range_bin, azimuth_bin], range increasing
    with row index from SONAR_RANGE_MIN to SONAR_RANGE_MAX, azimuth spanning
    -FOV/2 .. +FOV/2 across columns. This orientation is an assumption based
    on HoloOcean's documented config -- verify against your build's actual
    output (see verify_sonar_shape() below) and adjust the transpose/flip
    here if it doesn't match.

    Returns list of (sensor_world_pos, hit_world_pos_or_None) per azimuth ray,
    in the FLU body frame projected into a single horizontal plane (elevation
    ignored -- see module docstring).
    """
    if sonar_array is None:
        return []

    sonar_array = np.asarray(sonar_array)
    if sonar_array.ndim != 2:
        return []

    n_range_bins, n_azimuth_bins = sonar_array.shape
    ranges = np.linspace(SONAR_RANGE_MIN, SONAR_RANGE_MAX, n_range_bins)
    azimuths = np.linspace(-SONAR_AZIMUTH_FOV_DEG / 2, SONAR_AZIMUTH_FOV_DEG / 2,
                            n_azimuth_bins)

    R = rotation_matrix_rpy(*rotation)
    sensor_pos = np.array(location, dtype=np.float64)

    # normalize per-column for a rough intensity-independent threshold
    col_max = sonar_array.max(axis=0, keepdims=True)
    col_max[col_max < 1e-6] = 1.0
    norm = sonar_array / col_max

    results = []
    for col in range(n_azimuth_bins):
        above = np.where(norm[:, col] > SONAR_INTENSITY_THRESHOLD)[0]
        az_rad = np.radians(azimuths[col])
        far_point_local = np.array([SONAR_RANGE_MAX * np.cos(az_rad),
                                     SONAR_RANGE_MAX * np.sin(az_rad), 0.0])
        far_point_world = sensor_pos + R @ far_point_local

        if len(above) == 0:
            # no return -> treat whole ray as free, no occupied hit
            results.append((sensor_pos, far_point_world, False))
            continue

        r_hit = ranges[above[0]]
        hit_local = np.array([r_hit * np.cos(az_rad), r_hit * np.sin(az_rad), 0.0])
        hit_world = sensor_pos + R @ hit_local
        results.append((sensor_pos, hit_world, True))

    return results


def verify_sonar_shape(run_dir, agent_name="auv0"):
    """
    Scans the WHOLE log (not just the first frame) to find real sonar
    captures. HoloOcean always returns a valid-shaped array even before the
    sensor's first actual capture (Hz-gated), so checking only the first
    non-None frame can report an uninitialized all-zero buffer and look
    like "no returns" even when later frames have real data.
    """
    path = os.path.join(run_dir, f"{agent_name}_sensor_log.pkl")
    with open(path, "rb") as f:
        records = pickle.load(f)

    best_max = -1.0
    best_tick = None
    best_shape = None
    n_nonzero_frames = 0
    n_total_frames = 0

    for r in records:
        arr = r.get("sonar")
        if arr is None:
            continue
        arr = np.asarray(arr)
        n_total_frames += 1
        frame_max = float(arr.max())
        if frame_max > 1e-6:
            n_nonzero_frames += 1
        if frame_max > best_max:
            best_max = frame_max
            best_tick = r.get("tick")
            best_shape = arr.shape

    print(f"[verify_sonar_shape] scanned {n_total_frames} sonar frames, "
          f"{n_nonzero_frames} had nonzero content")
    if best_shape is not None:
        print(f"[verify_sonar_shape] best frame: tick={best_tick}, "
              f"shape={best_shape}, max_intensity={best_max:.4f}")
    if n_nonzero_frames == 0:
        print("[verify_sonar_shape] EVERY frame in the entire run was zero -- "
              "this is a real no-returns condition, not a first-frame artifact.")
    return best_shape


# ---------------------------------------------------------------------------
# Per-agent local mapper
# ---------------------------------------------------------------------------

class AgentLocalMapper:
    def __init__(self, name):
        self.name = name
        self.own_grid = VoxelGrid()
        # last_shared: voxel index (tuple) -> log-odds value at last broadcast
        self.last_shared = {}
        # received: other_agent_name -> {voxel index tuple: log-odds value}
        self.received = {}

    def process_record(self, record):
        loc = record.get("location")
        rot = record.get("rotation")
        sonar = record.get("sonar")
        if loc is None or rot is None or sonar is None:
            return
        for sensor_pos, end_pos, is_hit in sonar_hits_world(sonar, loc, rot):
            trace_ray_update(self.own_grid, sensor_pos, end_pos, hit=is_hit)

    def compute_delta(self):
        """Voxels whose log-odds changed >= DELTA_THRESHOLD since last shared."""
        delta = {}
        nz = np.argwhere(np.abs(self.own_grid.log_odds) > 1e-6)
        for idx in nz:
            key = tuple(idx)
            current = float(self.own_grid.log_odds[key])
            previous = self.last_shared.get(key, 0.0)
            if abs(current - previous) >= DELTA_THRESHOLD:
                delta[key] = current
        return delta

    def commit_broadcast(self, delta):
        self.last_shared.update(delta)

    def receive(self, sender_name, delta):
        if sender_name not in self.received:
            self.received[sender_name] = {}
        self.received[sender_name].update(delta)

    def fused_log_odds(self):
        """Own belief + latest known contribution from every peer (channel-
        filter style: peer contributions are their latest snapshot, not an
        accumulating stream, so re-broadcasts don't double count)."""
        fused = self.own_grid.log_odds.copy()
        for _, peer_map in self.received.items():
            for (i, j, k), val in peer_map.items():
                fused[i, j, k] = np.clip(fused[i, j, k] + val, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)
        return fused


# ---------------------------------------------------------------------------
# Distributed fusion driver
# ---------------------------------------------------------------------------

def run_distributed_fusion(run_dir):
    log_files = sorted(glob.glob(os.path.join(run_dir, "*_sensor_log.pkl")))
    if not log_files:
        raise FileNotFoundError(f"No *_sensor_log.pkl files found in {run_dir}")

    agent_logs = {}
    for path in log_files:
        name = os.path.basename(path).replace("_sensor_log.pkl", "")
        with open(path, "rb") as f:
            agent_logs[name] = pickle.load(f)

    agent_names = list(agent_logs.keys())
    mappers = {name: AgentLocalMapper(name) for name in agent_names}

    max_len = max(len(records) for records in agent_logs.values())
    total_broadcast_voxels = 0
    total_full_grid_voxels = 0
    grid_voxel_count = mappers[agent_names[0]].own_grid.log_odds.size

    print(f"Replaying {max_len} ticks across {len(agent_names)} agents...")

    for t in range(max_len):
        positions = {}
        for name in agent_names:
            records = agent_logs[name]
            if t >= len(records):
                continue
            mappers[name].process_record(records[t])
            loc = records[t].get("location")
            if loc is not None:
                positions[name] = np.array(loc)

        if t > 0 and t % BROADCAST_INTERVAL_TICKS == 0:
            deltas = {name: mappers[name].compute_delta() for name in agent_names}

            for sender in agent_names:
                delta = deltas[sender]
                if not delta:
                    continue
                total_broadcast_voxels += len(delta)
                total_full_grid_voxels += grid_voxel_count

                for receiver in agent_names:
                    if receiver == sender:
                        continue
                    if sender not in positions or receiver not in positions:
                        continue
                    dist = np.linalg.norm(positions[sender] - positions[receiver])
                    if dist <= ACOUSTIC_MAX_RANGE:
                        mappers[receiver].receive(sender, delta)

                mappers[sender].commit_broadcast(delta)

    print(f"Replay complete.")
    if total_full_grid_voxels > 0:
        compression_ratio = 1.0 - (total_broadcast_voxels / total_full_grid_voxels)
        print(f"[metrics] total compressed voxels transmitted: {total_broadcast_voxels}")
        print(f"[metrics] equivalent full-grid-every-broadcast voxels: {total_full_grid_voxels}")
        print(f"[metrics] communication savings from delta compression: "
              f"{compression_ratio * 100:.1f}%")

    return mappers, agent_names


# ---------------------------------------------------------------------------
# Evaluation / visualization
# ---------------------------------------------------------------------------

def evaluate_and_save(mappers, agent_names, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_ref = mappers[agent_names[0]].own_grid
    fused_probs = {name: 1.0 / (1.0 + np.exp(-mappers[name].fused_log_odds()))
                   for name in agent_names}

    # Cross-agent map agreement: mean absolute difference in occupancy
    # probability between each pair of agents' fused views. Low = converged.
    print("[metrics] pairwise fused-map agreement (mean abs prob diff, lower=better):")
    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            a, b = agent_names[i], agent_names[j]
            diff = np.mean(np.abs(fused_probs[a] - fused_probs[b]))
            print(f"    {a} vs {b}: {diff:.4f}")

    # Save fused grids for the next phase (damage probability layering).
    out_dir = os.path.join(run_dir, "occupancy")
    os.makedirs(out_dir, exist_ok=True)
    for name in agent_names:
        atomic_save_npy(os.path.join(out_dir, f"{name}_fused_logodds.npy"),
                         mappers[name].fused_log_odds())
    print(f"[occupancy] fused log-odds grids saved -> {out_dir}")

    # Top-down (max over z) occupancy visualization per agent, side by side.
    fig, axes = plt.subplots(1, len(agent_names), figsize=(5 * len(agent_names), 5))
    if len(agent_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, agent_names):
        top_down = np.max(fused_probs[name], axis=2).T  # (y, x) for imshow
        im = ax.imshow(top_down, origin="lower", cmap="viridis", vmin=0, vmax=1,
                        extent=[grid_ref.env_min[0], grid_ref.env_max[0],
                                grid_ref.env_min[1], grid_ref.env_max[1]])
        ax.set_title(f"{name} fused occupancy (top-down)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="P(occupied)")
    plot_path = os.path.join(out_dir, "fused_occupancy_topdown.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[occupancy] top-down occupancy plot saved -> {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------

def find_latest_run_dir(logs_root="logs"):
    runs = sorted(glob.glob(os.path.join(logs_root, "run_*")))
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {logs_root}")
    return runs[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None,
                         help="Path to a logs/run_<ts> directory from swarm_sim.py. "
                              "Defaults to the most recent run.")
    parser.add_argument("--verify_sonar_shape_only", action="store_true",
                         help="Just print the sonar array shape from the log and exit "
                              "(sanity-check before trusting the mapping).")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    print(f"Using run_dir: {run_dir}")

    if args.verify_sonar_shape_only:
        verify_sonar_shape(run_dir)
        sys.exit(0)

    verify_sonar_shape(run_dir)
    mappers, agent_names = run_distributed_fusion(run_dir)
    evaluate_and_save(mappers, agent_names, run_dir)
