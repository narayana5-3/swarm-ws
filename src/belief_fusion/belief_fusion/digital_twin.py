"""
Digital twin visualization.

Combines everything the pipeline has produced so far -- structural
occupancy, fused damage probability, uncertainty, and CBBA task allocation
-- into a single interactive 3D view of the inspected structure's current
health state. This is the final synthesis layer: every other module in
this project produces one piece of evidence; this one assembles them into
the thing a human (or a report figure, or a demo) actually looks at.

Also implements the "longitudinal differential digital twin" element from
the project design: given a SECOND run (a later inspection pass over the
same structure), computes and highlights where damage probability has
INCREASED since the previous pass -- i.e. genuine change detection over
time, not just a snapshot. This is the patentable element beyond a static
health map: tracking how the structure's condition evolves across repeated
inspections.

Output is a standalone interactive HTML file (Plotly) -- rotatable,
zoomable, hoverable, and viewable in any browser without needing Python
running, which also makes it a strong demo asset on its own.

Design notes / simplifications (documented, not hidden):
  - Only voxels the occupancy grid is reasonably confident are structure
    (P(occupied) > STRUCTURE_PROB_THRESHOLD) are rendered -- showing every
    voxel in the grid (mostly open water) would be dense and uninformative.
  - Differential mode assumes both runs used the same ENV_MIN/ENV_MAX/
    VOXEL_SIZE (true by default here, since both come from the same
    occupancy_mapping.py constants) so voxel indices line up directly
    without needing spatial registration/ICP -- reasonable for two passes
    of the same simulated scenario, would need real registration for two
    genuinely independent real-world scans.

Run:
    python digital_twin.py --run_dir logs/run_<ts>
    python digital_twin.py --run_dir logs/run_<ts> --compare_run_dir logs/run_<earlier_ts>
"""

import os
import glob
import pickle
import argparse
import numpy as np

from occupancy_mapping.occupancy_mapping import ENV_MIN, VOXEL_SIZE, find_latest_run_dir

STRUCTURE_PROB_THRESHOLD = 0.6   # min occupancy probability to render as structure
DAMAGE_INCREASE_THRESHOLD = 0.1  # min probability increase to flag as "worsening"


def _load_agent_grids(run_dir, damage_probability_dir=None):
    """Loads the first available agent's fused occupancy + damage + uncertainty
    grids for a run (post-fusion, these should already closely agree across
    agents -- see occupancy_mapping.py's map-agreement metric)."""
    occ_paths = sorted(glob.glob(os.path.join(run_dir, "occupancy", "*_fused_logodds.npy")))
    if not occ_paths:
        raise FileNotFoundError(
            f"No occupancy grids found in {run_dir}/occupancy -- run "
            f"occupancy_mapping.py for this run first.")

    damage_probability_dir = damage_probability_dir or os.path.join(run_dir, "damage_probability")
    damage_paths = sorted(glob.glob(os.path.join(damage_probability_dir, "*_damage_prob.npy")))
    uncertainty_paths = sorted(glob.glob(os.path.join(damage_probability_dir, "*_uncertainty.npy")))

    occ_log_odds = np.load(occ_paths[0])
    occ_prob = 1.0 / (1.0 + np.exp(-occ_log_odds))

    damage_prob = np.load(damage_paths[0]) if damage_paths else np.full_like(occ_prob, 0.5)
    uncertainty = np.load(uncertainty_paths[0]) if uncertainty_paths else np.ones_like(occ_prob)

    if not damage_paths:
        print(f"[digital_twin] WARNING: no damage_probability grids found in "
              f"{damage_probability_dir} -- run damage_probability_mapping.py "
              f"first for a meaningful health overlay. Rendering structure only.")

    return occ_prob, damage_prob, uncertainty


def _load_task_allocation(run_dir):
    path = os.path.join(run_dir, "task_allocation", "cbba_assignment.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def build_digital_twin(run_dir, damage_probability_dir=None, compare_run_dir=None,
                        out_path=None):
    import plotly.graph_objects as go

    occ_prob, damage_prob, uncertainty = _load_agent_grids(run_dir, damage_probability_dir)
    structure_mask = occ_prob > STRUCTURE_PROB_THRESHOLD
    idx = np.argwhere(structure_mask)

    if len(idx) == 0:
        raise ValueError(
            f"No voxels above STRUCTURE_PROB_THRESHOLD={STRUCTURE_PROB_THRESHOLD} "
            f"-- either the occupancy map is too sparse (short sweep) or the "
            f"threshold needs lowering.")

    world_pts = ENV_MIN + (idx + 0.5) * VOXEL_SIZE
    damage_vals = damage_prob[structure_mask]
    uncertainty_vals = uncertainty[structure_mask]

    traces = []

    # Main structure, colored by damage probability -- the primary health view
    traces.append(go.Scatter3d(
        x=world_pts[:, 0], y=world_pts[:, 1], z=world_pts[:, 2],
        mode="markers",
        marker=dict(
            size=3,
            color=damage_vals,
            colorscale="RdYlGn_r",   # green=healthy, red=high damage probability
            cmin=0, cmax=1,
            colorbar=dict(title="P(damage)", x=1.02),
            opacity=0.85,
        ),
        text=[f"P(damage)={d:.2f}<br>uncertainty={u:.2f}"
              for d, u in zip(damage_vals, uncertainty_vals)],
        hoverinfo="text",
        name="Structure (colored by damage probability)"
    ))

    # Task allocation overlay, if available
    task_result = _load_task_allocation(run_dir)
    if task_result is not None:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i, (name, info) in enumerate(task_result["agents"].items()):
            color = colors[i % len(colors)]
            start = info["start_position"]
            task_positions = [t["position"] for t in info["assigned_tasks"]]

            traces.append(go.Scatter3d(
                x=[start[0]], y=[start[1]], z=[start[2]],
                mode="markers", marker=dict(size=6, color=color, symbol="diamond"),
                name=f"{name} start", showlegend=True
            ))

            if task_positions:
                path_pts = [start] + task_positions
                xs, ys, zs = zip(*path_pts)
                traces.append(go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines+markers",
                    line=dict(color=color, width=4),
                    marker=dict(size=5, color=color, symbol="square"),
                    name=f"{name} planned inspection path ({len(task_positions)} tasks)"
                ))
    else:
        print("[digital_twin] No task_allocation/cbba_assignment.pkl found -- "
              "rendering health map only, without planned inspection paths. "
              "Run cbba_task_allocation.py first to include them.")

    # Longitudinal differential overlay, if a comparison run is given
    if compare_run_dir is not None:
        prev_occ_prob, prev_damage_prob, _ = _load_agent_grids(compare_run_dir)
        prev_structure_mask = prev_occ_prob > STRUCTURE_PROB_THRESHOLD

        # only compare voxels BOTH runs agree are structure, so we're not
        # confusing "newly discovered structure" with "worsening damage"
        both_structure = structure_mask & prev_structure_mask
        delta = damage_prob - prev_damage_prob
        worsening_mask = both_structure & (delta > DAMAGE_INCREASE_THRESHOLD)

        worsening_idx = np.argwhere(worsening_mask)
        if len(worsening_idx) > 0:
            worsening_pts = ENV_MIN + (worsening_idx + 0.5) * VOXEL_SIZE
            worsening_delta = delta[worsening_mask]
            traces.append(go.Scatter3d(
                x=worsening_pts[:, 0], y=worsening_pts[:, 1], z=worsening_pts[:, 2],
                mode="markers",
                marker=dict(size=6, color="magenta", symbol="x",
                           line=dict(width=1, color="black")),
                text=[f"damage probability increased by {d:.2f} since previous inspection"
                      for d in worsening_delta],
                hoverinfo="text",
                name=f"WORSENING since previous inspection ({len(worsening_idx)} voxels)"
            ))
            print(f"[digital_twin] {len(worsening_idx)} voxels show damage "
                  f"probability increase > {DAMAGE_INCREASE_THRESHOLD} since "
                  f"the comparison run -- flagged in magenta.")
        else:
            print(f"[digital_twin] No voxels show meaningful damage increase "
                  f"vs. the comparison run (threshold={DAMAGE_INCREASE_THRESHOLD}).")

    fig = go.Figure(data=traces)
    title = "Digital Twin: Structural Health Map"
    if compare_run_dir is not None:
        title += " with Longitudinal Change Detection"
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   aspectmode="data"),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_dir = os.path.join(run_dir, "digital_twin")
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_path or os.path.join(out_dir, "digital_twin.html")
    fig.write_html(out_path)
    print(f"[digital_twin] interactive visualization saved -> {out_path}")
    print(f"[digital_twin] open this file directly in any browser -- no "
          f"Python needed to view it, good for demos/presentations too.")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None,
                         help="Path to a logs/run_<ts> directory. Defaults to most recent.")
    parser.add_argument("--damage_probability_dir", type=str, default=None,
                         help="Defaults to <run_dir>/damage_probability.")
    parser.add_argument("--compare_run_dir", type=str, default=None,
                         help="Path to an EARLIER logs/run_<ts> directory (a "
                              "previous inspection pass) to enable longitudinal "
                              "change detection -- highlights voxels where "
                              "damage probability has increased since then.")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    print(f"Using run_dir: {run_dir}")
    build_digital_twin(run_dir, args.damage_probability_dir, args.compare_run_dir)
