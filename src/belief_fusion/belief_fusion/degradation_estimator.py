"""
Degradation prognosis: for each detected damage point, estimates how
urgent it is -- specifically, answers "how long can we let this be before
it becomes a big problem," which nothing else in the pipeline does yet.

Two modes, used automatically depending on what data is available:

1. TREND-BASED (the real, defensible estimate): if a PRIOR inspection pass
   over the same structure exists (the same two-run comparison
   digital_twin.py already uses for its differential view), computes the
   actual growth rate of damage probability at each detected point between
   the two passes, and linearly extrapolates forward to estimate how many
   days until it crosses a critical severity threshold. This is a genuine
   trend measurement, not a guess -- but it's still a FIRST-ORDER LINEAR
   extrapolation, and real crack propagation is frequently nonlinear
   (often accelerating as a structure weakens) -- stated as a caveat on
   every trend-based estimate, not hidden in a footnote.

2. SEVERITY-HEURISTIC (fallback, used when there's no prior pass to
   measure a trend from -- e.g. the very first inspection of a structure):
   maps current damage probability + observation confidence to a
   qualitative urgency band and a recommended re-inspection interval,
   using conservative, clearly-labeled heuristic thresholds -- NOT a
   calibrated structural-engineering standard, and the output says so
   explicitly. This is a recommendation for when to look again, not a
   prediction of failure timing.

Every output record states explicitly which mode produced it (`basis`
field) so a dashboard or report can never present a heuristic guess as if
it were a measured trend, or vice versa.

Run:
    python degradation_estimator.py --run_dir logs/run_<ts>
    python degradation_estimator.py --run_dir logs/run_<later_ts> --compare_run_dir logs/run_<earlier_ts>
"""

import os
import glob
import json
import argparse
import numpy as np

from occupancy_mapping.occupancy_mapping import ENV_MIN, VOXEL_SIZE, find_latest_run_dir

STRUCTURE_PROB_THRESHOLD = 0.6     # matches digital_twin.py's structure mask
DAMAGE_DETECTION_THRESHOLD = 0.5   # min damage probability to count as "a detection" at all
CRITICAL_SEVERITY_THRESHOLD = 0.9  # damage probability considered "critical"
MIN_PEAK_SEPARATION_VOXELS = 6     # matches cbba_task_allocation.py's local-maxima spacing

# Severity-heuristic urgency bands (used only when no prior-pass trend data
# exists). Thresholds and intervals are conservative, illustrative
# engineering-judgment defaults -- NOT calibrated against a real structural
# standard. Always presented with that caveat, never as a hard prediction.
URGENCY_BANDS = [
    (0.90, "CRITICAL", 7,   "High-confidence severe detection -- recommend immediate follow-up inspection and engineering review."),
    (0.75, "HIGH",      30,  "Strong detection -- recommend re-inspection within a month."),
    (0.60, "MODERATE",  90,  "Moderate-confidence detection -- recommend re-inspection within a quarter."),
    (0.50, "LOW",       180, "Weak/borderline detection -- recommend monitoring at next scheduled inspection."),
]


def _urgency_band(severity):
    for min_sev, band, days, note in URGENCY_BANDS:
        if severity >= min_sev:
            return band, days, note
    return "NONE", None, "Below detection threshold."


def extract_damage_detections(damage_prob_grid, occ_prob_grid,
                                threshold=DAMAGE_DETECTION_THRESHOLD):
    """
    Connected-component labeling of voxels above threshold, masked to
    confirmed structure -- each connected region collapses to ONE
    detection at its peak severity location.

    NOTE: this deliberately does NOT use the local-maxima approach
    cbba_task_allocation.py uses for risk_score. That distinction matters:
    risk_score's mask covers the entire swept structure surface (one huge
    continuous band), so connected-component labeling collapsed it into a
    single meaningless mega-region there. Damage detections above a 0.5
    probability threshold are the opposite case -- they should be a
    handful of small, genuinely SEPARATE regions, not one connected band.
    Tested directly: a single flat 5x5x5-voxel damage plateau produced 8
    duplicate local-maxima detections (tied peaks on a flat plateau, spaced
    farther apart than the dedup radius); connected-component labeling
    correctly collapses it to exactly 1.
    """
    from scipy.ndimage import label

    structure_mask = occ_prob_grid > STRUCTURE_PROB_THRESHOLD
    detection_mask = (damage_prob_grid > threshold) & structure_mask

    labeled, n_components = label(detection_mask)
    detections = []
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        comp_values = damage_prob_grid[comp_mask]
        comp_indices = np.argwhere(comp_mask)
        peak_local_idx = int(np.argmax(comp_values))
        idx = comp_indices[peak_local_idx]
        severity = float(comp_values[peak_local_idx])

        world_pos = ENV_MIN + (idx + 0.5) * VOXEL_SIZE
        detections.append({"position": world_pos.tolist(), "voxel_idx": idx.tolist(),
                             "current_severity": severity, "region_size_voxels": int(comp_mask.sum())})
    return detections


def estimate_prognosis(run_dir, compare_run_dir=None, time_delta_days=None):
    """
    Main entry point. If compare_run_dir is given, uses the trend-based
    estimate; otherwise falls back to the severity heuristic for every
    detection.

    time_delta_days: how many days elapsed between compare_run_dir (the
    earlier pass) and run_dir (the current pass). Required for a
    meaningful trend-based days-to-critical estimate -- if not given but
    compare_run_dir is, growth rate is reported per-comparison-pass rather
    than per-day (still useful, just not in day units).
    """
    from .digital_twin import _load_agent_grids

    occ_prob, damage_prob, uncertainty = _load_agent_grids(run_dir)
    detections = extract_damage_detections(damage_prob, occ_prob)

    prior_damage_prob = None
    if compare_run_dir:
        prior_occ_prob, prior_damage_prob, _ = _load_agent_grids(compare_run_dir)

    records = []
    for det in detections:
        idx = tuple(det["voxel_idx"])
        current_severity = det["current_severity"]

        if prior_damage_prob is not None:
            prior_severity = float(prior_damage_prob[idx])
            delta = current_severity - prior_severity

            if delta > 1e-6:
                if time_delta_days:
                    growth_rate_per_day = delta / time_delta_days
                    remaining = CRITICAL_SEVERITY_THRESHOLD - current_severity
                    days_to_critical = (remaining / growth_rate_per_day
                                         if growth_rate_per_day > 0 else None)
                    record = {
                        "position": det["position"],
                        "current_severity": current_severity,
                        "prior_severity": prior_severity,
                        "growth_rate_per_day": growth_rate_per_day,
                        "estimated_days_to_critical": (
                            max(0, days_to_critical) if days_to_critical is not None else None),
                        "basis": "trend-based (linear extrapolation from measured growth "
                                 "between two inspection passes)",
                        "caveat": "Assumes constant linear growth. Real crack propagation is "
                                  "often nonlinear (frequently accelerating) -- treat this as a "
                                  "conservative first estimate, not a guarantee.",
                    }
                else:
                    record = {
                        "position": det["position"],
                        "current_severity": current_severity,
                        "prior_severity": prior_severity,
                        "growth_per_pass": delta,
                        "basis": "trend-based (growth measured between two inspection passes, "
                                 "no elapsed-time value given -- pass time_delta_days for a "
                                 "days-to-critical estimate)",
                        "caveat": "Growth rate is per-inspection-pass, not per unit time.",
                    }
            else:
                record = {
                    "position": det["position"],
                    "current_severity": current_severity,
                    "prior_severity": prior_severity,
                    "growth_rate_per_day": 0.0 if time_delta_days else None,
                    "estimated_days_to_critical": None,
                    "basis": "trend-based (no measured increase since prior pass)",
                    "caveat": "Stable or improved since last inspection -- no growth-based "
                              "urgency escalation, but see severity band for baseline priority.",
                }
        else:
            record = {
                "position": det["position"],
                "current_severity": current_severity,
                "basis": "severity-heuristic (no prior inspection pass available -- "
                         "urgency reflects current severity only, not a measured trend)",
                "caveat": "First inspection of this location -- no growth trend available yet. "
                          "Re-run this estimator after a follow-up pass for a trend-based estimate.",
            }

        band, reinspect_days, note = _urgency_band(current_severity)
        record["urgency_band"] = band
        record["recommended_reinspection_days"] = reinspect_days
        record["recommendation"] = note
        records.append(record)

    records.sort(key=lambda r: r["current_severity"], reverse=True)
    return records


def save_and_report(records, run_dir):
    out_dir = os.path.join(run_dir, "prognosis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prognosis.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n{'=' * 90}\nDEGRADATION PROGNOSIS -- {len(records)} detection(s)\n{'=' * 90}")
    for r in records:
        pos = np.round(r["position"], 1)
        line = f"[{r['urgency_band']:8s}] pos={pos}  severity={r['current_severity']:.2f}"
        if "estimated_days_to_critical" in r and r["estimated_days_to_critical"] is not None:
            line += f"  -> est. {r['estimated_days_to_critical']:.0f} days to critical (TREND-BASED)"
        elif r["basis"].startswith("severity-heuristic"):
            line += f"  -> re-inspect within {r['recommended_reinspection_days']} days (heuristic, no trend data)"
        print(line)
    print(f"\nSaved -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--compare_run_dir", type=str, default=None,
                         help="An earlier inspection pass, for trend-based estimates.")
    parser.add_argument("--time_delta_days", type=float, default=None,
                         help="Days elapsed between compare_run_dir and run_dir.")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    print(f"Using run_dir: {run_dir}")
    records = estimate_prognosis(run_dir, args.compare_run_dir, args.time_delta_days)
    save_and_report(records, run_dir)
