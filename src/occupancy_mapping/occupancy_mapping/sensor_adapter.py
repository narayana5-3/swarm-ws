"""
Adapts the AUV model's real gpu_lidar-based sonar substitute (see
src/simulation/models/auv/model.sdf) into the synthetic [range_bin,
azimuth_bin] intensity array occupancy_mapping.py's sonar_hits_world() and
damage_probability_mapping.py's AgentDamageMapper.process_frame() both
expect.

Project DAVE's real ROS2 multibeam sonar plugin never landed upstream (see
GAZEBO_MIGRATION_HANDOFF.md) so the AUV uses a gpu_lidar sensor bridged to a
plain sensor_msgs/LaserScan instead -- one measured RANGE per azimuth
column, not a full range-intensity return profile per bin the way real
sonar hardware (and HoloOcean's simulated sonar) provides. This builds the
synthetic profile the reused ray-geometry functions expect: a single
intensity spike at the bin matching the real measured range, so their
existing per-column argmax-above-threshold hit-detection logic keeps
working completely unmodified.
"""

import numpy as np

# Matches occupancy_mapping.py's SONAR_RANGE_MIN/MAX and the AUV model's
# gpu_lidar <range> tags.
DEFAULT_N_RANGE_BINS = 200


def laserscan_to_sonar_array(ranges, range_min, range_max,
                              n_range_bins=DEFAULT_N_RANGE_BINS):
    """
    ranges: sequence of per-azimuth-column measured distances, in the order
    LaserScan reports them (angle_min -> angle_max), which is the same
    -FOV/2 -> +FOV/2 azimuth ordering sonar_hits_world() assumes for its
    columns.

    Returns a (n_range_bins, n_azimuth_bins) array with a single 1.0 spike
    per column at the bin nearest the real measured range, or an all-zero
    column when that azimuth had no real return (range >= range_max, i.e.
    Gazebo's "nothing hit" convention, or non-finite).
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    n_azimuth_bins = len(ranges)
    array = np.zeros((n_range_bins, n_azimuth_bins), dtype=np.float64)
    bin_edges = np.linspace(range_min, range_max, n_range_bins)

    for col, r in enumerate(ranges):
        if not np.isfinite(r) or r >= range_max - 1e-3 or r < range_min:
            continue
        row = int(np.clip(np.searchsorted(bin_edges, r), 0, n_range_bins - 1))
        array[row, col] = 1.0

    return array


def quaternion_to_rpy_deg(x, y, z, w):
    """Standard ZYX Euler extraction, converted to degrees to match
    occupancy_mapping.py's rotation_matrix_rpy(roll_deg, pitch_deg, yaw_deg)."""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])
