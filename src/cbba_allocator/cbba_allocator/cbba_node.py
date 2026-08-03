"""
Live ROS2 adaptation of cbba_task_allocation.py's allocate() driver.

Periodically (allocation_period_sec) re-reads the swarm's fused risk_score
grids that belief_fusion_node writes to run_dir/damage_probability/*_risk_score.npy,
extracts discrete inspection tasks via extract_tasks_from_risk_grid()
(unchanged), and runs the acoustic-range-gated sequential single-task
auction (CBBAAgent / run_sequential_auction, both unchanged) using each
agent's LATEST live position instead of a replayed log's last position --
the direct live equivalent of get_last_positions() in the original driver.
Re-running periodically as new sonar/camera coverage updates the risk grid
is the live version of "adaptive task reallocation" the module docstring
describes; a fixed period is a simpler, equally faithful trigger than
watching for a grid-change threshold and was chosen for this pass (a
change-triggered version is a straightforward follow-up: compare
np.abs(risk - previous_risk).sum() against a threshold before re-auctioning).

Publishes each agent's assigned task sequence as a geometry_msgs/PoseArray
on /auv{i}/assigned_waypoints -- swarm_control consumes this directly as
the route to plan through.
"""

import os
import glob
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose

from cbba_allocator.cbba_task_allocation import (
    CBBAAgent, extract_tasks_from_risk_grid, run_sequential_auction, ACOUSTIC_MAX_RANGE,
)


class CBBAAllocatorNode(Node):
    def __init__(self):
        super().__init__("cbba_allocator_node")

        self.declare_parameter("num_auvs", 3)
        self.declare_parameter("run_dir", os.path.expanduser("~/swarm_ws/live_run"))
        self.declare_parameter("allocation_period_sec", 15.0)

        num_auvs = self.get_parameter("num_auvs").value
        self.run_dir = os.path.expanduser(self.get_parameter("run_dir").value)
        allocation_period = self.get_parameter("allocation_period_sec").value

        self.agent_names = [f"auv{i}" for i in range(num_auvs)]
        self.positions = {name: None for name in self.agent_names}
        self.allocation_in_progress = threading.Lock()

        for name in self.agent_names:
            self.create_subscription(
                Odometry, f"/{name}/odom",
                self._make_odom_cb(name), qos_profile_sensor_data)

        self.waypoint_pubs = {
            name: self.create_publisher(PoseArray, f"/{name}/assigned_waypoints", 10)
            for name in self.agent_names
        }

        self.create_timer(allocation_period, self._allocate_tick)

        self.get_logger().info(
            f"cbba_allocator_node: re-auctioning every {allocation_period}s "
            f"for {self.agent_names}, reading {self.run_dir}/damage_probability/")

    def _make_odom_cb(self, name):
        def cb(msg):
            p = msg.pose.pose.position
            self.positions[name] = np.array([p.x, p.y, p.z])
        return cb

    def _load_combined_risk_grid(self):
        risk_dir = os.path.join(self.run_dir, "damage_probability")
        paths = sorted(glob.glob(os.path.join(risk_dir, "*_risk_score.npy")))
        if not paths:
            return None
        # Swarm-fused per-agent views should closely agree post-fusion; take
        # the element-wise max across agents as a conservative combined view
        # (never misses a detection any single agent's file already has).
        grids = [np.load(p) for p in paths]
        return np.maximum.reduce(grids)

    def _allocate_tick(self):
        """Kicks off _run_allocation in a background thread rather than
        running it inline -- extract_tasks_from_risk_grid's scipy
        maximum_filter over the full 200x200x140 grid measured 100+ seconds
        under real CPU contention on this machine (0.3s for the actual
        auction itself), and a single-threaded rclpy executor blocks ALL
        other callbacks (including this node's own odom subscriptions) for
        as long as a timer callback runs. allocation_in_progress skips this
        tick rather than queuing a second overlapping run if the previous
        one hasn't finished yet."""
        if not self.allocation_in_progress.acquire(blocking=False):
            self.get_logger().info("Previous allocation still running, skipping this tick.")
            return
        threading.Thread(target=self._run_allocation, daemon=True).start()

    def _run_allocation(self):
        try:
            self._allocate_once()
        finally:
            self.allocation_in_progress.release()

    def _allocate_once(self):
        if any(self.positions[name] is None for name in self.agent_names):
            self.get_logger().info("Waiting for all agents' odometry before first auction...")
            return

        try:
            risk_grid = self._load_combined_risk_grid()
        except (ValueError, EOFError, OSError) as exc:
            # atomic_save_npy (see occupancy_mapping.py) prevents torn reads
            # going forward, but this stays defensive against any other
            # transient read race rather than crashing the whole node.
            self.get_logger().warn(f"Risk grid read failed this tick, will retry: {exc}")
            return
        if risk_grid is None:
            self.get_logger().info("No risk_score grids yet -- belief_fusion hasn't snapshotted.")
            return

        tasks = extract_tasks_from_risk_grid(risk_grid)
        if not tasks:
            self.get_logger().info("No tasks above RISK_THRESHOLD yet.")
            return

        agents = {name: CBBAAgent(name, self.positions[name]) for name in self.agent_names}
        run_sequential_auction(agents, tasks, acoustic_range=ACOUSTIC_MAX_RANGE)

        tasks_by_id = {t["task_id"]: t for t in tasks}
        for name in self.agent_names:
            msg = PoseArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "world"
            for task_id in agents[name].path:
                pos = tasks_by_id[task_id]["position"]
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = (float(v) for v in pos)
                pose.orientation.w = 1.0
                msg.poses.append(pose)
            self.waypoint_pubs[name].publish(msg)

        assignment_summary = {name: len(agents[name].path) for name in self.agent_names}
        self.get_logger().info(f"Auctioned {len(tasks)} tasks -> {assignment_summary}")


def main(args=None):
    rclpy.init(args=args)
    node = CBBAAllocatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
