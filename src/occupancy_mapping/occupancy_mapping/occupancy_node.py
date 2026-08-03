"""
Live ROS2 adaptation of occupancy_mapping.py's run_distributed_fusion().

Deliberate simplification vs. one-node-per-agent: this single node hosts
every agent's AgentLocalMapper and performs the acoustic-range-gated delta
relay internally, the same way run_distributed_fusion() iterated over all
agents' logs in one process. The interesting, citable artifact here is the
range-gating logic and delta-compression math itself (see occupancy_mapping.py's
module docstring), not literal OS-process separation between agents -- so
this keeps the reused fusion classes and their tested behavior completely
unchanged, just driven by live ROS2 callbacks instead of replayed pickle logs.

Bridges to the rest of the pipeline via the same run_dir/occupancy/*.npy
file layout the reused code (PlanningEnvironment, damage_probability_mapping.py)
already expects -- so those consumers need zero changes to work with a live
run instead of a replayed one.
"""

import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from occupancy_mapping.occupancy_mapping import AgentLocalMapper, ACOUSTIC_MAX_RANGE, atomic_save_npy
from occupancy_mapping.sensor_adapter import laserscan_to_sonar_array, quaternion_to_rpy_deg


class OccupancyMappingNode(Node):
    def __init__(self):
        super().__init__("occupancy_mapping_node")

        self.declare_parameter("num_auvs", 3)
        self.declare_parameter("run_dir", os.path.expanduser("~/swarm_ws/live_run"))
        self.declare_parameter("broadcast_period_sec", 3.0)
        self.declare_parameter("snapshot_period_sec", 5.0)

        num_auvs = self.get_parameter("num_auvs").value
        self.run_dir = os.path.expanduser(self.get_parameter("run_dir").value)
        broadcast_period = self.get_parameter("broadcast_period_sec").value
        snapshot_period = self.get_parameter("snapshot_period_sec").value

        self.agent_names = [f"auv{i}" for i in range(num_auvs)]
        self.mappers = {name: AgentLocalMapper(name) for name in self.agent_names}
        self.positions = {name: None for name in self.agent_names}
        self.rotations = {name: None for name in self.agent_names}

        for name in self.agent_names:
            self.create_subscription(
                Odometry, f"/{name}/odom",
                self._make_odom_cb(name), qos_profile_sensor_data)
            self.create_subscription(
                LaserScan, f"/{name}/sonar/scan",
                self._make_sonar_cb(name), qos_profile_sensor_data)

        self.create_timer(broadcast_period, self._broadcast_tick)
        self.create_timer(snapshot_period, self._snapshot_tick)

        self.get_logger().info(
            f"occupancy_mapping_node: tracking {self.agent_names}, "
            f"writing snapshots to {self.run_dir}/occupancy/")

    def _make_odom_cb(self, name):
        def cb(msg):
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            self.positions[name] = np.array([p.x, p.y, p.z])
            self.rotations[name] = quaternion_to_rpy_deg(o.x, o.y, o.z, o.w)
        return cb

    def _make_sonar_cb(self, name):
        def cb(msg):
            pos = self.positions[name]
            rot = self.rotations[name]
            if pos is None or rot is None:
                return  # no pose yet -- skip this frame rather than guess
            sonar_array = laserscan_to_sonar_array(
                msg.ranges, msg.range_min, msg.range_max)
            record = {"location": pos, "rotation": rot, "sonar": sonar_array}
            self.mappers[name].process_record(record)
        return cb

    def _broadcast_tick(self):
        deltas = {name: self.mappers[name].compute_delta() for name in self.agent_names}
        for sender in self.agent_names:
            delta = deltas[sender]
            if not delta:
                continue
            for receiver in self.agent_names:
                if receiver == sender:
                    continue
                p_s, p_r = self.positions[sender], self.positions[receiver]
                if p_s is None or p_r is None:
                    continue
                if np.linalg.norm(p_s - p_r) <= ACOUSTIC_MAX_RANGE:
                    self.mappers[receiver].receive(sender, delta)
            self.mappers[sender].commit_broadcast(delta)

    def _snapshot_tick(self):
        out_dir = os.path.join(self.run_dir, "occupancy")
        os.makedirs(out_dir, exist_ok=True)
        for name in self.agent_names:
            atomic_save_npy(os.path.join(out_dir, f"{name}_fused_logodds.npy"),
                             self.mappers[name].fused_log_odds())


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyMappingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
