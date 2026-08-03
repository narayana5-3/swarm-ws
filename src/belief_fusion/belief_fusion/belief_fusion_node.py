"""
Live ROS2 adaptation of damage_probability_mapping.py's run_damage_fusion(),
same single-node simplification as occupancy_mapping_node (see that file's
docstring for why): one process hosts every agent's AgentDamageMapper and
performs the acoustic-range-gated delta relay internally.

Fuses each agent's sonar ray geometry (reusing occupancy_mapping's
sensor_adapter, same as occupancy_node) with damage_detection's live
per-frame damage-probability image to build a shared damage-probability +
uncertainty map, then reuses evaluate_and_save() and
estimate_prognosis()/save_and_report() from the HoloOcean-phase code
COMPLETELY UNCHANGED -- they already read/write exactly the run_dir file
layout this live pipeline populates (run_dir/occupancy/*.npy from
occupancy_mapping_node, run_dir/damage_probability/*.npy from here,
run_dir/prognosis/prognosis.json for the dashboard).
"""

import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image

from occupancy_mapping.sensor_adapter import laserscan_to_sonar_array, quaternion_to_rpy_deg
from occupancy_mapping.occupancy_mapping import ACOUSTIC_MAX_RANGE
from belief_fusion.damage_probability_mapping import AgentDamageMapper, evaluate_and_save
from belief_fusion.degradation_estimator import estimate_prognosis, save_and_report


class BeliefFusionNode(Node):
    def __init__(self):
        super().__init__("belief_fusion_node")

        self.declare_parameter("num_auvs", 3)
        self.declare_parameter("run_dir", os.path.expanduser("~/swarm_ws/live_run"))
        self.declare_parameter("broadcast_period_sec", 3.0)
        self.declare_parameter("snapshot_period_sec", 10.0)

        num_auvs = self.get_parameter("num_auvs").value
        self.run_dir = os.path.expanduser(self.get_parameter("run_dir").value)
        broadcast_period = self.get_parameter("broadcast_period_sec").value
        snapshot_period = self.get_parameter("snapshot_period_sec").value

        self.agent_names = [f"auv{i}" for i in range(num_auvs)]
        self.mappers = {name: AgentDamageMapper(name) for name in self.agent_names}
        self.positions = {name: None for name in self.agent_names}
        self.rotations = {name: None for name in self.agent_names}
        self.latest_sonar_array = {name: None for name in self.agent_names}

        for name in self.agent_names:
            self.create_subscription(
                Odometry, f"/{name}/odom",
                self._make_odom_cb(name), qos_profile_sensor_data)
            self.create_subscription(
                LaserScan, f"/{name}/sonar/scan",
                self._make_sonar_cb(name), qos_profile_sensor_data)
            self.create_subscription(
                Image, f"/{name}/damage_prob",
                self._make_damage_cb(name), qos_profile_sensor_data)

        self.create_timer(broadcast_period, self._broadcast_tick)
        self.create_timer(snapshot_period, self._snapshot_tick)

        self.get_logger().info(
            f"belief_fusion_node: fusing damage probability for {self.agent_names}, "
            f"writing snapshots to {self.run_dir}/damage_probability/")

    def _make_odom_cb(self, name):
        def cb(msg):
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            self.positions[name] = np.array([p.x, p.y, p.z])
            self.rotations[name] = quaternion_to_rpy_deg(o.x, o.y, o.z, o.w)
        return cb

    def _make_sonar_cb(self, name):
        def cb(msg):
            self.latest_sonar_array[name] = laserscan_to_sonar_array(
                msg.ranges, msg.range_min, msg.range_max)
        return cb

    def _make_damage_cb(self, name):
        def cb(msg):
            pos = self.positions[name]
            rot = self.rotations[name]
            sonar_array = self.latest_sonar_array[name]
            if pos is None or rot is None or sonar_array is None:
                return  # no synced pose/sonar yet -- skip this frame
            damage_prob_map = (
                np.frombuffer(msg.data, dtype=np.uint8)
                .reshape(msg.height, msg.width).astype(np.float32) / 255.0)
            self.mappers[name].process_frame(pos, rot, sonar_array, damage_prob_map)
        return cb

    def _broadcast_tick(self):
        deltas = {name: self.mappers[name].compute_delta() for name in self.agent_names}
        for sender in self.agent_names:
            damage_delta, count_delta = deltas[sender]
            if not damage_delta and not count_delta:
                continue
            for receiver in self.agent_names:
                if receiver == sender:
                    continue
                p_s, p_r = self.positions[sender], self.positions[receiver]
                if p_s is None or p_r is None:
                    continue
                if np.linalg.norm(p_s - p_r) <= ACOUSTIC_MAX_RANGE:
                    self.mappers[receiver].receive(sender, damage_delta, count_delta)
            self.mappers[sender].commit_broadcast(damage_delta, count_delta)

    def _snapshot_tick(self):
        try:
            evaluate_and_save(self.mappers, self.agent_names, self.run_dir)
        except Exception as exc:  # noqa: BLE001 -- log and keep the node alive
            self.get_logger().warn(f"evaluate_and_save skipped this tick: {exc}")
            return

        try:
            records = estimate_prognosis(self.run_dir)
            save_and_report(records, self.run_dir)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"prognosis estimate skipped this tick: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = BeliefFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
