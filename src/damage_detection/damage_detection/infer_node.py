"""
Live per-agent crack-detection ROS2 node.

Single node handling every agent's camera stream through one shared
InferenceEngine instance (one model in memory, not N duplicate copies) --
subscribes to each /auv{i}/camera/image_raw, runs infer.py's
InferenceEngine on each incoming frame, publishes the resulting
damage-probability map as a mono8 Image (probability * 255) on
/auv{i}/damage_prob for belief_fusion to consume.

No trained checkpoint ships with this repo (training takes real wall-clock
time independent of this pipeline build-out -- see GAZEBO_MIGRATION_HANDOFF.md
Section 5.6). If checkpoint_path doesn't resolve to a real file, this node
falls back to InferenceEngine.untrained() so the full topic/shape/fusion
chain can still be exercised and verified end-to-end; a clear warning is
logged so "the demo looks quiet" is never confused with "the wiring is
broken." Point checkpoint_path at a real train.py output before demo day.

inference_rate_hz throttles inference per agent (default 2Hz, well below
the camera's 15Hz publish rate) -- running CPU inference on every frame
across 3 agents measured at 400%+ CPU on this machine, which starved every
other node's ROS2 executor (confirmed live: swarm_control's path planner
went from 6s/leg in isolation to 30s/leg with this node running
unthrottled). Damage detections don't need to be captured every frame to
be useful for the fusion layer's belief update; 2Hz is a deliberate
CPU/responsiveness trade for the live demo, not a correctness requirement.
"""

import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from damage_detection.infer import InferenceEngine


def bgr_from_image_msg(msg):
    """Minimal, allocation-light Image -> HxWx3 BGR uint8 conversion, no
    cv_bridge dependency (avoids adding another package dependency for a
    single, simple case: ros_gz_bridge always publishes rgb8 here, see the
    AUV camera sensor topic bridge in simulation.launch.py)."""
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        return arr[:, :, ::-1]
    return arr


def image_msg_from_mono_prob(prob_map, stamp, frame_id):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = prob_map.shape
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = msg.width
    msg.data = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8).tobytes()
    return msg


class DamageInferenceNode(Node):
    def __init__(self):
        super().__init__("damage_inference_node")

        self.declare_parameter("num_auvs", 3)
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("inference_rate_hz", 2.0)

        num_auvs = self.get_parameter("num_auvs").value
        checkpoint_path = self.get_parameter("checkpoint_path").value
        self.min_period_sec = 1.0 / self.get_parameter("inference_rate_hz").value
        self.last_inference_time = {}

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.engine = InferenceEngine(checkpoint_path)
            self.get_logger().info(f"Loaded trained checkpoint: {checkpoint_path}")
        else:
            self.engine = InferenceEngine.untrained()
            self.get_logger().warn(
                "No trained checkpoint given/found (checkpoint_path="
                f"'{checkpoint_path}') -- running an UNTRAINED model. "
                "Damage probability output is meaningless noise; this only "
                "verifies the topic/shape/fusion wiring. Train a real "
                "checkpoint with damage_detection/train.py before the demo.")

        self.agent_names = [f"auv{i}" for i in range(num_auvs)]
        self.publishers_ = {}
        for name in self.agent_names:
            self.create_subscription(
                Image, f"/{name}/camera/image_raw",
                self._make_camera_cb(name), qos_profile_sensor_data)
            self.publishers_[name] = self.create_publisher(
                Image, f"/{name}/damage_prob", 10)

        self.get_logger().info(
            f"damage_inference_node: running inference for {self.agent_names}")

    def _make_camera_cb(self, name):
        def cb(msg):
            now = time.monotonic()
            last = self.last_inference_time.get(name, 0.0)
            if now - last < self.min_period_sec:
                return
            self.last_inference_time[name] = now

            bgr = bgr_from_image_msg(msg)
            prob_map = self.engine.infer(bgr)
            out = image_msg_from_mono_prob(prob_map, msg.header.stamp, f"{name}/damage_prob")
            self.publishers_[name].publish(out)
        return cb


def main(args=None):
    rclpy.init(args=args)
    node = DamageInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
