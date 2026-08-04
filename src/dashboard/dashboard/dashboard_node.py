"""
Live web dashboard for the demo.

Wraps live_dashboard.py's render_dashboard()/DashboardState (both reused
completely unchanged -- render_dashboard() is a pure function, exactly as
designed) in a Flask MJPEG stream, replacing the HoloOcean-phase's
cv2.imshow desktop window with a webpage judges can watch on any browser.

A single shared DashboardState is updated by ROS2 subscription callbacks
(camera, damage probability, odometry) and a periodic timer that re-reads
the same run_dir/damage_probability + run_dir/prognosis files
belief_fusion_node already writes -- then Flask's own thread renders and
serves whatever the state currently holds. Flask runs in a background
thread (its dev server blocks) while rclpy.spin() owns the main thread.
"""

import os
import glob
import json
import threading
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, PointStamped
from flask import Flask, Response, jsonify, render_template_string

from occupancy_mapping.occupancy_mapping import ENV_MIN, ENV_MAX
from dashboard.live_dashboard import DashboardState, render_dashboard
from damage_detection.infer import make_overlay

INDEX_HTML = """
<!doctype html>
<title>Swarm Mission Dashboard</title>
<body style="margin:0;background:#111;">
  <img src="/stream.mjpg" style="width:100%;height:auto;display:block;">
</body>
"""


def _bgr_from_rgb8(msg):
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, 3)[:, :, ::-1]


class DashboardNode(Node):
    def __init__(self):
        super().__init__("dashboard_node")

        self.declare_parameter("num_auvs", 3)
        self.declare_parameter("run_dir", os.path.expanduser("~/swarm_ws/live_run"))
        self.declare_parameter("port", 8080)
        self.declare_parameter("refresh_period_sec", 5.0)

        num_auvs = self.get_parameter("num_auvs").value
        self.run_dir = os.path.expanduser(self.get_parameter("run_dir").value)
        port = self.get_parameter("port").value
        refresh_period = self.get_parameter("refresh_period_sec").value

        agent_names = [f"auv{i}" for i in range(num_auvs)]
        self.state = DashboardState(agent_names)
        self.state.status_message = "Waiting for swarm..."
        self.state_lock = threading.Lock()
        self.latest_damage_prob = {name: None for name in agent_names}
        # Current auction cycle's assigned tasks (id/position/value), refreshed
        # from disk -- see _refresh_from_disk for why coverage is scoped to
        # "this cycle" rather than a lifetime total.
        self.current_tasks_by_id = {}
        self.captured_task_ids_this_cycle = set()

        for name in agent_names:
            self.create_subscription(
                Odometry, f"/{name}/odom", self._make_odom_cb(name), qos_profile_sensor_data)
            self.create_subscription(
                Image, f"/{name}/camera/image_raw", self._make_camera_cb(name),
                qos_profile_sensor_data)
            self.create_subscription(
                Image, f"/{name}/damage_prob", self._make_damage_cb(name),
                qos_profile_sensor_data)
            self.create_subscription(
                PoseArray, f"/{name}/assigned_waypoints", self._make_waypoints_cb(name), 10)
            self.create_subscription(
                PointStamped, f"/{name}/task_reached", self._make_task_reached_cb(name), 10)

        self.create_timer(refresh_period, self._refresh_from_disk)
        self._start_flask(port)

        self.get_logger().info(
            f"dashboard_node: serving http://0.0.0.0:{port}/ for {agent_names}")

    # -- ROS2 callbacks: update shared state -------------------------------

    def _make_odom_cb(self, name):
        def cb(msg):
            p = msg.pose.pose.position
            with self.state_lock:
                self.state.agent_positions[name] = np.array([p.x, p.y, p.z])
        return cb

    def _make_camera_cb(self, name):
        def cb(msg):
            bgr = _bgr_from_rgb8(msg)
            with self.state_lock:
                self.state.latest_camera_frame = bgr
                self.state.latest_camera_agent = name
                prob = self.latest_damage_prob.get(name)
                if prob is not None and prob.shape == bgr.shape[:2]:
                    self.state.latest_damage_overlay = make_overlay(bgr, prob)
        return cb

    def _make_damage_cb(self, name):
        def cb(msg):
            prob = (np.frombuffer(msg.data, dtype=np.uint8)
                    .reshape(msg.height, msg.width).astype(np.float32) / 255.0)
            self.latest_damage_prob[name] = prob
        return cb

    def _make_waypoints_cb(self, name):
        def cb(msg):
            with self.state_lock:
                self.state.agent_current_task[name] = (
                    f"{len(msg.poses)} waypoint(s) assigned" if msg.poses else None)
                if msg.poses:
                    self.state.recent_detections_log.insert(
                        0, f"tick {self.state.tick}: {name} assigned "
                           f"{len(msg.poses)} task waypoint(s)")
                    del self.state.recent_detections_log[20:]
        return cb

    def _make_task_reached_cb(self, name):
        def cb(msg):
            pos = np.array([msg.point.x, msg.point.y, msg.point.z])
            with self.state_lock:
                task_id, task = self._nearest_task(pos)
                if task_id is None or task_id in self.captured_task_ids_this_cycle:
                    return  # no current-cycle task snapshot yet, or already logged
                self.captured_task_ids_this_cycle.add(task_id)
                self.state.tasks_captured += 1
                self.state.cumulative_value += task["value"]
                self.state.recent_detections_log.insert(
                    0, f"tick {self.state.tick}: {name} reached task near "
                       f"({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}), value={task['value']:.1f}")
                del self.state.recent_detections_log[20:]
        return cb

    def _nearest_task(self, pos, tolerance_m=3.0):
        """Matches a reached position back to a task_id in the current
        cycle's snapshot. Positions round-trip through PoseArray -> the
        planner -> here, so an exact task should match well within
        tolerance; anything farther is stale (task list already rolled to a
        new cycle) rather than a real match."""
        best_id, best_task, best_dist = None, None, tolerance_m
        for task_id, task in self.current_tasks_by_id.items():
            dist = float(np.linalg.norm(np.array(task["position"]) - pos))
            if dist < best_dist:
                best_id, best_task, best_dist = task_id, task, dist
        return best_id, best_task

    # -- Periodic disk refresh: risk map + prognosis -----------------------

    def _refresh_from_disk(self):
        risk_dir = os.path.join(self.run_dir, "damage_probability")
        paths = sorted(glob.glob(os.path.join(risk_dir, "*_risk_score.npy")))
        if paths:
            try:
                combined = np.maximum.reduce([np.load(p) for p in paths])
                with self.state_lock:
                    self.state.top_down_risk = np.max(combined, axis=2)
                    self.state.top_down_extent = (
                        ENV_MIN[0], ENV_MAX[0], ENV_MIN[1], ENV_MAX[1])
            except (ValueError, EOFError, OSError) as exc:
                self.get_logger().warn(f"risk map refresh skipped this tick: {exc}")

        prognosis_path = os.path.join(self.run_dir, "prognosis", "prognosis.json")
        if os.path.exists(prognosis_path):
            try:
                with open(prognosis_path) as f:
                    records = json.load(f)
                with self.state_lock:
                    self.state.prognosis_records = records
                    self.state.status_message = f"{len(records)} detection(s) tracked"
            except (json.JSONDecodeError, OSError) as exc:
                self.get_logger().warn(f"prognosis refresh skipped this tick: {exc}")

        tasks_path = os.path.join(self.run_dir, "tasks", "current_tasks.json")
        if os.path.exists(tasks_path):
            try:
                with open(tasks_path) as f:
                    tasks = json.load(f)
                with self.state_lock:
                    new_ids = {t["task_id"] for t in tasks}
                    if new_ids != set(self.current_tasks_by_id.keys()):
                        # cbba_allocator re-auctioned -- a new cycle's worth of
                        # tasks means a fresh coverage denominator, not adding
                        # onto the last cycle's counts (see cbba_node.py's
                        # _save_current_tasks for why this can't be a lifetime
                        # running total).
                        self.current_tasks_by_id = {t["task_id"]: t for t in tasks}
                        self.captured_task_ids_this_cycle = set()
                        self.state.tasks_captured = 0
                        self.state.cumulative_value = 0.0
                        self.state.total_tasks_estimate = len(tasks)
                        self.state.total_value = sum(t["value"] for t in tasks)
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                self.get_logger().warn(f"task list refresh skipped this tick: {exc}")

        with self.state_lock:
            self.state.tick += 1

    # -- Flask -------------------------------------------------------------

    def _start_flask(self, port):
        app = Flask(__name__)

        @app.route("/")
        def index():
            return render_template_string(INDEX_HTML)

        @app.route("/stream.mjpg")
        def stream():
            def gen():
                while True:
                    with self.state_lock:
                        frame = render_dashboard(self.state)
                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + buf.tobytes() + b"\r\n")
                    time.sleep(0.2)  # ~5fps -- a status dashboard, not a video feed
            return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

        @app.route("/prognosis.json")
        def prognosis():
            with self.state_lock:
                return jsonify(self.state.prognosis_records)

        thread = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=port, threaded=True,
                                    use_reloader=False),
            daemon=True)
        thread.start()


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
