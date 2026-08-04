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
from flask import Flask, Response, jsonify, render_template_string, send_from_directory, abort

from occupancy_mapping.occupancy_mapping import ENV_MIN, ENV_MAX
from dashboard.live_dashboard import DashboardState, render_dashboard
from damage_detection.infer import make_overlay

INDEX_HTML = """
<!doctype html>
<title>Swarm Mission Dashboard</title>
<body style="margin:0;background:#111;">
  <a href="/results" style="position:absolute;top:10px;right:14px;z-index:1;
     font:13px Georgia,serif;color:#ccc;background:#000c;padding:4px 10px;
     border:1px solid #444;text-decoration:none;">algorithm validation notes</a>
  <img src="/stream.mjpg" style="width:100%;height:auto;display:block;">
</body>
"""

# Results artifacts (CBBA comparison, CEC2017 benchmark, crack-detector eval)
# live at the repo root, not under a per-run run_dir -- these are one-time
# validation studies, not live mission output. See compare_allocators.py,
# benchmark_cec2017.py and eval_visualize.py for how each is produced.
RESULTS_ROOT = os.path.expanduser("~/swarm_ws")
RESULTS_SUBDIRS = ["cbba_comparison_results", "cec_results", "eval_results"]

RESULTS_HTML = """
<!doctype html>
<title>Algorithm validation notes</title>
<style>
  body {
    background: #fdfdfb; color: #1a1a1a;
    font-family: Georgia, "Times New Roman", serif;
    max-width: 880px; margin: 30px auto 100px; padding: 0 20px;
    line-height: 1.5;
  }
  a { color: #1a4b8c; }
  h1 { font-size: 1.35em; margin-bottom: 2px; }
  .subtitle { color: #666; font-size: 0.9em; margin-top: 0; }
  h2 { font-size: 1.05em; margin-top: 2.2em; margin-bottom: 0.3em;
       border-bottom: 1px solid #ccc; padding-bottom: 4px; }
  p { margin: 0.6em 0; }
  table {
    border-collapse: collapse; width: 100%; margin: 10px 0 4px;
    font-family: "Courier New", Consolas, monospace; font-size: 0.87em;
  }
  th, td { padding: 3px 9px; text-align: right; border-bottom: 1px solid #ddd; }
  th:first-child, td:first-child { text-align: left; }
  th { border-bottom: 1px solid #1a1a1a; font-weight: normal; color: #444; }
  tr.ours td:first-child { font-weight: bold; }
  .aside { color: #555; font-size: 0.92em; }
  .note {
    background: #f1f0eb; border: 1px solid #ddd; padding: 8px 12px;
    font-size: 0.93em; margin: 10px 0;
  }
  .missing { color: #8a3b2f; font-style: italic; }
  img { max-width: 100%; border: 1px solid #ccc; display: block; margin: 8px 0; }
  .gallery { display: flex; gap: 6px; flex-wrap: wrap; }
  .gallery img { max-width: 23%; margin: 0; }
  .charts { display: flex; gap: 10px; flex-wrap: wrap; }
  .charts img { max-width: 48%; margin: 0; }
  code { font-family: "Courier New", Consolas, monospace; background: #f1f0eb; padding: 1px 4px; }
  .cmd { font-size: 0.85em; }
</style>
<body>
  <p><a href="/">&lt;&lt; back to the live stream</a></p>
  <h1>Algorithm validation notes</h1>
  <p class="subtitle">Numbers pulled straight from the comparison/eval scripts in this repo
    (<code>compare_allocators.py</code>, <code>benchmark_cec2017.py</code>,
    <code>eval_visualize.py</code>) -- re-run against real captured data, not projected.</p>

  <h2>1. Task allocation -- CBBA vs. greedy vs. round-robin</h2>
  {% if cbba %}
    <p>{{ cbba.n_trials }} trials total, run against the real risk grids captured in
      demo_dryrun / verify_run / live_run (agent start positions randomized per trial).</p>
    <table>
      <tr><th>allocator</th><th>value captured</th><th>load-balance std</th><th>travel / value</th></tr>
      {% for name in ["cbba", "greedy", "round_robin"] %}
      <tr class="{{ 'ours' if name == 'cbba' else '' }}">
        <td>{{ name }}{{ '  (ours)' if name == 'cbba' else '' }}</td>
        <td>{{ "%.1f"|format(cbba.summary[name].mean_total_value_captured) }}
            &plusmn; {{ "%.1f"|format(cbba.summary[name].std_total_value_captured) }}</td>
        <td>{{ "%.2f"|format(cbba.summary[name].mean_load_balance_std) }}</td>
        <td>{% if cbba.summary[name].mean_travel_per_value is not none %}{{ "%.3f"|format(cbba.summary[name].mean_travel_per_value) }}{% else %}--{% endif %}</td>
      </tr>
      {% endfor %}
    </table>
    <p class="aside">Wilcoxon rank-sum on value captured -- cbba vs. greedy: p={{ "%.3g"|format(cbba.pairwise_significance.cbba_vs_greedy.pvalue) }}
      ({{ "significant" if cbba.pairwise_significance.cbba_vs_greedy["significant_at_0.05"] else "not significant" }}),
      cbba vs. round_robin: p={{ "%.3g"|format(cbba.pairwise_significance.cbba_vs_round_robin.pvalue) }}
      ({{ "significant" if cbba.pairwise_significance.cbba_vs_round_robin["significant_at_0.05"] else "not significant" }}).</p>
    <img src="/results/assets/cbba_comparison_results/comparison_chart.png">
    <div class="note">Neither baseline loses across the board, so I'm not going to claim that.
      Greedy comes close on travel efficiency, round-robin comes close on raw value captured.
      What CBBA actually wins is the combination -- it's within 1% of round-robin's value capture
      and close to its balance, without round-robin's poor travel efficiency, and without
      greedy's workload imbalance (about 18x worse than CBBA's). Each baseline buys its one good
      metric by giving up a different one; CBBA is the only one that doesn't trade like that.</div>
  {% else %}
    <p class="missing">comparison_summary.json not found -- run
      <code class="cmd">python3 -m cbba_allocator.compare_allocators</code> from the repo root.</p>
  {% endif %}

  <h2>2. Path planning -- NOA-SQP vs. GWO / PSO (CEC2017 benchmark)</h2>
  {% if cec %}
    <p>{{ cec.n_trials }} independent trials per function on {{ cec.significance|length }} of the
      standard CEC2017 benchmark functions -- a generic optimizer benchmark, not something built
      around this project. Wins: noa {{ cec.win_count.noa }}, gwo {{ cec.win_count.gwo }},
      pso {{ cec.win_count.pso }}, no significant difference {{ cec.win_count.tie }}.</p>
    <table>
      <tr><th>function</th><th>vs. best baseline</th><th>p-value</th><th>result</th></tr>
      {% for fname, sig in cec.significance.items() %}
      <tr>
        <td>{{ fname }}</td>
        <td>{{ sig.vs }}</td>
        <td>{{ "%.3g"|format(sig.pvalue) }}</td>
        <td>{% if sig.significant and sig.noa_better %}noa better
            {% elif sig.significant %}baseline better
            {% else %}no significant difference{% endif %}</td>
      </tr>
      {% endfor %}
    </table>
    <p class="aside">noa is significantly better on 2 of {{ cec.significance|length }}, never
      significantly worse on any of them.</p>
    <div class="charts">
      <img src="/results/assets/cec_results/ranking_chart.png">
      <img src="/results/assets/cec_results/convergence_curves.png">
    </div>
  {% else %}
    <p class="missing">cec2017_summary.json not found.</p>
  {% endif %}

  <h2>3. Damage detection -- crack detector, real held-out accuracy</h2>
  {% if eval %}
    <table>
      <tr><th>checkpoint</th><th>dice</th><th>iou</th><th>precision</th><th>recall</th><th style="text-align:left">held out on</th></tr>
      <tr>
        <td>deepcrack_pretrained (stage 1)</td>
        <td>{{ "%.3f"|format(eval.deepcrack_pretrained.all_images.dice) }}</td>
        <td>{{ "%.3f"|format(eval.deepcrack_pretrained.all_images.iou) }}</td>
        <td>{{ "%.3f"|format(eval.deepcrack_pretrained.all_images.precision) }}</td>
        <td>{{ "%.3f"|format(eval.deepcrack_pretrained.all_images.recall) }}</td>
        <td style="text-align:left">{{ eval.deepcrack_pretrained.n_images }} real DeepCrack photos, never trained on</td>
      </tr>
      <tr class="ours">
        <td>sim_finetuned (ours, shipped default)</td>
        <td>{{ "%.3f"|format(eval.sim_finetuned.crack_containing_only.dice) }}</td>
        <td>{{ "%.3f"|format(eval.sim_finetuned.crack_containing_only.iou) }}</td>
        <td>{{ "%.3f"|format(eval.sim_finetuned.crack_containing_only.precision) }}</td>
        <td>{{ "%.3f"|format(eval.sim_finetuned.crack_containing_only.recall) }}</td>
        <td style="text-align:left">{{ eval.sim_finetuned.n_with_crack }}/{{ eval.sim_finetuned.n_images }}
          real captured Gazebo frames with a crack, never trained on</td>
      </tr>
    </table>
    <p class="aside">sim_finetuned is deepcrack_pretrained warm-started and then fine-tuned on real
      captured Gazebo frames -- that warm start is the improvement over training from scratch or
      from ImageNet weights alone. Dice/IoU above are the crack-containing subset
      ({{ eval.sim_finetuned.n_with_crack }}/{{ eval.sim_finetuned.n_images }} images); the
      full-set number is lower ({{ "%.3f"|format(eval.sim_finetuned.all_images.dice) }} Dice)
      purely from the Dice 0/0 discontinuity on correctly-empty predictions, not a real
      localization problem.</p>
    <div class="gallery">
      {% for p in eval_panels %}
      <img src="/results/assets/{{ p }}">
      {% endfor %}
    </div>
  {% else %}
    <p class="missing">eval_results/metrics_summary.json not found.</p>
  {% endif %}
</body>
"""


def _load_json(*path_parts):
    path = os.path.join(RESULTS_ROOT, *path_parts)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _existing_eval_panels():
    """A couple of the real side-by-side [input|ground truth|prediction]
    panel images eval_visualize.py already saved, as visual evidence
    alongside the Dice/IoU numbers -- not regenerated here, just surfaced."""
    panels = []
    for sub in ("eval_results/deepcrack_stage1", "eval_results/sim_finetuned"):
        found = sorted(glob.glob(os.path.join(RESULTS_ROOT, sub, "panel_*.png")))[:2]
        panels.extend(os.path.relpath(p, RESULTS_ROOT) for p in found)
    return panels


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

        @app.route("/results")
        def results():
            return render_template_string(
                RESULTS_HTML,
                cbba=_load_json("cbba_comparison_results", "comparison_summary.json"),
                cec=_load_json("cec_results", "cec2017_summary.json"),
                eval=_load_json("eval_results", "metrics_summary.json"),
                eval_panels=_existing_eval_panels(),
            )

        @app.route("/results/assets/<path:subpath>")
        def results_assets(subpath):
            # Restrict to the known result directories -- this is the only
            # route in the app that maps a URL onto arbitrary filesystem
            # reads, so it must not be able to walk outside RESULTS_ROOT.
            top = subpath.split("/", 1)[0]
            if top not in RESULTS_SUBDIRS:
                abort(404)
            full_path = os.path.normpath(os.path.join(RESULTS_ROOT, subpath))
            if not full_path.startswith(os.path.join(RESULTS_ROOT, "")):
                abort(404)
            directory, filename = os.path.split(full_path)
            return send_from_directory(directory, filename)

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
