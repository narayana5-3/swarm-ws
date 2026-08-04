"""
Live dashboard for the integrated mission demo (live_mission.py). Renders
a single dashboard frame from the current in-memory mission state: latest
camera + damage overlay, top-down coverage/risk map, swarm/task status, and
the degradation prognosis panel (urgency + estimated time-to-critical) --
built specifically to be the thing a judge watches update in real time
during a live demo, not a report figure.

Split deliberately into two layers:
  - render_dashboard(state) -> image: pure function, fully testable without
    a display (no cv2.imshow call inside it). This is what's actually
    tested here.
  - show_dashboard(state): the thin cv2.imshow wrapper around it, which
    needs a real display and is used by live_mission.py directly.

DashboardState is a plain mutable object the mission loop updates each
tick/detection -- no threading/locking here, since the mission loop is
expected to be single-threaded and synchronous with HoloOcean's own
blocking tick() calls (update state, then render, in the same loop
iteration -- no concurrent access to guard against).
"""

import numpy as np
import cv2

PANEL_SIZE = (1400, 900)
URGENCY_COLORS_BGR = {
    "CRITICAL": (40, 40, 220),
    "HIGH": (30, 120, 230),
    "MODERATE": (30, 200, 230),
    "LOW": (180, 180, 180),
    "NONE": (210, 210, 210),
}


class DashboardState:
    def __init__(self, agent_names, total_tasks_estimate=0):
        self.agent_names = agent_names
        self.agent_positions = {name: np.zeros(3) for name in agent_names}
        self.agent_current_task = {name: None for name in agent_names}
        self.latest_camera_frame = None
        self.latest_camera_agent = None
        self.latest_damage_overlay = None
        self.top_down_risk = None          # 2D array for imshow, or None
        self.top_down_extent = None        # (xmin, xmax, ymin, ymax)
        self.prognosis_records = []        # from degradation_estimator.py
        self.total_tasks_estimate = total_tasks_estimate
        self.tasks_captured = 0
        self.cumulative_value = 0.0
        self.total_value = 0.0
        self.tick = 0
        self.status_message = "Initializing..."
        self.recent_detections_log = []    # list of short strings, most recent first


def _fit_into(img, w, h):
    if img is None:
        canvas = np.full((h, w, 3), 230, dtype=np.uint8)
        cv2.putText(canvas, "no data yet", (w // 2 - 60, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (150, 150, 150), 1, cv2.LINE_AA)
        return canvas
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _panel_header(img, text, w):
    cv2.rectangle(img, (0, 0), (w, 28), (60, 60, 60), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _render_camera_panel(state, w, h):
    frame = state.latest_damage_overlay if state.latest_damage_overlay is not None \
        else state.latest_camera_frame
    panel = _fit_into(frame, w, h - 28)
    header = f"Camera feed ({state.latest_camera_agent or '--'})"
    full = np.full((h, w, 3), 255, dtype=np.uint8)
    full[28:] = panel
    return _panel_header(full, header, w)


def _render_risk_map_panel(state, w, h):
    if state.top_down_risk is not None:
        norm = np.clip(state.top_down_risk, 0, 1)
        heat = (norm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        heat_color = cv2.rotate(heat_color, cv2.ROTATE_90_COUNTERCLOCKWISE)
        panel_img = heat_color
    else:
        panel_img = None
    panel = _fit_into(panel_img, w, h - 28)

    if state.top_down_extent is not None and panel_img is not None:
        xmin, xmax, ymin, ymax = state.top_down_extent
        ih, iw = panel.shape[:2]
        for name, pos in state.agent_positions.items():
            fx = (pos[0] - xmin) / max(xmax - xmin, 1e-6)
            fy = 1 - (pos[1] - ymin) / max(ymax - ymin, 1e-6)
            px, py = int(fx * iw), int(fy * ih)
            if 0 <= px < iw and 0 <= py < ih:
                cv2.circle(panel, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)

    full = np.full((h, w, 3), 255, dtype=np.uint8)
    full[28:] = panel
    return _panel_header(full, "Shared risk map (top-down)", w)


def _render_status_panel(state, w, h):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img = _panel_header(img, "Swarm status", w)
    y = 45
    for name in state.agent_names:
        task = state.agent_current_task.get(name)
        task_str = f"-> {task}" if task else "(idle / patrolling)"
        pos = state.agent_positions.get(name, np.zeros(3))
        line = f"{name}: pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})  {task_str}"
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        y += 20

    y += 10
    cv2.putText(img, f"Tick: {state.tick}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (30, 30, 30), 1, cv2.LINE_AA)
    y += 20
    frac = state.cumulative_value / max(state.total_value, 1e-6) if state.total_value else 0
    cv2.putText(img, f"Coverage (this cycle): {state.tasks_captured}/{state.total_tasks_estimate} "
                       f"tasks ({frac*100:.0f}% risk value)", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    y += 25

    cv2.putText(img, "Recent events:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (30, 30, 30), 1, cv2.LINE_AA)
    y += 18
    for msg in state.recent_detections_log[:6]:
        cv2.putText(img, f"  {msg}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (90, 90, 90), 1, cv2.LINE_AA)
        y += 16

    return img


def _render_prognosis_panel(state, w, h):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img = _panel_header(img, "Degradation prognosis", w)
    y = 45
    if not state.prognosis_records:
        cv2.putText(img, "No detections above threshold yet.", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        return img

    for r in state.prognosis_records[:8]:
        band = r.get("urgency_band", "NONE")
        color = URGENCY_COLORS_BGR.get(band, (150, 150, 150))
        cv2.rectangle(img, (10, y - 12), (26, y + 2), color, -1)
        sev = r.get("current_severity", 0)
        if r.get("estimated_days_to_critical") is not None:
            detail = f"~{r['estimated_days_to_critical']:.0f}d to critical (trend)"
        elif r.get("recommended_reinspection_days") is not None:
            detail = f"re-inspect in {r['recommended_reinspection_days']}d"
        else:
            detail = ""
        line = f"[{band}] severity={sev:.2f}  {detail}"
        cv2.putText(img, line, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        y += 22

    return img


def render_dashboard(state):
    """Pure function: DashboardState -> BGR image (numpy array). No display
    calls -- fully testable headless."""
    W, H = PANEL_SIZE
    half_w, half_h = W // 2, H // 2

    top_left = _render_camera_panel(state, half_w, half_h)
    top_right = _render_risk_map_panel(state, half_w, half_h)
    bottom_left = _render_status_panel(state, half_w, half_h)
    bottom_right = _render_prognosis_panel(state, half_w, half_h)

    top = np.hstack([top_left, top_right])
    bottom = np.hstack([bottom_left, bottom_right])

    banner = np.full((34, W, 3), 20, dtype=np.uint8)
    cv2.putText(banner, "LIVE MISSION DASHBOARD -- Risk-Prioritized Belief-Fusion Swarm",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.vstack([banner, top, bottom])
    return canvas


def show_dashboard(state, window_name="Mission Dashboard"):
    """Thin display wrapper -- needs a real display, not used in headless
    testing. Non-blocking (waitKey(1)) so it doesn't stall the sim loop."""
    img = render_dashboard(state)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)


if __name__ == "__main__":
    # smoke test: render with a fully-populated fake state, no display needed
    state = DashboardState(agent_names=["auv0", "auv1", "auv2", "auv3"], total_tasks_estimate=40)
    state.agent_positions = {"auv0": np.array([1.2, -3.4, -5.0]), "auv1": np.array([10, 5, -2]),
                              "auv2": np.array([-5, 8, -6]), "auv3": np.array([3, -8, -3])}
    state.agent_current_task = {"auv0": "task_012", "auv1": None, "auv2": "task_005", "auv3": None}
    state.latest_camera_frame = np.random.randint(80, 180, size=(256, 256, 3), dtype=np.uint8)
    state.latest_camera_agent = "auv0"
    state.top_down_risk = np.random.uniform(0, 1, size=(80, 80))
    state.top_down_extent = (-50, 50, -50, 50)
    state.tick = 1234
    state.tasks_captured = 16
    state.cumulative_value = 1200.0
    state.total_value = 1900.0
    state.recent_detections_log = ["tick 1200: crack detected near (12.3, 4.1, -8.2), severity 0.81",
                                     "tick 1150: task_012 reassigned to auv0 (new high-risk detection)"]
    state.prognosis_records = [
        {"urgency_band": "CRITICAL", "current_severity": 0.92, "estimated_days_to_critical": 12},
        {"urgency_band": "HIGH", "current_severity": 0.78, "recommended_reinspection_days": 30},
        {"urgency_band": "MODERATE", "current_severity": 0.63, "recommended_reinspection_days": 90},
    ]

    img = render_dashboard(state)
    print(f"rendered dashboard: shape={img.shape}, dtype={img.dtype}")
    print(f"non-white/gray fraction: {(img < 200).any(axis=2).mean():.3f}")
    cv2.imwrite("/tmp/dashboard_smoke_test.png", img)
    print("saved -> /tmp/dashboard_smoke_test.png")
    assert img.shape[:2] == (PANEL_SIZE[1] + 34, PANEL_SIZE[0])
    print("OK")
