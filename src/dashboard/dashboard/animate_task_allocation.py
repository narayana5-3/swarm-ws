"""
Animates the iterative task-reallocation process (iterative_task_allocation.py's
saved history) into a video: watch the swarm's agents move out to their
CBBA-assigned tasks round by round, tasks lighting up as "captured," with a
running coverage counter -- built specifically to be a compelling thing to
actually watch in a live demo, not just a static plot.

Run (after iterative_task_allocation.py has produced a history file):
    python animate_task_allocation.py --run_dir logs/run_<ts>
"""

import os
import pickle
import argparse
import numpy as np
import cv2

from cbba_task_allocation import get_last_positions
from occupancy_mapping import find_latest_run_dir

FRAME_SIZE = (1000, 1000)
FRAMES_PER_ROUND_TRAVEL = 25    # smooth movement frames while agents travel to tasks
FRAMES_PER_ROUND_PAUSE = 20     # hold frames showing the round's summary
FINAL_HOLD_FRAMES = 60
FPS = 20

AGENT_COLORS_BGR = [
    (200, 120, 30), (30, 140, 220), (40, 180, 40), (30, 30, 200),
    (180, 40, 180), (40, 180, 200),
]


def load_history(run_dir):
    path = os.path.join(run_dir, "task_allocation", "iterative_allocation_history.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {path} -- run iterative_task_allocation.py first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["round_history"], data["all_tasks"], data["total_task_value"]


def _world_to_pixel(pos, bounds, margin=80):
    (xmin, xmax, ymin, ymax) = bounds
    x, y = pos[0], pos[1]
    w, h = FRAME_SIZE[0] - 2 * margin, FRAME_SIZE[1] - 2 * margin
    px = margin + int((x - xmin) / max(xmax - xmin, 1e-6) * w)
    py = FRAME_SIZE[1] - margin - int((y - ymin) / max(ymax - ymin, 1e-6) * h)
    return px, py


def animate(run_dir, out_path=None):
    round_history, all_tasks, total_value = load_history(run_dir)
    tasks_by_id = {t["task_id"]: t for t in all_tasks}

    agent_name_set = set()
    for r in round_history:
        agent_name_set |= set(r["assigned"].keys())
        agent_name_set |= set(r["agent_positions_after"].keys())
    agent_names = sorted(agent_name_set)
    if not agent_names:
        raise ValueError("No agents found in history -- was the run empty?")

    initial_positions = get_last_positions(run_dir, agent_names)
    colors = {name: AGENT_COLORS_BGR[i % len(AGENT_COLORS_BGR)] for i, name in enumerate(agent_names)}

    all_positions = [np.array(t["position"]) for t in all_tasks] + \
        [np.array(p) for p in initial_positions.values()]
    all_xy = np.array([[p[0], p[1]] for p in all_positions])
    xmin, ymin = all_xy.min(axis=0) - 5
    xmax, ymax = all_xy.max(axis=0) + 5
    bounds = (xmin, xmax, ymin, ymax)

    out_path = out_path or os.path.join(run_dir, "task_allocation", "iterative_allocation_animation.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, FRAME_SIZE)

    captured_task_ids = set()
    captured_color = {}
    agent_positions = dict(initial_positions)
    cumulative_value = [0.0]  # mutable box so the nested draw_frame can read the live value

    def draw_frame(agent_positions_now, round_text):
        img = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 255, dtype=np.uint8)

        for t in all_tasks:
            px, py = _world_to_pixel(t["position"], bounds)
            radius = max(3, int(3 + t["value"] / 15))
            if t["task_id"] in captured_task_ids:
                color = captured_color.get(t["task_id"], (150, 150, 150))
                cv2.circle(img, (px, py), radius, color, -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), radius, (60, 60, 60), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (px, py), radius, (210, 210, 210), -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), radius, (160, 160, 160), 1, cv2.LINE_AA)

        for name, pos in agent_positions_now.items():
            px, py = _world_to_pixel(pos, bounds)
            color = colors[name]
            pts = np.array([[px, py - 12], [px - 10, py + 8], [px + 10, py + 8]], dtype=np.int32)
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
            cv2.putText(img, name, (px + 14, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (30, 30, 30), 1, cv2.LINE_AA)

        cv2.rectangle(img, (0, 0), (FRAME_SIZE[0], 60), (245, 245, 245), -1)
        cv2.putText(img, "Risk-Prioritized Iterative Task Reallocation", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, round_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

        frac = min(cumulative_value[0] / max(total_value, 1e-6), 1.0)
        bar_x0, bar_y0, bar_w, bar_h = 20, FRAME_SIZE[1] - 40, FRAME_SIZE[0] - 40, 20
        cv2.rectangle(img, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (200, 200, 200), 1)
        cv2.rectangle(img, (bar_x0, bar_y0), (bar_x0 + int(bar_w * frac), bar_y0 + bar_h),
                       (40, 180, 40), -1)
        cv2.putText(img, f"Risk coverage: {frac*100:.0f}%", (bar_x0, bar_y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

        return img

    # opening title card
    for _ in range(30):
        img = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 255, dtype=np.uint8)
        cv2.putText(img, "Risk-Prioritized Distributed Belief-Fusion Swarm", (40, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, "Iterative Adaptive Task Reallocation", (40, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(img, f"{len(all_tasks)} detected high-risk points  |  {len(agent_names)} AUVs",
                    (40, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
        writer.write(img)

    for round_data in round_history:
        round_num = round_data["round"]
        assigned = round_data["assigned"]

        agent_routes = {}
        for name in agent_names:
            route = [agent_positions[name]]
            for task_id in assigned.get(name, []):
                route.append(np.array(tasks_by_id[task_id]["position"]))
            agent_routes[name] = route

        max_legs = max((len(r) - 1 for r in agent_routes.values()), default=0)
        for leg in range(max_legs):
            for frame_i in range(FRAMES_PER_ROUND_TRAVEL):
                t = frame_i / FRAMES_PER_ROUND_TRAVEL
                current_positions = {}
                for name, route in agent_routes.items():
                    if leg < len(route) - 1:
                        current_positions[name] = route[leg] + t * (route[leg + 1] - route[leg])
                    else:
                        current_positions[name] = route[-1]
                round_text = f"Round {round_num}/{len(round_history)}  --  in progress"
                img = draw_frame(current_positions, round_text)
                writer.write(img)

            for name, task_ids in assigned.items():
                if leg < len(task_ids):
                    tid = task_ids[leg]
                    captured_task_ids.add(tid)
                    captured_color[tid] = colors[name]
                    cumulative_value[0] += tasks_by_id[tid]["value"]

        for name in agent_names:
            if agent_routes[name]:
                agent_positions[name] = agent_routes[name][-1]

        round_summary = (f"Round {round_num}/{len(round_history)} complete: "
                          f"{round_data['n_assigned']} tasks captured this round")
        for _ in range(FRAMES_PER_ROUND_PAUSE):
            img = draw_frame(agent_positions, round_summary)
            writer.write(img)

    final_text = f"All rounds complete -- {len(captured_task_ids)}/{len(all_tasks)} tasks covered"
    for _ in range(FINAL_HOLD_FRAMES):
        img = draw_frame(agent_positions, final_text)
        writer.write(img)

    writer.release()
    print(f"[animate] saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    print(f"Using run_dir: {run_dir}")
    animate(run_dir, args.out_path)
