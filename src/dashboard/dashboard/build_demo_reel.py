"""
Assembles every visual asset this project produces into ONE polished,
continuous demo video with title/section cards -- built specifically for
presenting to a judging panel, where a single watchable file that tells the
whole story beats a folder of separate outputs someone has to be walked
through manually.

Gracefully skips any section whose source asset doesn't exist yet (prints
a note rather than failing), so you can run this at any point in the
pipeline and get the best reel possible from what's been generated so far,
then re-run it again later once more pieces exist.

Sections, in order (skipped if the underlying file is missing):
  1. Title card
  2. AUV point-of-view footage (from make_demo_video.py)
  3. Shared occupancy map building up over time (from make_demo_video.py)
  4. Crack detection: real-footage overlay panels (from damage_detection/infer.py)
  5. Shared damage probability + uncertainty heatmap (static, from
     damage_probability_mapping.py)
  6. Risk-prioritized task allocation (static, from cbba_task_allocation.py)
  7. Iterative adaptive task reallocation -- ANIMATED (from
     animate_task_allocation.py) -- the centerpiece section
  8. Digital twin -- screenshot + note that the interactive version is
     available separately (video can't embed a live rotatable 3D view)
  9. Closing summary card with key numbers

Run:
    python build_demo_reel.py --run_dir logs/run_<ts> \
        --damage_inference_dir damage_detection/damage_inference/auv0
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2

TARGET_SIZE = (1280, 720)
FPS = 24
SECTION_CARD_SECONDS = 2.0
STATIC_IMAGE_SECONDS = 4.0
TITLE_CARD_SECONDS = 3.5
CLOSING_CARD_SECONDS = 5.0


class ReelWriter:
    def __init__(self, out_path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, FPS, TARGET_SIZE)
        self.out_path = out_path
        self.sections_included = []
        self.sections_skipped = []

    def write_frame(self, img):
        if img.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
            img = cv2.resize(img, TARGET_SIZE)
        self.writer.write(img)

    def write_video_file(self, path, caption=None, caption_seconds=1.5):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        n_written = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, TARGET_SIZE)
            if caption and n_written < caption_seconds * FPS:
                frame = _overlay_caption(frame, caption)
            self.writer.write(frame)
            n_written += 1
        cap.release()
        return n_written > 0

    def write_static_image(self, path, caption=None, seconds=STATIC_IMAGE_SECONDS):
        img = cv2.imread(path)
        if img is None:
            return False
        img = _fit_image(img, TARGET_SIZE)
        if caption:
            img = _overlay_caption(img, caption, persistent=True)
        n_frames = int(seconds * FPS)
        for _ in range(n_frames):
            self.write_frame(img)
        return True

    def write_title_card(self, lines, seconds=SECTION_CARD_SECONDS, big=False):
        img = np.full((TARGET_SIZE[1], TARGET_SIZE[0], 3), 255, dtype=np.uint8)
        y = TARGET_SIZE[1] // 2 - (len(lines) * 30) // 2
        for i, (text, scale, color, thickness) in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            x = (TARGET_SIZE[0] - tw) // 2
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        thickness, cv2.LINE_AA)
            y += th + 25
        n_frames = int(seconds * FPS)
        for _ in range(n_frames):
            self.write_frame(img)

    def add_section(self, name, found):
        if found:
            self.sections_included.append(name)
        else:
            self.sections_skipped.append(name)
            print(f"[demo_reel] SKIPPED '{name}' -- source asset not found yet.")

    def finish(self):
        self.writer.release()
        print(f"\n[demo_reel] included: {self.sections_included}")
        print(f"[demo_reel] skipped (asset not found): {self.sections_skipped}")
        print(f"[demo_reel] saved -> {self.out_path}")


def _fit_image(img, target_size):
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
    x_off, y_off = (tw - new_w) // 2, (th - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _overlay_caption(img, text, persistent=False):
    img = img.copy()
    h, w = img.shape[:2]
    bar_h = 50
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    alpha = 0.75 if persistent else 0.65
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.putText(img, text, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def build_reel(run_dir, damage_inference_dir=None, out_path=None):
    out_path = out_path or os.path.join(run_dir, "demo_reel.mp4")
    reel = ReelWriter(out_path)

    # 1. Title card
    reel.write_title_card([
        ("Risk-Prioritized Distributed Belief-Fusion Swarm", 0.85, (20, 20, 20), 2),
        ("Autonomous Underwater Infrastructure Inspection", 0.6, (80, 80, 80), 1),
    ], seconds=TITLE_CARD_SECONDS)

    # 2. AUV point-of-view footage
    pov_paths = sorted(glob.glob(os.path.join(run_dir, "*_pov.mp4")))
    if pov_paths:
        reel.write_title_card([("Multi-AUV Swarm -- Onboard Camera Feed", 0.7, (20, 20, 20), 2)])
        found = reel.write_video_file(pov_paths[0], caption="Live AUV camera feed during autonomous sweep")
    else:
        found = False
    reel.add_section("AUV POV footage", found)

    # 3. Occupancy buildup
    occ_anim_path = os.path.join(run_dir, "occupancy_buildup.mp4")
    if os.path.exists(occ_anim_path):
        reel.write_title_card([("Distributed Occupancy Mapping", 0.7, (20, 20, 20), 2),
                                ("Delta-compressed belief fusion across the swarm", 0.5, (90, 90, 90), 1)])
        found = reel.write_video_file(occ_anim_path, caption="Shared structural map building up in real time")
    else:
        found = False
    reel.add_section("Occupancy buildup animation", found)

    # 4. Crack detection overlays
    if damage_inference_dir:
        overlay_paths = sorted(glob.glob(os.path.join(damage_inference_dir, "overlays", "*.png")))[:8]
        if overlay_paths:
            reel.write_title_card([("AI-Based Crack Detection", 0.7, (20, 20, 20), 2),
                                    ("Real-time inference on live camera footage", 0.5, (90, 90, 90), 1)])
            for p in overlay_paths:
                reel.write_static_image(p, caption="Damage probability overlay (red/yellow = high confidence)",
                                          seconds=1.2)
            found = True
        else:
            found = False
    else:
        found = False
    reel.add_section("Crack detection overlays", found)

    # 5. Damage probability + uncertainty heatmap
    dmg_plot = os.path.join(run_dir, "damage_probability", "damage_and_uncertainty_topdown.png")
    if os.path.exists(dmg_plot):
        reel.write_title_card([("Shared Damage-Probability Map + Uncertainty Heatmap", 0.65, (20, 20, 20), 2)])
        found = reel.write_static_image(dmg_plot, caption="Fused across the swarm via acoustic-mesh belief sharing")
    else:
        found = False
    reel.add_section("Damage probability + uncertainty map", found)

    # 6. Risk-prioritized task allocation (static)
    cbba_plot = os.path.join(run_dir, "task_allocation", "cbba_assignment.png")
    if os.path.exists(cbba_plot):
        reel.write_title_card([("Risk-Prioritized Task Allocation", 0.7, (20, 20, 20), 2),
                                ("Decentralized consensus auction, CBBA-inspired", 0.5, (90, 90, 90), 1)])
        found = reel.write_static_image(cbba_plot, caption="Each agent bids on inspection targets by risk value")
    else:
        found = False
    reel.add_section("Task allocation (static)", found)

    # 7. Iterative adaptive reallocation -- ANIMATED (centerpiece)
    iter_anim = os.path.join(run_dir, "task_allocation", "iterative_allocation_animation.mp4")
    if os.path.exists(iter_anim):
        reel.write_title_card([("Adaptive Multi-Round Task Reallocation", 0.7, (20, 20, 20), 2),
                                ("Watch the swarm cover the full detected risk backlog", 0.5, (90, 90, 90), 1)])
        found = reel.write_video_file(iter_anim)
    else:
        found = False
    reel.add_section("Iterative reallocation animation", found)

    # 8. Digital twin (screenshot only -- interactive HTML can't embed in video)
    twin_dir = os.path.join(run_dir, "digital_twin")
    twin_screenshot = None
    if os.path.isdir(twin_dir):
        pngs = glob.glob(os.path.join(twin_dir, "*.png"))
        twin_screenshot = pngs[0] if pngs else None
    if twin_screenshot:
        reel.write_title_card([("Digital Twin: Structural Health Map", 0.7, (20, 20, 20), 2),
                                ("Full interactive 3D version available separately", 0.5, (90, 90, 90), 1)])
        found = reel.write_static_image(twin_screenshot,
                                          caption="Interactive HTML version: rotate, zoom, hover for details")
    else:
        found = False
    reel.add_section("Digital twin", found)
    if not found and os.path.exists(os.path.join(twin_dir, "digital_twin.html")):
        print("[demo_reel] NOTE: digital_twin.html exists but no screenshot was found. "
              "Open it and save a screenshot as digital_twin/*.png to include it in the reel.")

    # 9. Closing summary card
    summary_lines = [("Demonstration Complete", 0.8, (20, 20, 20), 2)]
    stats = _gather_stats(run_dir)
    for stat_line in stats:
        summary_lines.append((stat_line, 0.5, (70, 70, 70), 1))
    reel.write_title_card(summary_lines, seconds=CLOSING_CARD_SECONDS)

    reel.finish()
    return out_path


def _gather_stats(run_dir):
    lines = []
    summary_path = os.path.join(run_dir, "task_allocation", "iterative_allocation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        lines.append(f"{s['total_tasks']} risk points detected, "
                      f"{s['fraction_value_captured']*100:.0f}% covered in "
                      f"{s['n_rounds_run']} dispatch rounds")
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--damage_inference_dir", type=str, default=None,
                         help="e.g. damage_detection/damage_inference/auv0")
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    build_reel(args.run_dir, args.damage_inference_dir, args.out_path)
