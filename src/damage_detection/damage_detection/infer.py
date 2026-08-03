"""
Runs a trained crack-detection checkpoint (see train.py) on camera frames,
producing a per-frame damage-probability map.

This file is referenced throughout the reused codebase (README.md,
damage_probability_mapping.py's module docstring) but was missing from the
handed-off package -- written fresh here, per the handoff doc's own
instruction ("thin wrapper around build_model() + a loaded checkpoint").
Preprocessing (BGR->RGB, /255, resize to the checkpoint's image_size) and
the checkpoint format ({"model_state_dict", "model_type", "image_size"})
match train.py/eval_visualize.py exactly, so a checkpoint trained with
train.py loads here unmodified.

Two ways to use it:
  - InferenceEngine: the reusable in-memory class, used by
    damage_detection's live ROS2 node (infer_node.py) to run inference on
    each incoming camera frame.
  - CLI (offline): replays a logged sensor pickle the same way the
    original HoloOcean-phase tooling did, for offline evaluation/demo-reel
    generation -- python infer.py --sensor_log <path> --model model.pt
    --out_dir damage_inference/auv0
"""

import os
import glob
import pickle
import argparse

import numpy as np
import cv2
import torch

from .model_factory import build_model

# torch defaults to intra-op parallelism across every visible core, which
# spiked CPU to 400%+ across 3 agents even at a throttled inference rate
# (confirmed live: this alone was slowing swarm_control's path planner from
# 6s/leg to 30s/leg by starving its executor of CPU). Live inference on a
# small CPU model doesn't need that; capped so this node leaves headroom for
# the rest of the pipeline instead of claiming every core for itself.
torch.set_num_threads(2)


class InferenceEngine:
    """Loads a checkpoint once, runs inference on individual BGR uint8 frames."""

    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        model_type = checkpoint.get("model_type", "small_unet")
        self.image_size = checkpoint.get("image_size", 256)
        self.model, _ = build_model(model_type, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = model_type

    @classmethod
    def untrained(cls, model_type="small_unet", device=None):
        """Builds a randomly-initialized model for pipeline wiring checks
        when no trained checkpoint exists yet -- predictions are
        meaningless, but every downstream shape/topic/fusion path can be
        exercised and verified before a real checkpoint is available."""
        engine = cls.__new__(cls)
        engine.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        engine.model, engine.image_size = build_model(model_type, pretrained=False)
        engine.model = engine.model.to(engine.device)
        engine.model.eval()
        engine.model_type = model_type
        return engine

    def infer(self, bgr_frame):
        """bgr_frame: HxWx3 uint8 (OpenCV/ROS convention). Returns an HxW
        float32 probability map resized back to the ORIGINAL frame
        resolution, so callers never need to know the model's internal
        input size."""
        orig_h, orig_w = bgr_frame.shape[:2]
        resized = cv2.resize(bgr_frame, (self.image_size, self.image_size))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        return cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def make_overlay(bgr_frame, prob_map, threshold=0.5):
    """Red overlay on detected-crack pixels, for quick visual sanity-checking."""
    overlay = bgr_frame.copy()
    mask = prob_map > threshold
    overlay[mask] = (0, 0, 255)
    return cv2.addWeighted(overlay, 0.4, bgr_frame, 0.6, 0)


def run_on_sensor_log(sensor_log_path, model_path, out_dir, agent_name=None):
    os.makedirs(out_dir, exist_ok=True)
    engine = InferenceEngine(model_path)

    with open(sensor_log_path, "rb") as f:
        records = pickle.load(f)

    results = []
    for i, record in enumerate(records):
        frame = record.get("camera")
        if frame is None:
            continue
        prob_map = engine.infer(frame)
        results.append({
            "tick": record.get("tick", i),
            "location": record.get("location"),
            "rotation": record.get("rotation"),
            "damage_prob_map": prob_map,
        })
        if i < 20:  # only save a handful of overlay images, not the whole run
            overlay = make_overlay(frame, prob_map)
            cv2.imwrite(os.path.join(out_dir, f"overlay_{i:04d}.png"), overlay)

    out_path = os.path.join(out_dir, "damage_inference.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[infer] {len(results)} frames processed -> {out_path}")
    return out_path


def find_latest_run_dir(logs_root="logs"):
    runs = sorted(glob.glob(os.path.join(logs_root, "run_*")))
    if not runs:
        raise FileNotFoundError(f"No run_* directories found under {logs_root}")
    return runs[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor_log", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    run_on_sensor_log(args.sensor_log, args.model, args.out_dir)
