"""
Synthetic crack injection.

Generates (damaged_image, ground_truth_mask) pairs by procedurally
compositing crack-like patterns onto clean images -- works on ANY source
image, including frames already logged from HoloOcean, without touching the
sim scene itself (no Unreal Editor access needed).

Crack generation approach: a random-walk "spine" with locally-varying
width and branching, rendered with slight darkening + a thin bright rim
(common in real crack photos due to shadow/highlight at the edges), then
alpha-blended onto the source image. This is a simplification of techniques
used in synthetic-defect-injection literature for crack/pothole/corrosion
detection training data.

Usage:
    from crack_injection import inject_crack, batch_generate_from_frames

    damaged, mask = inject_crack(clean_image)

    # or, generate a whole training set from a HoloOcean log's camera frames:
    batch_generate_from_frames("logs/run_.../auv0_sensor_log.pkl",
                                out_dir="synthetic_dataset/auv0")
"""

import os
import pickle
import numpy as np
import cv2


def _random_walk_spine(h, w, n_points=40, step_size=None, rng=None):
    """A jittered random walk across the image, roughly diagonal, as the
    crack's centerline."""
    rng = rng or np.random.default_rng()
    step_size = step_size or max(h, w) / n_points

    start_edge = rng.integers(0, 4)
    if start_edge == 0:
        pos = np.array([rng.uniform(0, w), 0.0])
        base_dir = np.array([rng.uniform(-0.3, 0.3), 1.0])
    elif start_edge == 1:
        pos = np.array([w, rng.uniform(0, h)])
        base_dir = np.array([-1.0, rng.uniform(-0.3, 0.3)])
    elif start_edge == 2:
        pos = np.array([rng.uniform(0, w), h])
        base_dir = np.array([rng.uniform(-0.3, 0.3), -1.0])
    else:
        pos = np.array([0.0, rng.uniform(0, h)])
        base_dir = np.array([1.0, rng.uniform(-0.3, 0.3)])

    base_dir = base_dir / np.linalg.norm(base_dir)
    points = [pos.copy()]

    for _ in range(n_points):
        jitter = rng.normal(0, 0.35, size=2)
        direction = base_dir + jitter
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        base_dir = 0.85 * base_dir + 0.15 * direction  # smooth turning
        base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-6)
        pos = pos + base_dir * step_size
        points.append(pos.copy())
        if not (0 <= pos[0] <= w and 0 <= pos[1] <= h):
            break

    return np.array(points)


def _draw_crack_mask(h, w, n_cracks=1, rng=None):
    rng = rng or np.random.default_rng()
    mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(n_cracks):
        spine = _random_walk_spine(h, w, rng=rng)
        base_width = rng.uniform(1.0, 3.5)

        for i in range(len(spine) - 1):
            p1 = tuple(spine[i].astype(int))
            p2 = tuple(spine[i + 1].astype(int))
            local_width = max(1, int(round(base_width * rng.uniform(0.6, 1.3))))
            cv2.line(mask, p1, p2, color=1.0, thickness=local_width,
                     lineType=cv2.LINE_AA)

        # occasional short branch, common in real crack patterns
        if rng.random() < 0.5 and len(spine) > 10:
            branch_start_idx = rng.integers(len(spine) // 4, 3 * len(spine) // 4)
            branch_start = spine[branch_start_idx]
            branch_dir = rng.normal(0, 1, size=2)
            branch_dir /= (np.linalg.norm(branch_dir) + 1e-6)
            branch_len = rng.uniform(10, min(h, w) * 0.2)
            branch_end = branch_start + branch_dir * branch_len
            cv2.line(mask, tuple(branch_start.astype(int)),
                     tuple(branch_end.astype(int)),
                     color=1.0, thickness=max(1, int(base_width * 0.6)),
                     lineType=cv2.LINE_AA)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = np.clip(mask, 0, 1)
    return mask


def inject_crack(image, n_cracks=None, darkness=0.55, rim_brightness=0.15, rng=None):
    """
    image: HxWx3 uint8 or float array (any range; will be handled).
    Returns (damaged_image, mask) where mask is HxW float32 in [0,1] --
    the pixel-accurate ground truth for training/evaluation.
    """
    rng = rng or np.random.default_rng()
    image = np.asarray(image)
    was_float = image.dtype != np.uint8
    if was_float:
        img_min, img_max = image.min(), image.max()
        img_u8 = ((image - img_min) / max(img_max - img_min, 1e-6) * 255).astype(np.uint8)
    else:
        img_u8 = image.copy()

    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    elif img_u8.shape[2] == 4:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGBA2BGR)

    h, w = img_u8.shape[:2]
    n_cracks = n_cracks if n_cracks is not None else rng.integers(1, 3)
    mask = _draw_crack_mask(h, w, n_cracks=n_cracks, rng=rng)

    # Dark crack body + slightly bright rim (dilated mask minus mask itself)
    dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    rim = np.clip(dilated - mask, 0, 1)

    damaged = img_u8.astype(np.float32)
    for c in range(3):
        damaged[:, :, c] = damaged[:, :, c] * (1 - darkness * mask)
        damaged[:, :, c] = damaged[:, :, c] + 255 * rim_brightness * rim
    damaged = np.clip(damaged, 0, 255).astype(np.uint8)

    return damaged, mask


def batch_generate_from_frames(sensor_log_pkl, out_dir, max_frames=200,
                                damage_fraction=0.5, seed=0, prefix=""):
    """
    Reads a HoloOcean sensor log, uses its logged camera frames as base
    images, and generates a labeled synthetic-crack dataset:
        out_dir/images/<prefix>frame_XXXX.png
        out_dir/masks/<prefix>frame_XXXX.png   (0/255, crack=255)
    damage_fraction of frames get a crack injected; the rest are kept clean
    (all-zero mask) so the model also learns what "no damage" looks like.

    prefix: set this to e.g. "auv0_" when calling once per agent so you can
    write multiple agents' frames into the SAME out_dir to build a combined,
    larger training set without filename collisions.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    with open(sensor_log_pkl, "rb") as f:
        records = pickle.load(f)

    frames = [r["camera"] for r in records if r.get("camera") is not None]
    frames = frames[:max_frames]
    if not frames:
        print(f"[crack_injection] no camera frames found in {sensor_log_pkl}")
        return

    n_written = 0
    for i, frame in enumerate(frames):
        arr = np.asarray(frame)
        apply_damage = rng.random() < damage_fraction
        if apply_damage:
            damaged, mask = inject_crack(arr, rng=rng)
        else:
            damaged = arr
            if damaged.dtype != np.uint8:
                dmin, dmax = damaged.min(), damaged.max()
                damaged = ((damaged - dmin) / max(dmax - dmin, 1e-6) * 255).astype(np.uint8)
            if damaged.ndim == 3 and damaged.shape[2] == 4:
                damaged = cv2.cvtColor(damaged, cv2.COLOR_RGBA2BGR)
            mask = np.zeros(damaged.shape[:2], dtype=np.float32)

        img_path = os.path.join(out_dir, "images", f"{prefix}frame_{i:04d}.png")
        mask_path = os.path.join(out_dir, "masks", f"{prefix}frame_{i:04d}.png")
        cv2.imwrite(img_path, damaged)
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        n_written += 1

    print(f"[crack_injection] wrote {n_written} (image, mask) pairs -> {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor_log", type=str, required=True,
                         help="Path to a <agent>_sensor_log.pkl from swarm_sim.py")
    parser.add_argument("--out_dir", type=str, default="synthetic_dataset")
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--damage_fraction", type=float, default=0.5)
    parser.add_argument("--prefix", type=str, default="",
                         help="Filename prefix, e.g. 'auv0_' -- set differently "
                              "per agent to merge multiple agents' frames into "
                              "the same --out_dir without overwriting.")
    parser.add_argument("--seed", type=int, default=0,
                         help="Use a different seed per agent call too, or "
                              "you'll get identical crack patterns/placement "
                              "across agents (less training diversity).")
    args = parser.parse_args()
    batch_generate_from_frames(args.sensor_log, args.out_dir,
                                max_frames=args.max_frames,
                                damage_fraction=args.damage_fraction,
                                seed=args.seed, prefix=args.prefix)
