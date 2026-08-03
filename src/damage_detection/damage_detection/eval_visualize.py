"""
Evaluate a trained crack detector against KNOWN ground truth -- the
synthetic-injected dataset itself, not unlabeled real camera footage.

Checking predictions against real (uninjected) sim frames, like the
overlays in infer.py's output, can't actually tell you whether the model
localizes cracks correctly, because there's no ground truth crack there to
compare against. This script runs on data where the true crack mask is
known, so it gives an honest answer.

Outputs:
  - Quantitative Dice / IoU / precision / recall over the dataset.
  - Side-by-side panel images: [input | ground truth | predicted probability
    heatmap | predicted binary mask] for visual inspection.

Usage:
    python eval_visualize.py --data_dir synthetic_dataset --model model.pt \
        --out_dir eval_results --num_visualize 12
"""

import os
import glob
import argparse
import numpy as np
import cv2
import torch

from model_factory import build_model


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "small_unet")  # backward-compatible default
    image_size = checkpoint.get("image_size", 256)
    model, _ = build_model(model_type, pretrained=False)  # weights come from checkpoint, not download
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, image_size


def compute_metrics(pred_binary, gt_binary, eps=1e-6):
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_binary.sum() + gt_binary.sum() + eps)
    precision = (intersection + eps) / (pred_binary.sum() + eps)
    recall = (intersection + eps) / (gt_binary.sum() + eps)
    return dice, iou, precision, recall


def evaluate(data_dir, model_path, out_dir, num_visualize=12, threshold=0.5,
             device=None, val_list_path=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_model(model_path, device)

    if val_list_path is None:
        # auto-detect the held-out list train.py saves next to the model
        guess = os.path.splitext(model_path)[0] + "_val_files.txt"
        if os.path.exists(guess):
            val_list_path = guess

    if val_list_path is not None:
        with open(val_list_path) as f:
            image_paths = [line.strip() for line in f if line.strip()]
        print(f"[eval] using held-out file list: {val_list_path} "
              f"({len(image_paths)} images NEVER seen during training)")
    else:
        image_paths = sorted(sum([glob.glob(os.path.join(data_dir, "images", ext))
                                   for ext in ("*.png", "*.jpg", "*.jpeg")], []))
        print(f"[eval] WARNING: no held-out file list found -- evaluating on "
              f"ALL of {data_dir}, which may include images the model was "
              f"TRAINED on. Results will look better than true generalization. "
              f"Pass --val_list explicitly, or use a separate dataset dir "
              f"generated with a different --seed for an honest check.")

    if not image_paths:
        raise FileNotFoundError(f"No images found for evaluation.")

    os.makedirs(out_dir, exist_ok=True)

    all_dice, all_iou, all_precision, all_recall = [], [], [], []
    n_gt_has_crack = 0
    n_visualized = 0

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            mask_path = img_path.replace(
                os.sep + "images" + os.sep, os.sep + "masks" + os.sep)
            mask_path = os.path.splitext(mask_path)[0] + ".png"
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            img_resized = cv2.resize(img, (image_size, image_size))
            gt_resized = cv2.resize(gt_mask, (image_size, image_size),
                                     interpolation=cv2.INTER_NEAREST)

            rgb = img_resized[:, :, ::-1].astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_binary = prob > threshold
            gt_binary = gt_resized > 127

            if gt_binary.sum() > 0:
                n_gt_has_crack += 1

            dice, iou, precision, recall = compute_metrics(pred_binary, gt_binary)
            all_dice.append(dice)
            all_iou.append(iou)
            all_precision.append(precision)
            all_recall.append(recall)

            if n_visualized < num_visualize:
                _save_panel(img_resized, gt_resized, prob, pred_binary,
                            os.path.join(out_dir, f"panel_{idx:04d}.png"),
                            dice, iou)
                n_visualized += 1

    print(f"[eval] {len(image_paths)} images evaluated "
          f"({n_gt_has_crack} contain a crack per ground truth)")
    print(f"[eval] mean Dice:      {np.mean(all_dice):.4f}")
    print(f"[eval] mean IoU:       {np.mean(all_iou):.4f}")
    print(f"[eval] mean Precision: {np.mean(all_precision):.4f}")
    print(f"[eval] mean Recall:    {np.mean(all_recall):.4f}")
    print(f"[eval] {n_visualized} side-by-side panels saved -> {out_dir}")
    print("[eval] NOTE: precision/recall on all-background images (no "
          "injected crack) can be undefined/misleading -- the mean above "
          "includes them. For a cleaner crack-only score, filter to images "
          "where the ground truth actually contains a crack.")

    return {
        "dice": float(np.mean(all_dice)),
        "iou": float(np.mean(all_iou)),
        "precision": float(np.mean(all_precision)),
        "recall": float(np.mean(all_recall)),
    }


def _save_panel(img, gt_mask, prob_map, pred_binary, out_path, dice, iou):
    h, w = img.shape[:2]

    gt_vis = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

    heat = (prob_map * 255).astype(np.uint8)
    heat_vis = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    pred_vis = cv2.cvtColor((pred_binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    labels = ["input", "ground truth", "pred probability", "pred binary"]
    panels = [img, gt_vis, heat_vis, pred_vis]

    labeled_panels = []
    for label, panel in zip(labels, panels):
        panel = panel.copy()
        cv2.putText(panel, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        labeled_panels.append(panel)

    combined = np.concatenate(labeled_panels, axis=1)
    cv2.putText(combined, f"dice={dice:.2f} iou={iou:.2f}",
                (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                cv2.LINE_AA)
    cv2.imwrite(out_path, combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Fallback: dataset dir to scan if no held-out "
                              "list is found/given. Prefer --val_list or the "
                              "auto-detected one from training for an honest "
                              "generalization check.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--val_list", type=str, default=None,
                         help="Explicit path to a held-out file list (as "
                              "saved by train.py alongside the model, or "
                              "your own list of genuinely unseen images).")
    parser.add_argument("--out_dir", type=str, default="eval_results")
    parser.add_argument("--num_visualize", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate(args.data_dir, args.model, args.out_dir,
              num_visualize=args.num_visualize, threshold=args.threshold,
              val_list_path=args.val_list)
