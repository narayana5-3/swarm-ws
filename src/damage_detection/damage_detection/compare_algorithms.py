"""
Statistical comparison of crack-detection configurations:
small_unet (baseline, from scratch) vs. transfer_unet (ImageNet-pretrained
ResNet34 encoder) vs. vit_segformer (pretrained SegFormer) vs., optionally,
transfer_unet_deepcrack (transfer_unet warm-started from a DeepCrack real-
photo pretraining checkpoint -- see prepare_deepcrack.py).

Runs N independent trials per configuration (different random seed each
time -> different train/val split + different augmentation stream), then
uses a Wilcoxon rank-sum test on the resulting Dice scores to check whether
differences are real, not noise -- matching the rigor standard used in the
swarm's own path-planning base paper (Luo et al. 2026), which runs 30-40
independent trials per configuration for exactly this reason.

Usage:
    python compare_algorithms.py --data_dir synthetic_dataset --n_trials 10 \
        --epochs 30 --out_dir comparison_results

    # include the DeepCrack-pretrained arm as a 4th configuration:
    python compare_algorithms.py --data_dir synthetic_dataset --n_trials 10 \
        --epochs 30 --deepcrack_checkpoint deepcrack_pretrained.pt \
        --out_dir comparison_results

Note: transfer_unet and vit_segformer need their pretrained weights
downloaded on first run (from PyPI-external hosts) -- make sure you have
internet access, or pass --no_pretrained to test the pipeline without them
(results won't be meaningful for the "does pretraining help" question in
that mode, only useful for verifying the harness works).

Note on the DeepCrack arm: only the FINE-TUNING stage is re-run per trial
(different seed each time) -- the DeepCrack pretraining checkpoint itself is
reused as-is across all trials, not re-trained per trial. This is
deliberate: re-running DeepCrack pretraining many times for statistical
significance isn't necessary (we're not testing whether pretraining is
stochastically consistent, that's well-established) -- what's actually
being tested is whether warm-starting fine-tuning from it consistently
beats fine-tuning from ImageNet-only weights, which the fine-tuning-stage
trials do test properly.
"""

import os
import json
import argparse
import numpy as np

from train import train
from eval_visualize import evaluate


def build_configs(deepcrack_checkpoint=None):
    """Each config: (name, model_type, init_from_checkpoint or None)."""
    configs = [
        ("small_unet", "small_unet", None),
        ("transfer_unet", "transfer_unet", None),
        ("vit_segformer", "vit_segformer", None),
    ]
    if deepcrack_checkpoint:
        configs.append(("transfer_unet_deepcrack", "transfer_unet", deepcrack_checkpoint))
    return configs


def run_comparison(data_dir, out_dir, n_trials=10, epochs=30, batch_size=8,
                    lr=None, pretrained=True, base_seed=0, deepcrack_checkpoint=None):
    os.makedirs(out_dir, exist_ok=True)
    configs = build_configs(deepcrack_checkpoint)
    config_names = [c[0] for c in configs]
    results = {name: {"dice": [], "iou": [], "precision": [], "recall": []}
               for name in config_names}

    for name, model_type, init_ckpt in configs:
        print(f"\n{'=' * 70}\n{name}: running {n_trials} independent trials\n{'=' * 70}")
        for trial in range(n_trials):
            seed = base_seed + trial
            print(f"\n--- {name} trial {trial + 1}/{n_trials} (seed={seed}) ---")

            trial_model_path = os.path.join(out_dir, f"{name}_trial{trial}.pt")
            train(data_dir, out_path=trial_model_path, epochs=epochs,
                  batch_size=batch_size, lr=lr, seed=seed,
                  model_type=model_type, pretrained=pretrained,
                  init_from_checkpoint=init_ckpt)

            trial_eval_dir = os.path.join(out_dir, f"{name}_trial{trial}_eval")
            metrics = evaluate(data_dir, trial_model_path, trial_eval_dir,
                                num_visualize=0)  # skip panel images, just want numbers

            for key in results[name]:
                results[name][key].append(metrics[key])

            # trial models can be large (esp. vit_segformer); remove after
            # extracting metrics to avoid filling disk across many trials
            if os.path.exists(trial_model_path):
                os.remove(trial_model_path)

    _summarize_and_test(results, config_names, out_dir, n_trials)
    return results


def _summarize_and_test(results, config_names, out_dir, n_trials):
    from scipy.stats import ranksums

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 70}\nSUMMARY ({n_trials} trials each)\n{'=' * 70}")
    summary = {}
    for name in config_names:
        dice_scores = results[name]["dice"]
        summary[name] = {
            "mean_dice": float(np.mean(dice_scores)),
            "std_dice": float(np.std(dice_scores)),
            "mean_iou": float(np.mean(results[name]["iou"])),
            "std_iou": float(np.std(results[name]["iou"])),
            "mean_precision": float(np.mean(results[name]["precision"])),
            "mean_recall": float(np.mean(results[name]["recall"])),
            "all_dice_scores": dice_scores,
        }
        print(f"{name:24s}  Dice = {summary[name]['mean_dice']:.4f} "
              f"+/- {summary[name]['std_dice']:.4f}   "
              f"IoU = {summary[name]['mean_iou']:.4f} "
              f"+/- {summary[name]['std_iou']:.4f}")

    print(f"\n{'=' * 70}\nPAIRWISE SIGNIFICANCE (Wilcoxon rank-sum on Dice scores)\n{'=' * 70}")
    pairwise = {}
    for i in range(len(config_names)):
        for j in range(i + 1, len(config_names)):
            a, b = config_names[i], config_names[j]
            stat, pvalue = ranksums(summary[a]["all_dice_scores"],
                                     summary[b]["all_dice_scores"])
            significant = pvalue < 0.05
            pairwise[f"{a}_vs_{b}"] = {"statistic": float(stat), "pvalue": float(pvalue),
                                        "significant_at_0.05": bool(significant)}
            marker = "*** SIGNIFICANT" if significant else "not significant"
            print(f"{a} vs {b}: p={pvalue:.4g}  [{marker}]")

    best_config = max(config_names, key=lambda m: summary[m]["mean_dice"])
    print(f"\nBest mean Dice: {best_config} "
          f"({summary[best_config]['mean_dice']:.4f})")
    print("NOTE: check the pairwise significance above before claiming this "
          "config is definitively 'best' -- a higher mean with a "
          "not-significant p-value against the runner-up means the "
          "difference could be noise, not a real effect.")

    out_path = os.path.join(out_dir, "comparison_summary.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "pairwise_significance": pairwise,
                    "n_trials": n_trials}, f, indent=2)
    print(f"\nFull results saved -> {out_path}")

    _plot_comparison(summary, config_names, out_dir)


def _plot_comparison(summary, config_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.5 * len(config_names) + 2, 5))
    data = [summary[name]["all_dice_scores"] for name in config_names]
    bp = ax.boxplot(data, tick_labels=config_names, patch_artist=True)
    palette = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
    ax.set_ylabel("Dice score (held-out)")
    ax.set_title(f"Crack detection configuration comparison "
                  f"({len(data[0])} trials each)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plot_path = os.path.join(out_dir, "comparison_boxplot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Boxplot saved -> {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="comparison_results")
    parser.add_argument("--n_trials", type=int, default=10,
                         help="Independent trials per configuration. Paper's own "
                              "convention uses 30-40; 10 is a faster starting "
                              "point given training cost, raise if time allows.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None,
                         help="Defaults per-config: 1e-3 normally, 1e-4 for "
                              "the DeepCrack-pretrained arm (fine-tuning an "
                              "already-converged checkpoint). Override to "
                              "force one value for every config.")
    parser.add_argument("--no_pretrained", action="store_true",
                         help="Test the harness without downloading pretrained "
                              "weights -- results won't be meaningful for the "
                              "real comparison, only useful for a dry run.")
    parser.add_argument("--deepcrack_checkpoint", type=str, default=None,
                         help="Path to a DeepCrack-pretrained transfer_unet "
                              "checkpoint (see prepare_deepcrack.py + "
                              "train.py --init_from_checkpoint). If given, "
                              "adds a 4th configuration: transfer_unet "
                              "fine-tuned from this checkpoint instead of "
                              "ImageNet-only weights.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_comparison(args.data_dir, args.out_dir, n_trials=args.n_trials,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    pretrained=not args.no_pretrained, base_seed=args.seed,
                    deepcrack_checkpoint=args.deepcrack_checkpoint)

