"""
Converts the DeepCrack dataset (Liu et al. 2019, Neurocomputing -- real,
pixel-level annotated concrete/pavement crack photos, 300 train + 237 test
images) into the same images/masks folder layout our synthetic-dataset
pipeline already uses, so it plugs directly into train.py / eval_visualize.py
/ compare_algorithms.py with zero other code changes.

Get the dataset first:
    git clone --depth 1 https://github.com/yhlleo/DeepCrack.git
    cd DeepCrack/dataset && unzip DeepCrack.zip -d extracted

Then run this script:
    python prepare_deepcrack.py --deepcrack_dir DeepCrack/dataset/extracted \
        --out_dir deepcrack_dataset

Verified against the real dataset: masks are clean binary (0/255), image
and mask filenames match by stem (e.g. train_img/11111.jpg <->
train_lab/11111.png) -- exactly the format our CrackDataset already expects.
"""

import os
import shutil
import argparse
import glob


def prepare(deepcrack_dir, out_dir, include_test=True):
    images_out = os.path.join(out_dir, "images")
    masks_out = os.path.join(out_dir, "masks")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    splits = [("train_img", "train_lab", "train")]
    if include_test:
        splits.append(("test_img", "test_lab", "test"))

    n_copied = 0
    for img_dir, lab_dir, prefix in splits:
        img_folder = os.path.join(deepcrack_dir, img_dir)
        lab_folder = os.path.join(deepcrack_dir, lab_dir)
        if not os.path.isdir(img_folder):
            print(f"[prepare_deepcrack] WARNING: {img_folder} not found, skipping")
            continue

        for img_path in sorted(glob.glob(os.path.join(img_folder, "*"))):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(lab_folder, stem + ".png")
            if not os.path.exists(mask_path):
                print(f"[prepare_deepcrack] WARNING: no mask for {img_path}, skipping")
                continue

            ext = os.path.splitext(img_path)[1]
            out_name = f"{prefix}_{stem}"
            shutil.copy(img_path, os.path.join(images_out, out_name + ext))
            shutil.copy(mask_path, os.path.join(masks_out, out_name + ".png"))
            n_copied += 1

    print(f"[prepare_deepcrack] {n_copied} image/mask pairs -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepcrack_dir", type=str, required=True,
                         help="Path to the extracted DeepCrack folder "
                              "(containing train_img/train_lab/test_img/test_lab)")
    parser.add_argument("--out_dir", type=str, default="deepcrack_dataset")
    parser.add_argument("--train_only", action="store_true",
                         help="Only use the 300 training images, hold out "
                              "the 237 test images entirely (e.g. to use "
                              "DeepCrack's own test split as a genuinely "
                              "independent real-world sanity check later).")
    args = parser.parse_args()

    prepare(args.deepcrack_dir, args.out_dir, include_test=not args.train_only)
