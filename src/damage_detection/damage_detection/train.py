"""
Train SmallUNet on a synthetic-injected crack dataset (from crack_injection.py).

Usage:
    python train.py --data_dir synthetic_dataset --epochs 30 --out model.pt

Loss: combined BCE + Dice, standard for segmentation with heavy class
imbalance (crack pixels are a small fraction of the image, plain BCE alone
tends to collapse to predicting "no crack" everywhere).
"""

import os
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_factory import build_model


def augment_pair(img, mask, rng):
    """
    img: HxWx3 uint8. mask: HxW uint8 (0/255). Applied together so the crack
    location stays consistent between image and mask.
    """
    h, w = img.shape[:2]

    # random horizontal / vertical flip
    if rng.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if rng.random() < 0.5:
        img = np.ascontiguousarray(img[::-1, :])
        mask = np.ascontiguousarray(mask[::-1, :])

    # random small rotation
    angle = rng.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # random crop (80-100% of area) then resize back -- adds scale/position diversity
    scale = rng.uniform(0.8, 1.0)
    ch, cw = int(h * scale), int(w * scale)
    top = rng.integers(0, h - ch + 1)
    left = rng.integers(0, w - cw + 1)
    img = img[top:top + ch, left:left + cw]
    mask = mask[top:top + ch, left:left + cw]
    img = cv2.resize(img, (w, h))
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # brightness/contrast jitter -- image only, mask unaffected
    alpha = rng.uniform(0.85, 1.15)  # contrast
    beta = rng.uniform(-20, 20)      # brightness
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return img, mask


class CrackDataset(Dataset):
    def __init__(self, image_paths, mask_dir, image_size=256, augment=False, seed=0):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        if not self.image_paths:
            raise FileNotFoundError("Empty image path list passed to CrackDataset.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, mask_stem + ".png")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = augment_pair(img, mask, self.rng)

        img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)

        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return img_t, mask_t


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets, bce_fn):
    return bce_fn(logits, targets) + dice_loss(logits, targets)


def train(data_dir, out_path="model.pt", epochs=30, batch_size=8, lr=None,
          image_size=None, val_fraction=0.15, device=None, augment=True, seed=0,
          model_type="small_unet", pretrained=True, init_from_checkpoint=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}, augmentation: {augment}, model_type: {model_type}")

    model, default_image_size = build_model(model_type, pretrained=pretrained)
    image_size = image_size or default_image_size

    if init_from_checkpoint:
        # warm-start from a previous checkpoint (e.g. a DeepCrack-pretrained
        # model) instead of ImageNet/random init -- this is the actual
        # mechanism for the "pretrain on real crack photos, fine-tune on
        # synthetic sim-domain data" two-stage workflow.
        ckpt = torch.load(init_from_checkpoint, map_location=device)
        ckpt_model_type = ckpt.get("model_type", "unknown")
        if ckpt_model_type != model_type:
            print(f"WARNING: init_from_checkpoint was trained as "
                  f"'{ckpt_model_type}' but you requested '{model_type}' -- "
                  f"loading anyway, but this will likely fail or silently "
                  f"not transfer most weights if the architectures differ.")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Initialized from checkpoint: {init_from_checkpoint} "
              f"(model_type={ckpt_model_type})")

    model = model.to(device)

    all_paths = sorted(sum([glob.glob(os.path.join(data_dir, "images", ext))
                             for ext in ("*.png", "*.jpg", "*.jpeg")], []))
    if not all_paths:
        raise FileNotFoundError(
            f"No images found in {data_dir}/images -- run crack_injection.py first.")
    mask_dir = os.path.join(data_dir, "masks")

    rng = np.random.default_rng(seed)
    shuffled = all_paths.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]

    train_set = CrackDataset(train_paths, mask_dir, image_size=image_size,
                              augment=augment, seed=seed)
    val_set = CrackDataset(val_paths, mask_dir, image_size=image_size,
                            augment=False, seed=seed)  # val stays clean, unaugmented

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # save which files were held out, so eval_visualize.py can check against
    # ONLY genuinely unseen data instead of accidentally re-scoring training images
    val_list_path = os.path.splitext(out_path)[0] + "_val_files.txt"
    with open(val_list_path, "w") as f:
        f.write("\n".join(val_paths))
    print(f"Held-out validation file list saved -> {val_list_path} "
          f"({len(val_paths)} images, never seen during training)")

    if lr is None:
        lr = 1e-4 if init_from_checkpoint else 1e-3
        print(f"lr not specified, using default {lr} "
              f"({'fine-tuning from checkpoint' if init_from_checkpoint else 'training from scratch/pretrained-backbone'})")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = combined_loss(logits, masks, bce_fn)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = combined_loss(logits, masks, bce_fn)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_set)

        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "image_size": image_size,
                        "model_type": model_type}, out_path)
            print(f"  -> saved best model to {out_path}")

    print(f"Training complete. Best val_loss={best_val_loss:.4f}, model at {out_path}")
    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Output dir from crack_injection.py "
                              "(contains images/ and masks/ subfolders).")
    parser.add_argument("--out", type=str, default="model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None,
                         help="Defaults to 1e-3 for training from scratch/"
                              "pretrained-backbone, or 1e-4 when using "
                              "--init_from_checkpoint -- fine-tuning an "
                              "already-converged checkpoint at full learning "
                              "rate tends to destabilize it rather than adapt "
                              "it (this was observed directly: a DeepCrack-"
                              "pretrained model fine-tuned on synthetic data "
                              "at lr=1e-3 got stuck at a much worse loss than "
                              "training from ImageNet-only weights did). "
                              "Override explicitly if you want a different value.")
    parser.add_argument("--image_size", type=int, default=None,
                         help="Defaults to the chosen model's expected input size.")
    parser.add_argument("--model_type", type=str, default="small_unet",
                         choices=["small_unet", "transfer_unet", "vit_segformer"],
                         help="Which architecture to train: small_unet (baseline, "
                              "from scratch), transfer_unet (pretrained ResNet34 "
                              "encoder), vit_segformer (pretrained SegFormer).")
    parser.add_argument("--no_pretrained", action="store_true",
                         help="Train transfer_unet/vit_segformer from random init "
                              "instead of pretrained weights (mainly for testing "
                              "without internet access to weight hosts).")
    parser.add_argument("--init_from_checkpoint", type=str, default=None,
                         help="Warm-start from a previous checkpoint (e.g. a "
                              "DeepCrack-pretrained model) instead of ImageNet/"
                              "random init. Use this for the pretrain-on-real, "
                              "fine-tune-on-synthetic two-stage workflow: train "
                              "once on DeepCrack, then pass that checkpoint here "
                              "when training on synthetic_dataset.")
    parser.add_argument("--no_augment", action="store_true",
                         help="Disable training-time augmentation (flip/rotate/"
                              "crop/brightness jitter) -- on by default.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(args.data_dir, out_path=args.out, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, image_size=args.image_size,
          augment=not args.no_augment, seed=args.seed,
          model_type=args.model_type, pretrained=not args.no_pretrained,
          init_from_checkpoint=args.init_from_checkpoint)
