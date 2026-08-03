"""
Model factory for the three-way crack-detection comparison.

Three architectures behind a common interface (model.forward(x) -> logits
of shape (B, 1, H, W)), so train.py / eval_visualize.py / infer.py can swap
between them via a single --model_type flag without any other code changes.

1. small_unet    -- SmallUNet from model.py. Baseline, trained from scratch,
                     no pretrained weights, fastest, least data-hungry.
2. transfer_unet -- U-Net with a pretrained ResNet34 encoder (ImageNet
                     weights via segmentation_models_pytorch). Pretrained
                     low-level visual features (edges, textures) transfer
                     even though ImageNet has nothing underwater in it --
                     this is the standard "why not train from scratch"
                     answer, and should generalize better than the baseline
                     especially on a still-modest training set size.
3. vit_segformer -- SegFormer (Xie et al. 2021), a transformer-based
                     segmentation model, pretrained on ImageNet/ADE20k via
                     HuggingFace transformers. Matches the "CNNs and ViTs"
                     direction flagged in the project's own literature
                     review. Transformers' global attention can help with
                     thin, branching, discontinuous structures like cracks
                     where CNNs' local receptive fields sometimes miss the
                     larger pattern -- heavier to train, more data-hungry.

NOTE ON PRETRAINED WEIGHTS: downloading them requires internet access to
PyPI-external hosts (smp's own weight URLs, huggingface.co) that aren't
reachable from Claude's sandbox. This file was built and structurally
verified there using pretrained=False / randomly-initialized configs (shape
checks only). Run with real pretrained weights (the default) on your own
machine, where you have full internet access.
"""

import torch
import torch.nn as nn

from .model import SmallUNet


MODEL_IMAGE_SIZES = {
    "small_unet": 256,
    "transfer_unet": 256,
    "vit_segformer": 512,  # SegFormer's pretrained checkpoints expect this
}


def build_model(model_type, pretrained=True):
    """
    Returns (model, image_size). image_size is the input resolution this
    model expects -- use it when constructing the dataset/dataloader.
    """
    if model_type == "small_unet":
        return SmallUNet(), MODEL_IMAGE_SIZES[model_type]

    elif model_type == "transfer_unet":
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
        return model, MODEL_IMAGE_SIZES[model_type]

    elif model_type == "vit_segformer":
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        if pretrained:
            hf_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig(num_labels=1)
            hf_model = SegformerForSemanticSegmentation(config)
        return SegformerWrapper(hf_model), MODEL_IMAGE_SIZES[model_type]

    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                          f"Choose from {list(MODEL_IMAGE_SIZES.keys())}")


class SegformerWrapper(nn.Module):
    """
    Wraps HuggingFace's SegformerForSemanticSegmentation to match the same
    forward(x) -> logits interface as the other two models. SegFormer's raw
    output is at 1/4 resolution (a quirk of its architecture); this
    upsamples back to input resolution so loss functions and downstream
    code never need to know the difference.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        out = self.hf_model(pixel_values=x)
        logits = out.logits  # (B, num_labels, H/4, W/4)
        logits = nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


if __name__ == "__main__":
    # smoke test: verify all three build and produce correctly-shaped output,
    # using pretrained=False so this runs without needing external downloads
    for model_type in MODEL_IMAGE_SIZES:
        model, image_size = build_model(model_type, pretrained=False)
        dummy = torch.randn(2, 3, image_size, image_size)
        out = model(dummy)
        expected = (2, 1, image_size, image_size)
        status = "OK" if tuple(out.shape) == expected else "MISMATCH"
        print(f"{model_type}: input {tuple(dummy.shape)} -> output {tuple(out.shape)} "
              f"(expected {expected}) [{status}]")
