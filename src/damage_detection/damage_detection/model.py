"""
Lightweight U-Net for crack segmentation.

Deliberately small (few channels, 3 downsampling levels) since this needs
to train reasonably fast on a single-GPU / possibly VRAM-constrained
machine, and inputs are HoloOcean camera frames (moderate resolution, not
megapixel photography). Swap in a larger backbone (e.g. a pretrained
ResNet encoder) later once you have real headroom and want to fine-tune on
a public crack dataset for stronger generic features.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """
    Binary crack segmentation. Input: (B, 3, H, W) in [0,1].
    Output: (B, 1, H, W) logits (apply sigmoid for probability).
    H and W should be divisible by 8 (3 downsampling steps).
    """

    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        c = base_channels

        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(c * 4, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        self.out_conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


if __name__ == "__main__":
    # smoke test: forward pass shape check
    model = SmallUNet()
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print(f"input {dummy.shape} -> output {out.shape}")
    assert out.shape == (2, 1, 256, 256)
    print("OK")
