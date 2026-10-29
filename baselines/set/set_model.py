"""
set_model.py — SET (CVPR 2025) on a YOLO11n backbone.

Original SET uses ResNet-50 + FPN + FCOS. This version keeps the two training-time
augmentations (HBS background smoothing + API adversarial perturbation) and plugs
them onto YOLO11n's P3/P4/P5 feature maps. Inference is identical to vanilla
YOLO11n (zero extra latency).
"""

import math
import torch
import torch.nn as nn

from yolocore.model import YOLO11n
from yolocore.loss_utils import swap_train_head

HEAD_CH = (64, 128, 256)
STRIDES = (8, 16, 32)

# SET original hyper-parameters (configs/aitod/fcos_r50_set.py)
REG_FACTOR_RANGE = [0, 0.01, 0.1, 0.5, 1, 2, 5]
REG_FACTORS = [4, 4, 4]      # index 4 -> reg_factor = 1.0
SET_SCALE = 1.0


class DNResBlock(nn.Module):
    """HBS: scale-adaptive background smoothing block."""

    def __init__(self, in_c=256, reduction=4, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, in_c // reduction, kernel_size, 1, padding, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_c // reduction, in_c, kernel_size, 1, padding, bias=True))

    def forward(self, x):
        return x + self.conv_block(x)


class SETModel(YOLO11n):
    """YOLO11n detector + SET training-time denoisers. At inference time the
    denoisers are not used; the model is a plain YOLO11n."""

    def __init__(self, nc=1):
        super().__init__(nc=nc)
        kernel_sizes = [int(math.log2(s)) // 2 * 2 + 1 for s in STRIDES]
        self.denoisers = nn.ModuleList(
            DNResBlock(in_c=c, reduction=4, kernel_size=k)
            for c, k in zip(HEAD_CH, kernel_sizes))
        self.reg_factor_range = list(REG_FACTOR_RANGE)
        self.reg_factors = list(REG_FACTORS)
        self.scale = SET_SCALE

    def forward_features(self, x):
        """backbone + neck, returns [P3, P4, P5] feature maps (grad-tracked)."""
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        x5 = self.b5(x4)
        x6 = self.b6(x5)
        x7 = self.b7(x6)
        x8 = self.b8(x7)
        x9 = self.b9(x8)
        x10 = self.b10(x9)

        x11 = self.h11(x10)
        x12 = self.h12([x11, x6])
        x13 = self.h13(x12)
        x14 = self.h14(x13)
        x15 = self.h15([x14, x4])
        x16 = self.h16(x15)          # P3 (stride 8)

        x17 = self.h17(x16)
        x18 = self.h18([x17, x13])
        x19 = self.h19(x18)          # P4 (stride 16)

        x20 = self.h20(x19)
        x21 = self.h21([x20, x10])
        x22 = self.h22(x21)          # P5 (stride 32)

        return [x16, x19, x22]

    def forward_head(self, feats):
        return self.h23(feats)


def build_set(nc=1, train=True):
    model = SETModel(nc=nc)
    if train:
        swap_train_head(model)
    return model
