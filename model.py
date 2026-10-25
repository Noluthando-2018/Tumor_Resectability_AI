"""
model.py

2.5D EfficientNet-B0 for binary tumor resectability classification.

Input  : [B, 60, 224, 224]
         4 modalities × 3 directions × 5 slices
Output : [B, 1]  raw logit
"""

import torch
import torch.nn as nn
from torchvision import models

IN_CHANNELS = 60


def build_model(pretrained=True, dropout_p=0.4):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model   = models.efficientnet_b0(weights=weights)

    # Adapt first conv: 3 → 60 channels
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels  = IN_CHANNELS,
        out_channels = old_conv.out_channels,
        kernel_size  = old_conv.kernel_size,
        stride       = old_conv.stride,
        padding      = old_conv.padding,
        bias         = False,
    )
    if pretrained:
        with torch.no_grad():
            w_mean = old_conv.weight.data.mean(dim=1, keepdim=True)
            new_conv.weight.data = w_mean.repeat(1, IN_CHANNELS, 1, 1)
    else:
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    model.features[0][0] = new_conv

    # Replace classifier head
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p / 2),
        nn.Linear(256, 1),
    )
    return model


if __name__ == "__main__":
    model = build_model(pretrained=False)
    dummy = torch.zeros(2, IN_CHANNELS, 224, 224)
    out   = model(dummy)
    print("Input  channels :", IN_CHANNELS)
    print("Output shape    :", out.shape)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters      : {total:,}")
    print("Model ready ✓")
