"""From-scratch tiny CNN backbone (baseline, NO pretraining).

Kept as the control condition in the comparison study: it is trained entirely
from the (small) RSNA training set, so its score isolates the value of a
pretrained, radiology-adapted representation (DINOv2) in the same pipeline.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TinyCNNBackbone(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.embed_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):  # x: [N, 1|3, H, W] in [0,1]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.head(self.net(x).flatten(1))

    def freeze(self, unfreeze_last_n: int = 2):
        # Control condition: train the whole tiny CNN, nothing frozen.
        for p in self.parameters():
            p.requires_grad = True
