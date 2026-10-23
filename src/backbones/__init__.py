"""Image backbones for RSNA Knee (controlled variable of the comparison study).

Every backbone implements the SAME interface so the rest of the pipeline is
agnostic to which one is used:
    - `embed_dim`: int, the per-slice embedding width it returns.
    - `forward(x)`: x is [N, 1, H, W] or [N, 3, H, W] in [0,1];
      returns [N, embed_dim]. 1-channel is replicated; 3-channel is a 2.5D
      group of adjacent slices. DINOv2 also exposes `forward_tokens`.
    - `freeze(unfreeze_last_n)`: freeze all params, then unfreeze the last
      `unfreeze_last_n` transformer blocks + the final norm (fine-tuning recipe
      from the top public notebooks: encoder LR is ~100x below the head LR).

Backbones:
    - TinyCNNBackbone : from-scratch small CNN (baseline; no pretraining).
    - DINOv2SmallBackbone / DINOv2BaseBackbone : offline DINOv2 ViT-S/Base.
"""
from .dinov2 import (
    DINOv2ViT,
    DINOv2SmallBackbone,
    DINOv2BaseBackbone,
)
from .tiny_cnn import TinyCNNBackbone

__all__ = [
    "DINOv2ViT",
    "DINOv2SmallBackbone",
    "DINOv2BaseBackbone",
    "TinyCNNBackbone",
]
