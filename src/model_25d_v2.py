"""Next-Gen 2.5D Study-Level Knee Abnormality Detection Model (v2).

Key innovations over goodpjw2008 (LOG #556):
1. Pluggable Backbones: convnext_tiny (28M), convnext_small (49M), or coatnet_rmlp2_rw_384.
2. Hierarchical Spatial Modeling:
   - 2.5D 3-slice adjacent window (3-channel input)
   - 6 Anatomical Slots (3 planes x 2 fat-suppressed states)
   - Continuous Depth Positional MLP: Linear(1, dim) -> GELU -> Linear(dim, dim)
   - Intra-Series 1D Depth Context: Conv1D across slice depth within each series
   - Inter-Series Global Study Transformer: 2-layer Norm-First Transformer over study tokens
3. 12 Finding-Specific Attention Pooling Heads + per-finding linear classifier.
"""
from __future__ import annotations
import math
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = ["Sagittal", "Coronal", "Axial"]
N_SLOTS = 6  # 3 planes x (fat-sat, non-fat-sat)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def slot_of(plane: str, fat_sup: bool | int) -> int:
    return PLANES.index(plane) + 3 * (1 - int(fat_sup))


def make_views(x: torch.Tensor, res: int, train: bool, group: int, fov: float = 0.92) -> torch.Tensor:
    """x: uint8 [N, 3, H, W] with N = n_series * group.
    Applies shared series-level affine spatial jitter (rot, scale, shift) and photometric jitter (gamma, gain).
    """
    n = x.shape[0]
    ns = n // group
    dev = x.device
    x = x.float().div_(255.0)
    if train:
        ang = (torch.rand(ns, device=dev) - 0.5) * 2 * math.radians(12)
        sc = fov * (1 + (torch.rand(ns, device=dev) - 0.5) * 0.24)
        tx = (torch.rand(ns, device=dev) - 0.5) * 0.12
        ty = (torch.rand(ns, device=dev) - 0.5) * 0.12
        gamma = torch.exp((torch.rand(ns, device=dev) - 0.5) * 0.5)
        gain = 1 + (torch.rand(ns, device=dev) - 0.5) * 0.2
    else:
        ang = torch.zeros(ns, device=dev)
        sc = torch.full((ns,), fov, device=dev)
        tx = ty = torch.zeros(ns, device=dev)
        gamma = gain = None

    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    theta = theta.repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)

    if x.shape[-1] > 1.5 * res:
        x = F.avg_pool2d(x, 2) if x.shape[-1] >= 2 * res else F.interpolate(x, scale_factor=0.75, mode="bilinear", antialias=True)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    if train and gamma is not None and gain is not None:
        g = gamma.repeat_interleave(group).view(n, 1, 1, 1)
        x = x.clamp_min(1e-4).pow(g) * gain.repeat_interleave(group).view(n, 1, 1, 1)

    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


class IntraSeriesDepthConv(nn.Module):
    """1D residual depth context along slice depth k within each series."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),  # depthwise
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),                         # pointwise
            nn.Dropout(0.1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B * S, K, D]
        res = x
        x_t = x.transpose(1, 2)  # [B*S, D, K]
        out = self.conv(x_t).transpose(1, 2)  # [B*S, K, D]
        return self.norm(res + out)


class KneeNetV2(nn.Module):
    """Next-Gen 2.5D Multi-Slice Knee Model with Hierarchical Spatial Aggregation."""
    def __init__(
        self,
        backbone: str = "convnext_small",
        pretrained: bool = True,
        dim: int = 384,
        depth: int = 2,
        drop: float = 0.2,
        drop_path: float = 0.1,
        use_depth_conv: bool = True,
    ):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        num_feat = self.enc.num_features

        self.proj = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(num_feat, dim),
            nn.LayerNorm(dim),
        )
        self.slot_emb = nn.Embedding(N_SLOTS, dim)
        self.pos_emb = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        
        self.depth_conv = IntraSeriesDepthConv(dim) if use_depth_conv else nn.Identity()

        # Global cross-slot Study-level Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None

        # 12 finding-specific attention pooling heads
        self.att = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, len(LABELS)),
        )
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))

    def head(self, feat: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """feat [B, S, K, D], mask [B, S] bool, pos [B, S, K] in 0..1 -> logits [B, 12]."""
        b, s, k, d = feat.shape
        slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)

        # 1. Base tokens with slot identity and slice-depth continuous position
        h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))

        # 2. Intra-series depth smoothing
        h = self.depth_conv(h.view(b * s, k, -1)).view(b, s, k, -1)

        # 3. Global Study Transformer over all tokens
        h = h.view(b, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)  # safeguard against empty study

        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=pad)

        # 4. Finding-specific attention pooling
        a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(dim=1)  # [B, T, 12]
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)  # [B, 12, D]

        # 5. Class logit computation
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor, res: int = 256, train: bool = False) -> torch.Tensor:
        """x uint8 [B, S, K, 3, H, W]."""
        b, s, k = x.shape[:3]
        keep = mask.view(-1)  # only run 2D backbone on series that actually exist
        xv = x.view(b * s, k, *x.shape[3:])[keep].flatten(0, 1)

        # Run 2D encoder with memory_format=channels_last
        f = self.enc(make_views(xv, res, train, k).contiguous(memory_format=torch.channels_last))
        feat = f.new_zeros(b * s, k, f.shape[-1])
        feat[keep] = f.view(-1, k, f.shape[-1])

        return self.head(feat.view(b, s, k, -1), mask, pos)
