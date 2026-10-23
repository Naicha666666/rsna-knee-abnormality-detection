"""Imaging-only model for RSNA Knee (slot attention over 6 MRI slots).

Architecture mirrors the highest-voted public notebooks (see LOG.md):
- A pluggable image BACKBONE encodes each MRI slice -> per-slice embedding.
  The backbone is the controlled variable of the comparison study:
    * TinyCNNBackbone        : from-scratch small CNN (control condition).
    * DINOv2SmallBackbone    : DINOv2 ViT-S/14 (offline weights).
    * DINOv2BaseBackbone     : DINOv2 ViT-B/14 (offline weights).
- Slot pooling -> 6 slot embeddings (one per plane/fluid/fat slot), gated by a
  presence mask so missing series are ignored.
- A PER-DIAGNOSIS attention head (one SlotHead per label) attends over the 6
  slots so, e.g., the ACL head focuses on Sagittal fluid-sensitive series while
  the OA head spreads attention across planes.
- The radiology report is NOT fed at inference (test.csv has no Report column).
  Report text is consumed offline into graded soft targets + weights by
  report_extractor.py, so this model is pure vision and submission-safe.

Training helpers (used by train.py / experiments.py):
- EMA shadow weights for inference.
- Encoder/head LR split (head LR; backbone LR scaled by `backbone.lr_scale`,
  ~0.01 for DINOv2, 1.0 for the tiny CNN). DINOv2 is frozen except its last
  ENCODER_UNFREEZE_LAST blocks (+ final norm), per community consensus.
- Pairwise ranking loss supplementing BCE.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (N_LABELS, IMG_SIZE, SLICES_PER_SLOT, N_SLOTS, SLOTS,
                    ENCODER_LR_FACTOR, RANK_LOSS_WEIGHT, SLOT_EMBED,
                    BACKBONE_EMBED, DINOV2_SMALL_PATH, DINOV2_BASE_PATH,
                    LABELS, SLOT_PRIOR_BIAS, EMA_DECAY, AUG_ROT_DEG,
                    AUG_SCALE, AUG_SHIFT, AUG_INTENSITY,
                    RANK_POS, RANK_NEG)
from backbones import (TinyCNNBackbone, DINOv2SmallBackbone, DINOv2BaseBackbone)


class SlotImageEncoder(nn.Module):
    """Encode one 2.5D group per slot -> [B, NS, feat], gated by presence mask.

    DINOv2 tokens are pooled three ways (CLS + patch mean + top-k focal) so
    diffuse disease and focal tears both survive the slot summary. TinyCNN
    has no token grid, so its embedding is used as-is.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.use_tokens = hasattr(backbone, "forward_tokens")
        self.feat_dim = backbone.embed_dim * (3 if self.use_tokens else 1)

    def forward(self, imgs, mask):
        # imgs: [B, NS, 3, H, W] (uint8 0-255 or float 0-1 / 0-255)
        B, NS, C, H, W = imgs.shape
        x = imgs.reshape(B * NS, C, H, W)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        elif x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        if self.use_tokens:
            tok = self.backbone.forward_tokens(x)      # [B*NS, 1+L, D]
            cls = tok[:, 0]
            patch = tok[:, 1:]
            k = max(1, patch.shape[1] // 8)
            focal = patch.topk(k, dim=1).values.mean(1)
            feat = torch.cat([cls, patch.mean(1), focal], dim=1)
        else:
            feat = self.backbone(x)
        feat = feat.view(B, NS, -1)
        return feat * mask.unsqueeze(-1)


class SlotHead(nn.Module):
    """Per-diagnosis attention over slot embeddings, with a weak anatomical prior.

    Each finding is read on particular sequences, so a shared mean would dilute
    the one slot that carries the evidence. The prior is only a 1.7x softmax
    tilt (0.55) and the learned query can override it.
    """
    def __init__(self, dim: int, n_slot: int = N_SLOTS, n_out: int = N_LABELS,
                 hidden: int = 256, p: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out_w = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.out_b = nn.Parameter(torch.zeros(n_out))
        self.hidden = hidden
        prior = torch.zeros(n_out, n_slot)
        # Slots: 0 SAG_FLUID_FS, 1 COR_FLUID_FS, 2 AX_FLUID_FS,
        #        3 AX_STRUCT, 4 COR_STRUCT, 5 SAG_STRUCT (the v13 list, which the
        # ablation left as the best-measured one).
        preferred = {
            "ACL": (0, 5), "MCL": (1, 4),
            "Medial Meniscus": (0, 1, 4),
            "Lateral Meniscus": (0, 1, 4),
            "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
            "PF OA": (0, 2, 3, 5), "Effusion": (0, 2, 3),
            "Synovitis": (0, 2, 3), "Baker's": (0,),
            "Contusion": (0, 1, 2, 3), "Fracture": (0, 1, 2, 3, 4, 5),
        }
        for target, slots in preferred.items():
            if target in LABELS:
                prior[LABELS.index(target), list(slots)] = SLOT_PRIOR_BIAS
        self.register_buffer("slot_prior", prior)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb                 # [B, S, H]
        att = (torch.einsum("bsh,oh->bos", h, self.query) / math.sqrt(self.hidden)
               + self.slot_prior.unsqueeze(0))
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return torch.einsum("boh,oh->bo", ctx, self.out_w) + self.out_b


class RSNAMultimodal(nn.Module):
    def __init__(self, backbone: nn.Module, slot_dim: int = SLOT_EMBED,
                 head_out: int = 64):
        super().__init__()
        self.img_enc = SlotImageEncoder(backbone)
        self.head = SlotHead(self.img_enc.feat_dim)

    def forward(self, imgs, mask):  # -> logits [B, N_LABELS]
        slots = self.img_enc(imgs, mask)
        return self.head(slots, mask)

    def predict(self, imgs, mask):
        return torch.sigmoid(self.forward(imgs, mask))


def augment(imgs):
    """Small affine + intensity jitter over a whole bag. No flips.

    Horizontal flip would undo laterality normalisation. Vertical flip would
    swap femur and tibia — the axis the compartmental OA targets are defined on.
    `imgs` is [B, S, C, H, W] in [0, 1] (float).
    """
    B, S, C, H, W = imgs.shape
    x = imgs.float()
    ang = math.radians((torch.rand(1).item() - 0.5) * 2 * AUG_ROT_DEG)
    sc = 1.0 + (torch.rand(1).item() - 0.5) * 2 * AUG_SCALE
    tx = (torch.rand(1).item() - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(1).item() - 0.5) * 2 * AUG_SHIFT
    cos, sin = math.cos(ang) / sc, math.sin(ang) / sc
    theta = x.new_tensor([[cos, -sin, tx], [sin, cos, ty]])
    theta = theta.unsqueeze(0).repeat(B * S, 1, 1)
    flat = x.reshape(B * S, C, H, W)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    x = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",
                      align_corners=False).reshape(B, S, C, H, W)
    scale = 1.0 + (torch.rand(1, device=x.device) - 0.5) * 2 * AUG_INTENSITY
    return (x * scale).clamp(0, 1)


def build_model(backbone_name: str = "tiny_cnn", unfreeze_last: int = 2,
                slot_dim: int = SLOT_EMBED) -> RSNAMultimodal:
    """Construct the model for a given backbone (the controlled variable)."""
    if backbone_name == "tiny_cnn":
        bb = TinyCNNBackbone(out_dim=BACKBONE_EMBED["tiny_cnn"])
        bb.lr_scale = 1.0                       # train the whole CNN at head LR
    elif backbone_name == "dinov2_small":
        bb = DINOv2SmallBackbone(weights_path=DINOV2_SMALL_PATH,
                                 unfreeze_last=unfreeze_last)
        bb.lr_scale = ENCODER_LR_FACTOR
    elif backbone_name == "dinov2_base":
        bb = DINOv2BaseBackbone(weights_path=DINOV2_BASE_PATH,
                                unfreeze_last=unfreeze_last)
        bb.lr_scale = ENCODER_LR_FACTOR
    else:
        raise ValueError(f"unknown backbone: {backbone_name}")
    return RSNAMultimodal(bb, slot_dim=slot_dim)


def get_param_groups(model: "RSNAMultimodal", head_lr: float):
    """Heads get `head_lr`; the backbone gets head_lr * backbone.lr_scale.
    Frozen params (requires_grad=False) are omitted so OneCycleLR step counts
    and AdamW state stay on the trainable set."""
    head_params, enc_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith("head.") or any(
            k in name for k in ("heads.", "classifier.", "slot_pool"))
        (head_params if is_head else enc_params).append(p)
    enc_lr = head_lr * getattr(model.img_enc.backbone, "lr_scale",
                               ENCODER_LR_FACTOR)
    groups = [{"params": head_params, "lr": head_lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": enc_lr})
    return groups


class EMA:
    """Exponential moving average of model weights for inference/submit.

    Decay is warmed up so the first few hundred steps are not anchored to the
    random head initialisation (most of a short fold).
    """
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.step = 0
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        if self.decay <= 0:
            return
        self.step += 1
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        src = model.state_dict()
        for k, v in self.state.items():
            s = src[k]
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(s.detach(), alpha=1.0 - decay)
            else:
                v.copy_(s)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.state)

    def state_dict(self):
        return self.state


class PairwiseRankingLoss(nn.Module):
    """Soft pairwise ranking on confidently graded pairs inside the batch.

    AUC only reads order, so this optimises the metric directly. A target with
    no usable pair (rare findings in a small batch) contributes nothing rather
    than a constant — which is why the weight stays small and this supplements
    BCE instead of replacing it.
    """
    def __init__(self, pos_cut: float = RANK_POS, neg_cut: float = RANK_NEG):
        super().__init__()
        self.pos_cut = pos_cut
        self.neg_cut = neg_cut

    def forward(self, logits, targets, weights=None):
        usable = (torch.ones_like(targets, dtype=torch.bool) if weights is None
                  else weights > 0)
        parts = []
        for j in range(logits.shape[1]):
            pos = logits[(targets[:, j] > self.pos_cut) & usable[:, j], j]
            neg = logits[(targets[:, j] < self.neg_cut) & usable[:, j], j]
            if pos.numel() and neg.numel():
                parts.append(F.softplus(-(pos[:, None] - neg[None, :])).mean())
        return torch.stack(parts).mean() if parts else logits.new_tensor(0.0)
