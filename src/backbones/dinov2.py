"""Minimal, offline DINOv2 Vision Transformer (ViT-S / ViT-B).

Self-contained implementation that matches the official `facebookresearch/dinov2`
checkpoint keys (dinov2_vits14 / dinov2_vitb14, no register tokens) so we can
load weights that were saved locally (see src/download_dinov2.py) WITHOUT any
internet access at submit time -- a hard requirement of this Code Competition.

Only the [CLS] token embedding is used as the per-slice representation.

Input convention: forward() takes [N, 1, H, W] or [N, 3, H, W] in [0,1].
1-channel inputs are replicated to 3 (legacy grayscale path). 3-channel
inputs are treated as a 2.5D group of adjacent slices. ImageNet
normalization is applied internally.
"""
from __future__ import annotations
import math
import os
import glob

import torch
import torch.nn as nn


# ImageNet statistics used by the DINOv2 pretraining pipeline.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B, nh, N, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _LayerScale(nn.Module):
    """Per-channel learnable scale (DINOv2 uses this after attn/mlp)."""
    def __init__(self, dim: int, init: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init)

    def forward(self, x):
        return x * self.gamma


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, mlp_ratio)
        self.ls2 = _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2ViT(nn.Module):
    """DINOv2 ViT (patch14). `embed_dim`/`depth`/`num_heads` select S vs B.

    Architecture matches the official released `dinov2_vit{s,b}14_pretrain` weights
    (patch embedding is a Conv2d keyed `patch_embed.weight/bias`; each block has
    LayerScale `ls1/ls2`; positional embedding is interpolated from 518->224).
    """

    def __init__(self, img_size: int = 224, patch_size: int = 14,
                 embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze(self, unfreeze_last_n: int = 2):
        """Freeze everything, then unfreeze the last `unfreeze_last_n` blocks
        and the final norm. Early blocks stay frozen: there is not enough
        supervision here to improve generic edge filters and enough to damage them.
        """
        for p in self.parameters():
            p.requires_grad = False
        n = len(self.blocks)
        for i in range(max(0, n - unfreeze_last_n), n):
            for p in self.blocks[i].parameters():
                p.requires_grad = True
        for p in self.norm.parameters():
            p.requires_grad = True

    @torch.jit.unused
    def _interpolate_pos_embed(self, sd_pos: torch.Tensor) -> torch.Tensor:
        """Resize the checkpoint's pos_embed to our num_patches if they differ."""
        n = self.num_patches + 1
        if sd_pos.shape[1] == n:
            return sd_pos
        cls = sd_pos[:, :1]
        patch = sd_pos[:, 1:]
        old = int(math.sqrt(patch.shape[1]))
        patch = patch.reshape(1, old, old, -1).permute(0, 3, 1, 2)
        patch = torch.nn.functional.interpolate(
            patch, size=int(math.sqrt(self.num_patches)),
            mode="bicubic", align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, self.num_patches, -1)
        return torch.cat([cls, patch], dim=1)

    def load_pretrained(self, path: str):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k: v for k, v in sd.items()
              if not k.startswith("head.") and k != "mask_token"}
        # Remap facebookresearch/timm-style keys to our module naming.
        # Official `dinov2_vit{s,b}14_pretrain` weights expose the patch
        # embedding as `patch_embed.proj.{weight,bias}`; our model uses a plain
        # Conv2d registered as `patch_embed.{weight,bias}`.
        remap = {}
        for k, v in sd.items():
            if k == "patch_embed.proj.weight":
                remap["patch_embed.weight"] = v
            elif k == "patch_embed.proj.bias":
                remap["patch_embed.bias"] = v
            else:
                remap[k] = v
        sd = remap
        if "pos_embed" in sd and sd["pos_embed"].shape != self.pos_embed.shape:
            try:
                sd["pos_embed"] = self._interpolate_pos_embed(sd["pos_embed"])
            except Exception:
                sd.pop("pos_embed", None)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        return missing, unexpected

    def _prepare(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward_tokens(self, x):
        """Return all tokens [B, 1+L, D] after the final norm (CLS first)."""
        x = self._prepare(x)
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_features(self, x):
        return self.forward_tokens(x)[:, 0]

    def forward(self, x):
        return self.forward_features(x)


def _resolve_weights(path_env: str | None, variant: str) -> str | None:
    """Resolve a DINOv2 .pth path from an explicit env override, else glob a
    mounted Kaggle dataset directory for common weight filenames.

    Kaggle mounts dataset inputs under /kaggle/input/<slug>, but the exact
    layout varies (sometimes nested under /kaggle/input/datasets/<slug>), so we
    try explicit candidates AND fall back to a recursive search of /kaggle/input.
    """
    fname = f"{variant}.pth"            # e.g. dinov2_vits14.pth
    if path_env and os.path.exists(path_env):
        return path_env
    if path_env and os.path.isdir(path_env):
        cand = glob.glob(os.path.join(path_env, "*.pth"))
        cand += glob.glob(os.path.join(path_env, "*.safetensors"))
        if cand:
            return sorted(cand)[0]

    # Local repo copy (src/download_dinov2.py) plus every Kaggle mount we have seen.
    _here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(_here, "data", "dinov2_weights"),
        "/kaggle/input/rsna-dinov2-weights",
        "/kaggle/input/datasets/rsna-dinov2-weights",
        "/kaggle/input/dinov2",
        "/kaggle/input/dinov2/PyTorch/small",
        "/kaggle/input/dinov2/PyTorch/base",
        f"/kaggle/input/{variant}",
    ]
    for base in candidates:
        if os.path.isdir(base):
            for fn in (fname, f"dinov2_{variant}.pth"):
                p = os.path.join(base, fn)
                if os.path.exists(p):
                    return p
            cand = glob.glob(os.path.join(base, "*.pth"))
            if cand:
                return sorted(cand)[0]

    # Final fallback: recursive search so we never miss a non-standard mount.
    if os.path.isdir("/kaggle/input"):
        hits = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
        if hits:
            return sorted(hits)[0]
        hits = glob.glob("/kaggle/input/**/dinov2_vit*.pth", recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def _debug_input_tree(label: str) -> None:
    """Print what is actually mounted under /kaggle/input when weights are
    missing, so a non-attached dataset is immediately diagnosable."""
    print(f"[{label}] DEBUG /kaggle/input tree:")
    if not os.path.isdir("/kaggle/input"):
        print("  /kaggle/input does not exist")
        return
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - 2  # relative to /kaggle/input
        if depth > 3:
            continue
        indent = "  " * (depth + 1)
        print(f"{indent}{os.path.basename(root)}/")
        if depth >= 2:
            for f in files:
                print(f"{indent}  {f}")


# The ViT grid must follow the configured input size: pos_embed is added by
# slicing `pos_embed[:, :n_token]`, so a backbone built for 224 (256 patches) fed
# 168 px input (144 patches) would silently take the first 145 of a 257-row table
# in row-major order -- wrong positions, no error. Build it at IMG_SIZE instead.
try:
    from config import IMG_SIZE as _CFG_IMG
except Exception:  # pragma: no cover - standalone import in smoke tests
    _CFG_IMG = 224


class DINOv2SmallBackbone(nn.Module):
    """DINOv2 ViT-S/14 backbone wrapper (embed_dim=384)."""

    def __init__(self, weights_path: str | None = None, unfreeze_last: int = 2,
                 img_size: int = _CFG_IMG):
        super().__init__()
        self.vit = DINOv2ViT(img_size=img_size, embed_dim=384, depth=12,
                             num_heads=6)
        self.embed_dim = 384
        path = _resolve_weights(weights_path, "dinov2_vits14")
        if path:
            miss, unexp = self.vit.load_pretrained(path)
            print(f"[DINOv2-small] loaded {path} | missing={len(miss)} "
                  f"unexpected={len(unexp)}")
        else:
            print("[DINOv2-small] WARNING: no weights found -> random init")
            _debug_input_tree("DINOv2-small")
        self.vit.freeze(unfreeze_last)

    def forward(self, x):
        return self.vit(x)

    def forward_tokens(self, x):
        return self.vit.forward_tokens(x)

    def freeze(self, n: int = 2):
        self.vit.freeze(n)


class DINOv2BaseBackbone(nn.Module):
    """DINOv2 ViT-B/14 backbone wrapper (embed_dim=768)."""

    def __init__(self, weights_path: str | None = None, unfreeze_last: int = 2,
                 img_size: int = _CFG_IMG):
        super().__init__()
        self.vit = DINOv2ViT(img_size=img_size, embed_dim=768, depth=12,
                             num_heads=12)
        self.embed_dim = 768
        path = _resolve_weights(weights_path, "dinov2_vitb14")
        if path:
            miss, unexp = self.vit.load_pretrained(path)
            print(f"[DINOv2-base] loaded {path} | missing={len(miss)} "
                  f"unexpected={len(unexp)}")
        else:
            print("[DINOv2-base] WARNING: no weights found -> random init")
            _debug_input_tree("DINOv2-base")
        self.vit.freeze(unfreeze_last)

    def forward(self, x):
        return self.vit(x)

    def forward_tokens(self, x):
        return self.vit.forward_tokens(x)

    def freeze(self, n: int = 2):
        self.vit.freeze(n)
