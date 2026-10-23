"""KneeNet-2.5D Hierarchical Spatial Model Training on AMD R9700 (ROCm).

Features:
- Backbone: convnext_small.fb_in22k_ft_in1k_384 (Res=384, 49M params)
- 1D Intra-series Depthwise Residual Convolution along slice depth K
- Continuous Depth Positional MLP: [0, 1] -> f_dim
- 2-layer Global Study Transformer across slice tokens
- 12 Finding-Specific Attention Pooling Heads
- Supervision: 70% 7-Source LLM Soft Targets + 30% U2-Flash v6m (Calibrated Synovitis/Effusion)
- Strict Gold-58 zero-gradient validation every epoch.
- Outputs saved: corpus0_topk_avg.pt, corpus0_best_epoch.pt, corpus0.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")

import tv_stub
tv_stub.install()

import config
import raptor
from dataset import RSNADataset
from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS

class KneeNet25DClassifier(raptor.RaptorClassifier):
    """Next-Gen 2.5D Hierarchical Spatial Model with 1D Depth Context & Transformer."""
    def __init__(self, backbone: nn.Module, f_dim: int, n_out: int = 12, drop: float = 0.2, layers: int = 2):
        super().__init__(backbone, f_dim, n_out, drop)
        
        # 1D Residual Depth Convolution along slice depth K
        self.depth_conv = nn.Sequential(
            nn.Conv1d(f_dim, f_dim, kernel_size=3, padding=1, groups=f_dim),
            nn.GELU(),
            nn.Conv1d(f_dim, f_dim, kernel_size=1),
            nn.Dropout(0.1),
        )
        self.norm_depth = nn.LayerNorm(f_dim)
        
        # Continuous Depth Positional MLP: [0, 1] -> f_dim
        self.pos_emb = nn.Sequential(
            nn.Linear(1, f_dim),
            nn.GELU(),
            nn.Linear(f_dim, f_dim),
        )
        
        # Cross-slice Study-level Transformer Encoder
        el = nn.TransformerEncoderLayer(
            d_model=f_dim,
            nhead=8,
            dim_feedforward=2 * f_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(el, layers, enable_nested_tensor=False)

    def head(self, feats, kmask=None):
        # feats: [B, K, D]
        b, k, d = feats.shape
        
        # 1. Continuous depth position grid
        grid = torch.linspace(0.0, 1.0, k, device=feats.device, dtype=feats.dtype).view(1, k, 1)
        p = self.pos_emb(grid).expand(b, k, d)
        
        # 2. 1D Intra-series residual depth conv
        feats_t = feats.transpose(1, 2)  # [B, D, K]
        conv_out = self.depth_conv(feats_t).transpose(1, 2)  # [B, K, D]
        h = self.norm_depth(feats + conv_out) + p
        
        # 3. Global Study Transformer over all slice tokens
        pad = (kmask < 0.5) if kmask is not None else None
        h = self.tf(h, src_key_padding_mask=pad)
        
        # 4. Finding-specific attention pooling
        a = self.att(h)  # [B, K, 12]
        if kmask is not None:
            a = a.masked_fill(kmask.unsqueeze(-1) < 0.5, -1e4)
        a = torch.softmax(a, dim=1)
        
        # 5. Output Class Logits
        pooled = torch.einsum("bkn,bkd->bnd", a, h)  # [B, 12, D]
        return (pooled * self.cls_w).sum(-1) + self.cls_b

def main():
    parser = argparse.ArgumentParser(description="Train KneeNet-2.5D ConvNeXt-Small 384 with T0 Targets")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test on small subset")
    parser.add_argument("--budget", type=float, default=36.0, help="Time budget in hours")
    args = parser.parse_args()

    out_dir = "/home/eason/rsna-knee/experiments/local_kneenet_25d_384_t0_consensus"
    if args.smoke:
        out_dir += "_smoke"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 85, flush=True)
    print(f"LAUNCHING KneeNet-2.5D (ConvNeXt-S 384) TRAINING: {'SMOKE TEST' if args.smoke else 'FULL 16 EPOCHS'}", flush=True)
    print("=" * 85, flush=True)

    # 1. Target Configuration
    targets_npz = Path("/home/eason/rsna-knee/experiments/nightly_targets/targets_t0_consensus.npz")
    assert targets_npz.exists(), f"Target file not found: {targets_npz}"
    
    arm_data = np.load(targets_npz)
    t_matrix = arm_data["targets"]
    w_matrix = arm_data["weights"]
    npz_uids = list(arm_data["uids"])
    print(f"[2.5D] Loaded verified T0 targets: {t_matrix.shape}, weights: {w_matrix.shape}", flush=True)

    # 2. Hyperparameters
    SEED = 42
    ARCH = "convnext_small.fb_in22k_ft_in1k_384"
    config.RAPTOR_ARCH = ARCH
    raptor.RAPTOR_ARCH = ARCH
    config.RAPTOR_RES = 384
    config.CALIBRATE_SYNOVITIS = True
    config.CORPUS_K_TRAIN = 12
    config.CORPUS_K_EVAL = 42
    config.CORPUS_EPOCHS = 1 if args.smoke else 16
    config.CORPUS_BATCH = 2
    config.CORPUS_LEGAL_WINDOWS = False

    # 3. Inject Targets into Dataset
    orig_init = RSNADataset.__init__
    def custom_init(self, *iargs, **ikwargs):
        orig_init(self, *iargs, **ikwargs)
        assert len(self.uids) == len(npz_uids), f"UID length mismatch: {len(self.uids)} vs {len(npz_uids)}"
        self.targets = t_matrix.copy()
        self.weights = w_matrix.copy()
        print(f"[2.5D Inject] ds.targets and ds.weights successfully set to T0 consensus tensors!", flush=True)

    RSNADataset.__init__ = custom_init

    # 4. Inject 2.5D Architecture into Raptor
    orig_build_backbone = raptor.build_raptor_backbone
    def custom_build_backbone(arch=None):
        arch_to_use = ARCH
        print(f"[2.5D Inject] Building backbone: {arch_to_use}", flush=True)
        return orig_build_backbone(arch_to_use)

    def custom_build_head(bb, head=None):
        print(f"[2.5D Inject] Building KneeNet25DClassifier head with f_dim={bb.num_features}", flush=True)
        return KneeNet25DClassifier(bb, f_dim=bb.num_features, n_out=12)

    raptor.build_raptor_backbone = custom_build_backbone
    raptor.build_head = custom_build_head

    # 5. Sanity Check Corpus
    vols, masks, ids = open_corpus()
    assert vols.shape == (4407, 64, 336, 336), f"Unexpected corpus shape: {vols.shape}"
    del vols, masks, ids

    print(f"Configuration locked: Backbone={config.RAPTOR_ARCH}, Res={config.RAPTOR_RES}", flush=True)
    print(f"Epochs={config.CORPUS_EPOCHS}, Batch={config.CORPUS_BATCH}, Seed={SEED}", flush=True)
    print(f"Output Directory: {out_dir}", flush=True)
    print("=" * 85, flush=True)

    t0 = time.time()
    run_corpus_submit(
        epochs=config.CORPUS_EPOCHS,
        max_studies=40 if args.smoke else None,
        batch=config.CORPUS_BATCH,
        k_train=config.CORPUS_K_TRAIN,
        k_eval=12 if args.smoke else config.CORPUS_K_EVAL,
        out_dir=out_dir,
        seed=SEED,
        do_submit=False,
        time_budget_h=args.budget,
    )
    elapsed = (time.time() - t0) / 60.0
    print("=" * 85, flush=True)
    print(f"KneeNet-2.5D Training Finished in {elapsed:.1f} minutes!", flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    main()
