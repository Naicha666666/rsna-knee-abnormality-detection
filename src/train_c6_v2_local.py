"""C6-v2 Training on AMD R9700 (ROCm): CoAtNet-2 384 with Nightly T0 Consensus Targets.

Configuration:
- Backbone: coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k (Res=384)
- Corpus Geometry: 64-slice corpus (D=64, BS=2, k_train=12, k_eval=42, Epochs=16, Seed=42)
- Supervision: 70% 7-Source LLM Soft Targets + 30% U2-Flash v6m (Calibrated Synovitis/Effusion)
- Validation: 58 official radiologist Gold studies evaluated every epoch, zero training gradients.
- Outputs saved: corpus0_topk_avg.pt, corpus0_best_epoch.pt, corpus0.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")

import tv_stub
tv_stub.install()

import config
from dataset import RSNADataset
from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS

def main():
    parser = argparse.ArgumentParser(description="Train C6-v2 CoAtNet-2 384 with T0 Targets")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test on small subset")
    parser.add_argument("--budget", type=float, default=36.0, help="Time budget in hours")
    args = parser.parse_args()

    out_dir = "/home/eason/rsna-knee/experiments/local_c6_v2_384_t0_consensus"
    if args.smoke:
        out_dir += "_smoke"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 85, flush=True)
    print(f"LAUNCHING C6-v2 (CoAtNet-2 384) TRAINING: {'SMOKE TEST' if args.smoke else 'FULL 16 EPOCHS'}", flush=True)
    print("=" * 85, flush=True)

    # 1. Target Configuration
    targets_npz = Path("/home/eason/rsna-knee/experiments/nightly_targets/targets_t0_consensus.npz")
    assert targets_npz.exists(), f"Target file not found: {targets_npz}"
    
    arm_data = np.load(targets_npz)
    t_matrix = arm_data["targets"]
    w_matrix = arm_data["weights"]
    npz_uids = list(arm_data["uids"])
    print(f"[C6-v2] Loaded verified T0 targets: {t_matrix.shape}, weights: {w_matrix.shape}", flush=True)

    # 2. Hyperparameters
    SEED = 42
    config.RAPTOR_ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
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
        print(f"[C6-v2 Inject] ds.targets and ds.weights successfully set to T0 consensus tensors!", flush=True)

    RSNADataset.__init__ = custom_init

    # 4. Sanity Check Corpus
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
    print(f"C6-v2 Training Finished in {elapsed:.1f} minutes!", flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    main()
