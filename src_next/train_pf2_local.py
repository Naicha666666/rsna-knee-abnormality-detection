"""PF-v2 on R9700: PF OA label-only specialist, ALL windows (LOG #498).

Splits the PF-v1 specialisation effect (LOG #497: PF-v1 0.884 vs C10 0.860 public PF OA, +0.024)
into its two bundled variables. PF-v2 = PF-v1 minus the window restriction, i.e. C10 plus ONE
variable (CORPUS_TRAIN_LABELS = ["PF OA"]); window policy, k_eval = 42, corpus, soft labels,
optimiser, epochs, seed all equal C10.

Pre-registered decision (LOG #498), PF public AUC X = 12*LB - 5.5 (+-0.006) from a PF-only probe:
  label effect   L = X - 0.860 (C10)      window effect   W = 0.884 (PF-v1) - X
  each +-0.012:  |.| >= 0.012 -> measurable, sign as read; |.| < 0.012 -> no measurable effect.
  Single seed per reader; retraining variance not counted.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")

import tv_stub
tv_stub.install()

import config

# the single variable
config.RAPTOR_ARCH = "convnext_small.fb_in22k_ft_in1k_384"   # = C10
# the specialist (vs C10)
config.CORPUS_TRAIN_LABELS = ["PF OA"]
config.CORPUS_CENTRE_ALLOW = None                            # = C10 (all windows)
config.SOFT_LABEL_PARQUETS = [
    "labels_kimi_soft.parquet",
    "labels_qwen_soft.parquet",
    "labels_grok_soft.parquet",
    "labels_gemma4_soft.parquet",
]
config.CORPUS_K_TRAIN = 12
config.CORPUS_K_EVAL = 42
config.CORPUS_EPOCHS = 16
config.CORPUS_BATCH = 2
config.CORPUS_LEGAL_WINDOWS = False

from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS  # noqa: E402


def main():
    default_out = os.path.join(os.path.dirname(HERE), "experiments",
                               "local_pf2_64_k12_ep16_convnext_pfoa_allwin")
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("PF-v2 on AMD R9700 (ROCm): PF OA label-only specialist, all windows", flush=True)
    print(f"labels {config.CORPUS_TRAIN_LABELS} | centres {config.CORPUS_CENTRE_ALLOW}", flush=True)
    print(f"Output Directory: {out}", flush=True)
    print(f"Backbone: {config.RAPTOR_ARCH} | res={config.RAPTOR_RES} | "
          f"LR backbone {config.RAPTOR_LR_BACKBONE} head {config.RAPTOR_LR_HEAD} | "
          f"LLRD {config.RAPTOR_LLRD} | legal windows {LEGAL_WINDOWS}", flush=True)
    print(f"Corpus Geometry: D={CORPUS_D} slices | slots={CORPUS_SLOTS}", flush=True)
    print(f"Hyperparameters: k_train={config.CORPUS_K_TRAIN} | k_eval={config.CORPUS_K_EVAL} | "
          f"bs={config.CORPUS_BATCH} | ep={config.CORPUS_EPOCHS} | budget={budget}h", flush=True)
    print(f"Soft Label Blend: regex + {config.SOFT_LABEL_PARQUETS}", flush=True)
    print("=" * 80, flush=True)

    assert config.RAPTOR_ARCH == "convnext_small.fb_in22k_ft_in1k_384"
    assert config.CORPUS_TRAIN_LABELS == ["PF OA"]
    assert config.CORPUS_CENTRE_ALLOW is None
    assert config.RAPTOR_RES == 384
    assert (config.RAPTOR_LR_BACKBONE, config.RAPTOR_LR_HEAD, config.RAPTOR_LLRD) == (2e-5, 1e-3, None)
    assert LEGAL_WINDOWS is False
    assert CORPUS_D == 64
    assert (config.CORPUS_K_TRAIN, config.CORPUS_K_EVAL,
            config.CORPUS_EPOCHS, config.CORPUS_BATCH) == (12, 42, 16, 2)
    assert len(config.SOFT_LABEL_PARQUETS) == 4

    vols, masks, ids = open_corpus()
    assert vols.shape == (4407, 64, 336, 336), vols.shape
    del vols, masks, ids

    run_corpus_submit(
        epochs=config.CORPUS_EPOCHS,
        batch=config.CORPUS_BATCH,
        k_train=config.CORPUS_K_TRAIN,
        k_eval=config.CORPUS_K_EVAL,
        out_dir=out,
        seed=42,
        do_submit=False,
        time_budget_h=budget,
    )
    print("=" * 80, flush=True)
    print("PF-v2 Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
