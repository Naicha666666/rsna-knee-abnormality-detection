"""C14 on AMD R9700 (ROCm): ConvNeXt V2-Base 384 with U2-FLASH Prompt v6 Soft Labels.

State-of-the-Art Architecture & Weak Supervision Intervention:
1. Backbone: convnextv2_base.fcmae_ft_in22k_in1k_384 (87.97M params, 1024-dim features).
   - Global Response Normalization (GRN) to prevent feature collapse on tiny abnormalities.
   - Fully Convolutional Masked AutoEncoder (FCMAE) pretraining for rich volumetric feature representations.
   - Native 384x384 high resolution matches physical MRI slice geometry.
2. Label Quality: U2-FLASH Prompt v6 Structured Extraction + Multi-LLM Ensemble + Clinical Ordinal Calibration.
3. R9700 32GB VRAM Acceleration: native BF16, k_train=12, k_eval=42, 16 Epochs.
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

SEED = int(os.environ.get("RSNA_SEED", "42"))
SMOKE = os.environ.get("RSNA_SMOKE") == "1"

# Backbone & High Resolution: ConvNeXt V2-Base 384
config.RAPTOR_ARCH = "convnextv2_base.fcmae_ft_in22k_in1k_384"
config.RAPTOR_RES = 384

# Ordinal Severity & Effusion->Synovitis Calibration
config.CALIBRATE_SYNOVITIS = True

# Soft Label Ensemble with Prompt v6
config.SOFT_LABEL_PARQUETS = [
    "labels_u2_flash_v6_soft.parquet",
    "labels_muse_glimmer_soft.parquet",
    "labels_kimi_soft.parquet",
    "labels_qwen_soft.parquet",
]
config.CORPUS_K_TRAIN = 12
config.CORPUS_K_EVAL = 42
config.CORPUS_EPOCHS = 16
config.CORPUS_BATCH = 2
config.CORPUS_LEGAL_WINDOWS = False

from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS  # noqa: E402


def main():
    default_out = os.path.join(os.path.dirname(HERE), "experiments",
                               "local_c14_64_k12_ep16_convnextv2_base_384_v6" + ("" if SEED == 42 else f"_s{SEED}"))
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C14 on AMD R9700 (ROCm): ConvNeXt V2-Base 384 with U2-FLASH v6 Soft Labels", flush=True)
    print(f"Output Directory: {out}", flush=True)
    print(f"Backbone: {config.RAPTOR_ARCH} | res={config.RAPTOR_RES} | "
          f"LR backbone {config.RAPTOR_LR_BACKBONE} head {config.RAPTOR_LR_HEAD} | "
          f"CALIBRATE_SYNOVITIS: {config.CALIBRATE_SYNOVITIS}", flush=True)
    print(f"Corpus Geometry: D={CORPUS_D} slices | slots={CORPUS_SLOTS}", flush=True)
    print(f"Hyperparameters: k_train={config.CORPUS_K_TRAIN} | k_eval={config.CORPUS_K_EVAL} | "
          f"bs={config.CORPUS_BATCH} | ep={config.CORPUS_EPOCHS} | budget={budget}h", flush=True)
    print(f"Soft Label Blend: regex + {config.SOFT_LABEL_PARQUETS}", flush=True)
    print("=" * 80, flush=True)

    assert config.RAPTOR_ARCH == "convnextv2_base.fcmae_ft_in22k_in1k_384"
    assert config.RAPTOR_RES == 384
    assert config.CALIBRATE_SYNOVITIS is True
    assert LEGAL_WINDOWS is False
    assert CORPUS_D == 64
    assert (config.CORPUS_K_TRAIN, config.CORPUS_K_EVAL,
            config.CORPUS_EPOCHS, config.CORPUS_BATCH) == (12, 42, 16, 2)
    assert len(config.SOFT_LABEL_PARQUETS) == 4

    vols, masks, ids = open_corpus()
    assert vols.shape == (4407, 64, 336, 336), vols.shape
    del vols, masks, ids

    run_corpus_submit(
        epochs=1 if SMOKE else config.CORPUS_EPOCHS,
        max_studies=8 if SMOKE else None,
        batch=config.CORPUS_BATCH,
        k_train=config.CORPUS_K_TRAIN,
        k_eval=config.CORPUS_K_EVAL,
        out_dir="/tmp/c14_smoke" if SMOKE else out,
        seed=SEED,
        do_submit=False,
        time_budget_h=budget,
    )
    print("=" * 80, flush=True)
    print("C14 Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
