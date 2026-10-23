"""C11 on R9700: RQ2 control -- C10 with GLOBAL MEAN pooling instead of per-diagnosis attention (LOG #482).

Question (RQ2): does diagnosis-specific attention MIL beat global study pooling, and is the
gap larger on focal findings (ACL, MCL, menisci, Contusion, Fracture, Baker's) than on
diffuse ones (the three OA compartments, Effusion, Synovitis)?

Offline motivation: swapping C10's learned attention for a plain mean at INFERENCE costs
0.035 macro gold (C6r 0.047) -- but that model was trained with attention, so the swap is
off-regime. This run trains the mean-pooling model properly.

Single variable vs C10: config.RAPTOR_POOL = "mean" (RaptorClassifier.head averages the
bag; the attention layer is built but receives no gradient). Everything else is C10's:
ConvNeXt-Small at 384, 64-slice corpus, regex + Kimi + Qwen + Grok + Gemma4 soft labels,
k=12/42, 16 epochs, batch 2, seed 42, LR 2e-5/1e-3, top-3 selection.

Evaluation: 58 gold only, NO submission (research track). Pre-registered: C10 top-3 gold
macro 0.9023. C11 <= 0.8843 (C10 - 0.018 gold noise) -> attention MIL helps (RQ2 yes);
within +-0.018 -> no detectable effect; >= 0.9203 -> mean pooling better. Secondary,
descriptive only: focal-7 vs diffuse-5 drop.

The checkpoint does NOT record the pooling mode: any loader must set
config.RAPTOR_POOL = "mean" before building the model.
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

config.RAPTOR_ARCH = "convnext_small.fb_in22k_ft_in1k_384"   # = C10
# the single variable vs C10
config.RAPTOR_POOL = "mean"
# everything below is C10's configuration, unchanged
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
                               "local_c11_64_k12_ep16_convnext_meanpool")
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C11 on AMD R9700 (ROCm): ConvNeXt-Small, GLOBAL MEAN pooling (RQ2 control)", flush=True)
    print(f"vs C10: RAPTOR_POOL={config.RAPTOR_POOL}; everything else frozen", flush=True)
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
    assert config.RAPTOR_POOL == "mean"
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
    print("C11 Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
