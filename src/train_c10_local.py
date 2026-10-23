"""C10 on R9700: ConvNeXt-Small arm -- C6r with the image backbone swapped (LOG #469).

Why: on the 58 gold the C-line CoAtNet arms correlate 0.90-0.93 (per-label Spearman
median) with the public CoAtNet family inside the 0.943 base -- as high as the public
readers correlate with each other -- so every fusion so far added a fifth copy of the
same reader (LOG #468). C8 (RadImageNet ResNet-50 on the SAME corpus) fell to 0.744,
so the backbone alone can decorrelate; C8 simply was too weak (gold 0.839). ConvNeXt
is the architecture the private legs of the 0.946 notebooks use (LOG #466).

Single variable vs C6r: RAPTOR_ARCH = convnext_small.fb_in22k_ft_in1k_384 (49.5M, native
384, ImageNet normalisation as the corpus already applies). Corpus, soft labels
(regex + Kimi + Qwen + Grok + Gemma4), k=12/42, 16 epochs, batch 2, seed 42, res 384,
backbone/head LR 2e-5/1e-3, no LLRD, loss, schedule and top-3 selection are unchanged.

Pre-registered offline gate (before any submission): top-3 gold macro-AUC >= 0.90 AND
per-label Spearman median vs the public CoAt family <= 0.85. Only if both pass, fuse into
the 0.943 base at a fixed W = 0.10 in percentile-rank space.

RAPTOR_ARCH must be set BEFORE raptor is first imported (build_raptor_backbone binds
its default arch at import time).
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

# B4 (LOG #533): RSNA_SEED reruns the identical configuration with another seed; default 42 = the
# original run (output directory unchanged), any other seed gets a _s<seed> suffix.
SEED = int(os.environ.get("RSNA_SEED", "42"))

# the single variable
config.RAPTOR_ARCH = "convnext_small.fb_in22k_ft_in1k_384"
# everything below is C6r's configuration, unchanged
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
                               "local_c10_64_k12_ep16_convnext_small" + ("" if SEED == 42 else f"_s{SEED}"))
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C10 on AMD R9700 (ROCm): ConvNeXt-Small arm", flush=True)
    print("vs C6r: backbone only; corpus/labels/k/epochs/batch/seed/res/LR frozen", flush=True)
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
        seed=SEED,
        do_submit=False,
        time_budget_h=budget,
    )
    print("=" * 80, flush=True)
    print("C10 Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
