"""C8 on R9700: heterogeneous SECOND ARM -- RadImageNet ResNet-50 instead of CoAtNet.

Single variable vs C6: the image backbone. Corpus (64-slice), soft-label blend (the
same four LLMs), k=12/42, 16 epochs, batch 2, seed 42, resolution 384, loss, schedule
and epoch-selection rule are all unchanged.

Why a different backbone rather than more CoAtNet tuning: the DINOv2 line saturated,
C7 showed the MIL bag is done at k=12, and every label-side avenue is closed (LOG
#355/#368). The public 4-arm ensemble's gain comes from architecture diversity, and
RadImageNet is pretrained on medical images rather than ImageNet, so its errors should
decorrelate from C6's. The arm is only worth keeping if the C6+C8 rank average beats
C6's 0.935 by more than the 0.004 noise band.

RAPTOR_ARCH must be set BEFORE raptor is first imported: build_raptor_backbone binds
its default arch at import time, so a later assignment would silently keep CoAtNet.
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
config.RAPTOR_ARCH = "resnet50"          # weights: resnet50_radimagenet.pt (ARCH_WEIGHTS)
# everything below is C6's configuration, unchanged
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

from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS  # noqa: E402


def main():
    default_out = os.path.join(os.path.dirname(HERE), "experiments",
                               "local_c8_64_k12_ep16_radimagenet")
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C8 heterogeneous arm on AMD R9700 (ROCm): RadImageNet ResNet-50", flush=True)
    print("vs C6: backbone only; corpus/labels/k/epochs/batch/seed/res frozen", flush=True)
    print(f"Output Directory: {out}", flush=True)
    print(f"Backbone: {config.RAPTOR_ARCH} | res={config.RAPTOR_RES}", flush=True)
    print(f"Corpus Geometry: D={CORPUS_D} slices | slots={CORPUS_SLOTS}", flush=True)
    print(f"Hyperparameters: k_train={config.CORPUS_K_TRAIN} | "
          f"k_eval={config.CORPUS_K_EVAL} | bs={config.CORPUS_BATCH} | "
          f"ep={config.CORPUS_EPOCHS} | budget={budget}h", flush=True)
    print(f"Soft Label Blend: regex + {config.SOFT_LABEL_PARQUETS}", flush=True)
    print("=" * 80, flush=True)

    # preflight assertions: a wrong one of these makes the run unattributable, and
    # finding out 25 minutes later costs a whole epoch (same reasoning as colab_c6)
    assert config.RAPTOR_ARCH == "resnet50"
    assert config.RAPTOR_RES == 384
    assert CORPUS_D == 64
    assert (config.CORPUS_K_TRAIN, config.CORPUS_K_EVAL,
            config.CORPUS_EPOCHS, config.CORPUS_BATCH) == (12, 42, 16, 2)
    assert len(config.SOFT_LABEL_PARQUETS) == 4

    vols, masks, ids = open_corpus()
    print(f"Corpus opened: shape={vols.shape}, n_studies={len(ids)}", flush=True)
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
    print("C8 Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
