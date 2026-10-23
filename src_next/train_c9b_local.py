"""C9b on R9700: DINOv2 ViT-B/14 arm with a ViT-appropriate backbone LR.

C9 (same backbone, CoAtNet's LR) collapsed in epoch 2 -- gold 0.7243 -> 0.5779 --
exactly as OneCycle reached peak LR, and never recovered (LOG #394). C9b changes the
backbone LR to 5e-6 with layer-wise decay 0.75. That is a SECOND variable vs C6 and is
declared as such in the pre-registration (LOG #395); head LR stays 1e-3.

Original C9 notes follow.


Single variable vs C6: the image backbone. Corpus (64-slice), soft-label blend (the
same four LLMs), k=12/42, 16 epochs, batch 2, seed 42, resolution 384, loss, schedule
and epoch-selection rule are all unchanged.

Why DINOv2 ViT-B after C8: C8 (RadImageNet ResNet-50, gold 0.8390) decorrelated from
C6 (per-label Spearman median 0.797) but was 0.068 weaker, and fusing it LOST 0.005 on
the LB (LOG #392). That rejects a WEAK heterogeneous arm, not heterogeneity itself.
DINOv2 ViT-B/14 is self-supervised and a pure Transformer, so it differs from the
supervised conv-attention CoAtNet in both pretraining and architecture, and is strong
enough to be a fair test.

Pre-registered (LOG #393): best gold >= 0.88 to count as a strong arm; then an EQUAL
0.5/0.5 rank fusion with C6, no weight search; fusion LB > 0.939 supports the
hypothesis, <= 0.939 rejects it.

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
config.RAPTOR_ARCH = "vit_base_patch14_dinov2"   # weights: dinov2_vitb14.pth (ARCH_WEIGHTS)
# the second, declared variable (LOG #395): LR adapted to a ViT backbone
config.RAPTOR_LR_BACKBONE = 5e-6
config.RAPTOR_LLRD = 0.75
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
                               "local_c9b_64_k12_ep16_dinov2b_llrd")
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C9b heterogeneous arm on AMD R9700 (ROCm): DINOv2 ViT-B/14, ViT-adapted backbone LR", flush=True)
    print("vs C6: backbone + backbone LR (5e-6, LLRD 0.75); corpus/labels/k/epochs/batch/seed/res/head LR frozen", flush=True)
    print(f"Output Directory: {out}", flush=True)
    print(f"Backbone: {config.RAPTOR_ARCH} | res={config.RAPTOR_RES}", flush=True)
    print(f"LR: backbone {config.RAPTOR_LR_BACKBONE} with LLRD {config.RAPTOR_LLRD} | "
          f"head {config.RAPTOR_LR_HEAD}", flush=True)
    print(f"Corpus Geometry: D={CORPUS_D} slices | slots={CORPUS_SLOTS}", flush=True)
    print(f"Hyperparameters: k_train={config.CORPUS_K_TRAIN} | "
          f"k_eval={config.CORPUS_K_EVAL} | bs={config.CORPUS_BATCH} | "
          f"ep={config.CORPUS_EPOCHS} | budget={budget}h", flush=True)
    print(f"Soft Label Blend: regex + {config.SOFT_LABEL_PARQUETS}", flush=True)
    print("=" * 80, flush=True)

    # preflight assertions: a wrong one of these makes the run unattributable, and
    # finding out 25 minutes later costs a whole epoch (same reasoning as colab_c6)
    assert config.RAPTOR_ARCH == "vit_base_patch14_dinov2"
    assert (config.RAPTOR_LR_BACKBONE, config.RAPTOR_LLRD, config.RAPTOR_LR_HEAD) == (5e-6, 0.75, 1e-3)
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
    print("C9b Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
