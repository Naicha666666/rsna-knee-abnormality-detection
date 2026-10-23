"""C7 trainer on the R9700 ROCm server.

Single variable vs C6: k_train 12 -> 16 (MIL bag size). Everything else is
frozen to C6 so a move on the hidden test is attributable:

    64-slice corpus, regex+kimi+qwen+grok+gemma4, k_eval=42, batch=2,
    16 epochs, seed=42, bf16, same pos_weight / OneCycle / top-3 average.

C6 used 58% of 32 GB at k=12 (24 images/step). k=16 is 32 images/step, about
77% VRAM if linear -- under the card. Wall clock scales with k: C6 was 7.17 h
so this is ~9.5 h.

Pre-register: if LB stays at C6's 0.935, bag size is saturated at 12. Gold-58
spread is 0.018 so it cannot adjudicate this. No training on Kaggle.
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

config.SOFT_LABEL_PARQUETS = [
    "labels_kimi_soft.parquet",
    "labels_qwen_soft.parquet",
    "labels_grok_soft.parquet",
    "labels_gemma4_soft.parquet",
]
config.CORPUS_K_TRAIN = 16
config.CORPUS_K_EVAL = 42
config.CORPUS_EPOCHS = 16
config.CORPUS_BATCH = 2

from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS


def main():
    default_out = os.path.join(
        os.path.dirname(HERE), "experiments", "local_c7_64_k16_ep16_4llm"
    )
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C7 MIL-bag ablation on AMD Radeon AI PRO R9700 (ROCm)", flush=True)
    print("vs C6: k_train 12 -> 16; labels/epochs/seed/eval bag frozen", flush=True)
    print(f"Output Directory: {out}", flush=True)
    print(f"Corpus Geometry: D={CORPUS_D} slices | slots={CORPUS_SLOTS}", flush=True)
    print(f"Hyperparameters: k_train={config.CORPUS_K_TRAIN} | "
          f"k_eval={config.CORPUS_K_EVAL} | bs={config.CORPUS_BATCH} | "
          f"ep={config.CORPUS_EPOCHS} | budget={budget}h", flush=True)
    print(f"Soft Label Blend: regex + {config.SOFT_LABEL_PARQUETS}", flush=True)
    print("=" * 80, flush=True)

    vols, masks, ids = open_corpus()
    print(f"Corpus opened: shape={vols.shape}, n_studies={len(ids)}", flush=True)
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
    print("C7 Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
