"""Local R9700 trainer: 64-slice corpus, regex+kimi+qwen, k=12, 16 epochs.

No hidden-test submit here -- checkpoint is copied to a Kaggle infer kernel later.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tv_stub
tv_stub.install()


def main():
    from config import CORPUS_EPOCHS, CORPUS_K_TRAIN, CORPUS_K_EVAL, CORPUS_BATCH
    from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS

    out = os.environ.get("RSNA_OUT_DIR", os.path.join(
        os.path.dirname(HERE), "experiments", "local_c5_64_k12_ep16"))
    os.makedirs(out, exist_ok=True)
    print(f"C5 local: D={CORPUS_D} slots={CORPUS_SLOTS} "
          f"k={CORPUS_K_TRAIN}/{CORPUS_K_EVAL} bs={CORPUS_BATCH} ep={CORPUS_EPOCHS}",
          flush=True)
    vols, masks, ids = open_corpus()
    print(f"opened corpus {vols.shape} n_ids={len(ids)}", flush=True)
    del vols, masks, ids

    run_corpus_submit(
        epochs=CORPUS_EPOCHS,
        batch=CORPUS_BATCH,
        k_train=CORPUS_K_TRAIN,
        k_eval=CORPUS_K_EVAL,
        out_dir=out,
        seed=42,
        do_submit=False,
        time_budget_h=36.0,
    )


if __name__ == "__main__":
    main()
