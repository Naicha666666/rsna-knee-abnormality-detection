"""Off-Kaggle corpus trainer (R9700 box, Colab L4, anywhere with a GPU).

Everything that defines the experiment lives in config.py / corpus.py; this file
only supplies the paths and the wall-clock budget, both from the environment:

    RSNA_OUT_DIR         where corpus0.pt and the log go
    RSNA_CORPUS_DIR      directory holding the memmap
    RSNA_CORPUS_SLOTS    44 (published memmap, default) or 64 (our C4 shards)
    RSNA_TIME_BUDGET_H   stop after this many hours (Colab sessions are capped)

No hidden-test submit here -- the checkpoint is carried to a Kaggle infer kernel,
which costs minutes of GPU instead of the hours a training kernel costs.
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
        os.path.dirname(HERE), "experiments", "local_run"))
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36"))
    from config import SOFT_LABEL_PARQUETS
    print(f"local run: D={CORPUS_D} slots={CORPUS_SLOTS} "
          f"k={CORPUS_K_TRAIN}/{CORPUS_K_EVAL} bs={CORPUS_BATCH} ep={CORPUS_EPOCHS} "
          f"budget={budget}h", flush=True)
    print(f"label sources: regex + {SOFT_LABEL_PARQUETS}", flush=True)
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
        time_budget_h=budget,
    )


if __name__ == "__main__":
    main()
