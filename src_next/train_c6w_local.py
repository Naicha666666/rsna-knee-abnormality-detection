"""C6w on R9700: EXPERIMENT 2 -- legal MIL windows for training AND evaluation.

Single variable vs C6r (the control, LOG #401/#402): the window rule. Every centre used
anywhere -- the training bag, the per-epoch gold evaluation that selects the epoch, and
later the test-set inference -- must have rows c-1, c, c+1 all non-empty and in the same
slot. Everything else is C6r value for value: 64-slice corpus, regex + Kimi + Qwen + Grok
+ Gemma4 soft labels, CoAtNet-2 at 384, k=12/42, 16 epochs, batch 2, seed 42, LR 2e-5 /
1e-3, no layer-wise LR decay, same code path (NativeDW, fused AdamW).

Why train as well as infer: experiment 2a changed only the inference windows on weights
trained with the reference rule and landed in the no-effect band (LB 0.933 vs 0.935). One
untested explanation is a train/inference window mismatch; this run removes it.

Pre-registered (LOG #403): LB > LB(C6r) + 0.004 counts as an improvement; the submission
uses the rule-selected corpus0.pt, and both candidates are saved so a weight-type mismatch
with C6r (which adopted top-3 average) can be split with one extra inference submission.
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
config.CORPUS_K_TRAIN = 12
config.CORPUS_K_EVAL = 42
config.CORPUS_EPOCHS = 16
config.CORPUS_BATCH = 2
# the single variable (must be set before corpus is imported: it reads the switch once)
config.CORPUS_LEGAL_WINDOWS = True

from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS  # noqa: E402


def main():
    default_out = os.path.join(os.path.dirname(HERE), "experiments",
                               "local_c6w_64_k12_ep16_4llm_legalwin")
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C6w EXPERIMENT 2 on AMD R9700 (ROCm): legal MIL windows for train + eval", flush=True)
    print("vs C6r: window rule only (train bag, gold eval, later test inference)", flush=True)
    print(f"Output Directory: {out}", flush=True)
    print(f"Backbone: {config.RAPTOR_ARCH} | res={config.RAPTOR_RES} | "
          f"LR backbone {config.RAPTOR_LR_BACKBONE} head {config.RAPTOR_LR_HEAD} | "
          f"LLRD {config.RAPTOR_LLRD} | legal windows {LEGAL_WINDOWS}", flush=True)
    print(f"Corpus Geometry: D={CORPUS_D} slices | slots={CORPUS_SLOTS}", flush=True)
    print(f"Hyperparameters: k_train={config.CORPUS_K_TRAIN} | k_eval={config.CORPUS_K_EVAL} | "
          f"bs={config.CORPUS_BATCH} | ep={config.CORPUS_EPOCHS} | budget={budget}h", flush=True)
    print(f"Soft Label Blend: regex + {config.SOFT_LABEL_PARQUETS}", flush=True)
    print("=" * 80, flush=True)

    assert config.RAPTOR_ARCH == "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
    assert config.RAPTOR_RES == 384
    assert (config.RAPTOR_LR_BACKBONE, config.RAPTOR_LR_HEAD, config.RAPTOR_LLRD) == (2e-5, 1e-3, None)
    assert LEGAL_WINDOWS is True, "the switch did not reach corpus.py"
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
    print("C6w Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
