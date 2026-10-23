"""C6b on R9700 (B5, LOG #533): C6r with BINARIZED training targets -- the single variable.

Question: S6 (LOG #521) observed that the labels whose reports disagree with the expert threshold (Effusion:
65% of gold negatives are reported positive; Contusion: 38%) still get strong image models, and read it as the
soft, severity-graded target absorbing the threshold mismatch. Binarizing the target at 0.5 removes that
graded information; if the reading is right, Effusion + Contusion should drop while extractable labels do not.

Everything else = C6r value for value (64-slice corpus, regex + Kimi + Qwen + Grok + Gemma4 soft labels,
CoAtNet-2 384, k 12/42, 16 epochs, bs 2, seed 42, reference windows, no LLRD); gold rows keep the gold override.
Control = the existing C6r weights (LB 0.933), not retrained.

Pre-registered reading (two pooled probes per model, other columns 0.5; k-column mean = (12*LB - (12-k)*0.5)/k):
  probe A = Effusion + Contusion (k = 2, one unit 0.006); probe B = ACL, MCL, Medial Meniscus, Baker's (k = 4,
  one unit 0.003). Mechanism supported iff  A(C6b) - A(C6r) <= -0.012  AND  |B(C6b) - B(C6r)| <= 0.003.
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
config.CORPUS_LEGAL_WINDOWS = False
config.CORPUS_BINARIZE_TARGETS = True          # B5: the single variable vs C6r

from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS  # noqa: E402


def main():
    default_out = os.path.join(os.path.dirname(HERE), "experiments",
                               "local_c6b_64_k12_ep16_4llm_binarized")
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C6b on AMD R9700 (ROCm): C6r + binarized targets (B5)", flush=True)
    print("vs C6r: CORPUS_BINARIZE_TARGETS = True only (hard targets at 0.5; gold override unchanged)", flush=True)
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
    assert LEGAL_WINDOWS is False
    assert CORPUS_D == 64
    assert (config.CORPUS_K_TRAIN, config.CORPUS_K_EVAL,
            config.CORPUS_EPOCHS, config.CORPUS_BATCH) == (12, 42, 16, 2)
    assert len(config.SOFT_LABEL_PARQUETS) == 4
    assert config.CORPUS_BINARIZE_TARGETS is True

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
    print("C6r Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
