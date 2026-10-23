"""C6n on R9700: EXPERIMENT 4 -- frozen BatchNorm statistics.

Single variable vs C6r (the control, LOG #401-#404): BatchNorm running statistics are frozen
at the ImageNet-pretrained values for the whole run (every BN layer stays in eval mode;
affine weight/bias still train), and the top-k weight average skips BN re-estimation.
Each step otherwise estimates batch statistics from 24 correlated windows of two knees,
and CoAtNet-2 has 25 BatchNorm layers (LOG #376). Loss is unweighted, as in C6r.

Everything else is C6r value for value: 64-slice corpus, same soft labels and pos_weight,
CoAtNet-2 at 384, k=12/42, 16 epochs, batch 2, seed 42, LR 2e-5 / 1e-3, no LLRD,
REFERENCE windows, same code path (NativeDW, fused AdamW).

Pre-registered (LOG #407): LB > LB(C6r) + 0.004 = 0.937 counts as an improvement. The
rule-selected corpus0.pt is submitted; if it is a different weight type from C6r's top-3
average, the top-3 average is submitted as well and that pair is the one judged.
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
config.CORPUS_CONF_WEIGHTED_BCE = False
# the single variable
config.CORPUS_FREEZE_BN = True

from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS  # noqa: E402


def main():
    default_out = os.path.join(os.path.dirname(HERE), "experiments",
                               "local_c6n_64_k12_ep16_4llm_freezebn")
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C6n EXPERIMENT 4 on AMD R9700 (ROCm): frozen BatchNorm statistics", flush=True)
    print("vs C6r: BN running statistics frozen at pretrained values only", flush=True)
    print(f"Output Directory: {out}", flush=True)
    print(f"Backbone: {config.RAPTOR_ARCH} | res={config.RAPTOR_RES} | "
          f"LR backbone {config.RAPTOR_LR_BACKBONE} head {config.RAPTOR_LR_HEAD} | "
          f"LLRD {config.RAPTOR_LLRD} | legal windows {LEGAL_WINDOWS} | "
          f"conf-weighted BCE {config.CORPUS_CONF_WEIGHTED_BCE} | freeze BN {config.CORPUS_FREEZE_BN}", flush=True)
    print(f"Corpus Geometry: D={CORPUS_D} slices | slots={CORPUS_SLOTS}", flush=True)
    print(f"Hyperparameters: k_train={config.CORPUS_K_TRAIN} | k_eval={config.CORPUS_K_EVAL} | "
          f"bs={config.CORPUS_BATCH} | ep={config.CORPUS_EPOCHS} | budget={budget}h", flush=True)
    print(f"Soft Label Blend: regex + {config.SOFT_LABEL_PARQUETS}", flush=True)
    print("=" * 80, flush=True)

    assert config.RAPTOR_ARCH == "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
    assert config.RAPTOR_RES == 384
    assert (config.RAPTOR_LR_BACKBONE, config.RAPTOR_LR_HEAD, config.RAPTOR_LLRD) == (2e-5, 1e-3, None)
    assert LEGAL_WINDOWS is False
    assert config.CORPUS_CONF_WEIGHTED_BCE is False
    assert config.CORPUS_FREEZE_BN is True
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
    print("C6n Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
