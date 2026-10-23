"""C6s on R9700: experiment 5 -- C6r with a different soft-label source set (LOG #424).

Single variable vs the C6r control: SOFT_LABEL_PARQUETS
    C6r: regex + Kimi + Qwen + Grok + Gemma4
    C6s: regex + Luna + Nemotron + Muse Glimmer + Qwen   ("NEW4", LOG #422-#423)
Blending is unchanged (equal-weight probability mean with the regex extractor).
Everything else is C6r's, value for value: 64-slice corpus, CoAtNet-2 at 384,
k=12/42, 16 epochs, batch 2, seed 42, reference MIL windows, no LLRD, unweighted BCE,
BN not frozen, NativeDW + fused AdamW.

Pre-registered criterion: LB > 0.937 (C6r 0.933 + 0.004) with the top-3-average weights.
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
    "labels_luna_soft.parquet",
    "labels_nemotron_soft.parquet",
    "labels_muse_glimmer_soft.parquet",
    "labels_qwen_soft.parquet",
]
config.CORPUS_K_TRAIN = 12
config.CORPUS_K_EVAL = 42
config.CORPUS_EPOCHS = 16
config.CORPUS_BATCH = 2
config.CORPUS_LEGAL_WINDOWS = False

from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS  # noqa: E402


def main():
    default_out = os.path.join(os.path.dirname(HERE), "experiments",
                               "local_c6s_64_k12_ep16_new4labels")
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C6s (experiment 5) on AMD R9700 (ROCm): C6r with the NEW4 soft-label set", flush=True)
    print("vs C6r: SOFT_LABEL_PARQUETS only (kimi/qwen/grok/gemma4 -> luna/nemotron/muse_glimmer/qwen)", flush=True)
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
    assert config.SOFT_LABEL_PARQUETS == ["labels_luna_soft.parquet", "labels_nemotron_soft.parquet",
                                          "labels_muse_glimmer_soft.parquet", "labels_qwen_soft.parquet"]
    assert not getattr(config, "CORPUS_CONF_WEIGHTED_BCE", False)
    assert not getattr(config, "CORPUS_FREEZE_BN", False)

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
    print("C6s Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
