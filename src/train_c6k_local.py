"""C6k on R9700: C6r with image-teacher distillation (LOG #457).

Single variable vs the C6r control (LB 0.933): the BCE target of every non-gold study
becomes 0.5 * (regex + Kimi + Qwen + Grok + Gemma4 mean) + 0.5 * image teacher, where the
teacher is the OUT-OF-FOLD mean of the public DINOv2 x20 OOF and RadImageNet e11 OOF
(data/labels/teacher_dino_rad_oof.parquet). lambda = 0.5 is fixed before training, not
tuned on the gold (0.3 / 0.5 / 0.6 all give 0.915-0.916 there). Gold studies stay
held out as validation exactly as in C6r.

Everything else is C6r's, value for value: 64-slice corpus, CoAtNet-2 at 384, k=12/42,
16 epochs, batch 2, seed 42, reference MIL windows, no LLRD, unweighted BCE, NativeDW +
fused AdamW.

Pre-registered criterion: LB >= 0.937 (C6r 0.933 + 0.004) with the top-3-average weights.
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
config.TEACHER_PARQUET = "teacher_dino_rad_oof.parquet"
config.TEACHER_LAMBDA = 0.5

from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS  # noqa: E402


def main():
    default_out = os.path.join(os.path.dirname(HERE), "experiments",
                               "local_c6k_64_k12_ep16_4llm_teacher05")
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("C6k on AMD R9700 (ROCm): C6r + image-teacher distillation", flush=True)
    print(f"vs C6r: teacher {config.TEACHER_PARQUET} lambda={config.TEACHER_LAMBDA}", flush=True)
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
    assert (config.TEACHER_PARQUET, config.TEACHER_LAMBDA) == ("teacher_dino_rad_oof.parquet", 0.5)

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
    print("C6k Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
