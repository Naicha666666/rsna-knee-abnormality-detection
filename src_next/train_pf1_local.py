"""PF-v1 on R9700: PF OA diagnosis specialist (LOG #490).

Strategy: RSNA_Knee_Bottleneck_Audit_Conversation_2026-10-01.md -- freeze the 0.943 public
stack and attack only the weakest public column, PF OA (public AUC ~0.896, LOG #477), whose
report target (0.912) beats every image model -> a representation / spatial problem.

Step-1 diagnostic (scratchpad/pfoa_window_diag.py): PF OA evidence sits in the central rows of
the sagittal fluid slot (single window at row 9: AUC 0.87 for both generalists, above their
whole-bag 0.831 / 0.847); axial-only is weak. The patellofemoral joint is a midline structure.

Specialist vs C10 (ConvNeXt-Small 384, same corpus, same soft labels, same optimiser):
  * CORPUS_TRAIN_LABELS = ["PF OA"]: BCE and epoch / top-k selection on PF OA only
  * CORPUS_CENTRE_ALLOW = central sagittal: fluid-slot centres 4-13 + structural-slot
    centres 20-29 (20 centres, none straddles a slot); the policy is stored in the checkpoint
    and restored at inference
  * k_eval = 20 (every allowed centre once, in gold evaluation and in test inference)

Pre-registered decision (LOG #490):
  offline sanity  top-3 gold PF OA >= 0.80 and not constant (21 positives: noise ~+-0.1,
                  so gold only rules out a broken model; descriptive refs C10 0.851, CoAt 0.879)
  decisive probe  submission = specialist PF OA column, the other 11 labels constant 0.5;
                  PF public AUC = 12*LB - 5.5 (+-0.006) vs the public stack's 0.896:
                  >= 0.902 -> replace the PF OA column of 56558421 in one clean full submission
                  0.890-0.901 -> no gain, stop PF-v1 (no hyper-parameter fishing)
                  < 0.890 -> worse; revisit the specialisation hypothesis itself
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

# B4 (LOG #533): RSNA_SEED reruns the identical configuration with another seed; default 42 = the
# original run (output directory unchanged), any other seed gets a _s<seed> suffix.
SEED = int(os.environ.get("RSNA_SEED", "42"))

# the single variable
config.RAPTOR_ARCH = "convnext_small.fb_in22k_ft_in1k_384"   # = C10
# the specialist (vs C10)
config.CORPUS_TRAIN_LABELS = ["PF OA"]
config.CORPUS_CENTRE_ALLOW = list(range(4, 14)) + list(range(20, 30))
config.SOFT_LABEL_PARQUETS = [
    "labels_kimi_soft.parquet",
    "labels_qwen_soft.parquet",
    "labels_grok_soft.parquet",
    "labels_gemma4_soft.parquet",
]
config.CORPUS_K_TRAIN = 12
config.CORPUS_K_EVAL = 20
config.CORPUS_EPOCHS = 16
config.CORPUS_BATCH = 2
config.CORPUS_LEGAL_WINDOWS = False

from corpus import run_corpus_submit, open_corpus, CORPUS_D, CORPUS_SLOTS, LEGAL_WINDOWS  # noqa: E402


def main():
    default_out = os.path.join(os.path.dirname(HERE), "experiments",
                               "local_pf1_64_k12_ep16_convnext_pfoa_censag" + ("" if SEED == 42 else f"_s{SEED}"))
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print("PF-v1 on AMD R9700 (ROCm): PF OA specialist, central-sagittal windows", flush=True)
    print(f"labels {config.CORPUS_TRAIN_LABELS} | centres {config.CORPUS_CENTRE_ALLOW}", flush=True)
    print(f"Output Directory: {out}", flush=True)
    print(f"Backbone: {config.RAPTOR_ARCH} | res={config.RAPTOR_RES} | "
          f"LR backbone {config.RAPTOR_LR_BACKBONE} head {config.RAPTOR_LR_HEAD} | "
          f"LLRD {config.RAPTOR_LLRD} | legal windows {LEGAL_WINDOWS}", flush=True)
    print(f"Corpus Geometry: D={CORPUS_D} slices | slots={CORPUS_SLOTS}", flush=True)
    print(f"Hyperparameters: k_train={config.CORPUS_K_TRAIN} | k_eval={config.CORPUS_K_EVAL} | "
          f"bs={config.CORPUS_BATCH} | ep={config.CORPUS_EPOCHS} | budget={budget}h", flush=True)
    print(f"Soft Label Blend: regex + {config.SOFT_LABEL_PARQUETS}", flush=True)
    print("=" * 80, flush=True)

    assert config.RAPTOR_ARCH == "convnext_small.fb_in22k_ft_in1k_384"
    assert config.CORPUS_TRAIN_LABELS == ["PF OA"]
    assert config.CORPUS_CENTRE_ALLOW == list(range(4, 14)) + list(range(20, 30))
    assert config.RAPTOR_RES == 384
    assert (config.RAPTOR_LR_BACKBONE, config.RAPTOR_LR_HEAD, config.RAPTOR_LLRD) == (2e-5, 1e-3, None)
    assert LEGAL_WINDOWS is False
    assert CORPUS_D == 64
    assert (config.CORPUS_K_TRAIN, config.CORPUS_K_EVAL,
            config.CORPUS_EPOCHS, config.CORPUS_BATCH) == (12, 20, 16, 2)
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
        seed=SEED,
        do_submit=False,
        time_budget_h=budget,
    )
    print("=" * 80, flush=True)
    print("PF-v1 Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
