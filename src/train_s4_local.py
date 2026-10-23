"""S4 on R9700: PF OA ordered-window study, arm T (true order) or K (order destroyed) (LOG #522/#523).

Pre-registration: results/s4_preregistration_draft.md (frozen, LOG #523). ONE script for both arms so
they are identical by construction; RSNA_S4_ARM=T|K selects S4_ORDER = "true" | "shuffled" and the output
directory, nothing else.
  * = PF-v1: ConvNeXt-S 384, corpus64, regex + 4 LLM soft labels, PF-OA-only BCE and selection,
    the 20 central-sagittal centres (fluid 4-13, structural 20-29), 16 epochs, bs 2, seed 42, same LR
    (backbone 2e-5, head 1e-3 -- the head group now includes the Transformer), bf16, top-3 average.
  * new for both arms: RAPTOR_HEAD = "s4seq" (2-layer Transformer + learned position embedding, then
    per-diagnosis attention pooling), training bag = a contiguous run of 12 centres (random start,
    same RNG sequence in both arms), k_eval = 20 (all centres).
  * T: positions = true index of each centre in the allow-list.
    K: positions = a fixed per-study permutation of those indices (SHA-256 of "S4-perm-v1:" + uid),
       manifest written next to the checkpoint.
RSNA_SMOKE=1: 8 training studies, 1 epoch, output under /tmp (smoke test only).
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
config.RAPTOR_ARCH = "convnext_small.fb_in22k_ft_in1k_384"   # = C10
# the specialist (= PF-v1)
config.CORPUS_TRAIN_LABELS = ["PF OA"]
config.CORPUS_CENTRE_ALLOW = list(range(4, 14)) + list(range(20, 30))
# S4 (LOG #523): the only arm-specific setting is S4_ORDER
ARM = os.environ.get("RSNA_S4_ARM", "")
assert ARM in ("T", "K"), "set RSNA_S4_ARM=T or K"
config.RAPTOR_HEAD = "s4seq"
config.S4_ORDER = {"T": "true", "K": "shuffled"}[ARM]
config.CORPUS_TRAIN_CONTIGUOUS = True
SMOKE = os.environ.get("RSNA_SMOKE") == "1"
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
    default_out = (f"/tmp/s4{ARM.lower()}_smoke" if SMOKE else
                   os.path.join(os.path.dirname(HERE), "experiments",
                                f"local_s4{ARM.lower()}_64_k12_ep16_convnext_pfoa_censag_seq"))
    out = os.environ.get("RSNA_OUT_DIR", default_out)
    os.makedirs(out, exist_ok=True)
    budget = float(os.environ.get("RSNA_TIME_BUDGET_H", "36.0"))

    print("=" * 80, flush=True)
    print(f"S4 arm {ARM} on AMD R9700 (ROCm): PF OA, ordered-window head, S4_ORDER={config.S4_ORDER}"
          + (" | SMOKE" if SMOKE else ""), flush=True)
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
    assert (config.RAPTOR_HEAD, config.CORPUS_TRAIN_CONTIGUOUS) == ("s4seq", True)
    assert config.S4_ORDER == {"T": "true", "K": "shuffled"}[ARM]
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
        epochs=1 if SMOKE else config.CORPUS_EPOCHS,
        max_studies=8 if SMOKE else None,
        batch=config.CORPUS_BATCH,
        k_train=config.CORPUS_K_TRAIN,
        k_eval=config.CORPUS_K_EVAL,
        out_dir=out,
        seed=42,
        do_submit=False,
        time_budget_h=budget,
    )
    print("=" * 80, flush=True)
    print(f"S4 arm {ARM} Training Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
