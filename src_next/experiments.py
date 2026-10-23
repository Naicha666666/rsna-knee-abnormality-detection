"""Reproducible experiment registry + runner for the RSNA Knee comparison study.

Philosophy (paper-grade): the training PROTOCOL is fixed; the only thing that
varies between experiments is the image BACKBONE (tiny_cnn / dinov2_small /
dinov2_base) plus, for one ablation, whether the encoder is fine-tuned. All
other settings (seed, CV scheme, slots, preprocessing, losses, EMA) are pinned in
config.py / train.py and shared by every run.

Each run writes a self-contained artifact under experiments/exp_<id>/:
    config.json   - exact configuration
    metrics.json  - macro-AUC, per-label AUC, fold scores, n studies/gold
    oof.npy       - out-of-fold predictions (NaN where unlabeled)
    provenance.json - environment + source hash + fold assignment + CLI (repro)
so any result can be regenerated and audited.

Usage:
    python experiments.py --exp exp002          # run one experiment
    python experiments.py --summary              # rebuild results/ from all runs
"""
from __future__ import annotations
import os
import sys
import json
import hashlib
import datetime
import platform

import numpy as np

from config import (LABELS, SEED, PROJECT_ROOT, BATCH_SIZE, EPOCHS, LR,
                    ENCODER_UNFREEZE_LAST)
import train as T

# On Kaggle the kernel is a single inlined file; artifacts must land in
# /kaggle/working (the exposed kernel output). Locally they land under the repo.
_KAGGLE_WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else None
EXP_DIR = os.path.join(_KAGGLE_WORKING, "experiments") if _KAGGLE_WORKING \
    else os.path.join(PROJECT_ROOT, "experiments")
RESULTS_DIR = os.path.join(_KAGGLE_WORKING, "results") if _KAGGLE_WORKING \
    else os.path.join(PROJECT_ROOT, "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


# ----------------------------- experiment registry -----------------------------
class ExpConfig:
    def __init__(self, exp_id, backbone, unfreeze_last=ENCODER_UNFREEZE_LAST,
                 max_studies=None, epochs=EPOCHS, folds=5,
                 batch=None, lr=LR, note=""):
        self.exp_id = exp_id
        self.backbone = backbone
        self.unfreeze_last = unfreeze_last
        self.max_studies = max_studies
        self.epochs = epochs
        self.folds = folds
        self.batch = batch or (2 if backbone.startswith("dinov2") else BATCH_SIZE)
        self.lr = lr
        self.note = note

    def to_dict(self):
        return dict(
            exp_id=self.exp_id, backbone=self.backbone,
            unfreeze_last=self.unfreeze_last, max_studies=self.max_studies,
            epochs=self.epochs, folds=self.folds, batch=self.batch, lr=self.lr,
            note=self.note, seed=SEED)


# Core 3-way comparison + 1 fine-tune ablation, each at FULL data (20 epochs),
# plus fast SCREENING runs (subset=600 studies, 5 epochs) used to (a) verify the
# DINOv2 offline-loading path on T4 and (b) get a cheap relative ranking before
# committing to the expensive full runs.
EXP_REGISTRY = {
    "exp001":   ExpConfig("exp001", "tiny_cnn", unfreeze_last=2,
                          note="control: from-scratch CNN, no pretraining"),
    "exp002":   ExpConfig("exp002", "dinov2_small", unfreeze_last=2,
                          note="DINOv2 ViT-S, fine-tune last 2 blocks"),
    "exp003":   ExpConfig("exp003", "dinov2_base", unfreeze_last=2,
                          note="DINOv2 ViT-B, fine-tune last 2 blocks"),
    "exp002f":  ExpConfig("exp002f", "dinov2_small", unfreeze_last=0,
                          note="ABLATION: DINOv2 ViT-S FROZEN (no fine-tune)"),
    "exp001_scr": ExpConfig("exp001_scr", "tiny_cnn", unfreeze_last=2,
                            max_studies=600, epochs=5,
                            note="screening: tiny_cnn subset"),
    "exp002_scr": ExpConfig("exp002_scr", "dinov2_small", unfreeze_last=2,
                            max_studies=600, epochs=5,
                            note="screening: dinov2_small subset"),
    "exp003_scr": ExpConfig("exp003_scr", "dinov2_base", unfreeze_last=2,
                            max_studies=600, epochs=5,
                            note="screening: dinov2_base subset"),
    # FULL controlled comparison (single fixed 80/20 split, N=1500, E=10).
    # folds=1 => identical group-disjoint split across backbones (see train.run_cv).
    "tiny_cnn_1500_ep10": ExpConfig("tiny_cnn_1500_ep10", "tiny_cnn",
                                    unfreeze_last=2, max_studies=1500, epochs=10,
                                    folds=1, note="FULL: from-scratch CNN, single split"),
    "dinov2_small_1500_ep10": ExpConfig("dinov2_small_1500_ep10", "dinov2_small",
                                        unfreeze_last=2, max_studies=1500, epochs=10,
                                        folds=1, note="FULL: DINOv2 ViT-S, single split"),
    "dinov2_base_1500_ep10": ExpConfig("dinov2_base_1500_ep10", "dinov2_base",
                                       unfreeze_last=2, max_studies=1500, epochs=10,
                                       folds=1, note="FULL: DINOv2 ViT-B, single split"),
    # v2 submit protocol: 2.5D cache + 4-fold rank-average (see train.run_submit).
    "sub_dinov2s_v2": ExpConfig("sub_dinov2s_v2", "dinov2_small",
                                unfreeze_last=2, max_studies=None, epochs=10,
                                folds=4, batch=8,
                                note="SUBMIT v2: DINOv2-S, cache+2.5D+4fold"),
}


# ----------------------------- provenance / reproducibility -----------------------------
def _source_hash() -> str:
    """SHA-256 over all src .py files (recursive) for exact-code pinning.

    On Kaggle the code is a single inlined file (no src/ tree), so we fall back
    to hashing the running script itself."""
    h = hashlib.sha256()
    if os.path.isdir(SRC_DIR):
        for root, _, files in os.walk(SRC_DIR):
            for f in sorted(files):
                if f.endswith(".py") and "__pycache__" not in root:
                    with open(os.path.join(root, f), "rb") as fh:
                        h.update(fh.read())
        return h.hexdigest()
    # Kaggle: hash the inlined run.py
    try:
        with open(sys.argv[0], "rb") as fh:
            return h.hexdigest()
    except Exception:
        return "kaggle-inline"


def _env_snapshot() -> dict:
    import torch
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        platform=platform.platform(),
    )


# ----------------------------- run one experiment -----------------------------
def run_cv_and_save(cfg: "ExpConfig"):
    """Run CV for an arbitrary ExpConfig and write its artifact directory."""
    exp_id = cfg.exp_id
    out_dir = os.path.join(EXP_DIR, exp_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== RUN {exp_id}: backbone={cfg.backbone} "
          f"unfreeze_last={cfg.unfreeze_last} studies="
          f"{cfg.max_studies or 'FULL'} epochs={cfg.epochs} folds={cfg.folds} ===")

    res = T.run_cv(backbone_name=cfg.backbone, unfreeze_last=cfg.unfreeze_last,
                   max_studies=cfg.max_studies, epochs=cfg.epochs,
                   folds=cfg.folds, batch=cfg.batch, lr=cfg.lr)

    metrics = dict(
        exp_id=exp_id, macro_auc=float(res["macro"]),
        mean_fold=float(np.mean(res["fold_scores"])),
        std_fold=float(np.std(res["fold_scores"])),
        fold_scores=[float(x) for x in res["fold_scores"]],
        per_label={LABELS[i]: (None if np.isnan(a) else float(a))
                   for i, a in enumerate(res["per_label"])},
        n_studies=len(res["idx"]),
        epochs=cfg.epochs, folds=cfg.folds,
        backbone=cfg.backbone, unfreeze_last=cfg.unfreeze_last,
        max_studies=cfg.max_studies, batch=cfg.batch, lr=cfg.lr,
    )
    prov = dict(
        env=_env_snapshot(),
        source_hash=_source_hash(),
        seed=SEED,
        argv=sys.argv,
        generated_at=datetime.datetime.utcnow().isoformat() + "Z",
        fold_splits=res["fold_splits"],
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    np.save(os.path.join(out_dir, "oof.npy"), res["oof"])

    print(f"[{exp_id}] macro-AUC={metrics['macro_auc']:.4f} "
          f"(mean fold {metrics['mean_fold']:.4f} +- {metrics['std_fold']:.4f})")
    print(f"[{exp_id}] saved -> {out_dir}")
    return metrics


def run_experiment(exp_id: str):
    if exp_id not in EXP_REGISTRY:
        raise KeyError(f"unknown experiment {exp_id}; known: {list(EXP_REGISTRY)}")
    return run_cv_and_save(EXP_REGISTRY[exp_id])


# ----------------------------- summarize all experiments -----------------------------
def _load_metrics(exp_id):
    p = os.path.join(EXP_DIR, exp_id, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def summarize():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for exp_id in EXP_REGISTRY:
        m = _load_metrics(exp_id)
        if not m:
            continue
        rows.append(m)
    rows.sort(key=lambda r: r["exp_id"])

    # results.csv
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    import csv
    fields = ["exp_id", "backbone", "unfreeze_last", "max_studies", "epochs",
              "folds", "batch", "macro_auc", "mean_fold", "std_fold"]
    fields += [f"auc_{l}" for l in LABELS]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in
                   ["exp_id", "backbone", "unfreeze_last", "max_studies",
                    "epochs", "folds", "batch", "macro_auc", "mean_fold",
                    "std_fold"]}
            for l in LABELS:
                v = r["per_label"].get(l)
                row[f"auc_{l}"] = ("" if v is None else f"{v:.4f}")
            w.writerow(row)

    # results.md (paper-style comparison table)
    md = ["# RSNA Knee — Method Comparison (paper-style)", "",
          "> Controlled experiment: the training protocol is FIXED; only the image "
          "backbone varies. CV macro-AUC is computed on the held-out GOLD portion "
          "(public labels are only 58 studies, so it is a *relative* ranking "
          "signal; the hidden test leaderboard is the final arbiter).", "",
          "## Results", "",
          "| exp_id | backbone | tune | data | ep | macro-AUC | mean±std fold |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        data = "FULL" if not r["max_studies"] else f"sub{r['max_studies']}"
        tune = "ft" if r["unfreeze_last"] > 0 else "frozen"
        md.append(
            f"| {r['exp_id']} | {r['backbone']} | {tune} | {data} | "
            f"{r['epochs']} | {r['macro_auc']:.4f} | "
            f"{r['mean_fold']:.4f}±{r['std_fold']:.4f} |")
    md += ["", "## Per-label AUC (held-out gold)", "",
           "| exp_id | " + " | ".join(LABELS) + " |",
           "|---" + "|".join(["---"] * len(LABELS)) + "|"]
    for r in rows:
        cells = []
        for l in LABELS:
            v = r["per_label"].get(l)
            cells.append("" if v is None else f"{v:.3f}")
        md.append(f"| {r['exp_id']} | " + " | ".join(cells) + " |")
    md += ["", "## Reproducibility", "",
           "Each `experiments/exp_<id>/` contains `config.json`, `metrics.json`, "
           "`oof.npy`, and `provenance.json` (env + source SHA-256 + fold "
           "assignment + CLI). Regenerate with `python experiments.py --exp "
           "<id>`.", "",
           "## Limitations", "",
           "- Public gold labels = 58 studies; CV macro-AUC is a noisy *relative* "
           "signal, not an absolute estimate. Final ranking uses the hidden test "
           "leaderboard (public + private).",
           "- Physical-scale crop and laterality normalization rely on DICOM "
           "headers present in the Kaggle environment; they degrade gracefully "
           "(fixed crop / no flip) if stripped."]
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {csv_path} and {os.path.join(RESULTS_DIR, 'RESULTS.md')}")
    return rows


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarize()
    elif len(sys.argv) > 2 and sys.argv[1] == "--exp":
        run_experiment(sys.argv[2])
    else:
        print("usage: python experiments.py --exp <id> | --summary")


if __name__ == "__main__":
    main()
