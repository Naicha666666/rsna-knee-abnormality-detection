"""Score a candidate label source: GAUGE 1 + decorrelation vs the existing blend.

Why both numbers: C3 measured that a source's SOLO GAUGE 1 does not predict what
it buys. Qwen scored 0.880 alone -- better than Kimi's 0.876 -- but correlated
r=0.94 with Kimi, and adding it moved the 58-gold blend by +0.0013. C6 then
added a decorrelated third source (r=0.76) and the blend went the other way.
Whatever the sign, the quantity that distinguishes those runs is CORRELATION
WITH WHAT IS ALREADY IN THE BLEND, not the solo score. This script prints both
so a candidate is never promoted on its solo number alone.

Usage:
    python src/score_label_source.py <labels.jsonl|.parquet|.csv> [--name NAME]

Caveat printed with the output: the reference correlations in LOG.md
(kimi-qwen 0.940, grok-kimi 0.760) were computed over all 4,407 studies. A
58-study probe estimates the same quantity with far more noise -- treat it as
a screen ("is this obviously redundant?"), not as a precise r.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from config import LABELS, PROJECT_ROOT

LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "train.csv")


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Same rank-sum formula as report_extractor.roc_auc, so the number here is
    comparable with every GAUGE 1 already recorded in LOG.md."""
    yt = np.asarray(y_true, dtype=float)
    n_pos, n_neg = yt.sum(), len(yt) - yt.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    return (ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _flatten_nested(records: list) -> pd.DataFrame:
    """`[{StudyInstanceUID, "ACL": {p, conf}, ...}]` -> flat p / p__conf columns.

    This is the shape `data/labels/gold58_export/README.md` asks external
    platforms to return (same as GROK_RUBRIC.txt has always specified), so
    hand-collected results drop straight in with no manual conversion step.
    """
    rows = []
    for r in records:
        rec = {"StudyInstanceUID": str(r["StudyInstanceUID"])}
        for lab in LABELS:
            cell = r[lab]
            rec[lab] = float(cell["p"] if isinstance(cell, dict) else cell)
            rec[lab + "__conf"] = float(cell["conf"] if isinstance(cell, dict) else 0.5)
        rows.append(rec)
    return pd.DataFrame(rows)


def load_labels(path: str) -> pd.DataFrame:
    if path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON array of records")
        return _flatten_nested(data)
    if path.endswith(".jsonl"):
        rows = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        rows[r["StudyInstanceUID"]] = r
                    except json.JSONDecodeError:
                        continue
        return pd.DataFrame(list(rows.values()))
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Unique-index a label frame. Some published parquets carry duplicate
    StudyInstanceUIDs (a resumed/retried labelling run appending twice); with
    duplicates, .loc[list_of_uids] expands rows and every downstream length
    check silently disagrees. Last write wins, matching _read_jsonl."""
    return df.drop_duplicates("StudyInstanceUID", keep="last").set_index(
        "StudyInstanceUID")


def gauge1(df: pd.DataFrame, gold: pd.DataFrame, verbose: bool = True) -> float:
    d = _indexed(df)
    common = [u for u in gold.index if u in d.index]
    if not common:
        raise SystemExit("no gold studies in this file")
    y = gold.loc[common, LABELS].to_numpy(float)
    aucs = []
    for j, lab in enumerate(LABELS):
        if len(np.unique(y[:, j])) < 2:
            if verbose:
                print(f"    {lab:18s} only one class")
            continue
        a = roc_auc(y[:, j], d.loc[common, lab].to_numpy(float))
        aucs.append(a)
        if verbose:
            print(f"    {lab:18s} {a:.3f}")
    return float(np.mean(aucs))


def correlations(df_new: pd.DataFrame) -> pd.DataFrame:
    """Per-label Pearson r against every existing source, on shared studies."""
    new = _indexed(df_new)
    out = {}
    # Full-corpus sources, plus every earlier 58-study probe. The probes matter
    # because the sharpest question about a new candidate is often how it
    # relates to the LAST candidate (e.g. opus-5 vs sonnet-5: same vendor --
    # does that predict correlation, or is convergence task-driven?), and that
    # comparison is impossible if only the blend's own sources are in the table.
    paths = ([(os.path.basename(p)[len("labels_"):-len("_soft.parquet")], p)
              for p in sorted(glob.glob(os.path.join(LABELS_DIR, "labels_*_soft.parquet")))]
             # .jsonl = kbench runs, .json = hand-collected external platforms;
             # globbing only .jsonl silently left every external probe out of
             # the table, including the same-vendor predecessor comparisons.
             + [("probe:" + os.path.basename(p)[len("labels_"):].rsplit("_gold58", 1)[0], p)
                for p in sorted(glob.glob(os.path.join(
                    LABELS_DIR, "gold58_probes", "labels_*_gold58.json*")))])
    for name, path in paths:
        other = _indexed(load_labels(path))
        common = [u for u in new.index if u in other.index]
        if len(common) < 5:
            continue
        rs = []
        for lab in LABELS:
            a = new.loc[common, lab].to_numpy(float)
            b = other.loc[common, lab].to_numpy(float)
            # a constant column has no correlation to report, not r=0
            if np.std(a) == 0 or np.std(b) == 0:
                rs.append(np.nan)
                continue
            rs.append(float(np.corrcoef(a, b)[0, 1]))
        out[name] = pd.Series(rs, index=LABELS)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    name = args.name or os.path.basename(args.labels)

    df_new = load_labels(args.labels)
    tr = pd.read_csv(TRAIN_CSV)
    gold = tr[tr[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")

    print(f"=== {name}: {len(df_new)} studies ===\n")
    print("GAUGE 1 (per label):")
    macro = gauge1(df_new, gold)
    print(f"\n  macro-AUC = {macro:.4f}")
    print("  bar: regex 0.865 | kimi 0.876 | qwen 0.880 | grok 0.887 | "
          "gemma4 0.840 | gpt_oss 0.822")

    corr = correlations(df_new)
    if corr.empty:
        print("\nno overlapping existing sources found in data/labels/")
        return
    print("\nPer-label Pearson r vs existing sources:")
    print(corr.round(3).to_string())
    print("\nMedian r per source (the redundancy screen):")
    med = corr.median().sort_values(ascending=False)
    for src, r in med.items():
        tag = "  REDUNDANT" if r >= 0.90 else ("  decorrelated" if r <= 0.80 else "")
        print(f"  {src:12s} {r:.3f}{tag}")
    print(f"\n  max median r = {med.max():.3f} vs {med.idxmax()}")
    print("\nNOTE: LOG.md's reference correlations (kimi-qwen 0.940, grok-kimi "
          "0.760) were measured over all 4,407 studies. This is "
          f"{len(df_new)} studies -- a redundancy screen, not a precise r.")


if __name__ == "__main__":
    main()
