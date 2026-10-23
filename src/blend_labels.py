"""Blend multiple LLM soft-label parquets into an ensembled label set.

Supports:
- Equal-weight or custom-weighted probability & confidence averaging.
- Optional integration of regex baseline.
- Automatic verification against the 58 Gold standard studies (GAUGE 1).
- Generates standard parquet format ready for dataset.py and training pipelines.

Usage:
    python src/blend_labels.py --sources grok,qwen,kimi,luna,nemotron --out data/labels/labels_5llm_mean.parquet
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LABELS, PROJECT_ROOT
from score_label_source import roc_auc, gauge1

LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "train.csv")


def resolve_source_path(name: str) -> str:
    name = name.strip()
    if name.endswith(".parquet") and os.path.exists(name):
        return name
    candidate = os.path.join(LABELS_DIR, f"labels_{name}_soft.parquet")
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.join(LABELS_DIR, name)
    if os.path.exists(candidate2):
        return candidate2
    raise FileNotFoundError(f"Cannot resolve label source: '{name}' in {LABELS_DIR}")


def build_blend(
    source_names: list[str],
    weights: list[float] | None = None,
    include_regex: bool = False,
    out_parquet: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load sources, average probabilities & confidences, and save to parquet."""
    tr = pd.read_csv(TRAIN_CSV)
    uids = tr["StudyInstanceUID"].astype(str).tolist()
    reports = tr["Report"].fillna("").tolist()

    loaded_sources = []
    names = []
    for s in source_names:
        s_clean = s.strip()
        path = resolve_source_path(s_clean)
        df = pd.read_parquet(path).drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")
        loaded_sources.append(df)
        names.append(s_clean)
        if verbose:
            print(f"Loaded {s_clean:12s}: {len(df)} studies from {os.path.basename(path)}")

    if include_regex:
        import report_extractor as re_ext
        if verbose:
            print("Generating Regex extractor baseline for all 4,407 reports...")
        reg_rows = []
        for uid, rep in zip(uids, reports):
            g, c = re_ext.extract(rep)
            row = {"StudyInstanceUID": uid}
            for lab in LABELS:
                row[lab] = g[lab]
                row[lab + "__conf"] = c[lab]
            reg_rows.append(row)
        df_reg = pd.DataFrame(reg_rows).set_index("StudyInstanceUID")
        loaded_sources.append(df_reg)
        names.append("regex")

    N_src = len(loaded_sources)
    if weights is None:
        w_arr = np.ones(N_src, dtype=float) / N_src
    else:
        w_arr = np.array(weights, dtype=float)
        w_arr = w_arr / w_arr.sum()

    if verbose:
        print("\nWeights per source:")
        for nm, w in zip(names, w_arr):
            print(f"  {nm:15s}: {w:.3f} ({w*100:.1f}%)")

    # Blend across all UIDs
    out_rows = []
    for uid in uids:
        rec = {"StudyInstanceUID": uid}
        for lab in LABELS:
            p_vals = []
            c_vals = []
            for df_src in loaded_sources:
                if uid in df_src.index:
                    p_vals.append(float(df_src.at[uid, lab]))
                    c_vals.append(float(df_src.at[uid, lab + "__conf"]))
                else:
                    p_vals.append(0.05)
                    c_vals.append(0.05)
            rec[lab] = float(np.dot(w_arr, p_vals))
            rec[lab + "__conf"] = float(np.dot(w_arr, c_vals))
        out_rows.append(rec)

    df_blend = pd.DataFrame(out_rows)

    if out_parquet:
        os.makedirs(os.path.dirname(os.path.abspath(out_parquet)), exist_ok=True)
        df_blend.to_parquet(out_parquet, index=False)
        if verbose:
            print(f"\nSaved blended parquet to {out_parquet} ({len(df_blend)} studies)")

    # Score against Gold standard
    gold = tr[tr[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")
    common = [u for u in df_blend["StudyInstanceUID"] if u in gold.index]
    if common:
        macro = gauge1(df_blend, gold, verbose=False)
        df_eval = df_blend.set_index("StudyInstanceUID")
        if verbose:
            print(f"\n=== GAUGE 1 Validation on {len(common)} Gold Studies ===")
            for lab in LABELS:
                y_true = gold.loc[common, lab].to_numpy(float)
                y_score = df_eval.loc[common, lab].to_numpy(float)
                a = roc_auc(y_true, y_score)
                print(f"  {lab:18s}: {a:.4f}")
            print(f"\n  Blended Macro-AUC = {macro:.4f}")

    return df_blend


def main():
    parser = argparse.ArgumentParser(description="Blend LLM soft labels")
    parser.add_argument(
        "--sources",
        type=str,
        default="grok,qwen,kimi,luna,nemotron",
        help="Comma-separated list of source names or paths",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional comma-separated weights (defaults to equal)",
    )
    parser.add_argument(
        "--include-regex",
        action="store_true",
        help="Include regex baseline in blend",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(LABELS_DIR, "labels_5llm_mean.parquet"),
        help="Output parquet destination",
    )
    args = parser.parse_args()

    src_list = [s.strip() for s in args.sources.split(",") if s.strip()]
    weights = [float(w) for w in args.weights.split(",")] if args.weights else None

    build_blend(
        source_names=src_list,
        weights=weights,
        include_regex=args.include_regex,
        out_parquet=args.out,
        verbose=True,
    )


if __name__ == "__main__":
    main()
