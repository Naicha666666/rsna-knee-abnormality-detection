"""Claude in-session labeler for RSNA Knee reports.

Same rubric, JSON schema, parquet layout and GAUGE 1 scorer as other backends.
Claude reads chunk in_XXX.json and produces out_XXX.json, which this script
merges into labels_claude_soft.parquet.

Usage:
  # check progress
  python src/claude_labeler.py status

  # merge completed out_XXX.json files
  python src/claude_labeler.py merge

  # run gauge on completed parquet
  python src/claude_labeler.py gauge
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import LABELS

CHUNK_DIR = os.path.join("data", "labels", "claude_chunks")
OUT_PARQUET = os.path.join("data", "labels", "labels_claude_soft.parquet")


def status():
    in_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "in_*.json")))
    out_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "out_*.json")))
    done_idxs = {os.path.basename(f).replace("out_", "").replace(".json", "") for f in out_files}
    total_in = sum(len(json.load(open(f, encoding="utf-8"))) for f in in_files)
    total_out = sum(len(json.load(open(f, encoding="utf-8"))) for f in out_files) if out_files else 0
    print(f"Chunks total : {len(in_files)} ({total_in} studies)")
    print(f"Chunks done  : {len(out_files)} ({total_out} studies)")
    print(f"Remaining    : {len(in_files) - len(out_files)} chunks ({total_in - total_out} studies)")
    # list first 5 pending
    pending = []
    for f in in_files:
        idx = os.path.basename(f).replace("in_", "").replace(".json", "")
        if idx not in done_idxs:
            pending.append(os.path.basename(f))
    if pending:
        print(f"Next pending : {pending[:5]}")
    return len(in_files) - len(out_files)


def rec_from_dict(uid: str, d: dict) -> dict:
    rec = {"StudyInstanceUID": str(uid)}
    for lab in LABELS:
        cell = d[lab]
        if not isinstance(cell, dict):
            raise ValueError(f"{uid} {lab} not an object: {cell!r}")
        rec[lab] = float(cell["p"])
        rec[lab + "__conf"] = float(cell["conf"])
        if not (0.0 <= rec[lab] <= 1.0 and 0.0 <= rec[lab + "__conf"] <= 1.0):
            raise ValueError(f"{uid} {lab} out of range")
    return rec


def merge():
    out_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "out_*.json")))
    recs, bad = [], []
    for p in out_files:
        data = json.load(open(p, encoding="utf-8"))
        if not isinstance(data, list):
            bad.append((p, "not a list"))
            continue
        for r in data:
            try:
                recs.append(rec_from_dict(r["StudyInstanceUID"], r))
            except Exception as e:
                bad.append((p, str(e)))
    if bad:
        print(f"WARNING: {len(bad)} bad records; first: {bad[:5]}")
    df_new = pd.DataFrame(recs)
    if os.path.exists(OUT_PARQUET):
        prev = pd.read_parquet(OUT_PARQUET)
        keep = prev[~prev["StudyInstanceUID"].isin(df_new["StudyInstanceUID"])]
        df = pd.concat([keep, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def gauge():
    import report_extractor as R
    tr = pd.read_csv(os.path.join("data", "raw", "train.csv"))
    gold = tr[tr[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")
    df = pd.read_parquet(OUT_PARQUET).set_index("StudyInstanceUID")
    common = [u for u in gold.index if u in df.index]
    if not common:
        raise SystemExit("parquet has none of the 58 gold studies yet")
    y = gold.loc[common, LABELS].to_numpy(float)
    aucs = []
    print(f"GAUGE 1 (Claude) on {len(common)} gold studies:")
    for j, lab in enumerate(LABELS):
        col = df.loc[common, lab].to_numpy(float)
        if len(np.unique(y[:, j])) < 2:
            print(f"  {lab:18s} only one class")
            continue
        a = R.roc_auc(y[:, j], col)
        aucs.append(a)
        print(f"  {lab:18s} AUC={a:.3f}")
    macro = float(np.mean(aucs))
    print(f"  macro-AUC={macro:.3f}   (regex extractor: 0.865, Kimi: 0.889)")
    return macro


def read_chunk(chunk_idx: int) -> list:
    """Return list of {uid, report} dicts for a given chunk index."""
    path = os.path.join(CHUNK_DIR, f"in_{chunk_idx:03d}.json")
    return json.load(open(path, encoding="utf-8"))


def write_chunk_out(chunk_idx: int, records: list):
    """Write labeled records to out_XXX.json and merge into parquet."""
    path = os.path.join(CHUNK_DIR, f"out_{chunk_idx:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f"Saved {path} ({len(records)} records)")
    merge()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if cmd == "status":
        status()
    elif cmd == "merge":
        merge()
    elif cmd == "gauge":
        gauge()
    else:
        print(f"Unknown command: {cmd}")
