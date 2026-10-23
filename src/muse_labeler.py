"""Muse Spark in-session labeler (no API, this session only).

Same contract as grok_labeler / claude_labeler:
  in : data/labels/muse_chunks/in_XXX.json  (list of {uid, report})
  out: data/labels/muse_chunks/out_XXX.json (list of {StudyInstanceUID, 12x{p,conf}})
  parquet: data/labels/labels_muse_soft.parquet (StudyInstanceUID + 12 p + 12 conf)

Rubric / schema / GAUGE identical to llm_labeler so the source is comparable.
Usage:
  python src/muse_labeler.py dump    # 4407 -> muse_chunks/in_*.json (chunk 12)
  python src/muse_labeler.py status
  python src/muse_labeler.py merge   # out_*.json -> labels_muse_soft.parquet
  python src/muse_labeler.py gauge
"""
from __future__ import annotations
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LABELS
from llm_labeler import RUBRIC  # noqa: F401  (single source of truth)

CHUNK_DIR = os.path.join("data", "labels", "muse_chunks")
OUT_PARQUET = os.path.join("data", "labels", "labels_muse_soft.parquet")


def dump(chunk_size: int = 12):
    os.makedirs(CHUNK_DIR, exist_ok=True)
    tr = pd.read_csv(os.path.join("data", "raw", "train.csv"))
    rows = tr[["StudyInstanceUID", "Report"]].to_dict("records")
    n = (len(rows) + chunk_size - 1) // chunk_size
    for i in range(n):
        part = rows[i * chunk_size:(i + 1) * chunk_size]
        payload = [{"uid": str(r["StudyInstanceUID"]),
                    "report": "" if pd.isna(r["Report"]) else str(r["Report"])}
                   for r in part]
        with open(os.path.join(CHUNK_DIR, f"in_{i:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    with open(os.path.join(CHUNK_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"n_reports": len(rows), "n_chunks": n, "chunk_size": chunk_size,
                   "rubric": "llm_labeler.RUBRIC", "backend": "muse-spark-in-session"}, f, indent=2)
    print(f"dumped {len(rows)} reports into {n} chunks -> {CHUNK_DIR}", flush=True)
    return n


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


def status():
    in_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "in_*.json")))
    out_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "out_*.json")))
    done = {os.path.basename(f).replace("out_", "").replace(".json", "") for f in out_files}
    tin = sum(len(json.load(open(f, encoding="utf-8"))) for f in in_files) if in_files else 0
    tout = sum(len(json.load(open(f, encoding="utf-8"))) for f in out_files) if out_files else 0
    print(f"Chunks total : {len(in_files)} ({tin} studies)")
    print(f"Chunks done  : {len(out_files)} ({tout} studies)")
    print(f"Remaining    : {len(in_files) - len(out_files)} chunks ({tin - tout} studies)")
    pend = [os.path.basename(f) for f in in_files
            if os.path.basename(f).replace("in_", "").replace(".json", "") not in done][:5]
    if pend:
        print(f"Next pending : {pend}")
    if os.path.exists(OUT_PARQUET):
        df = pd.read_parquet(OUT_PARQUET)
        print(f"Parquet now  : {OUT_PARQUET}: {df.shape[0]} rows x {df.shape[1]} cols")
    return len(in_files) - len(out_files)


def merge() -> pd.DataFrame:
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
            except Exception as e:  # noqa: BLE001
                bad.append((p, str(e)))
    if bad:
        print(f"WARNING: {len(bad)} bad records; first: {bad[:5]}")
    if not recs:
        raise SystemExit(f"no valid records in {CHUNK_DIR}")
    df_new = pd.DataFrame(recs)
    if os.path.exists(OUT_PARQUET):
        prev = pd.read_parquet(OUT_PARQUET)
        keep = prev[~prev["StudyInstanceUID"].isin(df_new["StudyInstanceUID"])]
        df = pd.concat([keep, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET}: {df.shape[0]} rows, {df.shape[1]} cols", flush=True)
    return df


def gauge():
    import report_extractor as R
    import numpy as np
    tr = pd.read_csv(os.path.join("data", "raw", "train.csv"))
    gold = tr[tr[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")
    df = pd.read_parquet(OUT_PARQUET).set_index("StudyInstanceUID")
    common = [u for u in gold.index if u in df.index]
    if not common:
        raise SystemExit("parquet holds none of the 58 gold studies")
    y = gold.loc[common, LABELS].to_numpy(float)
    aucs = []
    print(f"GAUGE 1 (Muse) on {len(common)} gold studies:")
    for j, lab in enumerate(LABELS):
        col = df.loc[common, lab].to_numpy(float)
        if len(np.unique(y[:, j])) < 2:
            print(f"  {lab:18s} only one class")
            continue
        a = R.roc_auc(y[:, j], col)
        aucs.append(a)
        print(f"  {lab:18s} AUC={a:.3f}")
    macro = float(np.mean(aucs))
    print(f"  macro-AUC={macro:.3f}   (regex 0.865 / kimi 0.876 / qwen 0.880 / grok 0.887)")
    return macro


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if cmd == "dump":
        dump()
    elif cmd == "status":
        status()
    elif cmd == "merge":
        merge()
    elif cmd == "gauge":
        gauge()
    else:
        print(f"Unknown command: {cmd}")
