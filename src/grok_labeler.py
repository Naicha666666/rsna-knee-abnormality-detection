"""Grok backend for the report labelling line (this TUI / agent, not api.x.ai).

Same rubric, JSON schema, parquet layout and GAUGE 1 scorer as the other
LLM backends (`llm_labeler`, `moonshot_labeler`, `ollama_labeler`). Records
are produced by Grok in-session (or by subagents) and merged here so the
downstream dataset contract stays one file: StudyInstanceUID + 12 p + 12 conf.
"""
from __future__ import annotations
import json
import os

import pandas as pd

from config import LABELS
from llm_labeler import RUBRIC, _SCHEMA, gauge

OUT_NAME = "labels_grok_soft.parquet"


def rec_from_dict(uid: str, d: dict) -> dict:
    rec = {"StudyInstanceUID": str(uid)}
    for lab in LABELS:
        cell = d[lab]
        if not isinstance(cell, dict):
            raise ValueError(f"{uid} {lab} not an object")
        rec[lab] = float(cell["p"])
        rec[lab + "__conf"] = float(cell["conf"])
        if not (0.0 <= rec[lab] <= 1.0 and 0.0 <= rec[lab + "__conf"] <= 1.0):
            raise ValueError(f"{uid} {lab} out of range")
    return rec


def merge_records(records: list, out_parquet: str) -> pd.DataFrame:
    """Append/upsert records into the parquet (resumable)."""
    recs = [rec_from_dict(r["StudyInstanceUID"], r) if LABELS[0] in r
            and isinstance(r.get(LABELS[0]), dict) else r for r in records]
    df_new = pd.DataFrame(recs)
    if os.path.exists(out_parquet):
        prev = pd.read_parquet(out_parquet)
        keep = prev[~prev["StudyInstanceUID"].isin(df_new["StudyInstanceUID"])]
        df = pd.concat([keep, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet}: {df.shape[0]} rows", flush=True)
    return df


def load_reports(train_csv: str, gold_only: bool = False) -> dict:
    tr = pd.read_csv(train_csv)
    if gold_only:
        tr = tr[tr[LABELS].notna().all(axis=1)]
    return {str(u): ("" if pd.isna(r) else str(r))
            for u, r in zip(tr["StudyInstanceUID"], tr["Report"])}


def dump_remaining_chunks(train_csv: str, out_dir: str, chunk_size: int = 12) -> int:
    """Write the 4349 non-gold reports as in_XXX.json lists of {uid, report}."""
    os.makedirs(out_dir, exist_ok=True)
    tr = pd.read_csv(train_csv)
    rest = tr.loc[~tr[LABELS].notna().all(axis=1), ["StudyInstanceUID", "Report"]]
    rows = rest.to_dict("records")
    n_chunks = (len(rows) + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        chunk = rows[i * chunk_size:(i + 1) * chunk_size]
        payload = [{"uid": str(r["StudyInstanceUID"]),
                    "report": "" if pd.isna(r["Report"]) else str(r["Report"])}
                   for r in chunk]
        path = os.path.join(out_dir, f"in_{i:03d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    manifest = {"n_reports": len(rows), "n_chunks": n_chunks,
                "chunk_size": chunk_size, "out_dir": out_dir}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"dumped {len(rows)} reports into {n_chunks} chunks -> {out_dir}",
          flush=True)
    return n_chunks


def merge_chunk_dir(out_dir: str, parquet: str) -> pd.DataFrame:
    """Upsert every out_XXX.json in out_dir into the grok parquet."""
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, "out_*.json")))
    recs, bad = [], []
    for p in files:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            bad.append((p, "not a list"))
            continue
        for r in data:
            try:
                recs.append(rec_from_dict(r["StudyInstanceUID"], r))
            except Exception as e:
                bad.append((p, str(e)))
    if bad:
        print(f"WARNING {len(bad)} bad records; first: {bad[:8]}", flush=True)
    if not recs:
        raise SystemExit(f"no valid records in {out_dir}")
    return merge_records(recs, parquet)
