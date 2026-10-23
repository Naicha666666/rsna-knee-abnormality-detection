"""Validate + merge externally-labelled chunks into a soft-label parquet.

Written after three distinct failure modes showed up in one afternoon of
hand-collecting labels from chat platforms (LOG.md #353 / #356):

  1. TRUNCATION   - the platform cut the reply mid-record; the file does not
                    even parse. Loud, easy to spot.
  2. SILENT DROP  - the file parses, every record is complete, but one study
                    is simply missing from the middle. **Invisible** unless the
                    uid set is compared against what was actually asked for.
  3. KEY DRIFT    - a label arrives under a near-miss name ("Bakers", "PF_OA").

So this checks the uid set against `full_export/manifest.json` (the record of
what each chunk asked for) rather than against itself, normalises label keys,
and reports exactly which studies are still missing so they can be re-run
alone instead of re-running a whole 100-study chunk.

Usage:
    python src/merge_external_labels.py <chunk_dir> --name <source>
    # e.g. python src/merge_external_labels.py ../data/labels/full_export/gpt55 --name gpt55
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import pandas as pd

from config import LABELS, PROJECT_ROOT

EXPORT_DIR = os.path.join(PROJECT_ROOT, "data", "labels", "full_export")
LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels")

# Near-miss spellings seen from real platforms; same idea as
# annotate_gemma4_32c.KEY_MAP, kept here so a hand-collected file does not
# have to be corrected by hand.
KEY_MAP = {
    "bakers": "Baker's", "baker s": "Baker's", "baker's": "Baker's",
    "pf_oa": "PF OA", "pf oa": "PF OA", "pfoa": "PF OA",
    "medial_oa": "Medial OA", "lateral_oa": "Lateral OA",
    "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus",
}


def _canon(k: str) -> str:
    return KEY_MAP.get(k.strip().lower(), k.strip())


def load_chunk(path: str) -> tuple[list, list]:
    """Return (records, problems) for one out_*.json."""
    try:
        data = json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [], [f"{os.path.basename(path)}: UNPARSEABLE ({e}) -- likely truncated"]
    if not isinstance(data, list):
        return [], [f"{os.path.basename(path)}: not a JSON array"]

    recs, problems = [], []
    for i, r in enumerate(data):
        r = {_canon(k): v for k, v in r.items()}
        uid = r.get("StudyInstanceUID")
        if not uid:
            problems.append(f"{os.path.basename(path)}[{i}]: no StudyInstanceUID")
            continue
        missing = [l for l in LABELS if l not in r]
        if missing:
            problems.append(f"{os.path.basename(path)}[{uid[-8:]}]: missing {missing}")
            continue
        rec = {"StudyInstanceUID": str(uid)}
        ok = True
        for lab in LABELS:
            cell = r[lab]
            try:
                p = float(cell["p"] if isinstance(cell, dict) else cell)
                c = float(cell["conf"] if isinstance(cell, dict) else 0.5)
            except (TypeError, ValueError, KeyError):
                problems.append(f"{os.path.basename(path)}[{uid[-8:]}]: bad {lab}")
                ok = False
                break
            if not (0.0 <= p <= 1.0 and 0.0 <= c <= 1.0):
                problems.append(f"{os.path.basename(path)}[{uid[-8:]}]: {lab} out of [0,1]")
                ok = False
                break
            rec[lab], rec[lab + "__conf"] = p, c
        if ok:
            recs.append(rec)
    return recs, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk_dir", help="directory holding out_*.json")
    ap.add_argument("--name", required=True, help="source name -> labels_<name>_soft.parquet")
    ap.add_argument("--manifest", default=os.path.join(EXPORT_DIR, "manifest.json"))
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    expected = {u for uids in manifest["chunks"].values() for u in uids}

    recs, problems = [], []
    files = sorted(glob.glob(os.path.join(args.chunk_dir, "out_*.json")))
    for p in files:
        r, pr = load_chunk(p)
        recs += r
        problems += pr

    by_uid = {r["StudyInstanceUID"]: r for r in recs}   # last write wins
    got, missing = set(by_uid), expected - set(by_uid)
    extra = set(by_uid) - expected

    print(f"files: {len(files)}   usable records: {len(recs)}   unique: {len(got)}")
    print(f"expected {len(expected)}  ->  have {len(got)}  missing {len(missing)}  "
          f"unexpected {len(extra)}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for x in problems[:20]:
            print(f"  {x}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")

    if missing:
        # Name the chunks the gaps fall in, so only those get re-run.
        per_chunk = {}
        for cid, uids in manifest["chunks"].items():
            n = len(missing & set(uids))
            if n:
                per_chunk[cid] = n
        print(f"\nmissing studies by chunk (re-run only these):")
        for cid, n in sorted(per_chunk.items()):
            print(f"  paste_{cid}.txt: {n} missing")
        with open(os.path.join(args.chunk_dir, "_missing_uids.json"), "w") as f:
            json.dump(sorted(missing), f, indent=1)
        print(f"  full list -> {args.chunk_dir}/_missing_uids.json")

    if extra:
        print(f"\nWARNING {len(extra)} uid(s) not in the manifest (hallucinated or "
              f"from another export); they are dropped.")
        for u in list(extra)[:5]:
            print(f"  {u}")
        by_uid = {u: r for u, r in by_uid.items() if u in expected}

    if not by_uid:
        raise SystemExit("nothing usable to write")

    df = pd.DataFrame(list(by_uid.values()))
    out = os.path.join(LABELS_DIR, f"labels_{args.name}_soft.parquet")
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}: {df.shape[0]} rows x {df.shape[1]} cols "
          f"({df.shape[0] / len(expected) * 100:.1f}% coverage)")
    print(f"score it with:  python src/score_label_source.py {out} --name {args.name}")


if __name__ == "__main__":
    main()
