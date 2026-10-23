"""Skim pulled notebooks: metadata, cell map, unique signals vs known 0.936 lineage."""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
KNOWN = [
    "dinov2", "dinov3", "raptor", "coatnet", "radimagenet", "slothead",
    "rank", "blend", "ensemble", "meniscus", "overlay", "calibrat",
    "pos_weight", "swa", "memmap", "k_eval", "crop_mm", "soft_label",
    "llm", "weak-label", "gated", "crossfit", "fingerprint",
]

def cell_src(c):
    s = c.get("source", "")
    if isinstance(s, list):
        return "".join(s)
    return str(s)

def first_md(nb):
    for c in nb.get("cells", []):
        if c.get("cell_type") == "markdown":
            t = cell_src(c).strip()
            if t:
                return t[:800]
    return ""

def hits(text):
    low = text.lower()
    return [k for k in KNOWN if k in low]

rows = []
for meta_path in sorted(ROOT.glob("*/kernel-metadata.json")):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    nbs = list(meta_path.parent.glob("*.ipynb"))
    nb = json.loads(nbs[0].read_text(encoding="utf-8")) if nbs else {"cells": []}
    cells = nb.get("cells", [])
    full = "\n".join(cell_src(c) for c in cells)
    print("=" * 78)
    print(f"ID: {meta.get('id')}")
    print(f"title: {meta.get('title')}")
    print(f"gpu={meta.get('enable_gpu')} inet={meta.get('enable_internet')} votes=?")
    print(f"datasets: {meta.get('dataset_sources')}")
    print(f"kernels: {meta.get('kernel_sources')}")
    print(f"models: {meta.get('model_sources')}")
    print(f"cells={len(cells)} chars={len(full)} keywords={hits(full)}")
    print("--- first markdown ---")
    print(first_md(nb)[:600])
    print("--- cell map ---")
    for i, c in enumerate(cells):
        src = cell_src(c)
        preview = " | ".join(src.strip().splitlines()[:2])[:140]
        print(f"  [{i:02d}] {c.get('cell_type')[:4]:4s} L={src.count(chr(10))+1:4d} {preview}")
    print()
