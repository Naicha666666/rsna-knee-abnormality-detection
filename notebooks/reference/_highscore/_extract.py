from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "_extract"
OUT.mkdir(exist_ok=True)

WANTED = {
    "rsna-knee-0-937-weak-label-dinov2-meniscus-resid": [0, 6, 8],
    "the-bee-s-knees-final-rsna-push": [0, 8, 11, 13, 14, 15, 16],
    "rsna-knee-v6-outer-weight-overlay": [9],
    "rsna-knee-infer-gpu-hsv44fixc-calw": [0, 1, 6],
    "rsna-knee-e0-v16-crossfit-gated": [1, 6],
    "rsna-knee-dinov2-multibranch-gm": [0, 2],
    "rsna-knee-full-4-arm-ensemble-v55": [0],
    "rsna-knee-abnormality-detectionv1": [3],
}

def src(c):
    s = c.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)

for folder, idxs in WANTED.items():
    nbs = list((ROOT / folder).glob("*.ipynb"))
    if not nbs:
        print("missing", folder)
        continue
    nb = json.loads(nbs[0].read_text(encoding="utf-8"))
    dest = OUT / folder
    dest.mkdir(exist_ok=True)
    for i in idxs:
        c = nb["cells"][i]
        text = src(c)
        (dest / f"cell_{i:02d}.txt").write_text(text, encoding="utf-8")
        print(f"{folder} [{i}] {c['cell_type']} {len(text)} chars")
print("done")
