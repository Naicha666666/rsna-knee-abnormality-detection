import json
from pathlib import Path

p = Path(__file__).with_name("bend-the-knee-to-the-dinosaurs.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
print("nbformat", nb.get("nbformat"), "cells", len(nb.get("cells", [])))
print("metadata keys", list(nb.get("metadata", {}).keys()))
meta = nb.get("metadata", {})
print(json.dumps({k: meta[k] for k in meta if k in ("kaggle", "kernelspec", "language_info", "papermill")}, indent=2, default=str)[:4000])
print("----CELLS----")
out_dir = Path(__file__).with_name("_cells")
out_dir.mkdir(exist_ok=True)
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
    first = src.strip().splitlines()[:3]
    preview = " | ".join(first)[:180]
    nout = len(c.get("outputs", []) or [])
    print(f"[{i:02d}] {c['cell_type']:8s} lines={src.count(chr(10))+1:4d} outs={nout:2d}  {preview}")
    (out_dir / f"cell_{i:02d}_{c['cell_type']}.txt").write_text(src, encoding="utf-8")
    # dump text outputs
    texts = []
    for o in c.get("outputs", []) or []:
        if o.get("output_type") == "stream":
            texts.append("".join(o.get("text", [])))
        elif o.get("output_type") in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "text/plain" in data:
                t = data["text/plain"]
                texts.append("".join(t) if isinstance(t, list) else str(t))
        elif o.get("output_type") == "error":
            texts.append("ERROR: " + "\n".join(o.get("traceback", [])[-8:]))
    if texts:
        (out_dir / f"cell_{i:02d}_out.txt").write_text("\n---\n".join(texts), encoding="utf-8")
