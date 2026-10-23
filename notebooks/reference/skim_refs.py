"""Skim digest: full markdown + code cell signatures only."""
import json, glob, os

REF = os.path.dirname(os.path.abspath(__file__))

def cell_sig(src):
    lines = [l for l in src.splitlines() if l.strip()]
    head = "\n".join(lines[:3])
    return head[:240]

def extract(path):
    nb = json.load(open(path, encoding="utf-8"))
    slug = os.path.basename(path).replace(".ipynb", "")
    out = [f"\n{'='*78}\nNOTEBOOK: {slug}\n{'='*78}"]
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        ctype = cell.get("cell_type")
        if ctype == "markdown":
            out.append(f"\n--- [md] ---\n{src}")
        elif ctype == "code":
            out.append(f"\n--- [code #{i}] ---\n{cell_sig(src)}")
    return "\n".join(out)

files = sorted(glob.glob(os.path.join(REF, "*.ipynb")))
text = "\n".join(extract(f) for f in files)
outp = os.path.join(REF, "_skim.txt")
open(outp, "w", encoding="utf-8").write(text)
print("wrote", outp, "chars=", len(text))
