"""Extract readable digest (markdown + key code) from reference notebooks."""
import json, glob, os, re

REF = os.path.dirname(os.path.abspath(__file__))
KEYWORDS = ["class ", "def ", "Dataset", "Model", "train", "fit(" "auc", "metric",
            "pseudo", "report", "SeriesInstanceUID", "dicom", "sitk", "Image",
            "fold", "concat", "emb", "head", "sigmoid", "ResNet", "DINO",
            "backbone", "preprocess", "window", "normalize", "slice", "series",
            "study", "ensemble", "pipeline", "tokeniz", "bert", "clip"]

def is_key(code):
    return any(k.lower() in code.lower() for k in KEYWORDS)

def extract(path):
    nb = json.load(open(path, encoding="utf-8"))
    slug = os.path.basename(path).replace(".ipynb", "")
    out = [f"\n{'='*78}\nNOTEBOOK: {slug}\n{'='*78}"]
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        ctype = cell.get("cell_type")
        if ctype == "markdown":
            out.append(f"\n--- [markdown] ---\n{src}")
        elif ctype == "code" and is_key(src):
            s = src if len(src) <= 1600 else src[:1600] + "\n...[truncated]"
            out.append(f"\n--- [code #{i}] ---\n{s}")
    return "\n".join(out)

files = sorted(glob.glob(os.path.join(REF, "*.ipynb")))
digest = []
for f in files:
    digest.append(extract(f))
text = "\n".join(digest)
outp = os.path.join(REF, "_digest.txt")
open(outp, "w", encoding="utf-8").write(text)
print("wrote", outp, "chars=", len(text))
print("notebooks:", [os.path.basename(f) for f in files])
