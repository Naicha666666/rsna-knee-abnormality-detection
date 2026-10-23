import json
from pathlib import Path

p = Path(__file__).with_name("rsna-knee-data-structure-eda-baseline.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
out = Path(__file__).with_name("_extracted")
for i in [30, 31]:
    src = "".join(nb["cells"][i].get("source", []))
    fp = out / f"cell_{i:02d}.py"
    fp.write_text(src, encoding="utf-8")
    print("wrote", fp.name, len(src))
