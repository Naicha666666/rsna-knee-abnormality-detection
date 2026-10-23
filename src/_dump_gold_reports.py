import json
import os
import pandas as pd
from config import LABELS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(HERE, "data", "raw", "train.csv"))
gold = tr[tr[LABELS].notna().all(axis=1)]
recs = [{"uid": str(u), "report": "" if pd.isna(r) else str(r)}
        for u, r in zip(gold["StudyInstanceUID"], gold["Report"])]
out = os.path.join(HERE, "data", "labels", "_gold58_reports.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(recs, f, ensure_ascii=False, indent=1)
print(f"gold {len(recs)} chars {sum(len(x['report']) for x in recs)} -> {out}")
