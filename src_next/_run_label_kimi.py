"""Label all 4407 reports with Moonshot kimi-k2.6. ~1.6h, resumable."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import moonshot_labeler as M

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "labels", "labels_kimi_soft.parquet")
tr = pd.read_csv(os.path.join(HERE, "data", "raw", "train.csv"))
reports = {str(u): r for u, r in zip(tr["StudyInstanceUID"], tr["Report"].fillna(""))}
print(f"labelling {len(reports)} reports -> {OUT}", flush=True)
M.label(reports, OUT, workers=4, progress_every=200, checkpoint_every=100)
