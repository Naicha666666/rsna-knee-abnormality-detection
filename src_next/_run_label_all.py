"""Label all 4407 reports with the local Qwen3.8-27B. ~10h, resumable."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import ollama_labeler as O

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "data", "labels", "labels_qwen_soft.parquet")
tr = pd.read_csv(os.path.join(HERE, "data", "raw", "train.csv"))
reports = {str(u): r for u, r in
           zip(tr["StudyInstanceUID"], tr["Report"].fillna(""))}
print(f"labelling {len(reports)} reports -> {OUT}", flush=True)
O.label(reports, OUT, progress_every=100, checkpoint_every=50)
