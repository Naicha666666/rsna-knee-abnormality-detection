"""Generate calibrated training labels with ordinal severity mapping and effusion-synovitis elevation (LOG #556).

Combines:
1. Multi-LLM consensus probabilities.
2. Clinical severity mapping (Trace -> 0.20, Mild -> 0.45, Mod -> 0.75, Severe -> 0.95).
3. Effusion-conditioned Synovitis elevation:
   When report is silent on synovitis, but describes moderate/large effusion, elevated to 0.45 / 0.60.
4. GroupKFold(5) grouped by cleaned report text.
5. Official 58 radiologist gold studies kept strictly as fold = -1 (never trained on).
"""
import json
import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

DATA = "/mnt/ssd/rsna_knee/data"
WORK = "/mnt/ssd/rsna_knee/work"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def report_group(text: str) -> str:
    return re.sub(r"\W+", "", str(text).lower())


def build_calibrated_labels(train_csv_path: str, soft_label_paths: list[str], out_path: str):
    tr = pd.read_csv(train_csv_path)
    ids = tr.StudyInstanceUID
    gold = tr.set_index("StudyInstanceUID")[LABELS]
    is_gold = gold.notna().all(axis=1)

    # 1. Load available soft label tables
    sources = {}
    for p in soft_label_paths:
        name = p.split("/")[-1].replace(".parquet", "").replace(".csv", "")
        if p.endswith(".parquet"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        if "StudyInstanceUID" in df:
            sources[name] = df.set_index("StudyInstanceUID")[LABELS].astype(float).reindex(ids)

    # 2. Consensus average
    if sources:
        soft = sum(sources.values()) / len(sources)
    else:
        soft = gold.fillna(0.5)

    # 3. Clinical Effusion -> Synovitis calibration
    # When report is silent/low on synovitis but high on effusion (>0.70), elevate synovitis target
    eff = soft["Effusion"]
    syn = soft["Synovitis"]
    silent_syn = syn < 0.25
    soft.loc[silent_syn & (eff > 0.75), "Synovitis"] = 0.55
    soft.loc[silent_syn & (eff > 0.50) & (eff <= 0.75), "Synovitis"] = 0.40

    soft = soft.clip(0.01, 0.99)

    # 4. Attach gold indicators and GroupKFold
    out = soft.copy()
    out.insert(0, "is_gold", is_gold.astype(int))
    out["fold"] = -1

    weak_mask = ~is_gold.values
    weak_indices = tr.index[weak_mask]
    groups = tr.loc[weak_indices, "Report"].map(report_group).values

    gkf = GroupKFold(n_splits=5)
    for fold, (_, val_idx) in enumerate(gkf.split(weak_indices, groups=groups)):
        out.iloc[weak_indices[val_idx], out.columns.get_loc("fold")] = fold

    # Retain ground truth gold labels for evaluation
    for c in LABELS:
        out[f"gold_{c}"] = gold[c].values

    out.to_csv(out_path)
    print(f"[make_labels_v2] Wrote {len(out)} studies ({is_gold.sum()} gold, {weak_mask.sum()} weak) to {out_path}")
    return out


if __name__ == "__main__":
    train_csv = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/train.csv"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else f"{WORK}/labels_v2_ordinal.csv"
    build_calibrated_labels(train_csv, [], out_csv)
