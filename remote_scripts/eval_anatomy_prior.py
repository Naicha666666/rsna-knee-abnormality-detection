"""Compute Fixed Anatomy Prior Baseline for B11c-2a.

Evaluates how well a static anatomical prior (predicting the population mode/median slice)
performs compared to the C14 Probe on the locked 250-study manifest.
"""
import json
from pathlib import Path
import numpy as np
import torch

MANIFEST_PATH = Path("/home/eason/rsna-knee/scratchpad/b11c2_manifest_250.json")
TEACHER_BUNDLE_PATH = Path("/home/eason/rsna-knee/data/labels/purified_teacher_1200.pt")
PROBE_RESULTS_PATH = Path("/home/eason/rsna-knee/scratchpad/b11c2a_probe_results.json")

manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
uids_250 = manifest["uids"]
tier1_labels = manifest["tier1_labels"]

bundle = torch.load(TEACHER_BUNDLE_PATH, map_location="cpu")
labels = bundle["labels"]
label_to_idx = {l: i for i, l in enumerate(labels)}
registered_Q = bundle["registered_Q"]
reg_map = bundle["registration_map"]

probe_results = {r["label"]: r for r in json.load(open(PROBE_RESULTS_PATH, encoding="utf-8"))}

TARGET_CANDIDATES = {
    "ACL": reg_map["sagittal"],
    "MCL": reg_map["coronal"],
    "Medial OA": reg_map["coronal"],
    "Lateral Meniscus": reg_map["sagittal"],
    "Contusion": reg_map["coronal"]
}

full_comparison_table = []

for lab in tier1_labels:
    l_idx = label_to_idx[lab]
    cand_slices = TARGET_CANDIDATES[lab]
    n_cands = len(cand_slices)
    random_hit1 = (1.0 / n_cands) * 100
    random_hit3 = (3.0 / n_cands) * 100

    peak_slices = []
    for uid in uids_250:
        q_vec = np.array([registered_Q[uid][l_idx, c].item() for c in cand_slices])
        if np.max(q_vec) > 0:
            peak_slices.append(int(np.argmax(q_vec)))

    n_teacher = len(peak_slices)
    if n_teacher == 0:
        continue

    # Fixed Anatomy Prior: predict the overall mode slice across the population
    counts = np.bincount(peak_slices, minlength=n_cands)
    mode_top1 = int(np.argmax(counts))
    mode_top3 = list(np.argsort(counts)[::-1][:3])

    anatomy_hit1 = (np.sum(np.array(peak_slices) == mode_top1) / n_teacher) * 100
    anatomy_hit3 = (np.sum([p in mode_top3 for p in peak_slices]) / n_teacher) * 100

    pr = probe_results[lab]

    full_comparison_table.append({
        "label": lab,
        "n_teacher": n_teacher,
        "random_hit1": random_hit1,
        "anatomy_hit1": anatomy_hit1,
        "probe_hit1": pr["probe_hit1"],
        "delta_hit1": pr["probe_hit1"] - anatomy_hit1,
        "anatomy_hit3": anatomy_hit3,
        "probe_hit3": pr["probe_hit3"],
        "delta_hit3": pr["probe_hit3"] - anatomy_hit3,
        "spearman": pr["spearman"],
        "ndcg3": pr["ndcg3"]
    })

print("=" * 110)
print("B11c-2a EXTENDED BENCHMARK: RANDOM vs FIXED ANATOMY PRIOR vs FROZEN C14 PROBE")
print("=" * 110)
header = f"| {'Label':<17} | {'N':>3} | {'Rand H@1':>8} | {'Anat H@1':>8} | {'Probe H@1':>9} | {'Δ(Probe-Anat)':>13} | {'Anat H@3':>8} | {'Probe H@3':>9} | {'Spearman':>8} |"
sep = f"|{'-'*19}|{'-'*5}|{'-'*10}|{'-'*10}|{'-'*11}|{'-'*15}|{'-'*10}|{'-'*11}|{'-'*10}|"
print(header)
print(sep)
for r in full_comparison_table:
    print(f"| {r['label']:<17} | {r['n_teacher']:3d} | {r['random_hit1']:7.1f}% | {r['anatomy_hit1']:7.1f}% | {r['probe_hit1']:8.1f}% | {r['delta_hit1']:+12.1f}% | {r['anatomy_hit3']:7.1f}% | {r['probe_hit3']:8.1f}% | {r['spearman']:8.3f} |")
print("=" * 110)

out_file = Path("/home/eason/rsna-knee/scratchpad/b11c2a_extended_results.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(full_comparison_table, f, indent=2)
