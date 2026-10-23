"""Build Immutable 250-Study Manifest for B11c-2 Spatial Distillation.

Strict Isolation Rules:
1. 58 Gold cases (_gold58_reports.json) are PERMANENTLY EXCLUDED (Firewall).
2. Must belong to the 1,200 Purified Teacher studies.
3. Must contain supervised spatial evidence in at least one of the 5 Tier-1 targets:
   Tier-1 = ['ACL', 'MCL', 'Medial OA', 'Lateral Meniscus', 'Contusion']
4. Fixed random seed (seed=42) for 100% deterministic reproducibility.
5. Emits SHA-256 manifest hash to freeze the split.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

GOLD_PATH = Path("data/labels/_gold58_reports.json")
TEACHER_BUNDLE_PATH = Path("data/labels/purified_teacher_1200.pt")
OUT_MANIFEST = Path("scratchpad/b11c2_manifest_250.json")

# On remote, fallback to /home/eason/rsna-knee/
if not GOLD_PATH.exists():
    GOLD_PATH = Path("/home/eason/rsna-knee/gold58_reports.json")
if not TEACHER_BUNDLE_PATH.exists():
    TEACHER_BUNDLE_PATH = Path("/home/eason/rsna-knee/data/labels/purified_teacher_1200.pt")

import torch

print(f"Loading Gold reports from: {GOLD_PATH}")
gold_data = json.load(open(GOLD_PATH, encoding="utf-8"))
gold_uids = set(c["uid"] for c in gold_data)
print(f"Total Gold held-out cases: {len(gold_uids)}")

print(f"Loading Purified Teacher bundle from: {TEACHER_BUNDLE_PATH}")
bundle = torch.load(TEACHER_BUNDLE_PATH, map_location="cpu")
teacher_uids = bundle["uids"]
labels = bundle["labels"]
tier1_labels = bundle["metadata"]["tier1_labels"]
print(f"Total teacher studies: {len(teacher_uids)}")
print(f"Tier-1 Targets: {tier1_labels}")

label_to_idx = {l: i for i, l in enumerate(labels)}
tier1_indices = [label_to_idx[l] for l in tier1_labels]

registered_mask = bundle["registered_mask"] # uid -> (12, 64)

# Filter eligible candidates: not in gold, and has at least one tier-1 supervision
candidates = []
for uid in teacher_uids:
    if uid in gold_uids:
        continue
    mask = registered_mask[uid] # (12, 64)
    # Check if any tier1 label has active supervision
    tier1_active = [l for l in tier1_labels if mask[label_to_idx[l]].sum().item() > 0]
    if len(tier1_active) > 0:
        candidates.append((uid, tier1_active))

print(f"Total eligible Tier-1 studies outside Gold: {len(candidates)}")

# Deterministic selection with seed=42
rng = np.random.RandomState(42)
indices = rng.permutation(len(candidates))

# Prioritize multi-target coverage while sampling 250 studies
selected = [candidates[i] for i in indices[:250]]
selected_uids = [item[0] for item in selected]

assert len(selected_uids) == 250
assert len(set(selected_uids) & gold_uids) == 0, "FATAL: Overlap with Gold detected!"

# Label distribution statistics
counts = {l: 0 for l in tier1_labels}
for _, active_list in selected:
    for l in active_list:
        counts[l] += 1

print("\n=== Manifest 250-Study Tier-1 Representation ===")
for l in tier1_labels:
    print(f"  {l:<20}: {counts[l]:3d} studies ({counts[l]/250*100:5.1f}%)")

# Calculate SHA-256 checksum of the sorted UID list
uid_str = "\n".join(sorted(selected_uids))
manifest_hash = hashlib.sha256(uid_str.encode('utf-8')).hexdigest()
print(f"\nManifest SHA-256 Hash: {manifest_hash}")

manifest_payload = {
    "manifest_version": "b11c2_dev_250_v1",
    "sha256": manifest_hash,
    "seed": 42,
    "study_count": 250,
    "tier1_labels": tier1_labels,
    "label_representation": counts,
    "gold_overlap_check": "PASS (0 overlap with 58 Gold)",
    "uids": selected_uids
}

OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
    json.dump(manifest_payload, f, indent=2)

print(f"Successfully saved manifest to {OUT_MANIFEST}")
