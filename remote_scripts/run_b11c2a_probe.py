"""B11c-2a: Frozen Representation Probe Benchmark.

Goal:
Measure whether frozen C14 backbone (ConvNeXt V2-Base 384) features F in R^1024
already encode MSK spatial lesion relevance Q_teacher without any retraining.

Methodology:
1. Fixed 250-study manifest (scratchpad/b11c2_manifest_250.json, SHA256 locked).
2. Frozen C14 model: /home/eason/rsna-knee/experiments/local_c14_64_k12_ep16_convnextv2_base_384_v6/corpus0_topk_avg.pt.
3. Feature extraction for all candidate windows (36 slices per study: 12 coronal, 12 sagittal, 12 axial).
4. 5-Fold Cross-Validation Linear/Ridge probe for each Tier-1 label:
   Tier-1 Targets: ACL, MCL, Medial OA, Lateral Meniscus, Contusion.
5. Descriptive Benchmark Output Table:
   | Label | N teacher | Random Hit@1 | Probe Hit@1 | Probe Hit@3 | Spearman | NDCG@3 |
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/eason/rsna-knee/src')
sys.path.insert(0, '/home/eason/rsna-knee/pylib-cu')
try:
    import tv_stub
    tv_stub.install()
except ImportError:
    pass

from raptor import build_raptor_backbone, build_head

MANIFEST_PATH = Path("scratchpad/b11c2_manifest_250.json")
if not MANIFEST_PATH.exists():
    MANIFEST_PATH = Path("/home/eason/rsna-knee/scratchpad/b11c2_manifest_250.json")

TEACHER_BUNDLE_PATH = Path("data/labels/purified_teacher_1200.pt")
if not TEACHER_BUNDLE_PATH.exists():
    TEACHER_BUNDLE_PATH = Path("/home/eason/rsna-knee/data/labels/purified_teacher_1200.pt")

CORPUS_DIR = Path("/home/eason/rsna-knee/data/corpus64")
CKPT_PATH = Path("/home/eason/rsna-knee/experiments/local_c14_64_k12_ep16_convnextv2_base_384_v6/corpus0_topk_avg.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 1. Load Manifest & Teacher
manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
uids_250 = manifest["uids"]
tier1_labels = manifest["tier1_labels"]
print(f"Loaded manifest with {len(uids_250)} studies. SHA256={manifest['sha256'][:16]}...")

bundle = torch.load(TEACHER_BUNDLE_PATH, map_location="cpu")
labels = bundle["labels"]
label_to_idx = {l: i for i, l in enumerate(labels)}
registered_Q = bundle["registered_Q"]       # uid -> (12, 64)
registered_mask = bundle["registered_mask"] # uid -> (12, 64)
reg_map = bundle["registration_map"]

cor_slices = reg_map["coronal"]   # [32..43]
sag_slices = reg_map["sagittal"]  # [1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16]
ax_slices = reg_map["axial"]      # [52..63]

# Candidate slices per Tier-1 target based on clinical primary plane
TARGET_CANDIDATES = {
    "ACL": sag_slices,               # Sagittal
    "MCL": cor_slices,               # Coronal
    "Medial OA": cor_slices,         # Coronal
    "Lateral Meniscus": sag_slices,  # Sagittal
    "Contusion": cor_slices          # Coronal (marrow edema)
}

# 2. Build C14 Backbone
print("\nLoading C14 ConvNeXt V2-Base 384 model...")
t0 = time.time()
ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
arch = ck.get("arch")
bb = build_raptor_backbone(arch)
model = build_head(bb, ck.get("head") or "mil")
model.load_state_dict({k: v for k, v in ck["state_dict"].items()})
model.to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"Model loaded and frozen in {time.time() - t0:.2f}s.")

# 3. Load Corpus Memory Maps
print("\nLoading corpus64 memory maps...")
t0 = time.time()
ids_p0 = np.load(CORPUS_DIR / "ids64_p0.npy", allow_pickle=True)
ids_p1 = np.load(CORPUS_DIR / "ids64_p1.npy", allow_pickle=True)
vols_p0 = np.load(CORPUS_DIR / "vols64_p0.npy", mmap_mode="r")
vols_p1 = np.load(CORPUS_DIR / "vols64_p1.npy", mmap_mode="r")

id_to_loc = {}
for i, uid in enumerate(ids_p0):
    id_to_loc[str(uid)] = (0, i)
for i, uid in enumerate(ids_p1):
    id_to_loc[str(uid)] = (1, i)
print(f"Corpus index ready in {time.time() - t0:.2f}s.")

# 4. Extract Features for Candidate Slices
# Candidate slices needed: set(cor_slices + sag_slices)
needed_slices = sorted(list(set(cor_slices + sag_slices + ax_slices)))
print(f"\nExtracting features for {len(needed_slices)} windows across {len(uids_250)} studies ({len(needed_slices)*len(uids_250)} total forward passes)...")

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

study_features = {} # uid -> dict(c -> 1024-d np array)
t_ext = time.time()

batch_triplets = []
batch_meta = [] # (uid, c)

def flush_batch():
    if not batch_triplets:
        return
    inp = torch.cat(batch_triplets, dim=0).to(DEVICE)
    with torch.no_grad():
        feats = model.norm(model.backbone(inp)) # (B, 1024)
        feats_np = feats.cpu().numpy()
    for (u, c), f in zip(batch_meta, feats_np):
        study_features[u][c] = f
    batch_triplets.clear()
    batch_meta.clear()

# Feature caching for instant reuse
FEATURE_CACHE_PATH = Path("/home/eason/rsna-knee/scratchpad/b11c2a_features_250.pt")
FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

if FEATURE_CACHE_PATH.exists():
    print(f"Loading cached features from {FEATURE_CACHE_PATH}...")
    study_features = torch.load(FEATURE_CACHE_PATH, map_location="cpu")
    print(f"Loaded features for {len(study_features)} studies.")
else:
    for idx, uid in enumerate(uids_250):
        study_features[uid] = {}
        part, loc = id_to_loc[uid]
        vol = vols_p0[loc] if part == 0 else vols_p1[loc]
        
        for c in needed_slices:
            lo = max(0, c - 1)
            hi = min(63, c + 1)
            triplet_arr = np.stack([vol[lo], vol[c], vol[hi]], axis=0).astype(np.float32) / 255.0
            triplet_t = torch.from_numpy(triplet_arr).unsqueeze(0)
            triplet_t = F.interpolate(triplet_t, size=(384, 384), mode="bilinear", align_corners=False)
            triplet_t = (triplet_t.to(DEVICE) - MEAN) / STD
            
            batch_triplets.append(triplet_t)
            batch_meta.append((uid, c))
            
            if len(batch_triplets) >= 64:
                flush_batch()

        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(uids_250)}] Extracted studies ({time.time() - t_ext:.1f}s)...")

    flush_batch()
    print(f"All features extracted in {time.time() - t_ext:.2f}s! Saving to cache...")
    torch.save(study_features, FEATURE_CACHE_PATH)

# 5. Descriptive Probe Benchmark Evaluation
print("\n" + "=" * 80)
print("B11c-2a: FROZEN REPRESENTATION PROBE BENCHMARK RESULTS")
print("=" * 80)

def compute_ndcg_at_k(pred_scores, true_relevance, k=3):
    ideal_order = np.argsort(true_relevance)[::-1]
    pred_order = np.argsort(pred_scores)[::-1]
    
    dcg = 0.0
    idcg = 0.0
    for i in range(min(k, len(true_relevance))):
        dcg += (2**true_relevance[pred_order[i]] - 1) / np.log2(i + 2)
        idcg += (2**true_relevance[ideal_order[i]] - 1) / np.log2(i + 2)
    return dcg / (idcg + 1e-8) if idcg > 0 else 0.0

results_table = []

for lab in tier1_labels:
    l_idx = label_to_idx[lab]
    cand_slices = TARGET_CANDIDATES[lab]
    n_cands = len(cand_slices)
    random_hit1 = 1.0 / n_cands

    # Filter studies where this label is positive with active teacher Q
    eligible_uids = []
    study_X = [] # List of (12, 1024)
    study_Y = [] # List of (12,)
    peak_slices = []

    for uid in uids_250:
        q_vec = np.array([registered_Q[uid][l_idx, c].item() for c in cand_slices])
        if np.max(q_vec) > 0:
            eligible_uids.append(uid)
            study_X.append(np.stack([study_features[uid][c] for c in cand_slices], axis=0))
            study_Y.append(q_vec)
            peak_slices.append(int(np.argmax(q_vec)))

    n_teacher = len(eligible_uids)
    print(f"Target '{lab}': Found {n_teacher} studies with active Q supervision.")
    if n_teacher < 5:
        continue

    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((n_teacher, n_cands))

    for train_idx, val_idx in kf.split(eligible_uids):
        # Flatten training pairs
        X_train = np.concatenate([study_X[i] for i in train_idx], axis=0) # (N_train*12, 1024)
        Y_train = np.concatenate([study_Y[i] for i in train_idx], axis=0) # (N_train*12,)
        
        clf = Ridge(alpha=100.0)
        clf.fit(X_train, Y_train)
        
        for i in val_idx:
            oof_preds[i] = clf.predict(study_X[i])

    # Compute Metrics across all out-of-fold studies
    hit1_cnt = 0
    hit3_cnt = 0
    spearman_list = []
    ndcg3_list = []

    for i in range(n_teacher):
        pred_s = oof_preds[i]
        true_q = study_Y[i]
        true_peak = peak_slices[i]
        
        pred_peak = int(np.argmax(pred_s))
        pred_top3 = list(np.argsort(pred_s)[::-1][:3])
        
        if pred_peak == true_peak:
            hit1_cnt += 1
        if true_peak in pred_top3:
            hit3_cnt += 1
            
        rho, _ = spearmanr(pred_s, true_q)
        if not np.isnan(rho):
            spearman_list.append(rho)
            
        ndcg3_list.append(compute_ndcg_at_k(pred_s, true_q, k=3))

    probe_hit1 = (hit1_cnt / n_teacher) * 100
    probe_hit3 = (hit3_cnt / n_teacher) * 100
    mean_spearman = np.mean(spearman_list) if spearman_list else 0.0
    mean_ndcg3 = np.mean(ndcg3_list)

    results_table.append({
        "label": lab,
        "n_teacher": n_teacher,
        "random_hit1": random_hit1 * 100,
        "probe_hit1": probe_hit1,
        "probe_hit3": probe_hit3,
        "spearman": mean_spearman,
        "ndcg3": mean_ndcg3
    })

# Print Table
header = f"| {'Label':<18} | {'N teacher':>9} | {'Random Hit@1':>12} | {'Probe Hit@1':>11} | {'Probe Hit@3':>11} | {'Spearman':>8} | {'NDCG@3':>6} |"
sep = f"|{'-'*20}|{'-'*11}|{'-'*14}|{'-'*13}|{'-'*13}|{'-'*10}|{'-'*8}|"
print(header)
print(sep)
for r in results_table:
    print(f"| {r['label']:<18} | {r['n_teacher']:9d} | {r['random_hit1']:11.1f}% | {r['probe_hit1']:10.1f}% | {r['probe_hit3']:10.1f}% | {r['spearman']:8.3f} | {r['ndcg3']:6.3f} |")
print("=" * 80)

# Save JSON results
out_json = Path("scratchpad/b11c2a_probe_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results_table, f, indent=2)
print(f"Saved benchmark results to {out_json}")
