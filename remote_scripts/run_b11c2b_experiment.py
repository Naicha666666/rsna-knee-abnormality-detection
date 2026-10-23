"""B11c-2b: Spatially Guided Representation Adaptation (Three-Arm Intervention Experiment).

Core Scientific Question:
When C14 already encodes baseline spatial features, does case-specific spatial supervision
reshape the representation (Stage 4) and provide diagnosis-level improvements over fixed anatomy?

Strict Experimental Controls:
- Primary Endpoint: Δ(C - B) (Gold-58 Diagnosis Macro AUC of Arm C vs Arm B)
- Secondary Endpoint: Δ(C - A) (Gold-58 Diagnosis Macro AUC of Arm C vs Arm A)
- Exact Parity:
  * Same 250 manifest studies (SHA256: 1bbdfebbd55909cb)
  * Same frozen/trainable architecture: Stages 0..2 frozen, Stage 4 (stage 3) + Heads trainable
  * Same optimizer (AdamW, lr_bb=2e-5, lr_head=1e-4), same batch size, same epochs (6)
  * Same diagnosis loss (BCE with soft labels)
  * Same lambda = 0.15 for Arms B and C
  * Only variable swapped: Q_anatomy vs Q_VLM
  * 58 Gold permanently held-out for validation only
"""
import copy
import json
import os
os.environ["RSNA_CORPUS_SLOTS"] = "64"
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
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

import config
from corpus import open_corpus, centers_from_mask, read_windows
from dataset import RSNADataset
from raptor import build_raptor_backbone, build_head, RaptorClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using execution device: {DEVICE}")

# 1. Load Data, Manifest, and Teachers
MANIFEST_PATH = Path("/home/eason/rsna-knee/scratchpad/b11c2_manifest_250.json")
TEACHER_BUNDLE_PATH = Path("/home/eason/rsna-knee/data/labels/purified_teacher_1200.pt")
CKPT_PATH = Path("/home/eason/rsna-knee/experiments/local_c14_64_k12_ep16_convnextv2_base_384_v6/corpus0_topk_avg.pt")

manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
train_uids = manifest["uids"]
tier1_labels = manifest["tier1_labels"]
print(f"Loaded train manifest: {len(train_uids)} studies (SHA256={manifest['sha256'][:16]}...)")

bundle = torch.load(TEACHER_BUNDLE_PATH, map_location="cpu")
labels = bundle["labels"]
label_to_idx = {l: i for i, l in enumerate(labels)}
tier1_indices = [label_to_idx[l] for l in tier1_labels]
registered_Q = bundle["registered_Q"]
reg_map = bundle["registration_map"]

TARGET_CANDIDATES = {
    "ACL": reg_map["sagittal"],
    "MCL": reg_map["coronal"],
    "Medial OA": reg_map["coronal"],
    "Lateral Meniscus": reg_map["sagittal"],
    "Contusion": reg_map["coronal"]
}

# Precompute Static Anatomy Priors (population mode distribution across candidates)
Q_anatomy_dict = {}
for lab in tier1_labels:
    cand_slices = TARGET_CANDIDATES[lab]
    l_idx = label_to_idx[lab]
    peak_counts = np.zeros(len(cand_slices), dtype=np.float32)
    for u in train_uids:
        q_vec = np.array([registered_Q[u][l_idx, c].item() for c in cand_slices])
        if np.max(q_vec) > 0:
            peak_counts[int(np.argmax(q_vec))] += 1.0
    # Soft distribution centered at anatomy mode
    if np.sum(peak_counts) > 0:
        dist = peak_counts / np.sum(peak_counts)
    else:
        dist = np.ones(len(cand_slices), dtype=np.float32) / len(cand_slices)
    Q_anatomy_dict[lab] = torch.from_numpy(dist).float()

print("Precomputed Q_anatomy priors for Tier-1 targets.")

# Load Corpus and Gold-58 Data
print("Opening corpus memmap...")
vols, masks, cids = open_corpus()
row_of = {u: i for i, u in enumerate(cids)}

ds = RSNADataset("train")
y_raw = ds.y_raw
has_gold = np.isfinite(y_raw).all(axis=1)
gold_ids = [u for i, u in enumerate(ds.uids) if has_gold[i] and u in row_of]
idx_of = {u: i for i, u in enumerate(ds.uids)}
gold_y = np.stack([y_raw[idx_of[u]] for u in gold_ids]).astype(np.float32)
N_GOLD = len(gold_ids)
print(f"Loaded {N_GOLD} Gold studies for held-out evaluation.")

# Soft Labels for training studies
import pandas as pd
soft_path = Path("/home/eason/rsna-knee/data/labels/labels_u2_flash_v6_soft.parquet")
if not soft_path.exists():
    soft_path = Path("data/labels/labels_u2_flash_v6_soft.parquet")
soft_df = pd.read_parquet(soft_path).set_index("StudyInstanceUID")
train_y_dict = {}
for u in train_uids:
    if u in soft_df.index:
        train_y_dict[u] = soft_df.loc[u, labels].values.astype(np.float32)
    else:
        train_y_dict[u] = y_raw[idx_of[u]].astype(np.float32)

# Model Wrapper with Auxiliary Relevance Head
class ModelWithRelevance(nn.Module):
    def __init__(self, base_model, f_dim=1024, n_targets=5):
        super().__init__()
        self.base = base_model
        # Auxiliary Relevance Head: maps window feature -> 5 Tier-1 relevance logits
        self.relevance_head = nn.Sequential(
            nn.Linear(f_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_targets)
        )

    def forward(self, x, chunk=None):
        # x is [B, K, 3, H, W]
        b, k = x.shape[:2]
        flat = x.flatten(0, 1) # [B*K, 3, H, W]
        step = chunk or flat.shape[0]
        
        # Pass through backbone
        feats_list = []
        for i in range(0, flat.shape[0], step):
            f = self.base.backbone(flat[i:i+step])
            f = self.base.norm(f)
            feats_list.append(f)
        feats = torch.cat(feats_list, dim=0) # [B*K, f_dim]
        
        # MIL Classification head
        logits = self.base.forward_from_features(feats, b, k) # [B, 12]
        
        # Relevance logits: [B, K, 5]
        rel_logits = self.relevance_head(feats).view(b, k, -1)
        
        return logits, rel_logits, feats

def build_experiment_model():
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    arch = ck.get("arch")
    bb = build_raptor_backbone(arch)
    base_model = build_head(bb, ck.get("head") or "mil")
    base_model.load_state_dict({k: v for k, v in ck["state_dict"].items()})
    
    # Freeze stages 0, 1, 2
    for p in base_model.parameters():
        p.requires_grad = False
        
    # Unfreeze Stage 4 (the 4th stage, index 3 in timm ConvNeXt V2)
    for p in base_model.backbone.stages[3].parameters():
        p.requires_grad = True
    for p in base_model.norm.parameters():
        p.requires_grad = True
        
    # Unfreeze MIL Head
    for p in base_model.att.parameters():
        p.requires_grad = True
    base_model.cls_w.requires_grad = True
    base_model.cls_b.requires_grad = True
    
    wrapper = ModelWithRelevance(base_model, f_dim=1024, n_targets=5)
    wrapper.to(DEVICE)
    return wrapper

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

def eval_on_gold(model, k_eval=42):
    model.eval()
    preds = []
    rng = np.random.RandomState(42)
    
    with torch.no_grad():
        for uid in gold_ids:
            row = row_of[uid]
            mask = masks[row]
            cs = centers_from_mask(mask, k_eval, train=False, rng=rng)
            # Read windows (returns torch tensor (K, 3, res, res) in [0, 1])
            wins = read_windows(vols[row], cs, config.RAPTOR_RES)
            wins_t = (wins.to(DEVICE) - MEAN) / STD
            wins_t = wins_t.unsqueeze(0) # (1, K, 3, H, W)
            
            logits, _, _ = model(wins_t, chunk=24)
            prob = torch.sigmoid(logits).cpu().numpy()[0]
            preds.append(prob)
            
    preds_arr = np.stack(preds, axis=0) # (N_GOLD, 12)
    
    # Compute AUCs
    aucs = {}
    for i, l in enumerate(labels):
        try:
            aucs[l] = roc_auc_score(gold_y[:, i], preds_arr[:, i])
        except Exception:
            aucs[l] = 0.5
            
    macro_12 = np.mean([aucs[l] for l in labels])
    macro_tier1 = np.mean([aucs[l] for l in tier1_labels])
    return macro_tier1, macro_12, aucs

def run_arm(arm_name: str, lambda_spatial: float, use_vlm: bool):
    print("\n" + "=" * 80)
    print(f"LAUNCHING {arm_name}: lambda={lambda_spatial}, use_vlm={use_vlm}")
    print("=" * 80)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = build_experiment_model()
    
    # Optimizer
    bb_params = [p for p in model.base.backbone.stages[3].parameters() if p.requires_grad] + [p for p in model.base.norm.parameters() if p.requires_grad]
    head_params = [p for p in model.base.att.parameters()] + [model.base.cls_w, model.base.cls_b] + [p for p in model.relevance_head.parameters()]
    
    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": 2e-5, "weight_decay": 1e-4},
        {"params": head_params, "lr": 1e-4, "weight_decay": 1e-4}
    ])
    
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    epochs = 6
    batch_size = 4
    K_train = 12
    
    best_gold_t1 = 0.0
    best_gold_12 = 0.0
    best_aucs = {}
    
    # Initial Baseline evaluation (Epoch 0)
    e0_t1, e0_12, e0_aucs = eval_on_gold(model)
    print(f"Initial Checkpoint (Epoch 0): Tier-1 AUC={e0_t1:.4f} | All-12 AUC={e0_12:.4f}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.RandomState(42 + epoch)
        perm = rng.permutation(len(train_uids))
        
        epoch_diag_loss = 0.0
        epoch_spatial_loss = 0.0
        n_steps = 0
        
        for b_start in range(0, len(train_uids), batch_size):
            batch_uids = [train_uids[i] for i in perm[b_start : b_start + batch_size]]
            if len(batch_uids) < batch_size:
                continue
                
            batch_wins = []
            batch_y = []
            batch_spatial_targets = [] # List of tuples: (study_idx, target_idx, target_dist)
            
            for b_idx, uid in enumerate(batch_uids):
                row = row_of[uid]
                mask = masks[row]
                # Sample K_train centers
                cs = centers_from_mask(mask, K_train, train=True, rng=rng)
                wins = read_windows(vols[row], cs, config.RAPTOR_RES)
                wins_t = (wins.to(DEVICE) - MEAN) / STD
                batch_wins.append(wins_t)
                batch_y.append(train_y_dict[uid])
                
                # Check spatial teacher supervision
                if lambda_spatial > 0:
                    for t_idx, lab in enumerate(tier1_labels):
                        l_idx = label_to_idx[lab]
                        cand_slices = TARGET_CANDIDATES[lab]
                        
                        # Find which window centers match the candidate slices
                        # cs is list of K_train centers
                        overlap_indices = [k_pos for k_pos, c in enumerate(cs) if c in cand_slices]
                        if len(overlap_indices) >= 3:
                            if use_vlm:
                                q_vec = np.array([registered_Q[uid][l_idx, cs[k_pos]].item() for k_pos in overlap_indices])
                                if np.max(q_vec) > 0:
                                    target_dist = torch.from_numpy(q_vec / np.sum(q_vec)).float().to(DEVICE)
                                    batch_spatial_targets.append((b_idx, t_idx, overlap_indices, target_dist))
                            else:
                                # Arm B: Fixed Anatomy Prior
                                q_vec = np.array([Q_anatomy_dict[lab][cand_slices.index(cs[k_pos])].item() for k_pos in overlap_indices])
                                target_dist = torch.from_numpy(q_vec / (np.sum(q_vec) + 1e-8)).float().to(DEVICE)
                                batch_spatial_targets.append((b_idx, t_idx, overlap_indices, target_dist))
                                
            inp_batch = torch.stack(batch_wins, dim=0) # [B, K, 3, H, W]
            target_y = torch.tensor(np.stack(batch_y, axis=0), device=DEVICE, dtype=torch.float32)
            
            optimizer.zero_grad()
            logits, rel_logits, _ = model(inp_batch, chunk=24)
            
            # Diagnosis Loss
            loss_diag = bce_loss_fn(logits, target_y)
            
            # Spatial Loss
            loss_spatial = torch.tensor(0.0, device=DEVICE)
            if lambda_spatial > 0 and len(batch_spatial_targets) > 0:
                s_losses = []
                for b_idx, t_idx, k_indices, target_dist in batch_spatial_targets:
                    # Predicted relevance logits for this target across the candidate windows
                    pred_logits = rel_logits[b_idx, k_indices, t_idx] # [n_overlap]
                    # Soft cross-entropy
                    log_p = F.log_softmax(pred_logits, dim=-1)
                    s_loss = -(target_dist * log_p).sum()
                    s_losses.append(s_loss)
                if s_losses:
                    loss_spatial = torch.stack(s_losses).mean()
                    
            total_loss = loss_diag + lambda_spatial * loss_spatial
            total_loss.backward()
            optimizer.step()
            
            epoch_diag_loss += loss_diag.item()
            epoch_spatial_loss += loss_spatial.item()
            n_steps += 1
            
        t1_auc, a12_auc, cur_aucs = eval_on_gold(model)
        print(f"Epoch {epoch:2d}/{epochs} | Loss_diag={epoch_diag_loss/n_steps:.4f} | Loss_spatial={epoch_spatial_loss/n_steps:.4f} | Gold Tier-1 AUC={t1_auc:.4f} | All-12={a12_auc:.4f}")
        
        if t1_auc > best_gold_t1:
            best_gold_t1 = t1_auc
            best_gold_12 = a12_auc
            best_aucs = cur_aucs
            
    print(f"Finished {arm_name}. Peak Gold Tier-1 AUC: {best_gold_t1:.4f} (All-12: {best_gold_12:.4f})")
    return {
        "arm": arm_name,
        "peak_tier1_auc": best_gold_t1,
        "peak_all12_auc": best_gold_12,
        "aucs": best_aucs
    }

def main():
    print("=" * 80)
    print("STARTING B11c-2b THREE-ARM INTERVENTION BENCHMARK")
    print("=" * 80)
    t_start = time.time()
    
    # Arm A: Diagnosis Only (lambda = 0.0)
    res_A = run_arm("Arm A (Diagnosis Only)", lambda_spatial=0.0, use_vlm=False)
    
    # Arm B: Diagnosis + Fixed Anatomy Prior (lambda = 0.15)
    res_B = run_arm("Arm B (Diagnosis + Anatomy Prior)", lambda_spatial=0.15, use_vlm=False)
    
    # Arm C: Diagnosis + Purified VLM Spatial Teacher (lambda = 0.15)
    res_C = run_arm("Arm C (Diagnosis + VLM Spatial Teacher)", lambda_spatial=0.15, use_vlm=True)
    
    delta_C_B = res_C["peak_tier1_auc"] - res_B["peak_tier1_auc"]
    delta_C_A = res_C["peak_tier1_auc"] - res_A["peak_tier1_auc"]
    
    print("\n" + "=" * 90)
    print("B11c-2b THREE-ARM INTERVENTION BENCHMARK RESULTS (GOLD-58 HELD-OUT)")
    print("=" * 90)
    print(f"Arm A (Diagnosis Only)           : Tier-1 AUC = {res_A['peak_tier1_auc']:.4f} | All-12 AUC = {res_A['peak_all12_auc']:.4f}")
    print(f"Arm B (Diagnosis + Anatomy Prior): Tier-1 AUC = {res_B['peak_tier1_auc']:.4f} | All-12 AUC = {res_B['peak_all12_auc']:.4f}")
    print(f"Arm C (Diagnosis + VLM Teacher)  : Tier-1 AUC = {res_C['peak_tier1_auc']:.4f} | All-12 AUC = {res_C['peak_all12_auc']:.4f}")
    print("-" * 90)
    print(f"PRIMARY ENDPOINT   : Δ(C - B) = {delta_C_B:+.4f}")
    print(f"SECONDARY ENDPOINT : Δ(C - A) = {delta_C_A:+.4f}")
    print("=" * 90)
    
    print("\nTier-1 Diagnosis Level Breakdown (Gold-58 AUC):")
    header = f"| {'Target':<18} | {'Arm A':>8} | {'Arm B':>8} | {'Arm C':>8} | {'Δ(C - B)':>10} | {'Δ(C - A)':>10} |"
    sep = f"|{'-'*20}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*12}|{'-'*12}|"
    print(header)
    print(sep)
    for l in tier1_labels:
        auc_a = res_A["aucs"].get(l, 0.5)
        auc_b = res_B["aucs"].get(l, 0.5)
        auc_c = res_C["aucs"].get(l, 0.5)
        d_cb = auc_c - auc_b
        d_ca = auc_c - auc_a
        print(f"| {l:<18} | {auc_a:8.4f} | {auc_b:8.4f} | {auc_c:8.4f} | {d_cb:+10.4f} | {d_ca:+10.4f} |")
    print("=" * 90)
    
    # Save Benchmark Output
    out_payload = {
        "benchmark": "B11c-2b Three-Arm Intervention",
        "primary_endpoint_delta_C_B": delta_C_B,
        "secondary_endpoint_delta_C_A": delta_C_A,
        "res_A": res_A,
        "res_B": res_B,
        "res_C": res_C,
        "total_elapsed_minutes": (time.time() - t_start) / 60.0
    }
    
    out_path = Path("/home/eason/rsna-knee/scratchpad/b11c2b_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved full benchmark results to {out_path} in {(time.time() - t_start)/60:.1f} minutes!")

if __name__ == "__main__":
    main()
