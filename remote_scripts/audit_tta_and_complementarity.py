"""Rigorous Empirical Audit:
1. Mirror TTA Ablation on In-House Models (C14, C12, PF1) on Gold-58.
2. 0.949 Base vs In-House Per-Label Complementarity Audit on Gold-58.
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '/home/eason/rsna-knee/src')
os.environ.setdefault('RSNA_DATA_ROOT', '/home/eason/rsna-knee/data')
os.environ.setdefault('RSNA_CORPUS_DIR', '/home/eason/rsna-knee/data/corpus64')
os.environ.setdefault('RSNA_CORPUS_SLOTS', '64')

try:
    import tv_stub
    tv_stub.install()
except Exception:
    pass

import timm
import config
from corpus import open_corpus, centers_from_mask, read_windows
from dataset import RSNADataset, LABELS
from raptor import RaptorClassifier, build_raptor_backbone

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

# -----------------------------------------------------------------------------
# 1. Load Ground Truth Gold-58
# -----------------------------------------------------------------------------
print("Loading Gold-58 verified cohort and Corpus64...", flush=True)
vols, masks, cids = open_corpus()
row_of = {u: i for i, u in enumerate(cids)}
ds = RSNADataset('train')
y_raw = ds.y_raw
has_gold = np.isfinite(y_raw).all(axis=1)
gold_ids = [u for i, u in enumerate(ds.uids) if has_gold[i] and u in row_of]
idx_of = {u: i for i, u in enumerate(ds.uids)}
gold_y = np.stack([y_raw[idx_of[u]] for u in gold_ids]).astype(np.float32)
print(f"Verified Gold Studies: {len(gold_ids)} studies, shape: {gold_y.shape}", flush=True)

# -----------------------------------------------------------------------------
# 2. Anatomical Mirror TTA helper
# -----------------------------------------------------------------------------
SAG_CUTOFF_64 = 32

def mirror_triplets_64(triplets, center_slices):
    cs_arr = np.asarray(center_slices, dtype=np.int64)
    is_sag = (cs_arr < SAG_CUTOFF_64)
    sag_mask = torch.from_numpy(is_sag).to(triplets.device).view(-1, 1, 1, 1)
    return torch.where(sag_mask, triplets.flip(1), triplets.flip(-1))

def calc_all_aucs(y_true, y_pred):
    aucs = {}
    for j, lab in enumerate(LABELS):
        try:
            if not np.isfinite(y_pred[:, j]).all():
                aucs[lab] = 0.5
            else:
                aucs[lab] = float(roc_auc_score(y_true[:, j], y_pred[:, j]))
        except Exception:
            aucs[lab] = 0.5
    macro = float(np.mean(list(aucs.values())))
    tier1 = float(np.mean([aucs[l] for l in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA']]))
    return macro, tier1, aucs

# -----------------------------------------------------------------------------
# 3. Model Inference Engine (Plain vs Mirror TTA)
# -----------------------------------------------------------------------------
def eval_inhouse_model(ckpt_path, arch, res=384, k=42, allowed_centers=None):
    print(f"\nEvaluating: {arch} from {os.path.basename(ckpt_path)}...", flush=True)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    bb = timm.create_model(arch, pretrained=False, num_classes=0)
    model = RaptorClassifier(bb, f_dim=bb.num_features).to(dev)
    state = ck['state_dict']
    model.load_state_dict({k: v for k, v in state.items()})
    model.eval()

    preds_plain = []
    preds_tta = []

    with torch.no_grad():
        for u in gold_ids:
            r = row_of[u]
            cs = centers_from_mask(masks[r], k=k, train=False)
            if allowed_centers:
                cs = [c for c in cs if c in set(allowed_centers)] or cs
            x_raw = read_windows(vols[r], cs, res=res)  # (K, 3, res, res)
            
            # Plain
            x_p = ((x_raw.to(dev) - mean) / std).unsqueeze(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p_orig = torch.sigmoid(model(x_p, chunk=16).float())
            
            # Mirror TTA
            x_m_raw = mirror_triplets_64(x_raw, cs)
            x_m = ((x_m_raw.to(dev) - mean) / std).unsqueeze(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p_mir = torch.sigmoid(model(x_m, chunk=16).float())
            
            p_tta = 0.5 * (p_orig + p_mir)

            preds_plain.append(p_orig.cpu().numpy()[0])
            preds_tta.append(p_tta.cpu().numpy()[0])

    del model, bb, ck
    torch.cuda.empty_cache()
    gc.collect()

    return np.array(preds_plain), np.array(preds_tta)

# -----------------------------------------------------------------------------
# 4. Run Evaluations on In-House Models
# -----------------------------------------------------------------------------
# Model A: C14 (ConvNeXt-V2 Base 384)
ck_c14 = '/home/eason/rsna-knee/experiments/local_c14_64_k12_ep16_convnextv2_base_384_v6/corpus0_topk_avg.pt'
c14_plain, c14_tta = eval_inhouse_model(ck_c14, 'convnextv2_base.fcmae_ft_in22k_in1k_384', res=384, k=42)

# Model B: C12 (ConvNeXt Small 384)
ck_c12 = '/home/eason/rsna-knee/experiments/local_c12_64_k12_ep16_convnext_small_384_calibrated/corpus0_topk_avg.pt'
c12_plain, c12_tta = eval_inhouse_model(ck_c12, 'convnext_small.fb_in22k_ft_in1k_384', res=384, k=42)

# Model C: PF1 (ConvNeXt Small PF OA Specialist)
ck_pf1 = '/home/eason/rsna-knee/experiments/local_pf1_64_k12_ep16_convnext_pfoa_censag/corpus0_topk_avg.pt'
pf1_allow = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
pf1_plain, pf1_tta = eval_inhouse_model(ck_pf1, 'convnext_small.fb_in22k_ft_in1k_384', res=384, k=20, allowed_centers=pf1_allow)

# In-House Quad Ensemble
quad_plain = 0.60 * c14_plain + 0.40 * c12_plain
quad_plain[:, 6] = 0.50 * quad_plain[:, 6] + 0.50 * pf1_plain[:, 6]

quad_tta = 0.60 * c14_tta + 0.40 * c12_tta
quad_tta[:, 6] = 0.50 * quad_tta[:, 6] + 0.50 * pf1_tta[:, 6]

# -----------------------------------------------------------------------------
# 5. Load 0.949 Base Model on Gold-58
# -----------------------------------------------------------------------------
ck_0949_path = '/home/eason/rsna-knee/data/weights/nartaa/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
ck_0949 = torch.load(ck_0949_path, map_location='cpu', weights_only=False)

# Recorded official gold metrics from Nartaa:
eval_aucs = {k: float(v) for k, v in ck_0949.get('aucs', {}).items()}
m_eval_macro = float(ck_0949.get('gold_auc', 0.91895))

# Also run direct forward pass with bfloat16
class NartaaRaptor(nn.Module):
    def __init__(self, bb, f_dim=768, n=12):
        super().__init__()
        self.backbone = bb
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(nn.Linear(f_dim, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, f_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
    def encode(self, x):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        f = torch.cat([self.backbone(chunk) for chunk in flat.split(16)], dim=0)
        return f.view(B, K, -1)
    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb
    def forward(self, x):
        return self.head(self.encode(x))

bb_0949 = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False, num_classes=0, in_chans=3, img_size=320, global_pool='avg')
m_0949 = NartaaRaptor(bb_0949, f_dim=bb_0949.num_features).to(dev)
m_0949.load_state_dict(ck_0949['model'])
m_0949.eval()

preds_0949_eval = []
with torch.no_grad():
    for u in gold_ids:
        r = row_of[u]
        cs = centers_from_mask(masks[r], k=42, train=False)
        x_raw = read_windows(vols[r], cs, res=384)
        x_raw_320 = F.interpolate(x_raw, size=(320, 320), mode='bilinear', align_corners=False)
        x_p = ((x_raw_320.to(dev) - mean) / std).unsqueeze(0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p1 = torch.sigmoid(m_0949(x_p).float())
        x_m_raw = mirror_triplets_64(x_raw_320, cs)
        x_m = ((x_m_raw.to(dev) - mean) / std).unsqueeze(0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p2 = torch.sigmoid(m_0949(x_m).float())
        p_avg = (0.5 * (p1 + p2)).cpu().numpy()[0]
        preds_0949_eval.append(p_avg)

preds_0949_eval = np.array(preds_0949_eval)
m_live, t_live, live_aucs = calc_all_aucs(gold_y, preds_0949_eval)
if np.isfinite(preds_0949_eval).all() and m_live > 0.80:
    print(f"0.949 Base Evaluated on Gold-58 (Direct forward): Macro AUC = {m_live:.4f}")
    eval_aucs = live_aucs
    m_eval_macro = m_live
else:
    print(f"Using Official Checkpoint Recorded Gold-58 AUCs: Macro AUC = {m_eval_macro:.4f}")

# -----------------------------------------------------------------------------
# 6. OUTPUT 1: MIRROR TTA ABLATION TABLE
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("【PART 1: ANATOMICAL MIRROR TTA ABLATION STUDY ON GOLD-58】")
print("=" * 90)

m_c14_p, t_c14_p, aucs_c14_p = calc_all_aucs(gold_y, c14_plain)
m_c14_t, t_c14_t, aucs_c14_t = calc_all_aucs(gold_y, c14_tta)

m_c12_p, t_c12_p, aucs_c12_p = calc_all_aucs(gold_y, c12_plain)
m_c12_t, t_c12_t, aucs_c12_t = calc_all_aucs(gold_y, c12_tta)

m_qd_p, t_qd_p, aucs_qd_p = calc_all_aucs(gold_y, quad_plain)
m_qd_t, t_qd_t, aucs_qd_t = calc_all_aucs(gold_y, quad_tta)

print(f"{'Model Architecture':32s} | {'Plain AUC':>10s} | {'Mirror TTA':>10s} | {'Delta (Gain)':>12s} | {'Tier-1 Gain':>12s}")
print("-" * 90)
print(f"{'C14 (ConvNeXt-V2 Base 384)':32s} | {m_c14_p:10.4f} | {m_c14_t:10.4f} | {m_c14_t - m_c14_p:+12.4f} | {t_c14_t - t_c14_p:+12.4f}")
print(f"{'C12 (ConvNeXt Small 384)':32s} | {m_c12_p:10.4f} | {m_c12_t:10.4f} | {m_c12_t - m_c12_p:+12.4f} | {t_c12_t - t_c12_p:+12.4f}")
print(f"{'In-House Quad Ensemble':32s} | {m_qd_p:10.4f} | {m_qd_t:10.4f} | {m_qd_t - m_qd_p:+12.4f} | {t_qd_t - t_qd_p:+12.4f}")
print("-" * 90)

print("\n--- Per-Label Mirror TTA Breakdown on In-House Quad Ensemble ---")
print(f"{'Label':20s} | {'Plain':>8s} | {'Mirror TTA':>10s} | {'Delta':>8s} | {'Anatomy Plane':15s}")
print("-" * 75)
for lab in LABELS:
    plane = "Sagittal (Depth)" if lab in ['ACL', 'MCL', 'PF OA'] else ("Coronal (Width)" if "OA" in lab or "Meniscus" in lab else "Axial/Mixed")
    delta = aucs_qd_t[lab] - aucs_qd_p[lab]
    flag = "+++" if delta > 0.005 else ("+" if delta > 0 else "")
    print(f"{lab:20s} | {aucs_qd_p[lab]:8.4f} | {aucs_qd_t[lab]:10.4f} | {delta:+8.4f} {flag:3s} | {plane:15s}")

# -----------------------------------------------------------------------------
# 7. OUTPUT 2: 0.949 BASE VS IN-HOUSE COMPLEMENTARITY AUDIT
# -----------------------------------------------------------------------------
print("\n" + "=" * 90)
print("【PART 2: 0.949 BASE VS IN-HOUSE PER-LABEL COMPLEMENTARITY AUDIT ON GOLD-58】")
print("=" * 90)

# Simulate SuperFuse Blend using ranks or normalized probabilities
from scipy.stats import rankdata

def rank_cols(arr):
    out = np.zeros_like(arr)
    for c in range(arr.shape[1]):
        out[:, c] = rankdata(arr[:, c]) / max(1, len(arr) - 1)
    return out

# Convert predictions to percentile ranks
r_0949 = rank_cols(preds_0949_eval) if np.isfinite(preds_0949_eval).all() and m_live > 0.80 else None
r_quad = rank_cols(quad_tta)

print(f"{'Label':20s} | {'0.949 Base':>10s} | {'In-House Quad':>14s} | {'Delta (InH - 0.949)':>20s} | {'Status / Advantage':20s}")
print("-" * 95)
wins_inhouse = 0
for lab in LABELS:
    b_val = eval_aucs[lab]
    q_val = aucs_qd_t[lab]
    diff = q_val - b_val
    if diff > 0.0:
        wins_inhouse += 1
        comp_flag = "★ IN-HOUSE WINS"
    elif abs(diff) < 0.015:
        comp_flag = "≈ BALANCED"
    else:
        comp_flag = "BASE STRONG"
    print(f"{lab:20s} | {b_val:10.4f} | {q_val:14.4f} | {diff:+20.4f} | {comp_flag:20s}")

print("-" * 95)
tier1_0949 = float(np.mean([eval_aucs[l] for l in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA']]))
tier1_quad = float(np.mean([aucs_qd_t[l] for l in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA']]))
print(f"{'MACRO AUC SUMMARY':20s} | {m_eval_macro:10.4f} | {m_qd_t:14.4f} | {m_qd_t - m_eval_macro:+20.4f} | In-House Wins {wins_inhouse}/12")
print(f"{'TIER-1 AUC SUMMARY':20s} | {tier1_0949:10.4f} | {tier1_quad:14.4f} | {tier1_quad - tier1_0949:+20.4f} |")
print("=" * 95)
