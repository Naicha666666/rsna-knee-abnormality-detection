import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from corpus import open_corpus, centers_from_mask, read_windows
from dataset import RSNADataset, LABELS
from raptor import RaptorClassifier

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

# 1. Load Gold-58
vols, masks, cids = open_corpus()
row_of = {u: i for i, u in enumerate(cids)}
ds = RSNADataset('train')
y_raw = ds.y_raw
has_gold = np.isfinite(y_raw).all(axis=1)
gold_ids = [u for i, u in enumerate(ds.uids) if has_gold[i] and u in row_of]
idx_of = {u: i for i, u in enumerate(ds.uids)}
gold_y = np.stack([y_raw[idx_of[u]] for u in gold_ids]).astype(np.float32)

SAG_CUTOFF_64 = 32
def mirror_triplets_64(triplets, center_slices):
    cs_arr = np.asarray(center_slices, dtype=np.int64)
    is_sag = (cs_arr < SAG_CUTOFF_64)
    sag_mask = torch.from_numpy(is_sag).to(triplets.device).view(-1, 1, 1, 1)
    return torch.where(sag_mask, triplets.flip(1), triplets.flip(-1))

def eval_model(ckpt_path, arch, res=384, k=42, allowed_centers=None):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    bb = timm.create_model(arch, pretrained=False, num_classes=0)
    model = RaptorClassifier(bb, f_dim=bb.num_features).to(dev)
    model.load_state_dict({k: v for k, v in ck['state_dict'].items()})
    model.eval()
    preds = []
    with torch.no_grad():
        for u in gold_ids:
            r = row_of[u]
            cs = centers_from_mask(masks[r], k=k, train=False)
            if allowed_centers:
                cs = [c for c in cs if c in set(allowed_centers)] or cs
            x_raw = read_windows(vols[r], cs, res=res)
            x_p = ((x_raw.to(dev) - mean) / std).unsqueeze(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p1 = torch.sigmoid(model(x_p, chunk=16).float())
            x_m_raw = mirror_triplets_64(x_raw, cs)
            x_m = ((x_m_raw.to(dev) - mean) / std).unsqueeze(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p2 = torch.sigmoid(model(x_m, chunk=16).float())
            preds.append((0.5 * (p1 + p2)).cpu().numpy()[0])
    return np.array(preds)

# Evaluate In-House Quad
c14_tta = eval_model('/home/eason/rsna-knee/experiments/local_c14_64_k12_ep16_convnextv2_base_384_v6/corpus0_topk_avg.pt', 'convnextv2_base.fcmae_ft_in22k_in1k_384')
c12_tta = eval_model('/home/eason/rsna-knee/experiments/local_c12_64_k12_ep16_convnext_small_384_calibrated/corpus0_topk_avg.pt', 'convnext_small.fb_in22k_ft_in1k_384')
pf1_tta = eval_model('/home/eason/rsna-knee/experiments/local_pf1_64_k12_ep16_convnext_pfoa_censag/corpus0_topk_avg.pt', 'convnext_small.fb_in22k_ft_in1k_384', k=20, allowed_centers=[4,5,6,7,8,9,10,11,12,13,20,21,22,23,24,25,26,27,28,29])

quad_tta = 0.60 * c14_tta + 0.40 * c12_tta
quad_tta[:, 6] = 0.50 * quad_tta[:, 6] + 0.50 * pf1_tta[:, 6]

# 0.949 Base
ck_0949 = torch.load('/home/eason/rsna-knee/data/weights/nartaa/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt', map_location='cpu', weights_only=False)
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

preds_0949 = []
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
        preds_0949.append((0.5 * (p1 + p2)).cpu().numpy()[0])
preds_0949 = np.array(preds_0949)

# Percentile Rank Transforms
def to_rank(arr):
    r = np.zeros_like(arr)
    for c in range(arr.shape[1]):
        r[:, c] = rankdata(arr[:, c]) / max(1, len(arr) - 1)
    return r

r_base = to_rank(preds_0949)
r_quad = to_rank(quad_tta)

# Base AUCs
aucs_base = [roc_auc_score(gold_y[:, c], r_base[:, c]) for c in range(12)]
macro_base = np.mean(aucs_base)

# Uniform 0.20 blend
r_uni = 0.80 * r_base + 0.20 * r_quad
aucs_uni = [roc_auc_score(gold_y[:, c], r_uni[:, c]) for c in range(12)]
macro_uni = np.mean(aucs_uni)

# Heuristic PF OA override (0.20 for 11, 0.35 for PF OA)
r_heuri = 0.80 * r_base + 0.20 * r_quad
r_heuri[:, 6] = 0.65 * r_base[:, 6] + 0.35 * r_quad[:, 6]
aucs_heuri = [roc_auc_score(gold_y[:, c], r_heuri[:, c]) for c in range(12)]
macro_heuri = np.mean(aucs_heuri)

# Sweep optimal per-label weights
weights_sweep = np.linspace(0.0, 0.50, 11)  # w_inhouse from 0.0 to 0.50
optimal_w = {}
optimal_auc = {}
conservative_w = {}
conservative_auc = {}

# Priors based on clinical advantage
priors = {
    'Baker\'s': 0.35,
    'PF OA': 0.35,
    'Lateral OA': 0.30,
    'Lateral Meniscus': 0.25,
    'Contusion': 0.20,
    'Medial OA': 0.20,
    'ACL': 0.20,
    'Fracture': 0.15,
    'Effusion': 0.10,
    'Medial Meniscus': 0.10,
    'MCL': 0.05,
    'Synovitis': 0.05,
}

r_opt = np.zeros_like(r_base)
r_cons = np.zeros_like(r_base)

for c, lab in enumerate(LABELS):
    best_w = 0.0
    best_a = -1.0
    for w in weights_sweep:
        b_c = (1.0 - w) * r_base[:, c] + w * r_quad[:, c]
        a = roc_auc_score(gold_y[:, c], b_c)
        if a > best_a:
            best_a = a
            best_w = w
    optimal_w[lab] = best_w
    optimal_auc[lab] = best_a
    r_opt[:, c] = (1.0 - best_w) * r_base[:, c] + best_w * r_quad[:, c]

    # Conservative regularized weight (blended with clinical prior to prevent sample noise)
    cw = 0.5 * best_w + 0.5 * priors[lab]
    conservative_w[lab] = round(cw, 2)
    b_cw = (1.0 - cw) * r_base[:, c] + cw * r_quad[:, c]
    conservative_auc[lab] = roc_auc_score(gold_y[:, c], b_cw)
    r_cons[:, c] = b_cw

macro_opt = np.mean(list(optimal_auc.values()))
macro_cons = np.mean(list(conservative_auc.values()))

print("\n" + "=" * 105)
print("【PER-LABEL 12-WEIGHT BLEND OPTIMIZATION STUDY ON GOLD-58】")
print("=" * 105)
print(f"{'Label':20s} | {'Base 0.949':>10s} | {'Uniform 0.20':>12s} | {'Optimal W_Quad':>15s} | {'Opt AUC':>10s} | {'Conservative W':>15s} | {'Cons AUC':>10s}")
print("-" * 105)
for lab in LABELS:
    c = LABELS.index(lab)
    print(f"{lab:20s} | {aucs_base[c]:10.4f} | {aucs_uni[c]:12.4f} | {optimal_w[lab]:15.2f} | {optimal_auc[lab]:10.4f} | {conservative_w[lab]:15.2f} | {conservative_auc[lab]:10.4f}")

print("-" * 105)
print(f"{'MACRO AUC SUMMARY':20s} | {macro_base:10.4f} | {macro_uni:12.4f} | {'—':>15s} | {macro_opt:10.4f} | {'—':>15s} | {macro_cons:10.4f}")
print(f"Gain vs Base 0.949:      |  +0.0000   | {macro_uni - macro_base:+12.4f} |               | {macro_opt - macro_base:+10.4f}  |               | {macro_cons - macro_base:+10.4f}")
print("=" * 105)
