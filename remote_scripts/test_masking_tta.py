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
from corpus import open_corpus, centers_from_mask, read_windows
from dataset import RSNADataset, LABELS

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)

print("Loading Gold-58 cohort...", flush=True)
vols, masks, cids = open_corpus()
row_of = {u: i for i, u in enumerate(cids)}
ds = RSNADataset('train')
y_raw = ds.y_raw
has_gold = np.isfinite(y_raw).all(axis=1)
gold_ids = [u for i, u in enumerate(ds.uids) if has_gold[i] and u in row_of]
idx_of = {u: i for i, u in enumerate(ds.uids)}
gold_y = np.stack([y_raw[idx_of[u]] for u in gold_ids]).astype(np.float32)
print(f"Verified Gold Studies: {len(gold_ids)}", flush=True)

class RaptorClassifierMasked(nn.Module):
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
    def head(self, feats, mask=None):
        h = self.norm(feats)
        a = self.att(h)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if not mask.any():
                mask = torch.ones_like(mask, dtype=torch.bool)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            a = a.masked_fill(~mask, -10000.0)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb
    def forward(self, x, mask=None):
        return self.head(self.encode(x), mask=mask)

ck_path = '/home/eason/rsna-knee/data/weights/nartaa/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
ck = torch.load(ck_path, map_location='cpu', weights_only=False)
bb = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False, num_classes=0, in_chans=3, img_size=320, global_pool='avg')
model = RaptorClassifierMasked(bb, f_dim=bb.num_features).to(dev)
model.load_state_dict(ck['model'])
model.eval()

def mirror_triplets(triplets, center_slices):
    cs_arr = np.asarray(center_slices, dtype=np.int64)
    is_sag = (cs_arr < 32)
    sag_mask = torch.from_numpy(is_sag).to(triplets.device).view(-1, 1, 1, 1)
    return torch.where(sag_mask, triplets.flip(1), triplets.flip(-1))

preds_prob_nomask = []
preds_logit_nomask = []
preds_prob_masked = []
preds_logit_masked = []

print("Running Gold-58 inference across all 4 modes...", flush=True)
with torch.no_grad():
    for u in gold_ids:
        r = row_of[u]
        m_r = masks[r]
        cs = centers_from_mask(m_r, k=42, train=False)
        x_raw = read_windows(vols[r], cs, res=384)
        x_320 = F.interpolate(x_raw, size=(320, 320), mode='bilinear', align_corners=False)
        x_p = ((x_320.to(dev) - mean) / std).unsqueeze(0)
        x_m_raw = mirror_triplets(x_320, cs)
        x_m = ((x_m_raw.to(dev) - mean) / std).unsqueeze(0)
        
        # Valid window mask
        w_valid = torch.tensor([(m_r[max(0, c-1)] > 0 or m_r[c] > 0 or m_r[min(len(m_r)-1, c+1)] > 0) for c in cs], dtype=torch.bool, device=dev)
        
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l1_no = model(x_p).float()
            l2_no = model(x_m).float()
            l1_mk = model(x_p, mask=w_valid).float()
            l2_mk = model(x_m, mask=w_valid).float()
        
        # Prob averaging vs Logit averaging (No mask)
        p1_no = torch.sigmoid(l1_no); p2_no = torch.sigmoid(l2_no)
        preds_prob_nomask.append((0.5 * (p1_no + p2_no)).cpu().numpy()[0])
        preds_logit_nomask.append(torch.sigmoid(0.5 * (l1_no + l2_no)).cpu().numpy()[0])
        
        # Prob averaging vs Logit averaging (Masked)
        p1_mk = torch.sigmoid(l1_mk); p2_mk = torch.sigmoid(l2_mk)
        preds_prob_masked.append((0.5 * (p1_mk + p2_mk)).cpu().numpy()[0])
        preds_logit_masked.append(torch.sigmoid(0.5 * (l1_mk + l2_mk)).cpu().numpy()[0])

def eval_preds(preds, name):
    arr = np.array(preds)
    aucs = {}
    for j, lab in enumerate(LABELS):
        aucs[lab] = float(roc_auc_score(gold_y[:, j], arr[:, j]))
    macro = float(np.mean(list(aucs.values())))
    print(f"\n=== {name} ===")
    print(f"Macro AUC: {macro:.5f}")
    for k, v in aucs.items():
        print(f"  {k:20s}: {v:.4f}")
    return macro, aucs

m_p_no, a_p_no = eval_preds(preds_prob_nomask, "1. Prob TTA (No Mask) [Original Nartaa]")
m_l_no, a_l_no = eval_preds(preds_logit_nomask, "2. Logit TTA (No Mask)")
m_p_mk, a_p_mk = eval_preds(preds_prob_masked, "3. Prob TTA (Masked)")
m_l_mk, a_l_mk = eval_preds(preds_logit_masked, "4. Logit TTA (Masked) [Proposed SOTA]")
