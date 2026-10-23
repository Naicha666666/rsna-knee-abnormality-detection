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

def run_eval(res=320, tta_mode='prob_mean'):
    logits_orig_list = []
    logits_mirr_list = []
    probs_list = []
    with torch.no_grad():
        for u in gold_ids:
            r = row_of[u]
            cs = centers_from_mask(masks[r], k=42, train=False)
            x_raw = read_windows(vols[r], cs, res=384)
            if res != 384:
                x_in = F.interpolate(x_raw, size=(res, res), mode='bilinear', align_corners=False)
            else:
                x_in = x_raw
            
            x_p = ((x_in.to(dev) - mean) / std).unsqueeze(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l1 = m_0949(x_p).float()
                p1 = torch.sigmoid(l1)
            
            x_m_raw = mirror_triplets_64(x_in, cs)
            x_m = ((x_m_raw.to(dev) - mean) / std).unsqueeze(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l2 = m_0949(x_m).float()
                p2 = torch.sigmoid(l2)
            
            if tta_mode == 'prob_mean':
                p_final = (0.5 * (p1 + p2)).cpu().numpy()[0]
            elif tta_mode == 'logit_mean':
                p_final = torch.sigmoid(0.5 * (l1 + l2)).cpu().numpy()[0]
            elif tta_mode == 'no_tta':
                p_final = p1.cpu().numpy()[0]
            
            probs_list.append(p_final)
    
    preds = np.array(probs_list)
    aucs = [roc_auc_score(gold_y[:, c], preds[:, c]) for c in range(12)]
    return np.mean(aucs), aucs, preds

print("Testing 0.949 Base Model Variations on Gold-58...", flush=True)

macro_no_tta, aucs_no_tta, p_no_tta = run_eval(res=320, tta_mode='no_tta')
print(f"1. Res 320 No TTA:               Macro AUC = {macro_no_tta:.4f}", flush=True)

macro_320_prob, aucs_320_prob, p_320_prob = run_eval(res=320, tta_mode='prob_mean')
print(f"2. Res 320 Prob Mean Mirror TTA: Macro AUC = {macro_320_prob:.4f} (Original Nartaa setup)", flush=True)

macro_320_logit, aucs_320_logit, p_320_logit = run_eval(res=320, tta_mode='logit_mean')
print(f"3. Res 320 Logit Mean Mirror TTA:Macro AUC = {macro_320_logit:.4f}", flush=True)

macro_384_prob, aucs_384_prob, p_384_prob = run_eval(res=384, tta_mode='prob_mean')
print(f"4. Res 384 Native Mirror TTA:    Macro AUC = {macro_384_prob:.4f} (Native training resolution)", flush=True)

# Dual-resolution ensemble (320 + 384)
p_dual = 0.5 * p_320_prob + 0.5 * p_384_prob
macro_dual = np.mean([roc_auc_score(gold_y[:, c], p_dual[:, c]) for c in range(12)])
print(f"5. Dual Res (320 + 384) Ensemble:Macro AUC = {macro_dual:.4f}", flush=True)
