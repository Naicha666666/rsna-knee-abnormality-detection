"""保存预测并搜索 Base 0.949 与 S2026 的最佳融合排程。"""
import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '/home/eason/rsna-knee/src')
os.environ.setdefault('RSNA_DATA_ROOT', '/home/eason/rsna-knee/data')
os.environ.setdefault('RSNA_CORPUS_DIR', '/home/eason/rsna-knee/data/corpus64')
os.environ.setdefault('RSNA_CORPUS_SLOTS', '64')
os.environ["HIP_VISIBLE_DEVICES"] = "0"

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

# 1. 加载 Gold-58
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

def load_model(ckpt_path):
    bb = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False, num_classes=0, in_chans=3, img_size=384, global_pool='avg')
    m = NartaaRaptor(bb, f_dim=bb.num_features).to(dev)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ck['state_dict'] if 'state_dict' in ck else (ck['model'] if 'model' in ck else ck)
    clean_state = {}
    for k, v in state.items():
        clean_k = k
        if clean_k.startswith('module.'):
            clean_k = clean_k[7:]
        if clean_k == 'cls_w':
            clean_k = 'clsW'
        elif clean_k == 'cls_b':
            clean_k = 'clsb'
        clean_state[clean_k] = v
    m.load_state_dict(clean_state, strict=True)
    m.eval()
    return m

m_0949 = load_model('/home/eason/rsna-knee/data/weights/nartaa/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt')
m_s2026 = load_model('/home/eason/rsna-knee/experiments/local_c6_coatnet_s2026_v6/corpus0_topk_avg.pt')

def infer(m):
    probs = []
    with torch.no_grad():
        for u in gold_ids:
            r = row_of[u]
            cs = centers_from_mask(masks[r], k=42, train=False)
            x_raw = read_windows(vols[r], cs, res=384)
            x_p = ((x_raw.to(dev) - mean) / std).unsqueeze(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l1 = m(x_p).float()
            x_m_raw = mirror_triplets_64(x_raw, cs)
            x_m = ((x_m_raw.to(dev) - mean) / std).unsqueeze(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l2 = m(x_m).float()
            p = torch.sigmoid(0.5 * (l1 + l2))
            probs.append(p.cpu().numpy()[0])
    return np.array(probs)

p_0949 = infer(m_0949)
p_s2026 = infer(m_s2026)

np.save('/home/eason/rsna-knee/p_0949.npy', p_0949)
np.save('/home/eason/rsna-knee/p_s2026.npy', p_s2026)
np.save('/home/eason/rsna-knee/gold_y.npy', gold_y)
print("Saved predictions to npy files.", flush=True)

# 细粒度搜索全局加权
best_w, best_auc = 0.5, 0.0
for w in np.linspace(0.5, 0.98, 49):
    p_w = w * p_0949 + (1 - w) * p_s2026
    auc_w = np.mean([roc_auc_score(gold_y[:, c], p_w[:, c]) for c in range(12)])
    if auc_w > best_auc:
        best_auc = auc_w
        best_w = w

print(f"Global Best: w_base={best_w:.2f}, w_s2026={1-best_w:.2f} -> Macro AUC = {best_auc:.5f}", flush=True)

# 标签级最优加权
opt_weights = []
p_opt = np.zeros_like(p_0949)
for c in range(12):
    c_best_w, c_best_auc = 1.0, roc_auc_score(gold_y[:, c], p_0949[:, c])
    for w in np.linspace(0.0, 1.0, 21):
        cand = w * p_0949[:, c] + (1 - w) * p_s2026[:, c]
        cand_auc = roc_auc_score(gold_y[:, c], cand)
        if cand_auc > c_best_auc:
            c_best_auc = cand_auc
            c_best_w = w
    opt_weights.append(c_best_w)
    p_opt[:, c] = c_best_w * p_0949[:, c] + (1 - c_best_w) * p_s2026[:, c]
    print(f"  {LABELS[c]:18s}: w_base={c_best_w:.2f}, w_s2026={1-c_best_w:.2f} | AUC={c_best_auc:.5f} (Base={roc_auc_score(gold_y[:, c], p_0949[:, c]):.5f})", flush=True)

opt_auc = np.mean([roc_auc_score(gold_y[:, c], p_opt[:, c]) for c in range(12)])
base_auc = np.mean([roc_auc_score(gold_y[:, c], p_0949[:, c]) for c in range(12)])
print(f"Per-Label Optimized Ensemble: Macro AUC = {opt_auc:.5f} (Δ vs Base: {opt_auc - base_auc:+.5f})", flush=True)
