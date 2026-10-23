"""评估王牌 2 (CoAtNet-2 384, Seed 2026) 与原版 0.949 Base (Seed 42) 在 Gold-58 上的多模/双种子集成表现。"""
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, rankdata

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
N_GOLD = len(gold_ids)
print(f"Loaded Gold-58: {N_GOLD} cases, targets shape {gold_y.shape}", flush=True)

SAG_CUTOFF_64 = 32
def mirror_triplets_64(triplets, center_slices):
    cs_arr = np.asarray(center_slices, dtype=np.int64)
    is_sag = (cs_arr < SAG_CUTOFF_64)
    sag_mask = torch.from_numpy(is_sag).to(triplets.device).view(-1, 1, 1, 1)
    return torch.where(sag_mask, triplets.flip(1), triplets.flip(-1))

# 2. 模型定义
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
    if 'state_dict' in ck:
        state = ck['state_dict']
    elif 'model' in ck:
        state = ck['model']
    else:
        state = ck
    # 清洗 state dict 中的前缀与命名差异
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
    incompatible = m.load_state_dict(clean_state, strict=True)
    print(f"Loaded {os.path.basename(ckpt_path)}: missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)} (100% EXACT MATCH)", flush=True)
    m.eval()
    return m

# 加载模型
p_0949 = '/home/eason/rsna-knee/data/weights/nartaa/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
p_s2026_avg = '/home/eason/rsna-knee/experiments/local_c6_coatnet_s2026_v6/corpus0_topk_avg.pt'
p_s2026_best = '/home/eason/rsna-knee/experiments/local_c6_coatnet_s2026_v6/corpus0_best_epoch.pt'

print(f"Loading Base 0.949 from {p_0949}...", flush=True)
m_0949 = load_model(p_0949)

print(f"Loading S2026 Top-K Avg from {p_s2026_avg}...", flush=True)
m_s2026_avg = load_model(p_s2026_avg)

print(f"Loading S2026 Best Epoch from {p_s2026_best}...", flush=True)
m_s2026_best = load_model(p_s2026_best)

def predict_model(m, k=42, res=384, name="Model"):
    logits_list = []
    probs_list = []
    with torch.no_grad():
        for i, u in enumerate(gold_ids):
            r = row_of[u]
            cs = centers_from_mask(masks[r], k=k, train=False)
            x_raw = read_windows(vols[r], cs, res=res)
            if res != 384:
                x_in = F.interpolate(x_raw, size=(res, res), mode='bilinear', align_corners=False)
            else:
                x_in = x_raw
            
            x_p = ((x_in.to(dev) - mean) / std).unsqueeze(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l1 = m(x_p).float()
            
            x_m_raw = mirror_triplets_64(x_in, cs)
            x_m = ((x_m_raw.to(dev) - mean) / std).unsqueeze(0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                l2 = m(x_m).float()
            
            # Logit TTA
            l_mean = 0.5 * (l1 + l2)
            p_mean = torch.sigmoid(l_mean)
            
            logits_list.append(l_mean.cpu().numpy()[0])
            probs_list.append(p_mean.cpu().numpy()[0])
            if (i + 1) % 15 == 0 or (i + 1) == len(gold_ids):
                print(f"  [{name}] {i + 1}/{len(gold_ids)} studies inferred", flush=True)
            
    return np.array(logits_list), np.array(probs_list)

print("\n--- Running Inference on Gold-58 (Res 384 + Mirror TTA) ---", flush=True)
print("1. Inferring Base 0.949 (Seed 42)...", flush=True)
l_0949, p_0949 = predict_model(m_0949, name="Base 0.949")

print("2. Inferring WangPai 2 S2026 Top-K Avg...", flush=True)
l_s2026_avg, p_s2026_avg = predict_model(m_s2026_avg, name="S2026 Avg")

print("3. Inferring WangPai 2 S2026 Best Epoch...", flush=True)
l_s2026_best, p_s2026_best = predict_model(m_s2026_best, name="S2026 Best")

def calc_macro_auc(y_true, y_pred):
    aucs = [roc_auc_score(y_true[:, c], y_pred[:, c]) for c in range(12)]
    return np.mean(aucs), aucs

auc_0949_m, aucs_0949 = calc_macro_auc(gold_y, p_0949)
auc_avg_m, aucs_avg = calc_macro_auc(gold_y, p_s2026_avg)
auc_best_m, aucs_best = calc_macro_auc(gold_y, p_s2026_best)

print("\n" + "=" * 80)
print(f"=== 单模表现对比 (Gold-58 Macro AUC) ===")
print(f"Base 0.949 (Seed 42, OAI SWA):     Macro AUC = {auc_0949_m:.5f}")
print(f"王牌 2 S2026 Top-K Avg (Prompt v6): Macro AUC = {auc_avg_m:.5f}")
print(f"王牌 2 S2026 Best Epoch (Prompt v6):Macro AUC = {auc_best_m:.5f}")
print("=" * 80)

# 计算标签相关性 (Spearman rho)
rhos = [spearmanr(p_0949[:, c], p_s2026_avg[:, c]).statistic for c in range(12)]
print(f"\n=== Base 0.949 与 S2026 预测相关性 (Spearman rho) ===")
for c, name in enumerate(LABELS):
    print(f"  {name:18s}: rho = {rhos[c]:.4f} | Base={aucs_0949[c]:.4f} | S2026={aucs_avg[c]:.4f}")
print(f"  均值相关性 Mean rho: {np.mean(rhos):.4f}")
print("=" * 80)

# 测试双种子集成 (Multi-Seed Ensemble)
print("\n=== 多种子/双种子集成 (Multi-Seed Ensemble: Base 0.949 + S2026) ===")

# A. 概率平均 (50/50)
p_blend_50 = 0.5 * p_0949 + 0.5 * p_s2026_avg
auc_blend_50, aucs_blend_50 = calc_macro_auc(gold_y, p_blend_50)
print(f"1. 50/50 概率平均 (Prob Mean): Macro AUC = {auc_blend_50:.5f} (Δ vs Base: {auc_blend_50 - auc_0949_m:+.5f})")

# B. 对数几率平均 (Logit Mean)
l_blend_50 = 0.5 * l_0949 + 0.5 * l_s2026_avg
p_logit_50 = 1.0 / (1.0 + np.exp(-l_blend_50))
auc_logit_50, aucs_logit_50 = calc_macro_auc(gold_y, p_logit_50)
print(f"2. 50/50 对数几率平均 (Logit Mean): Macro AUC = {auc_logit_50:.5f} (Δ vs Base: {auc_logit_50 - auc_0949_m:+.5f})")

# C. 秩次平均 (Rank Mean)
r_0949 = np.stack([rankdata(p_0949[:, c]) / len(p_0949) for c in range(12)], axis=1)
r_s2026 = np.stack([rankdata(p_s2026_avg[:, c]) / len(p_s2026_avg) for c in range(12)], axis=1)
p_rank_50 = 0.5 * r_0949 + 0.5 * r_s2026
auc_rank_50, aucs_rank_50 = calc_macro_auc(gold_y, p_rank_50)
print(f"3. 50/50 秩次平均 (Rank Mean): Macro AUC = {auc_rank_50:.5f} (Δ vs Base: {auc_rank_50 - auc_0949_m:+.5f})")

# D. 权重网格搜索 (w_base vs w_s2026)
best_w, best_auc = 0.5, 0.0
for w in np.linspace(0.1, 0.9, 17):
    p_w = w * p_0949 + (1 - w) * p_s2026_avg
    auc_w, _ = calc_macro_auc(gold_y, p_w)
    if auc_w > best_auc:
        best_auc = auc_w
        best_w = w
print(f"4. 最优概率加权 (w_base={best_w:.2f}, w_s2026={1-best_w:.2f}): Macro AUC = {best_auc:.5f} (Δ vs Base: {best_auc - auc_0949_m:+.5f})")

# E. 与 Best Epoch 融合
p_best_blend = 0.5 * p_0949 + 0.5 * p_s2026_best
auc_best_blend, _ = calc_macro_auc(gold_y, p_best_blend)
print(f"5. Base 0.949 + S2026 Best Epoch (50/50): Macro AUC = {auc_best_blend:.5f} (Δ vs Base: {auc_best_blend - auc_0949_m:+.5f})")

print("=" * 80)
