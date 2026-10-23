"""Do C6 (CoAtNet) and C8 (RadImageNet ResNet-50) actually disagree?

Scores both checkpoints on the 58 gold studies through the same eval path training
used (deterministic 42-window bags), then reports each arm's macro-AUC, their
per-label rank correlation, and the macro-AUC of a weighted rank average over w.
The weight has to be chosen HERE (offline) and frozen before any submission.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, "/home/eason/rsna-knee/src")
os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")
import tv_stub; tv_stub.install()
import torch
import config
from corpus import open_corpus, centers_from_mask, read_windows
from dataset import RSNADataset
from train import macro_auc
from raptor import RaptorClassifier, build_raptor_backbone

CKPTS = {
    "C6_coatnet": "/home/eason/rsna-knee/experiments/local_c6_64_k12_ep16_4llm/corpus0.pt",
    "C8_radimagenet": "/home/eason/rsna-knee/experiments/local_c8_64_k12_ep16_radimagenet/corpus0.pt",
}
K_EVAL, RES, CHUNK = 42, config.RAPTOR_RES, config.RAPTOR_ENCODE_CHUNK
dev = "cuda"
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

vols, masks, cids = open_corpus()
row_of = {u: i for i, u in enumerate(cids)}
ds = RSNADataset("train")
y_raw = ds.y_raw
has_gold = np.isfinite(y_raw).all(axis=1)
gold_ids = [u for i, u in enumerate(ds.uids) if has_gold[i] and u in row_of]
idx_of = {u: i for i, u in enumerate(ds.uids)}
gold_y = np.stack([y_raw[idx_of[u]] for u in gold_ids]).astype(np.float32)
print(f"gold studies: {len(gold_ids)}")

preds = {}
for name, path in CKPTS.items():
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", config.RAPTOR_ARCH)
    print(f"{name}: arch={arch} gold_auc(recorded)={ck.get('gold_auc'):.4f} epoch={ck.get('epoch')}")
    bb = build_raptor_backbone(arch)
    model = RaptorClassifier(bb, f_dim=bb.num_features).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    out = []
    with torch.no_grad():
        for b0 in range(0, len(gold_ids), 2):
            xs = []
            for u in gold_ids[b0:b0 + 2]:
                r = row_of[u]
                cs = centers_from_mask(masks[r], K_EVAL, train=False)
                xs.append(read_windows(vols[r], cs, RES))
            x = torch.stack(xs, 0)
            b, kk = x.shape[:2]
            x = ((x.flatten(0, 1) - mean) / std).view(b, kk, *x.shape[2:])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x.to(dev), chunk=CHUNK).float()
            out.append(torch.sigmoid(o).cpu().numpy())
    preds[name] = np.concatenate(out)
    m, per = macro_auc(gold_y, preds[name])
    print(f"  recomputed gold macro-AUC = {m:.4f}")
    del model, bb
    torch.cuda.empty_cache()

import pandas as pd
LAB = config.LABELS
a, b = preds["C6_coatnet"], preds["C8_radimagenet"]
rho = [pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method="spearman") for j in range(len(LAB))]
print("\nper-label Spearman between the two arms on the 58 gold:")
for j, l in enumerate(LAB):
    print(f"  {l:18s} {rho[j]:+.3f}")
print(f"  median {np.median(rho):+.3f}  min {np.min(rho):+.3f}")
print("  reference: label-source arms that gave nothing were >=0.90 correlated")


def pct_rank(p):
    return np.stack([pd.Series(p[:, j]).rank(pct=True).to_numpy() for j in range(p.shape[1])], 1)


ra, rb = pct_rank(a), pct_rank(b)
print("\nweighted rank average, macro-AUC on the 58 gold:")
grid = {}
for w in np.round(np.arange(0.0, 1.01, 0.1), 2):
    fused = w * ra + (1 - w) * rb          # w = weight on C6
    grid[float(w)] = float(macro_auc(gold_y, fused)[0])
    print(f"  w_C6={w:.1f}  macro-AUC={grid[float(w)]:.4f}")
best_w = max(grid, key=grid.get)
print(f"\nbest w_C6 on gold = {best_w:.1f} ({grid[best_w]:.4f}); "
      f"w=1.0 (C6 alone) = {grid[1.0]:.4f}; w=0.5 = {grid[0.5]:.4f}")
print("NOTE 58-gold macro noise is ~0.018, so differences smaller than that are not real.")
np.savez("/home/eason/rsna-knee/experiments/gold_fuse_check.npz",
         gold_ids=np.array(gold_ids), gold_y=gold_y, c6=a, c8=b, rho=np.array(rho))
json.dump({"grid": grid, "rho_median": float(np.median(rho)), "best_w": float(best_w)},
          open("/home/eason/rsna-knee/experiments/gold_fuse_check.json", "w"), indent=1)
print("saved gold_fuse_check.npz / .json")
