"""Gold-58 predictions of the S4 arms T / K (LOG #526; written before either arm finished). Same eval path
as gold_pf_preds.py (k_eval 20, the checkpoint's window policy restored) plus the S4 head and its positions
(T: true allow-list index; K: the study's fixed permutation), restored from the checkpoint metadata.

Same deterministic 42-window eval path as gold_fuse_check.py; only predictions are
saved, for the correlation check against the public CoAt readers.

Derived from: Do C6 (CoAtNet) and C8 (RadImageNet ResNet-50) actually disagree?

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
from raptor import RaptorClassifier, build_raptor_backbone, build_head
from corpus import s4_positions

E = "/home/eason/rsna-knee/experiments/"
CKPTS = {
    "s4t": (E + "local_s4t_64_k12_ep16_convnext_pfoa_censag_seq/corpus0_topk_avg.pt", 20),
    "s4k": (E + "local_s4k_64_k12_ep16_convnext_pfoa_censag_seq/corpus0_topk_avg.pt", 20),
}
RES, CHUNK = config.RAPTOR_RES, config.RAPTOR_ENCODE_CHUNK
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
import time
for name, (path, K_EVAL) in CKPTS.items():
    ck = torch.load(path, map_location="cpu", weights_only=False)
    config.CORPUS_CENTRE_ALLOW = ck.get("centre_allow")      # read at call time by centers_from_mask
    config.RAPTOR_HEAD, config.S4_ORDER = ck.get("head") or "mil", ck.get("s4_order")
    assert config.RAPTOR_HEAD == "s4seq" and config.S4_ORDER == {"s4t": "true", "s4k": "shuffled"}[name], (config.RAPTOR_HEAD, config.S4_ORDER)
    print(f"{name}: centre_allow={ck.get('centre_allow')} train_labels={ck.get('train_labels')} k_eval={K_EVAL}")
    t0 = time.time()
    arch = ck.get("arch", config.RAPTOR_ARCH)
    print(f"{name}: arch={arch} gold_auc(recorded)={ck.get('gold_auc'):.4f} epoch={ck.get('epoch')}")
    bb = build_raptor_backbone(arch)
    model = build_head(bb, config.RAPTOR_HEAD).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    out = []
    with torch.no_grad():
        for b0 in range(0, len(gold_ids), 2):
            xs, ps = [], []
            for u in gold_ids[b0:b0 + 2]:
                r = row_of[u]
                cs = centers_from_mask(masks[r], K_EVAL, train=False)
                xs.append(read_windows(vols[r], cs, RES)); ps.append(s4_positions(u, cs))
            model.set_positions(torch.from_numpy(np.stack(ps)).long())
            x = torch.stack(xs, 0)
            b, kk = x.shape[:2]
            x = ((x.flatten(0, 1) - mean) / std).view(b, kk, *x.shape[2:])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x.to(dev), chunk=CHUNK).float()
            out.append(torch.sigmoid(o).cpu().numpy())
    preds[name] = np.concatenate(out)
    from sklearn.metrics import roc_auc_score
    j = list(config.LABELS).index("PF OA") if hasattr(config, "LABELS") else 6
    print(f"  PF OA gold AUC = {roc_auc_score(gold_y[:, j], preds[name][:, j]):.4f} (recorded top-3 {ck.get('gold_auc'):.4f}) | "
          f"{(time.time() - t0):.0f} s for 58 studies")
    del model, bb
    torch.cuda.empty_cache()

np.savez("/home/eason/rsna-knee/experiments/gold_s4_preds.npz",
         gold_ids=np.array(gold_ids), gold_y=gold_y, **preds)
print("saved gold_s4_preds.npz")
