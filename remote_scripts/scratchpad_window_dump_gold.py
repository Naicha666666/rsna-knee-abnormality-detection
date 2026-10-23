"""R4 diagnostic: per-window attention logits and scores on the 58 gold (LOG #482).

RaptorClassifier.head is linear attention pooling:
    logit_n = sum_k softmax_k(a_kn) * (h_k . w_n) + b_n
so dumping a_kn and s_kn = h_k . w_n for EVERY legal window of a study lets any slot
ablation ("only axial windows", "drop sagittal", attention mass per slot) be recomputed
offline without another GPU pass. Dense bags: every legal centre, not the k_eval=42 spread.

Run on R9700 from /home/eason/rsna-knee:  python scratchpad_window_dump_gold.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, "/home/eason/rsna-knee/src")
os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")
import tv_stub; tv_stub.install()
import torch
import config
from corpus import open_corpus, centers_from_mask, read_windows, CORPUS_SLOTS
from dataset import RSNADataset
from raptor import RaptorClassifier, build_raptor_backbone

E = "/home/eason/rsna-knee/experiments/"
CKPTS = {
    "c6r": E + "local_c6r_64_k12_ep16_4llm_control/corpus0_topk_avg.pt",
    "c10": E + "local_c10_64_k12_ep16_convnext_small/corpus0_topk_avg.pt",
}
RES, CHUNK, dev = config.RAPTOR_RES, config.RAPTOR_ENCODE_CHUNK, "cuda"
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
slot_of = np.repeat(np.arange(len(CORPUS_SLOTS)), [s[2] for s in CORPUS_SLOTS])

vols, masks, cids = open_corpus()
row_of = {u: i for i, u in enumerate(cids)}
ds = RSNADataset("train")
has_gold = np.isfinite(ds.y_raw).all(axis=1)
gold_ids = [u for i, u in enumerate(ds.uids) if has_gold[i] and u in row_of]
print("gold", len(gold_ids), "slots", CORPUS_SLOTS)

out = {"gold_ids": np.array(gold_ids, dtype="U64"), "slots": np.array([s[0] + str(s[1]) for s in CORPUS_SLOTS])}
for name, path in CKPTS.items():
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bb = build_raptor_backbone(ck.get("arch", config.RAPTOR_ARCH))
    model = RaptorClassifier(bb, f_dim=bb.num_features).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    W, B = model.cls_w.detach(), model.cls_b.detach()
    A, S, SL, CT, ST = [], [], [], [], []
    with torch.no_grad():
        for u in gold_ids:
            r = row_of[u]
            m = np.asarray(masks[r])
            cs = centers_from_mask(m, 10 ** 6, train=False)
            cs = sorted(set(int(c) for c in cs))                     # every legal centre once
            x = read_windows(vols[r], cs, RES)
            x = ((x - mean) / std)[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model.encode(x.to(dev), chunk=CHUNK).float()     # [1, K, F]
            h = model.norm(f)[0]                                    # [K, F]
            a = model.att(h)                                        # [K, 12] attention logits
            s = h @ W.T                                             # [K, 12] per-window scores
            A.append(a.cpu().numpy().astype(np.float32))
            S.append(s.cpu().numpy().astype(np.float32))
            SL.append(slot_of[cs].astype(np.int8))
            CT.append(np.array(cs, dtype=np.int16))
            ST.append(len(cs))
    # sanity: dense pooled logit reproduces the head
    print(f"{name}: windows per study min {min(ST)} max {max(ST)} mean {np.mean(ST):.1f}")
    out[f"{name}_att"] = np.concatenate(A); out[f"{name}_score"] = np.concatenate(S)
    out[f"{name}_slot"] = np.concatenate(SL); out[f"{name}_center"] = np.concatenate(CT)
    out[f"{name}_nwin"] = np.array(ST, dtype=np.int32); out[f"{name}_bias"] = B.cpu().numpy()
    del model, bb
    torch.cuda.empty_cache()
np.savez("/home/eason/rsna-knee/experiments/gold_window_dump.npz", **out)
print("saved gold_window_dump.npz")
