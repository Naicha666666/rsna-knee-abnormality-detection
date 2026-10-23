"""C6 weights on the 58 gold studies: reference windows vs legal windows. Sanity only --
58-gold cannot rank this (LOG #390/#392); the decision is the leaderboard. No training."""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/eason/rsna-knee/src")
os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")
import tv_stub; tv_stub.install()  # noqa: E401,E702
import torch  # noqa: E402
import config  # noqa: E402
from corpus import open_corpus, centers_from_mask, read_windows, LEGAL_WINDOWS  # noqa: E402
from dataset import RSNADataset  # noqa: E402
from train import macro_auc  # noqa: E402
from raptor import RaptorClassifier, build_raptor_backbone  # noqa: E402

assert LEGAL_WINDOWS is False
CK = "/home/eason/rsna-knee/experiments/local_c6_64_k12_ep16_4llm/corpus0.pt"
vols, masks, cids = open_corpus()
row_of = {u: i for i, u in enumerate(cids)}
ds = RSNADataset("train")
gold = np.isfinite(ds.y_raw).all(axis=1)
gold_ids = [u for i, u in enumerate(ds.uids) if gold[i] and u in row_of]
idx_of = {u: i for i, u in enumerate(ds.uids)}
gold_y = np.stack([ds.y_raw[idx_of[u]] for u in gold_ids]).astype(np.float32)

ck = torch.load(CK, map_location="cpu", weights_only=False)
bb = build_raptor_backbone(ck.get("arch", config.RAPTOR_ARCH))
model = RaptorClassifier(bb, f_dim=bb.num_features).cuda()
model.load_state_dict(ck["state_dict"])
model.eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def score(legal):
    out, changed = [], 0
    with torch.no_grad():
        for b0 in range(0, len(gold_ids), 2):
            xs = []
            for u in gold_ids[b0:b0 + 2]:
                r = row_of[u]
                cs = centers_from_mask(masks[r], 42, train=False, legal=legal)
                changed += int(cs != centers_from_mask(masks[r], 42, train=False, legal=False))
                xs.append(read_windows(vols[r], cs, config.RAPTOR_RES))
            x = torch.stack(xs, 0)
            b, kk = x.shape[:2]
            x = ((x.flatten(0, 1) - mean) / std).view(b, kk, *x.shape[2:])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(x.cuda(), chunk=config.RAPTOR_ENCODE_CHUNK).float()
            out.append(torch.sigmoid(o).cpu().numpy())
    return macro_auc(gold_y, np.concatenate(out))[0], changed


a_ref, _ = score(False)
a_leg, n_changed = score(True)
print(f"C6 gold, reference windows: {a_ref:.4f} (recorded 0.9068)")
print(f"C6 gold, legal windows:     {a_leg:.4f} | gold studies whose 42-window bag changed: {n_changed}/58")
print(f"delta {a_leg - a_ref:+.4f} -- sanity only; 58-gold macro noise is ~0.018")
