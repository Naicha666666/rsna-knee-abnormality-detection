"""S4 smoke diagnosis (LOG #524): the 8-study x 1-epoch smoke ended at loss 9.6. Compare the PF-weighted
BCE of the S4 head and the PF-v1 MIL head (a) at initialisation and (b) over a short training run with
the real optimiser / OneCycle code path on the same studies. No checkpoint is written."""
import os
os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")
import tv_stub; tv_stub.install()
import numpy as np, torch, torch.nn as nn
import config
config.RAPTOR_ARCH = "convnext_small.fb_in22k_ft_in1k_384"
config.CORPUS_CENTRE_ALLOW = list(range(4, 14)) + list(range(20, 30))
config.CORPUS_TRAIN_CONTIGUOUS = True
config.S4_ORDER = "true"
config.SOFT_LABEL_PARQUETS = ["labels_kimi_soft.parquet", "labels_qwen_soft.parquet", "labels_grok_soft.parquet", "labels_gemma4_soft.parquet"]
import corpus
from raptor import build_raptor_backbone, build_head
from dataset import RSNADataset
import r9700_opt

vols, masks, ids = corpus.open_corpus(); row_of = {u: i for i, u in enumerate(ids)}
ds = RSNADataset("train")
J = list(config.LABELS).index("PF OA")
gold = set(u for i, u in enumerate(ds.uids) if np.isfinite(ds.y_raw[i]).all())
tr = [u for u in ds.uids if u in row_of and u not in gold][:400]
tgt = {u: ds.targets[i][J] for i, u in enumerate(ds.uids) if u in set(tr)}
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
pw = torch.tensor(1.6, device="cuda")


def run(head, steps, lr_b=2e-5, lr_h=1e-3):
    config.RAPTOR_HEAD = head
    torch.manual_seed(42); rng = np.random.RandomState(42)
    m = build_head(build_raptor_backbone()).cuda(); r9700_opt.optimize_model_for_r9700(m, verbose=False)
    groups, max_lrs = corpus.build_param_groups(m, lr_b, lr_h)
    opt = torch.optim.AdamW(groups, weight_decay=0.02, fused=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lrs, total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); out = []
    for s in range(steps):
        sel = [tr[(2 * s + j) % len(tr)] for j in range(2)]
        xs, ps = [], []
        for u in sel:
            cs = corpus.centers_from_mask(masks[row_of[u]], 12, True, rng)
            xs.append(corpus.read_windows(vols[row_of[u]], cs, 384)); ps.append(corpus.s4_positions(u, cs))
        x = torch.stack(xs); b, k = x.shape[:2]
        x = ((x.flatten(0, 1) - mean) / std).view(b, k, *x.shape[2:]).cuda()
        if hasattr(m, "set_positions"):
            m.set_positions(torch.from_numpy(np.stack(ps)).long())
        y = torch.tensor([tgt[u] for u in sel], device="cuda")
        m.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = m(x).float()[:, J]
            loss = lossf(logit, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 3.0); opt.step(); sch.step()
        out.append((float(loss), float(logit.abs().max())))
    return out


for head in ("mil", "s4seq"):
    for steps in (4, 100):
        o = run(head, steps)
        l = np.array([a for a, _ in o]); g = np.array([b for _, b in o])
        print(f"{head:6s} {steps:3d} steps | loss first {l[0]:.3f} | mean last-quarter {l[-max(1, steps // 4):].mean():.3f} "
              f"| max |logit| {g.max():.1f} (last {g[-1]:.1f})", flush=True)
        torch.cuda.empty_cache()
