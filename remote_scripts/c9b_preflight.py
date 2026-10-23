"""C9b preflight on the R9700: DINOv2 ViT-B with layer-wise LR decay.
Builds the arm and the real optimiser/scheduler through corpus.build_param_groups,
checks the per-depth LR table and the OneCycle warm-up, times steps. No checkpoint."""
import os, sys, time
sys.path.insert(0, "/home/eason/rsna-knee/src")
os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")
import tv_stub; tv_stub.install()
import config
config.RAPTOR_ARCH = "vit_base_patch14_dinov2"        # before raptor is imported
config.RAPTOR_LR_BACKBONE = 5e-6
config.RAPTOR_LLRD = 0.75
import numpy as np, torch, torch.nn as nn
from raptor import RaptorClassifier, build_raptor_backbone
from corpus import build_param_groups

bb = build_raptor_backbone()
model = RaptorClassifier(bb, f_dim=bb.num_features).cuda()
groups, max_lrs = build_param_groups(model, config.RAPTOR_LR_BACKBONE, config.RAPTOR_LR_HEAD,
                                     config.RAPTOR_LLRD)
n_params = sum(p.numel() for g in groups for p in g["params"])
n_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"groups: {len(groups)} ({len(groups) - 1} backbone depths + head) | "
      f"params covered {n_params:,} of {n_model:,} trainable")
assert n_params == n_model, "some parameters are in no group (or in two)"
ids = [id(p) for g in groups for p in g["params"]]
assert len(ids) == len(set(ids)), "a parameter appears in more than one group"
for i, (g, lr) in enumerate(zip(groups, max_lrs)):
    tag = "head" if i == len(groups) - 1 else f"depth {i:2d}"
    if i in (0, 1, len(groups) // 2, len(groups) - 2, len(groups) - 1):
        print(f"  {tag}: max_lr {lr:.3e}  ({sum(p.numel() for p in g['params']):,} params)")

opt = torch.optim.AdamW(groups, weight_decay=0.02, fused=True)
steps_per, epochs = 4349 // 2, 16
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lrs, total_steps=epochs * steps_per,
                                            pct_start=0.15)
x = torch.randn(2, 12, 3, 384, 384, device="cuda")
y = torch.rand(2, 12, device="cuda")
lossf = nn.BCEWithLogitsLoss()
torch.cuda.reset_peak_memory_stats()
ts, nonfinite = [], 0
for i in range(14):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = lossf(model(x).float(), y)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    if torch.isfinite(gn):
        opt.step()
    else:
        nonfinite += 1
    sched.step()
    torch.cuda.synchronize()
    if i >= 4:
        ts.append(time.perf_counter() - t0)
step = float(np.median(ts))
print(f"median step {step * 1e3:.1f} ms | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | "
      f"non-finite grad steps {nonfinite}/14")
print(f"=> train part {step * steps_per / 60:.1f} min/epoch (C9 preflight was 266.2 ms / 9.6 min)")

# the peak backbone LR each depth will actually reach, vs C9 which ran every depth at 2e-5
top, emb, head = max_lrs[-2], max_lrs[0], max_lrs[-1]
print(f"peak LR reached at epoch 2.4: top block {top:.2e} ({100 * top / 2e-5:.0f}% of C9), "
      f"embedding {emb:.2e} ({100 * emb / 2e-5:.2f}% of C9), head {head:.0e} (unchanged)")
