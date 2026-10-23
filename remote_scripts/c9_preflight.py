"""C9 preflight on the R9700: build the DINOv2 ViT-B arm, load weights, time steps.
No training, no checkpoint."""
import os, sys, time
sys.path.insert(0, "/home/eason/rsna-knee/src")
os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")
import tv_stub; tv_stub.install()
import config
config.RAPTOR_ARCH = "vit_base_patch14_dinov2"      # before raptor is imported
import numpy as np, torch, torch.nn as nn
from raptor import RaptorClassifier, build_raptor_backbone, ARCH_WEIGHTS
print("ARCH_WEIGHTS entry:", ARCH_WEIGHTS.get(config.RAPTOR_ARCH))
bb = build_raptor_backbone()
pe = getattr(bb, "patch_embed", None)
print("backbone:", type(bb).__name__, "| num_features:", bb.num_features,
      "| dynamic_img_pad:", getattr(pe, "dynamic_img_pad", None),
      "| dynamic_img_size:", getattr(bb, "dynamic_img_size", None))
model = RaptorClassifier(bb, f_dim=bb.num_features).cuda()
try:
    import r9700_opt
    if r9700_opt.is_r9700():
        r9700_opt.optimize_model_for_r9700(model)
except Exception as e:
    print("[R9700 Opt] note:", e)
n_dw = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d)
           and m.groups == m.in_channels and m.groups > 1)
print("depthwise convs:", n_dw)
fa = sorted({(type(m).__name__, bool(m.fused_attn)) for m in model.modules() if hasattr(m, "fused_attn")})
print("attention modules (class, fused_attn):", fa)
with torch.no_grad():
    t = bb.patch_embed(torch.zeros(1, 3, 384, 384, device="cuda"))
print("patch tokens for a 384 input:", tuple(t.shape), "(28x28=784 expected after padding to 392)")
x = torch.randn(2, 12, 3, 384, 384, device="cuda")
y = torch.rand(2, 12, device="cuda")
lossf = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)
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
    torch.cuda.synchronize()
    if i >= 4:
        ts.append(time.perf_counter() - t0)
step = float(np.median(ts))
print(f"median step {step * 1e3:.1f} ms | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | non-finite grad steps {nonfinite}/14")
print(f"=> train part {step * (4349 // 2) / 60:.1f} min/epoch, ~{step * (4349 // 2) * 16 / 3600:.2f} h for 16 epochs (+ per-epoch eval)")
print("references: C6 26.9 min/epoch (7.2 h); C8 ~6 min/epoch (1.6 h)")
