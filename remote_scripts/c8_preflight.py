"""C8 preflight on the R9700: build the arm, load RadImageNet weights, time a few
steps. No training, no checkpoint."""
import os, sys, time
sys.path.insert(0, "/home/eason/rsna-knee/src")
os.environ.setdefault("RSNA_DATA_ROOT", "/home/eason/rsna-knee/data")
os.environ.setdefault("RSNA_CORPUS_DIR", "/home/eason/rsna-knee/data/corpus64")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")
import tv_stub; tv_stub.install()
import config
config.RAPTOR_ARCH = "resnet50"
import numpy as np, torch, torch.nn as nn
from raptor import RaptorClassifier, build_raptor_backbone, ARCH_WEIGHTS
print("ARCH_WEIGHTS:", ARCH_WEIGHTS)
bb = build_raptor_backbone()            # default must now be resnet50
print("backbone:", type(bb).__name__, "| num_features:", bb.num_features)
model = RaptorClassifier(bb, f_dim=bb.num_features).cuda()
try:
    import r9700_opt
    if r9700_opt.is_r9700():
        r9700_opt.optimize_model_for_r9700(model)
except Exception as e:
    print("[R9700 Opt] note:", e)
n_dw = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d)
           and m.groups == m.in_channels and m.groups > 1)
print("depthwise convs in this backbone:", n_dw, "(ResNet-50 has none -> NativeDW swap is a no-op)")
x = torch.randn(2, 12, 3, 384, 384, device="cuda")
y = torch.rand(2, 12, device="cuda")
lossf = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-6, fused=True)
ts = []
for i in range(12):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = lossf(model(x).float(), y)
    loss.backward(); opt.step(); torch.cuda.synchronize()
    if i >= 4:
        ts.append(time.perf_counter() - t0)
step = float(np.median(ts))
print(f"median step {step * 1e3:.1f} ms | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(f"=> projected {step * (4349 // 2) / 60:.1f} min/epoch, {step * (4349 // 2) * 16 / 3600:.2f} h for 16 epochs")
print("C6 reference: 26.9 min/epoch, 7.2 h")
