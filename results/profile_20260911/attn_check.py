import collections, os, sys, time
sys.path.insert(0, "/home/eason/rsna-knee/src")
os.environ.setdefault("RSNA_CORPUS_SLOTS", "64")
import tv_stub; tv_stub.install()
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
print("torch", torch.__version__, "| AOTRITON_EXPERIMENTAL =", os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"), flush=True)
print("flash available:", torch.backends.cuda.is_flash_attention_available(),
      "| enabled flash/mem_eff/math:", torch.backends.cuda.flash_sdp_enabled(),
      torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(), flush=True)

# SDPA kernels at CoAtNet-rmlp-2 stage-2-like shape: 24 images, 16 heads x 32, 24x24 tokens
q = torch.randn(24, 16, 576, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
bias = torch.randn(1, 16, 576, 576, device="cuda", dtype=torch.bfloat16, requires_grad=True)
for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
    for use_bias in (False, True):
        try:
            with sdpa_kernel(be):
                for i in range(8):
                    if i == 3:
                        torch.cuda.synchronize(); t = time.perf_counter()
                    o = F.scaled_dot_product_attention(q, q, q, attn_mask=bias if use_bias else None)
                    o.float().sum().backward()
                torch.cuda.synchronize()
            print(f"  sdpa {be.name:20s} bias={use_bias!s:5s} ok   {1e3 * (time.perf_counter() - t) / 5:7.1f} ms fwd+bwd", flush=True)
        except Exception as e:
            print(f"  sdpa {be.name:20s} bias={use_bias!s:5s} FAIL {type(e).__name__}: {str(e).splitlines()[0][:100]}", flush=True)
        q.grad = None; bias.grad = None
if os.environ.get("SDPA_ONLY") == "1":
    sys.exit(0)

import timm
from raptor import RaptorClassifier, build_raptor_backbone
print("timm", timm.__version__, "| use_fused_attn():", timm.layers.use_fused_attn(), flush=True)
bb = build_raptor_backbone()
model = RaptorClassifier(bb, f_dim=bb.num_features).cuda()
fa = collections.Counter((type(m).__name__, bool(m.fused_attn)) for m in model.modules() if hasattr(m, "fused_attn"))
print("attention modules (class, fused_attn) -> count:", dict(fa), flush=True)
print("backbone top-level:", [n for n, _ in model.backbone.named_children()], flush=True)

x = torch.randn(2, 12, 3, 384, 384, device="cuda")
y = torch.rand(2, 12, device="cuda")
lossf = nn.BCEWithLogitsLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
model.train()
def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = lossf(model(x).float(), y)
    loss.backward(); opt.step(); torch.cuda.synchronize()
for _ in range(6):
    step()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        step()
ka = prof.key_averages()
print(ka.table(sort_by="self_device_time_total", row_limit=25, max_name_column_width=60), flush=True)

T = collections.defaultdict(float); st = {}
def hk(kind, name):
    def pre(*a):
        torch.cuda.synchronize(); st[kind + name] = time.perf_counter()
    def post(*a):
        torch.cuda.synchronize(); T[kind + " " + name] += time.perf_counter() - st[kind + name]
    return pre, post
mods = [("stem", model.backbone.stem)] + [(f"stage{i}", s) for i, s in enumerate(model.backbone.stages)]
handles = []
try:
    for name, m in mods:
        fp, fo = hk("fwd", name); bp, bo = hk("bwd", name)
        handles += [m.register_forward_pre_hook(fp), m.register_forward_hook(fo),
                    m.register_full_backward_pre_hook(bp), m.register_full_backward_hook(bo)]
    for _ in range(5):
        step()
    print("per-stage ms/step (5 steps, synchronised hooks):", flush=True)
    for name, _ in mods:
        f, b = 1e3 * T["fwd " + name] / 5, 1e3 * T["bwd " + name] / 5
        print(f"  {name:7s} fwd {f:7.1f}  bwd {b:7.1f}  bwd/fwd {b / max(f, 1e-9):4.2f}", flush=True)
except Exception as e:
    print("stage hooks failed:", type(e).__name__, str(e).splitlines()[0][:150], flush=True)
finally:
    for h in handles:
        h.remove()
