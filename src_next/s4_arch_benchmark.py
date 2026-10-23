"""S4 design review, part 3 (LOG #522): measured cost of PF-3D candidates on the R9700 (synthetic input,
bf16 autocast, NativeDW, fused AdamW -- the PF-v1 training path). Run on R9700 from src/.

  A  PF-v1 baseline      ConvNeXt-S + per-diagnosis attention MIL, B=2, K=12 windows (unordered)
  B  ordered sequence    same encoder; K=20 central-sagittal windows IN SLICE ORDER + learned slot/position
                         embedding + 2-layer Transformer over the sequence, then attention pooling
  B' ordered, K=12       contiguous random 12-window crop of B's sequence (cheaper training)
  C  thick window        same as A but each window stacks 7 rows (c-3..c+3) through an inflated stem
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")
import tv_stub; tv_stub.install()
import torch, torch.nn as nn
import config
from raptor import RaptorClassifier, build_raptor_backbone
import r9700_opt

dev = "cuda"
ARCH = "convnext_small.fb_in22k_ft_in1k_384"
RES = 384


class OrderedHead(nn.Module):
    def __init__(self, enc, f_dim, k_max=20, layers=2):
        super().__init__()
        self.backbone = enc
        self.norm = nn.LayerNorm(f_dim)
        self.pos = nn.Parameter(torch.zeros(1, k_max, f_dim)); nn.init.trunc_normal_(self.pos, std=0.02)
        el = nn.TransformerEncoderLayer(f_dim, 8, 2 * f_dim, dropout=0.1, batch_first=True, norm_first=True)
        self.seq = nn.TransformerEncoder(el, layers)
        self.att = nn.Sequential(nn.Linear(f_dim, 256), nn.Tanh(), nn.Linear(256, 1))
        self.cls = nn.Linear(f_dim, 1)

    def forward(self, x, pos_idx):          # x [B,K,3,H,W], pos_idx [B,K] positions in the 20-slot sequence
        b, k = x.shape[:2]
        h = self.norm(self.backbone(x.flatten(0, 1)).view(b, k, -1))
        h = self.seq(h + self.pos[0, pos_idx])
        a = torch.softmax(self.att(h), 1)
        return (a * self.cls(h)).sum(1)


def inflate_stem(enc, c_in=7):
    conv = enc.stem[0]
    new = nn.Conv2d(c_in, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(1, keepdim=True).repeat(1, c_in, 1, 1) * 3 / c_in)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    enc.stem[0] = new
    return enc


def bench(name, model, make_batch, steps=10, warm=5):
    model = model.to(dev)
    r9700_opt.optimize_model_for_r9700(model, verbose=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    torch.cuda.reset_peak_memory_stats(); ts = []
    for i in range(warm + steps):
        x, extra = make_batch()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, *extra) if extra else model(x)
            loss = out.float().pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize()
        if i >= warm:
            ts.append(time.perf_counter() - t0)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    n_head = n_par - sum(p.numel() for p in model.backbone.parameters()) / 1e6
    step = sorted(ts)[len(ts) // 2]
    print(f"{name:22s} params {n_par:6.1f} M (head {n_head:5.2f} M) | step {step * 1000:6.0f} ms | "
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:5.1f} GiB | epoch(2175 steps) ~{step * 2175 / 60:5.1f} min",
          flush=True)
    del model, opt; torch.cuda.empty_cache()


B = 2
bb = build_raptor_backbone(ARCH)
bench("A  PF-v1 MIL K=12", RaptorClassifier(bb, f_dim=bb.num_features),
      lambda: (torch.rand(B, 12, 3, RES, RES, device=dev), None))
bb = build_raptor_backbone(ARCH)
bench("B  ordered K=20", OrderedHead(bb, bb.num_features),
      lambda: (torch.rand(B, 20, 3, RES, RES, device=dev), (torch.arange(20, device=dev).repeat(B, 1),)))
bb = build_raptor_backbone(ARCH)
bench("B' ordered crop K=12", OrderedHead(bb, bb.num_features),
      lambda: (torch.rand(B, 12, 3, RES, RES, device=dev), (torch.arange(12, device=dev).repeat(B, 1) + 4,)))
bb = inflate_stem(build_raptor_backbone(ARCH), 7)
bench("C  thick 7-row K=12", RaptorClassifier(bb, f_dim=bb.num_features),
      lambda: (torch.rand(B, 12, 7, RES, RES, device=dev), None))
