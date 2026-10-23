"""S4 pre-training unit checks (LOG #524). Run on R9700 from src/. No training.
  1. default head unchanged: build_head() with RAPTOR_HEAD unset -> RaptorClassifier
  2. T and K: identical parameter count and names
  3. same seed -> identical contiguous crops in both arms (only positions differ)
  4. positions: T = true allow-list index (contiguous run); K = fixed per-study permutation, deterministic
  5. forward without set_positions fails; mismatched shape fails
  6. a K forward is invariant to permuting the bag together with its positions (order carries nothing
     beyond the positions), and T vs K positions give different outputs
"""
import tv_stub; tv_stub.install()
import numpy as np, torch
import config
from raptor import build_raptor_backbone, build_head, RaptorClassifier, S4SequenceClassifier
import corpus

ARCH = "convnext_small.fb_in22k_ft_in1k_384"
ALLOW = list(range(4, 14)) + list(range(20, 30))
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m) or c
res = []

res.append(ok(type(build_head(build_raptor_backbone(ARCH))) is RaptorClassifier, "default head is the MIL RaptorClassifier"))

config.CORPUS_CENTRE_ALLOW = ALLOW; config.RAPTOR_HEAD = "s4seq"; config.CORPUS_TRAIN_CONTIGUOUS = True
mT, mK = build_head(build_raptor_backbone(ARCH)), build_head(build_raptor_backbone(ARCH))
nT = sum(p.numel() for p in mT.parameters()); nK = sum(p.numel() for p in mK.parameters())
res.append(ok(isinstance(mT, S4SequenceClassifier) and nT == nK and
              [n for n, _ in mT.named_parameters()] == [n for n, _ in mK.named_parameters()],
              f"T and K identical parameters ({nT / 1e6:.2f} M)"))

mask = np.ones(64, np.uint8)
crops = {}
for order in ("true", "shuffled"):
    config.S4_ORDER = order
    rng = np.random.RandomState(42)
    crops[order] = [corpus.centers_from_mask(mask, 12, True, rng) for _ in range(50)]
res.append(ok(crops["true"] == crops["shuffled"], "same seed -> identical crops in T and K (50 draws)"))
c0 = crops["true"][0]
f_, s_ = c0[:6], c0[6:]
res.append(ok(len(c0) == 12 and all(4 <= c <= 13 for c in f_) and all(20 <= c <= 29 for c in s_)
              and all(b - a == 1 for a, b in zip(f_, f_[1:])) and [c - 16 for c in s_] == f_,
              f"crop = the same 6 contiguous depths in SagF and SagS: {c0}"))
starts = {crops["true"][i][0] for i in range(50)}
res.append(ok(starts <= set(range(4, 9)) and len(starts) > 1, f"slab start varies over depths 4..8: {sorted(starts)}"))

uid = "1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191"
config.S4_ORDER = "true"; pT = corpus.s4_positions(uid, c0)
config.S4_ORDER = "shuffled"; pK = corpus.s4_positions(uid, c0); pK2 = corpus.s4_positions(uid, c0)
perm = corpus.s4_perm(uid)
res.append(ok(list(pT) == [ALLOW.index(c) for c in c0], f"T positions = true indices {list(pT)}"))
res.append(ok(list(pK) == list(perm[pT]) and list(pK) == list(pK2) and sorted(perm) == list(range(20)),
              f"K positions = fixed permutation {list(pK)} (deterministic)"))
res.append(ok(list(corpus.s4_perm(uid + "x")) != list(perm), "different study -> different permutation"))

dev = "cuda"
m = mK.to(dev).eval()
x = torch.rand(1, 12, 3, 384, 384, device=dev)
try:
    m(x); res.append(ok(False, "forward without positions must fail"))
except RuntimeError:
    res.append(ok(True, "forward without set_positions fails"))
try:
    m.set_positions(torch.zeros(1, 5, dtype=torch.long)); m(x); res.append(ok(False, "shape mismatch must fail"))
except RuntimeError:
    res.append(ok(True, "positions of the wrong shape fail"))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    p = torch.from_numpy(pK).long()[None]
    m.set_positions(p); o1 = m(x).float()
    sh = torch.randperm(12)
    m.set_positions(p[:, sh]); o2 = m(x[:, sh]).float()
    m.set_positions(torch.from_numpy(pT).long()[None]); o3 = m(x).float()
res.append(ok(torch.allclose(o1, o2, atol=2e-2), f"bag+positions permuted together -> same output (max diff {float((o1 - o2).abs().max()):.4f})"))
res.append(ok(not torch.allclose(o1, o3, atol=1e-3), f"T vs K positions -> different output (max diff {float((o1 - o3).abs().max()):.4f})"))
print(f"\n{sum(res)}/{len(res)} checks passed")
raise SystemExit(0 if all(res) else 1)
