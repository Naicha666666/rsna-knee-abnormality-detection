"""Study-window MIL over a high-resolution timm backbone (the "R" line).

Why a second architecture at all: the six-slot DINOv2 line saturated at 0.855
+/- 0.005 and every imaging-side change from v8 to v14 landed inside the 0.004
run-to-run noise (see LOG.md). The public leaderboard median is 0.898, and the
scored public pipeline documents a SINGLE CoAtNet-384 arm at 0.914. The gap is
therefore in the representation and the input resolution, not in how the slots
are carved -- which is exactly what this line changes.

Differences from the DINOv2 line, deliberately:
  * 384 px input instead of 168, on an ImageNet-12k CoAtNet instead of DINOv2-S.
  * No slot structure and no anatomical prior. A study becomes one bag of 2.5D
    windows and a per-target attention decides which windows matter, so the
    architecture cannot be blamed on our slot bookkeeping.
  * Plain weighted BCE, no pairwise ranking term and no EMA, matching the
    reference arm. Those two are re-addable as separate single-variable runs.

Everything upstream of the pixels -- physical crop, percentile normalisation,
laterality, spatial sort, the report-derived targets and their weights -- is
reused unchanged from the existing modules, so this line inherits the parts the
DINOv2 experiments already validated.
"""
from __future__ import annotations
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn

from config import (LABELS, N_LABELS, CROP_MM, RAPTOR_ARCH, RAPTOR_RES,
                    RAPTOR_SLOTS, RAPTOR_SLICES_PER_SERIES, RAPTOR_K_TRAIN,
                    RAPTOR_K_EVAL, RAPTOR_WEIGHTS_NAME, RAPTOR_DROP,
                    LAT_MIN_AGREEMENT, SLOTS, RAPTOR_EVAL_SLICES_PER_SERIES,
                    RAPTOR_USE_SWA, RAPTOR_SWA_START_FRAC, RAPTOR_ENCODE_CHUNK)
import dicom_utils as du
import model as M

R_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet statistics: the backbone was pretrained with them, and a grayscale MR
# slice replicated into three channels still has to arrive on that scale.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def resolve_timm_weights(name: str = RAPTOR_WEIGHTS_NAME) -> str | None:
    """Find the offline timm checkpoint among the mounted Kaggle datasets.

    A code competition runs with the internet off, so timm cannot fetch from the
    hub: the state_dict is uploaded as a dataset and loaded by hand.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [os.path.join(here, "data", "timm_weights", name)]
    for root in ("/kaggle/input",):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, files in os.walk(root):
            if dirpath[len(root):].count(os.sep) > 4:
                dirnames[:] = []
                continue
            if name in files:
                cands.append(os.path.join(dirpath, name))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
# Each arch needs its own offline checkpoint: a code competition has no internet, so
# timm cannot fetch weights. RadImageNet ResNet-50 keys arrive torchvision-style under
# "backbone.N" and are remapped once, offline, by scratchpad/remap (see LOG).
ARCH_WEIGHTS = {
    "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k": "coatnet_rmlp_2_rw_384_in12k_ft_in1k.pt",
    "resnet50": "resnet50_radimagenet.pt",
    "vit_base_patch14_dinov2": "dinov2_vitb14.pth",   # official facebookresearch release
    # C10 (LOG #469): timm hub state_dict, saved offline with num_classes=0
    "convnext_small.fb_in22k_ft_in1k_384": "convnext_small_in22k_ft_in1k_384.pt",
    "convnextv2_base.fcmae_ft_in22k_in1k_384": "convnextv2_base_in22k_ft_in1k_384.pt",
}


def build_raptor_backbone(arch: str = RAPTOR_ARCH):
    # The stub only matters where torchvision is broken (ROCm wheels); everywhere
    # else it is a no-op. It must never be the reason a kernel dies -- adding it
    # for the R9700 box is exactly what broke the first C5 inference kernel, whose
    # single-file build had no tv_stub module to import.
    try:
        import tv_stub
        tv_stub.install()
    except ImportError:
        pass
    import timm
    # CoAtNet/MaxViT are conv-attention hybrids: no CLS token and no
    # interpolatable position embedding, so they pool by average. The "vit"
    # substring inside "coatnet" must not route them down the ViT path.
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "eva", "beit"))
    kw = dict(pretrained=False, num_classes=0, in_chans=3)
    kw["global_pool"] = "token" if is_vit else "avg"
    if is_vit:
        kw["dynamic_img_size"] = True
        # patch-14 ViTs cannot tile 384 (384 % 14 == 6) and timm asserts on it. Pad to
        # 392 inside the patch embed instead of changing the input resolution, so a ViT
        # arm sees the same 384 windows as the CoAtNet arm and the ensemble geometry
        # check (one res across members) still holds.
        kw["dynamic_img_pad"] = True
    bb = timm.create_model(arch, **kw)
    path = resolve_timm_weights(ARCH_WEIGHTS.get(arch, RAPTOR_WEIGHTS_NAME))
    if path:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        miss, unexp = bb.load_state_dict(sd, strict=False)
        print(f"[{arch}] loaded {path} | missing={len(miss)} unexpected={len(unexp)}",
              flush=True)
        if len(miss) > 8:
            print("   first missing:", list(miss)[:8], flush=True)
    else:
        print(f"[{arch}] WARNING: no offline weights found -> RANDOM INIT", flush=True)
    return bb


class RaptorClassifier(nn.Module):
    """Per-target attention pooling over a study's windows.

    One attention distribution per finding rather than one shared distribution:
    an ACL tear and a Baker's cyst are visible on different windows, and a single
    pooled study vector would average the evidence away.
    """

    def __init__(self, backbone: nn.Module, f_dim: int, n_out: int = N_LABELS,
                 drop: float = RAPTOR_DROP):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(nn.Linear(f_dim, 256), nn.Tanh(),
                                 nn.Dropout(drop), nn.Linear(256, n_out))
        self.cls_w = nn.Parameter(torch.zeros(n_out, f_dim))
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        nn.init.trunc_normal_(self.cls_w, std=0.02)
        # RQ2 control (LOG #482): "mean" replaces the per-diagnosis attention with one
        # global average over the bag's windows (the att layer then gets no gradient).
        # Read at construction so a training script can set config.RAPTOR_POOL first.
        import config as _cfg
        self.pool = getattr(_cfg, "RAPTOR_POOL", "attention")
        assert self.pool in ("attention", "mean"), self.pool

    def encode(self, x, chunk: int | None = None):   # x [B, K, 3, H, W]
        b, k = x.shape[:2]
        flat = x.flatten(0, 1)                # [B*K, 3, H, W]
        n = flat.shape[0]
        step = chunk or n
        if step >= n:
            f = self.backbone(flat)
        else:
            # Sub-batch the flattened image dimension so a single forward call's
            # peak activation memory is bounded by `chunk`, not by B*K -- a study
            # with a big window bag (dense eval) must not scale the batch that
            # actually hits the backbone. Training never needs this (B*K stays
            # small there), so callers that never pass `chunk` keep the old,
            # single-call behaviour exactly.
            f = torch.cat([self.backbone(flat[i:i + step])
                          for i in range(0, n, step)], dim=0)
        return f.view(b, k, -1)

    def head(self, feats, kmask=None):        # feats [B, K, F]
        h = self.norm(feats)
        if self.pool == "mean":
            a = torch.zeros(h.shape[0], h.shape[1], self.cls_b.shape[0],
                            device=h.device, dtype=h.dtype)
        else:
            a = self.att(h)                   # [B, K, n_out]
        if kmask is not None:
            a = a.masked_fill(kmask.unsqueeze(-1) < 0.5, -1e4)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def forward(self, x, kmask=None, chunk: int | None = None):
        return self.head(self.encode(x, chunk=chunk), kmask)


class S4SequenceClassifier(RaptorClassifier):
    """S4 (LOG #522/#523): a 2-layer Transformer over the bag with a learned position embedding,
    then the parent's per-diagnosis attention pooling. Positions are the window centres' indices in
    the central-sagittal allow-list (T) or a fixed per-study permutation of them (K); both arms have
    exactly these parameters. The caller MUST call set_positions() before every forward -- there is
    no default, so a forgotten call fails instead of silently dropping the positions."""

    def __init__(self, backbone: nn.Module, f_dim: int, n_out: int = N_LABELS,
                 drop: float = RAPTOR_DROP, k_max: int = 20, layers: int = 2):
        super().__init__(backbone, f_dim, n_out, drop)
        self.pos_embed = nn.Parameter(torch.zeros(1, k_max, f_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        el = nn.TransformerEncoderLayer(f_dim, 8, 2 * f_dim, dropout=0.1,
                                        batch_first=True, norm_first=True)
        self.seq = nn.TransformerEncoder(el, layers, enable_nested_tensor=False)
        self._pos = None

    def set_positions(self, pos):            # pos LongTensor [B, K], values in [0, k_max)
        self._pos = pos

    def head(self, feats, kmask=None):
        if self._pos is None:
            raise RuntimeError("S4SequenceClassifier: set_positions() was not called before forward")
        pos, self._pos = self._pos.to(feats.device), None
        if pos.shape != feats.shape[:2]:
            raise RuntimeError(f"positions {tuple(pos.shape)} do not match bag {tuple(feats.shape[:2])}")
        h = self.seq(self.norm(feats) + self.pos_embed[0, pos])
        a = torch.softmax(self.att(h) if kmask is None else
                          self.att(h).masked_fill(kmask.unsqueeze(-1) < 0.5, -1e4), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b


def build_head(bb, head: str | None = None):
    """The study head for a backbone: 'mil' (default, every run before S4) or 's4seq'."""
    import config as _cfg
    head = head or getattr(_cfg, "RAPTOR_HEAD", "mil")
    if head == "s4seq":
        return S4SequenceClassifier(bb, f_dim=bb.num_features)
    if head != "mil":
        raise ValueError(f"unknown head {head!r}")
    return RaptorClassifier(bb, f_dim=bb.num_features)


def build_raptor_model(arch: str = RAPTOR_ARCH):
    bb = build_raptor_backbone(arch)
    return RaptorClassifier(bb, f_dim=bb.num_features)


# --------------------------------------------------------------------------- #
# study volume cache
# --------------------------------------------------------------------------- #
def plan_slices(n_study: int, budget_gb: float) -> int:
    """Slices per series that fit the uint8 cache budget at RAPTOR_RES."""
    per = n_study * len(RAPTOR_SLOTS) * RAPTOR_RES * RAPTOR_RES
    n = RAPTOR_SLICES_PER_SERIES
    while n > 3 and n * per > budget_gb * (1024 ** 3):
        n -= 1
    return n


def build_window_cache(ds, indices, budget_gb: float, n_workers: int = 8,
                       force_n_per: int | None = None):
    """Decode each study's RAPTOR_SLOTS series once into a uint8 window cache.

    Returns (cache [N, S, R, R] uint8, present [N, n_series] float32, n_per).
    Slices are stored per series back to back, so a 2.5D window never straddles
    two series -- a triplet spanning a contrast change is not a 2.5D view of
    anything.

    `force_n_per` bypasses the memory-budget planner: the train cache has to hold
    every study at once for repeated-epoch access, so it is budget-bound, but a
    code-competition TEST split is a handful to a few hundred studies and can
    afford far more slices per series without touching RAPTOR_CACHE_BUDGET_GB.
    """
    from concurrent.futures import ThreadPoolExecutor

    indices = np.asarray(indices)
    n = len(indices)
    n_per = force_n_per if force_n_per is not None else plan_slices(n, budget_gb)
    n_series = len(RAPTOR_SLOTS)
    cache = np.zeros((n, n_series * n_per, RAPTOR_RES, RAPTOR_RES), np.uint8)
    present = np.zeros((n, n_series), np.float32)
    print(f"raptor cache: N={n} series={n_series} slices/series={n_per} "
          f"shape={cache.shape} = {cache.nbytes / 1024 ** 3:.2f} GB", flush=True)

    jobs, tag_pos = [], {}
    for li, gi in enumerate(indices):
        uid = ds.uids[int(gi)]
        slots = ds._slots_for_study(uid)
        for pos, si in enumerate(RAPTOR_SLOTS):
            folder = slots.get(si)
            if folder:
                jobs.append((li, pos, SLOTS[si][0], folder))
                tag_pos.setdefault(folder, None)

    def _lat(folder):
        return folder, du.read_laterality_full(folder)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for folder, tp in ex.map(_lat, list(tag_pos)):
            tag_pos[folder] = tp
    agree = [(t == p) for t, p in tag_pos.values() if t and p]
    use_fb = bool(agree) and (sum(agree) / len(agree)) >= LAT_MIN_AGREEMENT
    print(f"raptor laterality: {len(tag_pos)} series, "
          f"agree {(sum(agree) / len(agree)) if agree else float('nan'):.3f} "
          f"on {len(agree)} -> fallback {'ON' if use_fb else 'OFF'}", flush=True)
    sides = {f: (t or (p if use_fb else None)) for f, (t, p) in tag_pos.items()}

    def _decode(job):
        li, pos, plane, folder = job
        vol = du.load_series_stack(folder, plane, sides.get(folder), n_per,
                                   RAPTOR_RES, CROP_MM)
        return li, pos, (vol * 255.0).clip(0, 255).astype(np.uint8)

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for li, pos, vol in ex.map(_decode, jobs):
            cache[li, pos * n_per:(pos + 1) * n_per] = vol
            present[li, pos] = 1.0
            done += 1
            if done % 2000 == 0:
                print(f"  decoded {done}/{len(jobs)} slot-series", flush=True)
    print(f"raptor cache ready: {int(present.sum())} present series / "
          f"{n * n_series}", flush=True)
    return cache, present, n_per


def window_centers(n_per: int):
    """Local centre indices inside one series block (a triplet needs both sides)."""
    return list(range(1, max(2, n_per - 1)))


def take_windows(rows: np.ndarray, present: np.ndarray, n_per: int, picks):
    """rows [B, S, R, R] uint8 -> windows [B, K, 3, R, R] float32 in [0, 1].

    `picks` is a list of (series_pos, centre) per bag element; a pick on an
    absent series yields zeros and is masked out of the attention.
    """
    b = rows.shape[0]
    k = len(picks)
    out = np.zeros((b, k, 3, rows.shape[-2], rows.shape[-1]), np.float32)
    kmask = np.zeros((b, k), np.float32)
    for j, (pos, c) in enumerate(picks):
        base = pos * n_per
        tri = rows[:, [base + c - 1, base + c, base + c + 1]]
        out[:, j] = tri.astype(np.float32) / 255.0
        kmask[:, j] = present[:, pos]
    return out, kmask


def sample_picks(n_per: int, n_series: int, k: int, rng: np.random.RandomState):
    """k random (series, centre) picks, spread over series before repeating one."""
    cs = window_centers(n_per)
    pool = [(p, c) for p in range(n_series) for c in cs]
    if not pool:
        return [(0, 1)] * k
    if k >= len(pool):
        return pool
    sel = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in sorted(sel)]


def eval_picks(n_per: int, n_series: int, k: int):
    """Deterministic, evenly spread picks for inference."""
    cs = window_centers(n_per)
    pool = [(p, c) for p in range(n_series) for c in cs]
    if len(pool) <= k:
        return pool
    idx = np.linspace(0, len(pool) - 1, k).round().astype(int)
    return [pool[i] for i in idx]


def to_tensor(x, kmask, device, do_aug=False):
    t = torch.from_numpy(x).to(device)
    if do_aug:
        t = M.augment(t)
    b, k = t.shape[:2]
    t = (t - _MEAN.to(device)) / _STD.to(device) if t.shape[2] == 3 else t
    return t, torch.from_numpy(kmask).to(device)


@torch.no_grad()
def predict_windows(model, cache, present, idx, n_per, device, batch=2,
                    k_eval: int = RAPTOR_K_EVAL):
    model.eval()
    picks = eval_picks(n_per, present.shape[1], k_eval)
    out = []
    idx = np.asarray(idx)
    for b0 in range(0, len(idx), batch):
        sel = idx[b0:b0 + batch]
        x, km = take_windows(cache[sel], present[sel], n_per, picks)
        xt, kmt = to_tensor(x, km, device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            # A dense eval pool (R2: up to 18 centres x 6 slots = 108 windows/study)
            # would otherwise flatten to batch*108 images in one backbone call --
            # this is what OOM'd on R2's first run. Chunk regardless of pool size.
            z = model(xt, kmt, chunk=RAPTOR_ENCODE_CHUNK).float()
        out.append(torch.sigmoid(z).cpu().numpy())
    return (np.concatenate(out, 0) if out
            else np.zeros((0, N_LABELS), np.float32))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _recalibrate_bn(model, cache, present, idx, n_per, n_series, device, rng,
                    n_batches: int = 30, batch: int = RAPTOR_K_TRAIN):
    """Reset BatchNorm running stats, then re-accumulate them with cumulative
    averaging (momentum=None) over forward-only passes -- the same recipe as
    torch.optim.swa_utils.update_bn, hand-rolled because our forward signature
    takes (imgs, kmask) rather than the single positional input update_bn expects.

    SWA averages WEIGHTS across epochs, but a BatchNorm layer's running mean/var
    were accumulated for the trajectory of weights that produced them, not for the
    averaged weights the model now has; skipping this step is a common SWA bug
    that silently degrades an averaged model at eval time.
    """
    bn_layers = [m for m in model.modules()
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not bn_layers:
        return
    old_momenta = {m: m.momentum for m in bn_layers}
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None
    was_training = model.training
    model.train()
    idx = np.asarray(idx)
    with torch.no_grad():
        for _ in range(n_batches):
            sel = idx[rng.choice(len(idx), size=min(batch, len(idx)), replace=False)]
            picks = sample_picks(n_per, n_series, RAPTOR_K_TRAIN, rng)
            x, km = take_windows(cache[sel], present[sel], n_per, picks)
            xt, kmt = to_tensor(x, km, device, do_aug=False)
            model(xt, kmt)
    for m in bn_layers:
        m.momentum = old_momenta[m]
    model.train(was_training)


def train_windows(cache, present, y, w, tr_idx, va_idx, y_gold_va, epochs,
                  n_per, batch, device=None, seed: int = 42, time_budget_s=None):
    """One model on the window cache. Weighted BCE only -- no rank term.

    The rank term is re-addable as a separate single-variable run; the reference
    arm this line follows has neither, and starting a new architecture with our
    DINOv2 line's extras bolted on would make a disappointing result unattributable.
    SWA is the one addition kept from day one (config.RAPTOR_USE_SWA): it is a
    documented feature of the reference arm's own recipe ("_swa" checkpoint name),
    not a DINOv2-line import, and unlike EMA it needs no extra forward pass during
    training -- only a cheap BN recalibration once at the end.
    """
    from train import macro_auc

    device = device or R_DEVICE
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model = build_raptor_model().to(device)
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.")]
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": bb_params, "lr": RAPTOR_LR_BACKBONE},
         {"params": head_params, "lr": RAPTOR_LR_HEAD}], weight_decay=0.02)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    n_series = present.shape[1]

    tr_idx = np.asarray(tr_idx)
    steps_per = max(len(tr_idx) // batch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[RAPTOR_LR_BACKBONE, RAPTOR_LR_HEAD],
        total_steps=max(epochs * steps_per, 1), pct_start=0.15)

    do_sel = va_idx is not None and len(va_idx) > 0
    if do_sel:
        va_idx = np.asarray(va_idx)
        yd = y[va_idx].astype(float).copy()
        yd[w[va_idx] <= 0] = np.nan
        yd_bin = np.where(np.isfinite(yd), (yd > 0.5).astype(float), np.nan)
    best_score, best_state = -1.0, None
    swa_state, swa_n = None, 0
    swa_start_ep = int(round(epochs * RAPTOR_SWA_START_FRAC)) if RAPTOR_USE_SWA else epochs
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr_idx))
        loss_sum, nb = 0.0, 0
        for b0 in range(0, len(tr_idx) - batch + 1, batch):
            sel = tr_idx[order[b0:b0 + batch]]
            picks = sample_picks(n_per, n_series, RAPTOR_K_TRAIN, rng)
            x, km = take_windows(cache[sel], present[sel], n_per, picks)
            xt, kmt = to_tensor(x, km, device, do_aug=True)
            yt = torch.from_numpy(y[sel]).to(device)
            wt = torch.from_numpy(w[sel]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xt, kmt)
                loss = (bce(logits, yt) * wt).sum() / wt.sum().clamp(min=1.0)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += float(loss.item())
            nb += 1
        msg = f"    epoch {ep + 1}/{epochs}  loss={loss_sum / max(nb, 1):.4f}"
        if do_sel:
            pred = predict_windows(model, cache, present, va_idx, n_per, device)
            der, _ = macro_auc(yd_bin, pred)
            gld, _ = macro_auc(y_gold_va, pred) if y_gold_va is not None else (float("nan"), None)
            msg += f"  der={der:.4f}"
            if np.isfinite(gld):
                msg += f"  gold={gld:.4f}"
            if np.isfinite(der) and der > best_score:
                best_score = der
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        if RAPTOR_USE_SWA and ep >= swa_start_ep:
            cur = model.state_dict()
            if swa_state is None:
                swa_state = {k: v.detach().clone().float()
                             for k, v in cur.items() if v.dtype.is_floating_point}
            else:
                swa_n += 1
                for k, v in swa_state.items():
                    v += (cur[k].detach().float() - v) / (swa_n + 1)
            msg += f"  [swa n={swa_n + (1 if swa_state is not None else 0)}]"
        msg += f"  [{(time.time() - t0) / 60:.1f}m]"
        print(msg, flush=True)
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"    time budget hit after epoch {ep + 1}", flush=True)
            break

    if swa_state is not None and do_sel:
        # Load the averaged weights, restoring the non-float buffers (e.g. BN's
        # num_batches_tracked, an int) from whichever state the run ends on -- only
        # the running mean/var need recomputing, and _recalibrate_bn does that next.
        swa_full = model.state_dict()
        swa_full.update({k: v.to(swa_full[k].dtype) for k, v in swa_state.items()})
        model.load_state_dict(swa_full)
        _recalibrate_bn(model, cache, present, tr_idx, n_per, n_series, device, rng)
        pred = predict_windows(model, cache, present, va_idx, n_per, device)
        der, _ = macro_auc(yd_bin, pred)
        print(f"    SWA holdout der={der:.4f} (best single epoch was {best_score:.4f})",
              flush=True)
        if np.isfinite(der) and der > best_score:
            best_score = der
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        # else fall through: best_state (a single epoch) still wins below.

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def run_raptor_submit(max_studies: int | None = None, epochs: int = None,
                      batch: int = None, out_dir: str = ".",
                      budget_gb: float = None, seed: int = 42,
                      time_budget_h: float = 6.0, do_submit: bool = True):
    """Train one study-window model on a grouped 80/20 split and predict the test set.

    A single arm, not a fold ensemble: the DINOv2 line showed 8 folds scored the
    same as 4 (LOG.md v12), and at 384 px one arm is what the runtime buys. The
    published CoAtNet arm this follows is also a single split.
    """
    from config import (RAPTOR_EPOCHS, RAPTOR_BATCH, RAPTOR_CACHE_BUDGET_GB)
    from dataset import RSNADataset
    from train import _report_hash, _subset_indices, _write_submission, macro_auc
    from sklearn.model_selection import GroupShuffleSplit

    epochs = epochs or RAPTOR_EPOCHS
    batch = batch or RAPTOR_BATCH
    budget_gb = budget_gb or RAPTOR_CACHE_BUDGET_GB
    t0 = time.time()

    ds = RSNADataset("train")
    idx = _subset_indices(ds, max_studies)
    groups = np.array([_report_hash(ds.reports[i]) for i in idx])
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    tr, va = next(gss.split(idx, groups=groups))
    n_gold = int(np.isfinite(ds.y_raw[idx[va]]).any(axis=1).sum())
    print(f"raptor: {len(idx)} studies | train={len(tr)} holdout={len(va)} "
          f"(gold in holdout={n_gold}) | arch={RAPTOR_ARCH} res={RAPTOR_RES} "
          f"ep={epochs} batch={batch} k_train={RAPTOR_K_TRAIN}", flush=True)

    cache, present, n_per = build_window_cache(ds, idx, budget_gb)
    y, w = ds.targets[idx], ds.weights[idx]
    spent = (time.time() - t0) / 3600.0
    print(f"raptor: cache built in {spent * 60:.1f} min", flush=True)

    model, best = train_windows(
        cache, present, y, w, tr, va, ds.y_raw[idx[va]], epochs, n_per, batch,
        seed=seed, time_budget_s=max(600.0, (time_budget_h - spent) * 3600.0))
    print(f"raptor: best holdout derived-AUC={best:.4f}", flush=True)

    ckpt = os.path.join(out_dir, "raptor0.pt")
    torch.save({"state_dict": model.state_dict(), "arch": RAPTOR_ARCH,
                "res": RAPTOR_RES, "n_per": n_per, "slots": list(RAPTOR_SLOTS)},
               ckpt)
    print("saved", ckpt, flush=True)

    del cache, present
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if do_submit:
        test = RSNADataset("test")
        t_idx = np.arange(len(test.uids))
        # The test split is small enough that memory is not the constraint here, so
        # it decodes far more slices per series than the train cache affords, and
        # RAPTOR_K_EVAL=999 (see config.py) then votes over essentially all of them.
        tc, tp, tn = build_window_cache(test, t_idx, budget_gb,
                                        force_n_per=RAPTOR_EVAL_SLICES_PER_SERIES)
        preds = predict_windows(model, tc, tp, t_idx, tn, R_DEVICE)
        _write_submission(test.uids, preds, os.path.join(out_dir, "submission.csv"))
    print(f"raptor finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    return model
