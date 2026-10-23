"""Pre-decoded study corpus (memmap) + a matching test-time volume builder.

The R line decoded DICOMs into an in-RAM cache on every run, which cost 48-72
minutes per kernel AND imposed a memory ceiling that forced the depth-vs-breadth
tradeoff R2/R3 measured (LOG.md). The reference pipeline has neither problem
because it decodes the corpus ONCE, offline, into a uint8 memmap published as
two public Kaggle datasets, and streams per-study reads from disk:

    dreaddevelopment/knee-raptor-corpus       all_vols.npy    15.9 GB
    dreaddevelopment/knee-raptor-corpus-ext   extra_vols.npy   6.0 GB
    -> one (4407, 44, 336, 336) uint8 array, 100% of our train.csv studies
    C4 64-slice shards (self-built, same crop/span/percentile, more slices):
    easonyanyan/rsna-knee-corpus64-p0 / p1    vols64_p{0,1}.npy  ~15.9 GB each
    -> one (4407, 64, 336, 336) uint8 array

CRITICAL: the corpus covers TRAIN only. A code competition's hidden test set has
to be decoded at inference time, and it must be decoded EXACTLY the way the
corpus was, or the model sees a different distribution than it trained on. The
constants below were recovered from the reference's own inference notebook plus
the corpus masks, and `build_test_volume` reproduces that pipeline:

  * 5 slots. C1-C3 used [12, 10, 8, 6, 8] = 44 (the published memmap). C4 uses
    the 64-slice layout [18, 14, 12, 8, 12] documented by the 0.924 inference
    kernel -- same proportions, more coverage. Test-time volumes MUST use the
    same layout the model trained on.
  * span 0.06-0.94 of each series, NOT a central crop. Their comment: the
    collateral ligaments and lateral meniscus live in the peripheral slices a
    central crop throws away. (Our own pipeline uses the central 60%.)
  * per-SLOT percentile 2-98 over the picked slices, not per-volume 1-99.
  * centre crop CROP_MM=140 at the series' pixel spacing, resize to 336.
  * NO laterality normalisation. The reference simply does not flip; our DINOv2
    line spent real effort on tag+IPP laterality. Matching the corpus means not
    flipping either -- a deliberate deviation, flagged for a later A/B.
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np

# ---- corpus geometry --------------------------------------------------------
# C1-C3 trained on the published 44-slice memmap (12/10/8/6/8). The 0.924
# single-arm weights were trained on 64 slices at the same proportions
# (18/14/12/8/12) -- that corpus was never published, only the checkpoint.
# C4 rebuilds it with the same crop/span/percentile rules, only n_slices
# changes. 4407 x 64 x 336 x 336 uint8 = 31.8 GB, so it is dumped as TWO
# shards (~15.9 GB each) to fit Kaggle's 20 GB working dir, then mounted
# like the original two-part 44-slice corpus.
CORPUS_IMG = 336
CORPUS_CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.06, 0.94
PCT_LO, PCT_HI = 2.0, 98.0
# (plane, fluid_flag, n_slices); fluid_flag -1 means "any series in this plane"
CORPUS_SLOTS_44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
                   ("Coronal", 0, 6), ("Axial", -1, 8)]
CORPUS_SLOTS_64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
                   ("Coronal", 0, 8), ("Axial", -1, 12)]
# Which geometry this checkout is pointed at. RSNA_CORPUS_SLOTS=44|64 overrides it,
# so the two lines can share one file: C4/C5 run at 64 (own dumped shards), C2/C3/C6
# at 44 (the published memmap). Getting it wrong is NOT silent -- open_corpus below
# hard-exits when the memmap's depth disagrees with CORPUS_D.
_SLOT_CHOICE = os.environ.get("RSNA_CORPUS_SLOTS", "44")
CORPUS_SLOTS = CORPUS_SLOTS_64 if _SLOT_CHOICE == "64" else CORPUS_SLOTS_44
CORPUS_D = sum(s[2] for s in CORPUS_SLOTS)
assert CORPUS_D in (44, 64)
try:
    from config import CORPUS_LEGAL_WINDOWS as LEGAL_WINDOWS
except Exception:          # corpus.py used without the project config
    LEGAL_WINDOWS = False


def _find(*names, root=None):
    """First match for any of `names` under corpus search roots."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots = []
    if root:
        roots.append(root)
    env = os.environ.get("RSNA_CORPUS_DIR")
    if env:
        roots.append(env)
    roots.append(os.path.join(here, "data", "corpus64"))
    if os.path.isdir("/kaggle/input"):
        roots.append("/kaggle/input")
    for name in names:
        for r in roots:
            if not os.path.isdir(r):
                continue
            direct = os.path.join(r, name)
            if os.path.isfile(direct):
                return direct
            for dirpath, dirnames, files in os.walk(r):
                if dirpath[len(r):].count(os.sep) > 6:
                    dirnames[:] = []
                    continue
                if "competition" in dirpath.lower():
                    continue
                if name in files:
                    return os.path.join(dirpath, name)
    return None


class TwoPartVols:
    """Two memmap parts as one (N, D, H, W) array.

    Both stay memory-mapped and are never concatenated: every read is a single
    study, so materialising 32 GB would buy nothing. Row order is part 0 then
    part 1, matching how the id and mask files concatenate -- that ordering is
    the contract between volumes, masks and ids, so none may be sorted alone.
    """

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


def open_corpus():
    """Return (vols, masks [N,D] uint8, ids [N] str) or raise with what to attach.

    Prefers the C4 64-slice shards (vols64_p0/p1.npy). Falls back to the
    published 44-slice memmap only when CORPUS_D is 44, so a 64-slice train
    kernel cannot silently read the wrong geometry.
    """
    a = _find("vols64_p0.npy")
    b = _find("vols64_p1.npy")
    if a and b:
        am, bm = _find("masks64_p0.npy"), _find("masks64_p1.npy")
        ai, bi = _find("ids64_p0.npy"), _find("ids64_p1.npy")
        if not all((am, bm, ai, bi)):
            raise SystemExit(
                "64-slice vols found but masks/ids missing\n"
                f"  masks {am} {bm} | ids {ai} {bi}")
        vols = TwoPartVols(np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r"))
        masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
        ids = np.concatenate([np.load(ai, allow_pickle=True).astype(str),
                              np.load(bi, allow_pickle=True).astype(str)])
    else:
        av, bv = _find("all_vols.npy"), _find("extra_vols.npy")
        am, bm = _find("all_masks.npy"), _find("extra_masks.npy")
        ai, bi = _find("all_ids.npy"), _find("extra_ids.npy")
        if not all((av, bv, am, bm, ai, bi)):
            raise SystemExit(
                "corpus not found -- attach the 64-slice shards "
                "(vols64_p0.npy / vols64_p1.npy) or the published 44-slice "
                "dreaddevelopment/knee-raptor-corpus + -ext\n"
                f"  vols64_p0={a} vols64_p1={b} all_vols={av} extra_vols={bv}")
        vols = TwoPartVols(np.load(av, mmap_mode="r"), np.load(bv, mmap_mode="r"))
        masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
        ids = np.concatenate([np.load(ai, allow_pickle=True).astype(str),
                              np.load(bi, allow_pickle=True).astype(str)])
    if vols.shape[1] != CORPUS_D:
        raise SystemExit(
            f"corpus depth {vols.shape[1]} != CORPUS_D {CORPUS_D} -- "
            f"mounted the wrong slice layout")
    print(f"corpus: {vols.shape} uint8 memmap | masks {masks.shape} | ids {len(ids)}",
          flush=True)
    return vols, masks, ids



def _specialist_meta():
    """Window policy + trained labels recorded in every checkpoint (LOG #490), so inference
    can restore them; both are None for every run before the diagnosis specialist."""
    import config as _c
    allow = getattr(_c, "CORPUS_CENTRE_ALLOW", None)
    labs = getattr(_c, "CORPUS_TRAIN_LABELS", None)
    return {"centre_allow": sorted(int(c) for c in allow) if allow else None,
            "train_labels": list(labs) if labs else None,
            # S4 (LOG #523); "mil" / None / False for every earlier run
            "head": getattr(_c, "RAPTOR_HEAD", "mil"),
            "s4_order": getattr(_c, "S4_ORDER", None),
            "train_contiguous": bool(getattr(_c, "CORPUS_TRAIN_CONTIGUOUS", False))}


S4_PERM_TAG = "S4-perm-v1:"


def s4_perm(uid: str, n: int = 20):
    """The fixed per-study permutation of the n allow-list positions used by the S4 control arm K
    (LOG #523 §9.1): seeded by SHA-256 of a version tag + StudyInstanceUID, so training, gold
    evaluation and hidden-test inference all derive the same permutation from the uid alone."""
    import hashlib
    seed = int.from_bytes(hashlib.sha256((S4_PERM_TAG + str(uid)).encode()).digest()[:8], "little")
    return np.random.default_rng(seed).permutation(n)


def s4_positions(uid, centres):
    """Position indices for an S4 bag, or None when the head is not 's4seq'. T: index of each centre in
    the sorted allow-list; K: the study's fixed permutation of those indices. A centre outside the
    allow-list (a study whose allowed rows were all empty) maps to the nearest allowed index."""
    import config as _c
    if getattr(_c, "RAPTOR_HEAD", "mil") != "s4seq":
        return None
    allow = sorted(int(c) for c in (_c.CORPUS_CENTRE_ALLOW or []))
    if not allow:
        raise SystemExit("S4 needs CORPUS_CENTRE_ALLOW (the position vocabulary)")
    a = np.asarray(allow)
    idx = np.array([int(np.abs(a - int(c)).argmin()) for c in centres])
    order = getattr(_c, "S4_ORDER", None)
    if order == "true":
        return idx
    if order == "shuffled":
        return s4_perm(uid, len(allow))[idx]
    raise SystemExit(f"S4_ORDER must be 'true' or 'shuffled', got {order!r}")

def centers_from_mask(mask, k: int, train: bool, rng=None, legal=None):
    """`k` window centres over the valid span; a centre needs both neighbours.

    Training samples with replacement from the legal set (so every window is a
    fresh draw across epochs); evaluation spreads deterministically. Note this
    follows the reference exactly, including that a window MAY straddle a slot
    boundary: legality is judged on the valid span [lo, hi], not on which slot a
    position belongs to. Our own R line forbade straddling; matching the recipe
    that scored 0.914 comes first, and the no-straddle variant is a later A/B.

    legal=True (or CORPUS_LEGAL_WINDOWS when legal is None) keeps only centres whose
    rows c-1, c, c+1 are all non-empty and in the same slot. The audit found 12.9% of
    training windows straddle a slot (half of them a plane change) and 3.0% are all
    zero, and no attention mask is passed (LOG #377). If a study has no such centre
    the reference set is used unchanged.
    """
    valid = np.where(np.asarray(mask) > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, CORPUS_D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, CORPUS_D - 2))]
    if (LEGAL_WINDOWS if legal is None else legal):
        m = np.asarray(mask) > 0
        slot_of = np.repeat(np.arange(len(CORPUS_SLOTS)), [s[2] for s in CORPUS_SLOTS])
        strict = [c for c in cs if m[c - 1] and m[c] and m[c + 1]
                  and slot_of[c - 1] == slot_of[c + 1]]
        if strict:
            cs = strict
    # Diagnosis specialist (LOG #490): keep only the allowed centres. Read at call time so
    # a script can set config.CORPUS_CENTRE_ALLOW after this module was imported.
    import config as _cfg_c
    _allow = getattr(_cfg_c, "CORPUS_CENTRE_ALLOW", None)
    if _allow:
        _kept = [c for c in cs if c in set(_allow)]
        if _kept:
            cs = _kept
    if train and getattr(_cfg_c, "CORPUS_TRAIN_CONTIGUOUS", False):
        # S4 (LOG #523, amended #524): a contiguous slab in slice order, random start. When the
        # allowed centres form several equal-length runs of consecutive rows (the central SagF and
        # SagS bands sweep the SAME medio-lateral range in two contrasts), take the same depth range
        # from every run, so the bag is one anatomically contiguous slab seen in both series. A study
        # with fewer usable centres falls back to one run over the list, repeating centres in order.
        rng = rng or np.random
        runs = np.split(np.asarray(cs), np.where(np.diff(cs) != 1)[0] + 1)
        if len(runs) > 1 and k % len(runs) == 0 and len({len(r) for r in runs}) == 1 \
                and len(runs[0]) >= k // len(runs):
            m = k // len(runs)
            d0 = int(rng.randint(0, len(runs[0]) - m + 1))
            return [int(c) for r in runs for c in r[d0:d0 + m]]
        if len(cs) >= k:
            s0 = int(rng.randint(0, len(cs) - k + 1))
            return cs[s0:s0 + k]
        idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
        return [cs[i] for i in idx]
    if train:
        rng = rng or np.random
        pool = (cs * (k // len(cs) + 1))
        rng.shuffle(pool)
        return pool[:k]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def read_windows(vol_row, centers, res: int):
    """(K, 3, res, res) float32 in [0,1] from one study's (D, 336, 336) uint8 row.

    A window is three physically consecutive slices stacked as RGB, matching the
    reference's 2.5D definition.
    """
    import torch
    import torch.nn.functional as F

    vol = np.asarray(vol_row)
    d = vol.shape[0]
    out = []
    for c in centers:
        c = max(1, min(int(c), d - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        out.append(t)
    return torch.stack(out, 0)


# --------------------------------------------------------------------------- #
# test-time volume construction (must mirror how the corpus was built)
# --------------------------------------------------------------------------- #
def _pick_series_for_slot(rows, plane, fluid, used):
    """Prefer the requested fluid flag, but fall back to ANY unused series in the
    plane rather than leaving the slot empty -- which is why the reference's slot
    coverage reaches 100% on the fluid slots where our exact matching got 94-96%."""
    cands = [r for r in rows
             if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _order_series_files(folder):
    """(sorted file paths, median pixel spacing) for one series, ordered through-plane."""
    import pydicom

    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not files:
        return [], 1.0
    keyed, spacings = [], []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            # project the slice origin onto the slice normal: the true
            # through-plane coordinate, correct for oblique acquisitions where a
            # single fixed axis is not.
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        try:
            ps = float(ds.PixelSpacing[0])
        except Exception:
            ps = None
        if ps:
            spacings.append(ps)
        keyed.append((k, f))
    med_ps = float(np.median(spacings)) if spacings else 1.0
    if any(k is None for k, _ in keyed):
        return [f for _, f in keyed], med_ps      # arbitrary order beats dropping
    return [f for _, f in sorted(keyed, key=lambda t: t[0])], med_ps


def _read_px(path):
    import pydicom

    ds = pydicom.dcmread(path, force=True)
    a = ds.pixel_array.astype(np.float32)
    a = a * float(getattr(ds, "RescaleSlope", 1) or 1) \
        + float(getattr(ds, "RescaleIntercept", 0) or 0)
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = 0.0
    return a, ps


def _crop_resize(a, ps):
    """Centre-crop CORPUS_CROP_MM of tissue at this slice's spacing, then resize."""
    import cv2

    h, w = a.shape
    cpx = int(round(CORPUS_CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(a, (CORPUS_IMG, CORPUS_IMG), interpolation=cv2.INTER_AREA)


def build_test_volume(sid, rows, series_root):
    """(vol [D,336,336] uint8, mask [D] uint8) for one study, corpus-identical."""
    vol = np.zeros((CORPUS_D, CORPUS_IMG, CORPUS_IMG), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in CORPUS_SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = _order_series_files(
            os.path.join(series_root, sid, r["SeriesInstanceUID"]))
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * SPAN_LO), int(n * SPAN_HI) - 1
        hi = max(hi, lo)
        picks = (np.linspace(lo, hi, k).round().astype(int) if n > 1
                 else np.zeros(k, dtype=int))
        arrs, pss = [], []
        for p in picks:
            try:
                a, ps = _read_px(files[min(int(p), n - 1)])
            except Exception:
                a, ps = None, med_ps
            arrs.append(a)
            pss.append(ps if ps else med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]),
                                     [PCT_LO, PCT_HI])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= CORPUS_D:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            vol[idx] = (_crop_resize(aw, ps) * 255).astype(np.uint8)
            idx += 1
        if idx >= CORPUS_D:
            break
    mask = (vol.reshape(CORPUS_D, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def build_test_corpus(test_ds, n_workers: int = 8):
    """(vols [N,D,336,336] uint8, masks [N,D], uids) for the test split.

    Small enough to hold in RAM: a hidden test set is a handful to a few hundred
    studies, so 64*336*336 = 7.2 MB each stays trivial next to the train corpus
    this mirrors.
    """
    from concurrent.futures import ThreadPoolExecutor
    import time

    t0 = time.time()
    uids = list(test_ds.uids)
    rows_by_study = {}
    for uid in uids:
        rows_by_study[uid] = [
            {"SeriesInstanceUID": sid, "Anatomical_Plane": plane,
             "Fluid_Sensitive": fl, "Fat_Suppression": fat}
            for sid, plane, fl, fat in test_ds.study_to_series.get(uid, [])]
    vols = np.zeros((len(uids), CORPUS_D, CORPUS_IMG, CORPUS_IMG), np.uint8)
    masks = np.zeros((len(uids), CORPUS_D), np.uint8)

    def _one(i):
        uid = uids[i]
        return i, build_test_volume(uid, rows_by_study[uid], test_ds.series_root)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, (v, m) in ex.map(_one, range(len(uids))):
            vols[i], masks[i] = v, m
    filled = int((masks.sum(1) > 0).sum())
    print(f"test corpus: {vols.shape} | {filled}/{len(uids)} studies with data | "
          f"mean valid slices {masks.sum(1).mean():.1f}/{CORPUS_D} | "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return vols, masks, uids


def shard_bounds(n, shard, n_shards):
    """Half-open [lo, hi) split of n items. Shards cover 0..n without overlap."""
    if not (0 <= shard < n_shards) or n_shards < 1:
        raise ValueError(f"shard={shard} n_shards={n_shards}")
    return shard * n // n_shards, (shard + 1) * n // n_shards


def run_corpus_dump(shard=0, n_shards=2, out_dir=".", n_workers=8):
    """Decode train DICOMs into one 64-slice uint8 shard on disk.

    4407 x 64 x 336 x 336 does not fit Kaggle's 20 GB working dir (~31.8 GB),
    so this writes one half (~15.9 GB) as vols64_p{shard}.npy + masks + ids.
    Two CPU kernels (no GPU quota) produce the pair; the C4 train kernel
    memmaps both. Incremental memmap writes so peak RAM is a handful of studies.
    """
    import json
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dataset import RSNADataset

    ds = RSNADataset("train", use_report=False)
    if not ds.series_root or not os.path.isdir(ds.series_root):
        raise SystemExit(f"train series root missing: {ds.series_root}")
    uids = [str(u) for u in ds.uids]
    lo, hi = shard_bounds(len(uids), shard, n_shards)
    part = uids[lo:hi]
    nbytes = len(part) * CORPUS_D * CORPUS_IMG * CORPUS_IMG
    print(f"dump shard {shard}/{n_shards} studies {lo}:{hi} n={len(part)} | "
          f"({len(part)}, {CORPUS_D}, {CORPUS_IMG}, {CORPUS_IMG}) | "
          f"{nbytes / 1024 ** 3:.2f} GB | slots {CORPUS_SLOTS}", flush=True)
    free = shutil.disk_usage(out_dir).free
    print(f"disk {out_dir} free {free / 1024 ** 3:.1f} GB", flush=True)
    if free < nbytes * 1.05:
        raise SystemExit(
            f"not enough disk for shard {shard}: need {nbytes / 1024 ** 3:.1f} GB")

    rows_by_study = {}
    for uid in part:
        rows_by_study[uid] = [
            {"SeriesInstanceUID": sid, "Anatomical_Plane": plane,
             "Fluid_Sensitive": fl, "Fat_Suppression": fat}
            for sid, plane, fl, fat in ds.study_to_series.get(uid, [])]

    vol_path = os.path.join(out_dir, f"vols64_p{shard}.npy")
    mask_path = os.path.join(out_dir, f"masks64_p{shard}.npy")
    id_path = os.path.join(out_dir, f"ids64_p{shard}.npy")
    vols = np.lib.format.open_memmap(
        vol_path, mode="w+", dtype=np.uint8,
        shape=(len(part), CORPUS_D, CORPUS_IMG, CORPUS_IMG))
    masks = np.zeros((len(part), CORPUS_D), np.uint8)

    def _one(i):
        uid = part[i]
        v, m = build_test_volume(uid, rows_by_study[uid], ds.series_root)
        return i, v, m

    t0 = time.time()
    done, filled = 0, 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_one, i) for i in range(len(part))]
        for fut in as_completed(futs):
            i, v, m = fut.result()
            vols[i] = v
            masks[i] = m
            done += 1
            filled += int(m.sum() > 0)
            if done % 50 == 0 or done == len(part):
                print(f"  {done}/{len(part)}  filled {filled}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    vols.flush()
    del vols
    np.save(mask_path, masks)
    np.save(id_path, np.asarray(part, dtype=object))
    meta = {
        "shard": shard, "n_shards": n_shards, "lo": lo, "hi": hi,
        "n": len(part), "shape": [len(part), CORPUS_D, CORPUS_IMG, CORPUS_IMG],
        "slots": CORPUS_SLOTS, "crop_mm": CORPUS_CROP_MM,
        "span": [SPAN_LO, SPAN_HI], "pct": [PCT_LO, PCT_HI],
        "filled": int(filled),
        "mean_valid": float(masks.sum(1).mean()) if len(part) else 0.0,
    }
    with open(os.path.join(out_dir, f"manifest64_p{shard}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"dump done: {vol_path} | filled {filled}/{len(part)} | "
          f"mean valid {meta['mean_valid']:.1f}/{CORPUS_D} | "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return meta


# --------------------------------------------------------------------------- #
# training on the corpus (the reference arm's recipe)
# --------------------------------------------------------------------------- #
def build_param_groups(model, bb_lr, head_lr, llrd=None):
    """AdamW param groups plus the matching OneCycle max_lr list.

    llrd=None (the default) returns exactly the two groups every run through C8 used --
    whole backbone at bb_lr, head at head_lr, in that order -- so those runs reproduce.

    0 < llrd < 1 turns on layer-wise LR decay for ViT backbones (anything with
    `.blocks`): the patch/cls/pos embedding is depth 0, blocks[i] is depth i+1, the final
    norm is depth L+1, and a parameter at depth d trains at bb_lr * llrd ** (L + 1 - d).
    Added for C9b: the DINOv2 ViT-B arm collapsed in epoch 2, exactly when OneCycle
    reached its peak, while CoAtNet under the same schedule kept improving (LOG #394).
    """
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.")]
    bb = model.backbone
    if not llrd or not hasattr(bb, "blocks"):
        bb_params = [p for p in bb.parameters() if p.requires_grad]
        return ([{"params": bb_params, "lr": bb_lr},
                 {"params": head_params, "lr": head_lr}], [bb_lr, head_lr])
    n_blocks = len(bb.blocks)
    by_depth = {}
    for n, p in bb.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("blocks."):
            d = int(n.split(".")[1]) + 1
        elif n.startswith(("cls_token", "pos_embed", "reg_token", "mask_token", "patch_embed")):
            d = 0
        else:                                   # final norm / fc_norm
            d = n_blocks + 1
        by_depth.setdefault(d, []).append(p)
    groups, lrs = [], []
    for d in sorted(by_depth):
        lr = bb_lr * llrd ** (n_blocks + 1 - d)
        groups.append({"params": by_depth[d], "lr": lr})
        lrs.append(lr)
    groups.append({"params": head_params, "lr": head_lr})
    lrs.append(head_lr)
    return groups, lrs


def freeze_batchnorm_stats(model):
    """Put every BatchNorm layer in eval mode (running stats frozen); return the count.

    Must be called after each model.train(), which flips them back. The affine weight and
    bias keep requires_grad, so only the running mean/var are frozen. Experiment 4: each
    C-line step estimates batch statistics from 24 highly correlated windows of just two
    knees, and CoAtNet-2 has 25 BatchNorm layers (LOG #376).
    """
    import torch.nn as _nn
    n = 0
    for m in model.modules():
        if isinstance(m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d)):
            m.eval()
            n += 1
    return n


def train_on_corpus(vols, masks, row_of, train_ids, gold_ids, labels, gold_y,
                    res, k_train, k_eval, epochs, batch, bb_lr, head_lr,
                    device, seed=42, topk=3, time_budget_s=None,
                    encode_chunk=24, ckpt_path=None, llrd=None, sample_weights=None,
                    freeze_bn=False):
    """Fine-tune the study-window MIL model on the memmap corpus.

    Differences from our R line, all adopted from the reference arm:
      * `pos_weight` per finding from its prevalence -- we had NO class balancing,
        which on a macro-AUC metric penalises the rare findings (Fracture,
        Contusion) most.
      * a bigger MIL bag (k windows/study/step) -- R1-R3 used 4, which is a very
        small bag for attention pooling.
      * intensity jitter only. No affine. Their comment on flips is that
        laterality is signal, and our own line reached the same conclusion.
      * all 58 gold studies held out as the validation set, so every soft-labelled
        study trains. Epoch selection reads gold macro-AUC.
      * top-k epochs averaged by validation score (not a tail-window SWA).
    """
    import time
    import torch
    import torch.nn as nn
    from raptor import RaptorClassifier, build_raptor_backbone, build_head, _recalibrate_bn
    from train import macro_auc
    from config import RAPTOR_ARCH

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    bb = build_raptor_backbone()
    model = build_head(bb).to(device)
    _s4 = hasattr(model, "set_positions")
    if _s4:
        import config as _cfg_s4
        print(f"S4 head: {type(model).__name__} | order={_cfg_s4.S4_ORDER} | "
              f"contiguous={_cfg_s4.CORPUS_TRAIN_CONTIGUOUS} | allow={_cfg_s4.CORPUS_CENTRE_ALLOW}", flush=True)
        if _cfg_s4.S4_ORDER == "shuffled" and ckpt_path:
            _uids = list(train_ids) + list(gold_ids)
            _man = os.path.join(os.path.dirname(ckpt_path), "s4_perm_manifest.npz")
            np.savez(_man, uids=np.array(_uids), perms=np.stack([s4_perm(u) for u in _uids]),
                     tag=np.array(S4_PERM_TAG))
            print(f"S4 permutation manifest: {_man} ({len(_uids)} studies)", flush=True)
    try:
        import r9700_opt
        if r9700_opt.is_r9700():
            r9700_opt.optimize_model_for_r9700(model)
    except Exception as e:
        print(f"[R9700 Opt] Note: {e}", flush=True)

    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32,
                      device=device)
    print("pos_weight per finding:", np.round(pw.cpu().numpy(), 2).tolist(), flush=True)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    if sample_weights is not None:
        # Experiment 3 (LOG #405): weight each study-finding term by its label confidence.
        # Weighted mean, so an all-ones W reproduces the unweighted loss exactly.
        _lossf_el = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        _wm = np.stack([sample_weights[u] for u in train_ids])
        print(f"confidence-weighted BCE: W mean {_wm.mean():.3f} min {_wm.min():.3f} "
              f"max {_wm.max():.3f}", flush=True)

    groups, max_lrs = build_param_groups(model, bb_lr, head_lr, llrd)
    if llrd:
        print(f"layer-wise LR decay {llrd}: {len(groups) - 1} backbone groups, "
              f"lr {max_lrs[0]:.2e} (embed) .. {max_lrs[-2]:.2e} (top) | "
              f"head {max_lrs[-1]:.2e}", flush=True)
    fused_opt = False
    try:
        import r9700_opt
        fused_opt = r9700_opt.is_r9700()
    except Exception:
        pass
    opt = torch.optim.AdamW(groups, weight_decay=0.02, fused=fused_opt)
    steps_per = max(len(train_ids) // batch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lrs, total_steps=max(epochs * steps_per, 1),
        pct_start=0.15)
    use_cuda = device == "cuda"
    # The reference trains in bf16, but a T4 is Turing (cc 7.5): it has fp16 and fp32
    # conv engines and NOT bf16 for these shapes -- the reference's own inference code
    # says so in a comment. Requesting bf16 there does not fail loudly, it falls back
    # to fp32 activations and roughly doubles the memory, which is what OOM'd C1 at
    # the very step count R3 ran safely in fp16. Pick the dtype the card actually has,
    # and bring the GradScaler back when that means fp16.
    # NOT torch.cuda.is_bf16_supported(): on a T4 it returns True (it counts
    # emulation), so the first attempt at this fix silently stayed on bf16 and
    # OOM'd with byte-identical numbers. Compute capability is unambiguous --
    # native bf16 starts at Ampere (8.0); Turing (7.5) has fp16 tensor cores only.
    _cap = torch.cuda.get_device_capability()[0] if use_cuda else 0
    amp_dtype = torch.bfloat16 if _cap >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and amp_dtype is torch.float16)
    if use_cuda:
        print(f"amp dtype: {amp_dtype} (capability sm_{_cap}x) | "
              f"scaler {scaler.is_enabled()}", flush=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _bag(uids, k, train, jitter=None):
        xs, ps = [], []
        for u in uids:
            r = row_of[u]
            cs = centers_from_mask(masks[r], k, train, rng)
            xs.append(read_windows(vols[r], cs, res))
            if _s4:
                ps.append(s4_positions(u, cs))
        if _s4:   # consumed by the very next forward (S4SequenceClassifier fails if it is missing)
            model.set_positions(torch.from_numpy(np.stack(ps)).long())
        x = torch.stack(xs, 0)                        # (B, K, 3, res, res) in [0,1]
        if jitter is not None:
            x = (x * jitter).clamp(0, 1)              # jitter BEFORE normalisation
        b, kk = x.shape[:2]
        # the corpus stores [0,1]; an ImageNet-pretrained backbone expects its own
        # channel statistics (the reference runs NORM="imagenet" for its CoAtNet arm)
        return ((x.flatten(0, 1) - mean) / std).view(b, kk, *x.shape[2:])

    @torch.no_grad()
    def evaluate(k):
        model.eval()
        preds = []
        for b0 in range(0, len(gold_ids), 2):
            x = _bag(gold_ids[b0:b0 + 2], k, train=False).to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                o = model(x, chunk=encode_chunk).float()
            preds.append(torch.sigmoid(o).cpu().numpy())
        return macro_auc(gold_y, np.concatenate(preds))[0]

    best, best_state, keep = -1.0, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if freeze_bn:
            _n_bn = freeze_batchnorm_stats(model)
            if ep == 0:
                print(f"frozen BN statistics: {_n_bn} BatchNorm layers held in eval mode", flush=True)
        order = rng.permutation(len(train_ids))
        tot, nb, n_skip = 0.0, 0, 0
        for b0 in range(0, len(train_ids) - batch + 1, batch):
            sel = [train_ids[i] for i in order[b0:b0 + batch]]
            # intensity jitter only -- no affine, no flips (laterality is signal)
            x = _bag(sel, k_train, train=True,
                     jitter=1.0 + (rng.rand() - 0.5) * 0.20).to(device)
            y = torch.from_numpy(np.stack([labels[u] for u in sel])).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                if sample_weights is None:
                    loss = lossf(model(x).float(), y)
                else:
                    _w = torch.from_numpy(np.stack([sample_weights[u] for u in sel])).to(device)
                    loss = (_lossf_el(model(x).float(), y) * _w).sum() / _w.sum().clamp(min=1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)          # unscale before clipping, or the norm is wrong
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            # The GradScaler is only enabled for fp16 (see amp_dtype above), and it is
            # what skips inf/nan steps. On a bf16 card it is disabled, so `step` is a
            # bare `opt.step()` with NO such skip -- and clip_grad_norm_ multiplies
            # EVERY gradient by a nan total-norm, so one bad batch poisons the weights
            # permanently. That is exactly how the local R9700 run went loss=nan at
            # epoch 5 and never recovered. Guard the step ourselves on that path;
            # leave the fp16 path byte-identical so T4 runs stay comparable.
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            elif torch.isfinite(gn):
                opt.step()
            else:
                n_skip += 1
            sched.step()
            if np.isfinite(float(loss.item())):
                tot += float(loss.item())
                nb += 1
        au = evaluate(k_eval)
        print(f"    epoch {ep + 1}/{epochs}  loss={tot / max(nb, 1):.4f}  "
              f"gold={au:.4f}  [{(time.time() - t0) / 60:.1f}m]"
              + (f"  skip={n_skip}" if n_skip else ""), flush=True)
        state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        if au > best:
            best, best_state = au, state
            if ckpt_path:
                # A 16-epoch local run must not stake every epoch on reaching the
                # end of the loop; C5 lost 4 good epochs that way. Write via a
                # temp file + rename so a copy taken mid-training (e.g. to Drive
                # from a Colab cell) can never catch a half-written 270 MB file.
                torch.save({"state_dict": state, "res": res, "gold_auc": best,
                            "slots": CORPUS_SLOTS, "epoch": ep + 1,
                            "arch": RAPTOR_ARCH, **_specialist_meta()},
                           ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
        keep.append((au, state))
        keep.sort(key=lambda t: -t[0])
        keep[:] = keep[:max(1, topk)]
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"    time budget hit after epoch {ep + 1}", flush=True)
            break

    if len(keep) > 1:
        # Average the top-k epochs, then re-accumulate BatchNorm statistics: the
        # running stats belong to the trajectory that produced them, not to the
        # averaged weights. Accept the average only if it beats the best epoch.
        avg = {kk: sum(s[kk].float() for _, s in keep) / len(keep)
               for kk in keep[0][1] if keep[0][1][kk].dtype.is_floating_point}
        full = model.state_dict()
        full.update({kk: v.to(full[kk].dtype) for kk, v in avg.items()})
        model.load_state_dict(full)
        sub = [train_ids[i] for i in rng.choice(len(train_ids), size=min(120, len(train_ids)),
                                                replace=False)]

        _recal_batches = 30
        bn = [m for m in model.modules()
              if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                torch.nn.BatchNorm3d))]
        if bn and freeze_bn:
            # stats were never updated, so every kept epoch shares them: nothing to re-estimate
            print("    BN recalibration skipped (frozen BN statistics)", flush=True)
        if bn and not freeze_bn:
            old = {m: m.momentum for m in bn}
            for m in bn:
                m.reset_running_stats()
                m.momentum = None
            model.train()
            with torch.no_grad():
                for i in range(0, min(_recal_batches * batch, len(sub)), batch):
                    x = _bag(sub[i:i + batch], k_train, train=True).to(device)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                        model(x)
            for m in bn:
                m.momentum = old[m]
        au_avg = evaluate(k_eval)
        print(f"    top-{len(keep)} average: gold={au_avg:.4f} (best epoch {best:.4f})",
              flush=True)
        if ckpt_path:
            # Save BOTH candidates whichever the rule picks below. The same rule gave C6
            # the averaged weights and C7 the single best epoch, and C7's average was
            # only ever in memory, so that confound could not be split later (LOG #388).
            _base = ckpt_path[:-3] if ckpt_path.endswith(".pt") else ckpt_path
            _meta = {"res": res, "slots": CORPUS_SLOTS, "arch": RAPTOR_ARCH, **_specialist_meta()}
            torch.save({"state_dict": {kk: v.detach().cpu().clone()
                                       for kk, v in model.state_dict().items()},
                        "gold_auc": au_avg, "weights": "topk_avg", **_meta},
                       _base + "_topk_avg.pt")
            torch.save({"state_dict": best_state, "gold_auc": best,
                        "weights": "best_epoch", **_meta}, _base + "_best_epoch.pt")
            print(f"    saved both candidates: {_base}_topk_avg.pt / _best_epoch.pt",
                  flush=True)
        if au_avg > best:
            best = au_avg
            best_state = {kk: v.detach().cpu().clone()
                          for kk, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def _collect_ckpts(names):
    """Resolve every named checkpoint that exists, preserving order, no dups."""
    paths, seen = [], set()
    for name in names:
        p = _find(name)
        if not p:
            continue
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        paths.append(p)
        seen.add(ap)
    return paths


def run_corpus_infer(out_dir=".", ckpt_names=None, k_eval=None, ckpt_weights=None):
    """Score the hidden test from checkpoint(s) trained ANYWHERE. No training.

    Why this exists: submitting a *training* kernel makes Kaggle re-run the whole
    fit on every submission (C3 took ~5.5h to score), and it cannot score weights
    trained off-Kaggle at all -- which is every arm from C5 onwards. This mode
    mounts the weights instead, so a submission costs the DICOM decode plus a
    forward pass.

    Multiple checkpoints are rank-averaged per column (the metric only reads
    order). Geometry must match across members -- mixed 44/64-slice bags would
    be a silent distribution shift, so a mismatch exits loudly.

    `ckpt_weights` fuses members with UNEQUAL weight, which a heterogeneous
    ensemble needs: an arm that is materially weaker drags an equal-weight average
    below the strong arm alone (measured offline on the 58 gold: w=0.5 scored 0.8937
    vs 0.9068 for the CoAtNet arm by itself). Weights are chosen offline and baked in
    at build time -- never tuned against the leaderboard.

    The slice geometry comes FROM THE CHECKPOINT, never from config: C5/C6/C7
    trained on 64 slices while C1-C3 trained on 44. `build_test_volume` reads
    the module globals, so they are rebound here to whatever the checkpoint
    recorded.
    """
    import time
    import torch
    global CORPUS_SLOTS, CORPUS_D
    from config import RAPTOR_RES, CORPUS_K_EVAL, RAPTOR_ENCODE_CHUNK, RAPTOR_ARCH
    from dataset import RSNADataset
    from train import _write_submission, rank_average, _rankdata
    from raptor import RaptorClassifier, build_raptor_backbone, build_head

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ckpt_names or [
        "c7_corpus0.pt", "c6_corpus0.pt", "c5_corpus0.pt", "corpus0.pt",
    ]
    paths = _collect_ckpts(names)
    if not paths:
        raise SystemExit(f"no checkpoint found among {names} -- attach a weights dataset")
    print(f"ensemble: {len(paths)} checkpoint(s)", flush=True)

    loaded = []
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "epoch" in ck:
            print(f"WARNING: {path} is a MID-RUN snapshot (epoch {ck['epoch']}), "
                  f"not a finished run -- top-k averaging never happened", flush=True)
        loaded.append((path, ck))

    slots = [tuple(x) for x in loaded[0][1].get("slots", CORPUS_SLOTS)]
    res = int(loaded[0][1].get("res", RAPTOR_RES))
    for path, ck in loaded[1:]:
        s2 = [tuple(x) for x in ck.get("slots", CORPUS_SLOTS)]
        r2 = int(ck.get("res", RAPTOR_RES))
        if s2 != slots or r2 != res:
            raise SystemExit(
                f"geometry mismatch: {path} slots={s2} res={r2} vs {slots}/{res}")
    CORPUS_SLOTS = slots
    CORPUS_D = sum(x[2] for x in slots)
    # Diagnosis specialist (LOG #490): restore the training window policy from the
    # checkpoint; members of one ensemble must agree on it.
    _allows = {tuple(ck.get("centre_allow") or ()) for _p, ck in loaded}
    if len(_allows) != 1:
        raise SystemExit(f"window-policy mismatch across members: {_allows}")
    _allow = next(iter(_allows))
    import config as _cfg_inf
    _cfg_inf.CORPUS_CENTRE_ALLOW = list(_allow) or None
    print(f"  window centres: {'ALL' if not _allow else list(_allow)} | "
          f"trained labels: {loaded[0][1].get('train_labels') or 'ALL'}", flush=True)
    # S4 (LOG #523): restore head type and position mode from the checkpoint; members must agree.
    _heads = {(ck.get("head") or "mil", ck.get("s4_order")) for _p, ck in loaded}
    if len(_heads) != 1:
        raise SystemExit(f"head mismatch across members: {_heads}")
    _head, _order = next(iter(_heads))
    _cfg_inf.RAPTOR_HEAD, _cfg_inf.S4_ORDER = _head, _order
    if _head != "mil":
        print(f"  head: {_head} | S4 order: {_order}", flush=True)
    k = int(k_eval or CORPUS_K_EVAL)
    print(f"  res={res} D={CORPUS_D} k_eval={k} slots={CORPUS_SLOTS}", flush=True)
    print(f"  window rule: {'LEGAL (same slot, non-empty rows)' if LEGAL_WINDOWS else 'reference'}",
          flush=True)
    for path, ck in loaded:
        print(f"  {path} gold_auc={ck.get('gold_auc')} "
              f"arch={ck.get('arch', RAPTOR_ARCH)}", flush=True)

    test = RSNADataset("test")
    tv, tm, tuids = build_test_corpus(test)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    amp_dtype = (torch.bfloat16
                 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)

    member_preds = []
    member_default = []
    last_model = None
    for path, ck in loaded:
        # The backbone comes FROM THE CHECKPOINT: a heterogeneous ensemble (CoAtNet
        # arm + RadImageNet ResNet-50 arm) has a different architecture per member,
        # and f_dim differs with it (1024 vs 2048).
        bb = build_raptor_backbone(ck.get("arch", RAPTOR_ARCH))
        model = build_head(bb, ck.get("head") or "mil").to(device)
        # strict=True on purpose: a silently partial load here would submit a
        # half-random model and we would read the low score as a failed experiment.
        model.load_state_dict({kk: v for kk, v in ck["state_dict"].items()})
        model.eval()
        print(f"state_dict loaded strictly: {os.path.basename(path)}", flush=True)
        _s4 = hasattr(model, "set_positions")
        def _predict(legal):
            preds = []
            with torch.no_grad():
                for i in range(0, len(tuids), 2):
                    xs, ps = [], []
                    for r in range(i, min(i + 2, len(tuids))):
                        cs = centers_from_mask(tm[r], k, train=False, legal=legal)
                        xs.append(read_windows(tv[r], cs, res))
                        if _s4:
                            ps.append(s4_positions(tuids[r], cs))
                    if _s4:
                        model.set_positions(torch.from_numpy(np.stack(ps)).long())
                    x = torch.stack(xs, 0)
                    b, kk = x.shape[:2]
                    x = ((x.flatten(0, 1) - mean) / std).view(b, kk, *x.shape[2:])
                    with torch.autocast("cuda", dtype=amp_dtype,
                                        enabled=device == "cuda"):
                        o = model(x.to(device), chunk=RAPTOR_ENCODE_CHUNK).float()
                    preds.append(torch.sigmoid(o).cpu().numpy())
            return np.concatenate(preds)

        member_preds.append(_predict(None))
        if LEGAL_WINDOWS:
            # Same model, same decoded volumes, reference windows. This side output must
            # reproduce the member's standalone submission exactly, which is what shows
            # the window rule is the only thing that changed (LOG #401).
            member_default.append(_predict(False))
        last_model = model
        if device == "cuda" and len(loaded) > 1:
            del model
            torch.cuda.empty_cache()

    def _fuse(mp, tag):
        pre = f"[{tag}] " if LEGAL_WINDOWS else ""
        if len(mp) == 1:
            return mp[0]
        if ckpt_weights:
            w = np.asarray(ckpt_weights, dtype=np.float64)
            if len(w) != len(mp):
                raise SystemExit(f"{len(w)} weights for {len(mp)} members")
            w = w / w.sum()
            n = mp[0].shape[0]
            out = np.zeros_like(mp[0])
            for j in range(mp[0].shape[1]):
                out[:, j] = sum(wi * (_rankdata(p[:, j]) / n) for wi, p in zip(w, mp))
            print(f"{pre}weighted rank fusion of {len(mp)} members, "
                  f"weights={np.round(w, 4).tolist()} (pre-registered, not LB-tuned)",
                  flush=True)
            return out
        print(f"{pre}rank-averaged {len(mp)} members (equal weight)", flush=True)
        return rank_average(mp)

    fused = _fuse(member_preds, "submission")
    if LEGAL_WINDOWS:
        _write_submission(tuids, _fuse(member_default, "default-windows side output"),
                          os.path.join(out_dir, "_default_windows_submission.csv"))
        print("wrote default-window side output _default_windows_submission.csv", flush=True)
    _write_submission(tuids, fused, os.path.join(out_dir, "submission.csv"))
    print(f"corpus infer finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    return last_model


def run_corpus_submit(epochs=None, batch=None, k_train=None, k_eval=None,
                      res=None, out_dir=".", seed=42, do_submit=True,
                      time_budget_h=6.0, max_studies=None):
    """Train on the published memmap corpus, then decode + predict the test set.

    The train corpus covers only train.csv, so the test split is rebuilt from
    DICOM with `build_test_volume`, which mirrors the corpus's own preprocessing
    exactly (see this module's docstring). Getting that wrong is silent: the
    model would simply see a different distribution than it trained on.
    """
    import time
    import torch
    from config import (LABELS, RAPTOR_RES, CORPUS_EPOCHS, CORPUS_BATCH,
                        CORPUS_K_TRAIN, CORPUS_K_EVAL, RAPTOR_LR_BACKBONE,
                        RAPTOR_LR_HEAD, RAPTOR_ENCODE_CHUNK)
    from dataset import RSNADataset
    from train import _write_submission, macro_auc
    from raptor import predict_windows  # noqa: F401  (kept importable for parity)

    epochs = epochs or CORPUS_EPOCHS
    batch = batch or CORPUS_BATCH
    k_train = k_train or CORPUS_K_TRAIN
    k_eval = k_eval or CORPUS_K_EVAL
    res = res or RAPTOR_RES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    vols, masks, cids = open_corpus()
    row_of = {u: i for i, u in enumerate(cids)}

    ds = RSNADataset("train")
    y_raw = ds.y_raw
    has_gold = np.isfinite(y_raw).all(axis=1)
    covered = [u for u in ds.uids if u in row_of]
    print(f"corpus covers {len(covered)}/{len(ds.uids)} of train.csv", flush=True)

    gold_ids = [u for i, u in enumerate(ds.uids) if has_gold[i] and u in row_of]
    goldset = set(gold_ids)
    train_ids = [u for u in covered if u not in goldset]
    if max_studies:
        train_ids = train_ids[:max_studies]
    # soft targets from OUR report extractor (the reference's LLM-generated
    # labels_llm_soft.parquet is not distributed with its notebook)
    idx_of = {u: i for i, u in enumerate(ds.uids)}
    labels = {u: ds.targets[idx_of[u]].astype(np.float32) for u in train_ids}
    import config as _cfg_w
    sample_weights = None
    if getattr(_cfg_w, "CORPUS_CONF_WEIGHTED_BCE", False):
        sample_weights = {u: ds.weights[idx_of[u]].astype(np.float32) for u in train_ids}
    gold_y = np.stack([y_raw[idx_of[u]] for u in gold_ids]).astype(np.float32)
    _train_labels = getattr(_cfg_w, "CORPUS_TRAIN_LABELS", None)
    if _train_labels:
        # Diagnosis specialist (LOG #490): zero BCE weight on every other label (the
        # weighted-mean loss then averages over the kept labels only) and blank the other
        # gold columns so macro_auc -- epoch selection and top-k -- reads the kept ones only.
        from config import LABELS as _LAB
        _keep = np.array([l in set(_train_labels) for l in _LAB], np.float32)
        assert _keep.sum() == len(_train_labels), f"unknown label in {_train_labels}"
        base_w = sample_weights or {u: np.ones(len(_LAB), np.float32) for u in train_ids}
        sample_weights = {u: (base_w[u] * _keep).astype(np.float32) for u in train_ids}
        gold_y = np.where(_keep[None, :] > 0, gold_y, np.nan).astype(np.float32)
        print(f"specialist: training / selecting on {list(_train_labels)} only", flush=True)
    _allow = getattr(_cfg_w, "CORPUS_CENTRE_ALLOW", None)
    if _allow:
        print(f"specialist: window centres restricted to {sorted(_allow)}", flush=True)
    print(f"train {len(train_ids)} (soft) | gold-val {len(gold_ids)} | "
          f"res {res} | k {k_train}/{k_eval} | bs {batch} | ep {epochs}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "corpus0.pt")
    import config as _cfg
    model, best = train_on_corpus(
        vols, masks, row_of, train_ids, gold_ids, labels, gold_y, res,
        k_train, k_eval, epochs, batch, RAPTOR_LR_BACKBONE, RAPTOR_LR_HEAD,
        device, seed=seed, time_budget_s=time_budget_h * 3600.0,
        encode_chunk=RAPTOR_ENCODE_CHUNK, ckpt_path=ckpt,
        llrd=getattr(_cfg, "RAPTOR_LLRD", None), sample_weights=sample_weights,
        freeze_bn=getattr(_cfg, "CORPUS_FREEZE_BN", False))
    print(f"corpus run: best gold macro-AUC={best:.4f}", flush=True)

    from config import RAPTOR_ARCH as _ARCH
    torch.save({"state_dict": model.state_dict(), "res": res,
                "gold_auc": best, "slots": CORPUS_SLOTS, "arch": _ARCH, **_specialist_meta()}, ckpt)
    print("saved", ckpt, flush=True)

    if do_submit:
        if hasattr(model, "set_positions"):
            # S4 runs train with do_submit=False and are scored by run_corpus_infer, which sets
            # the positions; this legacy in-kernel path does not (fail-closed, LOG #523).
            raise SystemExit("do_submit is not supported for the S4 head; use corpus_infer")
        test = RSNADataset("test")
        tv, tm, tuids = build_test_corpus(test)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(tuids), 2):
                xs = []
                for r in range(i, min(i + 2, len(tuids))):
                    cs = centers_from_mask(tm[r], k_eval, train=False)
                    xs.append(read_windows(tv[r], cs, res))
                x = torch.stack(xs, 0)
                b, kk = x.shape[:2]
                x = ((x.flatten(0, 1) - mean) / std).view(b, kk, *x.shape[2:])
                amp_dtype = (torch.bfloat16
                             if device == "cuda"
                             and torch.cuda.get_device_capability()[0] >= 8
                             else torch.float16)
                with torch.autocast("cuda", dtype=amp_dtype,
                                    enabled=device == "cuda"):
                    o = model(x.to(device), chunk=RAPTOR_ENCODE_CHUNK).float()
                preds.append(torch.sigmoid(o).cpu().numpy())
        _write_submission(tuids, np.concatenate(preds),
                          os.path.join(out_dir, "submission.csv"))
    print(f"corpus run finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    return model
