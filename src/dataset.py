"""PyTorch Dataset for RSNA Knee — slot-based, imaging-only at inference.

Design notes (incorporating top public-notebook lessons, see LOG.md):
- A Study is the prediction unit. Its MRI content is mapped onto N_SLOTS fixed
  slots, one per (plane, fluid_sensitive, fat_suppression) combination defined in
  config.SLOTS. A slot may be empty -> a presence MASK (1 present / 0 missing)
  is returned so the attention head can ignore missing slots.
- Laterality normalization (right->left convention) is read from DICOM headers
  where present; it degrades to a no-op if headers are stripped (local sample).
- The radiology report is TRAIN-ONLY (test.csv has no Report column). We grade
  it offline into 12 soft targets + 12 per-label confidence weights. Where a gold
  label exists it overrides the soft target and gets GOLD_WEIGHT; where it does
  not, the soft target + confidence weight drive weak supervision.
"""
from __future__ import annotations
import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
# torch is imported lazily inside __getitem__/collate (so this module is importable
# even where torch isn't installed, e.g. for offline report-target inspection).
try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - only hit in torch-less environments
    class Dataset:  # placeholder base so the module still imports
        pass

from config import (LABELS, DATA_ROOT, RAW, IMG_SIZE, SLICES_PER_SLOT,
                    N_SLOTS, SLOTS, USE_REPORT_TARGETS, GROUP, N_GROUPS_MAX,
                    CACHE_BUDGET_GB, GOLD_WEIGHT, LAT_MIN_AGREEMENT,
                    DERIVED_W_FLOOR, USE_RECOVERED_FLAGS, SLOT_FALLBACK,
                    SOFT_LABEL_PARQUETS, TEACHER_PARQUET, TEACHER_LAMBDA)
import config as _cfg
import dicom_utils as du
import report_extractor as re


def _series_manifest(split: str) -> pd.DataFrame:
    """Load {split}_series.csv giving Study->Series metadata."""
    path = os.path.join(RAW, f"{split}_series.csv")
    if not os.path.exists(path) and DATA_ROOT:
        path = os.path.join(DATA_ROOT, f"{split}_series.csv")
    return pd.read_csv(path)


def build_study_index(split: str):
    """Return (labels_df, study_to_series) for the given split.

    On Kaggle the labeled CSVs live under DATA_ROOT (only mounted inside the
    Notebook); locally they live under RAW. Prefer DATA_ROOT when present so the
    same code path works in both environments.
    """
    labels_path = os.path.join(RAW, f"{split}.csv")
    if not os.path.exists(labels_path) and DATA_ROOT:
        labels_path = os.path.join(DATA_ROOT, f"{split}.csv")
    labels_df = pd.read_csv(labels_path)
    man = _series_manifest(split)
    study_to_series = defaultdict(list)
    for _, r in man.iterrows():
        study_to_series[r["StudyInstanceUID"]].append(
            (r["SeriesInstanceUID"], r["Anatomical_Plane"],
             int(r["Fluid_Sensitive"]), int(r["Fat_Suppression"])))
    return labels_df, study_to_series


def _match_slot(plane: str, fl: int, fat: int) -> int:
    """Index of the slot matching (plane, fluid_sensitive, fat_suppression), -1 if none."""
    for i, (p, f, fa) in enumerate(SLOTS):
        if p == plane and f == fl and fa == fat:
            return i
    return -1


class RSNADataset(Dataset):
    def __init__(self, split: str = "train", series_root: str | None = None,
                 use_report: bool = USE_REPORT_TARGETS):
        self.split = split
        self.series_root = series_root or (
            os.path.join(DATA_ROOT, f"{split}_series") if DATA_ROOT else None)
        self.use_report = use_report and split == "train"

        self._rec = None          # lazily built header-recovered slot flags
        self.labels_df, self.study_to_series = build_study_index(split)
        self.uids = self.labels_df["StudyInstanceUID"].tolist()
        # test.csv has NO Report column (competition design) — only train does.
        if "Report" in self.labels_df.columns:
            self.reports = self.labels_df["Report"].fillna("").tolist()
        else:
            self.reports = [""] * len(self.uids)

        if split == "train":
            # gold labels with NaN where missing (NaN = unlabeled study)
            self.y_raw = self.labels_df[LABELS].to_numpy(dtype=np.float32)
            self.has_gold = np.isfinite(self.y_raw).any(axis=1)
            if self.use_report:
                self._build_report_targets()
            else:
                # pure hard-label mode (only studies with labels are usable)
                self.targets = np.nan_to_num(self.y_raw, nan=0.0).astype(np.float32)
                self.weights = np.isfinite(self.y_raw).astype(np.float32)
        else:
            self.y_raw = None
            self.targets = None
            self.weights = None

    def _find_label_parquet(self, name):
        """Locate a soft-label parquet locally or among the Kaggle dataset mounts."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cands = [os.path.join(here, "data", "labels", name), name]
        root = "/kaggle/input"
        if os.path.isdir(root):
            for dirpath, dirnames, files in os.walk(root):
                if dirpath[len(root):].count(os.sep) > 4 or "competition" in dirpath.lower():
                    dirnames[:] = []
                    continue
                if name in files:
                    cands.append(os.path.join(dirpath, name))
        for c in cands:
            if os.path.exists(c):
                return c
        return None

    def _build_report_targets(self):
        """Grade every report -> soft target + confidence weight; gold overrides.

        With SOFT_LABEL_PARQUETS configured, each file's per-finding (p, conf) is
        averaged with the regex extractor's, source by source. Averaging
        PROBABILITIES rather than ranks: rank-averaging pins every label's mean
        to 0.5, which is harmless for AUC but wrong for a BCE target -- and it
        also scored lower on the 58 gold studies (0.885 vs 0.889).
        """
        import pandas as _pd

        extra = []
        for name in SOFT_LABEL_PARQUETS:
            path = self._find_label_parquet(name)
            if path is None:
                print(f"soft labels: {name} NOT FOUND -- falling back to the "
                      f"extractor alone for this source", flush=True)
                continue
            df = _pd.read_parquet(path).set_index("StudyInstanceUID")
            extra.append((name, df))
            print(f"soft labels: {name} ({len(df)} studies) from {path}", flush=True)

        T = np.zeros((len(self), len(LABELS)), np.float32)
        W = np.zeros((len(self), len(LABELS)), np.float32)
        n_missing = 0
        for i, rep in enumerate(self.reports):
            grades, conf = re.extract(rep)
            uid = str(self.uids[i])
            for j, lab in enumerate(LABELS):
                ps, cs = [grades[lab]], [conf[lab]]
                for _name, df in extra:
                    if uid in df.index:
                        ps.append(float(df.at[uid, lab]))
                        cs.append(float(df.at[uid, lab + "__conf"]))
                T[i, j] = float(np.mean(ps))
                W[i, j] = DERIVED_W_FLOOR + (1.0 - DERIVED_W_FLOOR) * float(np.mean(cs))
            if extra and not any(uid in df.index for _n, df in extra):
                n_missing += 1
        if extra:
            print(f"soft labels: {len(self) - n_missing}/{len(self)} studies got "
                  f"LLM labels ({n_missing} extractor-only)", flush=True)

        if getattr(_cfg, "CALIBRATE_SYNOVITIS", False):
            # LOG #556 / B5: Clinical Effusion -> Synovitis co-occurrence calibration
            eff_idx = list(LABELS).index("Effusion")
            syn_idx = list(LABELS).index("Synovitis")
            eff = T[:, eff_idx]
            syn = T[:, syn_idx]
            mask_syn_mod = (syn < 0.25) & (eff > 0.50) & (eff <= 0.75)
            mask_syn_large = (syn < 0.25) & (eff > 0.75)
            T[mask_syn_mod, syn_idx] = 0.40
            T[mask_syn_large, syn_idx] = 0.55
            print(f"[CALIBRATE_SYNOVITIS] Elevated {mask_syn_mod.sum()} mod + {mask_syn_large.sum()} large synovitis targets", flush=True)

        if TEACHER_PARQUET and TEACHER_LAMBDA > 0:
            path = self._find_label_parquet(TEACHER_PARQUET)
            if path is None:
                raise FileNotFoundError(f"teacher labels {TEACHER_PARQUET} not found")
            tdf = _pd.read_parquet(path).set_index("StudyInstanceUID")
            missing = [u for u in map(str, self.uids) if u not in tdf.index]
            if missing:
                raise ValueError(f"teacher labels miss {len(missing)} studies")
            P = tdf.loc[[str(u) for u in self.uids], list(LABELS)].to_numpy(np.float32)
            T = (1.0 - TEACHER_LAMBDA) * T + TEACHER_LAMBDA * P
            print(f"teacher labels: {TEACHER_PARQUET} lambda={TEACHER_LAMBDA} "
                  f"from {path}", flush=True)
        # B5 (LOG #533): hard targets from the soft report target, threshold 0.5. Read at call time
        # (default off -> every earlier run unchanged); the gold override below is applied as before.
        import config as _cfg_b
        if getattr(_cfg_b, "CORPUS_BINARIZE_TARGETS", False):
            _pos = (T >= 0.5).mean(0)
            print("binarized targets at 0.5 | soft mean per label "
                  f"{np.round(T.mean(0), 3).tolist()} -> positive rate {np.round(_pos, 3).tolist()}", flush=True)
            T = (T >= 0.5).astype(np.float32)
        # override with gold where present (annotated studies are up-weighted)
        for i in range(len(self)):
            yrow = self.y_raw[i]
            for j in range(len(LABELS)):
                if np.isfinite(yrow[j]):
                    T[i, j] = yrow[j]
                    W[i, j] = float(GOLD_WEIGHT)
        self.targets = T
        self.weights = W

    def __len__(self):
        return len(self.uids)

    def _best_series_folder(self, uid: str, sids):
        return max(
            (os.path.join(self.series_root, uid, s) for s in sids),
            key=lambda p: len(glob.glob(os.path.join(p, "*.dcm"))))

    def _recovered_flags(self):
        """series UID -> (fluid, fatsat) read off the DICOM header, once per split.

        The competition CSV's two flags are perfectly correlated in this corpus, so
        they carry one bit between them and a six-slot scheme cannot be built on
        them (see LOG.md v13/v14). The header carries the weighting and the fat
        suppression independently. A series whose header decides nothing is absent
        from the map and falls back to its CSV flags.
        """
        if self._rec is not None:
            return self._rec
        self._rec = {}
        if not USE_RECOVERED_FLAGS or self.series_root is None:
            return self._rec
        from concurrent.futures import ThreadPoolExecutor
        import time
        t0 = time.time()
        jobs = []
        for uid in self.uids:
            for sid, _p, _f, _a in self.study_to_series.get(uid, []):
                jobs.append((uid, sid))

        def _one(j):
            uid, sid = j
            return sid, du.recover_sequence(
                du.read_series_meta(os.path.join(self.series_root, uid, sid)))

        n_ok = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            for sid, rec in ex.map(_one, jobs):
                if rec is not None:
                    self._rec[sid] = (int(rec[0]), int(rec[1]))
                    n_ok += 1
        print(f"recovered sequence flags: {n_ok}/{len(jobs)} series "
              f"({n_ok / max(len(jobs), 1) * 100:.1f}%) in "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        return self._rec

    def _slots_for_study(self, uid: str):
        """Map slot index -> representative series folder for a study."""
        out = {}
        if self.series_root is None:
            return out
        rec = self._recovered_flags()
        series = self.study_to_series.get(uid, [])
        # (plane, flags) per series, preferring the header-recovered flags.
        tagged = [(sid, plane, rec.get(sid, (int(fl), int(fat))))
                  for sid, plane, fl, fat in series]
        for si, (p, f, a) in enumerate(SLOTS):
            cand = [sid for sid, plane, k in tagged if plane == p and k == (f, a)]
            if not cand and SLOT_FALLBACK and f == 0:
                # Structural slots are the scarcest; before leaving one empty take any
                # non-fat-suppressed series in the plane. Exact matches still win.
                cand = [sid for sid, plane, k in tagged
                        if plane == p and k[1] == 0]
            if cand:
                out[si] = self._best_series_folder(uid, cand)
        return out

    def _load_study_slots(self, uid: str):
        """Return (imgs [N_SLOTS, S, 1, H, W], mask [N_SLOTS]) for a study."""
        imgs = np.zeros(
            (N_SLOTS, SLICES_PER_SLOT, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        mask = np.zeros(N_SLOTS, dtype=np.float32)
        for si, folder in self._slots_for_study(uid).items():
            plane, fl, fat = SLOTS[si]
            lat = du.read_laterality(folder)
            imgs[si] = du.load_series(folder, plane, lat, SLICES_PER_SLOT, IMG_SIZE)
            mask[si] = 1.0
        return imgs, mask

    def __getitem__(self, idx):
        import torch
        uid = self.uids[idx]
        imgs, mask = self._load_study_slots(uid)
        if self.split == "train":
            return (torch.from_numpy(imgs), torch.from_numpy(mask),
                    torch.from_numpy(self.targets[idx]),
                    torch.from_numpy(self.weights[idx]))
        return torch.from_numpy(imgs), torch.from_numpy(mask), uid


def collate_train(batch):
    import torch
    imgs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])
    weights = torch.stack([b[3] for b in batch])
    return imgs, masks, targets, weights


def collate_test(batch):
    import torch
    imgs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    uids = [b[2] for b in batch]
    return imgs, masks, uids


def run_seq_diag(split: str = "train", limit: int | None = None,
                 n_workers: int = 16):
    """Read one header per series and report what the sequence recovery yields.

    A diagnostic, not a training step. The competition CSV's Fluid_Sensitive and
    Fat_Suppression are perfectly correlated here, so a six-slot scheme built on
    them has a slot that can never fill (see LOG.md v13). Recovering the weighting
    and the fat-suppression flag from the header is the fix the reference pipeline
    uses -- but the public sample DICOMs arrive with headers stripped, so whether
    the mounted competition data actually carries these fields is an empirical
    question. This answers it in ~20 minutes instead of inside a 4-hour run.
    """
    from concurrent.futures import ThreadPoolExecutor
    from collections import Counter
    import time

    t0 = time.time()
    ds = RSNADataset(split)
    rows = []
    for uid in ds.uids:
        for sid, plane, fl, fat in ds.study_to_series.get(uid, []):
            rows.append((uid, sid, plane, int(fl), int(fat)))
    if limit:
        rows = rows[:limit]
    print(f"seq-diag: {len(rows)} series over {len(ds.uids)} studies", flush=True)
    if ds.series_root is None:
        print("seq-diag: no series_root -> nothing to read")
        return

    def _one(r):
        uid, sid, plane, fl, fat = r
        folder = os.path.join(ds.series_root, uid, sid)
        return r, du.read_series_meta(folder)

    metas = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for r, meta in ex.map(_one, rows):
            metas.append((r, meta))
    print(f"seq-diag: headers read in {(time.time() - t0) / 60:.1f} min", flush=True)

    field_hits = Counter()
    weights = Counter()
    n_rec = 0
    cross = Counter()          # (csv_fl, csv_fat) -> (rec_fl, rec_fat)
    rec_by_series = {}
    for (uid, sid, plane, fl, fat), meta in metas:
        for k, v in meta.items():
            if str(v).strip():
                field_hits[k] += 1
        rec = du.recover_sequence(meta)
        if rec is None:
            weights["UNRECOVERED"] += 1
            continue
        n_rec += 1
        rfl, rfat, w = rec
        weights[w] += 1
        rec_by_series[(uid, sid)] = (int(rfl), int(rfat))
        cross[((fl, fat), (int(rfl), int(rfat)))] += 1

    n = len(metas)
    print(f"\nseq-diag: header field presence over {n} series")
    for k in du._SEQ_FIELDS:
        print(f"  {k:20s} {field_hits[k]:6d}  ({field_hits[k] / n * 100:5.1f}%)")
    print(f"\nseq-diag: recovered {n_rec}/{n} ({n_rec / n * 100:.1f}%)")
    print("  weight histogram:", dict(weights))
    print("\nseq-diag: CSV (fluid,fatsat) -> recovered (fluid,fatsat)")
    for (csv_k, rec_k), c in sorted(cross.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {csv_k} -> {rec_k}: {c}")

    # Which (plane, fluid, fatsat) cells actually exist under recovered flags, and
    # which six of them cover the most studies. Copying the reference's slot list is
    # not obviously right for this corpus, so let the data name the slots.
    cell_studies = Counter()
    for (uid, sid, plane, fl, fat), _m in metas:
        k = rec_by_series.get((uid, sid), (fl, fat))
        cell_studies[(plane, k[0], k[1])] += 0     # ensure the key exists
    seen = set()
    for (uid, sid, plane, fl, fat), _m in metas:
        k = rec_by_series.get((uid, sid), (fl, fat))
        cell = (plane, k[0], k[1])
        if (uid, cell) not in seen:
            seen.add((uid, cell))
            cell_studies[cell] += 1
    print("\nseq-diag: recovered (plane, fluid, fatsat) -> studies covered")
    for cell, c in sorted(cell_studies.items(), key=lambda kv: -kv[1]):
        print(f"  {cell}: {c}")

    # Greedy: the six cells covering the most (study x slot) content. Coverage is not
    # the only criterion -- a slot also has to be contrast-homogeneous -- but a cell
    # that reaches almost no studies cannot earn a sixth of the cache either way.
    top6 = [c for c, _ in sorted(cell_studies.items(), key=lambda kv: -kv[1])[:6]]
    print("seq-diag: top-6 recovered cells by study coverage:", top6)
    print(f"  their total coverage: {sum(cell_studies[c] for c in top6)}"
          f"/{len(by_study_uids := {r[0][0] for r in metas}) * 6}"
          f" ({sum(cell_studies[c] for c in top6) / (len(by_study_uids) * 6) * 100:.1f}%)")

    # What each slot list would fill, under CSV flags vs recovered flags.
    lists = {
        "v13_csv_flags": [("Sagittal", 1, 1), ("Coronal", 1, 1), ("Axial", 1, 1),
                          ("Axial", 0, 0), ("Coronal", 0, 0), ("Sagittal", 0, 0)],
        "reference": [("Sagittal", 1, 1), ("Coronal", 1, 1), ("Axial", 1, 1),
                      ("Sagittal", 1, 0), ("Coronal", 0, 0), ("Sagittal", 0, 0)],
    }
    by_study = {}
    for (uid, sid, plane, fl, fat), _m in metas:
        by_study.setdefault(uid, []).append(
            (plane, (fl, fat), rec_by_series.get((uid, sid))))
    for name, slots in lists.items():
        for src in ("csv", "recovered"):
            for fb in (False, True):
                filled = [0] * len(slots)
                for uid, series in by_study.items():
                    for i, (p, f, a) in enumerate(slots):
                        hit = False
                        for plane, csv_k, rec_k in series:
                            k = csv_k if src == "csv" else (rec_k or csv_k)
                            if plane == p and k == (f, a):
                                hit = True
                                break
                        # The reference's fallback: a non-fluid slot is the scarcest,
                        # so before leaving it empty it takes ANY non-fat-suppressed
                        # series in that plane.
                        if not hit and fb and f == 0:
                            for plane, csv_k, rec_k in series:
                                k = csv_k if src == "csv" else (rec_k or csv_k)
                                if plane == p and k[1] == 0:
                                    hit = True
                                    break
                        filled[i] += int(hit)
                tot = sum(filled)
                cells = len(by_study) * len(slots)
                tag = "+fallback" if fb else "exact    "
                print(f"seq-diag fill [{name} / {src:9s} / {tag}]: {filled} "
                      f"= {tot}/{cells} ({tot / cells * 100:.1f}%)")
    print(f"seq-diag done in {(time.time() - t0) / 60:.1f} min", flush=True)


def plan_n_groups(n_study: int, budget_gb: float = CACHE_BUDGET_GB) -> int:
    """How many 3-slice groups fit under the uint8 cache budget."""
    bytes_per = n_study * N_SLOTS * GROUP * IMG_SIZE * IMG_SIZE  # one group
    n_g = N_GROUPS_MAX
    while n_g > 1 and n_g * bytes_per > budget_gb * (1024 ** 3):
        n_g -= 1
    return n_g


def take_group(cache_rows: np.ndarray, g: int) -> np.ndarray:
    """cache_rows [..., S, H, W] -> [..., 3, H, W] for group g."""
    return cache_rows[..., g * GROUP:(g + 1) * GROUP, :, :]


def _resolve_laterality(jobs, tag_pos, enable_fallback: bool):
    """jobs is a list of (study_i, slot_i, plane, folder). tag_pos[folder]=(tag,pos)."""
    sides = {}
    for folder, (tag, pos) in tag_pos.items():
        sides[folder] = tag or (pos if enable_fallback else None)
    return sides


def build_slot_cache(ds: "RSNADataset", indices, n_groups: int | None = None,
                     n_workers: int = 8):
    """Decode selected studies once into an in-memory uint8 cache.

    Returns (cache [N, NS, S, H, W] uint8, mask [N, NS] float32, n_groups).
    Fine-tuning revisits the same pixels every epoch; reading them from the
    mount each time would make the epoch count a function of I/O.
    """
    from concurrent.futures import ThreadPoolExecutor

    indices = np.asarray(indices)
    n = len(indices)
    n_groups = n_groups or plan_n_groups(n)
    n_slices = n_groups * GROUP
    cache = np.zeros((n, N_SLOTS, n_slices, IMG_SIZE, IMG_SIZE), np.uint8)
    mask = np.zeros((n, N_SLOTS), np.float32)
    print(f"cache: N={n} groups={n_groups} shape={cache.shape} "
          f"= {cache.nbytes / 1024 ** 3:.2f} GB")

    # Resolve (study, slot) -> folder, then laterality tags/positions.
    jobs = []  # (local_i, slot_i, plane, folder)
    tag_pos = {}
    for li, gi in enumerate(indices):
        uid = ds.uids[int(gi)]
        for si, folder in ds._slots_for_study(uid).items():
            plane = SLOTS[si][0]
            jobs.append((li, si, plane, folder))
            if folder not in tag_pos:
                tag_pos[folder] = du.read_laterality_full(folder)

    both = [(t, p) for (t, p) in tag_pos.values() if t and p]
    agree = (float(np.mean([t == p for t, p in both])) if both else float("nan"))
    use_fb = bool(both) and np.isfinite(agree) and agree >= LAT_MIN_AGREEMENT
    print(f"laterality: {len(tag_pos)} series, tag/pos agree "
          f"{agree:.3f} on {len(both)} -> fallback {'ON' if use_fb else 'OFF'}")
    sides = _resolve_laterality(jobs, tag_pos, use_fb)

    def _one(job):
        li, si, plane, folder = job
        lat = sides.get(folder)
        try:
            vol = du.load_series_stack(folder, plane, lat, n_slices, IMG_SIZE)
        except Exception:
            vol = np.zeros((n_slices, IMG_SIZE, IMG_SIZE), np.float32)
        return li, si, vol

    CHUNK = 256
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0:c0 + CHUNK]
            for li, si, vol in pool.map(_one, block):
                cache[li, si] = np.clip(vol * 255.0, 0, 255).astype(np.uint8)
                mask[li, si] = 1.0 if vol.max() > 0 else 0.0
                done += 1
            if done % 512 < CHUNK:
                print(f"  decoded {done}/{len(jobs)} slot-series", flush=True)
    print(f"cache ready: {int(mask.sum())} present slots / {mask.size}")
    return cache, mask, n_groups


def load_study_groups(ds: "RSNADataset", uid: str, n_groups: int):
    """On-the-fly test path: [NS, n_groups, 3, H, W] uint8 + mask [NS]."""
    n_slices = n_groups * GROUP
    stack = np.zeros((N_SLOTS, n_slices, IMG_SIZE, IMG_SIZE), np.uint8)
    mask = np.zeros(N_SLOTS, np.float32)
    for si, folder in ds._slots_for_study(uid).items():
        plane = SLOTS[si][0]
        lat = du.read_laterality(folder)
        vol = du.load_series_stack(folder, plane, lat, n_slices, IMG_SIZE)
        stack[si] = np.clip(vol * 255.0, 0, 255).astype(np.uint8)
        if vol.max() > 0:
            mask[si] = 1.0
    groups = stack.reshape(N_SLOTS, n_groups, GROUP, IMG_SIZE, IMG_SIZE)
    return groups, mask
