from __future__ import annotations
import os, sys, time, gc, glob, hashlib
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

print("==================================================================", flush=True)
print("  RSNA KNEE NATIVE SHARED-CACHE QUAD ENSEMBLE STARTING", flush=True)
print("  Architecture: Single-Pass Corpus64 Memmap + C6 + C14 + C12 + PF1", flush=True)
print("==================================================================", flush=True)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

# -----------------------------------------------------------------------------
# 1. Configuration & Constants
# -----------------------------------------------------------------------------
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
N_LABELS = len(LABELS)

CORPUS_IMG = 336
CORPUS_CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.06, 0.94
PCT_LO, PCT_HI = 2.0, 98.0
CORPUS_SLOTS_64 = [
    ("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
    ("Coronal", 0, 8), ("Axial", -1, 12)
]
CORPUS_D = sum(s[2] for s in CORPUS_SLOTS_64)  # 64 slices
assert CORPUS_D == 64

CACHE_DIR = Path("/kaggle/working/_shared_cache")
VOLS_MMAP = CACHE_DIR / "vols64.npy"
MASKS_FILE = CACHE_DIR / "masks64.npy"
UIDS_FILE = CACHE_DIR / "uids64.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# -----------------------------------------------------------------------------
# 2. DICOM Preprocessing Utilities
# -----------------------------------------------------------------------------
def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None

def _order_series_files(folder):
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
        return [f for _, f in keyed], med_ps
    return [f for _, f in sorted(keyed, key=lambda t: t[0])], med_ps

def _read_px(path):
    ds = pydicom.dcmread(path, force=True)
    a = ds.pixel_array.astype(np.float32)
    a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = 0.0
    return a, ps

def _crop_resize(a, ps):
    h, w = a.shape
    cpx = int(round(CORPUS_CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(a, (CORPUS_IMG, CORPUS_IMG), interpolation=cv2.INTER_AREA)

def build_test_volume(sid, rows, series_root):
    vol = np.zeros((CORPUS_D, CORPUS_IMG, CORPUS_IMG), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in CORPUS_SLOTS_64:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = _order_series_files(os.path.join(series_root, sid, r["SeriesInstanceUID"]))
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * SPAN_LO), int(n * SPAN_HI) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else np.zeros(k, dtype=int)
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
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [PCT_LO, PCT_HI])
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

def build_shared_test_corpus(uids, study_to_series, series_root, n_workers=12):
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    N = len(uids)
    print(f"[Phase 0] Building shared Corpus64 cache for {N} studies ({n_workers} threads)...", flush=True)
    vols = np.lib.format.open_memmap(
        str(VOLS_MMAP), mode="w+", dtype=np.uint8, shape=(N, CORPUS_D, CORPUS_IMG, CORPUS_IMG)
    )
    masks = np.zeros((N, CORPUS_D), dtype=np.uint8)

    def _one(i):
        uid = uids[i]
        rows = study_to_series.get(uid, [])
        v, m = build_test_volume(uid, rows, series_root)
        return i, v, m

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, v, m in pool.map(_one, range(N)):
            vols[i] = v
            masks[i] = m
            if (i + 1) % 100 == 0 or (i + 1) == N:
                print(f"  [Phase 0] Decoded {i + 1}/{N} studies in {(time.time() - t0)/60:.1f} min", flush=True)

    vols.flush()
    del vols
    np.save(str(MASKS_FILE), masks)
    np.save(str(UIDS_FILE), np.asarray(uids, dtype=object))
    print(f"[Phase 0] COMPLETE: {VOLS_MMAP.stat().st_size / (1024**3):.2f} GB memmap created in {(time.time() - t0)/60:.1f} min!", flush=True)

# -----------------------------------------------------------------------------
# 3. Model Architecture & Window Extraction
# -----------------------------------------------------------------------------
def centers_from_mask(mask, k=42, allowed_centers=None):
    valid = np.where(np.asarray(mask) > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, CORPUS_D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, CORPUS_D - 2))]
    if allowed_centers:
        kept = [c for c in cs if c in set(allowed_centers)]
        if kept:
            cs = kept
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]

def read_windows(vol_row, centers, res=384):
    vol = np.asarray(vol_row)
    d = vol.shape[0]
    out = []
    for c in centers:
        c = max(1, min(int(c), d - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        out.append(t)
    return torch.stack(out, 0)

class RaptorClassifier(nn.Module):
    def __init__(self, backbone, f_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(
            nn.Linear(f_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n)
        )
        self.cls_w = nn.Parameter(torch.zeros(n, f_dim))
        self.cls_b = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.cls_w, std=0.02)
        self.n = n

    def encode(self, x, chunk=24):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        if chunk is not None and len(flat) > chunk:
            feats = [self.backbone(flat[i:i + chunk]) for i in range(0, len(flat), chunk)]
            f = torch.cat(feats, dim=0)
        else:
            f = self.backbone(flat)
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.cls_w).sum(-1) + self.cls_b
        return logits

    def forward(self, x, chunk=24):
        return self.head(self.encode(x, chunk=chunk))

class S4SequenceClassifier(RaptorClassifier):
    def __init__(self, backbone, f_dim=768, n=12, drop=0.2, k_max=20, layers=2):
        super().__init__(backbone, f_dim, n, drop)
        self.pos_embed = nn.Parameter(torch.zeros(1, k_max, f_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        el = nn.TransformerEncoderLayer(f_dim, 8, 2 * f_dim, dropout=0.1, batch_first=True, norm_first=True)
        self.seq = nn.TransformerEncoder(el, layers, enable_nested_tensor=False)
        self._pos = None

    def set_positions(self, pos):
        self._pos = pos

    def head(self, feats):
        if self._pos is not None:
            pos = self._pos.to(feats.device)
            feats = self.seq(feats + self.pos_embed[:, pos[0]])
            self._pos = None
        return super().head(feats)

def build_model(arch, ckpt_dict, device=DEVICE):
    import timm
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "eva", "beit"))
    kw = dict(pretrained=False, num_classes=0, in_chans=3)
    kw["global_pool"] = "token" if is_vit else "avg"
    if is_vit:
        kw["dynamic_img_size"] = True
        kw["dynamic_img_pad"] = True
    bb = timm.create_model(arch, **kw)
    f_dim = bb.num_features
    head_type = ckpt_dict.get("head") or "mil"
    if head_type == "s4seq":
        model = S4SequenceClassifier(bb, f_dim=f_dim, n=N_LABELS).to(device)
    else:
        model = RaptorClassifier(bb, f_dim=f_dim, n=N_LABELS).to(device)
    state = ckpt_dict["state_dict"]
    model.load_state_dict({kk: v for kk, v in state.items()})
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 4. Zero-I/O Forward Inference Pipeline
# -----------------------------------------------------------------------------
def predict_from_shared_cache(model, tv, tm, uids, res=384, k=42, allowed_centers=None, chunk=24):
    preds = []
    mean = _MEAN.to(DEVICE)
    std = _STD.to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(uids), 2):
            xs = []
            for r in range(i, min(i + 2, len(uids))):
                cs = centers_from_mask(tm[r], k=k, allowed_centers=allowed_centers)
                xs.append(read_windows(tv[r], cs, res=res))
            x = torch.stack(xs, 0)
            b, kk = x.shape[:2]
            x = ((x.flatten(0, 1) - _MEAN) / _STD).view(b, kk, *x.shape[2:])
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
                out = model(x.to(DEVICE), chunk=chunk).float()
            preds.append(torch.sigmoid(out).cpu().numpy())
    return np.concatenate(preds, axis=0)

def find_checkpoint(dataset_keyword, filename="corpus0_topk_avg.pt"):
    roots = ["/kaggle/input", "/kaggle/input/datasets"]
    cands = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dp, _, files in os.walk(r):
            if dataset_keyword in dp.lower() and filename in files:
                cands.append(os.path.join(dp, filename))
    if not cands:
        for r in roots:
            if not os.path.isdir(r):
                continue
            for dp, _, files in os.walk(r):
                if filename in files:
                    cands.append(os.path.join(dp, filename))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"Cannot find checkpoint {filename} matching {dataset_keyword}")

# -----------------------------------------------------------------------------
# 5. Main Execution Flow
# -----------------------------------------------------------------------------
def main():
    t_global_0 = time.time()
    
    # Locate test dataset
    comp_roots = ["/kaggle/input/rsna-knee-abnormality-detection", "/kaggle/input/competitions/rsna-knee-abnormality-detection"]
    comp_dir = None
    for cr in comp_roots:
        if os.path.isfile(os.path.join(cr, "test.csv")):
            comp_dir = cr
            break
    if not comp_dir:
        print("WARNING: Competition test.csv not found! Emitting placeholder submission.", flush=True)
        sub = pd.DataFrame({"StudyInstanceUID": ["dummy"], **{lbl: [0.5] for lbl in LABELS}})
        sub.to_csv("/kaggle/working/submission.csv", index=False)
        return

    test_csv = pd.read_csv(os.path.join(comp_dir, "test.csv"), dtype={"StudyInstanceUID": str})
    test_series = pd.read_csv(os.path.join(comp_dir, "test_series.csv"), dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    series_root = os.path.join(comp_dir, "test_series")
    if not os.path.isdir(series_root):
        series_root = os.path.join(comp_dir, "test_images")

    uids = test_csv["StudyInstanceUID"].tolist()
    study_to_series = {k: v.to_dict("records") for k, v in test_series.groupby("StudyInstanceUID")}
    print(f"[Test Dataset] Found {len(uids)} test studies.", flush=True)

    # Phase 0: Build Shared Cache ONCE
    build_shared_test_corpus(uids, study_to_series, series_root, n_workers=12)
    
    # Memory-map the shared cache
    tv = np.load(str(VOLS_MMAP), mmap_mode="r")
    tm = np.load(str(MASKS_FILE))
    print(f"[Shared Cache Attached] Shape: {tv.shape}, dtype: {tv.dtype}", flush=True)

    # -------------------------------------------------------------------------
    # Phase 1: Pure GPU Forward Passes (Zero-I/O)
    # -------------------------------------------------------------------------
    # 1. Model C6 (CoAtNet-2 384)
    t0 = time.time()
    p_c6_path = find_checkpoint("c6-weights", "c6_corpus0.pt")
    print(f"\n[1/4] Running C6 (CoAtNet-2 384) from {p_c6_path}...", flush=True)
    ck_c6 = torch.load(p_c6_path, map_location="cpu", weights_only=False)
    m_c6 = build_model("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", ck_c6)
    pred_c6 = predict_from_shared_cache(m_c6, tv, tm, uids, res=384, k=42, chunk=24)
    print(f"      C6 inference completed in {(time.time() - t0)/60:.1f} min! Preds: {pred_c6.shape}", flush=True)
    del m_c6, ck_c6; torch.cuda.empty_cache(); gc.collect()

    # 2. Model C14 (ConvNeXt V2-Base 384)
    t0 = time.time()
    p_c14_path = find_checkpoint("c14-weights", "corpus0_topk_avg.pt")
    print(f"\n[2/4] Running C14 (ConvNeXt V2-Base 384) from {p_c14_path}...", flush=True)
    ck_c14 = torch.load(p_c14_path, map_location="cpu", weights_only=False)
    m_c14 = build_model("convnextv2_base.fcmae_ft_in22k_in1k_384", ck_c14)
    pred_c14 = predict_from_shared_cache(m_c14, tv, tm, uids, res=384, k=42, chunk=24)
    print(f"      C14 inference completed in {(time.time() - t0)/60:.1f} min! Preds: {pred_c14.shape}", flush=True)
    del m_c14, ck_c14; torch.cuda.empty_cache(); gc.collect()

    # 3. Model C12 (ConvNeXt-Small 384)
    t0 = time.time()
    p_c12_path = find_checkpoint("c12-weights", "corpus0_topk_avg.pt")
    print(f"\n[3/4] Running C12 (ConvNeXt-Small 384) from {p_c12_path}...", flush=True)
    ck_c12 = torch.load(p_c12_path, map_location="cpu", weights_only=False)
    m_c12 = build_model("convnext_small.fb_in22k_ft_in1k_384", ck_c12)
    pred_c12 = predict_from_shared_cache(m_c12, tv, tm, uids, res=384, k=42, chunk=24)
    print(f"      C12 inference completed in {(time.time() - t0)/60:.1f} min! Preds: {pred_c12.shape}", flush=True)
    del m_c12, ck_c12; torch.cuda.empty_cache(); gc.collect()

    # 4. Model PF-v1 Specialist (PF OA Specialist)
    t0 = time.time()
    p_pf1_path = find_checkpoint("pf1-weights", "corpus0_topk_avg.pt")
    print(f"\n[4/4] Running PF-v1 Specialist from {p_pf1_path}...", flush=True)
    ck_pf1 = torch.load(p_pf1_path, map_location="cpu", weights_only=False)
    pf_allow = ck_pf1.get("centre_allow", [1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16])
    m_pf1 = build_model("convnext_small.fb_in22k_ft_in1k_384", ck_pf1)
    pred_pf1 = predict_from_shared_cache(m_pf1, tv, tm, uids, res=384, k=20, allowed_centers=pf_allow, chunk=24)
    print(f"      PF-v1 inference completed in {(time.time() - t0)/60:.1f} min! Preds: {pred_pf1.shape}", flush=True)
    del m_pf1, ck_pf1; torch.cuda.empty_cache(); gc.collect()

    # -------------------------------------------------------------------------
    # Phase 2: Per-Label Calibrated Rank Fusion
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Performing Calibrated Percentile Rank Fusion...", flush=True)
    df_c6 = pd.DataFrame(pred_c6, columns=LABELS)
    df_c14 = pd.DataFrame(pred_c14, columns=LABELS)
    df_c12 = pd.DataFrame(pred_c12, columns=LABELS)
    df_pf1 = pd.DataFrame(pred_pf1, columns=LABELS)

    r_c6 = df_c6.rank(method="average", pct=True)
    r_c14 = df_c14.rank(method="average", pct=True)
    r_c12 = df_c12.rank(method="average", pct=True)
    r_pf1 = df_pf1.rank(method="average", pct=True)

    # General Findings: CoAtNet-2 (0.50) + ConvNeXt-Base (0.30) + ConvNeXt-Small (0.20)
    r_general = 0.50 * r_c6 + 0.30 * r_c14 + 0.20 * r_c12

    # Final Ensemble Matrix
    final_df = r_general.copy()
    
    # Specific Boost: Inject PF-v1 Specialist into PF OA (W=0.40)
    final_df["PF OA"] = 0.60 * r_general["PF OA"] + 0.40 * r_pf1["PF OA"]

    # Final percentile ranking
    final_rank = final_df.rank(method="average", pct=True)

    # Format output submission
    sub = pd.DataFrame({"StudyInstanceUID": uids})
    for col in LABELS:
        sub[col] = final_rank[col]

    # -------------------------------------------------------------------------
    # Phase 3: Fail-Closed Validation Gate & Working Directory Cleanup
    # -------------------------------------------------------------------------
    print("\n[Phase 3] Running Fail-Closed Validation Gate...", flush=True)
    assert len(sub) == len(uids), f"Row count mismatch: {len(sub)} != {len(uids)}"
    assert sub.columns.tolist() == ["StudyInstanceUID", *LABELS], "Column schema drift"
    assert not sub["StudyInstanceUID"].isna().any(), "NaN in StudyInstanceUID"
    mat = sub[LABELS].to_numpy(np.float64)
    assert np.isfinite(mat).all(), "Non-finite predictions detected"
    assert ((mat >= 0.0) & (mat <= 1.0)).all(), "Predictions outside [0, 1]"
    if len(sub) > 10:
        assert (mat.std(axis=0) > 1e-6).all(), "Constant prediction column detected"

    out_file = Path("/kaggle/working/submission.csv")
    sub.to_csv(out_file, index=False)
    print(f"SUCCESS: submission.csv created! Shape: {sub.shape}, sha256: {hashlib.sha256(out_file.read_bytes()).hexdigest()[:16]}", flush=True)

    # Clean up the large shared memmap to save disk space
    try:
        del tv
        if VOLS_MMAP.exists():
            VOLS_MMAP.unlink()
        if MASKS_FILE.exists():
            MASKS_FILE.unlink()
        if UIDS_FILE.exists():
            UIDS_FILE.unlink()
        if CACHE_DIR.exists():
            import shutil
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
        print("[Cleanup] Removed temporary shared memmap cache. Disk is clean.", flush=True)
    except Exception as e:
        print(f"[Cleanup Note] Cache cleanup notice: {e}", flush=True)

    elapsed_min = (time.time() - t_global_0) / 60
    print(f"ALL PHASES COMPLETE in {elapsed_min:.1f} minutes! (Kaggle timeout limit is 540 min)", flush=True)

if __name__ == "__main__":
    main()
