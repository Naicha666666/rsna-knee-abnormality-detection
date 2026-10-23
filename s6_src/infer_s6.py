"""S6 ConvNeXt-V2-Base Reader: 12-Class Spatial Attention & Cross-Window Pooling Inference.

Decodes test DICOMs exactly matching the 64-slice corpus (336x336, 140mm crop, 2-98% percentile)
and evaluates S6SpatialConvNeXtV2 to produce calibrated pathology predictions.
"""
import os
import sys
import glob
import math
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]
C = len(LABELS)

CORPUS_D = 64
CORPUS_IMG = 336
CORPUS_CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.06, 0.94
PCT_LO, PCT_HI = 2.0, 98.0
CORPUS_SLOTS = [
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", -1, 12)
]

def build_spatial_pe(h, w, d):
    y = torch.arange(h).unsqueeze(1).repeat(1, w) / max(h - 1, 1)
    x = torch.arange(w).unsqueeze(0).repeat(h, 1) / max(w - 1, 1)
    pe = []
    for freq in [1.0, 2.0, 4.0, 8.0]:
        pe.extend([
            torch.sin(freq * math.pi * x), torch.cos(freq * math.pi * x),
            torch.sin(freq * math.pi * y), torch.cos(freq * math.pi * y)
        ])
    return torch.stack(pe, dim=-1).reshape(h * w, -1)

class S6SpatialConvNeXtV2(nn.Module):
    def __init__(self, backbone="convnextv2_base", chunk_size=2, hidden=384, drop=0.0, in_chans=3):
        super().__init__()
        self.backbone = backbone
        self.chunk_size = chunk_size
        self.enc = timm.create_model(backbone, pretrained=False, num_classes=0, in_chans=in_chans, global_pool="")
        d = self.enc.num_features
        self.sp_key = nn.Linear(d, 128)
        self.sp_pos = nn.Linear(16, 128, bias=False)
        self.sp_q = nn.Parameter(torch.zeros(C, 128))
        self.win_proj = nn.Linear(d, hidden)
        self.win_attn_q = nn.Parameter(torch.zeros(C, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(hidden, 1)

    def forward(self, windows, win_mask=None):
        B, W, Ch, H, W_img = windows.shape
        x = windows.view(B * W, Ch, H, W_img).float().div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        total_w = B * W
        feats = []
        for i in range(0, total_w, self.chunk_size):
            x_chunk = x[i:i + self.chunk_size]
            feats.append(self.enc(x_chunk))
        feat_map = torch.cat(feats, dim=0)

        BW, d, h_f, w_f = feat_map.shape
        N_pts = h_f * w_f
        feat_flat = feat_map.permute(0, 2, 3, 1).reshape(BW, N_pts, d)

        keys = self.sp_key(feat_flat)
        pe = build_spatial_pe(h_f, w_f, 128).to(x.device)
        pos = self.sp_pos(pe).unsqueeze(0)
        keys = keys + pos

        scores = torch.einsum("cd,bwd->bcw", self.sp_q, keys) / math.sqrt(128)
        sp_weights = F.softmax(scores, dim=-1)

        class_tokens = torch.einsum("bcw,bwd->bcd", sp_weights, feat_flat).view(B, W, C, d)
        class_tokens = self.win_proj(class_tokens)

        attn_scores = torch.einsum("ch,bwch->bcw", self.win_attn_q, class_tokens) / math.sqrt(class_tokens.shape[-1])
        if win_mask is not None:
            mask_exp = win_mask.unsqueeze(1).expand(B, C, W)
            attn_scores = attn_scores.masked_fill(~mask_exp, float("-inf"))

        win_weights = F.softmax(attn_scores, dim=-1)
        win_weights = torch.nan_to_num(win_weights, nan=0.0)

        study_tokens = torch.einsum("bcw,bwch->bch", win_weights, class_tokens)
        study_tokens = self.dropout(self.norm(study_tokens))
        logits = self.head(study_tokens).squeeze(-1)
        return logits

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

def _safe_read_px(path):
    try:
        ds = pydicom.dcmread(path, force=True)
        a = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        a = a * slope + inter
        ps = float(ds.PixelSpacing[0]) if "PixelSpacing" in ds else 0.0
        return a, ps
    except Exception:
        return None, 0.0

def _crop_resize(a, ps):
    h, w = a.shape
    cpx = int(round(CORPUS_CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(a, (CORPUS_IMG, CORPUS_IMG), interpolation=cv2.INTER_AREA)

def build_test_volume(sid, rows, series_root, max_workers=8):
    vol = np.zeros((CORPUS_D, CORPUS_IMG, CORPUS_IMG), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in CORPUS_SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        folder = os.path.join(series_root, sid, r["SeriesInstanceUID"])
        files, med_ps = _order_series_files(folder)
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * SPAN_LO), int(n * SPAN_HI) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else np.zeros(k, dtype=int)
        
        sel_files = [files[min(int(p), n - 1)] for p in picks]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_safe_read_px, sel_files))
            
        arrs = [r[0] for r in results]
        pss = [r[1] if r[1] > 0 else med_ps for r in results]
        
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

def extract_study_windows(vol, n_win=20, train=False):
    S, H, W = vol.shape
    cands = []
    for k in range(1, S - 1):
        cands.append((k - 1, k, k + 1))
    stride = max(1, len(cands) // n_win)
    picks = cands[::stride][:n_win]
    wins = np.stack([vol[[i, j, k]] for i, j, k in picks], axis=0)
    return wins

def find_test_data():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("test root not found under /kaggle/input")

def find_s6_weights():
    cands = [
        "/kaggle/input/rsna-knee-s6-convnextv2-weights/s6_convnextv2_base_fold0_best.pt",
        "/kaggle/input/easonyanyan/rsna-knee-s6-convnextv2-weights/s6_convnextv2_base_fold0_best.pt"
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    for d in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in d.lower() or "abnormality-detection" in d:
            continue
        hits = glob.glob(os.path.join(d, "**", "s6_convnextv2_base_fold0_best.pt"), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError("s6_convnextv2_base_fold0_best.pt not found under /kaggle/input")

def main():
    t0 = time.time()
    test_root = find_test_data()
    ckpt_path = find_s6_weights()
    print(f"[S6 Reader] Test root: {test_root}", flush=True)
    print(f"[S6 Reader] Checkpoint: {ckpt_path}", flush=True)

    test_df = pd.read_csv(os.path.join(test_root, "test.csv"))
    test_series_df = pd.read_csv(os.path.join(test_root, "test_series.csv"))
    series_dir = os.path.join(test_root, "test_series") if os.path.exists(os.path.join(test_root, "test_series")) else os.path.join(test_root, "test_images")

    test_ids = [str(x) for x in test_df["StudyInstanceUID"].tolist()]
    print(f"[S6 Reader] Found {len(test_ids)} test studies", flush=True)

    series_by_study = {}
    for r in test_series_df.to_dict("records"):
        sid = str(r["StudyInstanceUID"])
        series_by_study.setdefault(sid, []).append(r)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S6 Reader] Initializing model on device: {device}", flush=True)
    model = S6SpatialConvNeXtV2(backbone="convnextv2_base", chunk_size=2, hidden=384, drop=0.0).to(device)
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    model.eval()
    print("[S6 Reader] Weights loaded successfully!", flush=True)

    preds_list = []
    with torch.no_grad():
        for idx, sid in enumerate(test_ids):
            rows = series_by_study.get(sid, [])
            try:
                vol, mask = build_test_volume(sid, rows, series_dir)
                wins = extract_study_windows(vol, n_win=20, train=False)
                x = torch.from_numpy(wins).unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                if np.isnan(probs).any():
                    print(f"  [Study {idx+1}/{len(test_ids)}] {sid[:12]} NaN detected, defaulting to 0.5", flush=True)
                    probs = np.full(C, 0.5, dtype=np.float32)
            except Exception as e:
                print(f"  [Study {idx+1}/{len(test_ids)}] {sid[:12]} Error ({type(e).__name__}: {e}), defaulting to 0.5", flush=True)
                probs = np.full(C, 0.5, dtype=np.float32)
            preds_list.append(probs)
            if (idx + 1) % 20 == 0 or idx + 1 == len(test_ids):
                print(f"  [S6 Reader] Processed {idx+1}/{len(test_ids)} studies | {time.time()-t0:.1f}s", flush=True)

    out_df = pd.DataFrame(np.array(preds_list, dtype=np.float32), columns=LABELS)
    out_df.insert(0, "StudyInstanceUID", test_ids)
    out_csv = "/kaggle/working/_own_s6.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[S6 Reader] Successfully wrote {len(out_df)} predictions to {out_csv} in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
