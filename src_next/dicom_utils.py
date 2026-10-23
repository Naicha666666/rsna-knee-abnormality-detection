"""DICOM loading & slice preprocessing for RSNA Knee competition.

Incorporates lessons from top public notebooks (see LOG.md):
- Physical-scale crop: crop a constant mm extent (CROP_MM) then resize, so the
  encoder sees a consistent mm/pixel regardless of acquisition (0.71 mm/px @224).
- Volume-wise percentile normalization (1st-99th) over the WHOLE volume, not per slice.
- Central 60% slices (outer slices are mostly off-joint soft tissue).
- Laterality normalization (left/right): coronal & axial -> horizontal flip;
  sagittal -> reverse slice order. No-op when the tag is absent (graceful fallback).
- Header tags are read when present but the code degrades gracefully if stripped
  (our local test sample had them removed), falling back to fixed crops / no flip.
"""
from __future__ import annotations
import os
import glob
import re
import numpy as np
from PIL import Image
import pydicom

from config import (IMG_SIZE, SLICES_PER_SLOT, CROP_MM, WIN_PMIN, WIN_PMAX,
                    CENTRAL_SLICE_FRAC, LAT_MIN_OFFSET_MM)


def load_pixel_array(path: str) -> np.ndarray:
    """Read a DICOM file and return its 2D pixel array (H, W) as float32.
    Applies RescaleSlope/Intercept only if present."""
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or inter != 0.0:
        arr = arr * slope + inter
    return arr


def _pixel_spacing_mm(ds) -> float:
    """Return row spacing in mm/pixel, or 1.0 if missing."""
    try:
        ps = getattr(ds, "PixelSpacing", None)
        if ps is not None and len(ps) >= 1:
            return float(ps[0])
    except Exception:
        pass
    return 1.0


def _physical_crop(arr: np.ndarray, spacing: float, crop_mm: float,
                   out_size: int) -> np.ndarray:
    """Center-crop `crop_mm` millimetres of anatomy, then resize to out_size.

    Resize stays in float so per-slice min/max cannot scramble the volume-wise
    percentile window that runs after stacking.
    """
    crop_px = int(round(crop_mm / max(spacing, 1e-3)))
    h, w = arr.shape
    ch = min(crop_px, h); cw = min(crop_px, w)
    y0 = (h - ch) // 2; x0 = (w - cw) // 2
    arr = arr[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    img = Image.fromarray(arr, mode="F")
    img = img.resize((out_size, out_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _volume_normalize(vol: np.ndarray) -> np.ndarray:
    """Min-max scale a volume to [0,1] using its 1st/99th percentile."""
    lo, hi = np.percentile(vol, [WIN_PMIN, WIN_PMAX])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def normalise_laterality(vol: np.ndarray, plane: str, lat: str | None):
    """Map a right knee onto a left-knee convention.

    `vol` is [S, H, W]. Coronal/axial: horizontal flip. Sagittal: reverse the
    *slice order* (ML is the through-plane axis — flipping rows would be a
    vertical flip and would swap femur/tibia). Unknown laterality is a no-op.
    """
    if lat is None:
        return vol
    if str(lat).upper().startswith("R"):
        if str(plane).lower().startswith("sag"):
            return vol[::-1]
        return vol[:, :, ::-1]
    return vol


def _natural_key(p):
    """Digit-aware filename key so '.../10.dcm' sorts after '.../9.dcm'."""
    base = os.path.basename(p)
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", base)]


def _spatial_sort(files):
    """Order slices along the through-plane axis (the ImagePositionPatient
    component with the LARGEST spread across slices), not a fixed axis: for a
    sagittal knee the stack direction is medial-lateral (patient x), so sorting
    on z would scramble the 2.5D groups. Requires >=80% of slices positioned,
    else InstanceNumber (>=80%), else a natural filename sort."""
    n = len(files)
    pos, inst = [], []
    for p in files:
        ipp = inum = None
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            inum = getattr(ds, "InstanceNumber", None)
        except Exception:
            pass
        pos.append([float(v) for v in ipp] if ipp is not None and len(ipp) >= 3
                   else None)
        try:
            inum = int(inum) if inum is not None else None
        except Exception:
            inum = None
        inst.append(inum)
    n_pos = sum(v is not None for v in pos)
    if n_pos >= max(2, int(0.8 * n)):
        arr = np.array([v if v is not None else [np.nan] * 3 for v in pos],
                       dtype=float)
        axis = int(np.nanargmax(np.nanmax(arr, 0) - np.nanmin(arr, 0)))
        key = arr[:, axis]
        order = sorted(range(n), key=lambda i: (np.isnan(key[i]), key[i]))
        return [files[i] for i in order]
    n_inst = sum(v is not None for v in inst)
    if n_inst >= max(2, int(0.8 * n)):
        order = sorted(range(n), key=lambda i: (inst[i] is None, inst[i] or 0))
        return [files[i] for i in order]
    return sorted(files, key=_natural_key)


def load_series_stack(folder: str, plane: str = "Sagittal", lat: str | None = None,
                      n_slices: int = SLICES_PER_SLOT, out_size: int = IMG_SIZE,
                      crop_mm: float = CROP_MM) -> np.ndarray:
    """Load a series folder -> [n_slices, H, W] float32 in [0, 1].

    Headers are used when present (PixelSpacing, ImagePositionPatient for
    through-plane order). Laterality is applied to the stacked volume so a
    sagittal right-knee correction reverses slice order rather than flipping
    each frame vertically. Missing/empty folders return zeros.
    """
    empty = np.zeros((n_slices, out_size, out_size), dtype=np.float32)
    if not folder or not os.path.isdir(folder):
        return empty
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not files:
        return empty

    files = _spatial_sort(files)
    n = len(files)
    a = int(n * (1 - CENTRAL_SLICE_FRAC) / 2)
    b = max(a + 1, n - a)
    files = files[a:b]

    spacing = None
    planes = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if spacing is None:
                spacing = _pixel_spacing_mm(ds)
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            if slope != 1.0 or inter != 0.0:
                arr = arr * slope + inter
            arr = _physical_crop(arr, spacing or 1.0, crop_mm, out_size)
            planes.append(arr)
        except Exception:
            continue
    if not planes:
        return empty

    vol = np.stack(planes, 0)                 # [M, H, W]
    vol = _volume_normalize(vol)
    vol = normalise_laterality(vol, plane, lat)
    M = vol.shape[0]
    idx = np.linspace(0, M - 1, n_slices).round().astype(int)
    out = np.zeros((n_slices, out_size, out_size), dtype=np.float32)
    for i, fi in enumerate(idx):
        out[i] = vol[int(np.clip(fi, 0, M - 1))]
    return out


def load_series(folder: str, plane: str = "Sagittal", lat: str | None = None,
                n_slices: int = SLICES_PER_SLOT, out_size: int = IMG_SIZE,
                crop_mm: float = CROP_MM) -> np.ndarray:
    """Legacy wrapper: [n_slices, 1, H, W] float32 in [0, 1]."""
    stack = load_series_stack(folder, plane, lat, n_slices, out_size, crop_mm)
    return stack[:, None, :, :]


def _tag_side_from_ds(ds) -> str | None:
    lat = getattr(ds, "Laterality", None) or getattr(ds, "ImageLaterality", None)
    if not lat:
        return None
    s = str(lat).strip().upper()
    if s.startswith("L"):
        return "L"
    if s.startswith("R"):
        return "R"
    return None


def _position_side_from_ds(ds) -> str | None:
    """LPS: the right knee sits at negative patient-x. Near-isocentre is ignored."""
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None:
        return None
    try:
        x = float(ipp[0])
    except Exception:
        return None
    if abs(x) < LAT_MIN_OFFSET_MM:
        return None
    return "R" if x < 0 else "L"


def read_laterality_full(folder: str) -> tuple:
    """Return (tag_side, pos_side), each 'L'/'R'/None, from a series folder."""
    if not folder or not os.path.isdir(folder):
        return None, None
    files = glob.glob(os.path.join(folder, "*.dcm"))
    if not files:
        return None, None
    tag, xs = None, []
    for f in files[:20]:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            if tag is None:
                tag = _tag_side_from_ds(ds)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None:
                try:
                    xs.append(float(ipp[0]))
                except Exception:
                    pass
        except Exception:
            continue
    pos = None
    if xs:
        x = float(np.median(xs))
        if abs(x) >= LAT_MIN_OFFSET_MM:
            pos = "R" if x < 0 else "L"
    return tag, pos


# --------------------------------------------------------------------------- #
# sequence recovery (v14)
# --------------------------------------------------------------------------- #
# The competition CSV's Fluid_Sensitive and Fat_Suppression are perfectly correlated
# in this corpus -- every series is (0,0) or (1,1) -- so between them they carry one
# bit, and the six-slot scheme collapses to three (plane x that bit). The reference
# pipeline instead RECOVERS both from the header, where the weighting (T1 / PD / T2 /
# GRE) and the fat-suppression flag are independent. That is what makes a
# "fluid-sensitive, not fat-suppressed" slot a real thing rather than an empty one.
_SEQ_SEP = re.compile(r"[_\-/\\]+")
_FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat sat|fat_sat|spair|spir|stir|\bsp[ai]r\b|"
    # `t2 fs` without a right-hand boundary also matches "T2 FSE" -- fast spin echo,
    # which is not fat-suppressed. Anchor the fs token on both sides.
    r"chess|\bfsat\b|frfse fs\b|t2 fs\b|pd fs\b|suppress")
_T1_RX = re.compile(r"\bt1\b|t1w|t1 tse|t1 fse|mprage")
_T2_RX = re.compile(r"\bt2\b|t2w|t2 tse|t2 fse|stir")
_PD_RX = re.compile(r"\bpd\b|pdw|proton|dp\b|densidad")
# GE writes SAT_GEMS for spatial saturation, so ScanOptions is matched as exact
# tokens; a substring test on "SAT" fires on series that are not fat-suppressed.
_FATSAT_OPTS = {"FS", "FSAT", "SP", "SPIR", "SPAIR", "CHEMSAT"}

_SEQ_FIELDS = ("SeriesDescription", "SequenceName", "ScanOptions",
               "RepetitionTime", "EchoTime", "ScanningSequence")


def read_series_meta(folder: str) -> dict:
    """Header fields needed to recover (weighting, fat suppression) for a series.

    Reads at most a handful of files and stops at the first one that carries a
    SeriesDescription, because these fields are per-series constants. Returns {}
    when nothing is readable, which the caller treats as "fall back to the CSV".
    """
    out = {}
    if not folder or not os.path.isdir(folder):
        return out
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))[:5]
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        for k in _SEQ_FIELDS:
            v = getattr(ds, k, None)
            if v is None or k in out:
                continue
            if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                v = "|".join(str(x) for x in v)
            out[k] = str(v)
        if out.get("SeriesDescription"):
            break
    return out


def recover_sequence(meta: dict):
    """(fluid, fatsat, weight) from header fields, or None when unrecoverable.

    Mirrors the reference pipeline's `annotate()`: the description/sequence name
    decides the weighting when it says so, otherwise TR/TE decide, and a gradient-
    echo scanning sequence is called GRE rather than guessed at.
    """
    if not meta:
        return None
    desc = (meta.get("SeriesDescription", "") + " " +
            meta.get("SequenceName", "")).lower()
    desc = _SEQ_SEP.sub(" ", desc)
    opts = [t.strip().upper() for t in meta.get("ScanOptions", "").split("|")]
    fatsat = bool(_FATSAT_RX.search(desc)) or any(t in _FATSAT_OPTS for t in opts)

    def _num(k):
        try:
            return float(meta.get(k))
        except (TypeError, ValueError):
            return None

    tr, te = _num("RepetitionTime"), _num("EchoTime")
    gre = "GR" in meta.get("ScanningSequence", "").upper()
    t1 = bool(_T1_RX.search(desc))
    t2 = bool(_T2_RX.search(desc))
    pdw = bool(_PD_RX.search(desc))
    if t1 and not t2 and not pdw:
        weight = "T1"
    elif t2 and not pdw:
        weight = "T2"
    elif pdw:
        weight = "PD"
    elif gre:
        weight = "GRE"
    elif tr is not None and tr < 800:
        weight = "T1"
    elif te is not None and te > 60:
        weight = "T2"
    elif tr is not None and tr >= 800:
        weight = "PD"
    else:
        return None                      # nothing in the header decided it
    return (weight in ("PD", "T2")), fatsat, weight


def read_laterality(folder: str) -> str | None:
    """Return 'L'/'R' from DICOM tags, else None (no IPP fallback here).

    IPP fallback is decided at cache-build time after measuring agreement
    with the tag, because a wrong flip is worse than no flip.
    """
    tag, _pos = read_laterality_full(folder)
    return tag


# ---- backward-compatible one-file helper (used by early smoke tests) ----
def preprocess_dicom(path: str, size: int = IMG_SIZE) -> np.ndarray:
    arr = load_pixel_array(path)
    lo, hi = np.percentile(arr, [WIN_PMIN, WIN_PMAX])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0
