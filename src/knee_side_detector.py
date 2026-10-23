"""Knee Laterality and Orientation Verifier.

Ported from heliosli/rsna-knee-blend-gold (side_common.py).

In knee MRI, DICOM tags for laterality (Left vs Right) can be corrupted, missing,
or inconsistent. If a knee is mirrored horizontally:
- Medial Meniscus is swapped with Lateral Meniscus!
- Medial Compartment OA is swapped with Lateral Compartment OA!

This module provides 2D slice sampling and orientation detection to automatically
validate canonical lateral-at-+column orientation and fix inverted knees.
"""
from __future__ import annotations
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

SIDE_SIZE = 160
SIDE_FRACS = (0.4, 0.5, 0.6)
SIDE_PLANES = ("Coronal", "Axial")


def extract_side_slices(vol: np.ndarray, target_size: int = SIDE_SIZE) -> np.ndarray:
    """Extract and normalize canonical middle slices for laterality orientation checking.
    
    Args:
        vol: 3D numpy array [depth, H, W] of native volume
        target_size: output slice resolution (default 160)
        
    Returns:
        uint8 array [k, target_size, target_size] where k <= 3
    """
    n = int(vol.shape[0])
    if n < 1:
        return np.zeros((0, target_size, target_size), dtype=np.uint8)
    
    indices = sorted({int(round(f * (n - 1))) for f in SIDE_FRACS})
    out = []
    for i in indices:
        im = np.asarray(vol[i], dtype=np.float32)
        # Fast percentile computation with stride 4
        lo, hi = np.percentile(im[::4, ::4], [1.0, 99.0])
        denom = max(float(hi - lo), 1e-6)
        norm_im = np.clip((im - lo) / denom, 0.0, 1.0)
        
        if HAS_CV2:
            resized = cv2.resize(norm_im, (target_size, target_size), interpolation=cv2.INTER_AREA)
        else:
            # Fallback simple nearest/linear if cv2 is absent
            import scipy.ndimage
            scale_y = target_size / norm_im.shape[0]
            scale_x = target_size / norm_im.shape[1]
            resized = scipy.ndimage.zoom(norm_im, (scale_y, scale_x), order=1)
            
        out.append(np.rint(resized * 255.0).astype(np.uint8))
        
    return np.stack(out)


def build_side_net():
    """Create lightweight ResNet-34 side orientation classifier."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to build the side orientation network.")
    import timm
    return timm.create_model("resnet34", pretrained=False, num_classes=1, in_chans=1)


# ---------------------------------------------------------------------------
# Geometry fast path -- no torch, no cv2, no network required.
# ---------------------------------------------------------------------------
# Measured offline on the full training corpus
# (results/b2_laterality/laterality_headers.json, 4,407 studies x 17,509 series,
#  re-analysed 2026-10-10):
#   * `Laterality` tag is present in only 49.9% of studies (2,201 / 4,407) and
#     `ImageLaterality` is 0%; the competition CSVs (train_series.csv) carry NO
#     laterality column at all; only 20.5% of reports mention left/right.
#   * `x_median` (derived from ImagePositionPatient) is present for 100% of
#     studies, and all series of a study share its sign in 99.95% of cases.
#   * agreement(tag, sign(x_median)) is strongly monotone in |x_median|:
#       |x|<5mm 54.6% (n=33) | 5-10 59.5% (n=37) | 10-20 72.3% (n=47)
#       20-40 96.1% (n=152) | 40-80 99.1% (n=1971) | 80-150 98.5% (n=2032)
#     => |x_median| >= 20mm gives 98.70% agreement and covers 97.2% of studies.
# Conclusion: a trained ResNet-34 laterality head is NOT required. Use this
# 3-tier rule instead. It matters for the Stage-2 anchor bank, whose local
# 100mm zooms (PF / popliteal fossa) land on the wrong anatomical side if the
# knee is mirrored -- whole-knee readers do not need it (LOG #1523/#1525).
SIDE_GEOM_MIN_MM = 20.0
SIDE_GEOM_AGREEMENT = 0.987   # measured on |x_median| >= 20mm, n=4,158 series
SIDE_GEOM_COVERAGE = 0.972    # measured at study level, 4,285 / 4,407


def side_from_geometry(x_medians_mm):
    """Decide knee side from ImagePositionPatient x-coordinates.

    DICOM uses the LPS convention, so +x is towards the patient's LEFT.

    Args:
        x_medians_mm: iterable of per-series median x coordinates (mm). None
            entries are ignored.

    Returns:
        dict with keys: side ('L'/'R'/None), ambiguous (bool), reason,
        mean_x_mm, n, consistent.
    """
    xs = [float(v) for v in x_medians_mm if v is not None]
    if not xs:
        return {"side": None, "ambiguous": True, "reason": "no_geometry",
                "mean_x_mm": None, "n": 0, "consistent": False}

    mean_x = sum(xs) / len(xs)
    consistent = len({(x > 0) for x in xs}) == 1
    if abs(mean_x) < SIDE_GEOM_MIN_MM:
        # Near the isocenter the sign is a coin flip (54-72% agreement).
        return {"side": None, "ambiguous": True, "reason": "near_isocenter",
                "mean_x_mm": mean_x, "n": len(xs), "consistent": consistent}

    return {"side": "L" if mean_x > 0 else "R", "ambiguous": False,
            "reason": "geometry", "mean_x_mm": mean_x, "n": len(xs),
            "consistent": consistent}


def resolve_side(x_medians_mm, laterality_tag=None):
    """3-tier laterality rule: geometry -> DICOM tag -> no-flip.

    Tier 1 (|mean x| >= 20mm) is correct 98.7% of the time and fires for 97.2%
    of studies. Tier 2 uses the `Laterality` tag when present (half the corpus).
    Tier 3 returns side=None, which callers must treat as "do not mirror" --
    the safe default, since flipping a correct knee swaps medial with lateral.
    """
    geo = side_from_geometry(x_medians_mm)
    if not geo["ambiguous"]:
        geo["source"] = "geometry"
        return geo

    tag = None
    if laterality_tag is not None:
        t = str(laterality_tag).strip().upper()
        if t in ("L", "LEFT"):
            tag = "L"
        elif t in ("R", "RIGHT"):
            tag = "R"

    if tag is not None:
        return {"side": tag, "ambiguous": False, "source": "tag",
                "reason": geo["reason"], "mean_x_mm": geo["mean_x_mm"],
                "n": geo["n"], "consistent": geo["consistent"]}

    return {"side": None, "ambiguous": True, "source": "none",
            "reason": geo["reason"], "mean_x_mm": geo["mean_x_mm"],
            "n": geo["n"], "consistent": geo["consistent"]}


if __name__ == "__main__":
    # --- geometry fast-path tests (no torch / no cv2 needed) --------------
    assert resolve_side([-111.0, -108.0, -114.0])["side"] == "R"
    assert resolve_side([111.0, 108.0])["side"] == "L"
    assert resolve_side([3.0, -4.0])["ambiguous"] is True          # near isocenter
    assert resolve_side([3.0, -4.0], "L")["side"] == "L"            # tier 2
    assert resolve_side([3.0, -4.0], "L")["source"] == "tag"
    assert resolve_side([])["side"] is None                         # tier 3
    assert resolve_side([1.0], None)["source"] == "none"
    # exactly at the threshold is resolved (>=), not ambiguous
    assert resolve_side([SIDE_GEOM_MIN_MM])["ambiguous"] is False
    assert resolve_side([-SIDE_GEOM_MIN_MM])["side"] == "R"
    print("side_from_geometry / resolve_side self-test passed "
          "(thr=%.0fmm, agreement=%.3f, coverage=%.3f)."
          % (SIDE_GEOM_MIN_MM, SIDE_GEOM_AGREEMENT, SIDE_GEOM_COVERAGE))

    # --- slice extraction smoke test -------------------------------------
    dummy_vol = np.random.randint(0, 2000, size=(24, 256, 256), dtype=np.uint16)
    if HAS_CV2:
        slices = extract_side_slices(dummy_vol)
        assert slices.shape == (3, 160, 160)
        assert slices.dtype == np.uint8
        print("extract_side_slices self-test passed! Slices shape:", slices.shape)
    else:
        print("cv2 not available for local test, module syntax valid.")
