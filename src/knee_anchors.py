"""Anatomical Anchor Bank and Multi-View Geometry for Knee MRI.

Ported from heliosli/rsna-knee-blend-gold (OrthoFoundation compact pipeline).

Key innovations:
- Patellofemoral (PF) local 100mm view centered anterior-superior (solves PF OA edge-cutoff)
- Posterior (Post) local 100mm view centered on popliteal fossa (Baker's cyst / PCL)
- Coronal Bilateral local 100mm views for medial / lateral compartment cartilage & meniscus
- Thin-stack slab averaging: thin 3D volumes (0.4-1.5mm) dynamically merged to 3.5mm slabs
"""
from __future__ import annotations
import math
import numpy as np
import torch
import cv2

LOCAL_MM = 100.0
THIN_MM = 1.75
SLAB_MM = 3.5

# Canonical anatomical view centers: (row fraction, col fraction)
# Anterior is +column (sagittal) and -row (axial)
PF_CENTER = {"Sagittal": (0.38, 0.60), "Axial": (0.38, 0.50)}
POST_CENTER = {"Sagittal": (0.55, 0.35), "Axial": (0.62, 0.50)}

def knee_anchor(vol: np.ndarray, plane: str) -> tuple[tuple[float, float], str]:
    """Calculate (row_frac, col_frac) of the knee in the series in-plane frame.
    
    Reads foreground of the middle 3 slices; falls back to (0.5, 0.5) if not detected.
    """
    m = len(vol) // 2
    mid = np.asarray(vol[max(0, m - 1):m + 2]).astype(np.float32).mean(0)
    H, W = mid.shape
    hi = float(np.percentile(mid, 99.5))
    if hi <= 0:
        return (0.5, 0.5), "empty"
    fg = (mid > 0.08 * hi).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return (0.5, 0.5), "no_fg"
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = [int(v) for v in stats[k, :4]]
    if w >= 0.97 * W and h >= 0.97 * H:
        return (0.5, 0.5), "fills_field"
    comp = (lab == k)
    if plane in ("Sagittal", "Coronal"):
        width = comp[y:y + h].sum(1).astype(np.float32)
        if len(width) >= 15:
            kern = np.ones(15, np.float32) / 15
            width = np.convolve(width, kern, mode="same")
        row = (y + int(np.argmax(width))) / max(1, H - 1)
    else:
        row = (y + h / 2.0) / max(1, H - 1)
    col = (x + w / 2.0) / max(1, W - 1)
    return (float(row), float(col)), "ok"

def anchored_point(anchor: tuple[float, float], spacing_mm: tuple[float, float],
                   shape: tuple[int, int], d_row_mm: float, d_col_mm: float) -> tuple[float, float]:
    """Shift anchor point by physical millimetre offsets."""
    sy, sx = float(spacing_mm[0]), float(spacing_mm[1])
    H, W = shape
    return (
        min(1.0, max(0.0, anchor[0] + d_row_mm / (sy * max(1, H - 1)))),
        min(1.0, max(0.0, anchor[1] + d_col_mm / (sx * max(1, W - 1))))
    )

def series_views(plane: str, native_shape: tuple[int, int, int], spacings: tuple[float, float],
                 size: int, views: str = "v1", anchor: tuple[float, float] | None = None,
                 patella: tuple[float, float] | None = None):
    """Generate multi-view specifications for a series.
    
    Returns list of (view_id, crop_mm, center_xy)
    """
    shp = tuple(native_shape[-2:])
    a = anchor if anchor is not None else (0.5, 0.5)
    sp = spacings if spacings is not None else (1.0, 1.0)

    if views in ("v3k", "v4"):
        full = [(0, None, (0.5, 0.5))] if views == "v4" else []
        if plane == "Sagittal":
            pf = patella if patella is not None else anchored_point(a, sp, shp, -25.0, +40.0)
            return full + [
                (1, LOCAL_MM, a),                               # joint center local
                (4, LOCAL_MM, pf),                              # patellofemoral local
                (5, LOCAL_MM, anchored_point(a, sp, shp, +5.0, -35.0))  # posterior / Baker's local
            ]
        if plane == "Axial":
            pf = patella if patella is not None else anchored_point(a, sp, shp, -40.0, 0.0)
            return full + [
                (1, LOCAL_MM, a),
                (4, LOCAL_MM, pf),
                (5, LOCAL_MM, anchored_point(a, sp, shp, +35.0, 0.0))
            ]
        # Coronal: Medial (-30mm) and Lateral (+30mm) bilateral locals
        return full + [
            (2, LOCAL_MM, anchored_point(a, sp, shp, 0.0, -30.0)),
            (3, LOCAL_MM, anchored_point(a, sp, shp, 0.0, +30.0))
        ]

    # Baseline v1 / v2 fallback
    out = [(0, None, (0.5, 0.5)), (1, LOCAL_MM, (0.5, 0.5))]
    if views == "v2" and plane in PF_CENTER:
        out.append((4, LOCAL_MM, PF_CENTER[plane]))
    return out

def merge_thin_stack_slabs(vol: np.ndarray, pos: np.ndarray, spacing_mm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Dynamically average thin 3D acquisitions (0.4-1.5mm) into standard ~3.5mm slabs."""
    if 0 < spacing_mm < THIN_MM:
        g = max(1, int(round(SLAB_MM / spacing_mm)))
        n_new = len(pos) // g + (1 if len(pos) % g >= max(1, g // 2) else 0)
        if n_new >= 2:
            chunks = [vol[j * g:min(len(vol), (j + 1) * g)] for j in range(n_new)]
            merged_vol = np.stack([np.rint(c.astype(np.float32).mean(0)).astype(vol.dtype) for c in chunks])
            merged_pos = np.array([pos[j * g:min(len(pos), (j + 1) * g)].mean() for j in range(n_new)])
            new_spacing = float(np.median(np.diff(merged_pos)))
            return merged_vol, merged_pos, new_spacing
    return vol, pos, spacing_mm
