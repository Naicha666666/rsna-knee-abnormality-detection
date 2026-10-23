"""Export 3D Aggregated Images for VLM Multimodal Evaluation.

Generates two 3D representations per study from corpus64:
1. Montage Gallery (3x4 grid of 12 Coronal slices, slices 32..43, with tile labels S1..S12).
2. Tri-Planar View (1x3 grid: Sagittal slice 9, Coronal slice 37, Axial slice 58).
3. Counterfactual Blank Controls for both representations (pure black with same grid structure).
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CORPUS_DIR = Path("/home/eason/rsna-knee/data/corpus64")
OUT_DIR = Path("/home/eason/rsna-knee/experiments/b11_3d_montage_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = json.load(open("/home/eason/rsna-knee/b11_pilot_manifest.json"))

# Load corpus index
ids_p0 = np.load(CORPUS_DIR / "ids64_p0.npy", allow_pickle=True)
ids_p1 = np.load(CORPUS_DIR / "ids64_p1.npy", allow_pickle=True)
vols_p0 = np.load(CORPUS_DIR / "vols64_p0.npy", mmap_mode="r")
vols_p1 = np.load(CORPUS_DIR / "vols64_p1.npy", mmap_mode="r")

id_to_loc = {}
for i, uid in enumerate(ids_p0):
    id_to_loc[str(uid)] = (0, i)
for i, uid in enumerate(ids_p1):
    id_to_loc[str(uid)] = (1, i)

H, W = 336, 336

def create_montage(slices: list[np.ndarray], labels: list[str], rows: int, cols: int, is_blank: bool = False) -> Image.Image:
    canvas = Image.new("L", (cols * W, rows * H), color=0)
    draw = ImageDraw.Draw(canvas)
    
    for idx, (sl, lbl) in enumerate(zip(slices, labels)):
        r = idx // cols
        c = idx % cols
        x0, y0 = c * W, r * H
        
        if not is_blank:
            tile = Image.fromarray(sl)
            canvas.paste(tile, (x0, y0))
        
        # Draw border
        draw.rectangle([x0, y0, x0 + W - 1, y0 + H - 1], outline=128, width=2)
        # Draw label box
        draw.rectangle([x0 + 4, y0 + 4, x0 + 70, y0 + 32], fill=0, outline=255, width=1)
        draw.text((x0 + 10, y0 + 8), lbl, fill=255)
        
    return canvas

print(f"Starting 3D aggregation export for {len(manifest)} studies...")

for item in manifest:
    uid = item["study_id"]
    if uid not in id_to_loc:
        print(f"Study {uid} missing from corpus!")
        continue
    part, idx = id_to_loc[uid]
    vol = vols_p0[idx] if part == 0 else vols_p1[idx]
    
    # 1. Montage: 12 Coronal slices (32 to 43)
    cor_slices = [vol[s] for s in range(32, 44)]
    cor_labels = [f"S{i+1}" for i in range(12)]
    
    # Real montage
    im_montage = create_montage(cor_slices, cor_labels, rows=3, cols=4, is_blank=False)
    im_montage.save(OUT_DIR / f"{uid}_montage.png")
    
    # Blank montage control
    blank_montage = create_montage(cor_slices, cor_labels, rows=3, cols=4, is_blank=True)
    blank_montage.save(OUT_DIR / f"{uid}_montage_blank.png")
    
    # 2. Tri-planar View: Sagittal 9, Coronal 37, Axial 58
    tri_slices = [vol[9], vol[37], vol[58]]
    tri_labels = ["SAGITTAL", "CORONAL", "AXIAL"]
    
    # Real tri-planar
    im_tri = create_montage(tri_slices, tri_labels, rows=1, cols=3, is_blank=False)
    im_tri.save(OUT_DIR / f"{uid}_triplanar.png")
    
    # Blank tri-planar control
    blank_tri = create_montage(tri_slices, tri_labels, rows=1, cols=3, is_blank=True)
    blank_tri.save(OUT_DIR / f"{uid}_triplanar_blank.png")

print(f"Successfully exported 3D Aggregation datasets to {OUT_DIR}!")
