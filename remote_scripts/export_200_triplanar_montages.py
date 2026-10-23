"""Export Coronal, Sagittal, and Axial 12-Slice Montages for 200 Tri-Planar Cases.
Extracts:
- Coronal: slices 32..44 (12 slices) -> {uid}_montage.png
- Sagittal: idx12 (12 slices from slot 0..17) -> {uid}_sagittal_montage.png
- Axial: slices 52..64 (12 slices from slot 48..64) -> {uid}_axial_montage.png
"""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

CORPUS_DIR = Path("/home/eason/rsna-knee/data/corpus64")
OUT_DIR = Path("/home/eason/rsna-knee/vlm_cohort_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cohort = json.load(open("/home/eason/rsna-knee/cohort_200_triplanar_luna.json"))
print(f"Total targets: {len(cohort)} Tri-Planar cases for GPT-6 Luna")

t0 = time.time()
print("Loading corpus index and memory maps...")
ids_p0 = np.load(CORPUS_DIR / "ids64_p0.npy", allow_pickle=True)
ids_p1 = np.load(CORPUS_DIR / "ids64_p1.npy", allow_pickle=True)
vols_p0 = np.load(CORPUS_DIR / "vols64_p0.npy", mmap_mode="r")
vols_p1 = np.load(CORPUS_DIR / "vols64_p1.npy", mmap_mode="r")

id_to_loc = {}
for i, uid in enumerate(ids_p0):
    id_to_loc[str(uid)] = (0, i)
for i, uid in enumerate(ids_p1):
    id_to_loc[str(uid)] = (1, i)

print(f"Index built in {time.time() - t0:.2f}s. Mapping {len(id_to_loc)} volumes.")

H, W = 336, 336
rows, cols = 3, 4
labels = [f"S{i+1}" for i in range(12)]

sag_indices = np.round(np.linspace(1, 16, 12)).astype(int).tolist()
cor_indices = list(range(32, 44))
ax_indices = list(range(52, 64))

def create_montage(slices, labels):
    canvas = Image.new("L", (cols * W, rows * H), color=0)
    draw = ImageDraw.Draw(canvas)
    for idx, (sl, lbl) in enumerate(zip(slices, labels)):
        r = idx // cols
        c = idx % cols
        x0, y0 = c * W, r * H
        tile = Image.fromarray(sl)
        canvas.paste(tile, (x0, y0))
        draw.rectangle([x0, y0, x0 + W - 1, y0 + H - 1], outline=128, width=2)
        draw.rectangle([x0 + 4, y0 + 4, x0 + 70, y0 + 32], fill=0, outline=255, width=1)
        draw.text((x0 + 10, y0 + 8), lbl, fill=255)
    return canvas

n_cor, n_sag, n_ax = 0, 0, 0
for i, c in enumerate(cohort):
    uid = c["uid"]
    if uid not in id_to_loc:
        print(f"UID not found in corpus64: {uid}")
        continue
    part, idx = id_to_loc[uid]
    vol = vols_p0[idx] if part == 0 else vols_p1[idx]
    
    # 1. Coronal
    f_cor = OUT_DIR / f"{uid}_montage.png"
    if not f_cor.exists():
        im_cor = create_montage([vol[s] for s in cor_indices], labels)
        im_cor.save(f_cor)
        n_cor += 1
        
    # 2. Sagittal
    f_sag = OUT_DIR / f"{uid}_sagittal_montage.png"
    if not f_sag.exists():
        im_sag = create_montage([vol[s] for s in sag_indices], labels)
        im_sag.save(f_sag)
        n_sag += 1

    # 3. Axial
    f_ax = OUT_DIR / f"{uid}_axial_montage.png"
    if not f_ax.exists():
        im_ax = create_montage([vol[s] for s in ax_indices], labels)
        im_ax.save(f_ax)
        n_ax += 1

    if (i + 1) % 50 == 0:
        print(f"[{i+1}/{len(cohort)}] Export progress... ({time.time() - t0:.1f}s)")

print(f"Done! Exported new: {n_cor} coronal, {n_sag} sagittal, {n_ax} axial montages in {time.time() - t0:.2f}s.")
