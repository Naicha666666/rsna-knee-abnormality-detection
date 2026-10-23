"""Dump Coronal slices for the 20 B11 pilot studies from corpus64.
Saves PNG images:
- coronal_slice.png (slice 38, weight-bearing coronal)
- blank_slice.png (pure black)
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image

CORPUS_DIR = Path("/home/eason/rsna-knee/data/corpus64")
OUT_DIR = Path("/home/eason/rsna-knee/experiments/b11_pilot_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = json.load(open("/home/eason/rsna-knee/b11_pilot_manifest.json"))
study_ids = [m["study_id"] for m in manifest]

# Load corpus ids and vols
ids_p0 = np.load(CORPUS_DIR / "ids64_p0.npy", allow_pickle=True)
ids_p1 = np.load(CORPUS_DIR / "ids64_p1.npy", allow_pickle=True)
all_ids = np.concatenate([ids_p0, ids_p1])

vols_p0 = np.load(CORPUS_DIR / "vols64_p0.npy", mmap_mode="r")
vols_p1 = np.load(CORPUS_DIR / "vols64_p1.npy", mmap_mode="r")

id_to_loc = {}
for i, uid in enumerate(ids_p0):
    id_to_loc[str(uid)] = (0, i)
for i, uid in enumerate(ids_p1):
    id_to_loc[str(uid)] = (1, i)

# Central coronal slice is index 37 in the 64-slice corpus
# Coronal slot spans 32..43, centre is 37-38
SLICE_IDX = 37

for item in manifest:
    uid = item["study_id"]
    if uid not in id_to_loc:
        print(f"Study {uid} not in corpus!")
        continue
    part, idx = id_to_loc[uid]
    vol = vols_p0[idx] if part == 0 else vols_p1[idx]
    
    # Extract coronal weight-bearing slice
    slice_img = vol[SLICE_IDX]  # [336, 336] uint8
    
    # Save true image
    im = Image.fromarray(slice_img)
    im.save(OUT_DIR / f"{uid}_coronal.png")
    
    # Save blank control image
    blank = np.zeros_like(slice_img)
    Image.fromarray(blank).save(OUT_DIR / f"{uid}_blank.png")

print(f"Successfully extracted {len(manifest)} coronal & blank image pairs to {OUT_DIR}")
