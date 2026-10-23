"""Export 600 Coronal 12-Slice Montages for 3D VLM Training (400 U2 + 200 Luna)."""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

CORPUS_DIR = Path("/home/eason/rsna-knee/data/corpus64")
OUT_DIR = Path("/home/eason/rsna-knee/vlm_cohort_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

u2_cases = json.load(open("/home/eason/rsna-knee/cohort_400_u2_radimed.json"))
luna_cases = json.load(open("/home/eason/rsna-knee/cohort_200_gpt6_luna.json"))
all_uids = [c["uid"] for c in u2_cases] + [c["uid"] for c in luna_cases]

print(f"Total targets: {len(all_uids)} (400 U2 + 200 Luna)")

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

cor_labels = [f"S{i+1}" for i in range(12)]

exported = 0
skipped = 0
for i, uid in enumerate(all_uids):
    out_file = OUT_DIR / f"{uid}_montage.png"
    if out_file.exists():
        skipped += 1
        continue
    if uid not in id_to_loc:
        print(f"UID not found in corpus64: {uid}")
        continue
    part, idx = id_to_loc[uid]
    vol = vols_p0[idx] if part == 0 else vols_p1[idx]
    # Slices 32..44 are the 12 coronal slices
    cor_slices = [vol[s] for s in range(32, 44)]
    im = create_montage(cor_slices, cor_labels)
    im.save(out_file)
    exported += 1
    if exported % 50 == 0:
        print(f"Exported {exported} / {len(all_uids)} montages ({time.time() - t0:.1f}s)...")

print(f"Done! Exported {exported} new, {skipped} existing to {OUT_DIR} in {time.time() - t0:.2f}s.")
