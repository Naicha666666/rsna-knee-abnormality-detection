import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

CORPUS_DIR = Path('/home/eason/rsna-knee/data/corpus64')
OUT_DIR = Path('/home/eason/rsna-knee/vlm_cohort_images')
OUT_DIR.mkdir(parents=True, exist_ok=True)

cohort = json.load(open('/home/eason/rsna-knee/cohort_200_sagittal_luna.json'))
print(f'Total targets: {len(cohort)} Sagittal cases for GPT-6 Luna')

t0 = time.time()
print('Loading corpus index and memory maps...')
ids_p0 = np.load(CORPUS_DIR / 'ids64_p0.npy', allow_pickle=True)
ids_p1 = np.load(CORPUS_DIR / 'ids64_p1.npy', allow_pickle=True)
vols_p0 = np.load(CORPUS_DIR / 'vols64_p0.npy', mmap_mode='r')
vols_p1 = np.load(CORPUS_DIR / 'vols64_p1.npy', mmap_mode='r')

id_to_loc = {}
for i, uid in enumerate(ids_p0):
    id_to_loc[str(uid)] = (0, i)
for i, uid in enumerate(ids_p1):
    id_to_loc[str(uid)] = (1, i)

print(f'Index built in {time.time() - t0:.2f}s.')

H, W = 336, 336
rows, cols = 3, 4
idx12 = np.round(np.linspace(1, 16, 12)).astype(int).tolist()
sag_labels = [f'S{i+1}' for i in range(12)]

def create_montage(slices, labels):
    canvas = Image.new('L', (cols * W, rows * H), color=0)
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

exported = 0
skipped = 0
for i, c in enumerate(cohort):
    uid = c['uid']
    out_file = OUT_DIR / f'{uid}_sagittal_montage.png'
    if out_file.exists():
        skipped += 1
        continue
    part, idx = id_to_loc[uid]
    vol = vols_p0[idx] if part == 0 else vols_p1[idx]
    sag_slices = [vol[s] for s in idx12]
    im = create_montage(sag_slices, sag_labels)
    im.save(out_file)
    exported += 1
    if exported % 50 == 0:
        print(f'Exported {exported} / {len(cohort)} sagittal montages ({time.time() - t0:.1f}s)...')

print(f'Done! Exported {exported} new, {skipped} existing to {OUT_DIR} in {time.time() - t0:.2f}s.')
