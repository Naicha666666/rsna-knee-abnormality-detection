"""Export any missing Coronal 12-Slice Montages for the 1,000 ambiguous cohort on remote."""
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

CORPUS_DIR = Path("/home/eason/rsna-knee/data/corpus64")
OUT_DIR = Path("/home/eason/rsna-knee/vlm_cohort_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("/home/eason/rsna-knee/cohort_1000_ambiguous_manifest.json")

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
target_uids = [x["uid"] for x in manifest]

print(f"Total targets: {len(target_uids)}")

existing = [uid for uid in target_uids if (OUT_DIR / f"{uid}_montage.png").exists()]
missing = [uid for uid in target_uids if not (OUT_DIR / f"{uid}_montage.png").exists()]
print(f"Already existing: {len(existing)} | Need to export: {len(missing)}")

if missing:
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
    cor_indices = list(range(32, 44))

    exported = 0
    t_start = time.time()
    for uid in missing:
        if uid not in id_to_loc:
            print(f"WARNING: UID not found in corpus64: {uid}")
            continue
        part, idx = id_to_loc[uid]
        vol = vols_p0[idx] if part == 0 else vols_p1[idx]
        cor_slices = [vol[s] for s in cor_indices]
        im = create_montage(cor_slices, cor_labels)
        out_file = OUT_DIR / f"{uid}_montage.png"
        im.save(out_file, format="PNG")
        exported += 1
        if exported % 100 == 0 or exported == len(missing):
            print(f"Exported {exported}/{len(missing)} montages ({time.time() - t_start:.1f}s)...")

    print(f"All {exported} montages successfully exported in {time.time() - t_start:.2f}s!")
else:
    print("All 1,000 montages already exist! No export needed.")
