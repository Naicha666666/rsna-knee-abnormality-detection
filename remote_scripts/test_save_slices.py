import numpy as np
from PIL import Image

CORPUS_DIR = '/home/eason/rsna-knee/data/corpus64'
vols_p0 = np.load(f'{CORPUS_DIR}/vols64_p0.npy', mmap_mode='r')
v = vols_p0[0]

# Pick 12 slices across 0..17
idx12 = np.round(np.linspace(1, 16, 12)).astype(int)
print('Chosen 12 slice indices from 18:', idx12.tolist())

# Save a test 3x4 montage
H, W = 336, 336
rows, cols = 3, 4
from PIL import ImageDraw
canvas = Image.new('L', (cols * W, rows * H), color=0)
draw = ImageDraw.Draw(canvas)
for idx, s_i in enumerate(idx12):
    r = idx // cols
    c = idx % cols
    x0, y0 = c * W, r * H
    tile = Image.fromarray(v[s_i])
    canvas.paste(tile, (x0, y0))
    draw.rectangle([x0, y0, x0 + W - 1, y0 + H - 1], outline=128, width=2)
    draw.rectangle([x0 + 4, y0 + 4, x0 + 70, y0 + 32], fill=0, outline=255, width=1)
    draw.text((x0 + 10, y0 + 8), f'S{idx+1}', fill=255)

canvas.save('/home/eason/rsna-knee/sample_sagittal_montage.png')
print('Saved /home/eason/rsna-knee/sample_sagittal_montage.png')
