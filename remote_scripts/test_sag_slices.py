import numpy as np
from PIL import Image

CORPUS_DIR = '/home/eason/rsna-knee/data/corpus64'
vols_p0 = np.load(f'{CORPUS_DIR}/vols64_p0.npy', mmap_mode='r')
masks_p0 = np.load(f'{CORPUS_DIR}/masks64_p0.npy')
ids_p0 = np.load(f'{CORPUS_DIR}/ids64_p0.npy', allow_pickle=True)

print('First study ID:', ids_p0[0])
print('Slot 0 (Sagittal fluid-sensitive) valid mask for first 5 studies:')
for i in range(5):
    print(f'Study {i} ({ids_p0[i]}): valid {masks_p0[i, :18].sum()} / 18 slices, total valid {masks_p0[i].sum()} / 64')

# Check slice intensity min/max/mean for study 0
v = vols_p0[0]
for s in range(18):
    print(f'Slice {s}: min={v[s].min()}, max={v[s].max()}, mean={v[s].mean():.1f}')

