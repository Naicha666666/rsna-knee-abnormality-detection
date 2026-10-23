import numpy as np
from pathlib import Path

CORPUS_DIR = Path('/home/eason/rsna-knee/data/corpus64')
ids_p0 = np.load(CORPUS_DIR / 'ids64_p0.npy', allow_pickle=True)
ids_p1 = np.load(CORPUS_DIR / 'ids64_p1.npy', allow_pickle=True)
vols_p0 = np.load(CORPUS_DIR / 'vols64_p0.npy', mmap_mode='r')
vols_p1 = np.load(CORPUS_DIR / 'vols64_p1.npy', mmap_mode='r')

target_uid = '1.2.826.0.1.3680043.8.498.11557620559191469069130827959098335840'

if target_uid in ids_p0:
    idx = list(ids_p0).index(target_uid)
    vol = vols_p0[idx]
elif target_uid in ids_p1:
    idx = list(ids_p1).index(target_uid)
    vol = vols_p1[idx]
else:
    print('Not found')
    exit(1)

np.save('/home/eason/rsna-knee/case0_vol64.npy', np.array(vol))
print('Saved case0_vol64.npy shape:', vol.shape)
