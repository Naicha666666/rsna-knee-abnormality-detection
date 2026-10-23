import json
import numpy as np

CORPUS_DIR = '/home/eason/rsna-knee/data/corpus64'
ids_p0 = np.load(f'{CORPUS_DIR}/ids64_p0.npy', allow_pickle=True)
ids_p1 = np.load(f'{CORPUS_DIR}/ids64_p1.npy', allow_pickle=True)
masks_p0 = np.load(f'{CORPUS_DIR}/masks64_p0.npy')
masks_p1 = np.load(f'{CORPUS_DIR}/masks64_p1.npy')

id_to_loc = {}
for i, uid in enumerate(ids_p0):
    id_to_loc[str(uid)] = (0, i)
for i, uid in enumerate(ids_p1):
    id_to_loc[str(uid)] = (1, i)

cohort = json.load(open('/home/eason/rsna-knee/cohort_200_sagittal_luna.json'))
idx12 = np.round(np.linspace(1, 16, 12)).astype(int).tolist()
print('Chosen 12 slice indices:', idx12)

all_valid = 0
found = 0
for c in cohort:
    uid = c['uid']
    if uid in id_to_loc:
        found += 1
        part, idx = id_to_loc[uid]
        mask = masks_p0[idx] if part == 0 else masks_p1[idx]
        valid_in_slot0 = mask[:18].sum()
        valid_in_picked = sum(mask[s] for s in idx12)
        if valid_in_picked == 12:
            all_valid += 1
        else:
            print(f'UID {uid}: {valid_in_slot0}/18 in slot0, {valid_in_picked}/12 in picked')

print(f'Found in corpus64: {found}/{len(cohort)}')
print(f'All 12 chosen slices valid: {all_valid}/{len(cohort)}')
