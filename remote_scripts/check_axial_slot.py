import numpy as np

CORPUS_DIR = "/home/eason/rsna-knee/data/corpus64"
masks_p0 = np.load(f"{CORPUS_DIR}/masks64_p0.npy")
masks_p1 = np.load(f"{CORPUS_DIR}/masks64_p1.npy")

m = np.vstack([masks_p0, masks_p1])
print("Total studies in corpus64:", len(m))
print("Slot 0 (Sagittal fluid 0..18) valid mean:", m[:, 0:18].mean())
print("Slot 1 (Sagittal non-fluid 18..32) valid mean:", m[:, 18:32].mean())
print("Slot 2 (Coronal fluid 32..44) valid mean:", m[:, 32:44].mean())
print("Slot 3 (Coronal non-fluid 44..52) valid mean:", m[:, 44:52].mean())
print("Slot 4 (Axial 52..64) valid mean:", m[:, 52:64].mean())
print("Studies with 12/12 valid axial slices:", (m[:, 52:64].sum(1) == 12).sum(), "/", len(m))
