# =============================================================================
# RSNA Knee Abnormality Detection — Kaggle Notebook baseline template
# -----------------------------------------------------------------------------
# How to use in Kaggle:
#   1. Fork the competition notebook (GPU, internet ON for development).
#   2. Upload the `src/` folder (config.py, dicom_utils.py, dataset.py,
#      model.py, train.py, make_submission.py) as notebook files
#      (Add Data -> upload, or just paste them into the notebook editor).
#   3. Set `DATA_ROOT` below to the mounted competition input path.
#   4. Run this script. During DEV you may keep internet ON to pip install pydicom
#      if missing; for the COMMIT (submission) turn internet OFF and ensure all
#      packages are available (pydicom is in the Kaggle base image).
#   5. The notebook must output `submission.csv` (required filename).
# =============================================================================
import os, sys

# ---- point this at the competition input mounted in the notebook ----
DATA_ROOT = "/kaggle/input/rsna-knee-abnormality-detection"
# Make our src modules importable
SRC = "/kaggle/input/<your-src-dataset>/src"   # <-- replace with your uploaded src path
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)

import config as C
C.DATA_ROOT = DATA_ROOT  # override so dataset.py finds the DICOM trees

# Optional: ensure pydicom is present (dev only; comment out before commit w/ internet off)
# !pip install -q pydicom

# ----------------------------- 1) CV sanity --------------------------------
# Runs 5-fold stratified GroupKFold on the labeled studies, prints macro-AUC.
# With the full labeled set (notebook only) this validates the pipeline.
import train as T
T.main()   # prints per-fold + OOF macro-AUC

# ----------------------- 2) train final + submit ----------------------------
# Retrain on ALL labeled data, save EMA weights, predict test, write csv.
import torch
from torch.utils.data import DataLoader
import make_submission as MS
import model as M
from dataset import RSNADataset, collate_train, collate_test

LR, EPOCHS = C.LR, C.EPOCHS
model = M.RSNAMultimodal().to("cuda" if torch.cuda.is_available() else "cpu")
opt = torch.optim.AdamW(M.get_param_groups(model, LR), lr=LR)
bce = torch.nn.BCEWithLogitsLoss(reduction="none")
rankloss = M.PairwiseRankingLoss()
ema = M.EMA(model)
ds = RSNADataset("train")
loader = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=True, collate_fn=collate_train)
device = next(model.parameters()).device
for _ in range(EPOCHS):
    model.train()
    for imgs, masks, y, w in loader:
        imgs, masks, y, w = imgs.to(device), masks.to(device), y.to(device), w.to(device)
        opt.zero_grad()
        logits = model(imgs, masks)
        loss = (bce(logits, y) * w).sum() / w.sum().clamp(min=1.0) \
               + C.RANK_LOSS_WEIGHT * rankloss(logits, y)
        loss.backward(); opt.step(); ema.update(model)
ema.apply_to(model)
torch.save(ema.state_dict(), "/kaggle/working/baseline.pt")
print("final checkpoint saved")

MS.main(model_path="/kaggle/working/baseline.pt")
print("submission.csv written:", os.path.exists("/kaggle/working/submission.csv"))
