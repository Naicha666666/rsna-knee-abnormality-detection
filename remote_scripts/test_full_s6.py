import os, sys
os.environ["RSNA_CORPUS_SLOTS"] = "64"
sys.path.insert(0, "/home/eason/rsna-knee/src")
sys.path.insert(0, "/home/eason/rsna-knee/scratchpad")
import tv_stub
tv_stub.install()

import torch
import torch.nn.functional as F
import numpy as np
from corpus import open_corpus
from train_s6_convnextv2_base_r9700 import S6SpatialConvNeXtV2, extract_study_windows

print("Loading corpus...")
vols, masks, uids = open_corpus()

print("Initializing S6SpatialConvNeXtV2...")
init_ckpt = "/home/eason/rsna-knee/data/timm_weights/convnextv2_base_in22k_ft_in1k_384.pt"
model = S6SpatialConvNeXtV2(backbone="convnextv2_base", init_path=init_ckpt, chunk_size=8).cuda()

for i in range(5):
    vol = vols[i]
    print(f"\n--- Study {i}: {uids[i]} (vol min={vol.min()}, max={vol.max()}) ---")
    wins = extract_study_windows(vol, n_win=20, train=True)
    print(f"Windows shape: {wins.shape}, min={wins.min()}, max={wins.max()}")
    x = torch.from_numpy(wins).unsqueeze(0).cuda()
    
    logits = model(x)
    print(f"Logits shape: {logits.shape}, has_nan: {torch.isnan(logits).any().item()}, vals: {logits.detach().cpu().numpy()[0, :4]}")
    
    target = torch.full((1, 12), 0.5, device="cuda")
    loss = F.binary_cross_entropy_with_logits(logits, target)
    print(f"Loss: {loss.item()}")
    loss.backward()
    print("Backward pass OK!")
    model.zero_grad()
