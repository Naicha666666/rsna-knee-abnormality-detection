from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import gc
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

# The label extractor is defined in the cells above when this runs as a notebook. As a
# plain script it is imported from the package source, so the two paths share one
# definition rather than keeping a copy each.
if "extract" not in dir():
    import sys
    for _p in ("src", "../src", "../../src"):
        if (Path(_p) / "report_labeler.py").is_file():
            sys.path.insert(0, _p)
            break
    from report_labeler import extract  # noqa: F401

T0 = time.time()
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


IMG = 224                  # encoder input; a multiple of the patch size 14
CROP_MM = 160.0            # physical extent of the centre crop; a knee FOV is 140-180
GROUP = 3                  # slices per encoder input, stacked as the three channels
N_GROUP_MAX = 3            # groups per slot, before the memory budget is applied
CACHE_BUDGET_GB = 12.0     # ceiling for the training cache
HDR_THREADS = 16
PIX_THREADS = 12

EPOCHS = 12
BATCH_STUDIES = 8          # a study is a bag of up to N_SLOT slot images
LR_BACKBONE = 8e-6         # the encoder is being adapted, not retrained
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.02
UNFREEZE_LAST = 6          # trainable transformer blocks, counted from the output end
EVAL_BATCH = 12
TIME_BUDGET = 8.0 * 3600

# Six slots: three planes crossed with the acquisition axes. The fat-suppressed
# fluid-sensitive series exist for nearly every study; the T1 and the non-suppressed
# fluid-sensitive series are scarcer, which is what the presence mask is for.
SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

# The alternative: plane x the provided flag, ignoring the recovered weighting. Kept as
# a switch so the choice of slot definition can be varied while everything else is held
# fixed. Under this scheme a `Struct` slot mixes T1 series with non-fat-suppressed PD/T2
# series, which carry very different tissue contrast.
SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]

SLOT_SCHEME = os.environ.get("SLOT_SCHEME", "recovered")
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == "public" else SLOTS_RECOVERED
N_SLOT = len(SLOTS)

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
                        r"water excit|\btirm\b|\bsting\b|\bfatsup\b")
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")
