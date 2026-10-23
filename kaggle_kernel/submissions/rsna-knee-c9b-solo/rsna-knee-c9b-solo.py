from __future__ import annotations
import os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_THIS = sys.modules[__name__]
for _n in ['config', 'tv_stub', 'dicom_utils', 'report_extractor', 'dataset', 'backbones', 'backbones.dinov2', 'backbones.tiny_cnn', 'model', 'train', 'raptor', 'corpus', 'experiments', 'make_submission']:
    sys.modules[_n] = _THIS
import torch, numpy as np, pandas as pd


# ===================== config.py =====================
"""Config & constants for RSNA Knee Abnormality Detection baseline.

Paths auto-switch between local (EDA / smoke tests) and a Kaggle Notebook
(full data mounted under /kaggle/input/...). Tuned using public high-vote
notebooks (see LOG.md "参考社区公开 notebook" section).
"""

import os

# ---- 12 target labels (order = submission.csv column order) ----
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
N_LABELS = len(LABELS)

# ---- Image preprocessing (physical-scale aware, header-aware w/ fallback) ----
# v9: 168 instead of 224. The reference notebook's controlled coverage experiment
# (s168g4 vs s168g8) trades grid resolution for SLICE COVERAGE at a fixed memory
# budget: 160mm/168 = 0.95 mm/px and 13.3 mm per 14-px patch token, which lets the
# same 12 GB cache hold 5 groups (15 slices/slot) instead of 3 (9 slices/slot),
# and makes each encoder forward (168/224)^2 = 0.56x as expensive.
IMG_SIZE = 168            # resized slice size fed to the backbone
SLICES_PER_SLOT = 8       # legacy per-slice path (kept for smoke tests)
N_SLOTS = 6               # SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS,
                           # SAG_FLUID_NOFS, COR_T1, SAG_T1  (see SLOTS below)
CROP_MM = 160             # physical extent cropped before resize -> 0.71 mm/px
WIN_PMIN, WIN_PMAX = 1, 99  # percentile window for intensity normalization
CENTRAL_SLICE_FRAC = 0.6  # keep central 60% of each stack

# 2.5D grouping: GROUP consecutive slices are stacked as the 3 DINOv2 channels.
# Training draws one group per step (stack-axis aug); inference averages groups.
GROUP = 3
N_GROUPS_MAX = 8          # ceiling; plan_n_groups() cuts it to the cache budget
CACHE_BUDGET_GB = 12.0    # uint8 train cache ceiling (T4 ~29 GB RAM; 3 groups ≈ 11 GB)
LAT_MIN_OFFSET_MM = 5.0   # |patient-x| below this is not a usable side cue
LAT_MIN_AGREEMENT = 0.85  # enable IPP fallback only if it agrees with tags
GOLD_WEIGHT = 3.0         # annotated study weight vs a report-derived one
DERIVED_W_FLOOR = 0.25    # W = floor + (1-floor)*extractor_conf for report labels
SLOT_PRIOR_BIAS = 0.55    # weak anatomical prior on the slot attention
EMA_DECAY = 0.997
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.10
# 168 px made a fold ~32 min (v9: 4 folds in 128 min), so the 4-fold runs finished
# in 3.8 h of the 9 h cap. Raising the guardrail to 6.5 h lets an 8-fold run
# actually finish 8 folds; hidden-test decode + inference still leaves >1.5 h.
TRAIN_TIME_BUDGET_H = 6.5 # leave the rest of the 9h for hidden-test I/O
WEIGHT_DECAY = 0.02
RANK_POS, RANK_NEG = 0.60, 0.40  # graded-target cutoffs for the pairwise term
ONECYCLE_PCT_START = 0.15

# Epoch selection rule. "worse_of_two" = min(derived-holdout AUC, gold AUC), which is
# what v4-v9 shipped; "derived" scores on the report-derived holdout alone.
# v9's holdout has ~1100 studies but only ~14 gold ones, and in 2 of 4 folds the gold
# term is what bound the selection (fold 2: gold 0.767 vs derived 0.799) -- i.e. the
# shipped epoch was chosen by 14 studies. The reference pipeline dropped the gold term
# for exactly this reason.
SELECTION = "derived"        # "worse_of_two" | "derived"

# Slot definition: (plane, fluid_sensitive, fat_suppression)
# In this corpus Fluid_Sensitive and Fat_Suppression are PERFECTLY correlated: every
# series is (0,0) or (1,1) -- (1,0) and (0,1) do not occur, in train or test. So the
# old slot 3 ("Sagittal", 1, 0) matched ZERO series in all 4407 studies: it burnt 1/6
# of the cache and 1/6 of the encoder forwards on guaranteed-zero images. Meanwhile
# ("Axial", 0, 0) -- 1179 series, 857 studies -- had no slot at all and was discarded.
# Swapping the dead slot for it lifts slot fill 77.4% -> 80.7%. (The reference pipeline
# avoids this by RECOVERING fluid/fatsat from DICOM headers, where the two flags carry
# independent information; with the provided CSV columns they carry one bit between
# them. That recovery is the follow-up experiment, not this one.)
SLOTS = [
    ("Sagittal", 1, 1), ("Coronal", 1, 1), ("Axial", 1, 1),
    ("Axial", 0, 0), ("Coronal", 0, 0), ("Sagittal", 0, 0),
]

# v14: read the slot flags off the DICOM header instead of the competition CSV, and
# let a structural slot fall back to any non-fat-suppressed series in its plane.
# The header diagnostic (LOG.md, 2026-08-31) measured all eight combinations of
# {slot list} x {csv, recovered} x {exact, +fallback}; this one fills the most:
#   reference slots / recovered / +fallback  82.1%   <- v14
#   v13 slots       / csv       / exact      80.7%   <- v13
#   reference slots / csv       / exact      77.4%   <- v6-v12
# and it is also the homogeneous one: with CSV flags a "structural" slot mixes 5183
# T1 series with 3909 PD-non-fat-sat series, which carry very different contrast.
# The greedy top-6 recovered cells by study coverage reproduce this slot list exactly.
USE_RECOVERED_FLAGS = False   # v14 measured worse on all 4 folds
SLOT_FALLBACK = True

# ---- Augmentation policy: NO flips (see LOG.md) ----
# Vertical flip swaps femur/tibia (OA axis); horizontal flip undoes laterality norm.
# Only small rotation / scale / shift are allowed.
USE_FLIP = False

# ---- Model ----
ENCODER_UNFREEZE_LAST = 6   # unfreeze last N transformer blocks of the backbone
ENCODER_LR_FACTOR = 0.008    # encoder LR = HEAD_LR * this (8e-6 when HEAD_LR=1e-3)
SLOT_EMBED = 256             # slot embedding dim after the per-slot pooler

# ---- Backbone selection (controlled variable of the comparison study) ----
# One of: "tiny_cnn" | "dinov2_small" | "dinov2_base".
# Everything else in the protocol is FIXED across experiments so that the only
# thing that varies between runs is the image backbone (see experiments.py).
BACKBONE = os.environ.get("RSNA_BACKBONE", "tiny_cnn")

# Offline DINOv2 weights: a Kaggle dataset of state_dict .pth files produced by
# src/download_dinov2.py (run ONCE with internet), then attached to the
# submission kernel. Code Competition => NO internet at submit time, so weights
# must be available locally. Fallback: glob under the mounted dataset dir.
DINOV2_SMALL_PATH = os.environ.get("DINOV2_SMALL_PATH", None)
DINOV2_BASE_PATH = os.environ.get("DINOV2_BASE_PATH", None)

# Per-backbone slice embedding dim (DINOv2 ViT-S=384, ViT-B=768, tiny CNN=256).
BACKBONE_EMBED = {
    "tiny_cnn": 256,
    "dinov2_small": 384,
    "dinov2_base": 768,
}
RANK_LOSS_WEIGHT = 0.05      # pairwise ranking term; small so rare labels stay silent

# ---- Training ----
SEED = 42
N_FOLDS = 4                  # report-hash groups; rank-averaged at submit time
BATCH_SIZE = 8               # studies per step (2.5D = 6 encoder forwards, not 48)
# v5: v4's per-epoch selection score was still rising monotonically at ep12 in
# ALL 4 folds (no overfitting knee) -> the model was under-trained. 20 epochs
# keeps the run at ~4.4 h, inside TRAIN_TIME_BUDGET_H and the 9 h kernel cap.
EPOCHS = 20
LR = 1e-3                    # head LR; backbone is LR * ENCODER_LR_FACTOR
USE_EMA = True

# ---- Report-derived targets (crux of the winning approach) ----
# test.csv has NO Report column -> text is train-only. We grade reports into
# soft targets + per-target confidence (sample weight). See report_extractor.py.
USE_REPORT_TARGETS = True
REPORT_WEIGHT_FLOOR = 0.28   # silence on a finding pulls weakly, not as a confident negative

# ---- Data locations ----
# On Kaggle a Code Competition mounts the labeled data under
# /kaggle/input/<slug>/ (only available inside the Notebook / submission run; a
# plain `kernels push` run may NOT have it mounted). We therefore AUTO-DETECT the
# data root instead of hard-coding one path, so the kernel works in every layout.
KAGGLE_INPUT = "/kaggle/input/rsna-knee-abnormality-detection"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")


def _auto_data_root():
    """Return the directory holding train.csv/test.csv, or None if absent.

    Kaggle mounts the competition data under /kaggle/input/<slug>/ for a flat
    layout, but ALSO under /kaggle/input/competitions/<slug>/ (observed on a
    script-kernel push run: `ls /kaggle/input` -> ['competitions']). So we search
    both directly and recursively, rather than hard-coding one path.

    Search order:
      1. $RSNA_DATA_ROOT / $KAGGLE_DATA_ROOT (explicit override)
      2. known candidates: /kaggle/input/<slug>, /kaggle/input/competitions/<slug>,
         and a couple of variants
      3. any directory under /kaggle/input (or its 'competitions' subdir) that
         already contains a competition file (train.csv / test.csv /
         sample_submission.csv) -- catches the nested layout
      4. recursive fallback (depth <= 3) so we find the slug dir even if nested
      5. local RAW path (offline EDA / smoke tests)
    """
    import os
    # 1. explicit env override
    for envk in ("RSNA_DATA_ROOT", "KAGGLE_DATA_ROOT"):
        v = os.environ.get(envk)
        if v and os.path.isdir(v):
            return v
    # 2. known candidate paths (flat + nested)
    slug = "rsna-knee-abnormality-detection"
    for c in (KAGGLE_INPUT,
              f"/kaggle/input/competitions/{slug}",
              f"/kaggle/input/competitions/{slug}/competition",
              "/kaggle/input/competitions",
              "/kaggle/input/competition"):
        if os.path.isdir(c):
            return c
    # 3. directory directly under /kaggle/input that holds the data
    if os.path.isdir("/kaggle/input"):
        try:
            for d in sorted(os.listdir("/kaggle/input")):
                dd = os.path.join("/kaggle/input", d)
                if not os.path.isdir(dd):
                    continue
                for fn in ("train.csv", "test.csv", "sample_submission.csv"):
                    if os.path.exists(os.path.join(dd, fn)):
                        return dd
        except Exception:
            pass
        # 3b. nested: /kaggle/input/competitions/<slug>
        try:
            comp = "/kaggle/input/competitions"
            if os.path.isdir(comp):
                for d in sorted(os.listdir(comp)):
                    dd = os.path.join(comp, d)
                    if not os.path.isdir(dd):
                        continue
                    for fn in ("train.csv", "test.csv", "sample_submission.csv"):
                        if os.path.exists(os.path.join(dd, fn)):
                            return dd
        except Exception:
            pass
        # 3c. recursive fallback (depth <= 3) -- finds the slug dir even if nested
        try:
            for root, dirs, files in os.walk("/kaggle/input"):
                depth = root[len("/kaggle/input"):].count(os.sep)
                if depth > 3:
                    dirs[:] = []
                    continue
                for fn in ("train.csv", "test.csv", "sample_submission.csv"):
                    if fn in files:
                        return root
        except Exception:
            pass
    # 4. local fallback (offline EDA)
    if os.path.isdir(RAW):
        return RAW
    return None


DATA_ROOT = _auto_data_root()


# --------------------------------------------------------------------------- #
# R line: study-window MIL on a high-resolution timm backbone (see src/raptor.py)
# --------------------------------------------------------------------------- #
# The six-slot DINOv2 line saturated at 0.855 +/- 0.005 while the public median is
# 0.898 and a single published CoAtNet-384 arm reaches 0.914, so this line changes
# the representation and the input resolution rather than the slot bookkeeping.
RAPTOR_ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
RAPTOR_WEIGHTS_NAME = "coatnet_rmlp_2_rw_384_in12k_ft_in1k.pt"
RAPTOR_RES = 384
# CoAtNet-rmlp's relative-position MLP is built for its native 384 grid (confirmed by
# a local smoke test: 128px raises a hard shape mismatch inside scaled_dot_product_
# attention). Do NOT change RAPTOR_RES without first verifying a mismatched grid is
# actually safe for THIS checkpoint -- unlike DINOv2's absolute pos_embed, we have not
# proven CoAtNet tolerates it, interpolated or not.

# v2 (queued, not yet run -- this week has no Kaggle GPU quota; see LOG.md): all six
# slots instead of three. R1 measured that 384px caching is memory-bound (12 GB buys
# only ~20 total slices across ALL slots combined -- the reference notebook's "64
# slices, 5 slots" figure is an INFERENCE-time number for a tiny test set, not a
# trainable full-corpus cache budget; verified by hand -- see LOG.md). So the lever we
# can actually pull is VIEW DIVERSITY at roughly the same memory, not raw depth:
# R1 was 3 slots x 6 slices = 18 total, 10.89 GB. This is 6 slots x 4 slices = 24
# total, 14.53 GB -- still comfortably under the ~29 GB a T4 session showed room for
# in the DINOv2 line, without gambling on an unverified ceiling.
# R3: R2 (6 slots, 4 slices/series, +SWA) scored LB 0.874, BELOW R1's 0.891 (3 slots,
# 6 slices/series, no SWA) -- and R2's own derived-holdout AUC (0.8109) was also below
# R1's (0.8302) on the SAME 866-study split, so this is a real regression, not LB
# noise. But R2 changed THREE things from R1 at once (slot count, per-series depth,
# SWA), so the -0.017 cannot be attributed to any one of them -- a lapse from this
# project's single-variable discipline, made under quota pressure. SWA's own
# contribution was independently measured as negligible inside R2 (SWA holdout 0.8109
# vs best single epoch 0.8107), which leaves slot count and depth as the live
# suspects; they moved together (6 slots forced the ceiling from 6 to 4). R3 isolates
# depth alone: back to R1's 3 slots, SWA off, but the SAME 16 GB budget R2 used --
# which plan_slices() turns into 8 slices/series (24 total, matching R2's total slice
# COUNT exactly, just concentrated in 3 slots instead of spread over 6).
RAPTOR_SLOTS = (0, 1, 2)
RAPTOR_SLICES_PER_SERIES = 10     # ceiling for the TRAIN cache; plan_slices() shrinks
                                   # it further only if RAPTOR_CACHE_BUDGET_GB requires
                                   # (16 GB at 3 slots -> 8, so this ceiling never binds)
RAPTOR_K_TRAIN = 4                # windows per study per step (the MIL bag)
# "Exhaustive" eval, matching the reference's K_EVAL~=every window position: eval_picks
# already returns the WHOLE candidate pool once k >= its size (see raptor.py), so a
# large sentinel does this for free -- no separate "eval budget" knob to keep in sync.
RAPTOR_K_EVAL = 999
# The train cache is memory-bound because it holds all ~4407 studies at once for
# repeated-epoch access; the TEST cache does not (a code-competition hidden test set is
# a handful to a few hundred studies), so it can afford far more slices per series
# without touching RAPTOR_CACHE_BUDGET_GB at all -- passed as build_window_cache's
# force_n_per for the test-only call in run_raptor_submit.
RAPTOR_EVAL_SLICES_PER_SERIES = 20
# encode() flattens B*K windows into ONE backbone forward call. Training never hit
# this (batch=4 studies x K_TRAIN=4 windows = 16 images), but R2's exhaustive eval
# pool is 18 centres x 6 slots = 108 windows/study; at predict_windows' batch=2 that
# is 216 images at 384px through a 73M-param CoAtNet in one call -> confirmed CUDA
# OOM on a T4 (tried to allocate 7.59 GiB with only 7.28 GiB free). Chunk the
# flattened image dimension regardless of B*K instead of raising GPU-shaped knobs.
RAPTOR_ENCODE_CHUNK = 16  # matched to the measured-safe training count
RAPTOR_DROP = 0.2
RAPTOR_CACHE_BUDGET_GB = 16.0
RAPTOR_BATCH = 4                  # studies per step -> K_TRAIN*BATCH images/forward
RAPTOR_EPOCHS = 12
RAPTOR_LR_HEAD = 1e-3
RAPTOR_LR_BACKBONE = 2e-5         # the encoder is adapted, not retrained
RAPTOR_LLRD = None                # layer-wise LR decay for ViT backbones; None keeps the
                                  # two-group optimiser every run through C8 used (LOG #395)
RAPTOR_USE_SWA = False        # off for R3 -- isolating depth alone from R1
RAPTOR_SWA_START_FRAC = 0.5       # begin averaging once loss has mostly stopped falling


# --------------------------------------------------------------------------- #
# Corpus line: train on the published pre-decoded memmap (see src/corpus.py)
# --------------------------------------------------------------------------- #
# Reference recipe is k=12 windows x 12 epochs, but it was trained offline with no
# 9h cap. Measured from R3 (~880 train-images/min on a T4 at 384px), 4349 studies
# x 12 windows would be ~59 min/epoch -> 12 epochs = 12h, infeasible here. k=8 x 9
# epochs lands at ~40 min/epoch ~= 6h, inside the budget, and still doubles R1-R3's
# 4-window MIL bag. Holding all 58 gold out as validation (their choice) also frees
# the 866 studies our 80/20 split was spending on a derived-label holdout.
# Back to the C2/C3/C4 budget so C6 differs from C2 in the label source ALONE.
# The local R9700 max-score run (C5: 64 slices, k=12/42, 16 ep) keeps its own copy
# of this file on that box -- do NOT scp config.py there while C5 is running, or
# the restart would silently pick up these numbers instead.
CORPUS_K_TRAIN = 8
CORPUS_K_EVAL = 24
CORPUS_EPOCHS = 9
# C1 OOM'd in TRAINING at batch=4 x k=8 = 32 images/step (14.49 of 14.56 GiB used),
# so 16 -> 32 is not a safe extrapolation: R3's proven-safe 16 images/step was already
# using most of the card. Second time an extrapolated memory guess failed (R2 was the
# inference side at 216), so this now uses the MEASURED-safe count rather than a guess.
# Total images/epoch = n_studies * k_train is independent of batch, so dropping the
# batch to 2 keeps both the doubled MIL bag AND the ~40 min/epoch budget; BatchNorm
# still sees batch*k = 16 images per forward, not 2.
CORPUS_BATCH = 2          # x k_train = 16 images/step, the count R3 ran safely
CORPUS_LEGAL_WINDOWS = False  # True: MIL window centres only where rows c-1, c, c+1 are
                              # all non-empty AND in the same slot (LOG #401). False keeps
                              # the reference rule every run through C9b used.
CORPUS_CONF_WEIGHTED_BCE = False  # True: per-study, per-finding BCE weighted by ds.weights
                                  # (label confidence, LOG #405). False = unweighted, as C1-C6w.
CORPUS_FREEZE_BN = False          # True: BatchNorm running stats frozen at the pretrained values
                                  # during training; affine params still train (LOG #407).
# Diagnosis specialist (LOG #490). None = every run so far (12 labels, every window centre).
CORPUS_TRAIN_LABELS = None        # e.g. ["PF OA"]: BCE and epoch selection on these labels only;
                                  # the other heads still exist but get zero loss weight
CORPUS_CENTRE_ALLOW = None        # e.g. list(range(4, 14)): MIL window centres restricted to these
                                  # corpus rows in training, gold evaluation AND test inference


# --------------------------------------------------------------------------- #
# Soft-label sources (the L line). Empty list = regex extractor alone.
# --------------------------------------------------------------------------- #
# Measured on the 58 gold studies (GAUGE 1, same protocol for every source):
#   regex alone                    0.865
#   kimi-k2.6 alone                0.876-0.894  (temperature is locked at 1, so
#                                   four runs spread 0.018 -- it is NOT stable)
#   qwen alone                     0.880
#   grok alone                     0.887   (this session, no api.x.ai)
#   regex + kimi, probability mean 0.889   (C2, LB 0.913)
#   regex + kimi + qwen            0.8941  (C3: 2-source -> 3-source. Cashed out
#                                   at only +0.0013 on the 58 gold held-out.)
#   regex + kimi + grok            0.8931  (C6, below)
#   regex + qwen + grok            0.8974  (offline best -- NOT chosen: vs C2 it
#                                   swaps kimi AND adds grok, two variables.)
# Pairwise median Pearson over all 4407: kimi-qwen 0.940, grok-kimi 0.760,
# grok-qwen 0.761. Qwen is near-redundant with Kimi, which is why C3 barely moved.
# C6 therefore repeats C3's structural change (2 sources -> 3) with a DECORRELATED
# third source. GAUGE 1 ranks C6 a hair BELOW C3, but every gap in the table above
# is smaller than Kimi's own 0.018 run-to-run scatter on these 58 studies, so the
# gauge cannot separate them -- the training run is the adjudicator, and the
# hypothesis being tested is independence, not the offline number.
# Probability mean beats rank mean AND is the semantically right thing to feed a
# BCE target (rank-averaging flattens every label's mean to 0.5, which is a
# distortion for training even though it is fine for ranking). Files are averaged
# together with the extractor's own output, p with p and conf with conf.
SOFT_LABEL_PARQUETS = [
    "labels_kimi_soft.parquet",
    "labels_grok_soft.parquet",
]
# RSNA_SOFT_LABELS overrides the list (comma-separated filenames, empty = regex only).
# An arm should not need its own published copy of this file just to change which
# parquets it averages -- and a control run that differs from its treatment arm by a
# re-upload is not a control. train_local.py prints the resolved list at startup, so
# the log records what actually ran rather than what this file happened to say.
_env_srcs = os.environ.get("RSNA_SOFT_LABELS")
if _env_srcs is not None:
    SOFT_LABEL_PARQUETS = [x.strip() for x in _env_srcs.split(",") if x.strip()]

# Image-teacher distillation (LOG #457). After the report sources are averaged, the
# target becomes (1 - TEACHER_LAMBDA) * report_mean + TEACHER_LAMBDA * teacher_p, where
# teacher_p is an OUT-OF-FOLD image prediction (public DINOv2 x20 + RadImageNet OOF), so
# no study is labelled by a model that saw it. Gold studies are still overridden after
# this. On the 58 gold the C6r target goes 0.898 -> 0.916 at lambda 0.5, and the two
# sources correlate only 0.32-0.59 per label. Off by default: every existing arm is
# unchanged unless its script sets these.
TEACHER_PARQUET = None
TEACHER_LAMBDA = 0.0

# ===================== tv_stub.py =====================
"""Stub torchvision so timm can import on ROCm wheels whose torchvision crashes.

Must be imported BEFORE `import timm`. Harmless on CUDA where real torchvision
already works -- we only install the stub if torchvision is missing or broken.
"""
import sys
import types

import torch.nn as nn


def install():
    try:
        import torchvision  # noqa: F401
        from torchvision.ops.misc import FrozenBatchNorm2d  # noqa: F401
        return False
    except Exception:
        pass

    class _Lazy(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            sys.modules[name] = self

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            if name[:1].isupper():
                val = type(name, (), {})
                setattr(self, name, val)
                return val
            full = f"{self.__name__}.{name}"
            val = sys.modules.get(full) or _Lazy(full)
            setattr(self, name, val)
            return val

    tv = _Lazy("torchvision")
    for sub in (
        "torchvision.datasets", "torchvision.ops", "torchvision.ops.misc",
        "torchvision.models", "torchvision.models.feature_extraction",
        "torchvision.io", "torchvision.transforms",
        "torchvision.transforms.functional", "torchvision.utils",
    ):
        _Lazy(sub)

    class FrozenBatchNorm2d(nn.BatchNorm2d):
        pass

    class InterpolationMode:
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    sys.modules["torchvision.ops.misc"].FrozenBatchNorm2d = FrozenBatchNorm2d
    sys.modules["torchvision.transforms"].InterpolationMode = InterpolationMode
    sys.modules["torchvision.transforms.functional"].InterpolationMode = InterpolationMode
    tv.ops = sys.modules["torchvision.ops"]
    tv.datasets = sys.modules["torchvision.datasets"]
    tv.transforms = sys.modules["torchvision.transforms"]
    tv.models = sys.modules["torchvision.models"]
    return True

# ===================== dicom_utils.py =====================
"""DICOM loading & slice preprocessing for RSNA Knee competition.

Incorporates lessons from top public notebooks (see LOG.md):
- Physical-scale crop: crop a constant mm extent (CROP_MM) then resize, so the
  encoder sees a consistent mm/pixel regardless of acquisition (0.71 mm/px @224).
- Volume-wise percentile normalization (1st-99th) over the WHOLE volume, not per slice.
- Central 60% slices (outer slices are mostly off-joint soft tissue).
- Laterality normalization (left/right): coronal & axial -> horizontal flip;
  sagittal -> reverse slice order. No-op when the tag is absent (graceful fallback).
- Header tags are read when present but the code degrades gracefully if stripped
  (our local test sample had them removed), falling back to fixed crops / no flip.
"""
import os
import glob
import re
import numpy as np
from PIL import Image
import pydicom

from config import (IMG_SIZE, SLICES_PER_SLOT, CROP_MM, WIN_PMIN, WIN_PMAX,
                    CENTRAL_SLICE_FRAC, LAT_MIN_OFFSET_MM)


def load_pixel_array(path: str) -> np.ndarray:
    """Read a DICOM file and return its 2D pixel array (H, W) as float32.
    Applies RescaleSlope/Intercept only if present."""
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or inter != 0.0:
        arr = arr * slope + inter
    return arr


def _pixel_spacing_mm(ds) -> float:
    """Return row spacing in mm/pixel, or 1.0 if missing."""
    try:
        ps = getattr(ds, "PixelSpacing", None)
        if ps is not None and len(ps) >= 1:
            return float(ps[0])
    except Exception:
        pass
    return 1.0


def _physical_crop(arr: np.ndarray, spacing: float, crop_mm: float,
                   out_size: int) -> np.ndarray:
    """Center-crop `crop_mm` millimetres of anatomy, then resize to out_size.

    Resize stays in float so per-slice min/max cannot scramble the volume-wise
    percentile window that runs after stacking.
    """
    crop_px = int(round(crop_mm / max(spacing, 1e-3)))
    h, w = arr.shape
    ch = min(crop_px, h); cw = min(crop_px, w)
    y0 = (h - ch) // 2; x0 = (w - cw) // 2
    arr = arr[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    img = Image.fromarray(arr, mode="F")
    img = img.resize((out_size, out_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _volume_normalize(vol: np.ndarray) -> np.ndarray:
    """Min-max scale a volume to [0,1] using its 1st/99th percentile."""
    lo, hi = np.percentile(vol, [WIN_PMIN, WIN_PMAX])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def normalise_laterality(vol: np.ndarray, plane: str, lat: str | None):
    """Map a right knee onto a left-knee convention.

    `vol` is [S, H, W]. Coronal/axial: horizontal flip. Sagittal: reverse the
    *slice order* (ML is the through-plane axis — flipping rows would be a
    vertical flip and would swap femur/tibia). Unknown laterality is a no-op.
    """
    if lat is None:
        return vol
    if str(lat).upper().startswith("R"):
        if str(plane).lower().startswith("sag"):
            return vol[::-1]
        return vol[:, :, ::-1]
    return vol


def _natural_key(p):
    """Digit-aware filename key so '.../10.dcm' sorts after '.../9.dcm'."""
    base = os.path.basename(p)
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", base)]


def _spatial_sort(files):
    """Order slices along the through-plane axis (the ImagePositionPatient
    component with the LARGEST spread across slices), not a fixed axis: for a
    sagittal knee the stack direction is medial-lateral (patient x), so sorting
    on z would scramble the 2.5D groups. Requires >=80% of slices positioned,
    else InstanceNumber (>=80%), else a natural filename sort."""
    n = len(files)
    pos, inst = [], []
    for p in files:
        ipp = inum = None
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            inum = getattr(ds, "InstanceNumber", None)
        except Exception:
            pass
        pos.append([float(v) for v in ipp] if ipp is not None and len(ipp) >= 3
                   else None)
        try:
            inum = int(inum) if inum is not None else None
        except Exception:
            inum = None
        inst.append(inum)
    n_pos = sum(v is not None for v in pos)
    if n_pos >= max(2, int(0.8 * n)):
        arr = np.array([v if v is not None else [np.nan] * 3 for v in pos],
                       dtype=float)
        axis = int(np.nanargmax(np.nanmax(arr, 0) - np.nanmin(arr, 0)))
        key = arr[:, axis]
        order = sorted(range(n), key=lambda i: (np.isnan(key[i]), key[i]))
        return [files[i] for i in order]
    n_inst = sum(v is not None for v in inst)
    if n_inst >= max(2, int(0.8 * n)):
        order = sorted(range(n), key=lambda i: (inst[i] is None, inst[i] or 0))
        return [files[i] for i in order]
    return sorted(files, key=_natural_key)


def load_series_stack(folder: str, plane: str = "Sagittal", lat: str | None = None,
                      n_slices: int = SLICES_PER_SLOT, out_size: int = IMG_SIZE,
                      crop_mm: float = CROP_MM) -> np.ndarray:
    """Load a series folder -> [n_slices, H, W] float32 in [0, 1].

    Headers are used when present (PixelSpacing, ImagePositionPatient for
    through-plane order). Laterality is applied to the stacked volume so a
    sagittal right-knee correction reverses slice order rather than flipping
    each frame vertically. Missing/empty folders return zeros.
    """
    empty = np.zeros((n_slices, out_size, out_size), dtype=np.float32)
    if not folder or not os.path.isdir(folder):
        return empty
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not files:
        return empty

    files = _spatial_sort(files)
    n = len(files)
    a = int(n * (1 - CENTRAL_SLICE_FRAC) / 2)
    b = max(a + 1, n - a)
    files = files[a:b]

    spacing = None
    planes = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if spacing is None:
                spacing = _pixel_spacing_mm(ds)
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            if slope != 1.0 or inter != 0.0:
                arr = arr * slope + inter
            arr = _physical_crop(arr, spacing or 1.0, crop_mm, out_size)
            planes.append(arr)
        except Exception:
            continue
    if not planes:
        return empty

    vol = np.stack(planes, 0)                 # [M, H, W]
    vol = _volume_normalize(vol)
    vol = normalise_laterality(vol, plane, lat)
    M = vol.shape[0]
    idx = np.linspace(0, M - 1, n_slices).round().astype(int)
    out = np.zeros((n_slices, out_size, out_size), dtype=np.float32)
    for i, fi in enumerate(idx):
        out[i] = vol[int(np.clip(fi, 0, M - 1))]
    return out


def load_series(folder: str, plane: str = "Sagittal", lat: str | None = None,
                n_slices: int = SLICES_PER_SLOT, out_size: int = IMG_SIZE,
                crop_mm: float = CROP_MM) -> np.ndarray:
    """Legacy wrapper: [n_slices, 1, H, W] float32 in [0, 1]."""
    stack = load_series_stack(folder, plane, lat, n_slices, out_size, crop_mm)
    return stack[:, None, :, :]


def _tag_side_from_ds(ds) -> str | None:
    lat = getattr(ds, "Laterality", None) or getattr(ds, "ImageLaterality", None)
    if not lat:
        return None
    s = str(lat).strip().upper()
    if s.startswith("L"):
        return "L"
    if s.startswith("R"):
        return "R"
    return None


def _position_side_from_ds(ds) -> str | None:
    """LPS: the right knee sits at negative patient-x. Near-isocentre is ignored."""
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None:
        return None
    try:
        x = float(ipp[0])
    except Exception:
        return None
    if abs(x) < LAT_MIN_OFFSET_MM:
        return None
    return "R" if x < 0 else "L"


def read_laterality_full(folder: str) -> tuple:
    """Return (tag_side, pos_side), each 'L'/'R'/None, from a series folder."""
    if not folder or not os.path.isdir(folder):
        return None, None
    files = glob.glob(os.path.join(folder, "*.dcm"))
    if not files:
        return None, None
    tag, xs = None, []
    for f in files[:20]:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            if tag is None:
                tag = _tag_side_from_ds(ds)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None:
                try:
                    xs.append(float(ipp[0]))
                except Exception:
                    pass
        except Exception:
            continue
    pos = None
    if xs:
        x = float(np.median(xs))
        if abs(x) >= LAT_MIN_OFFSET_MM:
            pos = "R" if x < 0 else "L"
    return tag, pos


# --------------------------------------------------------------------------- #
# sequence recovery (v14)
# --------------------------------------------------------------------------- #
# The competition CSV's Fluid_Sensitive and Fat_Suppression are perfectly correlated
# in this corpus -- every series is (0,0) or (1,1) -- so between them they carry one
# bit, and the six-slot scheme collapses to three (plane x that bit). The reference
# pipeline instead RECOVERS both from the header, where the weighting (T1 / PD / T2 /
# GRE) and the fat-suppression flag are independent. That is what makes a
# "fluid-sensitive, not fat-suppressed" slot a real thing rather than an empty one.
_SEQ_SEP = re.compile(r"[_\-/\\]+")
_FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat sat|fat_sat|spair|spir|stir|\bsp[ai]r\b|"
    # `t2 fs` without a right-hand boundary also matches "T2 FSE" -- fast spin echo,
    # which is not fat-suppressed. Anchor the fs token on both sides.
    r"chess|\bfsat\b|frfse fs\b|t2 fs\b|pd fs\b|suppress")
_T1_RX = re.compile(r"\bt1\b|t1w|t1 tse|t1 fse|mprage")
_T2_RX = re.compile(r"\bt2\b|t2w|t2 tse|t2 fse|stir")
_PD_RX = re.compile(r"\bpd\b|pdw|proton|dp\b|densidad")
# GE writes SAT_GEMS for spatial saturation, so ScanOptions is matched as exact
# tokens; a substring test on "SAT" fires on series that are not fat-suppressed.
_FATSAT_OPTS = {"FS", "FSAT", "SP", "SPIR", "SPAIR", "CHEMSAT"}

_SEQ_FIELDS = ("SeriesDescription", "SequenceName", "ScanOptions",
               "RepetitionTime", "EchoTime", "ScanningSequence")


def read_series_meta(folder: str) -> dict:
    """Header fields needed to recover (weighting, fat suppression) for a series.

    Reads at most a handful of files and stops at the first one that carries a
    SeriesDescription, because these fields are per-series constants. Returns {}
    when nothing is readable, which the caller treats as "fall back to the CSV".
    """
    out = {}
    if not folder or not os.path.isdir(folder):
        return out
    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))[:5]
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        for k in _SEQ_FIELDS:
            v = getattr(ds, k, None)
            if v is None or k in out:
                continue
            if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                v = "|".join(str(x) for x in v)
            out[k] = str(v)
        if out.get("SeriesDescription"):
            break
    return out


def recover_sequence(meta: dict):
    """(fluid, fatsat, weight) from header fields, or None when unrecoverable.

    Mirrors the reference pipeline's `annotate()`: the description/sequence name
    decides the weighting when it says so, otherwise TR/TE decide, and a gradient-
    echo scanning sequence is called GRE rather than guessed at.
    """
    if not meta:
        return None
    desc = (meta.get("SeriesDescription", "") + " " +
            meta.get("SequenceName", "")).lower()
    desc = _SEQ_SEP.sub(" ", desc)
    opts = [t.strip().upper() for t in meta.get("ScanOptions", "").split("|")]
    fatsat = bool(_FATSAT_RX.search(desc)) or any(t in _FATSAT_OPTS for t in opts)

    def _num(k):
        try:
            return float(meta.get(k))
        except (TypeError, ValueError):
            return None

    tr, te = _num("RepetitionTime"), _num("EchoTime")
    gre = "GR" in meta.get("ScanningSequence", "").upper()
    t1 = bool(_T1_RX.search(desc))
    t2 = bool(_T2_RX.search(desc))
    pdw = bool(_PD_RX.search(desc))
    if t1 and not t2 and not pdw:
        weight = "T1"
    elif t2 and not pdw:
        weight = "T2"
    elif pdw:
        weight = "PD"
    elif gre:
        weight = "GRE"
    elif tr is not None and tr < 800:
        weight = "T1"
    elif te is not None and te > 60:
        weight = "T2"
    elif tr is not None and tr >= 800:
        weight = "PD"
    else:
        return None                      # nothing in the header decided it
    return (weight in ("PD", "T2")), fatsat, weight


def read_laterality(folder: str) -> str | None:
    """Return 'L'/'R' from DICOM tags, else None (no IPP fallback here).

    IPP fallback is decided at cache-build time after measuring agreement
    with the tag, because a wrong flip is worse than no flip.
    """
    tag, _pos = read_laterality_full(folder)
    return tag


# ---- backward-compatible one-file helper (used by early smoke tests) ----
def preprocess_dicom(path: str, size: int = IMG_SIZE) -> np.ndarray:
    arr = load_pixel_array(path)
    lo, hi = np.percentile(arr, [WIN_PMIN, WIN_PMAX])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0

# ===================== report_extractor.py =====================
"""Multilingual graded report extractor for RSNA Knee (v6 labeler).

Ported from the public Kaggle notebook "read the report, then the knee"
(romanrozen), whose labeler is a strict superset of the v4 lexicon we ran in
v3-v5. Beyond the nine-language stems it adds:

  * `unwrap`        - rejoin hard-wrapped report lines before clause splitting
  * directional negation - negation scopes forward/backward from the negator
    rather than over the whole clause ("... fracture ... without collapse")
  * `_grade_of`     - numeric grades read on a per-structure scale
  * `_score_paired` - side-aware scoring for the four paired structures
  * `_score_oa`     - compartment scoping + unlocalised-cartilage inheritance
  * synovitis backoff - orders the ~87% of reports that never write the word by
    inflammatory context, entering at low confidence (0.18)

Measured head-to-head against the v3-v5 extractor on the 58 annotated studies
(GAUGE 1, identical protocol): macro agreement AUC 0.822 -> 0.865, better on
10/12 targets (Medial OA +0.110, Contusion +0.081, PF OA +0.073, Effusion
+0.067, Synovitis +0.065). See LOG.md "v6".

The previous extractor is archived verbatim at
`experiments/_archive/report_extractor_v3.py` so the swap stays reproducible.

`extract(report) -> (grades, conf)` keeps the dataset.py contract.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

from config import LABELS

TARGETS = list(LABELS)


_PRE = str.maketrans({
    "ı": "i", "İ": "i", "I": "i", "ß": "ss", "đ": "d", "Đ": "d",
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae",
})


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.translate(_PRE).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("­", "")
    text = re.sub(r"[_\-/\\]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")


def unwrap(text: str) -> str:
    """Rejoin lines that a fixed-width layout broke mid-sentence.

    A large share of this corpus arrives hard-wrapped at some column, so a sentence is
    split across two lines with no punctuation at the break. Splitting on newlines then
    severs the finding from its anatomy - `... y parte de la raiz` / `meniscal posterior
    del menisco lateral con extrusion asociada` puts the tear in one clause and the
    meniscus in the next, and neither clause says anything on its own. A line that does
    not end in sentence punctuation is a continuation, not a statement.
    """
    if not isinstance(text, str):
        return ""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if out and out[-1] and not re.search(r"[.;:!?>*•]$", out[-1]) \
                and len(out[-1].split()) >= 4 and s and not s[:1].isupper():
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return "\n".join(out)


def clauses(text: str):
    """Split into clauses; attach `header:` lines to the value beneath them."""
    norm = normalize(unwrap(text) if FEATURES["unwrap"] else text)
    raw = [c.strip() for c in _SENT_SPLIT.split(norm) if c and c.strip()]
    merged = []
    for i, c in enumerate(raw):
        if c.endswith(":") and len(c.split()) <= 14 and i + 1 < len(raw):
            merged.append(c + " " + raw[i + 1])
            if FEATURES.get("heading_value_negation", True):
                # The bare heading names the finding and asserts nothing; emitting it
                # as its own clause re-introduces the false positive that the merged
                # `heading: value` clause exists to resolve.
                continue
        merged.append(c)
    out = []
    for c in merged:
        out.append(c)
        if len(c.split()) > 25:
            out.extend(p.strip() for p in c.split(",") if len(p.split()) > 2)
    return out


# Each rule below that is not obviously right is behind a flag, so that the notebook can
# turn it off and re-measure rather than assert that it helps.
FEATURES = {
    "unwrap": True,               # rejoin hard-wrapped lines before splitting
    "directional_negation": True,  # scope negation by direction instead of by clause
    "oa_inherit": True,            # unlocalised cartilage statements reach all three
    "graded_pathology": True,      # read numeric grades on a per-structure scale
    "synovitis_backoff": True,     # order the silent majority by inflammatory context
    # project-side addition (v6), measured in LOG.md: a `<finding>:` heading whose
    # value line denies the finding must read as a negative, not an assertion.
    "heading_value_negation": True,
}


def _rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts))


# --------------------------------------------------------------------------- #
# polarity
# --------------------------------------------------------------------------- #
# Negation is scoped by direction, not by clause. `Subchondral insufficiency fracture at
# the medial tibial plateau without articular surface collapse` contains a negator and
# asserts a fracture: `without` governs what follows it, and the fracture precedes it.
# Reading negation at clause scope turns that sentence into a denial - and it is the
# house style of one of the larger reporting sites here, so the error is systematic
# rather than occasional.
PRE_NEG = _rx(
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsence\b",
    r"\bno evidence\b", r"\bfree of\b", r"\bnone\b", r"\bneither\b", r"\bnor\b",
    r"\bsin\b", r"\bno hay\b", r"\bausencia\b", r"\bausentes?\b", r"\bno se\b",
    r"\bpas de\b", r"\bsans\b", r"\baucune?\b",
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b",
    r"\bkeine?[nmrs]?\b", r"\bohne\b", r"\bnicht\b", r"\bkein\b",
    r"\bnema\b", r"\bbez\b", r"\bnisu\b", r"\bnije\b",
    r"\bδεν\b", r"\bχωρις\b", r"ουδεν", r"\bουτε\b",
    r"\bбез\b", r"\bне\b", r"липсва", r"\bняма\b",
)

# Turkish and a few Croatian forms put the negator at the end of the sentence, so these
# govern what precedes them instead.
POST_NEG = _rx(
    r"\byok\b", r"\byoktur\b", r"izlenmemekte", r"saptanmadi", r"\bdegil\b",
    r"gozlenmemekte", r"mevcut degil", r"eslik etmiyor", r"\bizlenmedi\b",
    r"izlenmemistir", r"saptanmamistir", r"gorulmemistir", r"\bnema znakova\b",
    r"bez znakova",
)

NEGATION = _rx(PRE_NEG.pattern, POST_NEG.pattern, r"\bunremarkable\b")

NEG_WINDOW = 90

# Value-line denials: what a `<finding>:` heading is answered with when the finding is
# absent. Anchored at the start of the value, so these never fire mid-sentence.
VALUE_NEG = _rx(
    r"\bnone\b", r"\bnil\b", r"\bnegative\b", r"\bno\b", r"\bnot seen\b",
    r"\bninguna?\b", r"\bausentes?\b", r"\bnegativo\b",
    r"\baucune?\b", r"\bneant\b", r"\bpas de\b",
    r"\bkeine?[nmrs]?\b", r"\bnegativ\b",
    r"\bgeen\b",
    r"\byok(tur)?\b", r"\bmevcut degil\b",
    r"\bnema\b", r"\bbez\b",
    r"\bουδεν", r"\bοχι\b", r"\bκαμια\b",
    r"\bняма\b", r"\bне\b", r"\bлипсва",
    r"\(-\)", r"^-\s*$",
)


def _negated(clause: str, start: int, end: int) -> bool:
    """True when a negation trigger governs the span [start, end) of this clause."""
    for m in PRE_NEG.finditer(clause):
        if m.end() <= start and start - m.end() <= NEG_WINDOW:
            # `... intact but with a tear` - a contrast conjunction closes the scope.
            if not re.search(r"\b(but|however|ancak|fakat|pero|maar|aber|no i|ali|"
                             r"ωστοσο|αλλα|но)\b", clause[m.end():start]):
                return True
    for m in POST_NEG.finditer(clause):
        if m.start() >= end and m.start() - end <= NEG_WINDOW:
            return True
    # `Fractures :\nAucune.` / `Effusion: none.` -- the heading names the finding and
    # the value line denies it. The negator sits AFTER the anchor, so neither loop
    # above sees it, and the ported labeler grades the denial as an assertion (0.87
    # on a report that explicitly clears the finding). A negator separated from the
    # anchor by nothing but a colon governs it backwards.
    if FEATURES.get("heading_value_negation", True):
        rest = clause[end:]
        # `\w*` first: the anatomy patterns match stems (`fractur`), so the rest of the
        # word (`es`) still sits between the anchor and the colon.
        sep = re.match(r"^\w{0,4}[\s:;,\-\.]{0,6}", rest)
        val = rest[sep.end():] if sep else rest
        if val and (VALUE_NEG.match(val) or NORMALITY.match(val)
                    or NORMAL_PHRASE.match(val)):
            return True
    return False

NORMALITY = _rx(
    r"\bnormal", r"\bintact\b", r"\bpreserved\b", r"\bwithin normal limits\b",
    r"limites normales", r"\bconservad", r"\bintegr", r"\bnormales\b",
    r"\bdoga(l|ll)\b", r"korunmus", r"\bnormaldir\b", r"olagan",
    r"\buredn", r"\bocuvan", r"\bodrzan", r"\bintakt", r"\bprimjeren",
    r"\bodrzanog kontinuiteta", r"\bodržan",
    r"φυσιολογικ", r"ακεραι", r"δεν παρατηρουνται", r"δεν σημειωνονται",
    r"unauffallig", r"regelrecht", r"\bo\.?b\.?\b",
    r"нормал", r"запазен", r"съхранен", r"\bбез особености\b", r"интактн",
    r"\bgaaf\b", r"\bnormaal\b",
)

# "Negator + abnormality-noun" is how most of these languages assert normality:
# `sin alteraciones`, `ohne Auffalligkeiten`, `geen afwijkingen`, `bez osobitosti`.
# Read literally each one is a negation, so the old guard (`normality AND NOT negation`)
# threw every one of them away and left the structure looking unmentioned instead of
# explicitly clear. That matters to a rank metric: a ligament a radiologist looked at and
# called intact must rank below one the report never mentions, not equal to it.
NORMAL_PHRASE = _rx(
    r"\bsin alteracion", r"\bsin cambios\b", r"\bsin particularidad",
    r"\bsin hallazgos\b", r"\bsin lesion", r"\bsin signos de (rotura|lesion)",
    r"\bcontinu[oa]s?\b", r"\bcontinuidad conservada\b",
    r"\bno abnormalit", r"\bno significant abnormalit", r"\bunremarkable\b",
    r"\bno evidence of (tear|injury|abnormalit)",
    r"\bohne auffalligkeit", r"\bkein nachweis\b", r"\bohne befund\b",
    r"\bgeen afwijking", r"\bzonder afwijking",
    r"\bsans anomalie", r"\bpas d[e']anomalie",
    r"\bbez osobitosti\b", r"\bbez znakova (rupture|lezije)\b",
    r"\bbez patoloskih\b",
    r"χωρις αλλοιωσ", r"χωρις παθολογ", r"δεν παρατηρουνται (αξιολογα|παθολογ)",
    r"\bбез особености\b", r"\bбез патологич", r"\bбез данни за\b",
    r"\bozel bir ozellik yok", r"\bpatolojik bulgu (yok|izlenmemis)",
)

UNCERTAIN = _rx(
    r"\bpossible\b", r"\bprobable\b", r"\bsuspicious\b", r"\bsuspected?\b",
    r"cannot (be )?exclude", r"\bmay\b", r"\bquestionable\b", r"\bequivocal\b",
    r"\br/o\b", r"\bdd\b", r"\blikely\b", r"\bsuggest", r"\bcompatible with\b",
    r"\bposible\b", r"sin criterios categoricos", r"\bdudos", r"\bsugier",
    r"\bmuhtemel\b", r"\bolasi\b", r"\bsupheli\b", r"\bizlenim", r"\bdusundur",
    r"\bmoguce\b", r"\bvjerojatno\b", r"\bsumnja\b", r"\bmoze odgovarati\b",
    r"πιθαν", r"υποπτ",
    r"\bmoglich", r"\bverdachtig", r"\bfraglich", r"\bv\.?a\.?\b", r"\bwohl\b",
    r"\bвъзможно\b", r"\bвероятно\b", r"суспект",
    r"\bmogelijk\b", r"\bverdacht\b",
)

# ==== CELL ====

# --------------------------------------------------------------------------- #
# pathology vocabulary
# --------------------------------------------------------------------------- #
TEAR = _rx(
    r"\btear", r"\btorn\b", r"\brupture", r"\bdisruption\b", r"discontinuit",
    r"\bavuls", r"\bmacerat", r"\bbuckethandle\b", r"bucket handle",
    r"\brotura\b", r"\broturas\b", r"\bruptura", r"\bdesgarro", r"\broto\b",
    r"\bdechirure", r"\bdechire",
    r"\bscheur", r"\bruptuur", r"gescheurd",
    r"\briss\b", r"einriss", r"\bruptur", r"zerreiss", r"\blasion", r"\bausriss",
    r"\byirtik", r"\byirtig", r"\bkopma\b", r"butunluk kaybi", r"\brupturu\b",
    r"devamsizlik", r"\brupture\b", r"\bdevamliligi secilememis",
    r"\bpuknuce", r"\bprekid\b", r"\bpukotin", r"\bruptur",
    r"ρηξη", r"ρηξις", r"ρηγμα", r"ασυνεχεια",
    r"руптура", r"разкъсв", r"разрив", r"скъсв", r"\bлезия\b",
)

DEGEN = _rx(
    r"degenerat", r"\bmucoid\b", r"\bmyxoid\b", r"\bfray", r"\bfissur",
    r"dejeneratif", r"\bmukoid\b", r"degenerativn", r"εκφυλ", r"дегенерат",
    r"\bμυξοειδ", r"\bμυξωδ", r"\bmeniskopat", r"\bmeniscopath",
    r"\bmuco ?ide\b", r"aufgefasert", r"\bdejenerasyon\b",
)

INJURY = _rx(
    r"\binjur", r"\bsprain", r"\blesion", r"\blasion", r"\bedema\b", r"\boedema\b",
    r"\bodem\b", r"\bedem\b", r"\bοιδημα", r"\bодем", r"\bедем", r"\bstrain\b",
    r"\bhigh signal\b", r"\bsignal alteration\b", r"\bhiperintens", r"\bhyperintens",
    r"aumento de senal", r"alteracion de senal", r"cambio de senal",
    r"\bsignalanhebung", r"\bsignalalteration", r"verhoogd signaal", r"sinyal artis",
    r"αυξημενο σημα", r"повишен сигнал", r"\besguince\b",
    r"\bthicken", r"\bzadebljanje\b", r"\bverdikking\b", r"\bdistenzij",
    r"\blaksite\b", r"\blaxity\b", r"\bpartial\b", r"\bparcijaln", r"\bparcial",
    r"\bpartiel", r"\bpartiell",
)

# A numeric grade is the most precise thing a knee report says, and it means different
# things in different places: grade 3 of a meniscus is a tear by definition, grade 1 or 2
# is intrasubstance signal that never reaches the surface and is not one. A ligament runs
# the other way round - grade 1 is a stretch, grade 2 a partial tear. So the grade is
# read as a number and interpreted per structure rather than folded into one pathology
# vocabulary.
_GRADE_RX = re.compile(
    r"(?:grade|grad|grado|grau|derece|stupnja|stupanj|βαθμ|степен|icrs|outerbridge)"
    r"[\s:]*(?:grade\s*)?([1-4]|iv|iii|ii|i)\b"
)
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}


def _grade_of(clause: str):
    """Highest numeric grade stated in a clause, or None."""
    best = None
    for m in _GRADE_RX.finditer(clause):
        v = m.group(1)
        n = _ROMAN.get(v, None) if not v.isdigit() else int(v)
        if n is not None and (best is None or n > best):
            best = n
    return best

# --------------------------------------------------------------------------- #
# anatomy
# --------------------------------------------------------------------------- #
ANAT = {
    "ACL": _rx(
        r"anterior cruciate", r"\bacl\b",
        r"cruzado anterior", r"\blca\b",
        r"croise anterieur",
        r"voorste kruisband", r"\bvkb\b",
        r"vorderes kreuzband", r"vorderen kreuzband", r"vordere kreuzband",
        r"on capraz", r"\bocb\b", r"anterior capraz",
        r"prednji krizni", r"prednjeg krizn",
        r"προσθι[οα][^ ]* χιαστ", r"προσθιου χιαστου", r"χιαστο[^ ]* συνδεσμ",
        r"\bχιαστ\w*",
        r"предна кръстна", r"предната кръстна", r"предна кръста",
        r"cruciate ligaments", r"ligamentos cruzados", r"ligaments croises",
        r"kruisbanden", r"kreuzbander", r"capraz baglar", r"krizn[a-z]* ligament[a-z]*",
        r"χιαστοι συνδεσμ", r"χιαστων συνδεσμ", r"кръстните връзки", r"кръстни връзки",
    ),
    "MCL": _rx(
        r"medial collateral", r"\bmcl\b", r"tibial collateral",
        r"colateral medial", r"colateral interno", r"\blcm\b",
        r"collateral medial", r"collateral interne",
        r"mediale collaterale", r"binnenband", r"\b(mediale|laterale) banden\b",
        r"\bcollaterale banden\b",
        r"innenband", r"mediales? kollateral",
        r"\bic yan bag", r"medial kollateral", r"\biyb\b", r"medyal kollateral",
        r"medijalni kolateraln", r"medijalnog kolateraln",
        r"εσω πλαγι", r"εσωτερικο πλαγι", r"\bπλαγι\w* συνδεσμ", r"\bπλαγιοι\b",
        r"медиален колатерал", r"вътрешна странична", r"\bколатерал\w*",
        r"\bcolaterales\b", r"\bcollateraux\b", r"\bcollateralen\b", r"\bkolateralni\b",
        r"collateral ligaments", r"ligamentos colaterales", r"ligaments collateraux",
        r"collaterale banden", r"kollateralbander", r"seitenbander", r"yan baglar",
        r"kolateraln[a-z]* ligament[a-z]*", r"πλαγιοι συνδεσμ", r"πλαγιων συνδεσμ",
        r"колатерални връзки", r"страничните връзки",
    ),
    "Medial Meniscus": _rx(
        r"medial meniscus", r"\bmm\b(?= tear)", r"medial menisc",
        r"menisco medial", r"menisco interno",
        r"menisque medial", r"menisque interne",
        r"mediale meniscus", r"binnenmeniscus",
        r"innenmeniskus", r"medialen? meniskus", r"innenmeniskushinterhorn",
        r"medyal menisk", r"\bic menisk",
        r"medijalni meniskus", r"medijalnog meniskusa", r"medijalnom meniskusu",
        r"medijaln\w* menisk\w*", r"\bmedijalnog meniska\b", r"medijalni menisk",
        r"εσω μηνισκ", r"μηνισκ[^ ]* του εσω", r"εσω διαμερισμα[^.]{0,40}μηνισκ",
        r"медиалния менискус", r"медиален менискус", r"вътрешния менискус",
        r"oba meniska", r"both menisci", r"ambos meniscos", r"beide menisci",
        r"her iki menisku", r"amfoteroi\w* mhnisk", r"αμφοτερ\w* μηνισκ",
        r"двата менискуса", r"medial (and|&) lateral menisc",
    ),
    "Lateral Meniscus": _rx(
        r"lateral meniscus", r"lateral menisc",
        r"menisco lateral", r"menisco externo",
        r"menisque lateral", r"menisque externe",
        r"laterale meniscus", r"buitenmeniscus",
        r"aussenmeniskus", r"lateralen? meniskus", r"aussenmeniskushinterhorn",
        r"lateral menisk", r"\bdis menisk",
        r"lateralni meniskus", r"lateralnog meniskusa", r"lateralnom meniskusu",
        r"lateraln\w* menisk\w*", r"\blateralnog meniska\b",
        r"εξω μηνισκ", r"μηνισκ[^ ]* του εξω", r"εξω διαμερισμα[^.]{0,40}μηνισκ",
        r"латералния менискус", r"латерален менискус", r"външния менискус",
        r"oba meniska", r"both menisci", r"ambos meniscos", r"beide menisci",
        r"her iki menisku", r"αμφοτερ\w* μηνισκ",
        r"двата менискуса", r"medial (and|&) lateral menisc",
    ),
}

# Osteoarthritis is written as cartilage damage far more often than as a diagnosis.
OA_EVIDENCE = _rx(
    r"osteoarthrit", r"\barthros", r"\bgonarthros", r"\bosteoarthros",
    r"chondropath", r"chondromalac", r"condropat", r"condromalac", r"\bchondros",
    r"\bchondrosis\b", r"chondral (loss|defect|ulcer|thinning|injury|fissur|wear)",
    r"cartilage (loss|thinning|defect|fissur|wear|damage|heterogeneity|irregularit)",
    r"(loss|thinning|fissur|defect|ulcer|erosion|denudation) of[^.]{0,20}cartilage",
    r"articular cartilage[^.]{0,30}(loss|thin|fissur|defect|erosion|wear|irregular)",
    r"osteophyt", r"osteofit", r"osteofyt", r"osteofito", r"osteophyten", r"spurring",
    r"joint space narrowing", r"pinzamiento articular", r"reduced joint space",
    r"kikirdak kayb", r"kikirdak incelme", r"kondropati", r"kondral", r"kikirdak dejener",
    r"eklem aralig\w* daral", r"eklem mesafesi daral", r"kikirdak kalinlig\w* azal",
    r"kraakbeen", r"gonartrose", r"artrose", r"\bknorpel", r"arthrose", r"gonarthrose",
    r"hrskavic", r"hondromalac", r"artroz", r"osteoartrit", r"artrotsk", r"artrotick",
    r"\boa promjen", r"\boa\b", r"degenerativne promjene hrskav",
    r"χονδρ[^ ]*παθ", r"αρθριτ", r"αρθρωσ", r"οστεοφυτ", r"χονδρομαλακ",
    r"αρθρικου χονδρου", r"εξαλειψη του αρθρικου χονδρου", r"διαβρωση του αρθρικου χονδρ",
    r"λεπτυνση[^.]{0,30}χονδρ", r"φθορα[^.]{0,20}χονδρ",
    r"артроз", r"хондропат", r"остеофит", r"хрущял[^.]{0,40}(изтън|увред|дефект|липс)",
    r"изтъняване[^.]{0,30}хрущял", r"хондромалац",
    r"ulcera[s]? condral", r"cartilago[^.]{0,25}(perdida|adelgaz)",
    r"icrs grade", r"icrs\b", r"outerbridge", r"\bdenudation\b", r"denudacij",
    r"erozivne promjene", r"\berosion of[^.]{0,20}cartilage",
    r"kraakbeenlijden", r"kraakbeenverlies",
)

# --------------------------------------------------------------------------- #
# where in the joint a cartilage statement sits
# --------------------------------------------------------------------------- #
# Tibiofemoral structures. Used only inside a clause that already carries OA evidence,
# so bare "condyle" is safe here and would not be elsewhere.
TF_SITE = _rx(
    r"compartment", r"compartimento", r"compartiment", r"kompartman", r"kompartiment",
    r"kompartment", r"odjelj", r"διαμερισμα", r"компартм", r"\bотдел",
    r"femorotibial", r"tibiofemoral", r"femoro tibial", r"femorotibiaal",
    r"femorotibijaln", r"феморотибиал", r"\bft zglob", r"tibiofemoraln",
    r"condyle", r"condilo", r"kondyl", r"kondil", r"condyl", r"κονδυλ",
    r"кондил", r"\bplateau", r"\bplato\b", r"platillo", r"meseta", r"плато",
    r"tibiaplateau", r"tibijaln\w* plato", r"tibyal plato", r"tibia plato",
    r"κνημιαι", r"μηριαι", r"weightbearing", r"weightbaring", r"zona de carga",
    r"dragende deel", r"agirlik tasiyan", r"\bfemur\b", r"\btibia\b", r"\bfemoral\b",
    r"\btibial\b", r"\bfemura\b", r"\btibije\b", r"\bmesarthrio\b", r"μεσαρθριο",
)

# Patellofemoral structures.
PF_SITE = _rx(
    r"patellofemoral", r"femoropatellar", r"femoropatelar", r"patelofemoral",
    r"retropatellar", r"retrorotulian", r"trochlea", r"troclea", r"troklea",
    r"trochlear", r"trohlej", r"τροχιλ", r"\bpatella", r"\bpatellar", r"rotulian",
    r"\brotula\b", r"\bpatele\b", r"patellofemoraal", r"femoropatellair",
    r"επιγονατιδ", r"μηροεπιγονατιδ", r"пател", r"феморопател",
    r"anterior compartment", r"compartimento anterior", r"prednj\w* odjeljk",
    r"\bfp zglob", r"\bpf zglob", r"\bfaset", r"\bfacet", r"patellofemoraln",
)

SIDE_MEDIAL = _rx(
    r"\bmedial\w*", r"\bmedyal\w*", r"\bmedijaln\w*", r"\bmediaal\w*",
    r"\bmediale\w*", r"\binterno\b", r"\binterna\b", r"\binternos\b", r"\binterne\b",
    r"\binnen\w*", r"\bic\b", r"\bunutarnj\w*", r"\bεσω\w*", r"\bεσωτερικ\w*",
    r"\bмедиал\w*", r"\bвътреш\w*", r"\bbinnen\w*", r"\bmediaal\b", r"\bmediales?\b",
)
SIDE_LATERAL = _rx(
    r"\blateral\w*", r"\bexterno\b", r"\bexterna\b", r"\bexternos\b", r"\bexterne\b",
    r"\bdis\b", r"\blateraln\w*", r"\baussen\w*", r"\bbuiten\w*", r"\bεξω\w*",
    r"\bεξωτερικ\w*", r"\bлатерал\w*", r"\bвъншн\w*", r"\bvanjsk\w*",
)
SIDE_ANTERIOR = _rx(
    r"\banterior\w*", r"\bant\b", r"\bon\b", r"\bprednj\w*", r"\bvorder\w*",
    r"\bvoorste\b", r"\bπροσθι\w*", r"\bпредн\w*", r"\banteriyor\w*", r"\bavant\b",
    r"\banterieur\w*",
)

GLOBAL_OA = _rx(
    r"tri ?compartment", r"all three compartment", r"global(ised)? (oa|osteoarthrit)",
    r"\bgonarthros", r"\bgonartros", r"\bgonarthrose", r"\bgonartrose", r"gonartro",
    r"goanrtrot", r"gonartrot",
    r"osteoarthritis of the knee", r"artrosis (de |)(la )?rodilla", r"knee osteoarthrit",
    r"\bdiz osteoartrit", r"\bgonartroz", r"artroza koljena",
    r"οστεοαρθριτιδα", r"αρθριτιδα του γονατος", r"εκφυλιστικη οστεοαρθριτ",
    r"артроза на колянната", r"гонартроз",
    r"degenerative joint disease", r"\bdjd\b", r"three compartments",
    r"compartmens", r"compartments",
)

# --------------------------------------------------------------------------- #
# self-declaring findings
# --------------------------------------------------------------------------- #
DIRECT = {
    "Effusion": _rx(
        r"\beffusion", r"joint fluid", r"intra ?articular fluid", r"\bhydrops\b",
        r"\bhemarthros", r"\bhaemarthros",
        r"derrame articular", r"\bderrame\b", r"liquido articular", r"hemartrosis",
        r"epanchement",
        r"gewrichtsvocht", r"\bvocht\b", r"gewrichtseffusie", r"opzetting van suprapatell",
        r"gelenkerguss", r"\berguss\b", r"gelenksergu", r"gelenksflussigkeit",
        r"eklem\w* ic\w* sivi", r"efuzyon", r"eklem sivisi", r"eklem mesafesinde sivi",
        r"sivi (miktari|artisi|birikimi)", r"sivi artis", r"\bsivi\b[^.]{0,25}artmis",
        r"\bizljev", r"\bizliv", r"zglobn[^ ]* tekucin", r"\bhidrops\b",
        r"αρθρικ[^ ]* υγρ", r"υγρου ενδαρθρικα", r"ενδαρθρικ[^ ]* υγρ", r"ποσοτητα υγρου",
        r"ενδαρθρικ", r"αρθρικη συλλογη", r"υγρο στην αρθρωση", r"υγρου στην αρθρωση",
        r"συλλογη υγρου", r"ενθαρθρικ",
        r"ставен излив", r"излив", r"ставна течност", r"синовиална течност",
    ),
    "Synovitis": _rx(
        r"synovit", r"sinovit", r"synovial (thickening|proliferation|hypertroph)",
        r"thicken\w* synovial", r"hypertroph\w* of the synovium",
        r"synoviale? (verdikking|proliferatie)", r"verdikkingen van (het )?synovium",
        r"synovialitis", r"synovialis(verdickung|proliferation)", r"reizsynovial",
        r"sinovijalitis", r"sinovitis", r"zadebljanje sinovij", r"proliferacij\w* sinovij",
        r"sinovijaln\w* proliferacij",
        r"υμενιτιδα", r"συνοβιτιδα", r"υμενικ[^ ]* υπερτροφ", r"αρθρικου υμεν",
        r"παχυνση[^.]{0,20}υμεν", r"υμενα",
        r"синовит", r"синовиал[^ ]* (задебел|пролифер)",
        r"\bpannus\b", r"\bhoffit", r"sinovyal\w* (kalinlas|proliferas)",
        r"sinovyal hipertrof", r"\bartrit\b", r"\barthritis\b",
    ),
    "Baker's": _rx(
        r"baker", r"popliteal cyst", r"quiste popliteo", r"quistes popliteos",
        r"kyste poplite", r"popliteale? cyst", r"poplitealzyste", r"bakerzyste",
        r"popliteal kist", r"\bbakerova\b", r"poplitealn[^ ]* cist", r"popliteal\w* cist",
        r"κυστη baker", r"πολυχωρη συνοβιακη κυστη", r"κυστη του baker",
        r"συνοβιακη κυστη", r"κυστη τυπου baker",
        r"киста на бейкър", r"бейкърова киста", r"поплитеална киста", r"бекеров",
        r"gastrocnemio ?semimembranos", r"gastrocnemius semimembranosus burs",
    ),
    "Contusion": _rx(
        r"\bcontusion", r"bone bruise", r"bone marrow (o?edema|contusion)",
        r"marrow o?edema", r"\bkontuz", r"medular bone o?edema", r"osseous contusion",
        r"contusion osea", r"edema oseo", r"edema de medula osea", r"contusiones oseas",
        r"oedeme osseux", r"contusion osseuse",
        r"botcontusie", r"botoedeem", r"beenmergoedeem", r"botmergoedeem",
        r"knochenmarkodem", r"knochenodem", r"knochenmarksodem", r"kontusion",
        r"kemik kontuzyonu", r"kemik iligi odemi", r"kemik odemi", r"kemik iliginde odem",
        r"kontuzyonel kemik", r"kemik iligi odemleri",
        r"kostani edem", r"edem kosti", r"kontuzij", r"kostane srzi[^.]{0,20}edem",
        r"οστεομυελικ[^ ]* οιδημα", r"οστικο οιδημα", r"μυελικο οιδημα", r"οστικο μωλωπ",
        r"костномозъчен едем", r"костен едем", r"контузионен", r"костно мозъчен едем",
    ),
    "Fracture": _rx(
        r"\bfractur", r"\bfract\b",
        r"\bfractura", r"\bfracturas\b",
        r"\bfractuur", r"\bbreuk\b",
        r"\bfraktur", r"\bbruch\b",
        r"\bkirik\b", r"\bkirigi\b", r"\bkiri[kg]\w*",
        r"\bprijelom", r"impresijsk[^ ]* fraktur", r"impaktcij",
        r"καταγμα", r"καταγματ",
        r"фрактур", r"счупван", r"фисур",
        r"insufficiency fracture", r"stress fracture", r"avulsion fracture",
        r"subchondral fracture", r"subkondral kiri", r"impaction (fracture|injury)",
        r"osteochondral (fracture|impaction)", r"\bsegond\b", r"impactiefractuur",
        r"subchondrale impression", r"subchondraler? impress",
    ),
}

DECOY = {
    "Fracture": _rx(r"microfractur", r"\bfracture (risk|prophyla)"),
    "Baker's": _rx(r"meniscal cyst", r"quiste meniscal", r"parameniscal"),
}

PAIRED = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"}
OA_TARGETS = ["Medial OA", "Lateral OA", "PF OA"]

# --------------------------------------------------------------------------- #
# stems, for morphology the phrase lexicons cannot reach
# --------------------------------------------------------------------------- #
# A report that clears or tears both menisci in one breath - "normaal voorkomen
# menisci", "Normal medial and lateral menisci" - names neither side, so a
# side-qualified lexicon leaves both targets silent. The cruciates and collaterals
# already had their plural forms; the menisci were simply missed.
#
# It is only consulted when the clause names no side at all. Otherwise "tear of the
# medial meniscus, menisci otherwise intact" would fire the plural for the lateral side
# off a clause that is about the medial one.
PLURAL_MENISCI = _rx(
    r"\bmenisci\b", r"\bmeniscos\b", r"\bmenisques\b", r"\bmenisken\b",
    r"\bmeniskusi\b", r"\bmenisk\w*ler\b", r"\bμηνισκοι\b", r"\bμηνισκων\b",
    r"\bменискуси\b", r"\bменискусите\b", r"\bmenisci\w*\b",
)
ANY_SIDE = _rx(SIDE_MEDIAL.pattern, SIDE_LATERAL.pattern)

STEM_MENISCUS = _rx(r"menisc\w*", r"menisk\w*", r"μηνισκ\w*", r"мениск\w*")
STEM_CRUCIATE = _rx(r"cruciate", r"cruzado", r"croise", r"kruisband", r"kreuzband",
                    r"capraz bag\w*", r"krizn\w*", r"χιαστ\w*", r"кръстн\w*",
                    r"\bacl\b", r"\blca\b", r"\bvkb\b", r"\bocb\b", r"\bacb\b")
STEM_COLLATERAL = _rx(r"collateral\w*", r"colateral\w*", r"kollateral\w*",
                      r"collaterale\w*", r"kolateraln\w*", r"yan bag\w*",
                      r"πλαγι\w*", r"колатерал\w*", r"странич\w*",
                      r"innenband\w*", r"binnenband\w*", r"\bmcl\b", r"\blcm\b",
                      r"\biyb\b")

STEM_FRACTURE = _rx(r"fractur\w*", r"fraktur\w*", r"fractuur\w*", r"\bfract\b",
                    r"kiri[kgğ]\w*", r"prijelom\w*", r"lom kosti", r"\bbreuk\w*",
                    r"\bbruch\w*", r"καταγμα\w*", r"καταγματ\w*", r"фрактур\w*",
                    r"счупван\w*", r"fisur\w* (osea|oseas|kost)", r"fissur\w* kost")

# Decoys that steal a cruciate / collateral stem match for the wrong ligament.
POSTERIOR_ONLY = _rx(r"\bpcl\b", r"\blcp\b", r"\bhkb\b", r"\bacb\b",
                     r"posterior cruciate", r"cruzado posterior", r"croise posterieur",
                     r"achterste kruisband", r"hinteres kreuzband", r"arka capraz",
                     r"straznji krizn", r"οπισθι[οα]\w* χιαστ", r"задна кръстн",
                     r"задната кръстн")
LATERAL_COLL_ONLY = _rx(r"\blcl\b", r"\bfcl\b", r"lateral collateral",
                        r"fibular collateral", r"colateral lateral", r"colateral externo",
                        r"buitenband", r"aussenband", r"dis yan bag",
                        r"lateralni kolateraln", r"εξω πλαγι", r"латерален колатерал")

# ==== CELL ====

def _near(clause: str, stem_rx: re.Pattern, qual_rx: re.Pattern, window: int = 55):
    for m in stem_rx.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        if qual_rx.search(clause[lo:hi]):
            return True
    return False


STEM_RULES = {
    "ACL": (STEM_CRUCIATE, SIDE_ANTERIOR),
    "MCL": (STEM_COLLATERAL, SIDE_MEDIAL),
    "Medial Meniscus": (STEM_MENISCUS, SIDE_MEDIAL),
    "Lateral Meniscus": (STEM_MENISCUS, SIDE_LATERAL),
}


class _Matcher:
    def __init__(self, phrase_rx, stem=None, side=None, window=55):
        self.phrase_rx = phrase_rx
        self.stem = stem
        self.side = side
        self.window = window

    def search(self, clause):
        m = self.phrase_rx.search(clause)
        if m is not None:
            return m
        if self.stem is not None and _near(clause, self.stem, self.side, self.window):
            return self.stem.search(clause)
        return None


ANAT_MATCH = {t: _Matcher(ANAT[t], *STEM_RULES[t]) for t in PAIRED}
DIRECT_MATCH = {
    t: _Matcher(_rx(rx.pattern, STEM_FRACTURE.pattern) if t == "Fracture" else rx)
    for t, rx in DIRECT.items()
}

# --------------------------------------------------------------------------- #
# severity
# --------------------------------------------------------------------------- #
SEV_LOW = _rx(
    r"\bsmall\b", r"\bminimal\b", r"\btrace\b", r"\bmild\b", r"\bslight\b",
    r"\btiny\b", r"\bscant\b", r"\bdiscrete\b", r"\blow ?grade\b", r"\bincipient\b",
    r"\bleve\b", r"\bminim", r"\bpeque", r"\bfina\b", r"\bfino\b", r"\bligero\b", r"\bescaso\b", r"\bdiscreto\b",
    r"\bhafif\b", r"\baz miktarda\b", r"\bsilik\b",
    r"\bmanj\w*", r"\bblago\b", r"\bdiskretn", r"\bmalo\b", r"\bpocetn",
    r"\bgering", r"\bdiskret", r"\bkleine?r?\b", r"\bwenig\b", r"\bzarte?\b",
    r"\bbeperkte?\b", r"\bgeringe\b", r"\bweinig\b", r"\blichte?\b", r"\blicht\b",
    r"\bηπι", r"\bμικρ", r"\bελαχιστ", r"\bαρχομεν",
    r"\bминимал", r"\bлек", r"\bмалк", r"\bнеголям",
)

SEV_HIGH = _rx(
    r"\blarge\b", r"\bmarked\b", r"\bmassive\b", r"\bsevere\b", r"\bextensive\b",
    r"\bmoderate\b", r"\bgross\b", r"\bsignificant\b", r"\babundant\b", r"\btense\b",
    r"\bcomplete\b", r"\bfull ?thickness\b", r"\bhigh ?grade\b", r"\badvanced\b",
    r"\bmoderad", r"\bimportante\b", r"\bsevera?\b", r"\bmarcad", r"\bcuantios",
    r"\bespesor total\b", r"\bcompleta?\b",
    r"\bbelirgin\b", r"\byaygin\b", r"\bileri\b", r"\bciddi\b", r"\bbol\b", r"\bkomplet",
    r"\bopsezan\b", r"\bveliki\b", r"\bizrazit", r"\bznacajn", r"\bumjeren",
    r"\buznapredoval", r"\bpotpun", r"\bkompleksn",
    r"\bausgepragt", r"\bdeutlich", r"\bmassiv", r"\bmassig", r"\bgross",
    r"\buitgebreid", r"\bgevorderd", r"\bveel\b", r"\bmatige?\b", r"\bvolledig",
    r"\bμετρι", r"\bμεγαλ", r"\bεκτεταμεν", r"\bευμεγεθ", r"\bσοβαρ", r"\bπληρη",
    r"\bголям", r"\bизразен", r"\bзначим", r"\bумерен", r"\bобилен", r"\bпълн",
)

GRADE_HIGH = re.compile(r"grade?[ao]?\s*(3|4|iii|iv)\b|icrs grade (iii|iv|3|4)|"
                        r"stupnja iv|stupnja iii|\bgrado (3|4)\b|\bgrad (3|4)\b|"
                        r"\bgrade (3|4)\b")

DEGENERATIVE_MARROW = _rx(
    r"subchondral", r"subcondral", r"subkondral", r"supkondraln", r"subchondraln",
    r"υποχονδρι", r"υπαρθρικ", r"субхондрал", r"subchondrale?", r"subartikuler",
    r"\bcyst", r"\bquist", r"\bzyste\b", r"\bcistic", r"reactive", r"reactivo",
    r"degenerative", r"degenerativ", r"reaktiv", r"\bcisti\b",
)

TRAUMA = _rx(
    r"\bbruise\b", r"\bcontusion", r"\bkontuz", r"\btrauma", r"\bimpaction\b",
    r"\bpivot shift\b", r"\bkissing\b", r"\bacute\b", r"\bagudo\b", r"\bakut",
    r"\bpivot kaymasi\b", r"\bcontusion osseuse\b", r"\bbone bruise\b",
    r"\bbotcontusie\b", r"\bконтузион", r"\bμωλωπ", r"\bkontuzij", r"\bimpaktcij",
    r"\bimpakcij", r"\bfall\b", r"\binjury\b", r"\bimpression\b",
)

# Effusion-adjacent inflammatory signs, used only when a report never names synovitis.
SYNOVIAL_PROXY = _rx(
    r"bursit", r"burzit", r"\bbursa\b[^.]{0,30}(fluid|distend|sivi|tekucin|opzetting)",
    r"suprapatellar (bursitis|effusion|recess)", r"suprapatellar bursa",
    r"suprapatellar bursada", r"suprapatelarno", r"suprapatellaire recessus",
    r"hoffa", r"hoffit", r"plica", r"plika", r"πλικα", r"fat pad[^.]{0,20}(edema|oedema)",
    r"kapsul", r"capsul", r"καψ", r"капсул", r"\bpannus\b", r"\bsinov", r"\bsynov",
)


def _polarity(clause: str, span=None) -> str:
    """positive / negative / uncertain for a term matched at `span` in this clause."""
    if UNCERTAIN.search(clause):
        return "uncertain"
    if span is None or not FEATURES["directional_negation"]:
        if NEGATION.search(clause):
            return "negative"
    elif _negated(clause, span[0], span[1]):
        return "negative"
    if NORMALITY.search(clause):
        # `meniscus normal` denies; `normal alignment ... full thickness tear` does not.
        if TEAR.search(clause) or GRADE_HIGH.search(clause):
            return "positive"
        return "negative"
    return "positive"


def _severity(clause: str) -> float:
    """How emphatic a clause is about the finding it asserts.

    A numeric grade is deliberately not read here. It belongs to whichever structure the
    grade was written for, and a clause that grades the cartilage while mentioning the
    effusion in passing - `PF arthrotic change with reduced joint space, a smaller
    effusion and grade IV chondromalacia` - would otherwise report a severe effusion.
    """
    high = SEV_HIGH.search(clause) is not None
    low = SEV_LOW.search(clause) is not None
    if high and not low:
        return 1.0
    if low and not high:
        return 0.45
    if high and low:
        return 0.8
    return 0.75


def _grade(n_pos, n_neg, n_unc, best):
    """Map counted evidence onto a score in (0, 1) and a confidence."""
    if n_pos or n_unc:
        score = min(0.97, 0.50 + 0.45 * best + 0.015 * min(n_pos, 3))
        conf = min(1.0, 0.55 + 0.15 * n_pos)
    elif n_neg:
        score = max(0.04, 0.20 - 0.04 * n_neg)
        conf = min(0.9, 0.45 + 0.12 * n_neg)
    else:
        score, conf = 0.28, 0.05
    return score, conf


def _paired_weight(clause: str, meniscus: bool) -> float:
    """How strongly one clause asserts damage to a meniscus or a cruciate/collateral.

    Ordered, not calibrated. What has to hold is that a tear outranks a graded lesion,
    that the grade is read on the right scale for the structure, and that intrasubstance
    degeneration lands below both - the annotator marks a torn meniscus and leaves a
    degenerate one, and a lexicon that scores the two alike throws that ordering away.
    """
    g = _grade_of(clause) if FEATURES["graded_pathology"] else None
    tear = TEAR.search(clause) is not None
    if meniscus:
        if tear:
            base = 1.0
        elif g is not None:
            base = 0.95 if g >= 3 else 0.30
        elif DEGEN.search(clause):
            base = 0.35
        else:
            base = 0.45
    else:
        if tear:
            base = 1.0
        elif g is not None:
            base = 0.85 if g >= 2 else 0.30
        elif DEGEN.search(clause):
            base = 0.40
        else:
            base = 0.55
    if SEV_HIGH.search(clause) and not SEV_LOW.search(clause):
        base = min(1.0, base * 1.2)
    elif SEV_LOW.search(clause) and not SEV_HIGH.search(clause):
        base *= 0.7
    return base


def _score_paired(cls, tgt):
    """Evidence for one of the four side-specific ligament / meniscus targets."""
    anat_rx = ANAT_MATCH[tgt]
    path_rx = _rx(TEAR.pattern, DEGEN.pattern, INJURY.pattern)
    meniscus = "Meniscus" in tgt
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        hit = anat_rx.search(c)
        if hit is None and meniscus and PLURAL_MENISCI.search(c) \
                and not ANY_SIDE.search(c):
            hit = PLURAL_MENISCI.search(c)
        if hit is None:
            continue
        # Anchor the negation test on the pathology word, not on the anatomy word.
        # `Medial meniscus is not torn` negates the tear, and the negator stands after
        # the noun: scoping from the noun would read the sentence as an assertion.
        pm = path_rx.search(c)
        if pm is None and _grade_of(c) is None:
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c)
                                           and not NEGATION.search(c)):
                n_neg += 1
            continue
        span = (pm.start(), pm.end()) if pm is not None else None
        pol = _polarity(c, span)
        if pol == "positive":
            n_pos += 1
            best = max(best, _paired_weight(c, meniscus))
        elif pol == "negative":
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.45 * _paired_weight(c, meniscus))
    s, cf = _grade(n_pos, n_neg, n_unc, best)
    return s, cf, n_pos, n_neg


def _score_clauses(cls, anat_rx, path_rx=None, decoy_rx=None, context_penalty=None,
                   context_bonus=None):
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        m = anat_rx.search(c)
        if not m:
            continue
        if decoy_rx is not None and decoy_rx.search(c):
            continue
        if path_rx is not None and not path_rx.search(c):
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c)
                                           and not NEGATION.search(c)):
                n_neg += 1
            continue
        pol = _polarity(c, (m.start(), m.end()))
        if pol == "positive":
            n_pos += 1
            w = _severity(c)
            if context_penalty is not None and context_penalty.search(c):
                w *= 0.45
            if context_bonus is not None and context_bonus.search(c):
                w = min(1.0, w * 1.35)
            best = max(best, w)
        elif pol == "negative":
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.30)
    s, c = _grade(n_pos, n_neg, n_unc, best)
    return s, c, n_pos, n_neg


def _score_oa(cls):
    """Osteoarthritis, scoped to the three compartments.

    A cartilage statement is attributed by what else the clause names rather than by a
    compartment phrase: `medial` next to a tibiofemoral structure sends it to the medial
    compartment, `patella` or `trochlea` to the patellofemoral one, and a statement that
    names neither is a whole-joint assertion that counts for all three at a discount.
    That last case is what the public lexicon leaves silent, and it is the majority of
    the corpus - reports say `tricompartmental chondrosis`, not `chondrosis of the
    medial femorotibial compartment`.
    """
    acc = {t: {"pos": 0, "neg": 0, "unc": 0, "best": 0.0} for t in OA_TARGETS}
    g_pos, g_neg, g_best = 0, 0, 0.0

    for c in cls:
        m = OA_EVIDENCE.search(c)
        if not m:
            continue
        pol = _polarity(c, (m.start(), m.end()))
        sev = _severity(c)
        tf_med = _near(c, TF_SITE, SIDE_MEDIAL, 45)
        tf_lat = _near(c, TF_SITE, SIDE_LATERAL, 45)
        pf = PF_SITE.search(c) is not None
        hits = []
        if tf_med:
            hits.append("Medial OA")
        if tf_lat:
            hits.append("Lateral OA")
        if pf:
            hits.append("PF OA")

        if not hits:
            # No compartment named. Whole-joint statements ("tricompartmental
            # chondrosis", "gonarthrose") speak for all three; an unlocalised cartilage
            # remark is weaker evidence but still evidence, so it is carried at a
            # discount rather than dropped.
            if pol == "positive":
                g_pos += 1
                g_best = max(g_best, sev if GLOBAL_OA.search(c) else sev * 0.7)
            elif pol == "negative":
                g_neg += 1
            continue

        for t in hits:
            if pol == "positive":
                acc[t]["pos"] += 1
                acc[t]["best"] = max(acc[t]["best"], sev)
            elif pol == "negative":
                acc[t]["neg"] += 1
            else:
                acc[t]["unc"] += 1
                acc[t]["best"] = max(acc[t]["best"], 0.30)

    out = {}
    for t in OA_TARGETS:
        a = acc[t]
        pos, neg, unc, best = a["pos"], a["neg"], a["unc"], a["best"]
        if not (pos or unc) and g_pos and FEATURES["oa_inherit"]:
            # Inherit the whole-joint statement, unless this compartment was separately
            # and explicitly cleared.
            if neg:
                score, conf = _grade(0, neg, 0, 0.0)
                score = max(score, 0.35)
                conf *= 0.7
            else:
                score, conf = _grade(g_pos, 0, 0, g_best * 0.92)
                conf *= 0.75
        else:
            score, conf = _grade(pos, neg + g_neg, unc, best)
        out[t] = (score, conf, pos, neg)
    return out


def extract_raw(report: str) -> dict:
    """Twelve (score, confidence) pairs, plus the counts the coverage gauge reads."""
    cls = clauses(report)
    out = {}

    for tgt in PAIRED:
        s, c, npos, nneg = _score_paired(cls, tgt)
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    for tgt, (s, c, npos, nneg) in _score_oa(cls).items():
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    for tgt in ("Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"):
        if tgt == "Contusion":
            s, c, npos, nneg = _score_clauses(
                cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt),
                context_penalty=DEGENERATIVE_MARROW, context_bonus=TRAUMA)
        else:
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None,
                                              DECOY.get(tgt))
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    # --- synovitis backoff --------------------------------------------------- #
    # Eighty-eight per cent of reports never write the word, and the annotator marks it
    # on nearly half the studies: the label cannot be read off the term alone. What the
    # report does say is whether the joint is wet and irritated - an effusion, a
    # distended bursa, an inflamed fat pad - and that ordering is the only signal
    # available on the silent majority. It enters at low confidence, so it shapes the
    # ranking without asserting a finding.
    if (FEATURES["synovitis_backoff"] and out["Synovitis__npos"] == 0
            and out["Synovitis__nneg"] == 0):
        proxy = sum(1 for c in cls if SYNOVIAL_PROXY.search(c)
                    and _polarity(c) == "positive")
        eff = out["Effusion"]
        prior = 0.30 + 0.30 * max(0.0, (eff - 0.5) / 0.45) + 0.06 * min(proxy, 3)
        out["Synovitis"] = min(0.72, prior)
        out["Synovitis__conf"] = 0.18

    return out


# --------------------------------------------------------------------------- #
# adapter + offline gauges (project side, not part of the ported labeler)
# --------------------------------------------------------------------------- #
def extract(report: str):
    """Return (grades dict[label]->[0,1], conf dict[label]->(0,1])."""
    raw = extract_raw(report)
    grades = {t: float(raw[t]) for t in TARGETS}
    conf = {t: float(raw[t + "__conf"]) for t in TARGETS}
    return grades, conf


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    n_pos = yt.sum()
    n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_r = ranks[yt == 1].sum()
    return (sum_r - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def run_gauges(train_csv: str):
    """GAUGE 2 = silence rate over the corpus; GAUGE 1 = agreement AUC on the 58
    annotated studies. Silent here means the labeler saw neither a positive nor a
    negative assertion for that target."""
    df = pd.read_csv(train_csv)
    rep = df["Report"].fillna("").tolist()
    raws = [extract_raw(r) for r in rep]
    n = len(rep)

    print(f"GAUGE 2 - silence rate over {n} reports (low = good coverage):")
    for l in LABELS:
        silent = sum(1 for d in raws
                     if d[l + "__npos"] == 0 and d[l + "__nneg"] == 0)
        print(f"  {l:18s} silent={silent * 100 / n:5.1f}%")

    has_lab = df[LABELS].notna().any(axis=1)
    ann = df[has_lab].reset_index(drop=True)
    ann_raw = [extract_raw(r) for r in ann["Report"].fillna("")]
    print(f"\nGAUGE 1 - agreement AUC on {len(ann)} annotated studies:")
    aucs = []
    for l in LABELS:
        y = ann[l].astype(float).values
        if len(np.unique(y[np.isfinite(y)])) < 2:
            print(f"  {l:18s} only one class in annotations")
            continue
        scores = np.array([d[l] for d in ann_raw])
        auc = roc_auc(y, scores)
        if np.isnan(auc):
            print(f"  {l:18s} degenerate")
        else:
            print(f"  {l:18s} AUC={auc:.3f}")
            aucs.append(auc)
    if aucs:
        print(f"  macro-AUC={float(np.mean(aucs)):.3f}")


def self_test():
    """Semantic regression cases carried over from the v3 extractor, so the swap
    cannot silently break polarity, language coverage or severity ordering."""
    def _s(text):
        return extract(text)[0]

    cases = {
        "en_tear": "Complete tear of the anterior cruciate ligament.",
        "en_intact": "The anterior cruciate ligament is intact.",
        "es_group": "Ligamentos cruzados y colaterales dentro de limites normales.",
        "fr_heading": "Fractures :\nAucune.",
        "tr_effusion": "Eklem icerisinde yaygin sivi artisi mevcuttur.",
        "el_meniscus": "Ρηξη του εσω μηνισκου.",
        "eff_large": "Large joint effusion.",
        "eff_trace": "Trace joint effusion.",
        "eff_silent": "The menisci are intact.",
        "global_oa": "Tricompartmental osteoarthritis.",
        "degen_marrow": "Subchondral bone marrow oedema beneath a full thickness cartilage defect.",
        "true_bruise": "Bone bruise of the lateral femoral condyle after acute trauma.",
        "dir_neg": ("Subchondral insufficiency fracture at the medial tibial "
                    "plateau without articular surface collapse."),
    }
    R = {k: _s(v) for k, v in cases.items()}
    assert R["en_tear"]["ACL"] > 0.5, R["en_tear"]["ACL"]
    assert R["en_intact"]["ACL"] < 0.3, R["en_intact"]["ACL"]
    assert R["es_group"]["ACL"] < 0.3 and R["es_group"]["MCL"] < 0.3, (
        R["es_group"]["ACL"], R["es_group"]["MCL"])
    assert R["fr_heading"]["Fracture"] < 0.3, R["fr_heading"]["Fracture"]
    assert R["tr_effusion"]["Effusion"] > 0.5, R["tr_effusion"]["Effusion"]
    assert R["el_meniscus"]["Medial Meniscus"] > 0.5, R["el_meniscus"]["Medial Meniscus"]
    assert (R["eff_large"]["Effusion"] > R["eff_trace"]["Effusion"]
            > R["eff_silent"]["Effusion"]), (
        R["eff_large"]["Effusion"], R["eff_trace"]["Effusion"],
        R["eff_silent"]["Effusion"])
    assert min(R["global_oa"][t] for t in ("Medial OA", "Lateral OA", "PF OA")) > 0.5
    assert R["degen_marrow"]["Contusion"] < R["true_bruise"]["Contusion"], (
        R["degen_marrow"]["Contusion"], R["true_bruise"]["Contusion"])
    # the case the ported labeler exists for: a negator that governs forward only
    assert R["dir_neg"]["Fracture"] > 0.5, R["dir_neg"]["Fracture"]

    repairs = {
        ("Fracture", "Fractures :\nAucune."): "lt",
        ("Fracture", "No fracture is identified."): "lt",
        ("Fracture", "Fractures :\nFracture non deplacee du plateau tibial."): "gt",
        ("Effusion", "Effusion:\nNone."): "lt",
        ("Effusion", "Joint effusion: nil."): "lt",
        ("Effusion", "Moderate joint effusion."): "gt",
        ("Fracture", "Fracturas :\nNinguna."): "lt",
        ("Baker's", "Quiste de Baker :\nPresente."): "gt",
    }
    for (tgt, text), want in repairs.items():
        v = extract(text)[0][tgt]
        ok = (v < 0.3) if want == "lt" else (v > 0.5)
        assert ok, f"{want}: {tgt} scored {v:.2f} on {text!r}"
    print("extractor self-test passed")


# ===================== dataset.py =====================
"""PyTorch Dataset for RSNA Knee — slot-based, imaging-only at inference.

Design notes (incorporating top public-notebook lessons, see LOG.md):
- A Study is the prediction unit. Its MRI content is mapped onto N_SLOTS fixed
  slots, one per (plane, fluid_sensitive, fat_suppression) combination defined in
  config.SLOTS. A slot may be empty -> a presence MASK (1 present / 0 missing)
  is returned so the attention head can ignore missing slots.
- Laterality normalization (right->left convention) is read from DICOM headers
  where present; it degrades to a no-op if headers are stripped (local sample).
- The radiology report is TRAIN-ONLY (test.csv has no Report column). We grade
  it offline into 12 soft targets + 12 per-label confidence weights. Where a gold
  label exists it overrides the soft target and gets GOLD_WEIGHT; where it does
  not, the soft target + confidence weight drive weak supervision.
"""
import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
# torch is imported lazily inside __getitem__/collate (so this module is importable
# even where torch isn't installed, e.g. for offline report-target inspection).
try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - only hit in torch-less environments
    class Dataset:  # placeholder base so the module still imports
        pass

from config import (LABELS, DATA_ROOT, RAW, IMG_SIZE, SLICES_PER_SLOT,
                    N_SLOTS, SLOTS, USE_REPORT_TARGETS, GROUP, N_GROUPS_MAX,
                    CACHE_BUDGET_GB, GOLD_WEIGHT, LAT_MIN_AGREEMENT,
                    DERIVED_W_FLOOR, USE_RECOVERED_FLAGS, SLOT_FALLBACK,
                    SOFT_LABEL_PARQUETS, TEACHER_PARQUET, TEACHER_LAMBDA)
import dicom_utils as du
import report_extractor as _rext


def _series_manifest(split: str) -> pd.DataFrame:
    """Load {split}_series.csv giving Study->Series metadata."""
    path = os.path.join(RAW, f"{split}_series.csv")
    if not os.path.exists(path) and DATA_ROOT:
        path = os.path.join(DATA_ROOT, f"{split}_series.csv")
    return pd.read_csv(path)


def build_study_index(split: str):
    """Return (labels_df, study_to_series) for the given split.

    On Kaggle the labeled CSVs live under DATA_ROOT (only mounted inside the
    Notebook); locally they live under RAW. Prefer DATA_ROOT when present so the
    same code path works in both environments.
    """
    labels_path = os.path.join(RAW, f"{split}.csv")
    if not os.path.exists(labels_path) and DATA_ROOT:
        labels_path = os.path.join(DATA_ROOT, f"{split}.csv")
    labels_df = pd.read_csv(labels_path)
    man = _series_manifest(split)
    study_to_series = defaultdict(list)
    for _, r in man.iterrows():
        study_to_series[r["StudyInstanceUID"]].append(
            (r["SeriesInstanceUID"], r["Anatomical_Plane"],
             int(r["Fluid_Sensitive"]), int(r["Fat_Suppression"])))
    return labels_df, study_to_series


def _match_slot(plane: str, fl: int, fat: int) -> int:
    """Index of the slot matching (plane, fluid_sensitive, fat_suppression), -1 if none."""
    for i, (p, f, fa) in enumerate(SLOTS):
        if p == plane and f == fl and fa == fat:
            return i
    return -1


class RSNADataset(Dataset):
    def __init__(self, split: str = "train", series_root: str | None = None,
                 use_report: bool = USE_REPORT_TARGETS):
        self.split = split
        self.series_root = series_root or (
            os.path.join(DATA_ROOT, f"{split}_series") if DATA_ROOT else None)
        self.use_report = use_report and split == "train"

        self._rec = None          # lazily built header-recovered slot flags
        self.labels_df, self.study_to_series = build_study_index(split)
        self.uids = self.labels_df["StudyInstanceUID"].tolist()
        # test.csv has NO Report column (competition design) — only train does.
        if "Report" in self.labels_df.columns:
            self.reports = self.labels_df["Report"].fillna("").tolist()
        else:
            self.reports = [""] * len(self.uids)

        if split == "train":
            # gold labels with NaN where missing (NaN = unlabeled study)
            self.y_raw = self.labels_df[LABELS].to_numpy(dtype=np.float32)
            self.has_gold = np.isfinite(self.y_raw).any(axis=1)
            if self.use_report:
                self._build_report_targets()
            else:
                # pure hard-label mode (only studies with labels are usable)
                self.targets = np.nan_to_num(self.y_raw, nan=0.0).astype(np.float32)
                self.weights = np.isfinite(self.y_raw).astype(np.float32)
        else:
            self.y_raw = None
            self.targets = None
            self.weights = None

    def _find_label_parquet(self, name):
        """Locate a soft-label parquet locally or among the Kaggle dataset mounts."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cands = [os.path.join(here, "data", "labels", name), name]
        root = "/kaggle/input"
        if os.path.isdir(root):
            for dirpath, dirnames, files in os.walk(root):
                if dirpath[len(root):].count(os.sep) > 4 or "competition" in dirpath.lower():
                    dirnames[:] = []
                    continue
                if name in files:
                    cands.append(os.path.join(dirpath, name))
        for c in cands:
            if os.path.exists(c):
                return c
        return None

    def _build_report_targets(self):
        """Grade every report -> soft target + confidence weight; gold overrides.

        With SOFT_LABEL_PARQUETS configured, each file's per-finding (p, conf) is
        averaged with the regex extractor's, source by source. Averaging
        PROBABILITIES rather than ranks: rank-averaging pins every label's mean
        to 0.5, which is harmless for AUC but wrong for a BCE target -- and it
        also scored lower on the 58 gold studies (0.885 vs 0.889).
        """
        import pandas as _pd

        extra = []
        for name in SOFT_LABEL_PARQUETS:
            path = self._find_label_parquet(name)
            if path is None:
                print(f"soft labels: {name} NOT FOUND -- falling back to the "
                      f"extractor alone for this source", flush=True)
                continue
            df = _pd.read_parquet(path).set_index("StudyInstanceUID")
            extra.append((name, df))
            print(f"soft labels: {name} ({len(df)} studies) from {path}", flush=True)

        T = np.zeros((len(self), len(LABELS)), np.float32)
        W = np.zeros((len(self), len(LABELS)), np.float32)
        n_missing = 0
        for i, rep in enumerate(self.reports):
            grades, conf = _rext.extract(rep)
            uid = str(self.uids[i])
            for j, lab in enumerate(LABELS):
                ps, cs = [grades[lab]], [conf[lab]]
                for _name, df in extra:
                    if uid in df.index:
                        ps.append(float(df.at[uid, lab]))
                        cs.append(float(df.at[uid, lab + "__conf"]))
                T[i, j] = float(np.mean(ps))
                W[i, j] = DERIVED_W_FLOOR + (1.0 - DERIVED_W_FLOOR) * float(np.mean(cs))
            if extra and not any(uid in df.index for _n, df in extra):
                n_missing += 1
        if extra:
            print(f"soft labels: {len(self) - n_missing}/{len(self)} studies got "
                  f"LLM labels ({n_missing} extractor-only)", flush=True)
        if TEACHER_PARQUET and TEACHER_LAMBDA > 0:
            path = self._find_label_parquet(TEACHER_PARQUET)
            if path is None:
                raise FileNotFoundError(f"teacher labels {TEACHER_PARQUET} not found")
            tdf = _pd.read_parquet(path).set_index("StudyInstanceUID")
            missing = [u for u in map(str, self.uids) if u not in tdf.index]
            if missing:
                raise ValueError(f"teacher labels miss {len(missing)} studies")
            P = tdf.loc[[str(u) for u in self.uids], list(LABELS)].to_numpy(np.float32)
            T = (1.0 - TEACHER_LAMBDA) * T + TEACHER_LAMBDA * P
            print(f"teacher labels: {TEACHER_PARQUET} lambda={TEACHER_LAMBDA} "
                  f"from {path}", flush=True)
        # override with gold where present (annotated studies are up-weighted)
        for i in range(len(self)):
            yrow = self.y_raw[i]
            for j in range(len(LABELS)):
                if np.isfinite(yrow[j]):
                    T[i, j] = yrow[j]
                    W[i, j] = float(GOLD_WEIGHT)
        self.targets = T
        self.weights = W

    def __len__(self):
        return len(self.uids)

    def _best_series_folder(self, uid: str, sids):
        return max(
            (os.path.join(self.series_root, uid, s) for s in sids),
            key=lambda p: len(glob.glob(os.path.join(p, "*.dcm"))))

    def _recovered_flags(self):
        """series UID -> (fluid, fatsat) read off the DICOM header, once per split.

        The competition CSV's two flags are perfectly correlated in this corpus, so
        they carry one bit between them and a six-slot scheme cannot be built on
        them (see LOG.md v13/v14). The header carries the weighting and the fat
        suppression independently. A series whose header decides nothing is absent
        from the map and falls back to its CSV flags.
        """
        if self._rec is not None:
            return self._rec
        self._rec = {}
        if not USE_RECOVERED_FLAGS or self.series_root is None:
            return self._rec
        from concurrent.futures import ThreadPoolExecutor
        import time
        t0 = time.time()
        jobs = []
        for uid in self.uids:
            for sid, _p, _f, _a in self.study_to_series.get(uid, []):
                jobs.append((uid, sid))

        def _one(j):
            uid, sid = j
            return sid, du.recover_sequence(
                du.read_series_meta(os.path.join(self.series_root, uid, sid)))

        n_ok = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            for sid, rec in ex.map(_one, jobs):
                if rec is not None:
                    self._rec[sid] = (int(rec[0]), int(rec[1]))
                    n_ok += 1
        print(f"recovered sequence flags: {n_ok}/{len(jobs)} series "
              f"({n_ok / max(len(jobs), 1) * 100:.1f}%) in "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        return self._rec

    def _slots_for_study(self, uid: str):
        """Map slot index -> representative series folder for a study."""
        out = {}
        if self.series_root is None:
            return out
        rec = self._recovered_flags()
        series = self.study_to_series.get(uid, [])
        # (plane, flags) per series, preferring the header-recovered flags.
        tagged = [(sid, plane, rec.get(sid, (int(fl), int(fat))))
                  for sid, plane, fl, fat in series]
        for si, (p, f, a) in enumerate(SLOTS):
            cand = [sid for sid, plane, k in tagged if plane == p and k == (f, a)]
            if not cand and SLOT_FALLBACK and f == 0:
                # Structural slots are the scarcest; before leaving one empty take any
                # non-fat-suppressed series in the plane. Exact matches still win.
                cand = [sid for sid, plane, k in tagged
                        if plane == p and k[1] == 0]
            if cand:
                out[si] = self._best_series_folder(uid, cand)
        return out

    def _load_study_slots(self, uid: str):
        """Return (imgs [N_SLOTS, S, 1, H, W], mask [N_SLOTS]) for a study."""
        imgs = np.zeros(
            (N_SLOTS, SLICES_PER_SLOT, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        mask = np.zeros(N_SLOTS, dtype=np.float32)
        for si, folder in self._slots_for_study(uid).items():
            plane, fl, fat = SLOTS[si]
            lat = du.read_laterality(folder)
            imgs[si] = du.load_series(folder, plane, lat, SLICES_PER_SLOT, IMG_SIZE)
            mask[si] = 1.0
        return imgs, mask

    def __getitem__(self, idx):
        import torch
        uid = self.uids[idx]
        imgs, mask = self._load_study_slots(uid)
        if self.split == "train":
            return (torch.from_numpy(imgs), torch.from_numpy(mask),
                    torch.from_numpy(self.targets[idx]),
                    torch.from_numpy(self.weights[idx]))
        return torch.from_numpy(imgs), torch.from_numpy(mask), uid


def collate_train(batch):
    import torch
    imgs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    targets = torch.stack([b[2] for b in batch])
    weights = torch.stack([b[3] for b in batch])
    return imgs, masks, targets, weights


def collate_test(batch):
    import torch
    imgs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    uids = [b[2] for b in batch]
    return imgs, masks, uids


def run_seq_diag(split: str = "train", limit: int | None = None,
                 n_workers: int = 16):
    """Read one header per series and report what the sequence recovery yields.

    A diagnostic, not a training step. The competition CSV's Fluid_Sensitive and
    Fat_Suppression are perfectly correlated here, so a six-slot scheme built on
    them has a slot that can never fill (see LOG.md v13). Recovering the weighting
    and the fat-suppression flag from the header is the fix the reference pipeline
    uses -- but the public sample DICOMs arrive with headers stripped, so whether
    the mounted competition data actually carries these fields is an empirical
    question. This answers it in ~20 minutes instead of inside a 4-hour run.
    """
    from concurrent.futures import ThreadPoolExecutor
    from collections import Counter
    import time

    t0 = time.time()
    ds = RSNADataset(split)
    rows = []
    for uid in ds.uids:
        for sid, plane, fl, fat in ds.study_to_series.get(uid, []):
            rows.append((uid, sid, plane, int(fl), int(fat)))
    if limit:
        rows = rows[:limit]
    print(f"seq-diag: {len(rows)} series over {len(ds.uids)} studies", flush=True)
    if ds.series_root is None:
        print("seq-diag: no series_root -> nothing to read")
        return

    def _one(r):
        uid, sid, plane, fl, fat = r
        folder = os.path.join(ds.series_root, uid, sid)
        return r, du.read_series_meta(folder)

    metas = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for r, meta in ex.map(_one, rows):
            metas.append((r, meta))
    print(f"seq-diag: headers read in {(time.time() - t0) / 60:.1f} min", flush=True)

    field_hits = Counter()
    weights = Counter()
    n_rec = 0
    cross = Counter()          # (csv_fl, csv_fat) -> (rec_fl, rec_fat)
    rec_by_series = {}
    for (uid, sid, plane, fl, fat), meta in metas:
        for k, v in meta.items():
            if str(v).strip():
                field_hits[k] += 1
        rec = du.recover_sequence(meta)
        if rec is None:
            weights["UNRECOVERED"] += 1
            continue
        n_rec += 1
        rfl, rfat, w = rec
        weights[w] += 1
        rec_by_series[(uid, sid)] = (int(rfl), int(rfat))
        cross[((fl, fat), (int(rfl), int(rfat)))] += 1

    n = len(metas)
    print(f"\nseq-diag: header field presence over {n} series")
    for k in du._SEQ_FIELDS:
        print(f"  {k:20s} {field_hits[k]:6d}  ({field_hits[k] / n * 100:5.1f}%)")
    print(f"\nseq-diag: recovered {n_rec}/{n} ({n_rec / n * 100:.1f}%)")
    print("  weight histogram:", dict(weights))
    print("\nseq-diag: CSV (fluid,fatsat) -> recovered (fluid,fatsat)")
    for (csv_k, rec_k), c in sorted(cross.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {csv_k} -> {rec_k}: {c}")

    # Which (plane, fluid, fatsat) cells actually exist under recovered flags, and
    # which six of them cover the most studies. Copying the reference's slot list is
    # not obviously right for this corpus, so let the data name the slots.
    cell_studies = Counter()
    for (uid, sid, plane, fl, fat), _m in metas:
        k = rec_by_series.get((uid, sid), (fl, fat))
        cell_studies[(plane, k[0], k[1])] += 0     # ensure the key exists
    seen = set()
    for (uid, sid, plane, fl, fat), _m in metas:
        k = rec_by_series.get((uid, sid), (fl, fat))
        cell = (plane, k[0], k[1])
        if (uid, cell) not in seen:
            seen.add((uid, cell))
            cell_studies[cell] += 1
    print("\nseq-diag: recovered (plane, fluid, fatsat) -> studies covered")
    for cell, c in sorted(cell_studies.items(), key=lambda kv: -kv[1]):
        print(f"  {cell}: {c}")

    # Greedy: the six cells covering the most (study x slot) content. Coverage is not
    # the only criterion -- a slot also has to be contrast-homogeneous -- but a cell
    # that reaches almost no studies cannot earn a sixth of the cache either way.
    top6 = [c for c, _ in sorted(cell_studies.items(), key=lambda kv: -kv[1])[:6]]
    print("seq-diag: top-6 recovered cells by study coverage:", top6)
    print(f"  their total coverage: {sum(cell_studies[c] for c in top6)}"
          f"/{len(by_study_uids := {r[0][0] for r in metas}) * 6}"
          f" ({sum(cell_studies[c] for c in top6) / (len(by_study_uids) * 6) * 100:.1f}%)")

    # What each slot list would fill, under CSV flags vs recovered flags.
    lists = {
        "v13_csv_flags": [("Sagittal", 1, 1), ("Coronal", 1, 1), ("Axial", 1, 1),
                          ("Axial", 0, 0), ("Coronal", 0, 0), ("Sagittal", 0, 0)],
        "reference": [("Sagittal", 1, 1), ("Coronal", 1, 1), ("Axial", 1, 1),
                      ("Sagittal", 1, 0), ("Coronal", 0, 0), ("Sagittal", 0, 0)],
    }
    by_study = {}
    for (uid, sid, plane, fl, fat), _m in metas:
        by_study.setdefault(uid, []).append(
            (plane, (fl, fat), rec_by_series.get((uid, sid))))
    for name, slots in lists.items():
        for src in ("csv", "recovered"):
            for fb in (False, True):
                filled = [0] * len(slots)
                for uid, series in by_study.items():
                    for i, (p, f, a) in enumerate(slots):
                        hit = False
                        for plane, csv_k, rec_k in series:
                            k = csv_k if src == "csv" else (rec_k or csv_k)
                            if plane == p and k == (f, a):
                                hit = True
                                break
                        # The reference's fallback: a non-fluid slot is the scarcest,
                        # so before leaving it empty it takes ANY non-fat-suppressed
                        # series in that plane.
                        if not hit and fb and f == 0:
                            for plane, csv_k, rec_k in series:
                                k = csv_k if src == "csv" else (rec_k or csv_k)
                                if plane == p and k[1] == 0:
                                    hit = True
                                    break
                        filled[i] += int(hit)
                tot = sum(filled)
                cells = len(by_study) * len(slots)
                tag = "+fallback" if fb else "exact    "
                print(f"seq-diag fill [{name} / {src:9s} / {tag}]: {filled} "
                      f"= {tot}/{cells} ({tot / cells * 100:.1f}%)")
    print(f"seq-diag done in {(time.time() - t0) / 60:.1f} min", flush=True)


def plan_n_groups(n_study: int, budget_gb: float = CACHE_BUDGET_GB) -> int:
    """How many 3-slice groups fit under the uint8 cache budget."""
    bytes_per = n_study * N_SLOTS * GROUP * IMG_SIZE * IMG_SIZE  # one group
    n_g = N_GROUPS_MAX
    while n_g > 1 and n_g * bytes_per > budget_gb * (1024 ** 3):
        n_g -= 1
    return n_g


def take_group(cache_rows: np.ndarray, g: int) -> np.ndarray:
    """cache_rows [..., S, H, W] -> [..., 3, H, W] for group g."""
    return cache_rows[..., g * GROUP:(g + 1) * GROUP, :, :]


def _resolve_laterality(jobs, tag_pos, enable_fallback: bool):
    """jobs is a list of (study_i, slot_i, plane, folder). tag_pos[folder]=(tag,pos)."""
    sides = {}
    for folder, (tag, pos) in tag_pos.items():
        sides[folder] = tag or (pos if enable_fallback else None)
    return sides


def build_slot_cache(ds: "RSNADataset", indices, n_groups: int | None = None,
                     n_workers: int = 8):
    """Decode selected studies once into an in-memory uint8 cache.

    Returns (cache [N, NS, S, H, W] uint8, mask [N, NS] float32, n_groups).
    Fine-tuning revisits the same pixels every epoch; reading them from the
    mount each time would make the epoch count a function of I/O.
    """
    from concurrent.futures import ThreadPoolExecutor

    indices = np.asarray(indices)
    n = len(indices)
    n_groups = n_groups or plan_n_groups(n)
    n_slices = n_groups * GROUP
    cache = np.zeros((n, N_SLOTS, n_slices, IMG_SIZE, IMG_SIZE), np.uint8)
    mask = np.zeros((n, N_SLOTS), np.float32)
    print(f"cache: N={n} groups={n_groups} shape={cache.shape} "
          f"= {cache.nbytes / 1024 ** 3:.2f} GB")

    # Resolve (study, slot) -> folder, then laterality tags/positions.
    jobs = []  # (local_i, slot_i, plane, folder)
    tag_pos = {}
    for li, gi in enumerate(indices):
        uid = ds.uids[int(gi)]
        for si, folder in ds._slots_for_study(uid).items():
            plane = SLOTS[si][0]
            jobs.append((li, si, plane, folder))
            if folder not in tag_pos:
                tag_pos[folder] = du.read_laterality_full(folder)

    both = [(t, p) for (t, p) in tag_pos.values() if t and p]
    agree = (float(np.mean([t == p for t, p in both])) if both else float("nan"))
    use_fb = bool(both) and np.isfinite(agree) and agree >= LAT_MIN_AGREEMENT
    print(f"laterality: {len(tag_pos)} series, tag/pos agree "
          f"{agree:.3f} on {len(both)} -> fallback {'ON' if use_fb else 'OFF'}")
    sides = _resolve_laterality(jobs, tag_pos, use_fb)

    def _one(job):
        li, si, plane, folder = job
        lat = sides.get(folder)
        try:
            vol = du.load_series_stack(folder, plane, lat, n_slices, IMG_SIZE)
        except Exception:
            vol = np.zeros((n_slices, IMG_SIZE, IMG_SIZE), np.float32)
        return li, si, vol

    CHUNK = 256
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0:c0 + CHUNK]
            for li, si, vol in pool.map(_one, block):
                cache[li, si] = np.clip(vol * 255.0, 0, 255).astype(np.uint8)
                mask[li, si] = 1.0 if vol.max() > 0 else 0.0
                done += 1
            if done % 512 < CHUNK:
                print(f"  decoded {done}/{len(jobs)} slot-series", flush=True)
    print(f"cache ready: {int(mask.sum())} present slots / {mask.size}")
    return cache, mask, n_groups


def load_study_groups(ds: "RSNADataset", uid: str, n_groups: int):
    """On-the-fly test path: [NS, n_groups, 3, H, W] uint8 + mask [NS]."""
    n_slices = n_groups * GROUP
    stack = np.zeros((N_SLOTS, n_slices, IMG_SIZE, IMG_SIZE), np.uint8)
    mask = np.zeros(N_SLOTS, np.float32)
    for si, folder in ds._slots_for_study(uid).items():
        plane = SLOTS[si][0]
        lat = du.read_laterality(folder)
        vol = du.load_series_stack(folder, plane, lat, n_slices, IMG_SIZE)
        stack[si] = np.clip(vol * 255.0, 0, 255).astype(np.uint8)
        if vol.max() > 0:
            mask[si] = 1.0
    groups = stack.reshape(N_SLOTS, n_groups, GROUP, IMG_SIZE, IMG_SIZE)
    return groups, mask

# ===================== backbones/dinov2.py =====================
"""Minimal, offline DINOv2 Vision Transformer (ViT-S / ViT-B).

Self-contained implementation that matches the official `facebookresearch/dinov2`
checkpoint keys (dinov2_vits14 / dinov2_vitb14, no register tokens) so we can
load weights that were saved locally (see src/download_dinov2.py) WITHOUT any
internet access at submit time -- a hard requirement of this Code Competition.

Only the [CLS] token embedding is used as the per-slice representation.

Input convention: forward() takes [N, 1, H, W] or [N, 3, H, W] in [0,1].
1-channel inputs are replicated to 3 (legacy grayscale path). 3-channel
inputs are treated as a 2.5D group of adjacent slices. ImageNet
normalization is applied internally.
"""
import math
import os
import glob

import torch
import torch.nn as nn


# ImageNet statistics used by the DINOv2 pretraining pipeline.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B, nh, N, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _LayerScale(nn.Module):
    """Per-channel learnable scale (DINOv2 uses this after attn/mlp)."""
    def __init__(self, dim: int, init: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init)

    def forward(self, x):
        return x * self.gamma


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, mlp_ratio)
        self.ls2 = _LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2ViT(nn.Module):
    """DINOv2 ViT (patch14). `embed_dim`/`depth`/`num_heads` select S vs B.

    Architecture matches the official released `dinov2_vit{s,b}14_pretrain` weights
    (patch embedding is a Conv2d keyed `patch_embed.weight/bias`; each block has
    LayerScale `ls1/ls2`; positional embedding is interpolated from 518->224).
    """

    def __init__(self, img_size: int = 224, patch_size: int = 14,
                 embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze(self, unfreeze_last_n: int = 2):
        """Freeze everything, then unfreeze the last `unfreeze_last_n` blocks
        and the final norm. Early blocks stay frozen: there is not enough
        supervision here to improve generic edge filters and enough to damage them.
        """
        for p in self.parameters():
            p.requires_grad = False
        n = len(self.blocks)
        for i in range(max(0, n - unfreeze_last_n), n):
            for p in self.blocks[i].parameters():
                p.requires_grad = True
        for p in self.norm.parameters():
            p.requires_grad = True

    @torch.jit.unused
    def _interpolate_pos_embed(self, sd_pos: torch.Tensor) -> torch.Tensor:
        """Resize the checkpoint's pos_embed to our num_patches if they differ."""
        n = self.num_patches + 1
        if sd_pos.shape[1] == n:
            return sd_pos
        cls = sd_pos[:, :1]
        patch = sd_pos[:, 1:]
        old = int(math.sqrt(patch.shape[1]))
        patch = patch.reshape(1, old, old, -1).permute(0, 3, 1, 2)
        patch = torch.nn.functional.interpolate(
            patch, size=int(math.sqrt(self.num_patches)),
            mode="bicubic", align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, self.num_patches, -1)
        return torch.cat([cls, patch], dim=1)

    def load_pretrained(self, path: str):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k: v for k, v in sd.items()
              if not k.startswith("head.") and k != "mask_token"}
        # Remap facebookresearch/timm-style keys to our module naming.
        # Official `dinov2_vit{s,b}14_pretrain` weights expose the patch
        # embedding as `patch_embed.proj.{weight,bias}`; our model uses a plain
        # Conv2d registered as `patch_embed.{weight,bias}`.
        remap = {}
        for k, v in sd.items():
            if k == "patch_embed.proj.weight":
                remap["patch_embed.weight"] = v
            elif k == "patch_embed.proj.bias":
                remap["patch_embed.bias"] = v
            else:
                remap[k] = v
        sd = remap
        if "pos_embed" in sd and sd["pos_embed"].shape != self.pos_embed.shape:
            try:
                sd["pos_embed"] = self._interpolate_pos_embed(sd["pos_embed"])
            except Exception:
                sd.pop("pos_embed", None)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        return missing, unexpected

    def _prepare(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward_tokens(self, x):
        """Return all tokens [B, 1+L, D] after the final norm (CLS first)."""
        x = self._prepare(x)
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_features(self, x):
        return self.forward_tokens(x)[:, 0]

    def forward(self, x):
        return self.forward_features(x)


def _resolve_weights(path_env: str | None, variant: str) -> str | None:
    """Resolve a DINOv2 .pth path from an explicit env override, else glob a
    mounted Kaggle dataset directory for common weight filenames.

    Kaggle mounts dataset inputs under /kaggle/input/<slug>, but the exact
    layout varies (sometimes nested under /kaggle/input/datasets/<slug>), so we
    try explicit candidates AND fall back to a recursive search of /kaggle/input.
    """
    fname = f"{variant}.pth"            # e.g. dinov2_vits14.pth
    if path_env and os.path.exists(path_env):
        return path_env
    if path_env and os.path.isdir(path_env):
        cand = glob.glob(os.path.join(path_env, "*.pth"))
        cand += glob.glob(os.path.join(path_env, "*.safetensors"))
        if cand:
            return sorted(cand)[0]

    # Local repo copy (src/download_dinov2.py) plus every Kaggle mount we have seen.
    _here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(_here, "data", "dinov2_weights"),
        "/kaggle/input/rsna-dinov2-weights",
        "/kaggle/input/datasets/rsna-dinov2-weights",
        "/kaggle/input/dinov2",
        "/kaggle/input/dinov2/PyTorch/small",
        "/kaggle/input/dinov2/PyTorch/base",
        f"/kaggle/input/{variant}",
    ]
    for base in candidates:
        if os.path.isdir(base):
            for fn in (fname, f"dinov2_{variant}.pth"):
                p = os.path.join(base, fn)
                if os.path.exists(p):
                    return p
            cand = glob.glob(os.path.join(base, "*.pth"))
            if cand:
                return sorted(cand)[0]

    # Final fallback: recursive search so we never miss a non-standard mount.
    if os.path.isdir("/kaggle/input"):
        hits = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
        if hits:
            return sorted(hits)[0]
        hits = glob.glob("/kaggle/input/**/dinov2_vit*.pth", recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def _debug_input_tree(label: str) -> None:
    """Print what is actually mounted under /kaggle/input when weights are
    missing, so a non-attached dataset is immediately diagnosable."""
    print(f"[{label}] DEBUG /kaggle/input tree:")
    if not os.path.isdir("/kaggle/input"):
        print("  /kaggle/input does not exist")
        return
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - 2  # relative to /kaggle/input
        if depth > 3:
            continue
        indent = "  " * (depth + 1)
        print(f"{indent}{os.path.basename(root)}/")
        if depth >= 2:
            for f in files:
                print(f"{indent}  {f}")


# The ViT grid must follow the configured input size: pos_embed is added by
# slicing `pos_embed[:, :n_token]`, so a backbone built for 224 (256 patches) fed
# 168 px input (144 patches) would silently take the first 145 of a 257-row table
# in row-major order -- wrong positions, no error. Build it at IMG_SIZE instead.
try:
    from config import IMG_SIZE as _CFG_IMG
except Exception:  # pragma: no cover - standalone import in smoke tests
    _CFG_IMG = 224


class DINOv2SmallBackbone(nn.Module):
    """DINOv2 ViT-S/14 backbone wrapper (embed_dim=384)."""

    def __init__(self, weights_path: str | None = None, unfreeze_last: int = 2,
                 img_size: int = _CFG_IMG):
        super().__init__()
        self.vit = DINOv2ViT(img_size=img_size, embed_dim=384, depth=12,
                             num_heads=6)
        self.embed_dim = 384
        path = _resolve_weights(weights_path, "dinov2_vits14")
        if path:
            miss, unexp = self.vit.load_pretrained(path)
            print(f"[DINOv2-small] loaded {path} | missing={len(miss)} "
                  f"unexpected={len(unexp)}")
        else:
            print("[DINOv2-small] WARNING: no weights found -> random init")
            _debug_input_tree("DINOv2-small")
        self.vit.freeze(unfreeze_last)

    def forward(self, x):
        return self.vit(x)

    def forward_tokens(self, x):
        return self.vit.forward_tokens(x)

    def freeze(self, n: int = 2):
        self.vit.freeze(n)


class DINOv2BaseBackbone(nn.Module):
    """DINOv2 ViT-B/14 backbone wrapper (embed_dim=768)."""

    def __init__(self, weights_path: str | None = None, unfreeze_last: int = 2,
                 img_size: int = _CFG_IMG):
        super().__init__()
        self.vit = DINOv2ViT(img_size=img_size, embed_dim=768, depth=12,
                             num_heads=12)
        self.embed_dim = 768
        path = _resolve_weights(weights_path, "dinov2_vitb14")
        if path:
            miss, unexp = self.vit.load_pretrained(path)
            print(f"[DINOv2-base] loaded {path} | missing={len(miss)} "
                  f"unexpected={len(unexp)}")
        else:
            print("[DINOv2-base] WARNING: no weights found -> random init")
            _debug_input_tree("DINOv2-base")
        self.vit.freeze(unfreeze_last)

    def forward(self, x):
        return self.vit(x)

    def forward_tokens(self, x):
        return self.vit.forward_tokens(x)

    def freeze(self, n: int = 2):
        self.vit.freeze(n)

# ===================== backbones/tiny_cnn.py =====================
"""From-scratch tiny CNN backbone (baseline, NO pretraining).

Kept as the control condition in the comparison study: it is trained entirely
from the (small) RSNA training set, so its score isolates the value of a
pretrained, radiology-adapted representation (DINOv2) in the same pipeline.
"""
import torch
import torch.nn as nn


class TinyCNNBackbone(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.embed_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):  # x: [N, 1|3, H, W] in [0,1]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.head(self.net(x).flatten(1))

    def freeze(self, unfreeze_last_n: int = 2):
        # Control condition: train the whole tiny CNN, nothing frozen.
        for p in self.parameters():
            p.requires_grad = True

# ===================== model.py =====================
"""Imaging-only model for RSNA Knee (slot attention over 6 MRI slots).

Architecture mirrors the highest-voted public notebooks (see LOG.md):
- A pluggable image BACKBONE encodes each MRI slice -> per-slice embedding.
  The backbone is the controlled variable of the comparison study:
    * TinyCNNBackbone        : from-scratch small CNN (control condition).
    * DINOv2SmallBackbone    : DINOv2 ViT-S/14 (offline weights).
    * DINOv2BaseBackbone     : DINOv2 ViT-B/14 (offline weights).
- Slot pooling -> 6 slot embeddings (one per plane/fluid/fat slot), gated by a
  presence mask so missing series are ignored.
- A PER-DIAGNOSIS attention head (one SlotHead per label) attends over the 6
  slots so, e.g., the ACL head focuses on Sagittal fluid-sensitive series while
  the OA head spreads attention across planes.
- The radiology report is NOT fed at inference (test.csv has no Report column).
  Report text is consumed offline into graded soft targets + weights by
  report_extractor.py, so this model is pure vision and submission-safe.

Training helpers (used by train.py / experiments.py):
- EMA shadow weights for inference.
- Encoder/head LR split (head LR; backbone LR scaled by `backbone.lr_scale`,
  ~0.01 for DINOv2, 1.0 for the tiny CNN). DINOv2 is frozen except its last
  ENCODER_UNFREEZE_LAST blocks (+ final norm), per community consensus.
- Pairwise ranking loss supplementing BCE.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (N_LABELS, IMG_SIZE, SLICES_PER_SLOT, N_SLOTS, SLOTS,
                    ENCODER_LR_FACTOR, RANK_LOSS_WEIGHT, SLOT_EMBED,
                    BACKBONE_EMBED, DINOV2_SMALL_PATH, DINOV2_BASE_PATH,
                    LABELS, SLOT_PRIOR_BIAS, EMA_DECAY, AUG_ROT_DEG,
                    AUG_SCALE, AUG_SHIFT, AUG_INTENSITY,
                    RANK_POS, RANK_NEG)
from backbones import (TinyCNNBackbone, DINOv2SmallBackbone, DINOv2BaseBackbone)


class SlotImageEncoder(nn.Module):
    """Encode one 2.5D group per slot -> [B, NS, feat], gated by presence mask.

    DINOv2 tokens are pooled three ways (CLS + patch mean + top-k focal) so
    diffuse disease and focal tears both survive the slot summary. TinyCNN
    has no token grid, so its embedding is used as-is.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.use_tokens = hasattr(backbone, "forward_tokens")
        self.feat_dim = backbone.embed_dim * (3 if self.use_tokens else 1)

    def forward(self, imgs, mask):
        # imgs: [B, NS, 3, H, W] (uint8 0-255 or float 0-1 / 0-255)
        B, NS, C, H, W = imgs.shape
        x = imgs.reshape(B * NS, C, H, W)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        elif x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        if self.use_tokens:
            tok = self.backbone.forward_tokens(x)      # [B*NS, 1+L, D]
            cls = tok[:, 0]
            patch = tok[:, 1:]
            k = max(1, patch.shape[1] // 8)
            focal = patch.topk(k, dim=1).values.mean(1)
            feat = torch.cat([cls, patch.mean(1), focal], dim=1)
        else:
            feat = self.backbone(x)
        feat = feat.view(B, NS, -1)
        return feat * mask.unsqueeze(-1)


class SlotHead(nn.Module):
    """Per-diagnosis attention over slot embeddings, with a weak anatomical prior.

    Each finding is read on particular sequences, so a shared mean would dilute
    the one slot that carries the evidence. The prior is only a 1.7x softmax
    tilt (0.55) and the learned query can override it.
    """
    def __init__(self, dim: int, n_slot: int = N_SLOTS, n_out: int = N_LABELS,
                 hidden: int = 256, p: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out_w = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.out_b = nn.Parameter(torch.zeros(n_out))
        self.hidden = hidden
        prior = torch.zeros(n_out, n_slot)
        # Slots: 0 SAG_FLUID_FS, 1 COR_FLUID_FS, 2 AX_FLUID_FS,
        #        3 AX_STRUCT, 4 COR_STRUCT, 5 SAG_STRUCT (the v13 list, which the
        # ablation left as the best-measured one).
        preferred = {
            "ACL": (0, 5), "MCL": (1, 4),
            "Medial Meniscus": (0, 1, 4),
            "Lateral Meniscus": (0, 1, 4),
            "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
            "PF OA": (0, 2, 3, 5), "Effusion": (0, 2, 3),
            "Synovitis": (0, 2, 3), "Baker's": (0,),
            "Contusion": (0, 1, 2, 3), "Fracture": (0, 1, 2, 3, 4, 5),
        }
        for target, slots in preferred.items():
            if target in LABELS:
                prior[LABELS.index(target), list(slots)] = SLOT_PRIOR_BIAS
        self.register_buffer("slot_prior", prior)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb                 # [B, S, H]
        att = (torch.einsum("bsh,oh->bos", h, self.query) / math.sqrt(self.hidden)
               + self.slot_prior.unsqueeze(0))
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return torch.einsum("boh,oh->bo", ctx, self.out_w) + self.out_b


class RSNAMultimodal(nn.Module):
    def __init__(self, backbone: nn.Module, slot_dim: int = SLOT_EMBED,
                 head_out: int = 64):
        super().__init__()
        self.img_enc = SlotImageEncoder(backbone)
        self.head = SlotHead(self.img_enc.feat_dim)

    def forward(self, imgs, mask):  # -> logits [B, N_LABELS]
        slots = self.img_enc(imgs, mask)
        return self.head(slots, mask)

    def predict(self, imgs, mask):
        return torch.sigmoid(self.forward(imgs, mask))


def augment(imgs):
    """Small affine + intensity jitter over a whole bag. No flips.

    Horizontal flip would undo laterality normalisation. Vertical flip would
    swap femur and tibia — the axis the compartmental OA targets are defined on.
    `imgs` is [B, S, C, H, W] in [0, 1] (float).
    """
    B, S, C, H, W = imgs.shape
    x = imgs.float()
    ang = math.radians((torch.rand(1).item() - 0.5) * 2 * AUG_ROT_DEG)
    sc = 1.0 + (torch.rand(1).item() - 0.5) * 2 * AUG_SCALE
    tx = (torch.rand(1).item() - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(1).item() - 0.5) * 2 * AUG_SHIFT
    cos, sin = math.cos(ang) / sc, math.sin(ang) / sc
    theta = x.new_tensor([[cos, -sin, tx], [sin, cos, ty]])
    theta = theta.unsqueeze(0).repeat(B * S, 1, 1)
    flat = x.reshape(B * S, C, H, W)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    x = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros",
                      align_corners=False).reshape(B, S, C, H, W)
    scale = 1.0 + (torch.rand(1, device=x.device) - 0.5) * 2 * AUG_INTENSITY
    return (x * scale).clamp(0, 1)


def build_model(backbone_name: str = "tiny_cnn", unfreeze_last: int = 2,
                slot_dim: int = SLOT_EMBED) -> RSNAMultimodal:
    """Construct the model for a given backbone (the controlled variable)."""
    if backbone_name == "tiny_cnn":
        bb = TinyCNNBackbone(out_dim=BACKBONE_EMBED["tiny_cnn"])
        bb.lr_scale = 1.0                       # train the whole CNN at head LR
    elif backbone_name == "dinov2_small":
        bb = DINOv2SmallBackbone(weights_path=DINOV2_SMALL_PATH,
                                 unfreeze_last=unfreeze_last)
        bb.lr_scale = ENCODER_LR_FACTOR
    elif backbone_name == "dinov2_base":
        bb = DINOv2BaseBackbone(weights_path=DINOV2_BASE_PATH,
                                unfreeze_last=unfreeze_last)
        bb.lr_scale = ENCODER_LR_FACTOR
    else:
        raise ValueError(f"unknown backbone: {backbone_name}")
    return RSNAMultimodal(bb, slot_dim=slot_dim)


def get_param_groups(model: "RSNAMultimodal", head_lr: float):
    """Heads get `head_lr`; the backbone gets head_lr * backbone.lr_scale.
    Frozen params (requires_grad=False) are omitted so OneCycleLR step counts
    and AdamW state stay on the trainable set."""
    head_params, enc_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith("head.") or any(
            k in name for k in ("heads.", "classifier.", "slot_pool"))
        (head_params if is_head else enc_params).append(p)
    enc_lr = head_lr * getattr(model.img_enc.backbone, "lr_scale",
                               ENCODER_LR_FACTOR)
    groups = [{"params": head_params, "lr": head_lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": enc_lr})
    return groups


class EMA:
    """Exponential moving average of model weights for inference/submit.

    Decay is warmed up so the first few hundred steps are not anchored to the
    random head initialisation (most of a short fold).
    """
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.step = 0
        self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        if self.decay <= 0:
            return
        self.step += 1
        decay = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        src = model.state_dict()
        for k, v in self.state.items():
            s = src[k]
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(s.detach(), alpha=1.0 - decay)
            else:
                v.copy_(s)

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.state)

    def state_dict(self):
        return self.state


class PairwiseRankingLoss(nn.Module):
    """Soft pairwise ranking on confidently graded pairs inside the batch.

    AUC only reads order, so this optimises the metric directly. A target with
    no usable pair (rare findings in a small batch) contributes nothing rather
    than a constant — which is why the weight stays small and this supplements
    BCE instead of replacing it.
    """
    def __init__(self, pos_cut: float = RANK_POS, neg_cut: float = RANK_NEG):
        super().__init__()
        self.pos_cut = pos_cut
        self.neg_cut = neg_cut

    def forward(self, logits, targets, weights=None):
        usable = (torch.ones_like(targets, dtype=torch.bool) if weights is None
                  else weights > 0)
        parts = []
        for j in range(logits.shape[1]):
            pos = logits[(targets[:, j] > self.pos_cut) & usable[:, j], j]
            neg = logits[(targets[:, j] < self.neg_cut) & usable[:, j], j]
            if pos.numel() and neg.numel():
                parts.append(F.softplus(-(pos[:, None] - neg[None, :])).mean())
        return torch.stack(parts).mean() if parts else logits.new_tensor(0.0)

# ===================== train.py =====================
"""Training / CV for RSNA Knee (run inside Kaggle Notebook, internet OFF).

Key practices wired in from top public notebooks (see LOG.md):
- GroupKFold grouped by report-text hash so identical/duplicated reports never
  leak across folds.
- Per-label confidence weights (from report_extractor) so a study silent on a
  finding is a weak, not confident, negative; gold labels get weight 1.0.
- BCE (weighted) + pairwise ranking loss; EMA shadow weights for inference.
- Epoch selection: each epoch is scored on the holdout as min(derived-AUC,
  gold-AUC) and the best epoch's EMA state is restored ("worse_of_two").
- Macro-AUC evaluated only on the held-out GOLD-labeled portion (NaN-aware).
- Rank-averaged OOF across folds for the final ensembled prediction.

This module is the single source of truth for the training protocol. The
comparison study (experiments.py) calls run_cv() with different BACKBONEs while
keeping EVERYTHING else fixed, so differences in score reflect the backbone only.
"""
import os
import hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from config import (LABELS, N_FOLDS, BATCH_SIZE, EPOCHS, LR, SEED, PROJECT_ROOT,
                    IMG_SIZE,
                    USE_EMA, RANK_LOSS_WEIGHT, ENCODER_UNFREEZE_LAST,
                    TRAIN_TIME_BUDGET_H, WEIGHT_DECAY, ONECYCLE_PCT_START,
                    SELECTION)
from dataset import RSNADataset, build_slot_cache, take_group
import model as M

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)


# ---------------- offline metrics (no sklearn dependency) ----------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    return ranks


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney AUC; y_true binary (0/1), y_score continuous."""
    yt = y_true.astype(float)
    n_pos = yt.sum(); n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    sum_r = ranks[yt == 1].sum()
    return (sum_r - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def macro_auc(y_true, y_pred):
    aucs = []
    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_pred[:, i]
        ok = np.isfinite(yt)
        if ok.sum() < 2:
            aucs.append(float("nan")); continue
        yt, yp = yt[ok], yp[ok]
        if len(np.unique(yt)) < 2:
            aucs.append(float("nan")); continue
        aucs.append(roc_auc(yt, yp))
    return np.nanmean(aucs), aucs


def rank_average(fold_preds: list[np.ndarray]) -> np.ndarray:
    """Average predictions across folds in rank space (robust to per-fold drift)."""
    out = np.zeros_like(fold_preds[0])
    N = fold_preds[0].shape[0]
    for j in range(fold_preds[0].shape[1]):
        ranks = np.zeros(N)
        for p in fold_preds:
            ranks += _rankdata(p[:, j])
        ranks /= len(fold_preds)
        pooled = np.concatenate([p[:, j] for p in fold_preds])
        q = (ranks - 0.5) / N
        out[:, j] = np.quantile(pooled, q)
    return out


def _report_hash(report: str) -> int:
    return int(hashlib.md5(report.encode("utf-8", "ignore")).hexdigest(), 16)


# ---------------- training on an in-memory cache ----------------
def _prepare_imgs(cache, idx, g, device, do_aug=False):
    rows = torch.from_numpy(take_group(cache[idx], g))  # [B, NS, 3, H, W] uint8
    rows = rows.to(device)
    x = rows.float().div_(255.0)
    if do_aug:
        x = M.augment(x)
    return x


@torch.no_grad()
def predict_groups(model, cache, mask, idx, n_groups, device, batch=16):
    """Average logits over the 2.5D groups of each slot."""
    model.eval()
    out = []
    idx = np.asarray(idx)
    for b0 in range(0, len(idx), batch):
        sel = idx[b0:b0 + batch]
        m = torch.from_numpy(mask[sel]).to(device)
        acc = None
        use_amp = device == "cuda"
        for g in range(n_groups):
            x = _prepare_imgs(cache, sel, g, device, do_aug=False)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z = model(x, m).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / n_groups).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, len(LABELS)), np.float32)


def train_on_cache(cache, mask, y, w, tr_idx, epochs, backbone_name,
                   unfreeze_last, lr, batch, n_groups, device=None,
                   va_idx=None, y_gold_va=None):
    """Train one model on cached 2.5D groups. Returns (model, ema).

    Epoch selection (config.SELECTION): after each
    epoch the EMA weights are evaluated on the holdout twice — against the
    binarised derived targets (all holdout studies with weight>0) and, when
    >=6 gold studies are held out, against the gold labels. The epoch whose
    min(derived, gold) macro-AUC is best is restored at the end, so a bad
    final epoch cannot ship. Without a holdout the final EMA state is used.
    """
    device = device or DEVICE
    model = M.build_model(backbone_name, unfreeze_last)
    model.to(device)
    groups = M.get_param_groups(model, lr)
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=WEIGHT_DECAY)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    rankloss = M.PairwiseRankingLoss()
    ema = M.EMA(model) if USE_EMA else None
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    tr_idx = np.asarray(tr_idx)
    n = len(tr_idx)
    rng = np.random.RandomState(SEED)
    # Drop the leftover incomplete batch so OneCycleLR step counts match
    # and rank-loss pairs are not computed on a 1-2 study remainder.
    steps_per = max(n // batch, 1)
    total_steps = max(epochs * steps_per, 1)
    max_lrs = [g.get("lr", lr) for g in groups]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lrs, total_steps=total_steps,
        pct_start=ONECYCLE_PCT_START)

    do_select = (ema is not None and va_idx is not None and len(va_idx) > 0)
    if do_select:
        va_idx = np.asarray(va_idx)
        yd = y[va_idx].astype(float).copy()
        yd[w[va_idx] <= 0] = np.nan          # silent targets are not evidence
        yd_bin = np.where(np.isfinite(yd), (yd > 0.5).astype(float), np.nan)
        has_gold = (y_gold_va is not None and
                    np.isfinite(y_gold_va).any(axis=1).sum() >= 6)
    sel_state, sel_score = None, -1.0

    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        nb, loss_sum = 0, 0.0
        for b0 in range(0, n - batch + 1, batch):
            sel = tr_idx[order[b0:b0 + batch]]
            g = int(rng.randint(0, n_groups))
            imgs = _prepare_imgs(cache, sel, g, device, do_aug=True)
            masks = torch.from_numpy(mask[sel]).to(device)
            yt = torch.from_numpy(y[sel]).to(device)
            wt = torch.from_numpy(w[sel]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs, masks)
                loss_bce = (bce(logits, yt) * wt).sum() / wt.sum().clamp(min=1.0)
                loss = loss_bce + RANK_LOSS_WEIGHT * rankloss(logits, yt, wt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            loss_sum += float(loss.detach())
            nb += 1
        msg = f"    epoch {ep + 1}/{epochs}  loss={loss_sum / max(nb, 1):.4f}"
        if do_select:
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.apply_to(model)
            pred = predict_groups(model, cache, mask, va_idx, n_groups, device)
            der, _ = macro_auc(yd_bin, pred)
            gld, _ = (macro_auc(y_gold_va, pred) if has_gold
                      else (der, None))
            score = min(der, gld) if SELECTION == "worse_of_two" else der
            if np.isfinite(score) and score > sel_score:
                sel_score = score
                sel_state = {k: v.detach().clone()
                             for k, v in ema.state.items()}
            msg += (f"  sel[{SELECTION}]={score:.4f} (der={der:.4f}"
                    + (f", gold={gld:.4f}" if has_gold else "") + ")")
            model.load_state_dict(raw)
        print(msg, flush=True)

    if ema:
        if sel_state is not None:
            ema.state = sel_state            # checkpoint saves the best epoch
        ema.apply_to(model)
    model.eval()
    return model, ema


def _subset_indices(ds, max_studies: int | None):
    """If max_studies given, keep ALL gold studies + randomly sample the rest so
    the CV validation stays meaningful (and reproducible via SEED)."""
    idx = np.arange(len(ds))
    if max_studies is None or max_studies >= len(ds):
        return idx
    rng = np.random.RandomState(SEED)
    gold = np.where(ds.has_gold)[0]
    if max_studies <= len(gold):
        return rng.choice(gold, size=max_studies, replace=False)
    rest = np.setdiff1d(idx, gold)
    take = rng.choice(rest, size=min(max_studies - len(gold), len(rest)),
                      replace=False)
    return np.concatenate([gold, take])


def run_cv(backbone_name: str = "tiny_cnn", unfreeze_last: int = 2,
           max_studies: int | None = None, epochs: int = EPOCHS,
           folds: int = N_FOLDS, batch: int = BATCH_SIZE, lr: float = LR):
    """Run the fixed-protocol 5-fold CV for one backbone. Returns a dict with
    OOF predictions, fold scores, per-label AUCs and the split assignment
    (needed for reproducibility logging)."""
    ds = RSNADataset("train")
    idx = _subset_indices(ds, max_studies)
    reports = [ds.reports[i] for i in idx]
    strat = np.nansum(ds.y_raw[idx], axis=1).astype(int).clip(0, 5)
    groups = np.array([_report_hash(r) for r in reports])

    if folds == 1:
        # Single fixed group-disjoint 80/20 split (controlled comparison mode).
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
        tr, va = next(gss.split(idx, strat, groups))
        splits = [(tr, va)]
    else:
        skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(idx, strat, groups))

    cache, cmask, n_groups = build_slot_cache(ds, idx)
    y = ds.targets[idx]
    w = ds.weights[idx]
    oof = np.full((len(idx), len(LABELS)), np.nan, dtype=np.float32)
    fold_preds, fold_scores, fold_splits = [], [], []
    for fold, (tr, va) in enumerate(splits):
        fold_splits.append((idx[tr].tolist(), idx[va].tolist()))
        print(f"Fold {fold}: train={len(tr)} val={len(va)}", flush=True)
        model, _ema = train_on_cache(
            cache, cmask, y, w, tr, epochs, backbone_name, unfreeze_last,
            lr, batch, n_groups, va_idx=va, y_gold_va=ds.y_raw[idx[va]])
        pred = predict_groups(model, cache, cmask, va, n_groups, DEVICE)
        oof[va] = pred
        fold_preds.append(pred)
        golds = ds.y_raw[idx[va]]
        _, per = macro_auc(golds, pred)
        macro = np.nanmean([a for a in per if np.isfinite(a)])
        fold_scores.append(macro)
        print(f"Fold {fold}: macro-AUC={macro:.4f}  " +
              ", ".join(f"{l[:4]}:{a:.2f}" for l, a in zip(LABELS, per)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall, _ = macro_auc(ds.y_raw[idx], oof)
    per_label = []
    for j in range(len(LABELS)):
        yt = ds.y_raw[idx][:, j]; yp = oof[:, j]
        ok = np.isfinite(yt)
        per_label.append(roc_auc(yt[ok], yp[ok]) if ok.sum() >= 2 and
                         len(np.unique(yt[ok])) >= 2 else float("nan"))
    return dict(oof=oof, fold_preds=fold_preds, fold_scores=fold_scores,
                macro=float(overall), per_label=per_label,
                idx=idx.tolist(), fold_splits=fold_splits,
                strat=strat.tolist(), groups=groups.tolist())


def _write_submission(uids, preds, path):
    import pandas as pd
    sub = pd.DataFrame({"StudyInstanceUID": uids})
    for i, lab in enumerate(LABELS):
        sub[lab] = preds[:, i]
    sub = sub[["StudyInstanceUID"] + LABELS]
    sub.to_csv(path, index=False)
    print(f"wrote {path} ({len(sub)} rows)", flush=True)


def infer_test_ensemble(models, n_groups, device, series_root=None):
    """Decode the test set once, run each fold model, rank-average."""
    test = RSNADataset("test", series_root=series_root)
    uids = test.uids
    if not uids:
        return uids, np.zeros((0, len(LABELS)), np.float32)
    idx = np.arange(len(uids))
    cache, mask, ng = build_slot_cache(test, idx, n_groups=n_groups)
    fold_preds = []
    for mi, model in enumerate(models):
        print(f"infer test model {mi + 1}/{len(models)} ({len(uids)} studies)",
              flush=True)
        fold_preds.append(predict_groups(model, cache, mask, idx, ng, device))
    del cache, mask
    if len(fold_preds) == 1:
        return uids, fold_preds[0]
    return uids, rank_average(fold_preds)


def find_checkpoints(root: str = "/kaggle/input", max_depth: int = 4):
    """Every `*.pt` fold checkpoint mounted as a Kaggle dataset input.

    The DINOv2 backbone weights are `.pth`, so a `.pt` filter separates trained
    fold checkpoints from pretrained encoder weights without naming either one.
    """
    found = []
    if not os.path.isdir(root):
        return found
    for dirpath, dirnames, files in os.walk(root):
        if dirpath[len(root):].count(os.sep) > max_depth:
            dirnames[:] = []
            continue
        for f in files:
            if f.endswith(".pt"):
                found.append(os.path.join(dirpath, f))
    return sorted(found)


def run_infer_only(out_dir: str = ".", device: str | None = None):
    """Rank-average every mounted fold checkpoint over the test set. No training.

    Used to ensemble folds from SEPARATE training kernels (a code competition
    reruns the kernel on the hidden test, so cross-run ensembling cannot be done
    by combining local CSVs -- the checkpoints have to be carried into one
    inference kernel as a dataset).
    """
    import time
    t0 = time.time()
    dev = device or DEVICE
    ckpts = find_checkpoints()
    print(f"ensemble: {len(ckpts)} checkpoint(s) found", flush=True)
    for c in ckpts:
        print("   ", c, flush=True)
    if not ckpts:
        raise RuntimeError("no *.pt checkpoints under /kaggle/input")

    models, n_groups = [], None
    for c in ckpts:
        blob = torch.load(c, map_location="cpu", weights_only=False)
        bb = blob.get("backbone", "dinov2_small")
        ul = int(blob.get("unfreeze_last", ENCODER_UNFREEZE_LAST))
        ng = int(blob.get("n_groups", 1))
        # The cache is built once for all models, so the ensemble runs at the
        # smallest group count any member was trained with.
        n_groups = ng if n_groups is None else min(n_groups, ng)
        ck_img = int(blob.get("img_size", 224))
        # The cache is decoded once at config.IMG_SIZE, and the ViT grid is built
        # from it, so a checkpoint trained at another size would be fed the wrong
        # token count with the wrong pos_embed rows -- silently, since the slice
        # `pos_embed[:, :n_token]` always succeeds. Fail loudly instead.
        if ck_img != IMG_SIZE:
            raise RuntimeError(
                f"{os.path.basename(c)} was trained at img_size={ck_img} but this "
                f"kernel decodes at {IMG_SIZE}; ensemble one resolution per kernel")
        m = M.build_model(bb, unfreeze_last=ul)
        m.load_state_dict(blob["state_dict"])
        m.to(dev).eval()
        models.append(m)
        print(f"    loaded {os.path.basename(c)} backbone={bb} n_groups={ng}",
              flush=True)

    uids, preds = infer_test_ensemble(models, n_groups, dev)
    out_csv = os.path.join(out_dir, "submission.csv")
    _write_submission(uids, preds, out_csv)
    print(f"infer-only finished in {(time.time() - t0) / 60:.1f} min, "
          f"{len(models)} models", flush=True)
    return out_csv


def run_submit(backbone_name: str = "dinov2_small", unfreeze_last: int = 2,
               max_studies: int | None = None, epochs: int = EPOCHS,
               folds: int = N_FOLDS, batch: int = BATCH_SIZE, lr: float = LR,
               out_dir: str = ".", time_budget_h: float = TRAIN_TIME_BUDGET_H):
    """Train a report-hash fold ensemble and write submission.csv incrementally.

    After each finished fold we overwrite submission.csv with the rank-average
    of the models we have so far, so a time-budget stop still leaves a real
    (not placeholder) file.
    """
    import time
    t0 = time.time()
    budget = time_budget_h * 3600.0
    out_csv = os.path.join(out_dir, "submission.csv")

    ds = RSNADataset("train")
    idx = _subset_indices(ds, max_studies)
    reports = [ds.reports[i] for i in idx]
    strat = np.nansum(ds.y_raw[idx], axis=1).astype(int).clip(0, 5)
    groups = np.array([_report_hash(r) for r in reports])
    print(f"submit train: {len(idx)} studies / gold={int(ds.has_gold.sum())} "
          f"backbone={backbone_name} ep={epochs} folds={folds}", flush=True)

    if folds <= 1:
        splits = [(np.arange(len(idx)), np.array([], dtype=int))]
    else:
        skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(idx, strat, groups))

    cache, cmask, n_groups = build_slot_cache(ds, idx)
    y = ds.targets[idx]
    w = ds.weights[idx]

    models, fold_times = [], []
    for fold, (tr, va) in enumerate(splits):
        elapsed = time.time() - t0
        if fold > 0:
            mean_ft = float(np.mean(fold_times))
            if elapsed + 1.15 * mean_ft > budget:
                print(f"time budget: stopping after {fold} folds "
                      f"(elapsed={elapsed / 60:.1f}m, next~{mean_ft / 60:.1f}m)",
                      flush=True)
                break
        print(f"=== submit fold {fold + 1}/{len(splits)} "
              f"train={len(tr)} val={len(va)} ===", flush=True)
        ft0 = time.time()
        model, ema = train_on_cache(
            cache, cmask, y, w, tr, epochs, backbone_name, unfreeze_last,
            lr, batch, n_groups, va_idx=va, y_gold_va=ds.y_raw[idx[va]])
        if len(va):
            pred = predict_groups(model, cache, cmask, va, n_groups, DEVICE)
            golds = ds.y_raw[idx[va]]
            mac, per = macro_auc(golds, pred)
            print(f"fold {fold} val macro-AUC={mac:.4f}", flush=True)
        fold_times.append(time.time() - ft0)
        ckpt = os.path.join(out_dir, f"fold{fold}.pt")
        state = ema.state_dict() if ema else model.state_dict()
        torch.save({"state_dict": state, "backbone": backbone_name,
                    "unfreeze_last": unfreeze_last, "n_groups": n_groups,
                    "img_size": IMG_SIZE}, ckpt)
        models.append(model)
        print(f"fold {fold} done in {fold_times[-1]:.0f}s -> {ckpt}", flush=True)

    del cache, cmask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    uids, preds = infer_test_ensemble(models, n_groups, DEVICE)
    _write_submission(uids, preds, out_csv)
    print(f"submit finished in {(time.time() - t0) / 60:.1f} min, "
          f"{len(models)} folds", flush=True)
    return out_csv


def main():
    res = run_cv("tiny_cnn", unfreeze_last=2)
    print(f"\nOOF macro-AUC = {res['macro']:.4f}  "
          f"(mean fold = {np.mean(res['fold_scores']):.4f})")
    out = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "oof_preds.npy"), res["oof"])
    print("saved OOF ->", out)


# ===================== raptor.py =====================
"""Study-window MIL over a high-resolution timm backbone (the "R" line).

Why a second architecture at all: the six-slot DINOv2 line saturated at 0.855
+/- 0.005 and every imaging-side change from v8 to v14 landed inside the 0.004
run-to-run noise (see LOG.md). The public leaderboard median is 0.898, and the
scored public pipeline documents a SINGLE CoAtNet-384 arm at 0.914. The gap is
therefore in the representation and the input resolution, not in how the slots
are carved -- which is exactly what this line changes.

Differences from the DINOv2 line, deliberately:
  * 384 px input instead of 168, on an ImageNet-12k CoAtNet instead of DINOv2-S.
  * No slot structure and no anatomical prior. A study becomes one bag of 2.5D
    windows and a per-target attention decides which windows matter, so the
    architecture cannot be blamed on our slot bookkeeping.
  * Plain weighted BCE, no pairwise ranking term and no EMA, matching the
    reference arm. Those two are re-addable as separate single-variable runs.

Everything upstream of the pixels -- physical crop, percentile normalisation,
laterality, spatial sort, the report-derived targets and their weights -- is
reused unchanged from the existing modules, so this line inherits the parts the
DINOv2 experiments already validated.
"""
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn

from config import (LABELS, N_LABELS, CROP_MM, RAPTOR_ARCH, RAPTOR_RES,
                    RAPTOR_SLOTS, RAPTOR_SLICES_PER_SERIES, RAPTOR_K_TRAIN,
                    RAPTOR_K_EVAL, RAPTOR_WEIGHTS_NAME, RAPTOR_DROP,
                    LAT_MIN_AGREEMENT, SLOTS, RAPTOR_EVAL_SLICES_PER_SERIES,
                    RAPTOR_USE_SWA, RAPTOR_SWA_START_FRAC, RAPTOR_ENCODE_CHUNK)
import dicom_utils as du
import model as M

R_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet statistics: the backbone was pretrained with them, and a grayscale MR
# slice replicated into three channels still has to arrive on that scale.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def resolve_timm_weights(name: str = RAPTOR_WEIGHTS_NAME) -> str | None:
    """Find the offline timm checkpoint among the mounted Kaggle datasets.

    A code competition runs with the internet off, so timm cannot fetch from the
    hub: the state_dict is uploaded as a dataset and loaded by hand.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [os.path.join(here, "data", "timm_weights", name)]
    for root in ("/kaggle/input",):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, files in os.walk(root):
            if dirpath[len(root):].count(os.sep) > 4:
                dirnames[:] = []
                continue
            if name in files:
                cands.append(os.path.join(dirpath, name))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
# Each arch needs its own offline checkpoint: a code competition has no internet, so
# timm cannot fetch weights. RadImageNet ResNet-50 keys arrive torchvision-style under
# "backbone.N" and are remapped once, offline, by scratchpad/remap (see LOG).
ARCH_WEIGHTS = {
    "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k": "coatnet_rmlp_2_rw_384_in12k_ft_in1k.pt",
    "resnet50": "resnet50_radimagenet.pt",
    "vit_base_patch14_dinov2": "dinov2_vitb14.pth",   # official facebookresearch release
    # C10 (LOG #469): timm hub state_dict, saved offline with num_classes=0
    "convnext_small.fb_in22k_ft_in1k_384": "convnext_small_in22k_ft_in1k_384.pt",
}


def build_raptor_backbone(arch: str = RAPTOR_ARCH):
    # The stub only matters where torchvision is broken (ROCm wheels); everywhere
    # else it is a no-op. It must never be the reason a kernel dies -- adding it
    # for the R9700 box is exactly what broke the first C5 inference kernel, whose
    # single-file build had no tv_stub module to import.
    try:
        import tv_stub
        tv_stub.install()
    except ImportError:
        pass
    import timm
    # CoAtNet/MaxViT are conv-attention hybrids: no CLS token and no
    # interpolatable position embedding, so they pool by average. The "vit"
    # substring inside "coatnet" must not route them down the ViT path.
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "eva", "beit"))
    kw = dict(pretrained=False, num_classes=0, in_chans=3)
    kw["global_pool"] = "token" if is_vit else "avg"
    if is_vit:
        kw["dynamic_img_size"] = True
        # patch-14 ViTs cannot tile 384 (384 % 14 == 6) and timm asserts on it. Pad to
        # 392 inside the patch embed instead of changing the input resolution, so a ViT
        # arm sees the same 384 windows as the CoAtNet arm and the ensemble geometry
        # check (one res across members) still holds.
        kw["dynamic_img_pad"] = True
    bb = timm.create_model(arch, **kw)
    path = resolve_timm_weights(ARCH_WEIGHTS.get(arch, RAPTOR_WEIGHTS_NAME))
    if path:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        miss, unexp = bb.load_state_dict(sd, strict=False)
        print(f"[{arch}] loaded {path} | missing={len(miss)} unexpected={len(unexp)}",
              flush=True)
        if len(miss) > 8:
            print("   first missing:", list(miss)[:8], flush=True)
    else:
        print(f"[{arch}] WARNING: no offline weights found -> RANDOM INIT", flush=True)
    return bb


class RaptorClassifier(nn.Module):
    """Per-target attention pooling over a study's windows.

    One attention distribution per finding rather than one shared distribution:
    an ACL tear and a Baker's cyst are visible on different windows, and a single
    pooled study vector would average the evidence away.
    """

    def __init__(self, backbone: nn.Module, f_dim: int, n_out: int = N_LABELS,
                 drop: float = RAPTOR_DROP):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(nn.Linear(f_dim, 256), nn.Tanh(),
                                 nn.Dropout(drop), nn.Linear(256, n_out))
        self.cls_w = nn.Parameter(torch.zeros(n_out, f_dim))
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        nn.init.trunc_normal_(self.cls_w, std=0.02)
        # RQ2 control (LOG #482): "mean" replaces the per-diagnosis attention with one
        # global average over the bag's windows (the att layer then gets no gradient).
        # Read at construction so a training script can set config.RAPTOR_POOL first.
        import config as _cfg
        self.pool = getattr(_cfg, "RAPTOR_POOL", "attention")
        assert self.pool in ("attention", "mean"), self.pool

    def encode(self, x, chunk: int | None = None):   # x [B, K, 3, H, W]
        b, k = x.shape[:2]
        flat = x.flatten(0, 1)                # [B*K, 3, H, W]
        n = flat.shape[0]
        step = chunk or n
        if step >= n:
            f = self.backbone(flat)
        else:
            # Sub-batch the flattened image dimension so a single forward call's
            # peak activation memory is bounded by `chunk`, not by B*K -- a study
            # with a big window bag (dense eval) must not scale the batch that
            # actually hits the backbone. Training never needs this (B*K stays
            # small there), so callers that never pass `chunk` keep the old,
            # single-call behaviour exactly.
            f = torch.cat([self.backbone(flat[i:i + step])
                          for i in range(0, n, step)], dim=0)
        return f.view(b, k, -1)

    def head(self, feats, kmask=None):        # feats [B, K, F]
        h = self.norm(feats)
        if self.pool == "mean":
            a = torch.zeros(h.shape[0], h.shape[1], self.cls_b.shape[0],
                            device=h.device, dtype=h.dtype)
        else:
            a = self.att(h)                   # [B, K, n_out]
        if kmask is not None:
            a = a.masked_fill(kmask.unsqueeze(-1) < 0.5, -1e4)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def forward(self, x, kmask=None, chunk: int | None = None):
        return self.head(self.encode(x, chunk=chunk), kmask)


def build_raptor_model(arch: str = RAPTOR_ARCH):
    bb = build_raptor_backbone(arch)
    return RaptorClassifier(bb, f_dim=bb.num_features)


# --------------------------------------------------------------------------- #
# study volume cache
# --------------------------------------------------------------------------- #
def plan_slices(n_study: int, budget_gb: float) -> int:
    """Slices per series that fit the uint8 cache budget at RAPTOR_RES."""
    per = n_study * len(RAPTOR_SLOTS) * RAPTOR_RES * RAPTOR_RES
    n = RAPTOR_SLICES_PER_SERIES
    while n > 3 and n * per > budget_gb * (1024 ** 3):
        n -= 1
    return n


def build_window_cache(ds, indices, budget_gb: float, n_workers: int = 8,
                       force_n_per: int | None = None):
    """Decode each study's RAPTOR_SLOTS series once into a uint8 window cache.

    Returns (cache [N, S, R, R] uint8, present [N, n_series] float32, n_per).
    Slices are stored per series back to back, so a 2.5D window never straddles
    two series -- a triplet spanning a contrast change is not a 2.5D view of
    anything.

    `force_n_per` bypasses the memory-budget planner: the train cache has to hold
    every study at once for repeated-epoch access, so it is budget-bound, but a
    code-competition TEST split is a handful to a few hundred studies and can
    afford far more slices per series without touching RAPTOR_CACHE_BUDGET_GB.
    """
    from concurrent.futures import ThreadPoolExecutor

    indices = np.asarray(indices)
    n = len(indices)
    n_per = force_n_per if force_n_per is not None else plan_slices(n, budget_gb)
    n_series = len(RAPTOR_SLOTS)
    cache = np.zeros((n, n_series * n_per, RAPTOR_RES, RAPTOR_RES), np.uint8)
    present = np.zeros((n, n_series), np.float32)
    print(f"raptor cache: N={n} series={n_series} slices/series={n_per} "
          f"shape={cache.shape} = {cache.nbytes / 1024 ** 3:.2f} GB", flush=True)

    jobs, tag_pos = [], {}
    for li, gi in enumerate(indices):
        uid = ds.uids[int(gi)]
        slots = ds._slots_for_study(uid)
        for pos, si in enumerate(RAPTOR_SLOTS):
            folder = slots.get(si)
            if folder:
                jobs.append((li, pos, SLOTS[si][0], folder))
                tag_pos.setdefault(folder, None)

    def _lat(folder):
        return folder, du.read_laterality_full(folder)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for folder, tp in ex.map(_lat, list(tag_pos)):
            tag_pos[folder] = tp
    agree = [(t == p) for t, p in tag_pos.values() if t and p]
    use_fb = bool(agree) and (sum(agree) / len(agree)) >= LAT_MIN_AGREEMENT
    print(f"raptor laterality: {len(tag_pos)} series, "
          f"agree {(sum(agree) / len(agree)) if agree else float('nan'):.3f} "
          f"on {len(agree)} -> fallback {'ON' if use_fb else 'OFF'}", flush=True)
    sides = {f: (t or (p if use_fb else None)) for f, (t, p) in tag_pos.items()}

    def _decode(job):
        li, pos, plane, folder = job
        vol = du.load_series_stack(folder, plane, sides.get(folder), n_per,
                                   RAPTOR_RES, CROP_MM)
        return li, pos, (vol * 255.0).clip(0, 255).astype(np.uint8)

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for li, pos, vol in ex.map(_decode, jobs):
            cache[li, pos * n_per:(pos + 1) * n_per] = vol
            present[li, pos] = 1.0
            done += 1
            if done % 2000 == 0:
                print(f"  decoded {done}/{len(jobs)} slot-series", flush=True)
    print(f"raptor cache ready: {int(present.sum())} present series / "
          f"{n * n_series}", flush=True)
    return cache, present, n_per


def window_centers(n_per: int):
    """Local centre indices inside one series block (a triplet needs both sides)."""
    return list(range(1, max(2, n_per - 1)))


def take_windows(rows: np.ndarray, present: np.ndarray, n_per: int, picks):
    """rows [B, S, R, R] uint8 -> windows [B, K, 3, R, R] float32 in [0, 1].

    `picks` is a list of (series_pos, centre) per bag element; a pick on an
    absent series yields zeros and is masked out of the attention.
    """
    b = rows.shape[0]
    k = len(picks)
    out = np.zeros((b, k, 3, rows.shape[-2], rows.shape[-1]), np.float32)
    kmask = np.zeros((b, k), np.float32)
    for j, (pos, c) in enumerate(picks):
        base = pos * n_per
        tri = rows[:, [base + c - 1, base + c, base + c + 1]]
        out[:, j] = tri.astype(np.float32) / 255.0
        kmask[:, j] = present[:, pos]
    return out, kmask


def sample_picks(n_per: int, n_series: int, k: int, rng: np.random.RandomState):
    """k random (series, centre) picks, spread over series before repeating one."""
    cs = window_centers(n_per)
    pool = [(p, c) for p in range(n_series) for c in cs]
    if not pool:
        return [(0, 1)] * k
    if k >= len(pool):
        return pool
    sel = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in sorted(sel)]


def eval_picks(n_per: int, n_series: int, k: int):
    """Deterministic, evenly spread picks for inference."""
    cs = window_centers(n_per)
    pool = [(p, c) for p in range(n_series) for c in cs]
    if len(pool) <= k:
        return pool
    idx = np.linspace(0, len(pool) - 1, k).round().astype(int)
    return [pool[i] for i in idx]


def to_tensor(x, kmask, device, do_aug=False):
    t = torch.from_numpy(x).to(device)
    if do_aug:
        t = M.augment(t)
    b, k = t.shape[:2]
    t = (t - _MEAN.to(device)) / _STD.to(device) if t.shape[2] == 3 else t
    return t, torch.from_numpy(kmask).to(device)


@torch.no_grad()
def predict_windows(model, cache, present, idx, n_per, device, batch=2,
                    k_eval: int = RAPTOR_K_EVAL):
    model.eval()
    picks = eval_picks(n_per, present.shape[1], k_eval)
    out = []
    idx = np.asarray(idx)
    for b0 in range(0, len(idx), batch):
        sel = idx[b0:b0 + batch]
        x, km = take_windows(cache[sel], present[sel], n_per, picks)
        xt, kmt = to_tensor(x, km, device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            # A dense eval pool (R2: up to 18 centres x 6 slots = 108 windows/study)
            # would otherwise flatten to batch*108 images in one backbone call --
            # this is what OOM'd on R2's first run. Chunk regardless of pool size.
            z = model(xt, kmt, chunk=RAPTOR_ENCODE_CHUNK).float()
        out.append(torch.sigmoid(z).cpu().numpy())
    return (np.concatenate(out, 0) if out
            else np.zeros((0, N_LABELS), np.float32))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _recalibrate_bn(model, cache, present, idx, n_per, n_series, device, rng,
                    n_batches: int = 30, batch: int = RAPTOR_K_TRAIN):
    """Reset BatchNorm running stats, then re-accumulate them with cumulative
    averaging (momentum=None) over forward-only passes -- the same recipe as
    torch.optim.swa_utils.update_bn, hand-rolled because our forward signature
    takes (imgs, kmask) rather than the single positional input update_bn expects.

    SWA averages WEIGHTS across epochs, but a BatchNorm layer's running mean/var
    were accumulated for the trajectory of weights that produced them, not for the
    averaged weights the model now has; skipping this step is a common SWA bug
    that silently degrades an averaged model at eval time.
    """
    bn_layers = [m for m in model.modules()
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not bn_layers:
        return
    old_momenta = {m: m.momentum for m in bn_layers}
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None
    was_training = model.training
    model.train()
    idx = np.asarray(idx)
    with torch.no_grad():
        for _ in range(n_batches):
            sel = idx[rng.choice(len(idx), size=min(batch, len(idx)), replace=False)]
            picks = sample_picks(n_per, n_series, RAPTOR_K_TRAIN, rng)
            x, km = take_windows(cache[sel], present[sel], n_per, picks)
            xt, kmt = to_tensor(x, km, device, do_aug=False)
            model(xt, kmt)
    for m in bn_layers:
        m.momentum = old_momenta[m]
    model.train(was_training)


def train_windows(cache, present, y, w, tr_idx, va_idx, y_gold_va, epochs,
                  n_per, batch, device=None, seed: int = 42, time_budget_s=None):
    """One model on the window cache. Weighted BCE only -- no rank term.

    The rank term is re-addable as a separate single-variable run; the reference
    arm this line follows has neither, and starting a new architecture with our
    DINOv2 line's extras bolted on would make a disappointing result unattributable.
    SWA is the one addition kept from day one (config.RAPTOR_USE_SWA): it is a
    documented feature of the reference arm's own recipe ("_swa" checkpoint name),
    not a DINOv2-line import, and unlike EMA it needs no extra forward pass during
    training -- only a cheap BN recalibration once at the end.
    """
    from train import macro_auc

    device = device or R_DEVICE
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model = build_raptor_model().to(device)
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.")]
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": bb_params, "lr": RAPTOR_LR_BACKBONE},
         {"params": head_params, "lr": RAPTOR_LR_HEAD}], weight_decay=0.02)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    n_series = present.shape[1]

    tr_idx = np.asarray(tr_idx)
    steps_per = max(len(tr_idx) // batch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[RAPTOR_LR_BACKBONE, RAPTOR_LR_HEAD],
        total_steps=max(epochs * steps_per, 1), pct_start=0.15)

    do_sel = va_idx is not None and len(va_idx) > 0
    if do_sel:
        va_idx = np.asarray(va_idx)
        yd = y[va_idx].astype(float).copy()
        yd[w[va_idx] <= 0] = np.nan
        yd_bin = np.where(np.isfinite(yd), (yd > 0.5).astype(float), np.nan)
    best_score, best_state = -1.0, None
    swa_state, swa_n = None, 0
    swa_start_ep = int(round(epochs * RAPTOR_SWA_START_FRAC)) if RAPTOR_USE_SWA else epochs
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr_idx))
        loss_sum, nb = 0.0, 0
        for b0 in range(0, len(tr_idx) - batch + 1, batch):
            sel = tr_idx[order[b0:b0 + batch]]
            picks = sample_picks(n_per, n_series, RAPTOR_K_TRAIN, rng)
            x, km = take_windows(cache[sel], present[sel], n_per, picks)
            xt, kmt = to_tensor(x, km, device, do_aug=True)
            yt = torch.from_numpy(y[sel]).to(device)
            wt = torch.from_numpy(w[sel]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xt, kmt)
                loss = (bce(logits, yt) * wt).sum() / wt.sum().clamp(min=1.0)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += float(loss.item())
            nb += 1
        msg = f"    epoch {ep + 1}/{epochs}  loss={loss_sum / max(nb, 1):.4f}"
        if do_sel:
            pred = predict_windows(model, cache, present, va_idx, n_per, device)
            der, _ = macro_auc(yd_bin, pred)
            gld, _ = macro_auc(y_gold_va, pred) if y_gold_va is not None else (float("nan"), None)
            msg += f"  der={der:.4f}"
            if np.isfinite(gld):
                msg += f"  gold={gld:.4f}"
            if np.isfinite(der) and der > best_score:
                best_score = der
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        if RAPTOR_USE_SWA and ep >= swa_start_ep:
            cur = model.state_dict()
            if swa_state is None:
                swa_state = {k: v.detach().clone().float()
                             for k, v in cur.items() if v.dtype.is_floating_point}
            else:
                swa_n += 1
                for k, v in swa_state.items():
                    v += (cur[k].detach().float() - v) / (swa_n + 1)
            msg += f"  [swa n={swa_n + (1 if swa_state is not None else 0)}]"
        msg += f"  [{(time.time() - t0) / 60:.1f}m]"
        print(msg, flush=True)
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"    time budget hit after epoch {ep + 1}", flush=True)
            break

    if swa_state is not None and do_sel:
        # Load the averaged weights, restoring the non-float buffers (e.g. BN's
        # num_batches_tracked, an int) from whichever state the run ends on -- only
        # the running mean/var need recomputing, and _recalibrate_bn does that next.
        swa_full = model.state_dict()
        swa_full.update({k: v.to(swa_full[k].dtype) for k, v in swa_state.items()})
        model.load_state_dict(swa_full)
        _recalibrate_bn(model, cache, present, tr_idx, n_per, n_series, device, rng)
        pred = predict_windows(model, cache, present, va_idx, n_per, device)
        der, _ = macro_auc(yd_bin, pred)
        print(f"    SWA holdout der={der:.4f} (best single epoch was {best_score:.4f})",
              flush=True)
        if np.isfinite(der) and der > best_score:
            best_score = der
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        # else fall through: best_state (a single epoch) still wins below.

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def run_raptor_submit(max_studies: int | None = None, epochs: int = None,
                      batch: int = None, out_dir: str = ".",
                      budget_gb: float = None, seed: int = 42,
                      time_budget_h: float = 6.0, do_submit: bool = True):
    """Train one study-window model on a grouped 80/20 split and predict the test set.

    A single arm, not a fold ensemble: the DINOv2 line showed 8 folds scored the
    same as 4 (LOG.md v12), and at 384 px one arm is what the runtime buys. The
    published CoAtNet arm this follows is also a single split.
    """
    from config import (RAPTOR_EPOCHS, RAPTOR_BATCH, RAPTOR_CACHE_BUDGET_GB)
    from dataset import RSNADataset
    from train import _report_hash, _subset_indices, _write_submission, macro_auc
    from sklearn.model_selection import GroupShuffleSplit

    epochs = epochs or RAPTOR_EPOCHS
    batch = batch or RAPTOR_BATCH
    budget_gb = budget_gb or RAPTOR_CACHE_BUDGET_GB
    t0 = time.time()

    ds = RSNADataset("train")
    idx = _subset_indices(ds, max_studies)
    groups = np.array([_report_hash(ds.reports[i]) for i in idx])
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    tr, va = next(gss.split(idx, groups=groups))
    n_gold = int(np.isfinite(ds.y_raw[idx[va]]).any(axis=1).sum())
    print(f"raptor: {len(idx)} studies | train={len(tr)} holdout={len(va)} "
          f"(gold in holdout={n_gold}) | arch={RAPTOR_ARCH} res={RAPTOR_RES} "
          f"ep={epochs} batch={batch} k_train={RAPTOR_K_TRAIN}", flush=True)

    cache, present, n_per = build_window_cache(ds, idx, budget_gb)
    y, w = ds.targets[idx], ds.weights[idx]
    spent = (time.time() - t0) / 3600.0
    print(f"raptor: cache built in {spent * 60:.1f} min", flush=True)

    model, best = train_windows(
        cache, present, y, w, tr, va, ds.y_raw[idx[va]], epochs, n_per, batch,
        seed=seed, time_budget_s=max(600.0, (time_budget_h - spent) * 3600.0))
    print(f"raptor: best holdout derived-AUC={best:.4f}", flush=True)

    ckpt = os.path.join(out_dir, "raptor0.pt")
    torch.save({"state_dict": model.state_dict(), "arch": RAPTOR_ARCH,
                "res": RAPTOR_RES, "n_per": n_per, "slots": list(RAPTOR_SLOTS)},
               ckpt)
    print("saved", ckpt, flush=True)

    del cache, present
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if do_submit:
        test = RSNADataset("test")
        t_idx = np.arange(len(test.uids))
        # The test split is small enough that memory is not the constraint here, so
        # it decodes far more slices per series than the train cache affords, and
        # RAPTOR_K_EVAL=999 (see config.py) then votes over essentially all of them.
        tc, tp, tn = build_window_cache(test, t_idx, budget_gb,
                                        force_n_per=RAPTOR_EVAL_SLICES_PER_SERIES)
        preds = predict_windows(model, tc, tp, t_idx, tn, R_DEVICE)
        _write_submission(test.uids, preds, os.path.join(out_dir, "submission.csv"))
    print(f"raptor finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    return model

# ===================== corpus.py =====================
"""Pre-decoded study corpus (memmap) + a matching test-time volume builder.

The R line decoded DICOMs into an in-RAM cache on every run, which cost 48-72
minutes per kernel AND imposed a memory ceiling that forced the depth-vs-breadth
tradeoff R2/R3 measured (LOG.md). The reference pipeline has neither problem
because it decodes the corpus ONCE, offline, into a uint8 memmap published as
two public Kaggle datasets, and streams per-study reads from disk:

    dreaddevelopment/knee-raptor-corpus       all_vols.npy    15.9 GB
    dreaddevelopment/knee-raptor-corpus-ext   extra_vols.npy   6.0 GB
    -> one (4407, 44, 336, 336) uint8 array, 100% of our train.csv studies
    C4 64-slice shards (self-built, same crop/span/percentile, more slices):
    easonyanyan/rsna-knee-corpus64-p0 / p1    vols64_p{0,1}.npy  ~15.9 GB each
    -> one (4407, 64, 336, 336) uint8 array

CRITICAL: the corpus covers TRAIN only. A code competition's hidden test set has
to be decoded at inference time, and it must be decoded EXACTLY the way the
corpus was, or the model sees a different distribution than it trained on. The
constants below were recovered from the reference's own inference notebook plus
the corpus masks, and `build_test_volume` reproduces that pipeline:

  * 5 slots. C1-C3 used [12, 10, 8, 6, 8] = 44 (the published memmap). C4 uses
    the 64-slice layout [18, 14, 12, 8, 12] documented by the 0.924 inference
    kernel -- same proportions, more coverage. Test-time volumes MUST use the
    same layout the model trained on.
  * span 0.06-0.94 of each series, NOT a central crop. Their comment: the
    collateral ligaments and lateral meniscus live in the peripheral slices a
    central crop throws away. (Our own pipeline uses the central 60%.)
  * per-SLOT percentile 2-98 over the picked slices, not per-volume 1-99.
  * centre crop CROP_MM=140 at the series' pixel spacing, resize to 336.
  * NO laterality normalisation. The reference simply does not flip; our DINOv2
    line spent real effort on tag+IPP laterality. Matching the corpus means not
    flipping either -- a deliberate deviation, flagged for a later A/B.
"""
import glob
import json
import os

import numpy as np

# ---- corpus geometry --------------------------------------------------------
# C1-C3 trained on the published 44-slice memmap (12/10/8/6/8). The 0.924
# single-arm weights were trained on 64 slices at the same proportions
# (18/14/12/8/12) -- that corpus was never published, only the checkpoint.
# C4 rebuilds it with the same crop/span/percentile rules, only n_slices
# changes. 4407 x 64 x 336 x 336 uint8 = 31.8 GB, so it is dumped as TWO
# shards (~15.9 GB each) to fit Kaggle's 20 GB working dir, then mounted
# like the original two-part 44-slice corpus.
CORPUS_IMG = 336
CORPUS_CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.06, 0.94
PCT_LO, PCT_HI = 2.0, 98.0
# (plane, fluid_flag, n_slices); fluid_flag -1 means "any series in this plane"
CORPUS_SLOTS_44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
                   ("Coronal", 0, 6), ("Axial", -1, 8)]
CORPUS_SLOTS_64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
                   ("Coronal", 0, 8), ("Axial", -1, 12)]
# Which geometry this checkout is pointed at. RSNA_CORPUS_SLOTS=44|64 overrides it,
# so the two lines can share one file: C4/C5 run at 64 (own dumped shards), C2/C3/C6
# at 44 (the published memmap). Getting it wrong is NOT silent -- open_corpus below
# hard-exits when the memmap's depth disagrees with CORPUS_D.
_SLOT_CHOICE = os.environ.get("RSNA_CORPUS_SLOTS", "44")
CORPUS_SLOTS = CORPUS_SLOTS_64 if _SLOT_CHOICE == "64" else CORPUS_SLOTS_44
CORPUS_D = sum(s[2] for s in CORPUS_SLOTS)
assert CORPUS_D in (44, 64)
try:
    from config import CORPUS_LEGAL_WINDOWS as LEGAL_WINDOWS
except Exception:          # corpus.py used without the project config
    LEGAL_WINDOWS = False


def _find(*names, root=None):
    """First match for any of `names` under corpus search roots."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots = []
    if root:
        roots.append(root)
    env = os.environ.get("RSNA_CORPUS_DIR")
    if env:
        roots.append(env)
    roots.append(os.path.join(here, "data", "corpus64"))
    if os.path.isdir("/kaggle/input"):
        roots.append("/kaggle/input")
    for name in names:
        for r in roots:
            if not os.path.isdir(r):
                continue
            direct = os.path.join(r, name)
            if os.path.isfile(direct):
                return direct
            for dirpath, dirnames, files in os.walk(r):
                if dirpath[len(r):].count(os.sep) > 6:
                    dirnames[:] = []
                    continue
                if "competition" in dirpath.lower():
                    continue
                if name in files:
                    return os.path.join(dirpath, name)
    return None


class TwoPartVols:
    """Two memmap parts as one (N, D, H, W) array.

    Both stay memory-mapped and are never concatenated: every read is a single
    study, so materialising 32 GB would buy nothing. Row order is part 0 then
    part 1, matching how the id and mask files concatenate -- that ordering is
    the contract between volumes, masks and ids, so none may be sorted alone.
    """

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


def open_corpus():
    """Return (vols, masks [N,D] uint8, ids [N] str) or raise with what to attach.

    Prefers the C4 64-slice shards (vols64_p0/p1.npy). Falls back to the
    published 44-slice memmap only when CORPUS_D is 44, so a 64-slice train
    kernel cannot silently read the wrong geometry.
    """
    a = _find("vols64_p0.npy")
    b = _find("vols64_p1.npy")
    if a and b:
        am, bm = _find("masks64_p0.npy"), _find("masks64_p1.npy")
        ai, bi = _find("ids64_p0.npy"), _find("ids64_p1.npy")
        if not all((am, bm, ai, bi)):
            raise SystemExit(
                "64-slice vols found but masks/ids missing\n"
                f"  masks {am} {bm} | ids {ai} {bi}")
        vols = TwoPartVols(np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r"))
        masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
        ids = np.concatenate([np.load(ai, allow_pickle=True).astype(str),
                              np.load(bi, allow_pickle=True).astype(str)])
    else:
        av, bv = _find("all_vols.npy"), _find("extra_vols.npy")
        am, bm = _find("all_masks.npy"), _find("extra_masks.npy")
        ai, bi = _find("all_ids.npy"), _find("extra_ids.npy")
        if not all((av, bv, am, bm, ai, bi)):
            raise SystemExit(
                "corpus not found -- attach the 64-slice shards "
                "(vols64_p0.npy / vols64_p1.npy) or the published 44-slice "
                "dreaddevelopment/knee-raptor-corpus + -ext\n"
                f"  vols64_p0={a} vols64_p1={b} all_vols={av} extra_vols={bv}")
        vols = TwoPartVols(np.load(av, mmap_mode="r"), np.load(bv, mmap_mode="r"))
        masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
        ids = np.concatenate([np.load(ai, allow_pickle=True).astype(str),
                              np.load(bi, allow_pickle=True).astype(str)])
    if vols.shape[1] != CORPUS_D:
        raise SystemExit(
            f"corpus depth {vols.shape[1]} != CORPUS_D {CORPUS_D} -- "
            f"mounted the wrong slice layout")
    print(f"corpus: {vols.shape} uint8 memmap | masks {masks.shape} | ids {len(ids)}",
          flush=True)
    return vols, masks, ids



def _specialist_meta():
    """Window policy + trained labels recorded in every checkpoint (LOG #490), so inference
    can restore them; both are None for every run before the diagnosis specialist."""
    import config as _c
    allow = getattr(_c, "CORPUS_CENTRE_ALLOW", None)
    labs = getattr(_c, "CORPUS_TRAIN_LABELS", None)
    return {"centre_allow": sorted(int(c) for c in allow) if allow else None,
            "train_labels": list(labs) if labs else None}

def centers_from_mask(mask, k: int, train: bool, rng=None, legal=None):
    """`k` window centres over the valid span; a centre needs both neighbours.

    Training samples with replacement from the legal set (so every window is a
    fresh draw across epochs); evaluation spreads deterministically. Note this
    follows the reference exactly, including that a window MAY straddle a slot
    boundary: legality is judged on the valid span [lo, hi], not on which slot a
    position belongs to. Our own R line forbade straddling; matching the recipe
    that scored 0.914 comes first, and the no-straddle variant is a later A/B.

    legal=True (or CORPUS_LEGAL_WINDOWS when legal is None) keeps only centres whose
    rows c-1, c, c+1 are all non-empty and in the same slot. The audit found 12.9% of
    training windows straddle a slot (half of them a plane change) and 3.0% are all
    zero, and no attention mask is passed (LOG #377). If a study has no such centre
    the reference set is used unchanged.
    """
    valid = np.where(np.asarray(mask) > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, CORPUS_D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, CORPUS_D - 2))]
    if (LEGAL_WINDOWS if legal is None else legal):
        m = np.asarray(mask) > 0
        slot_of = np.repeat(np.arange(len(CORPUS_SLOTS)), [s[2] for s in CORPUS_SLOTS])
        strict = [c for c in cs if m[c - 1] and m[c] and m[c + 1]
                  and slot_of[c - 1] == slot_of[c + 1]]
        if strict:
            cs = strict
    # Diagnosis specialist (LOG #490): keep only the allowed centres. Read at call time so
    # a script can set config.CORPUS_CENTRE_ALLOW after this module was imported.
    import config as _cfg_c
    _allow = getattr(_cfg_c, "CORPUS_CENTRE_ALLOW", None)
    if _allow:
        _kept = [c for c in cs if c in set(_allow)]
        if _kept:
            cs = _kept
    if train:
        rng = rng or np.random
        pool = (cs * (k // len(cs) + 1))
        rng.shuffle(pool)
        return pool[:k]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def read_windows(vol_row, centers, res: int):
    """(K, 3, res, res) float32 in [0,1] from one study's (D, 336, 336) uint8 row.

    A window is three physically consecutive slices stacked as RGB, matching the
    reference's 2.5D definition.
    """
    import torch
    import torch.nn.functional as F

    vol = np.asarray(vol_row)
    d = vol.shape[0]
    out = []
    for c in centers:
        c = max(1, min(int(c), d - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        out.append(t)
    return torch.stack(out, 0)


# --------------------------------------------------------------------------- #
# test-time volume construction (must mirror how the corpus was built)
# --------------------------------------------------------------------------- #
def _pick_series_for_slot(rows, plane, fluid, used):
    """Prefer the requested fluid flag, but fall back to ANY unused series in the
    plane rather than leaving the slot empty -- which is why the reference's slot
    coverage reaches 100% on the fluid slots where our exact matching got 94-96%."""
    cands = [r for r in rows
             if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _order_series_files(folder):
    """(sorted file paths, median pixel spacing) for one series, ordered through-plane."""
    import pydicom

    files = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not files:
        return [], 1.0
    keyed, spacings = [], []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            # project the slice origin onto the slice normal: the true
            # through-plane coordinate, correct for oblique acquisitions where a
            # single fixed axis is not.
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        try:
            ps = float(ds.PixelSpacing[0])
        except Exception:
            ps = None
        if ps:
            spacings.append(ps)
        keyed.append((k, f))
    med_ps = float(np.median(spacings)) if spacings else 1.0
    if any(k is None for k, _ in keyed):
        return [f for _, f in keyed], med_ps      # arbitrary order beats dropping
    return [f for _, f in sorted(keyed, key=lambda t: t[0])], med_ps


def _read_px(path):
    import pydicom

    ds = pydicom.dcmread(path, force=True)
    a = ds.pixel_array.astype(np.float32)
    a = a * float(getattr(ds, "RescaleSlope", 1) or 1) \
        + float(getattr(ds, "RescaleIntercept", 0) or 0)
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = 0.0
    return a, ps


def _crop_resize(a, ps):
    """Centre-crop CORPUS_CROP_MM of tissue at this slice's spacing, then resize."""
    import cv2

    h, w = a.shape
    cpx = int(round(CORPUS_CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(a, (CORPUS_IMG, CORPUS_IMG), interpolation=cv2.INTER_AREA)


def build_test_volume(sid, rows, series_root):
    """(vol [D,336,336] uint8, mask [D] uint8) for one study, corpus-identical."""
    vol = np.zeros((CORPUS_D, CORPUS_IMG, CORPUS_IMG), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in CORPUS_SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = _order_series_files(
            os.path.join(series_root, sid, r["SeriesInstanceUID"]))
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * SPAN_LO), int(n * SPAN_HI) - 1
        hi = max(hi, lo)
        picks = (np.linspace(lo, hi, k).round().astype(int) if n > 1
                 else np.zeros(k, dtype=int))
        arrs, pss = [], []
        for p in picks:
            try:
                a, ps = _read_px(files[min(int(p), n - 1)])
            except Exception:
                a, ps = None, med_ps
            arrs.append(a)
            pss.append(ps if ps else med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]),
                                     [PCT_LO, PCT_HI])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= CORPUS_D:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            vol[idx] = (_crop_resize(aw, ps) * 255).astype(np.uint8)
            idx += 1
        if idx >= CORPUS_D:
            break
    mask = (vol.reshape(CORPUS_D, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def build_test_corpus(test_ds, n_workers: int = 8):
    """(vols [N,D,336,336] uint8, masks [N,D], uids) for the test split.

    Small enough to hold in RAM: a hidden test set is a handful to a few hundred
    studies, so 64*336*336 = 7.2 MB each stays trivial next to the train corpus
    this mirrors.
    """
    from concurrent.futures import ThreadPoolExecutor
    import time

    t0 = time.time()
    uids = list(test_ds.uids)
    rows_by_study = {}
    for uid in uids:
        rows_by_study[uid] = [
            {"SeriesInstanceUID": sid, "Anatomical_Plane": plane,
             "Fluid_Sensitive": fl, "Fat_Suppression": fat}
            for sid, plane, fl, fat in test_ds.study_to_series.get(uid, [])]
    vols = np.zeros((len(uids), CORPUS_D, CORPUS_IMG, CORPUS_IMG), np.uint8)
    masks = np.zeros((len(uids), CORPUS_D), np.uint8)

    def _one(i):
        uid = uids[i]
        return i, build_test_volume(uid, rows_by_study[uid], test_ds.series_root)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for i, (v, m) in ex.map(_one, range(len(uids))):
            vols[i], masks[i] = v, m
    filled = int((masks.sum(1) > 0).sum())
    print(f"test corpus: {vols.shape} | {filled}/{len(uids)} studies with data | "
          f"mean valid slices {masks.sum(1).mean():.1f}/{CORPUS_D} | "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return vols, masks, uids


def shard_bounds(n, shard, n_shards):
    """Half-open [lo, hi) split of n items. Shards cover 0..n without overlap."""
    if not (0 <= shard < n_shards) or n_shards < 1:
        raise ValueError(f"shard={shard} n_shards={n_shards}")
    return shard * n // n_shards, (shard + 1) * n // n_shards


def run_corpus_dump(shard=0, n_shards=2, out_dir=".", n_workers=8):
    """Decode train DICOMs into one 64-slice uint8 shard on disk.

    4407 x 64 x 336 x 336 does not fit Kaggle's 20 GB working dir (~31.8 GB),
    so this writes one half (~15.9 GB) as vols64_p{shard}.npy + masks + ids.
    Two CPU kernels (no GPU quota) produce the pair; the C4 train kernel
    memmaps both. Incremental memmap writes so peak RAM is a handful of studies.
    """
    import json
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dataset import RSNADataset

    ds = RSNADataset("train", use_report=False)
    if not ds.series_root or not os.path.isdir(ds.series_root):
        raise SystemExit(f"train series root missing: {ds.series_root}")
    uids = [str(u) for u in ds.uids]
    lo, hi = shard_bounds(len(uids), shard, n_shards)
    part = uids[lo:hi]
    nbytes = len(part) * CORPUS_D * CORPUS_IMG * CORPUS_IMG
    print(f"dump shard {shard}/{n_shards} studies {lo}:{hi} n={len(part)} | "
          f"({len(part)}, {CORPUS_D}, {CORPUS_IMG}, {CORPUS_IMG}) | "
          f"{nbytes / 1024 ** 3:.2f} GB | slots {CORPUS_SLOTS}", flush=True)
    free = shutil.disk_usage(out_dir).free
    print(f"disk {out_dir} free {free / 1024 ** 3:.1f} GB", flush=True)
    if free < nbytes * 1.05:
        raise SystemExit(
            f"not enough disk for shard {shard}: need {nbytes / 1024 ** 3:.1f} GB")

    rows_by_study = {}
    for uid in part:
        rows_by_study[uid] = [
            {"SeriesInstanceUID": sid, "Anatomical_Plane": plane,
             "Fluid_Sensitive": fl, "Fat_Suppression": fat}
            for sid, plane, fl, fat in ds.study_to_series.get(uid, [])]

    vol_path = os.path.join(out_dir, f"vols64_p{shard}.npy")
    mask_path = os.path.join(out_dir, f"masks64_p{shard}.npy")
    id_path = os.path.join(out_dir, f"ids64_p{shard}.npy")
    vols = np.lib.format.open_memmap(
        vol_path, mode="w+", dtype=np.uint8,
        shape=(len(part), CORPUS_D, CORPUS_IMG, CORPUS_IMG))
    masks = np.zeros((len(part), CORPUS_D), np.uint8)

    def _one(i):
        uid = part[i]
        v, m = build_test_volume(uid, rows_by_study[uid], ds.series_root)
        return i, v, m

    t0 = time.time()
    done, filled = 0, 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_one, i) for i in range(len(part))]
        for fut in as_completed(futs):
            i, v, m = fut.result()
            vols[i] = v
            masks[i] = m
            done += 1
            filled += int(m.sum() > 0)
            if done % 50 == 0 or done == len(part):
                print(f"  {done}/{len(part)}  filled {filled}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    vols.flush()
    del vols
    np.save(mask_path, masks)
    np.save(id_path, np.asarray(part, dtype=object))
    meta = {
        "shard": shard, "n_shards": n_shards, "lo": lo, "hi": hi,
        "n": len(part), "shape": [len(part), CORPUS_D, CORPUS_IMG, CORPUS_IMG],
        "slots": CORPUS_SLOTS, "crop_mm": CORPUS_CROP_MM,
        "span": [SPAN_LO, SPAN_HI], "pct": [PCT_LO, PCT_HI],
        "filled": int(filled),
        "mean_valid": float(masks.sum(1).mean()) if len(part) else 0.0,
    }
    with open(os.path.join(out_dir, f"manifest64_p{shard}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"dump done: {vol_path} | filled {filled}/{len(part)} | "
          f"mean valid {meta['mean_valid']:.1f}/{CORPUS_D} | "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return meta


# --------------------------------------------------------------------------- #
# training on the corpus (the reference arm's recipe)
# --------------------------------------------------------------------------- #
def build_param_groups(model, bb_lr, head_lr, llrd=None):
    """AdamW param groups plus the matching OneCycle max_lr list.

    llrd=None (the default) returns exactly the two groups every run through C8 used --
    whole backbone at bb_lr, head at head_lr, in that order -- so those runs reproduce.

    0 < llrd < 1 turns on layer-wise LR decay for ViT backbones (anything with
    `.blocks`): the patch/cls/pos embedding is depth 0, blocks[i] is depth i+1, the final
    norm is depth L+1, and a parameter at depth d trains at bb_lr * llrd ** (L + 1 - d).
    Added for C9b: the DINOv2 ViT-B arm collapsed in epoch 2, exactly when OneCycle
    reached its peak, while CoAtNet under the same schedule kept improving (LOG #394).
    """
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.")]
    bb = model.backbone
    if not llrd or not hasattr(bb, "blocks"):
        bb_params = [p for p in bb.parameters() if p.requires_grad]
        return ([{"params": bb_params, "lr": bb_lr},
                 {"params": head_params, "lr": head_lr}], [bb_lr, head_lr])
    n_blocks = len(bb.blocks)
    by_depth = {}
    for n, p in bb.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("blocks."):
            d = int(n.split(".")[1]) + 1
        elif n.startswith(("cls_token", "pos_embed", "reg_token", "mask_token", "patch_embed")):
            d = 0
        else:                                   # final norm / fc_norm
            d = n_blocks + 1
        by_depth.setdefault(d, []).append(p)
    groups, lrs = [], []
    for d in sorted(by_depth):
        lr = bb_lr * llrd ** (n_blocks + 1 - d)
        groups.append({"params": by_depth[d], "lr": lr})
        lrs.append(lr)
    groups.append({"params": head_params, "lr": head_lr})
    lrs.append(head_lr)
    return groups, lrs


def freeze_batchnorm_stats(model):
    """Put every BatchNorm layer in eval mode (running stats frozen); return the count.

    Must be called after each model.train(), which flips them back. The affine weight and
    bias keep requires_grad, so only the running mean/var are frozen. Experiment 4: each
    C-line step estimates batch statistics from 24 highly correlated windows of just two
    knees, and CoAtNet-2 has 25 BatchNorm layers (LOG #376).
    """
    import torch.nn as _nn
    n = 0
    for m in model.modules():
        if isinstance(m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d)):
            m.eval()
            n += 1
    return n


def train_on_corpus(vols, masks, row_of, train_ids, gold_ids, labels, gold_y,
                    res, k_train, k_eval, epochs, batch, bb_lr, head_lr,
                    device, seed=42, topk=3, time_budget_s=None,
                    encode_chunk=24, ckpt_path=None, llrd=None, sample_weights=None,
                    freeze_bn=False):
    """Fine-tune the study-window MIL model on the memmap corpus.

    Differences from our R line, all adopted from the reference arm:
      * `pos_weight` per finding from its prevalence -- we had NO class balancing,
        which on a macro-AUC metric penalises the rare findings (Fracture,
        Contusion) most.
      * a bigger MIL bag (k windows/study/step) -- R1-R3 used 4, which is a very
        small bag for attention pooling.
      * intensity jitter only. No affine. Their comment on flips is that
        laterality is signal, and our own line reached the same conclusion.
      * all 58 gold studies held out as the validation set, so every soft-labelled
        study trains. Epoch selection reads gold macro-AUC.
      * top-k epochs averaged by validation score (not a tail-window SWA).
    """
    import time
    import torch
    import torch.nn as nn
    from raptor import RaptorClassifier, build_raptor_backbone, _recalibrate_bn
    from train import macro_auc
    from config import RAPTOR_ARCH

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    bb = build_raptor_backbone()
    model = RaptorClassifier(bb, f_dim=bb.num_features).to(device)
    try:
        import r9700_opt
        if r9700_opt.is_r9700():
            r9700_opt.optimize_model_for_r9700(model)
    except Exception as e:
        print(f"[R9700 Opt] Note: {e}", flush=True)

    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32,
                      device=device)
    print("pos_weight per finding:", np.round(pw.cpu().numpy(), 2).tolist(), flush=True)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    if sample_weights is not None:
        # Experiment 3 (LOG #405): weight each study-finding term by its label confidence.
        # Weighted mean, so an all-ones W reproduces the unweighted loss exactly.
        _lossf_el = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        _wm = np.stack([sample_weights[u] for u in train_ids])
        print(f"confidence-weighted BCE: W mean {_wm.mean():.3f} min {_wm.min():.3f} "
              f"max {_wm.max():.3f}", flush=True)

    groups, max_lrs = build_param_groups(model, bb_lr, head_lr, llrd)
    if llrd:
        print(f"layer-wise LR decay {llrd}: {len(groups) - 1} backbone groups, "
              f"lr {max_lrs[0]:.2e} (embed) .. {max_lrs[-2]:.2e} (top) | "
              f"head {max_lrs[-1]:.2e}", flush=True)
    fused_opt = False
    try:
        import r9700_opt
        fused_opt = r9700_opt.is_r9700()
    except Exception:
        pass
    opt = torch.optim.AdamW(groups, weight_decay=0.02, fused=fused_opt)
    steps_per = max(len(train_ids) // batch, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lrs, total_steps=max(epochs * steps_per, 1),
        pct_start=0.15)
    use_cuda = device == "cuda"
    # The reference trains in bf16, but a T4 is Turing (cc 7.5): it has fp16 and fp32
    # conv engines and NOT bf16 for these shapes -- the reference's own inference code
    # says so in a comment. Requesting bf16 there does not fail loudly, it falls back
    # to fp32 activations and roughly doubles the memory, which is what OOM'd C1 at
    # the very step count R3 ran safely in fp16. Pick the dtype the card actually has,
    # and bring the GradScaler back when that means fp16.
    # NOT torch.cuda.is_bf16_supported(): on a T4 it returns True (it counts
    # emulation), so the first attempt at this fix silently stayed on bf16 and
    # OOM'd with byte-identical numbers. Compute capability is unambiguous --
    # native bf16 starts at Ampere (8.0); Turing (7.5) has fp16 tensor cores only.
    _cap = torch.cuda.get_device_capability()[0] if use_cuda else 0
    amp_dtype = torch.bfloat16 if _cap >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and amp_dtype is torch.float16)
    if use_cuda:
        print(f"amp dtype: {amp_dtype} (capability sm_{_cap}x) | "
              f"scaler {scaler.is_enabled()}", flush=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _bag(uids, k, train, jitter=None):
        xs = []
        for u in uids:
            r = row_of[u]
            cs = centers_from_mask(masks[r], k, train, rng)
            xs.append(read_windows(vols[r], cs, res))
        x = torch.stack(xs, 0)                        # (B, K, 3, res, res) in [0,1]
        if jitter is not None:
            x = (x * jitter).clamp(0, 1)              # jitter BEFORE normalisation
        b, kk = x.shape[:2]
        # the corpus stores [0,1]; an ImageNet-pretrained backbone expects its own
        # channel statistics (the reference runs NORM="imagenet" for its CoAtNet arm)
        return ((x.flatten(0, 1) - mean) / std).view(b, kk, *x.shape[2:])

    @torch.no_grad()
    def evaluate(k):
        model.eval()
        preds = []
        for b0 in range(0, len(gold_ids), 2):
            x = _bag(gold_ids[b0:b0 + 2], k, train=False).to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                o = model(x, chunk=encode_chunk).float()
            preds.append(torch.sigmoid(o).cpu().numpy())
        return macro_auc(gold_y, np.concatenate(preds))[0]

    best, best_state, keep = -1.0, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if freeze_bn:
            _n_bn = freeze_batchnorm_stats(model)
            if ep == 0:
                print(f"frozen BN statistics: {_n_bn} BatchNorm layers held in eval mode", flush=True)
        order = rng.permutation(len(train_ids))
        tot, nb, n_skip = 0.0, 0, 0
        for b0 in range(0, len(train_ids) - batch + 1, batch):
            sel = [train_ids[i] for i in order[b0:b0 + batch]]
            # intensity jitter only -- no affine, no flips (laterality is signal)
            x = _bag(sel, k_train, train=True,
                     jitter=1.0 + (rng.rand() - 0.5) * 0.20).to(device)
            y = torch.from_numpy(np.stack([labels[u] for u in sel])).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                if sample_weights is None:
                    loss = lossf(model(x).float(), y)
                else:
                    _w = torch.from_numpy(np.stack([sample_weights[u] for u in sel])).to(device)
                    loss = (_lossf_el(model(x).float(), y) * _w).sum() / _w.sum().clamp(min=1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)          # unscale before clipping, or the norm is wrong
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            # The GradScaler is only enabled for fp16 (see amp_dtype above), and it is
            # what skips inf/nan steps. On a bf16 card it is disabled, so `step` is a
            # bare `opt.step()` with NO such skip -- and clip_grad_norm_ multiplies
            # EVERY gradient by a nan total-norm, so one bad batch poisons the weights
            # permanently. That is exactly how the local R9700 run went loss=nan at
            # epoch 5 and never recovered. Guard the step ourselves on that path;
            # leave the fp16 path byte-identical so T4 runs stay comparable.
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            elif torch.isfinite(gn):
                opt.step()
            else:
                n_skip += 1
            sched.step()
            if np.isfinite(float(loss.item())):
                tot += float(loss.item())
                nb += 1
        au = evaluate(k_eval)
        print(f"    epoch {ep + 1}/{epochs}  loss={tot / max(nb, 1):.4f}  "
              f"gold={au:.4f}  [{(time.time() - t0) / 60:.1f}m]"
              + (f"  skip={n_skip}" if n_skip else ""), flush=True)
        state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        if au > best:
            best, best_state = au, state
            if ckpt_path:
                # A 16-epoch local run must not stake every epoch on reaching the
                # end of the loop; C5 lost 4 good epochs that way. Write via a
                # temp file + rename so a copy taken mid-training (e.g. to Drive
                # from a Colab cell) can never catch a half-written 270 MB file.
                torch.save({"state_dict": state, "res": res, "gold_auc": best,
                            "slots": CORPUS_SLOTS, "epoch": ep + 1,
                            "arch": RAPTOR_ARCH, **_specialist_meta()},
                           ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
        keep.append((au, state))
        keep.sort(key=lambda t: -t[0])
        keep[:] = keep[:max(1, topk)]
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"    time budget hit after epoch {ep + 1}", flush=True)
            break

    if len(keep) > 1:
        # Average the top-k epochs, then re-accumulate BatchNorm statistics: the
        # running stats belong to the trajectory that produced them, not to the
        # averaged weights. Accept the average only if it beats the best epoch.
        avg = {kk: sum(s[kk].float() for _, s in keep) / len(keep)
               for kk in keep[0][1] if keep[0][1][kk].dtype.is_floating_point}
        full = model.state_dict()
        full.update({kk: v.to(full[kk].dtype) for kk, v in avg.items()})
        model.load_state_dict(full)
        sub = [train_ids[i] for i in rng.choice(len(train_ids), size=min(120, len(train_ids)),
                                                replace=False)]

        _recal_batches = 30
        bn = [m for m in model.modules()
              if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                torch.nn.BatchNorm3d))]
        if bn and freeze_bn:
            # stats were never updated, so every kept epoch shares them: nothing to re-estimate
            print("    BN recalibration skipped (frozen BN statistics)", flush=True)
        if bn and not freeze_bn:
            old = {m: m.momentum for m in bn}
            for m in bn:
                m.reset_running_stats()
                m.momentum = None
            model.train()
            with torch.no_grad():
                for i in range(0, min(_recal_batches * batch, len(sub)), batch):
                    x = _bag(sub[i:i + batch], k_train, train=True).to(device)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda):
                        model(x)
            for m in bn:
                m.momentum = old[m]
        au_avg = evaluate(k_eval)
        print(f"    top-{len(keep)} average: gold={au_avg:.4f} (best epoch {best:.4f})",
              flush=True)
        if ckpt_path:
            # Save BOTH candidates whichever the rule picks below. The same rule gave C6
            # the averaged weights and C7 the single best epoch, and C7's average was
            # only ever in memory, so that confound could not be split later (LOG #388).
            _base = ckpt_path[:-3] if ckpt_path.endswith(".pt") else ckpt_path
            _meta = {"res": res, "slots": CORPUS_SLOTS, "arch": RAPTOR_ARCH, **_specialist_meta()}
            torch.save({"state_dict": {kk: v.detach().cpu().clone()
                                       for kk, v in model.state_dict().items()},
                        "gold_auc": au_avg, "weights": "topk_avg", **_meta},
                       _base + "_topk_avg.pt")
            torch.save({"state_dict": best_state, "gold_auc": best,
                        "weights": "best_epoch", **_meta}, _base + "_best_epoch.pt")
            print(f"    saved both candidates: {_base}_topk_avg.pt / _best_epoch.pt",
                  flush=True)
        if au_avg > best:
            best = au_avg
            best_state = {kk: v.detach().cpu().clone()
                          for kk, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def _collect_ckpts(names):
    """Resolve every named checkpoint that exists, preserving order, no dups."""
    paths, seen = [], set()
    for name in names:
        p = _find(name)
        if not p:
            continue
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        paths.append(p)
        seen.add(ap)
    return paths


def run_corpus_infer(out_dir=".", ckpt_names=None, k_eval=None, ckpt_weights=None):
    """Score the hidden test from checkpoint(s) trained ANYWHERE. No training.

    Why this exists: submitting a *training* kernel makes Kaggle re-run the whole
    fit on every submission (C3 took ~5.5h to score), and it cannot score weights
    trained off-Kaggle at all -- which is every arm from C5 onwards. This mode
    mounts the weights instead, so a submission costs the DICOM decode plus a
    forward pass.

    Multiple checkpoints are rank-averaged per column (the metric only reads
    order). Geometry must match across members -- mixed 44/64-slice bags would
    be a silent distribution shift, so a mismatch exits loudly.

    `ckpt_weights` fuses members with UNEQUAL weight, which a heterogeneous
    ensemble needs: an arm that is materially weaker drags an equal-weight average
    below the strong arm alone (measured offline on the 58 gold: w=0.5 scored 0.8937
    vs 0.9068 for the CoAtNet arm by itself). Weights are chosen offline and baked in
    at build time -- never tuned against the leaderboard.

    The slice geometry comes FROM THE CHECKPOINT, never from config: C5/C6/C7
    trained on 64 slices while C1-C3 trained on 44. `build_test_volume` reads
    the module globals, so they are rebound here to whatever the checkpoint
    recorded.
    """
    import time
    import torch
    global CORPUS_SLOTS, CORPUS_D
    from config import RAPTOR_RES, CORPUS_K_EVAL, RAPTOR_ENCODE_CHUNK, RAPTOR_ARCH
    from dataset import RSNADataset
    from train import _write_submission, rank_average, _rankdata
    from raptor import RaptorClassifier, build_raptor_backbone

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ckpt_names or [
        "c7_corpus0.pt", "c6_corpus0.pt", "c5_corpus0.pt", "corpus0.pt",
    ]
    paths = _collect_ckpts(names)
    if not paths:
        raise SystemExit(f"no checkpoint found among {names} -- attach a weights dataset")
    print(f"ensemble: {len(paths)} checkpoint(s)", flush=True)

    loaded = []
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "epoch" in ck:
            print(f"WARNING: {path} is a MID-RUN snapshot (epoch {ck['epoch']}), "
                  f"not a finished run -- top-k averaging never happened", flush=True)
        loaded.append((path, ck))

    slots = [tuple(x) for x in loaded[0][1].get("slots", CORPUS_SLOTS)]
    res = int(loaded[0][1].get("res", RAPTOR_RES))
    for path, ck in loaded[1:]:
        s2 = [tuple(x) for x in ck.get("slots", CORPUS_SLOTS)]
        r2 = int(ck.get("res", RAPTOR_RES))
        if s2 != slots or r2 != res:
            raise SystemExit(
                f"geometry mismatch: {path} slots={s2} res={r2} vs {slots}/{res}")
    CORPUS_SLOTS = slots
    CORPUS_D = sum(x[2] for x in slots)
    # Diagnosis specialist (LOG #490): restore the training window policy from the
    # checkpoint; members of one ensemble must agree on it.
    _allows = {tuple(ck.get("centre_allow") or ()) for _p, ck in loaded}
    if len(_allows) != 1:
        raise SystemExit(f"window-policy mismatch across members: {_allows}")
    _allow = next(iter(_allows))
    import config as _cfg_inf
    _cfg_inf.CORPUS_CENTRE_ALLOW = list(_allow) or None
    print(f"  window centres: {'ALL' if not _allow else list(_allow)} | "
          f"trained labels: {loaded[0][1].get('train_labels') or 'ALL'}", flush=True)
    k = int(k_eval or CORPUS_K_EVAL)
    print(f"  res={res} D={CORPUS_D} k_eval={k} slots={CORPUS_SLOTS}", flush=True)
    print(f"  window rule: {'LEGAL (same slot, non-empty rows)' if LEGAL_WINDOWS else 'reference'}",
          flush=True)
    for path, ck in loaded:
        print(f"  {path} gold_auc={ck.get('gold_auc')} "
              f"arch={ck.get('arch', RAPTOR_ARCH)}", flush=True)

    test = RSNADataset("test")
    tv, tm, tuids = build_test_corpus(test)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    amp_dtype = (torch.bfloat16
                 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)

    member_preds = []
    member_default = []
    last_model = None
    for path, ck in loaded:
        # The backbone comes FROM THE CHECKPOINT: a heterogeneous ensemble (CoAtNet
        # arm + RadImageNet ResNet-50 arm) has a different architecture per member,
        # and f_dim differs with it (1024 vs 2048).
        bb = build_raptor_backbone(ck.get("arch", RAPTOR_ARCH))
        model = RaptorClassifier(bb, f_dim=bb.num_features).to(device)
        # strict=True on purpose: a silently partial load here would submit a
        # half-random model and we would read the low score as a failed experiment.
        model.load_state_dict({kk: v for kk, v in ck["state_dict"].items()})
        model.eval()
        print(f"state_dict loaded strictly: {os.path.basename(path)}", flush=True)
        def _predict(legal):
            preds = []
            with torch.no_grad():
                for i in range(0, len(tuids), 2):
                    xs = []
                    for r in range(i, min(i + 2, len(tuids))):
                        cs = centers_from_mask(tm[r], k, train=False, legal=legal)
                        xs.append(read_windows(tv[r], cs, res))
                    x = torch.stack(xs, 0)
                    b, kk = x.shape[:2]
                    x = ((x.flatten(0, 1) - mean) / std).view(b, kk, *x.shape[2:])
                    with torch.autocast("cuda", dtype=amp_dtype,
                                        enabled=device == "cuda"):
                        o = model(x.to(device), chunk=RAPTOR_ENCODE_CHUNK).float()
                    preds.append(torch.sigmoid(o).cpu().numpy())
            return np.concatenate(preds)

        member_preds.append(_predict(None))
        if LEGAL_WINDOWS:
            # Same model, same decoded volumes, reference windows. This side output must
            # reproduce the member's standalone submission exactly, which is what shows
            # the window rule is the only thing that changed (LOG #401).
            member_default.append(_predict(False))
        last_model = model
        if device == "cuda" and len(loaded) > 1:
            del model
            torch.cuda.empty_cache()

    def _fuse(mp, tag):
        pre = f"[{tag}] " if LEGAL_WINDOWS else ""
        if len(mp) == 1:
            return mp[0]
        if ckpt_weights:
            w = np.asarray(ckpt_weights, dtype=np.float64)
            if len(w) != len(mp):
                raise SystemExit(f"{len(w)} weights for {len(mp)} members")
            w = w / w.sum()
            n = mp[0].shape[0]
            out = np.zeros_like(mp[0])
            for j in range(mp[0].shape[1]):
                out[:, j] = sum(wi * (_rankdata(p[:, j]) / n) for wi, p in zip(w, mp))
            print(f"{pre}weighted rank fusion of {len(mp)} members, "
                  f"weights={np.round(w, 4).tolist()} (pre-registered, not LB-tuned)",
                  flush=True)
            return out
        print(f"{pre}rank-averaged {len(mp)} members (equal weight)", flush=True)
        return rank_average(mp)

    fused = _fuse(member_preds, "submission")
    if LEGAL_WINDOWS:
        _write_submission(tuids, _fuse(member_default, "default-windows side output"),
                          os.path.join(out_dir, "_default_windows_submission.csv"))
        print("wrote default-window side output _default_windows_submission.csv", flush=True)
    _write_submission(tuids, fused, os.path.join(out_dir, "submission.csv"))
    print(f"corpus infer finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    return last_model


def run_corpus_submit(epochs=None, batch=None, k_train=None, k_eval=None,
                      res=None, out_dir=".", seed=42, do_submit=True,
                      time_budget_h=6.0, max_studies=None):
    """Train on the published memmap corpus, then decode + predict the test set.

    The train corpus covers only train.csv, so the test split is rebuilt from
    DICOM with `build_test_volume`, which mirrors the corpus's own preprocessing
    exactly (see this module's docstring). Getting that wrong is silent: the
    model would simply see a different distribution than it trained on.
    """
    import time
    import torch
    from config import (LABELS, RAPTOR_RES, CORPUS_EPOCHS, CORPUS_BATCH,
                        CORPUS_K_TRAIN, CORPUS_K_EVAL, RAPTOR_LR_BACKBONE,
                        RAPTOR_LR_HEAD, RAPTOR_ENCODE_CHUNK)
    from dataset import RSNADataset
    from train import _write_submission, macro_auc
    from raptor import predict_windows  # noqa: F401  (kept importable for parity)

    epochs = epochs or CORPUS_EPOCHS
    batch = batch or CORPUS_BATCH
    k_train = k_train or CORPUS_K_TRAIN
    k_eval = k_eval or CORPUS_K_EVAL
    res = res or RAPTOR_RES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    vols, masks, cids = open_corpus()
    row_of = {u: i for i, u in enumerate(cids)}

    ds = RSNADataset("train")
    y_raw = ds.y_raw
    has_gold = np.isfinite(y_raw).all(axis=1)
    covered = [u for u in ds.uids if u in row_of]
    print(f"corpus covers {len(covered)}/{len(ds.uids)} of train.csv", flush=True)

    gold_ids = [u for i, u in enumerate(ds.uids) if has_gold[i] and u in row_of]
    goldset = set(gold_ids)
    train_ids = [u for u in covered if u not in goldset]
    if max_studies:
        train_ids = train_ids[:max_studies]
    # soft targets from OUR report extractor (the reference's LLM-generated
    # labels_llm_soft.parquet is not distributed with its notebook)
    idx_of = {u: i for i, u in enumerate(ds.uids)}
    labels = {u: ds.targets[idx_of[u]].astype(np.float32) for u in train_ids}
    import config as _cfg_w
    sample_weights = None
    if getattr(_cfg_w, "CORPUS_CONF_WEIGHTED_BCE", False):
        sample_weights = {u: ds.weights[idx_of[u]].astype(np.float32) for u in train_ids}
    gold_y = np.stack([y_raw[idx_of[u]] for u in gold_ids]).astype(np.float32)
    _train_labels = getattr(_cfg_w, "CORPUS_TRAIN_LABELS", None)
    if _train_labels:
        # Diagnosis specialist (LOG #490): zero BCE weight on every other label (the
        # weighted-mean loss then averages over the kept labels only) and blank the other
        # gold columns so macro_auc -- epoch selection and top-k -- reads the kept ones only.
        from config import LABELS as _LAB
        _keep = np.array([l in set(_train_labels) for l in _LAB], np.float32)
        assert _keep.sum() == len(_train_labels), f"unknown label in {_train_labels}"
        base_w = sample_weights or {u: np.ones(len(_LAB), np.float32) for u in train_ids}
        sample_weights = {u: (base_w[u] * _keep).astype(np.float32) for u in train_ids}
        gold_y = np.where(_keep[None, :] > 0, gold_y, np.nan).astype(np.float32)
        print(f"specialist: training / selecting on {list(_train_labels)} only", flush=True)
    _allow = getattr(_cfg_w, "CORPUS_CENTRE_ALLOW", None)
    if _allow:
        print(f"specialist: window centres restricted to {sorted(_allow)}", flush=True)
    print(f"train {len(train_ids)} (soft) | gold-val {len(gold_ids)} | "
          f"res {res} | k {k_train}/{k_eval} | bs {batch} | ep {epochs}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "corpus0.pt")
    import config as _cfg
    model, best = train_on_corpus(
        vols, masks, row_of, train_ids, gold_ids, labels, gold_y, res,
        k_train, k_eval, epochs, batch, RAPTOR_LR_BACKBONE, RAPTOR_LR_HEAD,
        device, seed=seed, time_budget_s=time_budget_h * 3600.0,
        encode_chunk=RAPTOR_ENCODE_CHUNK, ckpt_path=ckpt,
        llrd=getattr(_cfg, "RAPTOR_LLRD", None), sample_weights=sample_weights,
        freeze_bn=getattr(_cfg, "CORPUS_FREEZE_BN", False))
    print(f"corpus run: best gold macro-AUC={best:.4f}", flush=True)

    from config import RAPTOR_ARCH as _ARCH
    torch.save({"state_dict": model.state_dict(), "res": res,
                "gold_auc": best, "slots": CORPUS_SLOTS, "arch": _ARCH, **_specialist_meta()}, ckpt)
    print("saved", ckpt, flush=True)

    if do_submit:
        test = RSNADataset("test")
        tv, tm, tuids = build_test_corpus(test)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(tuids), 2):
                xs = []
                for r in range(i, min(i + 2, len(tuids))):
                    cs = centers_from_mask(tm[r], k_eval, train=False)
                    xs.append(read_windows(tv[r], cs, res))
                x = torch.stack(xs, 0)
                b, kk = x.shape[:2]
                x = ((x.flatten(0, 1) - mean) / std).view(b, kk, *x.shape[2:])
                amp_dtype = (torch.bfloat16
                             if device == "cuda"
                             and torch.cuda.get_device_capability()[0] >= 8
                             else torch.float16)
                with torch.autocast("cuda", dtype=amp_dtype,
                                    enabled=device == "cuda"):
                    o = model(x.to(device), chunk=RAPTOR_ENCODE_CHUNK).float()
                preds.append(torch.sigmoid(o).cpu().numpy())
        _write_submission(tuids, np.concatenate(preds),
                          os.path.join(out_dir, "submission.csv"))
    print(f"corpus run finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    return model

# ===================== experiments.py =====================
"""Reproducible experiment registry + runner for the RSNA Knee comparison study.

Philosophy (paper-grade): the training PROTOCOL is fixed; the only thing that
varies between experiments is the image BACKBONE (tiny_cnn / dinov2_small /
dinov2_base) plus, for one ablation, whether the encoder is fine-tuned. All
other settings (seed, CV scheme, slots, preprocessing, losses, EMA) are pinned in
config.py / train.py and shared by every run.

Each run writes a self-contained artifact under experiments/exp_<id>/:
    config.json   - exact configuration
    metrics.json  - macro-AUC, per-label AUC, fold scores, n studies/gold
    oof.npy       - out-of-fold predictions (NaN where unlabeled)
    provenance.json - environment + source hash + fold assignment + CLI (repro)
so any result can be regenerated and audited.

Usage:
    python experiments.py --exp exp002          # run one experiment
    python experiments.py --summary              # rebuild results/ from all runs
"""
import os
import sys
import json
import hashlib
import datetime
import platform

import numpy as np

from config import (LABELS, SEED, PROJECT_ROOT, BATCH_SIZE, EPOCHS, LR,
                    ENCODER_UNFREEZE_LAST)
import train as T

# On Kaggle the kernel is a single inlined file; artifacts must land in
# /kaggle/working (the exposed kernel output). Locally they land under the repo.
_KAGGLE_WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else None
EXP_DIR = os.path.join(_KAGGLE_WORKING, "experiments") if _KAGGLE_WORKING \
    else os.path.join(PROJECT_ROOT, "experiments")
RESULTS_DIR = os.path.join(_KAGGLE_WORKING, "results") if _KAGGLE_WORKING \
    else os.path.join(PROJECT_ROOT, "results")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


# ----------------------------- experiment registry -----------------------------
class ExpConfig:
    def __init__(self, exp_id, backbone, unfreeze_last=ENCODER_UNFREEZE_LAST,
                 max_studies=None, epochs=EPOCHS, folds=5,
                 batch=None, lr=LR, note=""):
        self.exp_id = exp_id
        self.backbone = backbone
        self.unfreeze_last = unfreeze_last
        self.max_studies = max_studies
        self.epochs = epochs
        self.folds = folds
        self.batch = batch or (2 if backbone.startswith("dinov2") else BATCH_SIZE)
        self.lr = lr
        self.note = note

    def to_dict(self):
        return dict(
            exp_id=self.exp_id, backbone=self.backbone,
            unfreeze_last=self.unfreeze_last, max_studies=self.max_studies,
            epochs=self.epochs, folds=self.folds, batch=self.batch, lr=self.lr,
            note=self.note, seed=SEED)


# Core 3-way comparison + 1 fine-tune ablation, each at FULL data (20 epochs),
# plus fast SCREENING runs (subset=600 studies, 5 epochs) used to (a) verify the
# DINOv2 offline-loading path on T4 and (b) get a cheap relative ranking before
# committing to the expensive full runs.
EXP_REGISTRY = {
    "exp001":   ExpConfig("exp001", "tiny_cnn", unfreeze_last=2,
                          note="control: from-scratch CNN, no pretraining"),
    "exp002":   ExpConfig("exp002", "dinov2_small", unfreeze_last=2,
                          note="DINOv2 ViT-S, fine-tune last 2 blocks"),
    "exp003":   ExpConfig("exp003", "dinov2_base", unfreeze_last=2,
                          note="DINOv2 ViT-B, fine-tune last 2 blocks"),
    "exp002f":  ExpConfig("exp002f", "dinov2_small", unfreeze_last=0,
                          note="ABLATION: DINOv2 ViT-S FROZEN (no fine-tune)"),
    "exp001_scr": ExpConfig("exp001_scr", "tiny_cnn", unfreeze_last=2,
                            max_studies=600, epochs=5,
                            note="screening: tiny_cnn subset"),
    "exp002_scr": ExpConfig("exp002_scr", "dinov2_small", unfreeze_last=2,
                            max_studies=600, epochs=5,
                            note="screening: dinov2_small subset"),
    "exp003_scr": ExpConfig("exp003_scr", "dinov2_base", unfreeze_last=2,
                            max_studies=600, epochs=5,
                            note="screening: dinov2_base subset"),
    # FULL controlled comparison (single fixed 80/20 split, N=1500, E=10).
    # folds=1 => identical group-disjoint split across backbones (see train.run_cv).
    "tiny_cnn_1500_ep10": ExpConfig("tiny_cnn_1500_ep10", "tiny_cnn",
                                    unfreeze_last=2, max_studies=1500, epochs=10,
                                    folds=1, note="FULL: from-scratch CNN, single split"),
    "dinov2_small_1500_ep10": ExpConfig("dinov2_small_1500_ep10", "dinov2_small",
                                        unfreeze_last=2, max_studies=1500, epochs=10,
                                        folds=1, note="FULL: DINOv2 ViT-S, single split"),
    "dinov2_base_1500_ep10": ExpConfig("dinov2_base_1500_ep10", "dinov2_base",
                                       unfreeze_last=2, max_studies=1500, epochs=10,
                                       folds=1, note="FULL: DINOv2 ViT-B, single split"),
    # v2 submit protocol: 2.5D cache + 4-fold rank-average (see train.run_submit).
    "sub_dinov2s_v2": ExpConfig("sub_dinov2s_v2", "dinov2_small",
                                unfreeze_last=2, max_studies=None, epochs=10,
                                folds=4, batch=8,
                                note="SUBMIT v2: DINOv2-S, cache+2.5D+4fold"),
}


# ----------------------------- provenance / reproducibility -----------------------------
def _source_hash() -> str:
    """SHA-256 over all src .py files (recursive) for exact-code pinning.

    On Kaggle the code is a single inlined file (no src/ tree), so we fall back
    to hashing the running script itself."""
    h = hashlib.sha256()
    if os.path.isdir(SRC_DIR):
        for root, _, files in os.walk(SRC_DIR):
            for f in sorted(files):
                if f.endswith(".py") and "__pycache__" not in root:
                    with open(os.path.join(root, f), "rb") as fh:
                        h.update(fh.read())
        return h.hexdigest()
    # Kaggle: hash the inlined run.py
    try:
        with open(sys.argv[0], "rb") as fh:
            return h.hexdigest()
    except Exception:
        return "kaggle-inline"


def _env_snapshot() -> dict:
    import torch
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        platform=platform.platform(),
    )


# ----------------------------- run one experiment -----------------------------
def run_cv_and_save(cfg: "ExpConfig"):
    """Run CV for an arbitrary ExpConfig and write its artifact directory."""
    exp_id = cfg.exp_id
    out_dir = os.path.join(EXP_DIR, exp_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== RUN {exp_id}: backbone={cfg.backbone} "
          f"unfreeze_last={cfg.unfreeze_last} studies="
          f"{cfg.max_studies or 'FULL'} epochs={cfg.epochs} folds={cfg.folds} ===")

    res = T.run_cv(backbone_name=cfg.backbone, unfreeze_last=cfg.unfreeze_last,
                   max_studies=cfg.max_studies, epochs=cfg.epochs,
                   folds=cfg.folds, batch=cfg.batch, lr=cfg.lr)

    metrics = dict(
        exp_id=exp_id, macro_auc=float(res["macro"]),
        mean_fold=float(np.mean(res["fold_scores"])),
        std_fold=float(np.std(res["fold_scores"])),
        fold_scores=[float(x) for x in res["fold_scores"]],
        per_label={LABELS[i]: (None if np.isnan(a) else float(a))
                   for i, a in enumerate(res["per_label"])},
        n_studies=len(res["idx"]),
        epochs=cfg.epochs, folds=cfg.folds,
        backbone=cfg.backbone, unfreeze_last=cfg.unfreeze_last,
        max_studies=cfg.max_studies, batch=cfg.batch, lr=cfg.lr,
    )
    prov = dict(
        env=_env_snapshot(),
        source_hash=_source_hash(),
        seed=SEED,
        argv=sys.argv,
        generated_at=datetime.datetime.utcnow().isoformat() + "Z",
        fold_splits=res["fold_splits"],
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    np.save(os.path.join(out_dir, "oof.npy"), res["oof"])

    print(f"[{exp_id}] macro-AUC={metrics['macro_auc']:.4f} "
          f"(mean fold {metrics['mean_fold']:.4f} +- {metrics['std_fold']:.4f})")
    print(f"[{exp_id}] saved -> {out_dir}")
    return metrics


def run_experiment(exp_id: str):
    if exp_id not in EXP_REGISTRY:
        raise KeyError(f"unknown experiment {exp_id}; known: {list(EXP_REGISTRY)}")
    return run_cv_and_save(EXP_REGISTRY[exp_id])


# ----------------------------- summarize all experiments -----------------------------
def _load_metrics(exp_id):
    p = os.path.join(EXP_DIR, exp_id, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def summarize():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for exp_id in EXP_REGISTRY:
        m = _load_metrics(exp_id)
        if not m:
            continue
        rows.append(m)
    rows.sort(key=lambda r: r["exp_id"])

    # results.csv
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    import csv
    fields = ["exp_id", "backbone", "unfreeze_last", "max_studies", "epochs",
              "folds", "batch", "macro_auc", "mean_fold", "std_fold"]
    fields += [f"auc_{l}" for l in LABELS]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in
                   ["exp_id", "backbone", "unfreeze_last", "max_studies",
                    "epochs", "folds", "batch", "macro_auc", "mean_fold",
                    "std_fold"]}
            for l in LABELS:
                v = r["per_label"].get(l)
                row[f"auc_{l}"] = ("" if v is None else f"{v:.4f}")
            w.writerow(row)

    # results.md (paper-style comparison table)
    md = ["# RSNA Knee — Method Comparison (paper-style)", "",
          "> Controlled experiment: the training protocol is FIXED; only the image "
          "backbone varies. CV macro-AUC is computed on the held-out GOLD portion "
          "(public labels are only 58 studies, so it is a *relative* ranking "
          "signal; the hidden test leaderboard is the final arbiter).", "",
          "## Results", "",
          "| exp_id | backbone | tune | data | ep | macro-AUC | mean±std fold |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        data = "FULL" if not r["max_studies"] else f"sub{r['max_studies']}"
        tune = "ft" if r["unfreeze_last"] > 0 else "frozen"
        md.append(
            f"| {r['exp_id']} | {r['backbone']} | {tune} | {data} | "
            f"{r['epochs']} | {r['macro_auc']:.4f} | "
            f"{r['mean_fold']:.4f}±{r['std_fold']:.4f} |")
    md += ["", "## Per-label AUC (held-out gold)", "",
           "| exp_id | " + " | ".join(LABELS) + " |",
           "|---" + "|".join(["---"] * len(LABELS)) + "|"]
    for r in rows:
        cells = []
        for l in LABELS:
            v = r["per_label"].get(l)
            cells.append("" if v is None else f"{v:.3f}")
        md.append(f"| {r['exp_id']} | " + " | ".join(cells) + " |")
    md += ["", "## Reproducibility", "",
           "Each `experiments/exp_<id>/` contains `config.json`, `metrics.json`, "
           "`oof.npy`, and `provenance.json` (env + source SHA-256 + fold "
           "assignment + CLI). Regenerate with `python experiments.py --exp "
           "<id>`.", "",
           "## Limitations", "",
           "- Public gold labels = 58 studies; CV macro-AUC is a noisy *relative* "
           "signal, not an absolute estimate. Final ranking uses the hidden test "
           "leaderboard (public + private).",
           "- Physical-scale crop and laterality normalization rely on DICOM "
           "headers present in the Kaggle environment; they degrade gracefully "
           "(fixed crop / no flip) if stripped."]
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {csv_path} and {os.path.join(RESULTS_DIR, 'RESULTS.md')}")
    return rows


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarize()
    elif len(sys.argv) > 2 and sys.argv[1] == "--exp":
        run_experiment(sys.argv[2])
    else:
        print("usage: python experiments.py --exp <id> | --summary")


# ===================== make_submission.py =====================
"""Inference -> submission.csv for RSNA Knee (run inside Kaggle Notebook).

Loads held-out test studies (mounted in the notebook), runs the imaging-only
trained model (slots -> attention heads), and writes `submission.csv` with one
row per StudyInstanceUID. The report is NOT used at inference (test.csv has no
Report column), which is why the model is pure vision.
"""
import os
import numpy as np
import pandas as pd
import torch

from config import (LABELS, BATCH_SIZE, DATA_ROOT, PROJECT_ROOT, BACKBONE,
                    ENCODER_UNFREEZE_LAST, N_GROUPS_MAX)
from dataset import RSNADataset
import model as M
import train as T


def _device():
    """CUDA if a tiny op succeeds (avoids P100/sm_60 'no kernel image' crash)."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        x = torch.zeros(1, device="cuda")
        _ = (x + 1).item()
        del x
        return "cuda"
    except Exception:
        return "cpu"


def _out_dir():
    """Where to write submission.csv: /kaggle/working on Kaggle (exposed as a
    kernel output artifact), else the current directory for local smoke tests.
    Kaggle script kernels do NOT surface files written to cwd, so we must land
    the submission in /kaggle/working for `kaggle kernels output` to retrieve it.
    """
    return "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def _resolve_test_csv():
    """Find the competition test.csv among the usual mount locations.

    Returns the first existing candidate path, else None (so main() falls back
    to a placeholder rather than crashing on a path we didn't anticipate).
    """
    candidates = []
    if DATA_ROOT:
        candidates.append(os.path.join(DATA_ROOT, "test.csv"))
        candidates.append(os.path.join(DATA_ROOT, "sample_submission.csv"))
    slug = "rsna-knee-abnormality-detection"
    for base in (f"/kaggle/input/competitions/{slug}",
                 f"/kaggle/input/competitions/{slug}/competition",
                 "/kaggle/input/competitions",
                 "/kaggle/input/competition"):
        if os.path.isdir(base):
            candidates.append(os.path.join(base, "test.csv"))
            candidates.append(os.path.join(base, "sample_submission.csv"))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def main(model_path: str | None = None):
    # Real inference path: only when the competition test split is actually present.
    test_csv = _resolve_test_csv()
    if test_csv and os.path.exists(test_csv):
        try:
            # DICOM series live under {root}/test_series/{StudyUID}/{SeriesUID}/,
            # NOT next to test.csv. Prefer DATA_ROOT/test_series, then sibling of
            # test.csv; fall back to the dataset default (DATA_ROOT/test_series).
            parent = os.path.dirname(test_csv)
            candidates = []
            if DATA_ROOT:
                candidates.append(os.path.join(DATA_ROOT, "test_series"))
            candidates.append(os.path.join(parent, "test_series"))
            series_root = next((c for c in candidates if os.path.isdir(c)), None)
            print("test.csv:", test_csv, "| series_root:", series_root)
            paths = []
            if model_path:
                if os.path.isdir(model_path):
                    paths = sorted(
                        os.path.join(model_path, f)
                        for f in os.listdir(model_path)
                        if f.startswith("fold") and f.endswith(".pt"))
                elif os.path.exists(model_path):
                    paths = [model_path]
            if not paths:
                print("WARNING: no model checkpoint -> writing placeholder.")
                _write_placeholder()
                return

            device = _device()
            print("inference device:", device, "| ckpts:", paths)
            models, n_groups = [], N_GROUPS_MAX
            for p in paths:
                ckpt = torch.load(p, map_location="cpu")
                bb = BACKBONE
                unfr = ENCODER_UNFREEZE_LAST
                if isinstance(ckpt, dict):
                    bb = ckpt.get("backbone", bb)
                    unfr = ckpt.get("unfreeze_last", unfr)
                    n_groups = int(ckpt.get("n_groups", n_groups))
                    sd = ckpt.get("state_dict", ckpt)
                else:
                    sd = ckpt
                model = M.build_model(bb, unfr)
                model.load_state_dict(sd, strict=False)
                model.to(device).eval()
                models.append(model)
            uids, preds = T.infer_test_ensemble(
                models, n_groups, device, series_root=series_root)
            T._write_submission(uids, preds, os.path.join(_out_dir(),
                                                          "submission.csv"))
            return
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("WARNING: real inference failed (", e,
                  ") -> writing placeholder submission.csv.")
            _write_placeholder()
            return

    # No data / no checkpoint -> write a correctly-shaped placeholder so the file
    # always exists (the official submission run has the data and overwrites it).
    print("WARNING: DATA_ROOT/test.csv not available in this run -> writing a "
          "placeholder submission.csv (columns correct, dummy rows).")
    _write_placeholder()


def _write_placeholder():
    """Write submission.csv with the correct 13 columns and 3 dummy rows.

    Used when the competition test split isn't mounted in this particular run
    (e.g. a plain `kernels push` validation run). The real submission run
    overwrites it with actual predictions, so the file is always valid-shaped.
    """
    sub = pd.DataFrame({"StudyInstanceUID": [f"PLACEHOLDER_{i}" for i in range(3)]})
    for lab in LABELS:
        sub[lab] = 0.5
    sub = sub[["StudyInstanceUID"] + LABELS]
    sub.to_csv(os.path.join(_out_dir(), "submission.csv"), index=False)
    print("wrote placeholder submission.csv with", len(sub), "rows")



# ===================== RUN LOGIC =====================
# NOTE: config is BAKED IN at build time (Kaggle script kernels cannot receive
# CLI env vars through `kaggle kernels push`), so each experiment is a separate
# build+push with these tokens substituted by kaggle_kernel/_build.py.
CONFIG = dict(
    backbone='tiny_cnn',
    do_submit=False,
    max_studies=None,
    epochs=20,
    folds=5,
    mode='corpus_infer',
    shard=0,
    n_shards=2,
    k_eval=42,
    ckpt=['c9_corpus0.pt'],
    weights=None,
)
from torch.utils.data import DataLoader, Subset
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

def _pick_device():
    """Prefer CUDA (T4 via machine_shape=NvidiaTeslaT4), fall back to CPU.
    Kaggle's default torch (2.10+cu128) has no sm_60 kernels, so a P100 reports
    cuda=True then crashes. Probe with a tiny tensor and fall back first."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        name = torch.cuda.get_device_name(0)
        print("GPU:", name, "| capability:", torch.cuda.get_device_capability(0))
        x = torch.zeros(1, device="cuda")
        _ = (x + 1).item()
        del x
        torch.cuda.empty_cache()
        return "cuda"
    except Exception as e:
        print("WARNING: CUDA unusable (", e, ") -> falling back to CPU")
        return "cpu"


def run():
    print("=== ENV ===")
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
    device = _pick_device()
    print("using device:", device)

    # Always land a scoreable file first; overwritten once real preds exist.
    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    try:
        _write_placeholder()
    except Exception as e:
        print("placeholder write failed:", e)

    data_ok = bool(DATA_ROOT) and os.path.exists(os.path.join(DATA_ROOT, "train.csv"))
    print("=== DATA RESOLUTION ===")
    print("KAGGLE_INPUT candidate:", KAGGLE_INPUT)
    print("/kaggle/input exists:", os.path.isdir("/kaggle/input"))
    if os.path.isdir("/kaggle/input"):
        try:
            print("ls /kaggle/input:", sorted(os.listdir("/kaggle/input"))[:30])
        except Exception as e:
            print("  (ls failed:", e, ")")
    print("resolved DATA_ROOT:", DATA_ROOT, "| train.csv present:", data_ok)

    mode = CONFIG.get("mode", "train")
    backbone = CONFIG["backbone"]
    do_submit = CONFIG["do_submit"]
    max_studies = CONFIG["max_studies"]
    epochs = CONFIG["epochs"]
    folds = CONFIG["folds"]
    print(f"CONFIG mode={mode} backbone={backbone} do_submit={do_submit} "
          f"max_studies={max_studies} epochs={epochs} folds={folds}")
    # Rebind the training DEVICE after the CUDA probe (P100 may fall back).
    global DEVICE
    DEVICE = device

    if not data_ok:
        print("WARNING: competition training data not found in this environment.")
        print("         Writing a placeholder submission.csv; the official submission")
        print("         run will have the data and train for real. Skipping training.")
        main(model_path=None)
        print("DONE (placeholder)")
        return

    if mode == "corpus_dump":
        # CPU-only: decode train DICOMs into one 64-slice shard under /kaggle/working.
        out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
        run_corpus_dump(shard=int(CONFIG.get("shard", 0)),
                        n_shards=int(CONFIG.get("n_shards", 2)),
                        out_dir=out_dir)
        print("DONE")
        return

    if mode == "corpus_infer":
        # Score the hidden test from MOUNTED weights. No training at all.
        # Submitting a training kernel makes Kaggle refit on every rerun (~6h) and
        # cannot score weights trained off-Kaggle; this costs the decode + a forward
        # pass. Geometry (44 vs 64 slices) is read from the checkpoint, not config.
        out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
        run_corpus_infer(out_dir=out_dir, ckpt_names=CONFIG.get("ckpt"),
                         k_eval=CONFIG.get("k_eval"),
                         ckpt_weights=CONFIG.get("weights"))
        print("DONE")
        return

    if mode == "corpus":
        # Train on the published pre-decoded memmap corpus (no DICOM decode).
        out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
        run_corpus_submit(epochs=epochs, out_dir=out_dir,
                          do_submit=bool(do_submit), max_studies=max_studies)
        print("DONE")
        return

    if mode == "raptor":
        # The R line: one study-window model on a high-resolution timm backbone.
        out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
        run_raptor_submit(max_studies=max_studies, epochs=epochs,
                          out_dir=out_dir, do_submit=bool(do_submit))
        print("DONE")
        return

    if mode == "diag":
        # Header-only pass: answers whether the mounted DICOMs carry the fields the
        # sequence recovery needs, before spending a 4-hour run on the assumption.
        run_seq_diag("train")
        print("DONE")
        return

    if mode == "infer":
        # Ensemble mounted fold checkpoints over the test set; no training at all.
        # This is how folds trained in SEPARATE kernels get combined: a code
        # competition reruns the kernel on the hidden test, so cross-run
        # ensembling cannot be done by merging local submission CSVs.
        out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
        run_infer_only(out_dir=out_dir)
        print("DONE")
        return

    ds = RSNADataset("train")
    n_lab = int(ds.has_gold.sum())
    print(f"total studies={len(ds)}  gold-labeled={n_lab}")

    if do_submit:
        # v2: in-memory cache + 2.5D groups + report-hash fold ensemble.
        # Placeholder is already on disk (written at the top of run) so a
        # timeout still leaves a scoreable file; run_submit overwrites it
        # after each finished fold.
        out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
        # ViT-B at 224 px needed batch 4 to fit a T4; at 168 px the activations are
        # (168/224)^2 = 0.56x, so batch 8 fits and the optimisation stays comparable
        # to the ViT-S runs (the reference pipeline also runs its base arms at 8).
        bs = 4 if (backbone == "dinov2_base" and IMG_SIZE > 168) else 8
        run_submit(backbone_name=backbone, unfreeze_last=ENCODER_UNFREEZE_LAST,
                   max_studies=max_studies, epochs=epochs, folds=folds,
                   batch=bs, lr=LR, out_dir=out_dir)
        print("DONE")
        return

    # CV mode: run the fixed-protocol cross-validation for this backbone.
    tag = (f"_{max_studies}" if max_studies else "") + (f"_ep{epochs}" if epochs != 20 else "")
    exp_id = f"{backbone}{tag}"
    cfg = ExpConfig(exp_id=exp_id, backbone=backbone,
                   unfreeze_last=ENCODER_UNFREEZE_LAST, max_studies=max_studies,
                   epochs=epochs, folds=folds)
    run_cv_and_save(cfg)
    print("DONE")

def _c10_solo_check():
    """C9B solo submission (B3, LOG #529/#530): C-line single model scored on the hidden set,
    k_eval 42, all windows, arch vit_base_patch14_dinov2. Fail-closed contract check only; the submission is the
    model's own 12 columns."""
    import pandas as _pd, numpy as _np
    _p = os.path.join(os.environ.get("RSNA_WORK", "/kaggle/working"), "submission.csv")
    _sub = _pd.read_csv(_p, dtype={"StudyInstanceUID": str})
    assert _sub.columns.tolist() == ["StudyInstanceUID", *LABELS], "schema drift"
    assert not _sub["StudyInstanceUID"].str.startswith("PLACEHOLDER").any(), "placeholder rows"
    _v = _sub[LABELS].to_numpy(_np.float64)
    assert _np.isfinite(_v).all(), "non-finite predictions"
    if len(_sub) > 10:
        assert (_v.std(0) > 1e-9).all(), "a label column is constant"
    assert not CORPUS_CENTRE_ALLOW, f"C10 must use ALL windows, got {CORPUS_CENTRE_ALLOW}"
    print(f"[c9b solo] 12 labels kept | {len(_sub)} rows | all windows", flush=True)


if __name__ == "__main__":
    try:
        run()
        _c10_solo_check()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
