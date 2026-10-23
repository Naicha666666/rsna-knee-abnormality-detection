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
