# --- v4 configuration ----------------------------------------------------------------
# Only the knobs this version adds or changes. Everything above keeps its original value,
# so setting all of these to their "off" state reproduces the previous pipeline exactly.

import math
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

N_FOLDS       = 4          # report-hash groups; folds are trained until the budget runs out
MAX_FOLDS     = "auto"     # "auto" fits as many as the time budget allows, or an int
FOLD_TIME_PAD = 1.15       # safety factor on the measured per-fold time

USE_VFLIP     = False      # was True; a knee is not vertically symmetric
AUG_AFFINE    = True
AUG_ROT_DEG   = 8.0
AUG_SCALE     = 0.08
AUG_SHIFT     = 0.05
AUG_INTENSITY = 0.10

EMA_DECAY     = 0.997      # 0 disables the moving average
RANK_LOSS_W   = 0.05       # 0 disables the pairwise term
RANK_POS, RANK_NEG = 0.60, 0.40      # graded-target cutoffs that define a usable pair

LAT_FALLBACK        = "auto"   # "auto" | "on" | "off": infer side from x when the tag is absent
LAT_MIN_AGREEMENT   = 0.85     # required agreement with the tag before "auto" enables it
LAT_MIN_OFFSET_MM   = 5.0      # |x| below this is not a usable side cue

SELECTION     = "worse_of_two"   # "worse_of_two" | "derived"
GOLD_WEIGHT   = 3.0              # weight of an annotated study relative to a derived one

# ------------------------------------------------------------------ icefire theme
ICEFIRE  = sns.color_palette("icefire", 12)
ICE      = ICEFIRE[2]
FIRE     = ICEFIRE[9]
MIDLINE  = "#39414d"
CMAP     = "icefire"
CMAP_SEQ = sns.blend_palette(["#eef1f6", ICE, FIRE], as_cmap=True)

sns.set_theme(style="white", palette=ICEFIRE, font_scale=0.95)
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c8cdd4", "axes.labelcolor": MIDLINE,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 10,
    "text.color": MIDLINE, "xtick.color": MIDLINE, "ytick.color": MIDLINE,
    "grid.color": "#e8eaee", "axes.grid": True, "axes.grid.axis": "y",
    "legend.frameon": False, "figure.facecolor": "white",
})


def grad(n, reverse=False):
    """n colours along the icefire ramp, skipping its near-black centre."""
    cm = plt.get_cmap(CMAP)
    cold = np.linspace(0.02, 0.28, n - n // 2)
    warm = np.linspace(0.78, 0.99, n // 2)
    cols = [cm(p) for p in np.concatenate([cold, warm])][:n]
    return cols[::-1] if reverse else cols


# `log` is defined in the next cell, so this one reports with print().
print(f"v4: folds={N_FOLDS} vflip={USE_VFLIP} ema={EMA_DECAY} rank_w={RANK_LOSS_W} "
      f"lat_fallback={LAT_FALLBACK}")