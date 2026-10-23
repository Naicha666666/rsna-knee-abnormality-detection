"""Build a single self-contained run.py for Kaggle by inlining all src modules.

Why single-file: Kaggle script kernels execute the entry as /kaggle/src/script.py
and do NOT place sibling modules on sys.path, so a flat layout fails with
`No module named 'config'`. Inlining is the only layout guaranteed to work.

Tricks (verified against earlier failed attempts):
- Hoist a single `from __future__ import annotations` to the very top; strip any
  other `from __future__` lines.
- Register the combined module under every project name (config, dicom_utils,
  report_extractor, dataset, backbones, backbones.dinov2, backbones.tiny_cnn,
  model, train, experiments, make_submission) in sys.modules BEFORE any section
  runs, so `from config import ...` / `from backbones import ...` resolve to the
  same combined namespace.
- Rename dataset's `import report_extractor as re` -> `_rext` so it does NOT
  clobber the regex `re` used inside report_extractor (single shared namespace).
- Strip `if __name__ == "__main__":` blocks from library sections.
- train.py IS included now (the comparison study drives CV through it).
"""
import os
import re

base = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(os.path.dirname(base), "src")
bdir = os.path.join(src, "backbones")

order = [
    "config",
    "tv_stub",
    "dicom_utils",
    "report_extractor",
    "dataset",
    "backbones/dinov2",
    "backbones/tiny_cnn",
    "model",
    "train",
    "raptor",
    "corpus",
    "experiments",
    "make_submission",
]
NAMES = [
    "config", "tv_stub", "dicom_utils", "report_extractor", "dataset",
    "backbones", "backbones.dinov2", "backbones.tiny_cnn",
    "model", "train", "raptor", "corpus", "experiments", "make_submission",
]

header = '''from __future__ import annotations
import os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_THIS = sys.modules[__name__]
for _n in {names!r}:
    sys.modules[_n] = _THIS
import torch, numpy as np, pandas as pd
'''.format(names=NAMES)

runlogic = r'''
# ===================== RUN LOGIC =====================
# NOTE: config is BAKED IN at build time (Kaggle script kernels cannot receive
# CLI env vars through `kaggle kernels push`), so each experiment is a separate
# build+push with these tokens substituted by kaggle_kernel/_build.py.
CONFIG = dict(
    backbone=__BACKBONE__,
    do_submit=__DO_SUBMIT__,
    max_studies=__MAX_STUDIES__,
    epochs=__EPOCHS__,
    folds=__FOLDS__,
    mode=__MODE__,
    shard=__SHARD__,
    n_shards=__N_SHARDS__,
    k_eval=__K_EVAL__,
    ckpt=__CKPT__,
    weights=__WEIGHTS__,
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

if __name__ == "__main__":
    try:
        run()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

FUTURE_RE = re.compile(r"^\s*from __future__ import annotations\s*$")


def strip_future(txt: str) -> str:
    return "\n".join(l for l in txt.splitlines() if not FUTURE_RE.match(l))


def strip_main(txt: str) -> str:
    i = txt.find('if __name__ == "__main__":')
    if i != -1:
        txt = txt[:i].rstrip() + "\n"
    return txt


def patch_dataset(txt: str) -> str:
    txt = txt.replace("import report_extractor as re", "import report_extractor as _rext")
    txt = txt.replace("re.extract", "_rext.extract")
    return txt


def module_path(name: str) -> str:
    if name.startswith("backbones/"):
        return os.path.join(bdir, name[len("backbones/"):] + ".py")
    return os.path.join(src, name + ".py")


parts = [header]
for name in order:
    txt = open(module_path(name), encoding="utf-8").read()
    txt = strip_future(txt)
    txt = strip_main(txt)
    if name == "dataset":
        txt = patch_dataset(txt)
    parts.append(f"# ===================== {name}.py =====================\n" + txt)

parts.append(runlogic)
out = "\n\n".join(parts)

# Bake the experiment config into run.py (Kaggle script kernels can't receive
# CLI env vars through `kaggle kernels push`). Each experiment = one build+push.
_backbone = os.environ.get("RSNA_BACKBONE", "tiny_cnn")
_do_submit = os.environ.get("RSNA_DO_SUBMIT", "0") == "1"
_ms = os.environ.get("RSNA_MAX_STUDIES")
_max_studies = None if (not _ms or str(_ms).lower() in ("none", "0", "full")) else int(_ms)
_epochs = int(os.environ.get("RSNA_EPOCHS", "20"))
_seed = int(os.environ.get("RSNA_SEED", "42"))
_mode = os.environ.get("RSNA_MODE", "train")
assert _mode in ("train", "infer", "diag", "raptor", "corpus", "corpus_dump",
                 "corpus_infer"), _mode
_ke = os.environ.get("RSNA_K_EVAL")
_k_eval = int(_ke) if _ke else None
_cw = os.environ.get("RSNA_CKPT_WEIGHTS")
_weights = [float(x) for x in _cw.split(",") if x.strip()] if _cw else None
_ck = os.environ.get("RSNA_CKPT")
_ckpt = [x.strip() for x in _ck.split(",") if x.strip()] if _ck else None
_folds = int(os.environ.get("RSNA_FOLDS", "5"))
_shard = int(os.environ.get("RSNA_SHARD", "0"))
_n_shards = int(os.environ.get("RSNA_N_SHARDS", "2"))
out = (out.replace("__BACKBONE__", repr(_backbone))
          .replace("__DO_SUBMIT__", repr(_do_submit))
          .replace("__MAX_STUDIES__", repr(_max_studies))
          .replace("__EPOCHS__", repr(_epochs))
          .replace("__FOLDS__", repr(_folds))
          .replace("__MODE__", repr(_mode))
          .replace("__SHARD__", repr(_shard))
          .replace("__N_SHARDS__", repr(_n_shards))
          .replace("__K_EVAL__", repr(_k_eval))
          .replace("__CKPT__", repr(_ckpt))
          .replace("__WEIGHTS__", repr(_weights)))
# CORPUS_LEGAL_WINDOWS is a plain config constant too; bake it the same way.
if os.environ.get("RSNA_LEGAL_WINDOWS") == "1":
    _lw_src = "CORPUS_LEGAL_WINDOWS = False"
    assert out.count(_lw_src) == 1, "config.CORPUS_LEGAL_WINDOWS assignment not found"
    out = out.replace(_lw_src, "CORPUS_LEGAL_WINDOWS = True")
    print("baked CORPUS_LEGAL_WINDOWS = True")
# SEED is a plain constant in config.py, not a CONFIG token, so it is baked by
# rewriting the assignment (a seed repeat is a separate kernel, not a CLI flag).
if _seed != 42:
    _seed_src = "SEED = 42" + chr(10)
    assert out.count(_seed_src) == 1, "config.SEED assignment not found"
    out = out.replace(_seed_src, "SEED = %d%s" % (_seed, chr(10)))

open(os.path.join(base, "run.py"), "w", encoding="utf-8").write(out)
print("run.py written:", len(out), "bytes,", out.count("\n"), "lines,",
      "modules:", order, "| config:", _mode, _backbone, "submit", _do_submit,
      "max_studies", _max_studies, "epochs", _epochs, "folds", _folds)
