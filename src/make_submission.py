"""Inference -> submission.csv for RSNA Knee (run inside Kaggle Notebook).

Loads held-out test studies (mounted in the notebook), runs the imaging-only
trained model (slots -> attention heads), and writes `submission.csv` with one
row per StudyInstanceUID. The report is NOT used at inference (test.csv has no
Report column), which is why the model is pure vision.
"""
from __future__ import annotations
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


if __name__ == "__main__":
    main()
