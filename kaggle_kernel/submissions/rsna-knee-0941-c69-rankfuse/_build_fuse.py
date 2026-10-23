"""Build the (C6+C9b) x 0.941-fork rank-fusion notebook (LOG #400).

Derived from the C6 x fork build (LOG #370, LB 0.942). The ONLY change: the embedded
self-developed arm is the C6+C9b equal-weight fusion kernel (LB 0.931, SHA c00a7514)
instead of the C6 script. Fork cells, fork weight 0.5 and the fusion cell are unchanged.
Original notes follow.


Inputs are frozen by SHA-256 so the pushed notebook provably contains
  - the 0.941 Probe #22 fork notebook exactly as submitted (ref 56149721), and
  - the C6 inference script exactly as submitted (ref 56101443, LB 0.935).

The only new code is two cells:
  1. C6 arm, inserted BEFORE the fork. It runs the embedded C6 script in a
     subprocess. That script writes a placeholder /kaggle/working/submission.csv
     on start-up, so it must finish (and its CSV be moved aside) before the fork
     starts writing the same file.
  2. Fusion, appended AFTER the fork. Pre-registered flat weight W_C6 = 0.50,
     per-label percentile ranks. No fallback: if either arm is missing or
     malformed the cell raises, so a scoring failure shows up as an error
     instead of a score that silently belongs to only one arm.

Usage (from this directory):  python _build_fuse.py <fork.ipynb> <c6.py>
"""
import base64
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

FORK_SHA256 = "01f2d834a6c55726c7990e951a96c663405cb2a6bbc4929cae1fc5f07ce53cd6"
C6_SHA256 = "c00a751491ff405808fd03869bc60b162c602c97fe67552d4a90959f9d49544a"   # C6+C9b fusion kernel
W_C6 = 0.50

HERE = Path(__file__).resolve().parent
KERNEL_ID = "easonyanyan/rsna-knee-0941-c69-rankfuse"
CODE_FILE = "rsna-knee-0941-c69-rankfuse.ipynb"
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")
EXTRA_DATASETS = ["easonyanyan/rsna-knee-c6-weights",
                  "easonyanyan/rsna-knee-c9-weights",
                  "easonyanyan/timm-coatnet-rmlp2-384"]


def _cell(kind, text):
    cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


C6_MD = """## C6+C9b arm — runs first (fusion experiment, LOG #400)

The arm below is the C6+C9b equal-weight fusion kernel (LB 0.931), not the C6 script.


Exact code of `easonyanyan/rsna-knee-c6-infer-64slice` v1 (public LB 0.935), embedded
and checked by SHA-256, run in a subprocess. It goes before the 0.941 pipeline because it
writes a placeholder `/kaggle/working/submission.csv` on start-up; its real output is moved
to `/kaggle/working/_c6_arm/` so the unchanged pipeline below starts from a clean file.
"""

C6_CODE = '''# ======================= C6 ARM (subprocess, runs first) =======================
import base64 as _c6_b64, gzip as _c6_gz, hashlib as _c6_h, shutil as _c6_sh
import subprocess as _c6_sp, sys as _c6_sys, time as _c6_t
from pathlib import Path as _C6Path
import numpy as _c6_np, pandas as _c6_pd

C6_SHA256 = "__C6_SHA__"
_C6_B64 = "__C6_B64__"
_c6_src = _c6_gz.decompress(_c6_b64.b64decode(_C6_B64))
assert _c6_h.sha256(_c6_src).hexdigest() == C6_SHA256, "embedded C6 source drifted"

C6_DIR = _C6Path("/kaggle/working/_c6_arm")
C6_DIR.mkdir(parents=True, exist_ok=True)
C6_CSV = C6_DIR / "c6_submission.csv"
_c6_py = C6_DIR / "c6_infer.py"
_c6_py.write_bytes(_c6_src)
_c6_work = _C6Path("/kaggle/working/submission.csv")
if _c6_work.exists():
    _c6_work.unlink()

_c6_t0 = _c6_t.time()
_c6_r = _c6_sp.run([_c6_sys.executable, "-u", str(_c6_py)], cwd=str(C6_DIR),
                   capture_output=True, text=True)
(C6_DIR / "c6_infer.log").write_text(
    _c6_r.stdout + "\\n----- stderr -----\\n" + _c6_r.stderr, encoding="utf-8")
print(_c6_r.stdout[-6000:], flush=True)
if _c6_r.returncode != 0:
    print(_c6_r.stderr[-6000:], flush=True)
    raise RuntimeError(f"C6 arm exited {_c6_r.returncode}")

# The script exits 0 even when it only wrote a placeholder, so check the log for
# the lines that prove the real weights were used on the real test set.
for _c6_must in ("/easonyanyan/rsna-knee-c6-weights/c6_corpus0.pt",
                 "/easonyanyan/rsna-knee-c9-weights/c9_corpus0.pt",
                 "/easonyanyan/timm-coatnet-rmlp2-384/",
                 "weights=[0.5, 0.5]",
                 "state_dict loaded strictly",
                 "wrote /kaggle/working/submission.csv"):
    assert _c6_must in _c6_r.stdout, f"C6+C9b arm log is missing {_c6_must!r}"
assert _c6_r.stdout.count("state_dict loaded strictly") >= 2, "arm did not load both members strictly"
assert "placeholder submission.csv with" not in _c6_r.stdout.split("DATA RESOLUTION")[-1], \\
    "C6 arm fell back to a placeholder after data resolution"

_c6_sh.move(str(_c6_work), str(C6_CSV))
_c6_df = _c6_pd.read_csv(C6_CSV, dtype={"StudyInstanceUID": str})
assert not _c6_df["StudyInstanceUID"].str.startswith("PLACEHOLDER").any()
assert _c6_np.isfinite(_c6_df.iloc[:, 1:].to_numpy(_c6_np.float64)).all()
print(f"[C6 arm] OK: {len(_c6_df)} studies in {(_c6_t.time() - _c6_t0) / 60:.1f} min "
      f"-> {C6_CSV}", flush=True)
'''

FUSE_MD = """## (C6+C9b) × 0.941 rank fusion (LOG #400)

`W_C6` below is the weight of the self-developed arm, which is now C6+C9b. Kept at 0.50 as in the 0.942 run. Falsification: public LB <= 0.946.


Pre-registered, not tuned on the leaderboard: one flat weight `W_C6 = 0.50` for every
finding. Both submissions become per-label percentile ranks (`method="average"`); AUC only
reads order. Falsification: public LB ≤ 0.941 → C6 adds nothing the fork does not already have.
"""

FUSE_CODE = '''# ========================= C6 x 0.941 RANK FUSION ===========================
import hashlib as _fu_h, json as _fu_json
from pathlib import Path as _FuPath
import numpy as _fu_np, pandas as _fu_pd

W_C6 = __W_C6__
FUSE_LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
               "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
               "Contusion", "Fracture"]
_fu_cols = ["StudyInstanceUID"] + FUSE_LABELS
_fu_sub = _FuPath("/kaggle/working/submission.csv")
_fu_fork_copy = _FuPath("/kaggle/working/_fork_0941_submission.csv")

_fu_test = None
for _fu_cand in ("/kaggle/input/competitions/rsna-knee-abnormality-detection/test.csv",
                 "/kaggle/input/rsna-knee-abnormality-detection/test.csv"):
    if _FuPath(_fu_cand).exists():
        _fu_test = _fu_pd.read_csv(_fu_cand, dtype={"StudyInstanceUID": str})
        break
assert _fu_test is not None, "competition test.csv not found"
_fu_uids = set(_fu_test["StudyInstanceUID"].astype(str))

_fu_fork = _fu_pd.read_csv(_fu_sub, dtype={"StudyInstanceUID": str})
_fu_c6 = _fu_pd.read_csv(C6_CSV, dtype={"StudyInstanceUID": str})
for _fu_name, _fu_df in (("fork", _fu_fork), ("c6", _fu_c6)):
    assert _fu_df.columns.tolist() == _fu_cols, f"{_fu_name} schema: {_fu_df.columns.tolist()}"
    assert _fu_df["StudyInstanceUID"].is_unique, f"{_fu_name} duplicate UIDs"
    assert set(_fu_df["StudyInstanceUID"]) == _fu_uids, f"{_fu_name} UIDs != test.csv"
    assert _fu_np.isfinite(_fu_df[FUSE_LABELS].to_numpy(_fu_np.float64)).all(), f"{_fu_name} non-finite"
if len(_fu_fork) > 1:
    # the fork's Stage-1 fallback writes 0.5 everywhere; a constant file is not a real arm
    assert not (_fu_fork[FUSE_LABELS].nunique() == 1).all(), "fork output is constant"

_fu_c6 = _fu_c6.set_index("StudyInstanceUID").loc[_fu_fork["StudyInstanceUID"]].reset_index()
_fu_fr = _fu_fork[FUSE_LABELS].rank(method="average", pct=True)
_fu_cr = _fu_c6[FUSE_LABELS].rank(method="average", pct=True)

_fu_fork.to_csv(_fu_fork_copy, index=False)
_fu_out = _fu_fork[["StudyInstanceUID"]].copy()
for _fu_lab in FUSE_LABELS:
    _fu_out[_fu_lab] = (1.0 - W_C6) * _fu_fr[_fu_lab].to_numpy() + W_C6 * _fu_cr[_fu_lab].to_numpy()
assert _fu_np.isfinite(_fu_out[FUSE_LABELS].to_numpy(_fu_np.float64)).all()
_fu_out.to_csv(_fu_sub, index=False)


def _fu_sha(p):
    return _fu_h.sha256(_FuPath(p).read_bytes()).hexdigest()


_fu_agree = {lab: (float(_fu_fr[lab].corr(_fu_cr[lab])) if len(_fu_fr) > 2 else None)
             for lab in FUSE_LABELS}
_fu_receipt = {
    "experiment": "(C6+C9b) x 0.941 fork rank fusion (LOG #400)",
    "w_c6": W_C6, "n_studies": int(len(_fu_out)),
    "fork_sha256": _fu_sha(_fu_fork_copy), "c6_sha256": _fu_sha(C6_CSV),
    "fused_sha256": _fu_sha(_fu_sub), "c6_source_sha256": C6_SHA256,
    "rank_corr_fork_vs_c6": _fu_agree,
}
_FuPath("/kaggle/working/_fusion_receipt.json").write_text(
    _fu_json.dumps(_fu_receipt, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(_fu_json.dumps(_fu_receipt, indent=2, sort_keys=True), flush=True)
print("[fusion] PASS ->", _fu_sub, flush=True)
'''


def main(fork_path, c6_path):
    fork_bytes = Path(fork_path).read_bytes()
    c6_bytes = Path(c6_path).read_bytes()
    assert hashlib.sha256(fork_bytes).hexdigest() == FORK_SHA256, "fork notebook is not the submitted one"
    assert hashlib.sha256(c6_bytes).hexdigest() == C6_SHA256, "C6 script is not the submitted one"

    nb = json.loads(fork_bytes.decode("utf-8"))
    n_fork = len(nb["cells"])
    b64 = base64.b64encode(gzip.compress(c6_bytes, compresslevel=9, mtime=0)).decode("ascii")
    c6_code = C6_CODE.replace("__C6_SHA__", C6_SHA256).replace("__C6_B64__", b64)
    fuse_code = FUSE_CODE.replace("__W_C6__", repr(W_C6))

    out = copy.deepcopy(nb)
    # after the title markdown (cell 0), before "## 0. Run configuration"
    out["cells"][1:1] = [_cell("markdown", C6_MD), _cell("code", c6_code)]
    out["cells"] += [_cell("markdown", FUSE_MD), _cell("code", fuse_code)]
    assert len(out["cells"]) == n_fork + 4
    # every original cell is still there, in order, unmodified
    assert out["cells"][0] == nb["cells"][0]
    assert out["cells"][3:3 + n_fork - 1] == nb["cells"][1:]

    (HERE / CODE_FILE).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    meta_src = json.loads((Path(fork_path).parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta = {
        "id": KERNEL_ID,
        "title": "RSNA Knee 0941 C69 RankFuse",
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": list(meta_src["dataset_sources"]) + EXTRA_DATASETS,
        "kernel_sources": list(meta_src["kernel_sources"]),
        "competition_sources": list(meta_src["competition_sources"]),
        "model_sources": list(meta_src["model_sources"]),
        "docker_image": DOCKER_IMAGE,
        "machine_shape": "NvidiaTeslaT4",
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"fork cells {n_fork} -> fused {len(out['cells'])}; C6 embed {len(b64)} b64 chars; "
          f"datasets {len(meta['dataset_sources'])}; W_C6={W_C6}")
    print("notebook sha256:", hashlib.sha256((HERE / CODE_FILE).read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
