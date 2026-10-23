# =============================================================================
# RSNA Knee — Kaggle Benchmarks task: GAUGE 1 probe for candidate L-line models
# -----------------------------------------------------------------------------
# Framework: https://github.com/Kaggle/kaggle-benchmarks (`@kbench.task`).
# Create the notebook at https://www.kaggle.com/benchmarks/tasks/new (library
# pre-installed), then attach the competition dataset with "Add Input" -- a
# benchmarks notebook does NOT auto-mount it the way a competition notebook does.
#
# TWO CELLS:
#   CELL 1 -- everything that does not depend on the model choice, ending with
#             a printed list of the models your account can actually use.
#   CELL 2 -- you fill in MODEL, then run. Re-run it once per model you want to
#             compare; results accumulate on disk and the summary table at the
#             end scores every model probed so far.
#
# Output lands in /kaggle/working: each conversation appends its row to
# labels_<model>_gold58.jsonl the moment it returns (a kernel restart then
# costs one call, not the sweep), and each finished frame is also written as
# labels_<model>_gold58.parquet.
#
# Reading the result: a candidate has to clear the regex extractor's 0.865 by a
# REAL margin, not noise (Kimi's own run-to-run scatter on these 58 studies is
# 0.018). And a candidate correlated with what's already in the blend
# (kimi/qwen/grok/gemma4) is not worth adding even if its solo GAUGE 1 is high
# -- C3 proved that (regex+kimi+qwen, r=0.94 with kimi, bought +0.0013). It is
# decorrelation with the existing blend, not the solo score, that earns a full
# 4,407-report run.
# =============================================================================

# =============================================================================
# CELL 1 — setup + which models are available
# =============================================================================
import dataclasses
import json
import os

import numpy as np
import pandas as pd
import kaggle_benchmarks as kbench

# Force the plain-text ConsoleUI instead of the default Panel/Bokeh rich-widget
# renderer. We only need the printed results, and the widget comm channel
# ("No such comm target registered: jupyter.widget.control") is a known way for
# a Kaggle kernel session to wedge mid-run.
kbench.config.enable_console_mode(quiet=False, color=None)

OUT_DIR = "/kaggle/working"

# The Model Proxy RESERVES quota per request from `max_output_tokens`, not from
# what the call actually spends. Uncapped, an Opus-class model reserves ~$3.20 a
# call, so once the remaining daily quota drops below that EVERY request is
# refused with `403 ... max estimated cost of operation ($3.20) exceeds your
# available quota (based on max_output_tokens)` -- which is exactly how the
# opus-5 probe died after 11 of 58 studies while actually spending only $0.27.
# This task emits 24 numbers; observed output was 377-830 tokens for the
# non-reasoning models. 2000 leaves generous headroom and cuts the reservation
# by more than an order of magnitude. Raise it for a reasoning model (glm-5 used
# 4451 tokens/call) -- at the cost of a bigger reservation.
#
# The parameter is `max_tokens`, NOT `max_output_tokens`: kbench passes
# extra_api_params straight through to openai's chat.completions.parse()
# (actors/llms.py `_call_api`), which rejects the latter with
# `TypeError: Completions.parse() got an unexpected keyword argument`.
# "max_output_tokens" is only the proxy's own wording in its 403 message.
MAX_OUTPUT_TOKENS = 2000

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
# dataclass field names must be valid Python identifiers -> map field to label
FIELD_TO_LABEL = {
    "acl": "ACL", "mcl": "MCL",
    "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus",
    "medial_oa": "Medial OA", "lateral_oa": "Lateral OA", "pf_oa": "PF OA",
    "effusion": "Effusion", "synovitis": "Synovitis",
    "bakers_cyst": "Baker's", "contusion": "Contusion", "fracture": "Fracture",
}

# Same wording as annotate_gemma4_32c.py / llm_labeler.py's RUBRIC (minus the
# "output raw JSON" instruction -- kbench enforces the schema for us) so a
# GAUGE 1 gap is attributable to the MODEL, not a reworded prompt.
RUBRIC = """You grade knee MRI radiology reports for twelve findings.
Grade the report in whatever language it arrives in. Do not translate.

For EACH of the twelve findings, return two numbers:
  p    probability the finding is present in THIS knee, 0.0 to 1.0
  conf how much the report actually tells you about that finding, 0.0 to 1.0

The twelve findings:
  ACL              anterior cruciate ligament tear/rupture
  MCL              medial collateral ligament injury
  Medial Meniscus  medial meniscus tear
  Lateral Meniscus lateral meniscus tear
  Medial OA        medial compartment osteoarthritis / cartilage loss
  Lateral OA       lateral compartment osteoarthritis / cartilage loss
  PF OA            patellofemoral osteoarthritis / chondromalacia patellae
  Effusion         joint effusion / fluid in the joint
  Synovitis        synovitis, synovial thickening or proliferation
  Baker's          Baker's (popliteal) cyst
  Contusion        bone contusion / bone bruise / marrow oedema from trauma
  Fracture         fracture of any bone in the study"""


# FLAT schema, not a nested Finding(p, conf) dataclass -- a live run showed
# kbench reconstructs the OUTER dataclass but leaves nested dataclass fields as
# plain dicts (`AttributeError: 'dict' object has no attribute 'p'`), so the
# safe contract is one level deep, 24 scalar fields.
@dataclasses.dataclass
class KneeGrades:
    acl_p: float
    acl_conf: float
    mcl_p: float
    mcl_conf: float
    medial_meniscus_p: float
    medial_meniscus_conf: float
    lateral_meniscus_p: float
    lateral_meniscus_conf: float
    medial_oa_p: float
    medial_oa_conf: float
    lateral_oa_p: float
    lateral_oa_conf: float
    pf_oa_p: float
    pf_oa_conf: float
    effusion_p: float
    effusion_conf: float
    synovitis_p: float
    synovitis_conf: float
    bakers_cyst_p: float
    bakers_cyst_conf: float
    contusion_p: float
    contusion_conf: float
    fracture_p: float
    fracture_conf: float


def _get(obj, key):
    """obj[key] for a dict, getattr(obj, key) for a dataclass instance --
    covers both, since which one a given model's structured output lands as is
    exactly the ambiguity that broke the nested-Finding version."""
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "").replace("-", "_")


def _jsonl_path(tag: str) -> str:
    # gold58 runs keep the historical name (the end-of-cell summary globs for
    # it); a full-corpus slice carries its own range in the tag already, so
    # tacking "_gold58" on would be an outright lie about what is in the file.
    suffix = "" if "_full" in tag else "_gold58"
    return os.path.join(OUT_DIR, f"labels_{_slug(tag)}{suffix}.jsonl")


def _read_jsonl(model_id: str) -> pd.DataFrame:
    """Rebuild the label frame from disk, last write per study wins."""
    path = _jsonl_path(model_id)
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows[r["StudyInstanceUID"]] = r
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(list(rows.values()))


# Which model the sweep is currently on. The task function needs it to name its
# output file, and `llm` carries no stable public id, so CELL 2 sets it.
_current_model = {"id": "smoke"}


@kbench.task(name="knee_mri_report_grading")
def grade_report(llm, study_uid: str, report: str) -> dict:
    """Grades one report into the project's 12-label p/conf contract.

    Returns a plain dict (not bool/int/float/tuple) because the goal is to
    collect the structured labels for an OFFLINE GAUGE 1 macro-AUC, not to get
    kbench's own per-row leaderboard score.
    """
    grades = llm.prompt(f"{RUBRIC}\n\n<report>\n{report}\n</report>",
                        schema=KneeGrades,
                        extra_api_params={"max_tokens": MAX_OUTPUT_TOKENS})
    rec = {"StudyInstanceUID": study_uid}
    for field, label in FIELD_TO_LABEL.items():
        rec[label] = float(_get(grades, f"{field}_p"))
        rec[label + "__conf"] = float(_get(grades, f"{field}_conf"))

    # Flush THIS conversation before returning, like annotate_gemma4_32c.py
    # appends per record: a mid-sweep kernel death then costs one call, and the
    # partial file still scores.
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(_jsonl_path(_current_model["id"]), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
    return rec


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Same rank-sum formula as report_extractor.roc_auc / llm_labeler.gauge --
    keeps this number bit-comparable with the existing sources table."""
    yt = np.asarray(y_true, dtype=float)
    n_pos, n_neg = yt.sum(), len(yt) - yt.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_r = ranks[yt == 1].sum()
    return (sum_r - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def gauge1(df_labels: pd.DataFrame, gold_indexed: pd.DataFrame) -> float:
    d = df_labels.set_index("StudyInstanceUID")
    common = [u for u in gold_indexed.index if u in d.index]
    if not common:
        return float("nan")
    y = gold_indexed.loc[common, LABELS].to_numpy(float)
    aucs = []
    for j, lab in enumerate(LABELS):
        col = d.loc[common, lab].to_numpy(float)
        if len(np.unique(y[:, j])) < 2:
            continue
        aucs.append(roc_auc(y[:, j], col))
    return float(np.mean(aucs))


def _find_train_csv():
    """Locate train.csv without assuming a fixed mount path.

    A benchmarks notebook is not the competition notebook -- the data is
    attached by hand and the mount path varies (flat /kaggle/input/<slug>/,
    nested .../competitions/<slug>/, or a different slug if attached as a plain
    dataset). Same problem config._auto_data_root() solves for the training
    pipeline; reimplemented standalone since this file is pasted into a
    notebook rather than importing src/.
    """
    override = os.environ.get("RSNA_DATA_ROOT") or os.environ.get("KAGGLE_DATA_ROOT")
    if override and os.path.exists(os.path.join(override, "train.csv")):
        return os.path.join(override, "train.csv")
    root = "/kaggle/input"
    if os.path.isdir(root):
        for dirpath, _dirs, files in os.walk(root):
            if dirpath[len(root):].count(os.sep) > 3:
                continue
            if "train.csv" in files:
                return os.path.join(dirpath, "train.csv")
    raise FileNotFoundError(
        "train.csv not found under /kaggle/input -- attach the "
        "rsna-knee-abnormality-detection competition dataset with 'Add Input', "
        "or set RSNA_DATA_ROOT. Run `!find /kaggle/input -maxdepth 3` to see "
        "what is actually mounted.")


TRAIN_CSV = _find_train_csv()
tr = pd.read_csv(TRAIN_CSV)
gold_indexed = tr[tr[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")
gold_df = gold_indexed.reset_index().rename(
    columns={"StudyInstanceUID": "study_uid", "Report": "report"})
gold_df["report"] = gold_df["report"].fillna("").astype(str)

full_df = tr.rename(columns={"StudyInstanceUID": "study_uid", "Report": "report"})
full_df["report"] = full_df["report"].fillna("").astype(str)
full_df = full_df[["study_uid", "report"]].reset_index(drop=True)

print(f"train.csv: {TRAIN_CSV}")
print(f"gold studies: {len(gold_df)}   full corpus: {len(full_df)}")


def select_work(scope: str, row_range=None) -> pd.DataFrame:
    """Rows to evaluate: the 58 gold studies, or a slice of the full corpus.

    ROW_RANGE exists because the daily model-proxy quota (~$10) does not cover
    a full 4,407-study sweep in one day. Kaggle starts every batch run with an
    empty /kaggle/working, so there is no cross-run resume on the Kaggle side --
    the split has to be explicit: run rows [0, N) today, download the JSONL,
    run [N, end) tomorrow, and merge the two locally. A run that dies partway
    still leaves its completed rows in the JSONL, so the next range can simply
    be widened to cover whatever is missing.
    """
    if scope == "gold58":
        return gold_df[["study_uid", "report"]]
    if scope != "full":
        raise SystemExit(f"unknown scope {scope!r}")
    if row_range is None:
        return full_df
    a, b = row_range
    return full_df.iloc[a:b]


def list_available_models():
    """Print the model ids this account/token can actually use.

    The library exposes them through `kbench.llms`, but the exact accessor is
    not pinned down in the docs (mapping vs. registry object), so try the
    plausible shapes in order and fall back to showing what the object is --
    better than guessing an id and paying for a failed call.
    """
    print(f"default llm (kbench.llm): {kbench.llm}")
    print(f"judge llm (kbench.judge_llm): {getattr(kbench, 'judge_llm', 'n/a')}\n")

    ids = None
    try:
        ids = list(kbench.llms)
    except Exception:
        pass
    if not ids:
        for attr in ("keys", "available", "names", "list"):
            fn = getattr(kbench.llms, attr, None)
            if fn is None:
                continue
            try:
                ids = list(fn() if callable(fn) else fn)
                break
            except Exception:
                continue
    if not ids:
        env = os.environ.get("LLMS_AVAILABLE")
        if env:
            ids = [x.strip() for x in env.split(",") if x.strip()]

    if ids:
        print(f"{len(ids)} models available:")
        for i in sorted(str(x) for x in ids):
            print(f"  {i}")
    else:
        # Nothing enumerable -- show the object so the id format is visible.
        print("could not enumerate kbench.llms; inspect it directly:")
        print(f"  type: {type(kbench.llms)}")
        print(f"  repr: {kbench.llms!r}"[:2000])
        print(f"  attrs: {[a for a in dir(kbench.llms) if not a.startswith('_')]}")
        print("\nThe notebook's 'Add Models' panel lists the same set.")
    return ids


AVAILABLE_MODELS = list_available_models()

# Also DUMP the list to a file. Kaggle's API-downloadable kernel .log comes back
# empty, so stdout is invisible to anything driving this notebook over the API
# (`kaggle kernels output`) -- only files under /kaggle/working survive. Without
# this, picking a model id from outside the web UI is guesswork, and a wrong
# guess costs a whole run to discover.
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "available_models.json"), "w", encoding="utf-8") as f:
    json.dump({"models": sorted(map(str, AVAILABLE_MODELS or [])),
               "default_llm": str(kbench.llm),
               "judge_llm": str(getattr(kbench, "judge_llm", None))}, f, indent=1)
print(f"model list written to {OUT_DIR}/available_models.json")


# =============================================================================
# CELL 2 — pick the model, then run
# =============================================================================
MODEL = ""  # <-- 填这里: a full id from CELL 1's list, or a substring of one
            #     (e.g. "sonnet") -- resolved against AVAILABLE_MODELS below.
SCOPE = "gold58"   # "gold58" (58 studies) | "full" (4,407)
ROW_RANGE = None   # full scope only: (start, end) row slice, e.g. (0, 2204)
                   # for day 1 of a two-day split. None = the whole corpus.

assert MODEL, "set MODEL to one of the ids printed by CELL 1"


def _resolve_model(want: str) -> str:
    """Accept a substring so a probe does not have to hard-code an exact id.

    The exact ids on a given account are only knowable from CELL 1's output;
    guessing one costs a whole run to find out it was wrong. A unique
    case-insensitive substring match is unambiguous and fails loudly (listing
    the real ids) when it is not unique.
    """
    if not AVAILABLE_MODELS or want in AVAILABLE_MODELS:
        return want
    hits = [m for m in map(str, AVAILABLE_MODELS) if want.lower() in m.lower()]
    if len(hits) == 1:
        print(f"resolved {want!r} -> {hits[0]}")
        return hits[0]
    raise SystemExit(
        f"{want!r} matched {hits or 'nothing'}; pick one of: {sorted(map(str, AVAILABLE_MODELS))}")


MODEL = _resolve_model(MODEL)
# Tag the output file with the slice too, so a two-day split does not have both
# halves writing into one filename that the second run would start empty.
_tag = MODEL if SCOPE == "gold58" else (
    f"{MODEL}_full" + (f"_{ROW_RANGE[0]}_{ROW_RANGE[1]}" if ROW_RANGE else ""))
_current_model["id"] = _tag
llm = kbench.llms[MODEL]

work = select_work(SCOPE, ROW_RANGE)
print(f"scope={SCOPE}  rows={len(work)}  range={ROW_RANGE}  -> {_jsonl_path(_tag)}")

# --- cheap smoke test: ONE call, catches a bad id / schema mismatch before
#     spending a whole sweep of daily credit ---
_smoke_uid, _smoke_report = work.iloc[0][["study_uid", "report"]]
print(grade_report.run(llm=llm, study_uid=_smoke_uid, report=_smoke_report))

# --- the sweep ---
try:
    with kbench.client.enable_cache():
        results = grade_report.evaluate(
            llm=[llm],
            evaluation_data=work,
            on_failure="continue",
            max_attempts=3,
        )
    print(f"completed {len(results.completed_runs)} / {len(work)}, "
          f"errored {len(results.errored_runs)}")
    for run in results.errored_runs:
        print(f"  FAILED {run.params}: {run.error_message[:150]}")
except Exception as e:
    # Score whatever landed on disk rather than losing the spend.
    print(f"evaluate() raised ({type(e).__name__}: {e}); "
          f"scoring whatever was flushed to disk")

# Read back from the JSONL, not from `results` -- it is the only source that
# survives a partial sweep, and it dedupes retries by study.
df_labels = _read_jsonl(_tag)
if df_labels.empty:
    raise SystemExit(f"nothing on disk at {_jsonl_path(_tag)}")

# Score and PRINT before persisting: the number is what the run was paid for,
# and an ImportError in the writer must not swallow it (the benchmarks image
# ships without pyarrow/fastparquet, so to_parquet raises here).
# On a full-corpus slice this scores whatever gold studies happen to fall in
# the slice -- informative, but only the 58-study runs are directly comparable
# to the recorded GAUGE 1 table.
_d = df_labels.set_index("StudyInstanceUID")
_common = [u for u in gold_indexed.index if u in _d.index]
print(f"\n{_tag}: {len(df_labels)} studies graded, {len(_common)} of them gold")
if len(_common) >= 10:
    macro = gauge1(df_labels, gold_indexed)
    print(f"GAUGE 1 macro-AUC = {macro:.3f}  (on {len(_common)} gold studies)")
    for lab in LABELS:
        _y = gold_indexed.loc[_common, lab].to_numpy(float)
        if len(np.unique(_y)) < 2:
            print(f"    {lab:18s} only one class")
            continue
        print(f"    {lab:18s} {roc_auc(_y, _d.loc[_common, lab].to_numpy(float)):.3f}")
else:
    print("(too few gold studies in this slice to score; merge the slices "
          "locally and score there)")

out_base = os.path.join(OUT_DIR, f"labels_{_slug(_tag)}")
try:
    df_labels.to_parquet(out_base + ".parquet", index=False)
    print(f"  wrote {out_base}.parquet")
except ImportError:
    # No parquet engine in this image. The JSONL is already the durable copy
    # and the local repo converts it where pyarrow exists, so CSV is enough.
    df_labels.to_csv(out_base + ".csv", index=False)
    print(f"  no parquet engine here; wrote {out_base}.csv "
          f"(JSONL at {_jsonl_path(_tag)} is the source of truth)")

# --- summary over every model probed so far (re-run CELL 2 per model) ---
summary = {  # LOG.md #314-315, #320-323 -- the bar to clear
    "regex": 0.865, "kimi": 0.876, "qwen": 0.880, "grok": 0.887, "gemma4": 0.840,
}
probed = set()
for fn in sorted(os.listdir(OUT_DIR)):
    if not (fn.startswith("labels_") and fn.endswith("_gold58.jsonl")):
        continue
    name = fn[len("labels_"):-len("_gold58.jsonl")]
    if name == "smoke":
        continue
    df_i = _read_jsonl(name)
    if df_i.empty:
        continue
    summary[name] = gauge1(df_i, gold_indexed)
    probed.add(name)

print("\n=== GAUGE 1 summary (58 gold studies) ===")
for name, auc in sorted(summary.items(), key=lambda kv: -kv[1]):
    print(f"  {name:28s} {auc:.3f}{'  <- probed here' if name in probed else ''}")
