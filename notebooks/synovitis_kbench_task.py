# =============================================================================
# RSNA Knee — Synovitis-only labelling with openai/gpt-5.6-luna
# -----------------------------------------------------------------------------
# Why only Synovitis: it is the one label where every source is weak. Across 11
# candidates measured on the 58 gold studies, Synovitis sits at 0.652-0.809
# while the same models reach 0.88-0.99 on the other eleven. luna's 0.808 ties
# sonnet-5's 0.809 for best, at 1/7 the price -- hence luna for the full run.
#
# Why this file exists separately from gauge1_kbench_task.py: that script's
# 0.808 was produced by the FULL twelve-finding rubric. Asking for one finding
# is a DIFFERENT PROMPT, and deploying an unvalidated prompt across 4,407
# reports is how you spend $9 to learn nothing. So:
#
#   SCOPE = "gold58"  -> 58 studies, ~$0.13, checks the short prompt still
#                        lands near 0.808 before committing to the full run
#   SCOPE = "full"    -> all 4,407, ~$9
#
# The rubric below is a STRICT SUBSET of the twelve-label one: the other eleven
# findings are deleted and nothing is added. Any score difference is therefore
# attributable to "fewer labels in context", not to reworded instructions --
# the same single-variable discipline the rest of this project runs on.
#
# TWO CELLS, same shape as gauge1_kbench_task.py:
#   CELL 1 -- setup + task definition (no model calls)
#   CELL 2 -- pick SCOPE, run
# =============================================================================

# =============================================================================
# CELL 1 — setup
# =============================================================================
import dataclasses
import json
import os

import numpy as np
import pandas as pd
import kaggle_benchmarks as kbench

kbench.config.enable_console_mode(quiet=False, color=None)

OUT_DIR = "/kaggle/working"
MODEL = "openai/gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 300          # one finding = two numbers; 377 was the
                                  # 12-label figure, so this is generous.
                                  # Keeps the proxy's quota reservation small
                                  # (see LOG.md #350: the reservation, not the
                                  # spend, is what refuses requests).

# Verbatim subset of the twelve-label rubric: the p/conf semantics and the
# silence convention are unchanged, only the other eleven findings removed.
RUBRIC = """You grade knee MRI radiology reports for one finding.
Grade the report in whatever language it arrives in. Do not translate.

Return two numbers:
  p    probability the finding is present in THIS knee, 0.0 to 1.0
  conf how much the report actually tells you about that finding, 0.0 to 1.0

The finding:
  Synovitis        synovitis, synovial thickening or proliferation"""


@dataclasses.dataclass
class SynovitisGrade:
    synovitis_p: float
    synovitis_conf: float


def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _jsonl_path(scope: str) -> str:
    return os.path.join(OUT_DIR, f"labels_luna_synovitis_{scope}.jsonl")


_scope = {"name": "gold58"}


def _read_jsonl(scope: str) -> pd.DataFrame:
    path = _jsonl_path(scope)
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


@kbench.task(name="knee_synovitis_grading")
def grade_synovitis(llm, study_uid: str, report: str) -> dict:
    """Grades one report for Synovitis alone."""
    g = llm.prompt(f"{RUBRIC}\n\n<report>\n{report}\n</report>",
                   schema=SynovitisGrade,
                   extra_api_params={"max_tokens": MAX_OUTPUT_TOKENS})
    rec = {"StudyInstanceUID": study_uid,
           "Synovitis": float(_get(g, "synovitis_p")),
           "Synovitis__conf": float(_get(g, "synovitis_conf"))}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(_jsonl_path(_scope["name"]), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
    return rec


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    n_pos, n_neg = yt.sum(), len(yt) - yt.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    return (ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


ALL_LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def _find_train_csv():
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
        "train.csv not found under /kaggle/input -- attach the competition "
        "dataset with 'Add Input', or set RSNA_DATA_ROOT.")


TRAIN_CSV = _find_train_csv()
tr = pd.read_csv(TRAIN_CSV)
is_gold = tr[ALL_LABELS].notna().all(axis=1)
gold_indexed = tr[is_gold].set_index("StudyInstanceUID")
print(f"train.csv: {TRAIN_CSV}   total {len(tr)}   gold {len(gold_indexed)}")


# =============================================================================
# CELL 2 — pick the scope, then run
# =============================================================================
SCOPE = "gold58"      # <-- "gold58" to validate the short prompt (~$0.13),
                      #     "full" for all 4,407 (~$9) once it checks out
assert SCOPE in ("gold58", "full")
_scope["name"] = SCOPE

subset = tr[is_gold] if SCOPE == "gold58" else tr
work = subset.rename(columns={"StudyInstanceUID": "study_uid", "Report": "report"})
work["report"] = work["report"].fillna("").astype(str)
work = work[["study_uid", "report"]]
print(f"scope={SCOPE}  studies={len(work)}")

llm = kbench.llms[MODEL]
try:
    with kbench.client.enable_cache():
        results = grade_synovitis.evaluate(
            llm=[llm], evaluation_data=work,
            on_failure="continue", max_attempts=3,
        )
    print(f"completed {len(results.completed_runs)} / {len(work)}, "
          f"errored {len(results.errored_runs)}")
    for run in list(results.errored_runs)[:10]:
        print(f"  FAILED {run.params}: {run.error_message[:150]}")
except Exception as e:
    print(f"evaluate() raised ({type(e).__name__}: {e}); scoring what landed on disk")

df = _read_jsonl(SCOPE)
if df.empty:
    raise SystemExit(f"nothing at {_jsonl_path(SCOPE)}")
print(f"\n{len(df)} rows on disk -> {_jsonl_path(SCOPE)}")

# Score whatever gold studies are covered -- for SCOPE="full" this is the same
# GAUGE 1 protocol, just computed on the 58 gold rows inside the full sweep.
d = df.drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")
common = [u for u in gold_indexed.index if u in d.index]
if common:
    y = gold_indexed.loc[common, "Synovitis"].to_numpy(float)
    auc = roc_auc(y, d.loc[common, "Synovitis"].to_numpy(float))
    print(f"\nSynovitis AUC on {len(common)} gold studies = {auc:.4f}")
    print("  reference, SAME model with the 12-label prompt: 0.808")
    print("  other sources: sonnet-5 0.809 | opus-5 0.796 | C6 blend 0.763 | "
          "gemini-flash 0.726 | glm-5 0.698 | chatgpt-5.5 0.678 | glm-5.2 0.669 | "
          "hunyuan 0.673 | deepseek 0.670 | minimax 0.665 | qwen-max 0.665")
    if SCOPE == "gold58":
        print("\n  -> if this is near 0.808, the short prompt is sound and the "
              "full run is worth its ~$9. If it is materially lower, the "
              "12-label prompt is doing work and the short one should not ship.")

out_csv = os.path.join(OUT_DIR, f"labels_luna_synovitis_{SCOPE}.csv")
df.to_csv(out_csv, index=False)
print(f"wrote {out_csv}")
