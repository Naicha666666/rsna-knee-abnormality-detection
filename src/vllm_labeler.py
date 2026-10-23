"""Local vLLM (R9700) backend for the report labelling line.

Thin wrapper over `nvidia_labeler`: same rubric, schema, key normalisation,
validation, resumable loop and GAUGE 1 -- only the client points at the
OpenAI-compatible vLLM server on the R9700 instead of the NVIDIA API Catalog.
Muse Glimmer's reasoning is set to `low` (template default is `high`). No rate limit locally, so the pacer is set far above what 16 workers can reach;
concurrency is bounded by `--workers` alone.

Usage:
  python src/vllm_labeler.py smoke
  python src/vllm_labeler.py gold --workers 16
  python src/vllm_labeler.py full --workers 16
  python src/vllm_labeler.py gauge
"""
from __future__ import annotations
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import nvidia_labeler as NV
from llm_labeler import gauge

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://192.168.1.177:8000/v1")
DEFAULT_MODEL = "cyankiwi/Muse-Glimmer-30B-AWQ-INT4"
SLUG = "muse_glimmer"
NO_LIMIT_RPM = 100000

NV.MODEL_OPTS[DEFAULT_MODEL] = {
    "temperature": 0.0,
    "extra_body": {"chat_template_kwargs": {"reasoning_strength": "low"}, "seed": 0},
    "slug": SLUG}


# The container runs without a reasoning parser, so Muse Glimmer's thinking
# (`to=self ...`) comes back inside `content`, followed by `assistant to=user{json}`.
# Grade only what follows the last `to=user`; braces in the thinking would
# otherwise poison NV._extract_json's first-{ / last-} slice.
_nv_extract = NV._extract_json


def _extract_answer(text: str) -> dict:
    k = text.rfind("to=user")
    return _nv_extract(text[k + len("to=user"):] if k != -1 else text)


NV._extract_json = _extract_answer
NV.MAX_TOKENS = 4000           # thinking shares the budget; 8192 context


def _client():
    from openai import OpenAI
    return OpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=300, max_retries=0)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "gold", "full", "gauge"])
    ap.add_argument("--train-csv", default=os.path.join("data", "raw", "train.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    full_out = os.path.join("data", "labels", f"labels_{SLUG}_soft.parquet")

    if a.cmd == "smoke":
        uid, rep = next(iter(NV._reports(a.train_csv, gold_only=True).items()))
        t = time.time()
        d, usage = NV.chat(rep, _client(), a.model, max_attempts=1)
        NV._to_record(uid, d)
        print(f"ok in {time.time() - t:.1f}s | {usage}")
        print(json.dumps(d)[:400])
    elif a.cmd == "gold":
        out = a.out or os.path.join("data", "labels", "gold58_probes",
                                    f"labels_{SLUG}_gold58.parquet")
        NV.label(NV._reports(a.train_csv, gold_only=True), out, model=a.model,
                 workers=a.workers, rpm=NO_LIMIT_RPM, client=_client())
        gauge(out, a.train_csv)
    elif a.cmd == "full":
        out = a.out or full_out
        NV.label(NV._reports(a.train_csv, gold_only=False), out, model=a.model,
                 workers=a.workers, rpm=NO_LIMIT_RPM, client=_client())
        gauge(out, a.train_csv)
    elif a.cmd == "gauge":
        gauge(a.out or full_out, a.train_csv)
