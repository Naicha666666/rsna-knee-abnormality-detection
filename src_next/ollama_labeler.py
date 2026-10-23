"""Ollama backend for the report labelling line (local, free, no credentials).

Kept separate from `llm_labeler.py` on purpose: that module is the Anthropic
path and this one talks to a local Ollama server over its own HTTP API. The
provider-neutral parts -- the grading rubric, the JSON schema, the parquet
layout and the GAUGE 1 scorer -- are imported from there rather than copied, so
the two backends are graded by exactly the same ruler and a comparison between
them means something.

Two Ollama-specific traps this guards against:

  * **Silent context truncation.** Ollama defaults to a small `num_ctx` and
    quietly drops whatever does not fit. The longest report here is 4,743
    characters (~1.9k tokens for mixed Latin/Greek/Cyrillic text) and the rubric
    is ~700, so a default context would truncate the tail of long reports and
    label them from a fragment -- with no error anywhere. `NUM_CTX` is set well
    above the measured worst case, and `check_truncation()` verifies the server
    honoured it.
  * **Thinking output.** Qwen3 emits `<think>...</think>` before its answer
    unless told otherwise. That text is not JSON, so it has to be stripped
    before parsing rather than crashing the run at request 3,000.
"""
from __future__ import annotations
import json
import re
import time

import numpy as np
import pandas as pd

from config import LABELS
from llm_labeler import RUBRIC, _SCHEMA          # one rubric, one schema, both backends

DEFAULT_HOST = "http://192.168.1.177:11436"
# Qwen3.8-27B (Q4_K_M), not the 8B I first assumed "qwen 3.8" meant.
DEFAULT_MODEL = "qwen38-dflash2:latest"
# Longest report ~1.9k tokens + ~700 rubric + ~200 output, with headroom. Ollama
# truncates silently below this, so it is deliberately generous.
NUM_CTX = 8192
TIMEOUT_S = 300

_THINK = re.compile(r"<think>.*?</think>", re.S)


def _strip_think(text: str) -> str:
    """Remove Qwen3 reasoning blocks, then isolate the JSON object."""
    text = _THINK.sub("", text).strip()
    # a model may still wrap JSON in prose or a code fence
    if "```" in text:
        parts = [p for p in text.split("```") if "{" in p]
        if parts:
            text = parts[0]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
    i, j = text.find("{"), text.rfind("}")
    return text[i:j + 1] if i != -1 and j > i else text


def chat(report: str, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL,
         timeout: int = TIMEOUT_S) -> dict:
    """One report -> the 12x(p, conf) dict, via Ollama's /api/chat."""
    import requests

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"<report>\n{report}\n</report>"},
        ],
        # Ollama takes a JSON schema here and constrains decoding to it, which is
        # the same guarantee the Anthropic path gets from output_config.format.
        "format": _SCHEMA,
        "stream": False,
        "think": False,                    # newer Ollama honours this for Qwen3
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }
    r = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    return json.loads(_strip_think(body["message"]["content"]))


def check_truncation(host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL):
    """Confirm the server actually gave us NUM_CTX, not its default.

    A too-small context does not raise -- it silently drops the end of long
    reports. This sends the longest realistic prompt and reads back how many
    tokens the server says it evaluated.
    """
    import requests

    long_report = "Rodilla derecha. " * 400          # ~6.8k chars, worst case
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": RUBRIC},
                     {"role": "user", "content": long_report}],
        "stream": False, "think": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }
    r = requests.post(f"{host}/api/chat", json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    b = r.json()
    used = b.get("prompt_eval_count")
    print(f"prompt_eval_count={used} (num_ctx requested {NUM_CTX})")
    if used is not None and used >= NUM_CTX - 8:
        print("WARNING: prompt filled the context -- longer reports are being "
              "truncated silently; raise NUM_CTX")
    return used


def label(reports: dict, out_parquet: str, host: str = DEFAULT_HOST,
          model: str = DEFAULT_MODEL, progress_every: int = 25,
          checkpoint_every: int = 50):
    """Label {uid: report} sequentially, resumably, into the shared parquet layout.

    The full corpus takes ~10 hours on one local GPU (concurrency does not help;
    the server serialises). A run that long must survive interruption, so
    completed rows are flushed to `out_parquet` periodically and any uid already
    present there is skipped on restart -- rerunning the same command resumes
    instead of starting over.
    """
    import os

    done = {}
    if os.path.exists(out_parquet):
        prev = pd.read_parquet(out_parquet)
        done = {r["StudyInstanceUID"]: r for _, r in prev.iterrows()}
        print(f"resuming: {len(done)} already labelled in {out_parquet}", flush=True)

    todo = [(u, r) for u, r in reports.items() if str(u) not in done]
    print(f"{len(todo)} to label ({len(reports) - len(todo)} skipped)", flush=True)
    recs = [dict(r) for r in done.values()]
    errors = {}
    t0 = time.time()

    def _flush():
        pd.DataFrame(recs).to_parquet(out_parquet, index=False)

    for n, (uid, rep) in enumerate(todo, 1):
        try:
            d = chat(rep, host, model)
            rec = {"StudyInstanceUID": str(uid)}
            for lab in LABELS:
                rec[lab] = float(d[lab]["p"])
                rec[lab + "__conf"] = float(d[lab]["conf"])
            recs.append(rec)
        except Exception as e:
            errors[uid] = f"{type(e).__name__}: {str(e)[:120]}"
        if n % checkpoint_every == 0:
            _flush()
        if n % progress_every == 0:
            el = time.time() - t0
            print(f"  {n}/{len(todo)} | {el / n:.2f}s each | "
                  f"eta {(len(todo) - n) * el / n / 3600:.2f} h | "
                  f"errors {len(errors)}", flush=True)
    _flush()
    df = pd.DataFrame(recs)
    print(f"wrote {out_parquet}: {df.shape} | errors {len(errors)} | "
          f"{(time.time() - t0) / 3600:.2f} h", flush=True)
    if errors:
        print("first errors:", list(errors.items())[:5], flush=True)
    return df, errors


def probe_gold(train_csv: str, out_parquet: str, host: str = DEFAULT_HOST,
               model: str = DEFAULT_MODEL):
    """Label ONLY the 58 gold studies and score them. The decision gate.

    58 requests instead of 4,407: if this does not clear the regex extractor's
    0.865 GAUGE 1, the full run is not worth starting. Costs minutes and nothing
    else, and it is the same comparison the Anthropic path would be judged by.
    """
    from llm_labeler import gauge

    tr = pd.read_csv(train_csv)
    gold = tr[tr[LABELS].notna().all(axis=1)]
    reports = {str(u): r for u, r in
               zip(gold["StudyInstanceUID"], gold["Report"].fillna(""))}
    print(f"probing {len(reports)} gold studies with {model} at {host}", flush=True)
    label(reports, out_parquet, host, model)
    return gauge(out_parquet, train_csv)
