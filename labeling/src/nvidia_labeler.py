"""NVIDIA API Catalog (Nemotron) backend for the report labelling line.

Same arrangement as `moonshot_labeler`: its own file because the provider's calls
do not belong in another provider's module, while the rubric, the JSON schema and
the GAUGE 1 scorer are imported from `llm_labeler` -- one ruler for every backend.

Provider constraints this handles:

  * **40 requests per minute.** A shared pacer spaces request STARTS at least
    `60 / RPM` seconds apart across all worker threads, and a 429 backs off
    exponentially instead of hammering the limit. One report per request (like
    Kimi/Qwen), so a bad answer costs one re-request, not a whole chunk.
  * **Reasoning model.** Nemotron thinks when `enable_thinking` is on; for
    grading 12 fields that only burns output budget (Kimi returned EMPTY content
    that way), so it is switched off via `chat_template_kwargs`.
  * **Resumable.** Rows are flushed every 50 and any uid already in the parquet
    is skipped, so rerunning the same command after an interruption (or to retry
    the failures) continues instead of starting over.

The API key is read from `NVIDIA_API_KEY` only -- never written to a file.
"""
from __future__ import annotations
import json
import os
import sys
import threading
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import LABELS
from llm_labeler import RUBRIC, _SCHEMA, gauge  # one rubric, one schema, one ruler

BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
MAX_TOKENS = 2000
RPM = 38                       # provider limit is 40; keep a little headroom
TEMPERATURE = 0.0

# Per-model request knobs: reasoning OFF for grading (it only burns budget), and
# the sampling settings each model accepts. Kimi's temperature is locked at 1 on
# Moonshot's own API, so it is kept at 1 here with a fixed seed.
MODEL_OPTS = {
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "slug": "nemotron3_ultra"},
    "moonshotai/kimi-k3": {
        "temperature": 1.0,
        "extra_body": {"reasoning_effort": "none", "seed": 0},
        "slug": "nv_kimi_k3"},
}


def _client():
    from openai import OpenAI

    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise SystemExit("set NVIDIA_API_KEY in the environment "
                         "(never commit it to a file)")
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=180, max_retries=0)


class _Pacer:
    """Spaces request starts >= 60/rpm seconds apart, across threads."""

    def __init__(self, rpm: int):
        self.gap = 60.0 / rpm
        self.next_t = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.gap
        if t > now:
            time.sleep(t - now)


def _extract_json(text: str) -> dict:
    """Isolate the JSON object even if wrapped in a code fence or prose."""
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j <= i:
        raise ValueError(f"no JSON object in content: {text[:80]!r}")
    return json.loads(text[i:j + 1])


def _norm(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _to_record(uid: str, d: dict) -> dict:
    """Validate the 12x(p, conf) answer; raise rather than store a bad row.

    No decoding constraint here, and Nemotron occasionally spells the label keys
    without spaces/apostrophes ("MedialMeniscus", "Bakers") -- measured on 2/58
    gold reports. Keys are matched on their alphanumeric-lowercase form, which
    is unambiguous across the 12 labels; anything still missing raises.
    """
    by_norm = {_norm(k): v for k, v in d.items()}
    rec = {"StudyInstanceUID": uid}
    for lab in LABELS:
        v = by_norm[_norm(lab)]
        p, c = float(v["p"]), float(v["conf"])
        if not (0.0 <= p <= 1.0 and 0.0 <= c <= 1.0):
            raise ValueError(f"{lab} out of range: p={p} conf={c}")
        rec[lab] = p
        rec[lab + "__conf"] = c
    return rec


def chat(report: str, client=None, model: str = DEFAULT_MODEL,
         pacer: _Pacer | None = None, max_attempts: int = 6) -> tuple[dict, dict]:
    """One report -> (12x(p, conf) dict, usage info). Retries 429/5xx."""
    from openai import APIStatusError, APIConnectionError, APITimeoutError

    client = client or _client()
    opts = MODEL_OPTS.get(model, {"temperature": TEMPERATURE, "extra_body": {}})
    for attempt in range(max_attempts):
        if pacer:
            pacer.wait()
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": RUBRIC},
                          {"role": "user", "content": f"<report>\n{report}\n</report>"}],
                temperature=opts["temperature"],
                max_tokens=MAX_TOKENS,
                extra_body=opts["extra_body"],
                stream=False,
            )
        except APIStatusError as e:
            if e.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise
        except (APIConnectionError, APITimeoutError):
            if attempt < max_attempts - 1:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise
        ch = r.choices[0]
        if not ch.message.content:
            raise RuntimeError(f"empty content (finish={ch.finish_reason}) "
                               f"-- reasoning likely still on")
        usage = {"finish": ch.finish_reason,
                 "out_tokens": getattr(r.usage, "completion_tokens", None),
                 "in_tokens": getattr(r.usage, "prompt_tokens", None)}
        return _extract_json(ch.message.content), usage
    raise RuntimeError("unreachable")


def label(reports: dict, out_parquet: str, model: str = DEFAULT_MODEL,
          workers: int = 3, rpm: int = RPM, progress_every: int = 25,
          checkpoint_every: int = 50, client=None):
    """Label {uid: report} under the RPM limit, resumably, into the shared layout.

    `client` lets another OpenAI-compatible backend (e.g. local vLLM) reuse this
    loop; default is the NVIDIA API Catalog client."""
    from concurrent.futures import ThreadPoolExecutor

    done = {}
    if os.path.exists(out_parquet):
        prev = pd.read_parquet(out_parquet)
        done = {r["StudyInstanceUID"]: dict(r) for _, r in prev.iterrows()}
        print(f"resuming: {len(done)} already labelled", flush=True)
    todo = [(str(u), r) for u, r in reports.items() if str(u) not in done]
    print(f"{len(todo)} to label with {model} ({workers} workers, {rpm} rpm)",
          flush=True)

    client, pacer = client or _client(), _Pacer(rpm)
    recs, errors = list(done.values()), {}
    t0 = time.time()

    def _one(item):
        uid, rep = item
        try:
            d, _ = chat(rep, client, model, pacer)
            return uid, _to_record(uid, d), None
        except Exception as e:
            return uid, None, f"{type(e).__name__}: {str(e)[:160]}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, (uid, rec, err) in enumerate(ex.map(_one, todo), 1):
            if err:
                errors[uid] = err
            else:
                recs.append(rec)
            if n % checkpoint_every == 0:
                pd.DataFrame(recs).to_parquet(out_parquet, index=False)
            if n % progress_every == 0:
                el = time.time() - t0
                print(f"  {n}/{len(todo)} | {el / n:.2f}s each | "
                      f"eta {(len(todo) - n) * el / n / 60:.1f} min | "
                      f"errors {len(errors)}", flush=True)
    df = pd.DataFrame(recs)
    df.to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet}: {df.shape} | errors {len(errors)} | "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if errors:
        print("first errors:", list(errors.items())[:3], flush=True)
    return df, errors


def _reports(train_csv: str, gold_only: bool) -> dict:
    tr = pd.read_csv(train_csv)
    if gold_only:
        tr = tr[tr[LABELS].notna().all(axis=1)]
    return {str(u): r for u, r in zip(tr["StudyInstanceUID"], tr["Report"].fillna(""))}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "gold", "full", "gauge"])
    ap.add_argument("--train-csv", default=os.path.join("data", "raw", "train.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rpm", type=int, default=RPM)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    slug = MODEL_OPTS.get(a.model, {}).get("slug") or _norm(a.model)

    if a.cmd == "smoke":
        uid, rep = next(iter(_reports(a.train_csv, gold_only=True).items()))
        t = time.time()
        d, usage = chat(rep, model=a.model, max_attempts=1)
        _to_record(uid, d)
        print(f"ok in {time.time() - t:.1f}s | {usage}")
        print(json.dumps(d)[:400])
    elif a.cmd == "gold":
        out = a.out or os.path.join("data", "labels", "gold58_probes",
                                    f"labels_{slug}_gold58.parquet")
        label(_reports(a.train_csv, gold_only=True), out, model=a.model,
              workers=a.workers, rpm=a.rpm)
        gauge(out, a.train_csv)
    elif a.cmd == "full":
        out = a.out or os.path.join("data", "labels", "labels_nemotron_soft.parquet")
        label(_reports(a.train_csv, gold_only=False), out, workers=a.workers,
              rpm=a.rpm)
        gauge(out, a.train_csv)
    elif a.cmd == "gauge":
        gauge(a.out or os.path.join("data", "labels", "labels_nemotron_soft.parquet"),
              a.train_csv)
