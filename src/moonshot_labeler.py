"""Moonshot (Kimi) backend for the report labelling line.

A third backend beside `llm_labeler` (Anthropic) and `ollama_labeler` (local
Qwen). Kept in its own file for the same reason as the Ollama one -- a provider's
SDK calls do not belong in another provider's module -- but the rubric, the JSON
schema and the GAUGE 1 scorer are imported from `llm_labeler`, so all three
backends are judged by exactly the same ruler.

Two Kimi-specific facts, both measured rather than assumed:

  * **kimi-k2.6 is a reasoning model.** Its chain of thought goes to
    `reasoning_content` and the answer to `content`. Left on, it spent 1,200+
    output tokens thinking and returned an EMPTY `content` with
    `finish_reason: "length"` -- a failure that looks like a parse bug, not a
    budget one. `reasoning_effort: "none"` turns it off and the answer drops to
    ~192 output tokens (`thinking: {"type": "disabled"}` also works).
  * **temperature is fixed at 1.** The API rejects any other value
    ("invalid temperature: only 1 is allowed for this model"), so unlike the
    local Qwen run (temperature 0) this backend is NOT deterministic. Re-running
    it will not reproduce the same labels, and a GAUGE 1 difference smaller than
    its own run-to-run spread means nothing.
"""
from __future__ import annotations
import json
import os
import time

import pandas as pd

from config import LABELS
from llm_labeler import RUBRIC, _SCHEMA, gauge  # one rubric, one schema, one ruler

BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k2.6"
MAX_TOKENS = 2000


def _client():
    from openai import OpenAI

    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise SystemExit("set MOONSHOT_API_KEY in the environment "
                         "(never commit it to a file)")
    return OpenAI(api_key=key, base_url=BASE_URL)


def chat(report: str, client=None, model: str = DEFAULT_MODEL) -> dict:
    """One report -> the 12x(p, conf) dict."""
    client = client or _client()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": RUBRIC},
                  {"role": "user", "content": f"<report>\n{report}\n</report>"}],
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        # without this the model burns the whole budget on reasoning and returns
        # an empty content; see this module's docstring
        extra_body={"reasoning_effort": "none"},
    )
    msg = r.choices[0].message
    if not msg.content:
        raise RuntimeError(f"empty content (finish={r.choices[0].finish_reason}) "
                           f"-- reasoning likely still on")
    return json.loads(msg.content)


def label(reports: dict, out_parquet: str, model: str = DEFAULT_MODEL,
          workers: int = 4, progress_every: int = 25, checkpoint_every: int = 50):
    """Label {uid: report} concurrently and resumably into the shared layout."""
    from concurrent.futures import ThreadPoolExecutor

    done = {}
    if os.path.exists(out_parquet):
        prev = pd.read_parquet(out_parquet)
        done = {r["StudyInstanceUID"]: dict(r) for _, r in prev.iterrows()}
        print(f"resuming: {len(done)} already labelled", flush=True)
    todo = [(str(u), r) for u, r in reports.items() if str(u) not in done]
    print(f"{len(todo)} to label with {model} ({workers} workers)", flush=True)

    client = _client()
    recs, errors = list(done.values()), {}
    t0 = time.time()

    def _one(item):
        uid, rep = item
        try:
            d = chat(rep, client, model)
            rec = {"StudyInstanceUID": uid}
            for lab in LABELS:
                rec[lab] = float(d[lab]["p"])
                rec[lab + "__conf"] = float(d[lab]["conf"])
            return uid, rec, None
        except Exception as e:
            return uid, None, f"{type(e).__name__}: {str(e)[:120]}"

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
                      f"eta {(len(todo) - n) * el / n / 3600:.2f} h | "
                      f"errors {len(errors)}", flush=True)
    df = pd.DataFrame(recs)
    df.to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet}: {df.shape} | errors {len(errors)} | "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if errors:
        print("first errors:", list(errors.items())[:3], flush=True)
    return df, errors


def probe_gold(train_csv: str, out_parquet: str, model: str = DEFAULT_MODEL,
               workers: int = 4):
    """Label the 58 gold studies and score them against the same GAUGE 1."""
    tr = pd.read_csv(train_csv)
    gold = tr[tr[LABELS].notna().all(axis=1)]
    reports = {str(u): r for u, r in
               zip(gold["StudyInstanceUID"], gold["Report"].fillna(""))}
    label(reports, out_parquet, model, workers)
    return gauge(out_parquet, train_csv)
