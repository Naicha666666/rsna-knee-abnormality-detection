"""LLM soft labels for the 4,349 unannotated reports (the L line).

Why: every LB-moving change this project has made was on the supervision side
(epochs +0.012, a better report extractor +0.006), and C1 showed that swapping
the ENTIRE imaging preprocessing pipeline for the reference's moved nothing
(0.897 -> 0.892). The reference arm we are chasing trains on
`labels_llm_soft.parquet` -- LLM-generated soft labels, not distributed with
their notebook. Our regex extractor's agreement with the 58 gold studies
(GAUGE 1) is 0.865 macro-AUC; that number is the ceiling this line attacks.

Design notes:
  * Batch API, so 4,349 reports cost half. They are not latency-sensitive.
  * Structured outputs (`output_config.format`), so a malformed label dict is
    impossible rather than something to defend against downstream.
  * The rubric is the cacheable prefix and the report is the volatile suffix,
    so the per-request prompt is nearly free after the first.
  * Each finding gets a probability AND a confidence, mirroring what
    `report_extractor.extract()` already returns, so `dataset.py` consumes this
    through the same (grades, conf) contract with no changes.
  * Reports are multilingual (9+ languages observed) -- the prompt does not ask
    for translation, only for grading, because translation would double output
    tokens for no gain.

All 4,407 reports are labelled, the 58 gold ones included. That is not
contamination: the model reads only the report TEXT and never sees a gold label,
so scoring its output against gold is exactly the GAUGE 1 protocol the regex
extractor is measured with (0.865) -- and without labelling those 58 there is no
way to compare the two at all. Training still overrides those 58 studies with
their true gold labels, as it already did.
"""
from __future__ import annotations
import json
import os
import time

import numpy as np
import pandas as pd

from config import LABELS

MODEL = "claude-opus-5"
# 12 findings x (probability, confidence) is a small, fixed output; this cap is
# generous for it and keeps a malformed long generation from running away.
MAX_TOKENS = 2000

RUBRIC = """You grade knee MRI radiology reports for twelve findings.

The reports are written in many languages (English, Spanish, German, French,
Dutch, Turkish, Croatian, Greek, Bulgarian and others). Grade them in whatever
language they arrive in. Do not translate.

For EACH of the twelve findings, return two numbers:

  "p"    probability the finding is present in THIS knee, 0.0 to 1.0
  "conf" how much the report actually tells you about that finding, 0.0 to 1.0

The two are independent, and the difference matters:
  * report explicitly states the finding      -> p high,  conf high
  * report explicitly rules the finding out   -> p low,   conf high
  * report never mentions the structure       -> p at the population base rate,
                                                 conf LOW (near 0.05)
  * hedged / equivocal ("possible", "suspect") -> p mid,   conf moderate

A finding the radiologist looked at and called normal is NOT the same as a
finding never mentioned. The first is evidence; the second is silence. Silence
must come back with low conf so downstream training can down-weight it.

Grade severity into p: a "small" or "trace" effusion is not a large one; a
"complete tear" outranks "mild signal change". Read negation carefully,
including negation that scopes only part of a sentence -- "subchondral
insufficiency fracture without articular collapse" ASSERTS a fracture.

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
  Fracture         fracture of any bone in the study

An unlocalised osteoarthritis statement ("tricompartmental osteoarthritis")
raises all three OA compartments. A degenerative subchondral marrow signal under
a cartilage defect is OA, not a traumatic Contusion.

Output MINIFIED JSON on a single line: no spaces, no newlines, no indentation."""

_SCHEMA = {
    "type": "object",
    "properties": {
        lab: {
            "type": "object",
            "properties": {
                "p": {"type": "number", "minimum": 0, "maximum": 1},
                "conf": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["p", "conf"],
            "additionalProperties": False,
        }
        for lab in LABELS
    },
    "required": list(LABELS),
    "additionalProperties": False,
}


def _client():
    import anthropic
    # Zero-arg: resolves ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN, then an
    # `ant auth login` profile. Whichever the user set up works unchanged.
    return anthropic.Anthropic()


def _params(report: str):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    return MessageCreateParamsNonStreaming(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": RUBRIC,
                 # the rubric is identical across all 4,349 requests; the report
                 # below it is the only volatile part
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"<report>\n{report}\n</report>"}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )


def submit(reports: dict, out_json: str, chunk: int = 20000):
    """Submit reports (uid -> text) as Batch API jobs; record batch ids."""
    from anthropic.types.messages.batch_create_params import Request

    client = _client()
    uids = list(reports)
    batches = []
    for i in range(0, len(uids), chunk):
        part = uids[i:i + chunk]
        b = client.messages.batches.create(requests=[
            Request(custom_id=u, params=_params(reports[u])) for u in part])
        batches.append({"id": b.id, "n": len(part)})
        print(f"submitted batch {b.id} with {len(part)} requests", flush=True)
    json.dump({"model": MODEL, "batches": batches}, open(out_json, "w"), indent=1)
    return batches


def collect(state_json: str, out_parquet: str, poll_s: int = 60):
    """Poll the submitted batches, then write uid -> 24 columns as parquet."""
    client = _client()
    state = json.load(open(state_json))
    rows, errors = {}, {}
    for b in state["batches"]:
        while True:
            info = client.messages.batches.retrieve(b["id"])
            if info.processing_status == "ended":
                break
            print(f"{b['id']}: {info.processing_status}, "
                  f"processing {info.request_counts.processing}", flush=True)
            time.sleep(poll_s)
        print(f"{b['id']}: succeeded {info.request_counts.succeeded} "
              f"errored {info.request_counts.errored}", flush=True)
        # Results come back in ANY order -- keyed by custom_id, never position.
        for res in client.messages.batches.results(b["id"]):
            if res.result.type != "succeeded":
                errors[res.custom_id] = res.result.type
                continue
            msg = res.result.message
            text = next((c.text for c in msg.content if c.type == "text"), None)
            if text is None:
                errors[res.custom_id] = "no-text-block"
                continue
            try:
                rows[res.custom_id] = json.loads(text)
            except json.JSONDecodeError:
                errors[res.custom_id] = "bad-json"
    if errors:
        print(f"WARNING {len(errors)} requests without a usable label; "
              f"first few: {list(errors.items())[:5]}", flush=True)

    recs = []
    for uid, d in rows.items():
        rec = {"StudyInstanceUID": uid}
        for lab in LABELS:
            rec[lab] = float(d[lab]["p"])
            rec[lab + "__conf"] = float(d[lab]["conf"])
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet}: {df.shape}", flush=True)
    return df, errors


# --------------------------------------------------------------------------- #
# offline evaluation: the same GAUGE 1 the regex extractor is measured with
# --------------------------------------------------------------------------- #
def gauge(parquet: str, train_csv: str):
    """Agreement macro-AUC on the 58 gold studies, comparable to run_gauges().

    This is the decision number for the whole line: if it does not clear the
    regex extractor's 0.865, the LLM labels are not worth a training run.
    """
    import report_extractor as R

    tr = pd.read_csv(train_csv)
    gold = tr[tr[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")
    df = pd.read_parquet(parquet).set_index("StudyInstanceUID")
    common = [u for u in gold.index if u in df.index]
    if not common:
        raise SystemExit("parquet holds none of the 58 gold studies -- label "
                         "all 4407 reports, not just the unannotated ones, or "
                         "there is nothing to compare against the extractor")
    y = gold.loc[common, LABELS].to_numpy(float)
    aucs = []
    print(f"GAUGE 1 (LLM) on {len(common)} gold studies:")
    for j, lab in enumerate(LABELS):
        col = df.loc[common, lab].to_numpy(float)
        if len(np.unique(y[:, j])) < 2:
            print(f"  {lab:18s} only one class")
            continue
        a = R.roc_auc(y[:, j], col)
        aucs.append(a)
        print(f"  {lab:18s} AUC={a:.3f}")
    macro = float(np.mean(aucs))
    print(f"  macro-AUC={macro:.3f}   (regex extractor: 0.865)")
    return macro
