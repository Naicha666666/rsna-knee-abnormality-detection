"""Ollama Nemotron backend for knee MRI radiology report labeling.

Target:
  - Local host: http://192.168.1.177:11437
  - Model: nemotron-30b:latest (running on AMD Radeon AI PRO R9700 ROCm via Ollama)
  - Mode: 1 report per request ("每条一问")
  - Structured output with schema enforcement & think=False (~2.25s per report)
"""
from __future__ import annotations
import json
import os
import sys
import time
import urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LABELS

DEFAULT_HOST = "http://192.168.1.177:11437"
DEFAULT_MODEL = "nemotron-30b:latest"
NUM_CTX = 4096
NUM_PREDICT = 350
TIMEOUT_S = 60

# ── Optimized Clinical Rubric ──────────────────────────────────────
NEMOTRON_RUBRIC = """You are an expert musculoskeletal radiologist grading knee MRI reports for 12 pathological abnormalities.
Reports are written in multiple languages (English, Spanish, German, French, Dutch, Turkish, Croatian, Greek, Bulgarian, etc.).
Grade in whatever language the report arrives in without translating the text.

For EACH of the 12 target findings, output two numbers:
  "p": probability the pathological finding is present in this knee (0.0 to 1.0)
  "conf": confidence of evidence in the report (0.0 to 1.0)

Scoring guidelines:
- Finding explicitly confirmed positive -> p high (0.75 - 1.0), conf high (0.9 - 1.0)
- Structure explicitly inspected and normal / intact / ruled out -> p low (0.0 - 0.05), conf high (0.9 - 1.0)
- Equivocal / hedged / low-grade / partial ("suspect", "possible", "grade 1") -> p moderate (0.4 - 0.7), conf moderate (0.6 - 0.8)
- Structure NEVER mentioned in report -> p at baseline (~0.30 - 0.45), conf LOW (~0.05)
  *Silence is lack of evidence, not proof of normal!*

The 12 target abnormalities and anatomical boundaries:
1. ACL: Anterior Cruciate Ligament tear / rupture / discontinuous fibers / avulsion. Normal/intact ACL -> p=0.0.
2. MCL: Medial Collateral Ligament injury / sprain / tear / abnormal thickening / edema. Normal/intact MCL -> p=0.0.
3. Medial Meniscus: Medial meniscus tear (radial, horizontal, longitudinal, complex, bucket-handle, root tear, or maceration). Grade 1-2 intrasubstance degeneration without surfacing tear -> p mid-low (0.15 - 0.35). Normal/intact -> p=0.0.
4. Lateral Meniscus: Lateral meniscus tear (radial, horizontal, longitudinal, complex, flap, root tear). Normal/intact -> p=0.0.
5. Medial OA: Medial femorotibial osteoarthritis / chondrosis / cartilage thinning or ulcer on medial femoral condyle or medial tibial plateau.
6. Lateral OA: Lateral femorotibial osteoarthritis / chondrosis / cartilage thinning or ulcer on lateral femoral condyle or lateral tibial plateau.
7. PF OA: Patellofemoral osteoarthritis / chondromalacia patellae / cartilage defect on patella (facets) OR femoral trochlea (trochlear groove/sulcus). NOTE: Any trochlear defect (e.g. "medial trochlea", "tróclea femoral") belongs to PF OA, NOT Medial OA!
   *Special rule*: "Tricompartmental osteoarthritis/chondrosis" raises Medial OA, Lateral OA, and PF OA simultaneously (p >= 0.75).
8. Effusion: Joint effusion / hydrarthrosis / hemarthrosis / fluid in suprapatellar recess. (Trace/small -> p~0.5-0.7; moderate/large/massive -> p~0.85-1.0; no effusion -> p=0.0).
9. Synovitis: Synovitis, synovial membrane thickening, hypertrophy, or proliferation. If significant effusion or severe OA is present, secondary synovitis likelihood is elevated (p~0.6-0.8); if never mentioned, conf=0.05.
10. Baker's: Baker's cyst / popliteal cyst / gastrocnemius-semimembranosus bursal fluid collection. Normal/ruled out -> p=0.0.
11. Contusion: Traumatic bone contusion / bone bruise / trabecular microfracture. NOTE: Subchondral marrow edema directly underlying a degenerative cartilage defect is OA, NOT traumatic contusion!
12. Fracture: Acute fracture, avulsion fracture (Segond, tibial spine/eminence, fibular head), osteochondral fracture/fragment, or insufficiency fracture (e.g., subchondral insufficiency fracture / SONK) of patella, femur, tibia, or fibula.

Multilingual negation cues:
- Spanish: "sin signos de rotura", "conservado", "dentro de límites normales", "no se observan", "intacto" -> p=0.0, conf=0.95.
- German: "intakt", "regelrecht", "unauffällig", "keine Ruptur", "kein Nachweis" -> p=0.0, conf=0.95.
- French: "intègre", "sans anomalie", "pas de rupture" -> p=0.0, conf=0.95.
- Turkish: "salim", "doğal", "yırtık izlenmedi", "intakt" -> p=0.0, conf=0.95.
- Croatian/Serbian: "uredan", "očuvan", "bez znakova rupture" -> p=0.0, conf=0.95.
- Bulgarian: "съхранен", "без данни за руптура", "в норма" -> p=0.0, conf=0.95.

Output ONLY a minified JSON object with exactly these 12 keys:
{"ACL":{"p":0.0,"conf":0.0},"MCL":{"p":0.0,"conf":0.0},"Medial Meniscus":{"p":0.0,"conf":0.0},"Lateral Meniscus":{"p":0.0,"conf":0.0},"Medial OA":{"p":0.0,"conf":0.0},"Lateral OA":{"p":0.0,"conf":0.0},"PF OA":{"p":0.0,"conf":0.0},"Effusion":{"p":0.0,"conf":0.0},"Synovitis":{"p":0.0,"conf":0.0},"Baker's":{"p":0.0,"conf":0.0},"Contusion":{"p":0.0,"conf":0.0},"Fracture":{"p":0.0,"conf":0.0}}
"""

SCHEMA = {
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


def chat_one(report: str, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL,
             timeout: int = TIMEOUT_S, max_retries: int = 3) -> dict:
    """Label one report -> the 12x(p, conf) dict via Ollama's /api/chat."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": NEMOTRON_RUBRIC},
            {"role": "user", "content": f"<report>\n{report}\n</report>"},
        ],
        "format": SCHEMA,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{host}/api/chat"

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["message"]["content"].strip()
                # If wrapped in codeblock or extra text, isolate JSON
                if "```" in content:
                    parts = [p for p in content.split("```") if "{" in p]
                    if parts:
                        content = parts[0]
                        if content.lstrip().startswith("json"):
                            content = content.lstrip()[4:]
                i, j = content.find("{"), content.rfind("}")
                if i != -1 and j > i:
                    content = content[i:j + 1]
                parsed = json.loads(content)
                # Verify all keys exist
                valid = True
                for lab in LABELS:
                    if lab not in parsed or "p" not in parsed[lab] or "conf" not in parsed[lab]:
                        valid = False
                        break
                if valid:
                    return parsed
                raise ValueError("Parsed JSON missing required label keys")
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(1.0 * attempt)


def label_corpus(
    reports: dict[str, str],
    out_parquet: str,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    checkpoint_every: int = 25,
    progress_every: int = 10,
):
    """Sequentially labels {uid: report} into parquet. Resumes if file exists."""
    os.makedirs(os.path.dirname(os.path.abspath(out_parquet)), exist_ok=True)
    done: dict[str, dict] = {}
    if os.path.exists(out_parquet):
        try:
            prev = pd.read_parquet(out_parquet)
            done = {str(r["StudyInstanceUID"]): dict(r) for _, r in prev.iterrows()}
            print(f"Resuming: {len(done)} already labeled in {out_parquet}", flush=True)
        except Exception as e:
            print(f"Warning: could not load existing {out_parquet}: {e}", flush=True)

    todo = [(str(u), str(r)) for u, r in reports.items() if str(u) not in done]
    total_count = len(reports)
    print(f"Total reports: {total_count} | To do: {len(todo)} | Already done: {len(done)}", flush=True)

    if not todo:
        print("All reports are already labeled!", flush=True)
        return pd.read_parquet(out_parquet)

    recs = list(done.values())
    errors = {}
    t0 = time.time()
    last_flush = time.time()

    def _flush():
        nonlocal last_flush
        if recs:
            df = pd.DataFrame(recs)
            # Guarantee StudyInstanceUID is string
            df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
            df.to_parquet(out_parquet, index=False)
            last_flush = time.time()

    for idx, (uid, report_text) in enumerate(todo, 1):
        t_item_start = time.time()
        try:
            d = chat_one(report_text, host=host, model=model)
            rec = {"StudyInstanceUID": uid}
            for lab in LABELS:
                rec[lab] = float(d[lab]["p"])
                rec[lab + "__conf"] = float(d[lab]["conf"])
            recs.append(rec)
            done[uid] = rec
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:120]}"
            errors[uid] = err_msg
            print(f"  [Error {idx}/{len(todo)}] UID {uid}: {err_msg}", flush=True)

        # Progress reporting
        if idx % progress_every == 0 or idx == len(todo):
            elapsed = time.time() - t0
            speed = idx / elapsed
            remaining = len(todo) - idx
            eta_s = remaining / speed if speed > 0 else 0
            eta_str = f"{int(eta_s // 60)}m {int(eta_s % 60)}s" if eta_s < 3600 else f"{eta_s / 3600:.1f}h"
            print(
                f"[{len(done)}/{total_count}] ({(len(done)/total_count)*100:.1f}%) "
                f"Speed: {1/speed:.2f}s/it ({speed:.2f} it/s) | ETA: {eta_str} | Errors: {len(errors)}",
                flush=True
            )

        # Checkpointing
        if idx % checkpoint_every == 0 or idx == len(todo):
            _flush()

    _flush()
    print(f"Finished labeling! Output saved to {out_parquet} ({len(recs)} records).", flush=True)
    if errors:
        err_path = os.path.join(os.path.dirname(out_parquet), "nemotron_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        print(f"Encountered {len(errors)} errors (logged to {err_path}).", flush=True)

    return pd.DataFrame(recs)
