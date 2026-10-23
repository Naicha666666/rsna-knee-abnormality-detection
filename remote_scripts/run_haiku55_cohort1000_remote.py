"""Execute Claude Haiku 5.5 Multimodal Audit & Refinement on 1,000 Ambiguous Studies.

Runs on Linux remote machine (12 concurrent workers):
- Images from /home/eason/rsna-knee/vlm_cohort_images/{uid}_montage.png
- Manifest from /home/eason/rsna-knee/cohort_1000_ambiguous_manifest.json
- Cache in /home/eason/rsna-knee/haiku55_cohort1000_cache/{uid}.json
- Strict JSON parsing with fallback
- Output /home/eason/rsna-knee/labels_cohort1000_haiku55_refined.parquet
"""
import base64
import concurrent.futures
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("/home/eason/rsna-knee")
IMG_DIR = ROOT / "vlm_cohort_images"
MANIFEST_PATH = ROOT / "cohort_1000_ambiguous_manifest.json"
CACHE_DIR = ROOT / "haiku55_cohort1000_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = ROOT / "labels_cohort1000_haiku55_refined.parquet"

OPENROUTER_KEY = "YOUR_OPENROUTER_KEY"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-5.5"

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

def build_prompt(report_text: str, prior_dict: dict) -> str:
    priors_formatted = json.dumps(prior_dict, indent=2)
    return f"""You are an expert MSK radiologist performing a Multimodal Audit & Refinement on a knee MRI study.
You are provided with:
1. A 3x4 gallery montage of 12 consecutive coronal slices, labeled S1 (anterior) through S12 (posterior).
2. The official radiology report text.
3. The baseline 12-label soft probability indicators (Domain-Logit Prior) pre-computed from multi-LLM consensus.

[Baseline 12-Label Soft Indicators (Prior p)]:
{priors_formatted}

[Official Radiology Report]:
{report_text.strip()}

Task & Rules:
1. Examine the visual evidence across the 12 coronal slices (S1-S12) and cross-correlate with the report and baseline soft indicators.
2. For EACH of the 12 standard knee pathologies:
   - Provide:
     * "prior_p": the baseline probability given above.
     * "refined_p": calibrated diagnostic probability in [0.00, 1.00] based on visual evidence + report.
     * "status": "CONFIRMED" (prior aligns with image), "REFINED" (image refines ambiguity or reveals missed sign), or "NORMAL" (normal/negative).
     * "key_slices": array of prominent slice labels (e.g. ["S6", "S7"]) showing the abnormality, or [] if normal.
     * "visual_evidence": brief 1-phrase description of observed radiological signs (or "intact / normal").
     * "rationale": 1 concise sentence (under 15 words).
3. Anatomical Guidance:
   - Coronal plane is optimal for: MCL, Medial/Lateral OA (joint space narrowing & osteophytes), Meniscus body extrusion/tears, and tibial/femoral bone marrow contusions.
   - For ACL, PF OA, and Synovitis: note that assessment is partial on coronal slices; do not make drastic probability shifts unless definitive coronal signs (e.g. large suprapatellar effusion, avulsion fracture, or joint deformity) are present.
   - If image reveals incidental findings (e.g. joint effusion or bone contusion) omitted in the report, confirm visually and adjust refined_p.

Output STRICT JSON format only (no markdown, no backticks):
{{
  "clinical_impression": "1-2 sentence overall radiological impression",
  "diagnoses": {{
    "ACL": {{"prior_p": 0.05, "refined_p": 0.05, "status": "NORMAL", "key_slices": [], "visual_evidence": "normal fibers", "rationale": "Report negative, no secondary coronal signs."}},
    ...
  }}
}}"""

def encode_image(img_path: Path) -> str:
    im = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def call_haiku(prompt: str, img_data_url: str) -> dict:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "max_tokens": 2500,
        "temperature": 0.0,
        "reasoning": {"enabled": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_data_url}}
                ]
            }
        ]
    }
    for attempt in range(4):
        try:
            t0 = time.time()
            resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=90)
            lat = time.time() - t0
            resp.raise_for_status()
            data = resp.json()
            data["latency_sec"] = lat
            return data
        except Exception as e:
            if attempt == 3:
                raise e
            wait = 2.0 * (attempt + 1)
            time.sleep(wait)

def process_case(item: dict) -> dict:
    uid = item["uid"]
    cache_file = CACHE_DIR / f"{uid}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if "parsed" in cached and "diagnoses" in cached["parsed"]:
                return cached
        except Exception:
            pass

    rep_text = item.get("report", "")
    prior_p = item.get("prior_p", {})
    img_path = IMG_DIR / f"{uid}_montage.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing montage image for {uid}")

    img_url = encode_image(img_path)
    prompt = build_prompt(rep_text, prior_p)

    res = call_haiku(prompt, img_url)
    usage = res.get("usage", {})
    cost = usage.get("cost", 0.0)
    raw_content = res["choices"][0]["message"]["content"].strip()

    clean_text = raw_content
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        parsed = json.loads(clean_text)
    except Exception:
        parsed = {"diagnoses": {}}
        for lab in LABELS:
            m = re.search(rf'"{re.escape(lab)}"\s*:\s*\{{[^}}]*"refined_p"\s*:\s*([0-9.]+)', clean_text)
            if m:
                parsed["diagnoses"][lab] = {"refined_p": float(m.group(1)), "status": "REFINED"}

    record = {
        "uid": uid,
        "prior_p": prior_p,
        "latency_sec": res["latency_sec"],
        "usage": usage,
        "cost": cost,
        "parsed": parsed
    }
    cache_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record

def main():
    print("=" * 85)
    print("CLAUDE HAIKU 5.5 MULTIMODAL AUDIT: COHORT 1,000 AMBIGUOUS STUDIES")
    print(f"Model: {MODEL} | Endpoint: {ENDPOINT}")
    print(f"Cache Dir: {CACHE_DIR} | Image Dir: {IMG_DIR}")
    print("=" * 85)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    n_total = len(manifest)
    print(f"Loaded {n_total} target studies from manifest.")

    # Check cache status
    cached_count = sum(1 for item in manifest if (CACHE_DIR / f"{item['uid']}.json").exists())
    print(f"Existing cache hits: {cached_count}/{n_total} studies.")

    results = [None] * n_total
    t_start = time.time()
    completed_cnt = 0
    total_cost = 0.0

    # 12 workers for optimal rate-limit & latency balance
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_to_idx = {
            executor.submit(process_case, item): idx
            for idx, item in enumerate(manifest)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                rec = future.result()
                results[idx] = rec
                completed_cnt += 1
                c = rec.get("cost", 0.0)
                total_cost += c
                if completed_cnt % 25 == 0 or completed_cnt == n_total:
                    elapsed = time.time() - t_start
                    rate = completed_cnt / max(elapsed, 1e-3)
                    rem_cases = n_total - completed_cnt
                    eta_min = (rem_cases / max(rate, 1e-3)) / 60
                    print(f"[{completed_cnt:4d}/{n_total}] Done | Elapsed: {elapsed/60:4.1f}m | ETA: {eta_min:4.1f}m | Cost: ${total_cost:.4f} (~{total_cost*7.2:.2f}元)", flush=True)
            except Exception as e:
                print(f"[{idx+1:4d}/{n_total}] ERROR on {manifest[idx]['uid'][:16]}: {e}", flush=True)

    elapsed_total = time.time() - t_start
    print("\n" + "=" * 85)
    print(f"FINISHED ALL {completed_cnt}/{n_total} STUDIES IN {elapsed_total/60:.2f} MINUTES!")
    print(f"Total Cost: ${total_cost:.4f} USD (~{total_cost * 7.2:.2f} RMB)")
    print("=" * 85)

    # Save to Parquet
    rows = []
    for r in results:
        if r is None:
            continue
        uid = r["uid"]
        diags = r.get("parsed", {}).get("diagnoses", {})
        row = {"StudyInstanceUID": uid}
        for lab in LABELS:
            d = diags.get(lab, {})
            p_ref = d.get("refined_p", r["prior_p"].get(lab, 0.5))
            row[lab] = float(p_ref)
        rows.append(row)

    df_out = pd.DataFrame(rows).set_index("StudyInstanceUID")
    df_out.to_parquet(OUT_PARQUET)
    print(f"Successfully saved {len(df_out)} refined studies to {OUT_PARQUET}")

if __name__ == "__main__":
    main()
