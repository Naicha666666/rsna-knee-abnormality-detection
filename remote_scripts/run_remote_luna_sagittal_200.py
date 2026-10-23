"""Run 3D VLM Spatial Distillation on 200 Brand-New Sagittal Cases using OpenRouter openai/gpt-6-luna.
Reads 12-slice sagittal montages from /home/eason/rsna-knee/vlm_cohort_images/
Saves progress to /home/eason/rsna-knee/gpt6_luna_sagittal_200_cache.json
"""
import base64
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
import requests

OPENROUTER_KEY = "YOUR_OPENROUTER_KEY"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-6-luna"

CASES_PATH = Path("/home/eason/rsna-knee/cohort_200_sagittal_luna.json")
IMG_DIR = Path("/home/eason/rsna-knee/vlm_cohort_images")
CACHE_PATH = Path("/home/eason/rsna-knee/gpt6_luna_sagittal_200_cache.json")


def encode_img(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_teacher_prompt(report_text: str) -> str:
    return f"""You are an expert radiologist and medical vision AI teacher designed to train a Multiple Instance Learning (MIL) model.
Below is a 3D MRI volume of a patient's knee joint presented as a 3x4 montage gallery of 12 consecutive sagittal slices, labeled S1 through S12 (spanning the joint across the medial-to-lateral or lateral-to-medial dimension, with the central intercondylar notch containing the cruciate ligaments typically in slices S5-S8).
Inside each sagittal slice:
- Superior (femur) is at the top, Inferior (tibia) is at the bottom.
- Anterior (patella / extensor mechanism) is to one side, Posterior (popliteal fossa) is to the other side.
Here is the official radiology report text:

[Radiology Report]
{report_text}

Task:
Generate structured slice-level spatial supervision for the Multiple Instance Learning (MIL) attention head.
For each relevant pathology (especially ACL, MCL, Medial Meniscus tears, Lateral Meniscus tears, Contusion / Bone marrow edema, Baker's cyst, PF OA, Medial OA, Lateral OA, Effusion, Synovitis, Fracture):
1. Rate the diagnostic relevance of each slice on a discrete 0-3 scale:
   - 0 = irrelevant (no abnormal finding or non-diagnostic slice)
   - 1 = contextual / border (mild adjacent change)
   - 2 = relevant (clear abnormal signal, edema, or thinning)
   - 3 = strongly diagnostic (key slice demonstrating definitive pathology, e.g. acute ACL disruption, definitive meniscal tear, focal contusion, or cartilage defect)
2. For the most diagnostic slice (relevance=3), provide a normalized bounding box [ymin, xmin, ymax, xmax] (coordinates in [0.0, 1.0]) locating the lesion within that slice.

Output STRICT JSON:
{{
  "diagnoses": {{
    "<Diagnosis Name>": {{
      "primary_plane": "sagittal",
      "anatomy": ["anterior cruciate ligament"],
      "depth_zone": "medial" | "central" | "lateral",
      "relevant_slices": [
        {{"slice": "S6", "relevance": 3}},
        {{"slice": "S7", "relevance": 3}},
        {{"slice": "S5", "relevance": 2}}
      ],
      "roi": {{
        "slice": "S6",
        "bbox": [ymin, xmin, ymax, xmax]
      }},
      "finding": "Brief 1-sentence finding",
      "confidence": 0.0-1.0
    }}
  }}
}}
If normal with no abnormalities, output {{"diagnoses": {{}}}}."""


def call_single(item: dict, max_retries: int = 4) -> tuple[str, dict]:
    uid = item["uid"]
    report = item["report"]
    img_path = IMG_DIR / f"{uid}_sagittal_montage.png"
    if not img_path.exists():
        return uid, {"error": "Image file not found"}

    b64_img = encode_img(img_path)
    prompt = build_teacher_prompt(report)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_img}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.05,
        "max_tokens": 4096
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=90)
            if resp.status_code == 200:
                raw_text = resp.json()["choices"][0]["message"]["content"].strip()
                # Clean Markdown JSON fences if present
                clean_text = raw_text
                if clean_text.startswith("```"):
                    lines = clean_text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    clean_text = "\n".join(lines).strip()
                try:
                    parsed = json.loads(clean_text)
                    return uid, parsed
                except json.JSONDecodeError:
                    return uid, {"raw": raw_text, "parse_error": True}
            elif resp.status_code == 429:
                # Rate limit backoff
                wait = (attempt + 1) * 8
                print(f"[RateLimit 429] uid={uid[:12]}..., sleeping {wait}s...")
                time.sleep(wait)
            else:
                print(f"[HTTP {resp.status_code}] uid={uid[:12]}...: {resp.text[:120]}")
                time.sleep(3)
        except Exception as e:
            print(f"[Exception] attempt {attempt+1} uid={uid[:12]}...: {e}")
            time.sleep(4)

    return uid, {"error": f"Failed after {max_retries} retries"}


def main():
    cases = json.load(open(CASES_PATH, encoding="utf-8"))
    print(f"Loaded {len(cases)} Sagittal cohort studies for GPT-6 Luna distillation.")

    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.load(open(CACHE_PATH, encoding="utf-8"))
            print(f"Loaded existing cache with {len(cache)} entries.")
        except Exception as e:
            print(f"Could not load cache: {e}")

    remaining = [c for c in cases if c["uid"] not in cache]
    print(f"Remaining to process: {len(remaining)} / {len(cases)}")

    if not remaining:
        print("All cases already processed! Done.")
        return

    workers = 4
    batch_save_freq = 5
    processed_count = 0
    t0 = time.time()

    print(f"Starting ThreadPoolExecutor with {workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_uid = {executor.submit(call_single, item): item["uid"] for item in remaining}
        for future in concurrent.futures.as_completed(future_to_uid):
            uid = future_to_uid[future]
            try:
                ret_uid, result = future.result()
                cache[ret_uid] = result
                processed_count += 1

                n_diags = len(result.get("diagnoses", {})) if isinstance(result.get("diagnoses"), dict) else 0
                has_err = "error" in result or result.get("parse_error")
                status_str = "ERR" if has_err else f"{n_diags} findings"
                elapsed = time.time() - t0
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - processed_count) / rate if rate > 0 else 0

                print(f"[{len(cache)}/{len(cases)}] UID={uid[:12]}... -> {status_str} | Speed={rate*60:.1f} cases/min | ETA={eta/60:.1f}m")

                if processed_count % batch_save_freq == 0:
                    with open(CACHE_PATH, "w", encoding="utf-8") as f:
                        json.dump(cache, f, indent=2, ensure_ascii=False)

            except Exception as exc:
                print(f"UID {uid} generated an exception: {exc}")

    # Final save
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"Completed! Total cached: {len(cache)}. Time elapsed: {(time.time() - t0)/60:.1f} minutes.")


if __name__ == "__main__":
    main()
