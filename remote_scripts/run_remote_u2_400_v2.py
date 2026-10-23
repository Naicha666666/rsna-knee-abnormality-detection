"""Run 3D VLM Spatial Distillation on 400 Coronal cases using Unisound U2-Med (Fast & Robust v2).
Features:
- Uses model 'u2-med' (Unisound Medical VLM)
- Fast in-memory JPEG compression (quality 80) avoiding API gateway timeout
- Incremental checkpointing every 5 cases
- Resumes seamlessly from existing 183 completed cases
"""
import base64
import concurrent.futures
import io
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image
import requests

KEY_PATH = Path("/home/eason/rsna-knee/.unisound_key")
API_KEY = KEY_PATH.read_text(encoding="utf-8").strip()

ENDPOINT = "https://maas-api.unisound.com/v1/chat/completions"
MODEL_NAME = "u2-med"

CASES_PATH = Path("/home/eason/rsna-knee/cohort_400_u2_radimed.json")
IMG_DIR = Path("/home/eason/rsna-knee/vlm_cohort_images")
CACHE_PATH = Path("/home/eason/rsna-knee/u2_radimed_400_cache.json")


def encode_img(path: Path) -> str:
    im = Image.open(path)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_teacher_prompt(report_text: str) -> str:
    return f"""You are an expert radiologist and medical vision AI teacher designed to train a Multiple Instance Learning (MIL) model.
Below is a 3D MRI volume of a patient's knee joint presented as a 3x4 montage gallery of 12 consecutive coronal slices, labeled S1 (anterior) through S12 (posterior).
Here is the official radiology report text:

[Radiology Report]
{report_text}

Task:
Generate structured slice-level spatial supervision for the Multiple Instance Learning (MIL) attention head.
For each relevant pathology (especially Lateral OA, Medial OA, PF OA, ACL, MCL, Meniscus tears, Effusion, Synovitis, Baker's, Contusion, Fracture):
1. Rate the diagnostic relevance of each slice on a discrete 0-3 scale:
   - 0 = irrelevant (no abnormal finding or non-diagnostic slice)
   - 1 = contextual / border (mild adjacent change)
   - 2 = relevant (clear abnormal signal or structural thinning)
   - 3 = strongly diagnostic (key slice demonstrating definitive pathology, e.g. full-thickness ulcer, bone exposure, spur, or frank tear)
2. For the most diagnostic slice (relevance=3), provide a normalized bounding box [ymin, xmin, ymax, xmax] (coordinates in [0.0, 1.0]) locating the lesion within that slice (crucial for distinguishing lateral [outer] from medial [inner] compartments).

Output STRICT JSON:
{{
  "diagnoses": {{
    "<Diagnosis Name>": {{
      "primary_plane": "coronal",
      "anatomy": ["lateral compartment"],
      "depth_zone": "anterior",
      "relevant_slices": [
        {{"slice": "S6", "relevance": 3}},
        {{"slice": "S7", "relevance": 3}},
        {{"slice": "S5", "relevance": 2}}
      ],
      "roi": {{
        "slice": "S7",
        "bbox": [ymin, xmin, ymax, xmax]
      }},
      "finding": "Brief 1-sentence finding",
      "confidence": 0.0-1.0
    }}
  }}
}}
If normal with no abnormalities, output {{"diagnoses": {{}}}}."""


def call_single(item: dict, max_retries: int = 3) -> tuple[str, dict]:
    uid = item["uid"]
    report = item["report"]
    img_path = IMG_DIR / f"{uid}_montage.png"
    if not img_path.exists():
        return uid, {"error": "Image file not found"}

    b64_img = encode_img(img_path)
    prompt = build_teacher_prompt(report)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
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
                            "url": f"data:image/jpeg;base64,{b64_img}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1500
    }

    wait = 2
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=75)
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"].strip()
                    clean_text = content
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
                    except Exception:
                        try:
                            import json_repair
                            parsed = json_repair.repair_json(clean_text, return_objects=True)
                            if isinstance(parsed, dict):
                                return uid, parsed
                        except Exception:
                            pass
                        return uid, {"raw": content}
            elif resp.status_code == 429:
                time.sleep(wait * 2)
                wait *= 2
            else:
                time.sleep(wait)
        except Exception:
            time.sleep(wait)
            wait = min(wait * 2, 10)

    return uid, {"error": "Failed after max retries"}


def main():
    cases = json.load(open(CASES_PATH, encoding="utf-8"))
    print(f"Loaded {len(cases)} cases for U2-Med distillation.")

    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.load(open(CACHE_PATH, encoding="utf-8"))
            print(f"Loaded existing cache: {len(cache)} / {len(cases)} completed.")
        except Exception as e:
            print(f"Cache load error: {e}")
            cache = {}

    remaining = [c for c in cases if c["uid"] not in cache]
    print(f"Remaining to process: {len(remaining)} / {len(cases)}")

    if not remaining:
        print("All cases completed! Done.")
        return

    workers = 4
    done_cnt = 0
    t0 = time.time()
    print(f"Launching concurrent workers (concurrency={workers})...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_single, c): c["uid"] for c in remaining}
        for f in concurrent.futures.as_completed(futures):
            uid, res = f.result()
            done_cnt += 1
            cache[uid] = res
            status = "OK" if "error" not in res else "ERR"
            n_findings = len(res.get("diagnoses", {})) if isinstance(res.get("diagnoses"), dict) else 0
            elapsed = time.time() - t0
            rate = done_cnt / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - done_cnt) / rate if rate > 0 else 0

            print(f"[{len(cache)}/{len(cases)}] {uid[:12]}... -> {status} ({n_findings} findings) | Speed={rate*60:.1f} c/m | ETA={eta/60:.1f}m", flush=True)

            if done_cnt % 5 == 0 or done_cnt == len(remaining):
                with open(CACHE_PATH, "w", encoding="utf-8") as out_f:
                    json.dump(cache, out_f, indent=2, ensure_ascii=False)

    print(f"\nU2-Med run completed. Total in cache: {len(cache)}")


if __name__ == "__main__":
    main()
