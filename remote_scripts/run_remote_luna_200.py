"""Run 3D VLM Training / Distillation on 200 cases using OpenRouter openai/gpt-6-luna.
Reads coronal 12-slice montages from /home/eason/rsna-knee/vlm_cohort_images/
Saves results periodically to /home/eason/rsna-knee/gpt6_luna_200_cache.json
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

CASES_PATH = Path("/home/eason/rsna-knee/cohort_200_gpt6_luna.json")
IMG_DIR = Path("/home/eason/rsna-knee/vlm_cohort_images")
CACHE_PATH = Path("/home/eason/rsna-knee/gpt6_luna_200_cache.json")


def encode_img(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

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
      "primary_plane": "coronal" | "sagittal" | "axial",
      "anatomy": ["lateral compartment", ...],
      "depth_zone": "anterior" | "central" | "posterior",
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

def call_single(item: dict, max_retries: int = 4) -> tuple[str, dict]:
    uid = item["uid"]
    report = item["report"]
    img_path = IMG_DIR / f"{uid}_montage.png"
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
    
    wait = 2
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    try:
                        import json_repair
                        parsed = json_repair.repair_json(content, return_objects=True)
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
            wait = min(wait * 2, 20)
            
    return uid, {"error": "Failed after max retries"}

def main():
    cases = json.load(open(CASES_PATH, encoding="utf-8"))
    print(f"Loaded {len(cases)} cases for GPT-6 Luna.")

    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.load(open(CACHE_PATH, encoding="utf-8"))
            print(f"Loaded cache: {len(cache)} / {len(cases)} already completed.")
        except Exception as e:
            print(f"Cache load error: {e}")
            cache = {}

    valid_cache = {
        k: v for k, v in cache.items()
        if isinstance(v, dict) and "diagnoses" in v and len(v["diagnoses"]) > 0
    }
    cache = valid_cache
    remaining = [c for c in cases if c["uid"] not in cache]
    print(f"Valid in cache: {len(cache)}. Remaining to process: {len(remaining)} / {len(cases)}")

    if remaining:
        print("Launching concurrent workers (concurrency=2)...")
        done_cnt = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(call_single, c): c["uid"] for c in remaining}
            for f in concurrent.futures.as_completed(futures):
                uid, res = f.result()
                done_cnt += 1
                cache[uid] = res
                status = "OK" if "error" not in res else "ERR"
                print(f"[{done_cnt:03d}/{len(remaining)}] {uid[:18]}... {status}", flush=True)
                
                if done_cnt % 10 == 0 or done_cnt == len(remaining):
                    with open(CACHE_PATH, "w", encoding="utf-8") as out_f:
                        json.dump(cache, out_f, indent=2, ensure_ascii=False)
                    print(f"  --> Checkpoint saved ({len(cache)} total in cache)", flush=True)

    print(f"\nGPT-6 Luna run completed. Total in cache: {len(cache)}")

if __name__ == "__main__":
    main()

