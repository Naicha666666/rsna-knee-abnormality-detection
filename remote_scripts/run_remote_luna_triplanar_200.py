"""Production Tri-Planar 3D VLM Spatial Distillation on 200 Brand-New Cases using OpenRouter openai/gpt-6-luna.
Simultaneously feeds Coronal, Sagittal, and Axial montages along with official clinical radiology reports.
Provides joint 3D spatial grounding across X, Y, Z joint axes.
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

OPENROUTER_KEY = "YOUR_OPENROUTER_KEY"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-6-luna"

CASES_PATH = Path("/home/eason/rsna-knee/cohort_200_triplanar_luna.json")
IMG_DIR = Path("/home/eason/rsna-knee/vlm_cohort_images")
CACHE_PATH = Path("/home/eason/rsna-knee/gpt6_luna_triplanar_200_cache.json")


def load_and_compress_img(path: Path) -> str:
    with Image.open(path) as im:
        im = im.convert("L")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_teacher_prompt(report_text: str) -> str:
    return f"""You are an elite musculoskeletal (MSK) radiologist and medical vision AI teacher designing cross-planar supervision for a multi-view 3D Deep Learning model.
Below are three distinct imaging planes of a patient's knee joint from the same MRI study:
- Image 1: CORONAL montage (12 slices, S1 anterior to S12 posterior)
- Image 2: SAGITTAL montage (12 slices, S1 lateral to S12 medial, or vice versa)
- Image 3: AXIAL montage (12 slices, S1 superior / suprapatellar to S12 inferior / proximal tibia)

Here is the official clinical radiology report for this patient:

[Radiology Report]
{report_text}

Clinical Task:
Correlate the 3 imaging planes with the radiology report to perform comprehensive Tri-Planar 3D Spatial Localization.
For every pathology confirmed in the knee (ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's Cyst, Contusion, Fracture):
1. Specify 'primary_diagnostic_plane': ('sagittal' | 'coronal' | 'axial') - the optimal viewing plane for diagnosing this condition.
2. For planes where findings are visible, list ONLY relevant slices (relevance 2 or 3, skip normal slices):
   - 2 = clear abnormal finding / moderate lesion
   - 3 = pathognomonic / peak diagnostic slice
3. For each plane where relevance=3, provide a normalized bounding box [ymin, xmin, ymax, xmax] locating the lesion within that peak slice (coordinates within [0.0, 1.0]).
4. Provide 'cross_planar_synthesis': concise 1-sentence radiological explanation of how the planes corroborate each other (e.g. "Sagittal S6 confirms ACL midsubstance tear, Coronal S6 confirms empty notch sign").

Output STRICT JSON:
{{
  "diagnoses": {{
    "<Diagnosis Name>": {{
      "primary_diagnostic_plane": "sagittal" | "coronal" | "axial",
      "anatomy": ["..."],
      "coronal_slices": [
        {{"slice": "S6", "relevance": 3, "bbox": [0.40, 0.45, 0.60, 0.55]}}
      ],
      "sagittal_slices": [
        {{"slice": "S6", "relevance": 3, "bbox": [0.45, 0.40, 0.65, 0.60]}}
      ],
      "axial_slices": [
        {{"slice": "S8", "relevance": 2}}
      ],
      "cross_planar_synthesis": "..."
    }}
  }}
}}
"""


def process_case(case_info: dict, max_retries: int = 4) -> tuple[str, dict]:
    uid = case_info["uid"]
    f_cor = IMG_DIR / f"{uid}_montage.png"
    f_sag = IMG_DIR / f"{uid}_sagittal_montage.png"
    f_ax = IMG_DIR / f"{uid}_axial_montage.png"

    if not (f_cor.exists() and f_sag.exists() and f_ax.exists()):
        return uid, {"error": "Missing one or more montages"}

    try:
        b64_cor = load_and_compress_img(f_cor)
        b64_sag = load_and_compress_img(f_sag)
        b64_ax = load_and_compress_img(f_ax)
    except Exception as e:
        return uid, {"error": f"Image load failed: {e}"}

    prompt_text = build_teacher_prompt(case_info["report"])

    messages = [
        {
            "role": "system",
            "content": "You are an expert MSK radiologist and 3D VLM annotation specialist. Output strictly valid JSON without Markdown fences or commentary."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_cor}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_sag}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_ax}", "detail": "high"}}
            ]
        }
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kaggle.com/rsna-knee",
        "X-Title": "RSNA Knee Tri-Planar 3D Distillation"
    }

    delay = 2.0
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
            elapsed = time.time() - t0
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    if content.startswith("```"):
                        lines = content.splitlines()
                        clean = "\n".join([l for l in lines if not l.startswith("```")])
                        parsed = json.loads(clean)
                    else:
                        raise

                return uid, {
                    "status": "ok",
                    "elapsed_s": round(elapsed, 2),
                    "annotation": parsed,
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0)
                }
            elif resp.status_code in [429, 500, 502, 503, 504]:
                time.sleep(delay)
                delay *= 2.0
            else:
                return uid, {"error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
        except Exception as e:
            if attempt == max_retries - 1:
                return uid, {"error": f"Exception: {str(e)}"}
            time.sleep(delay)
            delay *= 2.0

    return uid, {"error": "Exceeded max retries"}


def main():
    cases = json.load(open(CASES_PATH))
    print(f"Total Tri-Planar Cohort: {len(cases)} cases")

    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.load(open(CACHE_PATH))
            print(f"Loaded existing cache with {len(cache)} entries.")
        except Exception:
            cache = {}

    to_process = [c for c in cases if c["uid"] not in cache or "error" in cache[c["uid"]]]
    print(f"Cases remaining to distill: {len(to_process)}")

    if not to_process:
        print("All cases completed!")
        return

    workers = 4
    start_time = time.time()
    completed = len(cache) - len([c for c in cache.values() if "error" in c])
    total_target = len(cases)

    print(f"Starting Multi-Threaded Tri-Planar Distillation with {workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(process_case, c): c for c in to_process}

        for future in concurrent.futures.as_completed(future_map):
            c = future_map[future]
            uid, res = future.result()
            cache[uid] = res

            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

            if "status" in res and res["status"] == "ok":
                completed += 1
                diag_count = len(res.get("annotation", {}).get("diagnoses", {}))
                elapsed_tot = time.time() - start_time
                speed = completed / (elapsed_tot / 60.0) if elapsed_tot > 0 else 0
                eta_m = (total_target - completed) / speed if speed > 0 else 0
                print(f"[{completed}/{total_target}] {uid[:18]}... -> OK ({diag_count} diagnoses) | Latency={res['elapsed_s']}s | Speed={speed:.1f} c/m | ETA={eta_m:.1f}m", flush=True)
            else:
                err_msg = res.get("error", "Unknown")[:60]
                print(f"[FAIL] {uid[:18]}... -> Error: {err_msg}", flush=True)

    print(f"\nFinished Tri-Planar Distillation in {(time.time() - start_time)/60:.1f} minutes!")


if __name__ == "__main__":
    main()
