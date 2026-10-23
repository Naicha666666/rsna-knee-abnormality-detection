"""Test 1 Tri-Planar Case with GPT-6 Luna via OpenRouter.
Feeds Coronal, Sagittal, and Axial montages simultaneously along with official radiology report.
Optimized for concise slice output and max_tokens=8192.
"""
import base64
import io
import json
import time
from pathlib import Path
from PIL import Image
import requests

OPENROUTER_KEY = "YOUR_OPENROUTER_KEY"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-6-luna"

CASES_PATH = Path("/home/eason/rsna-knee/cohort_200_triplanar_luna.json")
IMG_DIR = Path("/home/eason/rsna-knee/vlm_cohort_images")

cohort = json.load(open(CASES_PATH))
case = cohort[0]
uid = case["uid"]
print(f"Testing Case 0: UID={uid}")
print(f"Report Length: {len(case['report'])}")

def load_and_compress_img(path: Path) -> str:
    with Image.open(path) as im:
        im = im.convert("L")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

t0 = time.time()
f_cor = IMG_DIR / f"{uid}_montage.png"
f_sag = IMG_DIR / f"{uid}_sagittal_montage.png"
f_ax = IMG_DIR / f"{uid}_axial_montage.png"

b64_cor = load_and_compress_img(f_cor)
b64_sag = load_and_compress_img(f_sag)
b64_ax = load_and_compress_img(f_ax)
print(f"Loaded and compressed 3 montages in {time.time() - t0:.2f}s.")

prompt = f"""You are an elite musculoskeletal (MSK) radiologist and medical vision AI teacher designing cross-planar supervision for a multi-view 3D Deep Learning model.
Below are three distinct imaging planes of a patient's knee joint from the same MRI study:
- Image 1: CORONAL montage (12 slices, S1 anterior to S12 posterior)
- Image 2: SAGITTAL montage (12 slices, S1 lateral to S12 medial, or vice versa)
- Image 3: AXIAL montage (12 slices, S1 superior / suprapatellar to S12 inferior / proximal tibia)

Here is the official clinical radiology report for this patient:

[Radiology Report]
{case['report']}

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

messages = [
    {
        "role": "system",
        "content": "You are an expert MSK radiologist and 3D VLM annotation specialist. Output strictly valid JSON without Markdown fences or commentary."
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_cor}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_sag}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_ax}", "detail": "high"}}
        ]
    }
]

headers = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://kaggle.com/rsna-knee",
    "X-Title": "RSNA Knee Tri-Planar 3D Distillation"
}

payload = {
    "model": MODEL_NAME,
    "messages": messages,
    "temperature": 0.1,
    "max_tokens": 8192,
    "response_format": {"type": "json_object"}
}

t_api = time.time()
print("Sending Tri-Planar request to OpenRouter GPT-6 Luna (max_tokens=8192)...")
resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
print(f"Status Code: {resp.status_code} in {time.time() - t_api:.2f}s")

if resp.status_code == 200:
    res = resp.json()
    content = res["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    print(f"\nSuccessfully parsed {len(parsed.get('diagnoses', {}))} diagnoses!")
    for diag, info in parsed.get('diagnoses', {}).items():
        print(f"\n* Diagnosis: {diag}")
        print(f"  Primary Plane: {info.get('primary_diagnostic_plane')}")
        cor = [f"{s['slice']}(rel={s.get('relevance')})" for s in info.get('coronal_slices', [])]
        sag = [f"{s['slice']}(rel={s.get('relevance')})" for s in info.get('sagittal_slices', [])]
        ax = [f"{s['slice']}(rel={s.get('relevance')})" for s in info.get('axial_slices', [])]
        print(f"  Coronal Slices : {cor}")
        print(f"  Sagittal Slices: {sag}")
        print(f"  Axial Slices   : {ax}")
        print(f"  Synthesis      : {info.get('cross_planar_synthesis')}")
else:
    print(f"Error: {resp.text}")
