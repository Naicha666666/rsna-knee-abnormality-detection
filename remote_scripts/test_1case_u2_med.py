import base64
import json
import time
from pathlib import Path
import requests

KEY_PATH = Path("/home/eason/rsna-knee/.unisound_key")
API_KEY = KEY_PATH.read_text(encoding="utf-8").strip()
ENDPOINT = "https://maas-api.unisound.com/v1/chat/completions"
MODEL_NAME = "u2-med"

CASES_PATH = Path("/home/eason/rsna-knee/cohort_400_u2_radimed.json")
IMG_DIR = Path("/home/eason/rsna-knee/vlm_cohort_images")

cases = json.load(open(CASES_PATH))
item = cases[184]  # The first case that timed out previously
uid = item["uid"]
report = item["report"]
img_path = IMG_DIR / f"{uid}_montage.png"

print(f"Testing u2-med on Case 184: UID={uid}")
print(f"Image exists: {img_path.exists()}, size={img_path.stat().st_size} bytes")

b64_img = base64.b64encode(img_path.read_bytes()).decode("utf-8")

prompt = f"""You are an expert radiologist and medical vision AI teacher designed to train a Multiple Instance Learning (MIL) model.
Below is a 3D MRI volume of a patient's knee joint presented as a 3x4 montage gallery of 12 consecutive coronal slices, labeled S1 (anterior) through S12 (posterior).
Here is the official radiology report text:

[Radiology Report]
{report}

Task:
Generate structured slice-level spatial supervision for the Multiple Instance Learning (MIL) attention head.
For each relevant pathology (especially Lateral OA, Medial OA, PF OA, ACL, MCL, Meniscus tears, Effusion, Synovitis, Baker's, Contusion, Fracture):
1. Rate the diagnostic relevance of each slice on a discrete 0-3 scale:
   - 0 = irrelevant (no abnormal finding or non-diagnostic slice)
   - 1 = contextual / border (mild adjacent change)
   - 2 = relevant (clear abnormal signal or structural thinning)
   - 3 = strongly diagnostic (key slice demonstrating definitive pathology, e.g. full-thickness ulcer, bone exposure, spur, or frank tear)
2. For the most diagnostic slice (relevance=3), provide a normalized bounding box [ymin, xmin, ymax, xmax] (coordinates in [0.0, 1.0]) locating the lesion within that slice.

Output STRICT JSON:
{{
  "diagnoses": {{
    "<Diagnosis Name>": {{
      "primary_plane": "coronal",
      "anatomy": ["lateral compartment"],
      "depth_zone": "anterior",
      "relevant_slices": [
        {{"slice": "S6", "relevance": 3}},
        {{"slice": "S7", "relevance": 3}}
      ],
      "roi": {{
        "slice": "S7",
        "bbox": [0.4, 0.2, 0.6, 0.4]
      }},
      "finding": "Brief 1-sentence finding",
      "confidence": 0.85
    }}
  }}
}}
If normal with no abnormalities, output {{"diagnoses": {{}}}}."""

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
            ]
        }
    ],
    "temperature": 0.0,
    "max_tokens": 1500
}

t0 = time.time()
print(f"Sending request to Unisound {MODEL_NAME}...")
resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=45)
elapsed = time.time() - t0
print(f"Status: {resp.status_code}, Elapsed: {elapsed:.2f}s")
if resp.status_code == 200:
    content = resp.json()["choices"][0]["message"]["content"]
    print("Response:\n", content[:600])
else:
    print("Error:", resp.text)
