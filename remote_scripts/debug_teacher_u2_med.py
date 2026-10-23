import base64
import io
import json
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

cases = json.load(open(CASES_PATH))
item = cases[184]
uid = item["uid"]
report = item["report"]
img_path = IMG_DIR / f"{uid}_montage.png"

im = Image.open(img_path)
buf = io.BytesIO()
im.save(buf, format="JPEG", quality=80)
b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

prompt = f"""You are an expert radiologist and medical vision AI teacher.
Montage of 12 coronal slices S1-S12.
Report:
{report}

Return JSON with diagnoses and slice relevance 0-3 and ROI [ymin, xmin, ymax, xmax].
Format:
{{"diagnoses": {{"<Name>": {{"relevant_slices": [{{"slice": "S6", "relevance": 3}}], "roi": {{"slice": "S6", "bbox": [0.4, 0.2, 0.6, 0.4]}}, "finding": "...", "confidence": 0.85}}}}}}
"""

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        }
    ],
    "temperature": 0.0,
    "max_tokens": 1500
}

print(f"Sending full teacher prompt to {MODEL_NAME}...")
t0 = time.time()
try:
    resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=60)
    print(f"Status: {resp.status_code}, Elapsed: {time.time()-t0:.2f}s")
    print("Response:\n", resp.text[:600])
except Exception as e:
    print(f"Exception after {time.time()-t0:.2f}s: {e}")
