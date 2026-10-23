import base64
import json
import time
import io
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

# Test 1: Original PNG size
orig_bytes = img_path.read_bytes()
print(f"Original PNG size: {len(orig_bytes)} bytes")

# Test 2: Convert to compressed JPEG (quality 80)
im = Image.open(img_path)
buf = io.BytesIO()
im.save(buf, format="JPEG", quality=80)
jpeg_bytes = buf.getvalue()
print(f"Compressed JPEG size: {len(jpeg_bytes)} bytes (Reduction: {len(jpeg_bytes)/len(orig_bytes)*100:.1f}%)")

b64_img = base64.b64encode(jpeg_bytes).decode("utf-8")

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the knee abnormalities shown in this 3x4 montage based on the report:\n" + report[:150]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        }
    ],
    "temperature": 0.0,
    "max_tokens": 100
}

t0 = time.time()
print(f"Sending compressed JPEG ({len(jpeg_bytes)//1024} KB) to Unisound {MODEL_NAME} (timeout=30s)...")
try:
    resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=30)
    print(f"Status: {resp.status_code}, Elapsed: {time.time()-t0:.2f}s")
    print("Response:\n", resp.text[:300])
except Exception as e:
    print(f"Failed after {time.time()-t0:.2f}s: {e}")
