import requests
import time
from pathlib import Path

KEY_PATH = Path("/home/eason/rsna-knee/.unisound_key")
API_KEY = KEY_PATH.read_text(encoding="utf-8").strip()
ENDPOINT = "https://maas-api.unisound.com/v1/chat/completions"
MODEL_NAME = "u2-radimed"

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Hello! Reply with OK"}],
    "max_tokens": 10
}

print(f"Sending test request to {ENDPOINT} using key: {API_KEY[:10]}... (timeout=45s)")
t0 = time.time()
try:
    r = requests.post(ENDPOINT, json=payload, headers=headers, timeout=45)
    print("Status:", r.status_code, f"Elapsed: {time.time()-t0:.2f}s")
    print("Body:", r.text[:300])
except Exception as e:
    print(f"Request exception after {time.time()-t0:.2f}s: {e}")
