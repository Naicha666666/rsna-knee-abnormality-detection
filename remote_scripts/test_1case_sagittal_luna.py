import base64
import json
import time
from pathlib import Path
import requests

OPENROUTER_KEY = 'YOUR_OPENROUTER_KEY'
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
MODEL_NAME = 'openai/gpt-6-luna'

CASES_PATH = Path('/home/eason/rsna-knee/cohort_200_sagittal_luna.json')
IMG_DIR = Path('/home/eason/rsna-knee/vlm_cohort_images')

cases = json.load(open(CASES_PATH))
item = cases[0]
uid = item['uid']
report = item['report']
img_path = IMG_DIR / f'{uid}_sagittal_montage.png'

print(f'Testing Case 0: UID={uid}')
print(f'Image exists: {img_path.exists()}, size={img_path.stat().st_size} bytes')
print(f'Report (first 200 chars):\n{report[:200]}...\n')

b64_img = base64.b64encode(img_path.read_bytes()).decode('utf-8')

prompt = f"""You are an expert radiologist and medical vision AI teacher designed to train a Multiple Instance Learning (MIL) model.
Below is a 3D MRI volume of a patient's knee joint presented as a 3x4 montage gallery of 12 consecutive sagittal slices, labeled S1 through S12 (spanning the joint across the medial-to-lateral or lateral-to-medial dimension, with the central intercondylar notch containing the cruciate ligaments typically in slices S5-S8).
Inside each sagittal slice:
- Superior (femur) is at the top, Inferior (tibia) is at the bottom.
- Anterior (patella / extensor mechanism) is to one side, Posterior (popliteal fossa) is to the other side.
Here is the official radiology report text:

[Radiology Report]
{report}

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
      "depth_zone": "medial",
      "relevant_slices": [
        {{"slice": "S6", "relevance": 3}},
        {{"slice": "S7", "relevance": 3}},
        {{"slice": "S5", "relevance": 2}}
      ],
      "roi": {{
        "slice": "S6",
        "bbox": [0.35, 0.40, 0.65, 0.60]
      }},
      "finding": "Brief 1-sentence finding",
      "confidence": 0.85
    }}
  }}
}}
If normal with no abnormalities, output {{"diagnoses": {{}}}}."""

headers = {
    'Authorization': f'Bearer {OPENROUTER_KEY}',
    'Content-Type': 'application/json'
}
payload = {
    'model': MODEL_NAME,
    'messages': [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{b64_img}'
                    }
                }
            ]
        }
    ],
    'temperature': 0.05,
    'max_tokens': 4096
}

t0 = time.time()
print('Sending request to OpenRouter GPT-6 Luna...')
resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=90)
elapsed = time.time() - t0
print(f'Status: {resp.status_code}, Elapsed: {elapsed:.2f}s')

if resp.status_code == 200:
    data = resp.json()
    content = data['choices'][0]['message']['content']
    print('\nRaw Model Response:')
    print(content)
else:
    print('Error:', resp.text)
