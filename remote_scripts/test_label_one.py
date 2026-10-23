import urllib.request
import json
import re

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker\x27s", "Contusion", "Fracture",
]

RUBRIC = """You grade knee MRI radiology reports for twelve findings.
Grade the report in whatever language it arrives in. Do not translate.

For EACH of the twelve findings, return two numbers:
  "p"    probability the finding is present in THIS knee, 0.0 to 1.0
  "conf" how much the report actually tells you about that finding, 0.0 to 1.0

The twelve findings:
  ACL              anterior cruciate ligament tear/rupture
  MCL              medial collateral ligament injury
  Medial Meniscus  medial meniscus tear
  Lateral Meniscus lateral meniscus tear
  Medial OA        medial compartment osteoarthritis / cartilage loss
  Lateral OA       lateral compartment osteoarthritis / cartilage loss
  PF OA            patellofemoral osteoarthritis / chondromalacia patellae
  Effusion         joint effusion / fluid in the joint
  Synovitis        synovitis, synovial thickening or proliferation
  Baker s Bakers (popliteal) cyst
  Contusion        bone contusion / bone bruise / marrow oedema from trauma
  Fracture         fracture of any bone in the study

Return ONLY a valid JSON object matching this exact key structure:
{"ACL": {"p": ..., "conf": ...}, "MCL": {"p": ..., "conf": ...}, ...}
No prose, no explanations, no markdown formatting."""

report = """TECHNIQUE: Multiplanar MRI of the right knee without contrast.
FINDINGS:
There is a full-thickness tear of the anterior cruciate ligament with anterior translation of the tibia. PCL, MCL, and LCL are intact.
Complex tear involving the body and posterior horn of the medial meniscus. Lateral meniscus is intact.
Moderate joint effusion with trace synovitis.
No fracture or bone contusion. No Baker cyst."""

payload = {
    "model": "google/gemma-4-12B-it",
    "messages": [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f"<report>\n{report}\n</report>"}
    ],
    "temperature": 0.0,
    "max_tokens": 600
}

req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
res = json.loads(urllib.request.urlopen(req).read())
content = res["choices"][0]["message"]["content"].strip()
print("Raw output:")
print(content)

match = re.search(r"\{.*\}", content, re.DOTALL)
if match:
    data = json.loads(match.group(0))
    print("\nParsed successfully! Total keys:", len(data))
    for k in LABELS:
        item = data.get(k, {})
        print(f"  {k:18s}: p={item.get(p)} conf={item.get(conf)}")
