import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Let's inspect all submissions of our team and write a markdown summary table
subs = api.competition_submissions('rsna-knee-abnormality-detection')
records = []
for s in subs:
    rec = {
        "ref": str(getattr(s, "ref", getattr(s, "id", None))),
        "date": str(getattr(s, "date", ""))[:19],
        "public_score": float(s.public_score) if s.public_score is not None else None,
        "private_score": float(s.private_score) if s.private_score is not None else None,
        "description": getattr(s, "description", "")
    }
    records.append(rec)

print(f"Total submissions analyzed: {len(records)}")
with open("scratch/all_submissions_report.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

for r in records:
    pub = f"{r['public_score']:.3f}" if r['public_score'] is not None else "N/A"
    priv = f"{r['private_score']:.3f}" if r['private_score'] is not None else "N/A"
    drop = f"{r['public_score'] - r['private_score']:.3f}" if (r['public_score'] and r['private_score']) else "N/A"
    print(f"{r['ref']} | {r['date']} | Pub: {pub} | Priv: {priv} | Drop: {drop} | {r['description'][:50]}")
