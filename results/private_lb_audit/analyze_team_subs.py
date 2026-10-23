import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Fetch all submissions by iterating or checking attributes
subs = api.competition_submissions('rsna-knee-abnormality-detection')
print(f"Total submissions returned: {len(subs)}")

records = []
for s in subs:
    rec = {
        "ref": getattr(s, "ref", getattr(s, "id", None)),
        "date": str(getattr(s, "date", "")),
        "public_score": float(s.public_score) if s.public_score is not None else None,
        "private_score": float(s.private_score) if s.private_score is not None else None,
        "is_selected": getattr(s, "is_selected", getattr(s, "selected", None)),
        "description": getattr(s, "description", "")
    }
    records.append(rec)

# Sort by private score descending
sorted_by_priv = sorted(records, key=lambda x: (x["private_score"] is not None, x["private_score"]), reverse=True)

print("\n--- ALL SUBMISSIONS (Chronological) ---")
for r in records:
    sel = " [SELECTED]" if r["is_selected"] else ""
    print(f"{r['ref']} | {r['date'][:16]} | Pub: {r['public_score']} | Priv: {r['private_score']}{sel} | {r['description'][:65]}")

print("\n--- TOP 10 BY PRIVATE SCORE ---")
for r in sorted_by_priv[:10]:
    sel = " [SELECTED]" if r["is_selected"] else ""
    print(f"{r['ref']} | {r['date'][:16]} | Pub: {r['public_score']} | Priv: {r['private_score']}{sel} | {r['description'][:65]}")

with open("scratch/team_submissions_full.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
