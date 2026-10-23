import json
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

subs = api.competition_submissions('rsna-knee-abnormality-detection')
print(f"Total submissions fetched: {len(subs)}")

data = []
for s in subs:
    item = {
        "ref": str(getattr(s, "ref", getattr(s, "id", None))),
        "date": str(getattr(s, "date", None)),
        "description": getattr(s, "description", None),
        "status": str(getattr(s, "status", None)),
        "publicScore": str(getattr(s, "public_score", getattr(s, "publicScore", None))),
        "privateScore": str(getattr(s, "private_score", getattr(s, "privateScore", None))),
        "submittedBy": getattr(s, "submitted_by", getattr(s, "submittedBy", None)),
        "selected": getattr(s, "selected", getattr(s, "is_selected", None)),
        "teamId": getattr(s, "team_id", getattr(s, "teamId", None)),
        "errorDescription": getattr(s, "error_description", getattr(s, "errorDescription", None)),
    }
    data.append(item)

with open("scratch/submissions_with_private.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Saved to scratch/submissions_with_private.json")
for item in data[:30]:
    print(f"{item['ref']} | {item['date']} | Pub: {item['publicScore']} | Priv: {item['privateScore']} | Sel: {item['selected']} | {item['description'][:60]}")
