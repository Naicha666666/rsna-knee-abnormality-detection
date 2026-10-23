import json
from pathlib import Path

CACHE_PATH = Path("/home/eason/rsna-knee/u2_radimed_400_cache.json")
data = json.load(open(CACHE_PATH, encoding="utf-8"))

valid_items = {k: v for k, v in data.items() if "diagnoses" in v}
error_items = {k: v for k, v in data.items() if "error" in v}

print(f"Total in cache: {len(data)}")
print(f"Valid successfully completed cases: {len(valid_items)}")
print(f"Error / failed cases due to API timeout: {len(error_items)}")

# Clean cache to preserve only valid cases
with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(valid_items, f, indent=2, ensure_ascii=False)

print(f"Cleaned cache saved! Retained {len(valid_items)} valid cases.")
