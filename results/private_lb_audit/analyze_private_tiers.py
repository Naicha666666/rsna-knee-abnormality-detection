import sys, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Let's collect all leaderboard entries (or up to rank 1500) to get exact score distribution and thresholds
all_entries = []
page_token = None
page = 0

with api.build_kaggle_client() as kaggle:
    from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest
    
    while page < 20: # first 2000 teams covers Gold, Silver, Bronze, and down to 0.941
        page += 1
        request = ApiGetLeaderboardRequest()
        request.competition_name = 'rsna-knee-abnormality-detection'
        api._set_paging(request, 100, page_token)
        response = kaggle.competitions.competition_api_client.get_leaderboard(request)
        
        subs = response.submissions or []
        for i, s in enumerate(subs):
            global_rank = (page - 1) * 100 + i + 1
            all_entries.append({
                "rank": global_rank,
                "team_id": getattr(s, 'team_id', None),
                "team_name": getattr(s, 'team_name', ''),
                "score": float(getattr(s, 'score', 0.0)),
                "date": str(getattr(s, 'submission_date', ''))
            })
            
        page_token = response.next_page_token
        if not page_token:
            break
        time.sleep(0.1)

print(f"Fetched {len(all_entries)} entries.")
with open("scratch/private_lb_2000.json", "w", encoding="utf-8") as f:
    json.dump(all_entries, f, ensure_ascii=False, indent=2)

import pandas as pd
df = pd.DataFrame(all_entries)
print("\n--- Private Leaderboard Cutoffs ---")
for r in [1, 10, 50, 100, 200, 313, 500, 627, 700, 783, 800, 1000, 1500, 1700]:
    if r <= len(df):
        row = df.iloc[r-1]
        print(f"Rank {row['rank']}: Score {row['score']:.3f} | Team: {row['team_name']}")

print("\n--- Score Tier Value Counts ---")
print(df['score'].value_counts().sort_index(ascending=False))
