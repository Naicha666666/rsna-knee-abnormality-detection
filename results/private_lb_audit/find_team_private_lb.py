import sys, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

target_team_id = 16688751
page_token = None
page = 0
found = None
top_ranks = []
all_teams_snapshot = []

with api.build_kaggle_client() as kaggle:
    from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest
    
    while True:
        page += 1
        request = ApiGetLeaderboardRequest()
        request.competition_name = 'rsna-knee-abnormality-detection'
        api._set_paging(request, 100, page_token)
        response = kaggle.competitions.competition_api_client.get_leaderboard(request)
        
        subs = response.submissions or []
        for i, s in enumerate(subs):
            global_rank = (page - 1) * 100 + i + 1
            team_info = {
                "rank": global_rank,
                "team_id": getattr(s, 'team_id', None),
                "team_name": getattr(s, 'team_name', ''),
                "score": getattr(s, 'score', None),
                "date": str(getattr(s, 'submission_date', ''))
            }
            if global_rank <= 25:
                top_ranks.append(team_info)
            if global_rank % 100 == 0 or global_rank in [1, 10, 50, 100, 313, 627]:
                all_teams_snapshot.append(team_info)
                
            if team_info["team_id"] == target_team_id or 'X&Y' in team_info["team_name"]:
                found = team_info
                print(f"*** FOUND TARGET TEAM at Rank {global_rank}: {team_info} ***")
        
        if page % 5 == 0:
            last_item = subs[-1] if subs else None
            last_score = getattr(last_item, 'score', 'N/A') if last_item else 'N/A'
            print(f"Page {page} (Rank {page*100}), score at edge: {last_score}")
            
        page_token = response.next_page_token
        if not page_token:
            print("Reached end of leaderboard.")
            break
            
        if found and page >= 15: # Continue enough to get stats
            pass
            
        time.sleep(0.1)

with open("scratch/private_lb_top.json", "w", encoding="utf-8") as f:
    json.dump({"target": found, "top25": top_ranks, "snapshots": all_teams_snapshot}, f, ensure_ascii=False, indent=2)

print("Done. Saved snapshot.")
