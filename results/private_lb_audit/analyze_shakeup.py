import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

# Load public leaderboard
pub_df = pd.read_csv('scratch/leaderboard/rsna-knee-abnormality-detection-publicleaderboard-2026-10-23T19_48_11.csv')
print(f"Public LB teams: {len(pub_df)}")

# Load private leaderboard
with open('scratch/private_lb_2000.json', 'r', encoding='utf-8') as f:
    priv_entries = json.load(f)
priv_df = pd.DataFrame(priv_entries)
print(f"Private LB entries loaded: {len(priv_df)}")

# Merge on TeamId
merged = pd.merge(
    pub_df[['Rank', 'TeamId', 'TeamName', 'Score', 'LastSubmissionDate']].rename(
        columns={'Rank': 'pub_rank', 'Score': 'pub_score', 'LastSubmissionDate': 'pub_date'}
    ),
    priv_df[['rank', 'team_id', 'score', 'date']].rename(
        columns={'rank': 'priv_rank', 'score': 'priv_score', 'date': 'priv_date', 'team_id': 'TeamId'}
    ),
    on='TeamId'
)

merged['rank_diff'] = merged['pub_rank'] - merged['priv_rank'] # positive means climbed up, negative means dropped
merged['score_drop'] = merged['pub_score'] - merged['priv_score']

print("\n--- Team X&Y Status ---")
xy = merged[merged['TeamId'] == 16688751]
print(xy[['TeamName', 'pub_rank', 'pub_score', 'priv_rank', 'priv_score', 'rank_diff', 'score_drop']])

print("\n--- Top 15 Private LB Teams: Where did they come from? ---")
top15_priv = merged.sort_values('priv_rank').head(15)
for _, r in top15_priv.iterrows():
    diff_str = f"+{r['rank_diff']}" if r['rank_diff'] > 0 else f"{r['rank_diff']}"
    print(f"Priv #{r['priv_rank']:2d} ({r['priv_score']:.3f}) <- Pub #{r['pub_rank']:3d} ({r['pub_score']:.3f}) | Shift: {diff_str:>5} | Team: {r['TeamName']}")

print("\n--- Top 15 Public LB Teams: Where did they go? ---")
top15_pub = merged.sort_values('pub_rank').head(15)
for _, r in top15_pub.iterrows():
    diff_str = f"+{r['rank_diff']}" if r['rank_diff'] > 0 else f"{r['rank_diff']}"
    print(f"Pub #{r['pub_rank']:2d} ({r['pub_score']:.3f}) -> Priv #{r['priv_rank']:3d} ({r['priv_score']:.3f}) | Shift: {diff_str:>5} | Team: {r['TeamName']}")

print("\n--- Biggest Climbers (Top 10) ---")
climbers = merged.sort_values('rank_diff', ascending=False).head(10)
for _, r in climbers.iterrows():
    print(f"Pub #{r['pub_rank']} -> Priv #{r['priv_rank']} (+{r['rank_diff']}) | Score: {r['pub_score']} -> {r['priv_score']} | {r['TeamName']}")

print("\n--- Biggest Fallers (Top 10) ---")
fallers = merged.sort_values('rank_diff', ascending=True).head(10)
for _, r in fallers.iterrows():
    print(f"Pub #{r['pub_rank']} -> Priv #{r['priv_rank']} ({r['rank_diff']}) | Score: {r['pub_score']} -> {r['priv_score']} | {r['TeamName']}")
