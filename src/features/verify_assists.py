from pathlib import Path
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_parquet("data/processed/events_ENG-Premier_League_2024-2025.parquet")

print("=== 1. OWN-GOAL COLUMN CHECK ===")
og_cols = [c for c in df.columns if "own" in c.lower()]
print("Own-goal columns:", og_cols)
for c in og_cols:
    print(f"  {c} == True on Goal rows: {df.loc[df['type'] == 'Goal', c].fillna(False).sum()}")

goals = df[df["type"] == "Goal"].copy()
print("\n=== 2. SAMPLE GOAL ROWS WITH RELATED IDS ===")
cols = ["game_id", "event_id", "team", "player_id", "player", "related_event_id", "related_player_id", "q_Assisted"]
print(goals[cols].head(10).to_string(index=False))

assisted_goals = goals[goals["q_Assisted"].fillna(False)].copy()
print(f"\nAssisted goals (q_Assisted==True): {len(assisted_goals)}")
print(f"  with non-null related_player_id: {assisted_goals['related_player_id'].notna().sum()}")
print(f"  with non-null related_event_id:  {assisted_goals['related_event_id'].notna().sum()}")

# Join Goal -> assisting event via (game_id, related_event_id) == (game_id, event_id)
ev_lookup = df[["game_id", "event_id", "type", "team", "player_id", "player", "q_ShotAssist", "q_IntentionalAssist"]].dropna(subset=["event_id"]).copy()
merged = assisted_goals.merge(
    ev_lookup,
    left_on=["game_id", "related_event_id"],
    right_on=["game_id", "event_id"],
    how="left",
    suffixes=("_goal", "_assist"),
)
print("\n=== 3. LINKED ASSIST EVENTS (VIA related_event_id) ===")
print("Matched assisting events:", merged["event_id_assist"].notna().sum(), "/", len(assisted_goals))
print("Assisting event types:", merged["type_assist"].value_counts(dropna=False).to_dict())
same_player = (merged["related_player_id"] == merged["player_id_assist"]).sum()
print(f"related_player_id == linked event player_id: {same_player} / {merged['event_id_assist'].notna().sum()}")

print("\n=== 4. TOP 12 GOAL ASSISTERS IN 2024-25 PL ===")
top_assists = (
    merged.dropna(subset=["player_id_assist"])
    .groupby(["player_id_assist", "player_assist", "team_assist"])
    .size()
    .sort_values(ascending=False)
    .head(12)
)
print(top_assists.to_string())