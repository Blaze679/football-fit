from pathlib import Path
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROCESSED = Path("data/processed")
sample_file = PROCESSED / "events_ENG-Premier_League_2024-2025.parquet"
df = pd.read_parquet(sample_file)

print("=== 1. RELEVANT COLUMNS IN events_ENG-Premier_League_2024-2025.parquet ===")
rel_cols = [
    c for c in df.columns
    if any(k in c.lower() for k in ["goal", "assist", "keypass", "pen", "own", "is_", "type", "outcome", "related"])
]
print(rel_cols)

print("\n=== 2. GOAL EVENTS & OWN-GOAL CHECK (2024-25 PL) ===")
goals = df[(df["type"] == "Goal") | (df["is_goal"] == True)]
print(f"Rows with type=='Goal': {(df['type'] == 'Goal').sum()} | is_goal==True: {df['is_goal'].fillna(False).sum()}")
q_goal_cols = [c for c in goals.columns if c.startswith("q_") and goals[c].fillna(False).any()]
print("Active q_* flags on Goal rows (top 20 by count):")
print(goals[q_goal_cols].fillna(False).sum().sort_values(ascending=False).head(20).to_string())

if "q_OwnGoal" in goals.columns:
    og = goals[goals["q_OwnGoal"].fillna(False)]
    print(f"\nOwn goals (q_OwnGoal==True): {len(og)}")
    print(og[["game", "minute", "team", "player", "type", "outcome_type"]].head(5).to_string(index=False))

print("\n=== 3. ASSIST QUALIFIER COUNTS & OVERLAP (2024-25 PL) ===")
assist_cols = [c for c in ["q_IntentionalAssist", "q_Assisted", "q_ShotAssist", "q_KeyPass"] if c in df.columns]
for c in assist_cols:
    sub = df[df[c].fillna(False)]
    print(f"{c}: {len(sub)} rows | types={sub['type'].value_counts().to_dict()} | outcomes={sub['outcome_type'].value_counts().to_dict()}")

print("\n=== 4. TOP 10 GOAL SCORERS (EXCL. OWN GOALS) IN 2024-25 PL ===")
non_og = goals[~goals["q_OwnGoal"].fillna(False)] if "q_OwnGoal" in goals.columns else goals
print(non_og.groupby(["player_id", "player", "team"]).size().sort_values(ascending=False).head(10).to_string())

for c in ["q_IntentionalAssist", "q_Assisted", "q_ShotAssist"]:
    if c in df.columns:
        print(f"\n=== 5. TOP 10 BY {c} IN 2024-25 PL ===")
        sub = df[df[c].fillna(False)]
        print(sub.groupby(["player_id", "player", "team"]).size().sort_values(ascending=False).head(10).to_string())