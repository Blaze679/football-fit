import pandas as pd

p = pd.read_parquet("data/processed/player_season_minutes.parquet")
print("columns:", list(p.columns))
print(p.head(5).to_string())
print("rows:", len(p), "| distinct player_id:", p["player_id"].nunique())
for col in ("team", "league", "season"):
    if col in p.columns:
        print(col, "->", sorted(p[col].astype(str).unique())[:12])