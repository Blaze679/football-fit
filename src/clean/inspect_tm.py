from pathlib import Path
import pandas as pd

TM = Path("data/raw/transfermarkt")

for name in ["transfers.csv.gz", "players.csv.gz", "games.csv.gz"]:
    df = pd.read_csv(TM / name)
    print(f"=== {name} ({len(df):,} rows) ===")
    print("Columns:", list(df.columns))
    print(df.head(2).to_string(index=False))
    print()

raw = pd.read_csv(TM / "cohort_raw_summer2025.csv")
print(f"=== cohort_raw_summer2025.csv ({len(raw)} rows) ===")
print("Columns:", list(raw.columns))

tr = pd.read_csv(TM / "transfers.csv.gz")
sample_ids = raw["player_id"].dropna().unique()[:5]
print("\n=== Sample transfer histories for 5 raw-cohort players ===")
hist = tr[tr["player_id"].isin(sample_ids)].sort_values(["player_id", "transfer_date"])
cols = [c for c in ["player_id", "player_name", "transfer_season", "transfer_date", "from_club_name", "to_club_name", "transfer_fee"] if c in hist.columns]
print(hist[cols].to_string(index=False))