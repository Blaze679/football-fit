from pathlib import Path
import re
import sys
import unicodedata
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TM = Path("data/raw/transfermarkt")
PROCESSED = Path("data/processed")


def norm_text(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(s.split())


clubs_tm = pd.read_csv(TM / "clubs.csv.gz")
print(f"=== 1. clubs.csv.gz ({len(clubs_tm)} rows) ===")
print("Columns:", list(clubs_tm.columns))
print(clubs_tm.head(3).to_string(index=False))

# Check top-5 league clubs in clubs.csv.gz
top5_tm = clubs_tm[clubs_tm["domestic_competition_id"].isin(["GB1", "ES1", "IT1", "L1", "FR1"])]
print(f"\nTop-5 clubs in clubs.csv.gz: {len(top5_tm)}")
print(top5_tm[["club_id", "name", "domestic_competition_id", "url"]].head(10).to_string(index=False))

# Check player image_url coverage across all WhoScored players
ps = pd.read_parquet(PROCESSED / "player_style_features.parquet")
cohort = pd.read_parquet(PROCESSED / "cohort_matched.parquet")
players_tm = pd.read_csv(
    TM / "players.csv.gz",
    usecols=[
        "player_id", "name", "first_name", "last_name", "last_season",
        "current_club_id", "current_club_name", "country_of_citizenship",
        "foot", "height_in_cm", "market_value_in_eur", "image_url"
    ],
)
players_tm_recent = players_tm[players_tm["last_season"] >= 2023].copy()
players_tm_recent["norm_name"] = players_tm_recent["name"].map(norm_text)

# Unique WhoScored players
ws_unique = ps[["player_id", "player", "team", "league"]].drop_duplicates("player_id").copy()
ws_unique["norm_name"] = ws_unique["player"].map(norm_text)

# Direct match via cohort_matched first, then unique norm_name
cohort_map = (
    cohort.dropna(subset=["ws_player_id"])
    .drop_duplicates("ws_player_id")[["ws_player_id", "tm_player_id"]]
    .copy()
)
cohort_map["ws_player_id"] = cohort_map["ws_player_id"].astype(int)

ws_unique = ws_unique.merge(cohort_map, left_on="player_id", right_on="ws_player_id", how="left")
tm_unique_name = players_tm_recent.drop_duplicates("norm_name", keep=False)[["norm_name", "player_id"]].rename(
    columns={"player_id": "tm_id_by_name"}
)
ws_unique = ws_unique.merge(tm_unique_name, on="norm_name", how="left")
ws_unique["tm_player_id"] = ws_unique["tm_player_id"].combine_first(ws_unique["tm_id_by_name"])

ws_unique = ws_unique.merge(
    players_tm[["player_id", "image_url", "market_value_in_eur", "foot", "country_of_citizenship"]],
    left_on="tm_player_id",
    right_on="player_id",
    how="left",
)

print(f"\n=== 2. PLAYER IMAGE & METADATA COVERAGE ===")
print(f"Unique WhoScored players: {len(ws_unique)}")
print(f"Matched to TM player_id:  {ws_unique['tm_player_id'].notna().sum()} / {len(ws_unique)}")
print(f"With non-null image_url:  {ws_unique['image_url'].notna().sum()} / {len(ws_unique)}")
print("Sample image URLs:")
print(ws_unique.dropna(subset=["image_url"])[["player", "team", "image_url"]].head(5).to_string(index=False))