import unicodedata
from pathlib import Path
import pandas as pd


def norm_name(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().strip().split())


psm = pd.read_parquet("data/processed/player_season_minutes.parquet")
cand = pd.read_parquet("data/processed/cohort_summer2025_candidates.parquet")

print("=== player_season_minutes.parquet ===")
print("Shape:", psm.shape)
print("Columns:", list(psm.columns))
print(psm.head(5).to_string(index=False))

print("\n=== Duplicate player_ids in candidates ===")
dups = cand[cand.duplicated("player_id", keep=False)]
print(f"Duplicate rows: {len(dups)}")
if len(dups) > 0:
    print(dups[["player_id", "player_name", "from_club_name", "to_club_name", "transfer_date", "transfer_fee"]].to_string(index=False))

# Check unique WhoScored team names vs Transfermarkt club names
ws_teams = sorted(psm["team"].dropna().unique()) if "team" in psm.columns else []
tm_clubs = sorted(set(cand["from_club_name"].dropna()) | set(cand["to_club_name"].dropna()))
print(f"\nUnique WhoScored teams: {len(ws_teams)} | Unique TM cohort clubs: {len(tm_clubs)}")
print("Sample WS teams:", ws_teams[:20])
print("Sample TM clubs:", tm_clubs[:20])

# Test exact accent-free full-name matches in WhoScored
ws_names = set(psm["player"].dropna().map(norm_name)) if "player" in psm.columns else set()
cand["norm_player_name"] = cand["player_name"].map(norm_name)
cand["norm_full_name"] = cand["name"].map(norm_name)
exact_hit = cand["norm_player_name"].isin(ws_names) | cand["norm_full_name"].isin(ws_names)
print(f"\nExact accent-free name match in WhoScored (any season): {exact_hit.sum()} / {len(cand)}")