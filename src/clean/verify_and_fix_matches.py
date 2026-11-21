from pathlib import Path
import pandas as pd

OUT = Path("data/processed")
psm = pd.read_parquet(OUT / "player_season_minutes.parquet")
matched = pd.read_parquet(OUT / "cohort_matched.parquet")
club_map_df = pd.read_csv(OUT / "club_name_map.csv")

# 1. Check what WhoScored calls Parma and search events directly for the 9 unmatched GKs
ws_teams = sorted(psm["team"].dropna().unique())
print("=== WhoScored teams containing 'Parma' ===")
print([t for t in ws_teams if "parma" in t.lower()])

missing_gks = [
    "Silvestri", "Dmitrovic", "Dmitrović", "Bettinelli",
    "Lecomte", "Scuffet", "Travers", "Turner", "Chevalier", "Tenas"
]
print("\n=== Direct search in events_*.parquet for the 9 unmatched GKs ===")
found_rows = []
for pq in sorted(OUT.glob("events_*.parquet")):
    ev = pd.read_parquet(pq, columns=["season", "team", "player_id", "player"])
    ev = ev.dropna(subset=["player_id", "player"]).drop_duplicates(["season", "team", "player_id"])
    hit = ev[ev["player"].str.contains("|".join(missing_gks), case=False, na=False)]
    if len(hit):
        hit = hit.copy()
        hit["file"] = pq.name
        found_rows.append(hit)

if found_rows:
    found_df = pd.concat(found_rows, ignore_index=True)
    print(found_df.to_string(index=False))
else:
    print("None found in events_*.parquet.")

# 2. Fix club_name_map.csv for the 4 missed clubs
parma_ws = next((t for t in ws_teams if "parma" in t.lower()), None)
fix_clubs = {
    "Dortmund": "Borussia Dortmund",
    "Leipzig": "RB Leipzig",
    "Mönchengladbach": "Borussia M.Gladbach",
    "Parma": parma_ws,
}
for idx, r in club_map_df.iterrows():
    if r["tm_club"] in fix_clubs and fix_clubs[r["tm_club"]] is not None:
        club_map_df.at[idx, "ws_team"] = fix_clubs[r["tm_club"]]
        club_map_df.at[idx, "score"] = 1.0
        club_map_df.at[idx, "method"] = "override"
club_map_df.to_csv(OUT / "club_name_map.csv", index=False)
cmap = dict(zip(club_map_df["tm_club"], club_map_df["ws_team"]))

# 3. Reject false fuzzy matches (match_sim < 0.80) and update expected club columns
bad_fuzzy = (matched["match_method"] == "fuzzy_club_pool") & (matched["match_sim"] < 0.80)
matched.loc[bad_fuzzy, ["ws_player_id", "ws_player_name", "ws_teams_2425", "ws_teams_2526"]] = [pd.NA, None, "", ""]
matched.loc[bad_fuzzy, ["ws_min90_2425", "ws_min90_2526", "match_sim"]] = [0.0, 0.0, 0.0]
matched.loc[bad_fuzzy, "match_method"] = "unmatched"

matched["ws_from_club_expected"] = matched["tm_from_club"].map(cmap)
matched["ws_to_club_expected"] = matched["tm_to_club"].map(cmap)

# Flag players whose 2024-25 WhoScored club was ALREADY their 2025-26 destination club
matched["flag_already_at_dest_2425"] = matched.apply(
    lambda r: bool(r["ws_to_club_expected"]) and (r["ws_to_club_expected"] in str(r["ws_teams_2425"]).split(",")),
    axis=1,
)

matched.to_parquet(OUT / "cohort_matched.parquet", index=False)
matched.to_csv(OUT / "cohort_matched.csv", index=False)

print("\n=== UPDATED MATCH SUMMARY ===")
print(matched["match_method"].value_counts().to_string())
print(f"\nOutfield matched: {(matched['position'] != 'Goalkeeper').mul(matched['ws_player_id'].notna()).sum()} / {(matched['position'] != 'Goalkeeper').sum()}")
print(f"Goalkeepers matched: {(matched['position'] == 'Goalkeeper').mul(matched['ws_player_id'].notna()).sum()} / {(matched['position'] == 'Goalkeeper').sum()}")
print(f"Flagged already at destination club in 2024-25: {matched['flag_already_at_dest_2425'].sum()}")
print(matched[matched["flag_already_at_dest_2425"]][["tm_player_name", "position", "tm_from_club", "tm_to_club", "ws_teams_2425", "ws_teams_2526", "ws_min90_2425", "ws_min90_2526"]].to_string(index=False))