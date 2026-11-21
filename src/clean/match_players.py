from difflib import SequenceMatcher
from pathlib import Path
import re
import sys
import unicodedata
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path("data/processed")
psm = pd.read_parquet(OUT / "player_season_minutes.parquet")
cand_raw = pd.read_parquet(OUT / "cohort_summer2025_candidates.parquet")


def norm_text(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(s.split())


CLUB_STOP = {
    "fc", "cf", "ac", "as", "rc", "rcd", "ca", "ud", "us", "ss", "ssc",
    "aj", "ogc", "sco", "vfl", "vfb", "tsg", "fsv", "bsc", "sv", "1",
    "de", "del", "la", "le", "les", "st", "saint", "calcio", "sport",
    "club", "foot", "football", "hotspur", "albion", "wanderers", "rovers"
}


def norm_club(s):
    toks = [t for t in norm_text(s).split() if t not in CLUB_STOP]
    return " ".join(toks)


# 1. Deduplicate multi-move players in summer 2025: keep earliest from_club and latest to_club
cand_sorted = cand_raw.sort_values(["player_id", "transfer_date", "transfer_fee"], ascending=[True, True, True])
first_moves = cand_sorted.groupby("player_id", as_index=False).first()[
    ["player_id", "from_club_id", "from_club_name", "transfer_date"]
].rename(
    columns={
        "from_club_id": "earliest_from_club_id",
        "from_club_name": "earliest_from_club_name",
        "transfer_date": "earliest_transfer_date",
    }
)
cand = cand_sorted.drop_duplicates("player_id", keep="last").merge(first_moves, on="player_id", how="left")

# 2. Map Transfermarkt club names -> WhoScored team names
ws_teams = sorted(psm["team"].dropna().unique())
ws_club_norm = {t: norm_club(t) for t in ws_teams}

CLUB_OVERRIDES = {
    "Nott'm Forest": "Nottingham Forest",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Bor. M'gladbach": "Borussia M.Gladbach",
    "Mönchengladbach": "Borussia M.Gladbach",
    "Bor. Dortmund": "Borussia Dortmund",
    "Dortmund": "Borussia Dortmund",
    "E. Frankfurt": "Eintracht Frankfurt",
    "Frankfurt": "Eintracht Frankfurt",
    "B. Leverkusen": "Bayer Leverkusen",
    "Leverkusen": "Bayer Leverkusen",
    "Bayern Munich": "Bayern",
    "Paris SG": "PSG",
    "R. Strasbourg": "Strasbourg",
    "Stade Rennais": "Rennes",
    "Stade Brestois": "Brest",
    "Stade Reims": "Reims",
    "AS Saint-Étienne": "Saint-Etienne",
    "Olympique Lyon": "Lyon",
    "Celta de Vigo": "Celta Vigo",
    "CD Leganés": "Leganes",
    "UD Las Palmas": "Las Palmas",
    "CA Osasuna": "Osasuna",
    "RCD Mallorca": "Mallorca",
    "RCD Espanyol": "Espanyol",
    "Deportivo Alavés": "Deportivo Alaves",
    "Alavés": "Deportivo Alaves",
    "Atlético": "Atletico",
    "Hellas Verona": "Verona",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli",
    "Parma": "Parma Calcio 1913",
    "Parma Calcio 1913": "Parma Calcio 1913",
    "Como 1907": "Como",
    "1.FC Köln": "FC Koln",
    "1.FSV Mainz 05": "Mainz",
    "1.FC Union Berlin": "Union Berlin",
    "1.FC Heidenheim 1846": "Heidenheim",
    "TSG Hoffenheim": "Hoffenheim",
    "SC Freiburg": "Freiburg",
    "VfL Wolfsburg": "Wolfsburg",
    "VfL Bochum": "Bochum",
    "VfB Stuttgart": "Stuttgart",
    "Stuttgart": "Stuttgart",
    "FC St. Pauli": "St. Pauli",
    "FC Augsburg": "Augsburg",
    "SV Werder Bremen": "Werder Bremen",
    "RB Leipzig": "RB Leipzig",
    "Leipzig": "RB Leipzig",
    "Wolves": "Wolverhampton",
}

tm_clubs = sorted(
    set(cand["from_club_name"].dropna())
    | set(cand["to_club_name"].dropna())
    | set(cand["earliest_from_club_name"].dropna())
)
club_map = {}
club_rows = []
for c in tm_clubs:
    if c in CLUB_OVERRIDES and CLUB_OVERRIDES[c] in ws_teams:
        best_t, score, method = CLUB_OVERRIDES[c], 1.0, "override"
    else:
        nc = norm_club(c)
        best_t, score = "", 0.0
        for wt, wnc in ws_club_norm.items():
            if nc == wnc and nc != "":
                best_t, score = wt, 1.0
                break
            sc = SequenceMatcher(None, nc, wnc).ratio()
            if sc > score:
                best_t, score = wt, sc
        method = "exact_norm" if score == 1.0 else "fuzzy"
    club_map[c] = best_t if score >= 0.72 else None
    club_rows.append({"tm_club": c, "ws_team": club_map[c], "score": round(score, 3), "method": method})

club_map_df = pd.DataFrame(club_rows)
club_map_df.to_csv(OUT / "club_name_map.csv", index=False)

print("=== CLUB MAPPING CHECK (non-1.0 or unmapped) ===")
non_exact_clubs = club_map_df[club_map_df["score"] < 1.0]
print(non_exact_clubs.to_string(index=False) if len(non_exact_clubs) else "All clubs matched at 1.0!")

# 3. Build WhoScored player lookup (now includes all GK names)
psm["norm_player"] = psm["player"].map(norm_text)
ws_players = (
    psm.groupby(["player_id", "player", "norm_player"], as_index=False)
    .agg(
        teams_2425=("team", lambda s: sorted(set(s[psm.loc[s.index, "season"] == 2425]))),
        teams_2526=("team", lambda s: sorted(set(s[psm.loc[s.index, "season"] == 2526]))),
        min90_2425=("minutes_90", lambda s: round(float(s[psm.loc[s.index, "season"] == 2425].sum()), 1)),
        min90_2526=("minutes_90", lambda s: round(float(s[psm.loc[s.index, "season"] == 2526].sum()), 1)),
    )
)

# 4. Match each candidate player to WhoScored player_id
match_rows = []
for row in cand.itertuples():
    n1 = norm_text(row.player_name)
    n2 = norm_text(row.name)
    exp_from = {club_map.get(row.from_club_name), club_map.get(row.earliest_from_club_name)} - {None}
    exp_to = {club_map.get(row.to_club_name)} - {None}

    exact = ws_players[ws_players["norm_player"].isin({n1, n2} - {""})]
    chosen = None
    method = "unmatched"
    sim = 0.0

    if len(exact) == 1:
        chosen = exact.iloc[0]
        method = "exact_unique"
        sim = 1.0
    elif len(exact) > 1:
        scored = []
        for er in exact.itertuples():
            c_hit = len(set(er.teams_2425) & exp_from) + len(set(er.teams_2526) & exp_to)
            scored.append((c_hit, er.min90_2425 + er.min90_2526, er))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        chosen = scored[0][2]
        method = f"exact_disambig(club_hits={scored[0][0]})"
        sim = 1.0
    else:
        pool = ws_players[
            ws_players["teams_2425"].apply(lambda ts: bool(set(ts) & exp_from))
            | ws_players["teams_2526"].apply(lambda ts: bool(set(ts) & exp_to))
        ]
        best_sc = 0.0
        best_er = None
        toks_tm = set((n1 + " " + n2).split())
        for er in pool.itertuples():
            sc1 = SequenceMatcher(None, n1, er.norm_player).ratio()
            sc2 = SequenceMatcher(None, n2, er.norm_player).ratio()
            toks_ws = set(er.norm_player.split())
            tok_overlap = len(toks_tm & toks_ws) / max(1, min(len(toks_tm), len(toks_ws)))
            sc = max(sc1, sc2, 0.85 * tok_overlap if toks_ws else 0.0)
            if sc > best_sc:
                best_sc = sc
                best_er = er
        if best_er is not None and best_sc >= 0.80:
            chosen = best_er
            method = "fuzzy_club_pool"
            sim = round(best_sc, 3)

    ws_to_exp = club_map.get(row.to_club_name)
    teams_2425_list = chosen.teams_2425 if chosen is not None else []
    already_at_dest = bool(ws_to_exp) and (ws_to_exp in teams_2425_list)

    match_rows.append({
        "tm_player_id": row.player_id,
        "tm_player_name": row.player_name,
        "tm_full_name": row.name,
        "age_years": row.age_years,
        "position": row.position,
        "sub_position": row.sub_position,
        "tm_from_club": row.from_club_name,
        "tm_earliest_from_club": row.earliest_from_club_name,
        "tm_to_club": row.to_club_name,
        "transfer_date": row.transfer_date,
        "transfer_fee": row.transfer_fee,
        "ws_from_club_expected": club_map.get(row.from_club_name),
        "ws_to_club_expected": ws_to_exp,
        "ws_player_id": int(chosen.player_id) if chosen is not None else pd.NA,
        "ws_player_name": chosen.player if chosen is not None else None,
        "ws_teams_2425": ",".join(teams_2425_list),
        "ws_teams_2526": ",".join(chosen.teams_2526) if chosen is not None else "",
        "ws_min90_2425": chosen.min90_2425 if chosen is not None else 0.0,
        "ws_min90_2526": chosen.min90_2526 if chosen is not None else 0.0,
        "flag_already_at_dest_2425": already_at_dest,
        "match_method": method,
        "match_sim": sim,
    })

matched = pd.DataFrame(match_rows)
matched["ws_player_id"] = matched["ws_player_id"].astype("Int64")
matched.to_parquet(OUT / "cohort_matched.parquet", index=False)
matched.to_csv(OUT / "cohort_matched.csv", index=False)

print("\n=== PLAYER MATCH SUMMARY ===")
print(f"Unique senior candidates: {len(matched)}")
print(matched["match_method"].value_counts().to_string())
print(f"Outfield matched: {(matched['position'] != 'Goalkeeper').mul(matched['ws_player_id'].notna()).sum()} / {(matched['position'] != 'Goalkeeper').sum()}")
print(f"Goalkeepers matched: {(matched['position'] == 'Goalkeeper').mul(matched['ws_player_id'].notna()).sum()} / {(matched['position'] == 'Goalkeeper').sum()}")
print(f"Flagged already at destination club in 2024-25: {matched['flag_already_at_dest_2425'].sum()}")

print("\n=== NON-EXACT & UNMATCHED ROWS ===")
non_exact = matched[matched["match_method"] != "exact_unique"]
cols = ["tm_player_id", "tm_player_name", "position", "tm_from_club", "tm_to_club", "ws_player_id", "ws_player_name", "ws_teams_2425", "ws_teams_2526", "ws_min90_2425", "ws_min90_2526", "match_method", "match_sim"]
print(non_exact[cols].to_string(index=False))