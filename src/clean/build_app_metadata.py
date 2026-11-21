from difflib import SequenceMatcher
from pathlib import Path
import re
import sys
import unicodedata
import numpy as np
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


CLUB_STOP = {
    "fc", "cf", "ac", "as", "rc", "rcd", "ca", "ud", "us", "ss", "ssc",
    "aj", "ogc", "sco", "vfl", "vfb", "tsg", "fsv", "bsc", "sv", "1",
    "de", "del", "la", "le", "les", "st", "saint", "calcio", "sport",
    "club", "foot", "football", "hotspur", "albion", "wanderers", "rovers",
    "olympique", "stade", "real", "sporting", "athletic", "association"
}


def norm_club(s):
    toks = [t for t in norm_text(s).split() if t not in CLUB_STOP]
    return " ".join(toks)


# ==============================================================================
# 1. LINK ALL WHOSCORED CLUBS TO TRANSFERMARKT CLUBS.CSV.GZ
# ==============================================================================
cs = pd.read_parquet(PROCESSED / "club_style_features.parquet")
clubs_tm = pd.read_csv(TM / "clubs.csv.gz")
clubs_tm_top = clubs_tm[clubs_tm["domestic_competition_id"].isin(["GB1", "ES1", "IT1", "L1", "FR1"])].copy()
clubs_tm_top["norm_c"] = clubs_tm_top["name"].map(norm_club)

CLUB_NAME_FIX = {
    "Man Utd": "Manchester United Football Club",
    "Man City": "Manchester City Football Club",
    "Nottingham Forest": "Nottingham Forest Football Club",
    "Wolverhampton": "Wolverhampton Wanderers Football Club",
    "Tottenham": "Tottenham Hotspur Football Club",
    "Newcastle": "Newcastle United Football Club",
    "West Ham": "West Ham United Football Club",
    "Brighton": "Brighton & Hove Albion Football Club",
    "Leicester": "Leicester City Football Club",
    "Ipswich": "Ipswich Town Football Club",
    "Leeds": "Leeds United Football Club",
    "Burnley": "Burnley Football Club",
    "Sunderland": "Sunderland Association Football Club",
    "Atletico": "Club Atlético de Madrid S.A.D.",
    "Athletic Club": "Athletic Club Bilbao",
    "Deportivo Alaves": "Deportivo Alavés S.A.D.",
    "Leganes": "CD Leganés",
    "Celta Vigo": "Real Club Celta de Vigo S.A.D.",
    "Real Betis": "Real Betis Balompié S.A.D.",
    "Real Sociedad": "Real Sociedad de Fútbol S.A.D.",
    "Bayern": "FC Bayern München",
    "Borussia Dortmund": "Borussia Dortmund",
    "Borussia M.Gladbach": "Borussia Verein für Leibesübungen 1900 Mönchengladbach",
    "Bayer Leverkusen": "Bayer 04 Leverkusen Fußball",
    "Eintracht Frankfurt": "Eintracht Frankfurt Fußball AG",
    "FC Koln": "1.FC Köln",
    "Mainz": "1.FSV Mainz 05",
    "Union Berlin": "1.FC Union Berlin",
    "Heidenheim": "1.FC Heidenheim 1846",
    "St. Pauli": "FC St. Pauli",
    "Holstein Kiel": "Kieler Sportvereinigung Holstein von 1900",
    "Inter": "Football Club Internazionale Milano S.p.A.",
    "AC Milan": "Associazione Calcio Milan",
    "Roma": "Associazione Sportiva Roma",
    "Lazio": "Società Sportiva Lazio S.p.A.",
    "Napoli": "Società Sportiva Calcio Napoli",
    "Juventus": "Juventus Football Club",
    "PSG": "Paris Saint-Germain Football Club",
    "Paris FC": "Paris FC",
    "Lyon": "Olympique Lyonnais",
    "Marseille": "Olympique de Marseille",
    "Monaco": "Association Sportive de Monaco Football Club",
    "Lille": "Lille Olympique Sporting Club",
    "Nice": "Olympique Gymnaste Club Nice Côte d'Azur",
    "Rennes": "Stade Rennais Football Club",
    "Lens": "Racing Club de Lens",
    "Strasbourg": "Racing Club de Strasbourg Alsace",
    "Brest": "Stade Brestois 29",
    "Reims": "Stade de Reims",
    "Saint-Etienne": "AS Saint-Étienne",
    "Nantes": "Football Club de Nantes",
    "Toulouse": "Toulouse Football Club",
    "Auxerre": "Association de la Jeunesse Auxerroise",
    "Angers": "Angers SCO",
    "Le Havre": "Le Havre Athletic Club",
    "Montpellier": "Montpellier Hérault Sport Club",
    "Lorient": "Football Club Lorient-Bretagne Sud",
    "Metz": "Football Club de Metz",
}

ws_teams = sorted(cs["team"].dropna().unique())
club_meta_rows = []
for wt in ws_teams:
    matched_row = None
    if wt in CLUB_NAME_FIX:
        sub = clubs_tm_top[clubs_tm_top["name"] == CLUB_NAME_FIX[wt]]
        if not sub.empty:
            matched_row = sub.iloc[0]
    if matched_row is None:
        nwt = norm_club(wt)
        best_sc, best_r = 0.0, None
        for _, cr in clubs_tm_top.iterrows():
            sc = 1.0 if (nwt != "" and nwt == cr["norm_c"]) else SequenceMatcher(None, nwt, cr["norm_c"]).ratio()
            if sc > best_sc:
                best_sc, best_r = sc, cr
        if best_r is not None and best_sc >= 0.72:
            matched_row = best_r

    club_meta_rows.append({
        "team": wt,
        "tm_club_id": int(matched_row["club_id"]) if matched_row is not None else pd.NA,
        "tm_club_name": matched_row["name"] if matched_row is not None else wt,
        "stadium_name": matched_row["stadium_name"] if matched_row is not None else None,
        "stadium_seats": int(matched_row["stadium_seats"]) if (matched_row is not None and pd.notna(matched_row["stadium_seats"])) else pd.NA,
        "coach_name": matched_row["coach_name"] if matched_row is not None else None,
        "average_age": float(matched_row["average_age"]) if (matched_row is not None and pd.notna(matched_row["average_age"])) else np.nan,
        "squad_size": int(matched_row["squad_size"]) if (matched_row is not None and pd.notna(matched_row["squad_size"])) else pd.NA,
        "net_transfer_record": matched_row["net_transfer_record"] if matched_row is not None else None,
    })

club_meta = pd.DataFrame(club_meta_rows)
print(f"=== 1. CLUB METADATA LINKAGE ===")
print(f"Matched WhoScored clubs to TM clubs.csv.gz: {club_meta['tm_club_id'].notna().sum()} / {len(club_meta)}")
unmatched_c = club_meta[club_meta["tm_club_id"].isna()]["team"].tolist()
if unmatched_c:
    print("Unmatched clubs:", unmatched_c)

# Compute Top-5 League percentiles & plain-English Tactical Identity per club-season
cs_full = cs.merge(club_meta, on="team", how="left")
for col, inv in [
    ("poss_share", False),
    ("ppda", True),
    ("field_tilt", False),
    ("prog_pass_share", False),
    ("long_ball_share", False),
    ("def_line_height_m", False),
    ("wide_touch_share", False),
    ("tempo_passes_per_poss_min", False),
    ("box_touches_p90", False),
]:
    rank_s = cs_full.groupby("season")[col].rank(pct=True, ascending=not inv)
    cs_full[f"pct_{col}"] = (rank_s * 99).round(0).fillna(50).astype(int)


def club_identity(r) -> str:
    poss = r["pct_poss_share"]
    press = r["pct_ppda"]
    direct = r["pct_long_ball_share"]
    tilt = r["pct_field_tilt"]
    if poss >= 70 and press >= 65:
        return "High-Pressing Possession Dominator"
    if poss >= 70 and press < 65:
        return "Patient Ball-Control & Territory"
    if press >= 70 and direct >= 60:
        return "High-Tempo Vertical Gegenpresser"
    if press >= 70:
        return "Aggressive Front-Foot Presser"
    if direct >= 70 and tilt < 45:
        return "Direct Low-Block & Counter-Attack"
    if tilt >= 65:
        return "Front-Foot Territorial Attackers"
    if poss < 35:
        return "Compact Mid/Low-Block Transition"
    return "Balanced Hybrid System"


cs_full["tactical_identity"] = cs_full.apply(club_identity, axis=1)
cs_full.to_parquet(PROCESSED / "club_ui_metadata.parquet", index=False)

# ==============================================================================
# 2. LINK ALL WHOSCORED PLAYERS TO TRANSFERMARKT PORTRAITS & METADATA
# ==============================================================================
ps = pd.read_parquet(PROCESSED / "player_style_features.parquet")
cohort = pd.read_parquet(PROCESSED / "cohort_matched.parquet")
players_tm = pd.read_csv(
    TM / "players.csv.gz",
    usecols=[
        "player_id", "name", "first_name", "last_name", "last_season",
        "current_club_id", "current_club_name", "country_of_citizenship",
        "date_of_birth", "foot", "height_in_cm", "market_value_in_eur",
        "position", "sub_position", "image_url",
    ],
)
players_tm["date_of_birth"] = pd.to_datetime(players_tm["date_of_birth"], errors="coerce")
players_tm["age_2025_tm"] = ((pd.Timestamp("2025-07-01") - players_tm["date_of_birth"]).dt.days / 365.25).round(1)
players_tm_recent = players_tm[players_tm["last_season"] >= 2023].copy()
players_tm_recent["norm_name"] = players_tm_recent["name"].map(norm_text)

# Build lookup from norm_name -> list of candidate TM rows
tm_by_name = {}
for _, tr in players_tm_recent.iterrows():
    nn = tr["norm_name"]
    if nn:
        tm_by_name.setdefault(nn, []).append(tr)

# Known exact matches from cohort_matched
cohort_known = (
    cohort.dropna(subset=["ws_player_id", "tm_player_id"])
    .drop_duplicates("ws_player_id")
    .set_index("ws_player_id")["tm_player_id"]
    .to_dict()
)
tm_by_id = players_tm.set_index("player_id").to_dict("index")

team_to_tm_club = dict(zip(club_meta["team"], club_meta["tm_club_id"]))

ws_unique = ps.sort_values("minutes_90", ascending=False).drop_duplicates("player_id")[
    ["player_id", "player", "team", "league", "position", "sub_position", "age_2025"]
].copy()

p_meta_rows = []
for r in ws_unique.itertuples():
    ws_pid = int(r.player_id)
    tm_rec = None
    if ws_pid in cohort_known and int(cohort_known[ws_pid]) in tm_by_id:
        tm_rec = tm_by_id[int(cohort_known[ws_pid])]
        tm_pid = int(cohort_known[ws_pid])
    else:
        nn = norm_text(r.player)
        cands = tm_by_name.get(nn, [])
        if len(cands) == 1:
            tm_rec = cands[0]
            tm_pid = int(tm_rec["player_id"])
        elif len(cands) > 1:
            exp_cid = team_to_tm_club.get(r.team)
            # Rank by club match, then last_season >= 2024, then market_value
            cands_sorted = sorted(
                cands,
                key=lambda x: (
                    1 if (pd.notna(exp_cid) and x["current_club_id"] == exp_cid) else 0,
                    1 if x["last_season"] >= 2024 else 0,
                    float(x["market_value_in_eur"]) if pd.notna(x["market_value_in_eur"]) else 0.0,
                ),
                reverse=True,
            )
            tm_rec = cands_sorted[0]
            tm_pid = int(tm_rec["player_id"])
        else:
            tm_pid = pd.NA

    p_meta_rows.append({
        "player_id": ws_pid,
        "tm_player_id": tm_pid,
        "image_url": tm_rec["image_url"] if tm_rec is not None else None,
        "market_value_eur": float(tm_rec["market_value_in_eur"]) if (tm_rec is not None and pd.notna(tm_rec["market_value_in_eur"])) else np.nan,
        "foot": tm_rec["foot"] if tm_rec is not None else None,
        "height_cm": int(tm_rec["height_in_cm"]) if (tm_rec is not None and pd.notna(tm_rec["height_in_cm"])) else pd.NA,
        "nationality": tm_rec["country_of_citizenship"] if tm_rec is not None else None,
        "tm_position": tm_rec["position"] if tm_rec is not None else r.position,
        "tm_sub_position": tm_rec["sub_position"] if tm_rec is not None else r.sub_position,
        "tm_age_2025": tm_rec["age_2025_tm"] if tm_rec is not None else r.age_2025,
    })

p_meta_df = pd.DataFrame(p_meta_rows)
print(f"\n=== 2. PLAYER UI METADATA & PORTRAIT COVERAGE ===")
print(f"Unique WhoScored players: {len(p_meta_df)}")
print(f"Matched with portrait URL: {p_meta_df['image_url'].notna().sum()} / {len(p_meta_df)}")

# Merge onto player_style_features and compute position-peer percentiles (0..99)
ps_full = ps.merge(p_meta_df, on="player_id", how="left")
ps_full["position"] = ps_full["position"].fillna(ps_full["tm_position"]).fillna("Midfield")
ps_full["sub_position"] = ps_full["sub_position"].fillna(ps_full["tm_sub_position"]).fillna(ps_full["position"])
ps_full["age_2025"] = ps_full["age_2025"].fillna(ps_full["tm_age_2025"])

# Position-peer percentiles (calculated against players in same season & position with >=450 min90)
PCT_METRICS = [
    ("ga_p90", "Goal & Assist Output", False),
    ("shots_p90", "Shot Volume", False),
    ("box_touches_p90", "Penalty Box Presence", False),
    ("key_passes_p90", "Chance Creation (Key Passes)", False),
    ("throughballs_p90", "Line-Breaking Throughballs", False),
    ("prog_passes_p90", "Progressive Passing", False),
    ("f3_passes_succ_p90", "Final-Third Entry Passing", False),
    ("pass_acc", "Pass Retention Accuracy", False),
    ("takeons_att_p90", "1v1 Dribbling & Carrying", False),
    ("crosses_p90", "Wide Crossing Delivery", False),
    ("high_press_actions_p90", "High-Press Work Rate", False),
    ("tackles_int_p90", "Ball-Winning (Tackles & Int.)", False),
    ("ball_recoveries_p90", "Loose-Ball Recoveries", False),
    ("clearances_p90", "Box Clearance Defending", False),
    ("aerial_win_rate", "Aerial Duel Dominance", False),
    ("dispossessed_p90", "Ball Security (Low Turnovers)", True),
    ("gk_saves_p90", "Shot-Stopping Saves", False),
    ("gk_claims_p90", "Cross Claiming Command", False),
    ("gk_sweeper_p90", "Sweeper-Keeper Proactive Actions", False),
]

for col, _, _ in PCT_METRICS:
    ps_full[f"pct_{col}"] = 50

for (season, pos), grp in ps_full.groupby(["season", "position"]):
    qual = grp[grp["minutes_90"] >= 450]
    ref = qual if len(qual) >= 15 else grp
    for col, _, inv in PCT_METRICS:
        ref_vals = ref[col].dropna().values
        if len(ref_vals) == 0:
            continue
        vals = grp[col].fillna(np.nanmedian(ref_vals)).values
        # Percentile rank against reference pool
        pcts = np.array([np.mean(ref_vals <= v) * 99.0 for v in vals])
        if inv:
            pcts = 99.0 - pcts
        ps_full.loc[grp.index, f"pct_{col}"] = np.clip(np.round(pcts), 1, 99).astype(int)

# 6 Non-technical Scouting Pillar Scores (0-99)
ps_full["pillar_progression"] = ps_full[["pct_prog_passes_p90", "pct_f3_passes_succ_p90"]].mean(axis=1).round(0).astype(int)
ps_full["pillar_creativity"] = ps_full[["pct_key_passes_p90", "pct_throughballs_p90"]].mean(axis=1).round(0).astype(int)
ps_full["pillar_ball_security"] = ps_full[["pct_pass_acc", "pct_dispossessed_p90"]].mean(axis=1).round(0).astype(int)
ps_full["pillar_wide_dribbling"] = ps_full[["pct_takeons_att_p90", "pct_crosses_p90"]].mean(axis=1).round(0).astype(int)
ps_full["pillar_goal_threat"] = ps_full[["pct_ga_p90", "pct_shots_p90", "pct_box_touches_p90"]].mean(axis=1).round(0).astype(int)
ps_full["pillar_defending"] = ps_full[["pct_high_press_actions_p90", "pct_tackles_int_p90", "pct_ball_recoveries_p90"]].mean(axis=1).round(0).astype(int)

OUTFIELD_STRENGTH_CANDIDATES = [
    ("pct_prog_passes_p90", "Progressive Passer"),
    ("pct_key_passes_p90", "Chance Creator"),
    ("pct_throughballs_p90", "Throughball Vision"),
    ("pct_pass_acc", "Press-Resistant Distributor"),
    ("pct_takeons_att_p90", "1v1 Dribbler"),
    ("pct_crosses_p90", "Wide Crosser"),
    ("pct_box_touches_p90", "Penalty-Box Threat"),
    ("pct_shots_p90", "High-Volume Shooter"),
    ("pct_ga_p90", "End-Product Finisher"),
    ("pct_high_press_actions_p90", "Front-Foot Presser"),
    ("pct_tackles_int_p90", "Ball-Winning Disruptor"),
    ("pct_aerial_win_rate", "Aerial Dominant"),
]
GK_STRENGTH_CANDIDATES = [
    ("pct_gk_saves_p90", "High-Volume Shot Stopper"),
    ("pct_gk_claims_p90", "Commanding Cross Claimer"),
    ("pct_gk_sweeper_p90", "Proactive Sweeper-Keeper"),
    ("pct_pass_acc", "Composed Ball-Playing GK"),
]


def summarize_traits(r):
    cands = GK_STRENGTH_CANDIDATES if r["position"] == "Goalkeeper" else OUTFIELD_STRENGTH_CANDIDATES
    scored = [(int(r[c]), lbl) for c, lbl in cands]
    scored.sort(key=lambda x: x[0], reverse=True)
    top3 = [f"{lbl} ({p}th pct)" for p, lbl in scored[:3] if p >= 65]
    if not top3:
        top3 = [f"{lbl} ({p}th pct)" for p, lbl in scored[:2]]
    watch = [f"{lbl} ({p}th pct)" for p, lbl in reversed(scored[-2:]) if p <= 35]
    return " · ".join(top3), (" · ".join(watch) if watch else "No major statistical weaknesses")


traits = ps_full.apply(summarize_traits, axis=1)
ps_full["top_strengths"] = [t[0] for t in traits]
ps_full["watch_areas"] = [t[1] for t in traits]

# ==============================================================================
# 3. INSPECT & NAME THE 8 STYLE CLUSTERS IN player_style_space_2425.parquet
# ==============================================================================
space = pd.read_parquet(PROCESSED / "player_style_space_2425.parquet")
space = space.merge(
    ps_full[ps_full["season"] == 2425][["player_id", "team", "image_url", "market_value_eur", "top_strengths"]],
    on=["player_id", "team"],
    how="left",
)

print("\n=== 3. STYLE CLUSTER CENTROIDS (2024-25) ===")
cluster_labels = {}
for cid, grp in space.groupby("style_cluster"):
    top_pos = grp["position"].value_counts().index[0]
    top_sub = grp["sub_position"].value_counts().head(2).index.tolist()
    avg_x = grp["avg_touch_x"].mean()
    prog = grp["prog_passes_p90"].mean()
    kp = grp["key_passes_p90"].mean()
    shots = grp["shots_p90"].mean()
    tack = grp["tackles_int_p90"].mean()
    cross = grp["crosses_p90"].mean()
    clear = grp["clearances_p90"].mean()

    if top_pos == "Defender" and clear >= 3.2:
        archetype = "Ball-Playing & Aerial Centre-Back" if prog >= 4.0 else "Traditional Stopper Centre-Back"
    elif top_pos == "Defender":
        archetype = "Attacking Overlapping Full-Back" if cross >= 2.2 else "Two-Way Modern Full-Back"
    elif top_pos == "Midfield" and prog >= 5.5 and tack >= 2.5:
        archetype = "Deep-Lying Metronome & Anchor"
    elif top_pos == "Midfield":
        archetype = "Box-to-Box Engine & Presser" if tack >= 2.8 else "Creative Advanced Playmaker"
    elif top_pos == "Attack" and (cross >= 2.2 or kp >= 1.5):
        archetype = "Creative Wide Winger / Inside Forward"
    else:
        archetype = "Penalty-Box Poacher & Target Striker"

    # Ensure unique label if two clusters share a profile
    if archetype in cluster_labels.values():
        archetype = f"{archetype} (Type {cid + 1})"
    cluster_labels[int(cid)] = archetype
    ex_players = ", ".join(grp.sort_values("minutes_90", ascending=False)["player"].head(4).tolist())
    print(f"  Cluster {cid}: {archetype:38s} (n={len(grp):3d}, top={top_sub}) | e.g. {ex_players}")

space["archetype_name"] = space["style_cluster"].map(cluster_labels)
space.to_parquet(PROCESSED / "player_style_space_2425.parquet", index=False)

ps_full = ps_full.merge(
    space[["player_id", "team", "style_cluster", "archetype_name"]],
    on=["player_id", "team"],
    how="left",
)
ps_full["archetype_name"] = ps_full["archetype_name"].fillna(ps_full["sub_position"])
ps_full.to_parquet(PROCESSED / "player_ui_metadata.parquet", index=False)

print("\nSaved club_ui_metadata.parquet, player_ui_metadata.parquet, and updated player_style_space_2425.parquet.")