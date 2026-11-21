from pathlib import Path
import re
import sys
import unicodedata
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROCESSED = Path("data/processed")
TM = Path("data/raw/transfermarkt")


def norm_text(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(s.split())


def safe_div(num, den, fill=0.0):
    num_a = np.asarray(num, dtype=float)
    den_a = np.asarray(den, dtype=float)
    out = np.full_like(num_a, fill, dtype=float)
    mask = den_a > 0
    out[mask] = num_a[mask] / den_a[mask]
    return out


USE_COLS = [
    "league", "season", "game_id", "type", "outcome_type",
    "team_id", "team", "player_id", "player",
    "x", "y", "end_x", "end_y",
    "is_touch", "is_shot", "pass_length",
    "q_Longball", "q_Cross", "q_Throughball", "q_ShotAssist",
    "q_CornerTaken", "q_FreekickTaken", "q_ThrowIn", "q_GoalKick", "q_Head",
]

match_club_parts = []
club_coord_parts = []
player_agg_parts = []
player_coord_parts = []

for pq in sorted(PROCESSED.glob("events_*.parquet")):
    df = pd.read_parquet(pq, columns=USE_COLS)

    # Boolean masks
    is_pass = df["type"] == "Pass"
    is_succ_pass = is_pass & (df["outcome_type"] == "Successful")
    is_set_piece = (
        df["q_CornerTaken"].fillna(False)
        | df["q_FreekickTaken"].fillna(False)
        | df["q_ThrowIn"].fillna(False)
        | df["q_GoalKick"].fillna(False)
    )
    is_open_pass = is_pass & (~is_set_piece)
    is_succ_open_pass = is_succ_pass & (~is_set_piece)

    # Progressive pass: open-play successful pass moving >=25% closer to goal (105, 34)
    d_start = np.hypot(105.0 - df["x"], 34.0 - df["y"])
    d_end = np.hypot(105.0 - df["end_x"], 34.0 - df["end_y"])
    is_prog_pass = is_succ_open_pass & ((d_start - d_end) >= 0.25 * d_start) & (d_start > 5.0)
    is_vert_pass = is_succ_open_pass & ((df["end_x"] - df["x"]) >= 10.0)

    is_touch = df["is_touch"].fillna(False)
    is_open_touch = is_touch & (~is_set_piece)
    is_box_touch = is_touch & (df["x"] >= 88.5) & (df["y"].between(13.84, 54.16))
    is_wide_touch = is_open_touch & ((df["y"] < 13.6) | (df["y"] > 54.4))

    # Defensive & pressing actions
    is_tackle = df["type"] == "Tackle"
    is_interception = (df["type"] == "Interception") & (df["outcome_type"] == "Successful")
    is_challenge = df["type"] == "Challenge"
    is_foul_comm = (df["type"] == "Foul") & (df["outcome_type"] == "Unsuccessful")
    is_recovery = (df["type"] == "BallRecovery") & (df["outcome_type"] == "Successful")
    is_clearance = (df["type"] == "Clearance") & (df["outcome_type"] == "Successful")

    is_ppda_def = (is_tackle | is_interception | is_challenge | is_foul_comm) & (df["x"] >= 42.0)
    is_high_rec = (is_tackle | is_interception | is_recovery) & (df["x"] >= 63.0)
    is_def_line_ev = is_tackle | is_interception | is_challenge | is_clearance | is_recovery

    # --- A. Match-Club level aggregations (for Possession, PPDA, Field Tilt) ---
    m_df = pd.DataFrame({
        "league": df["league"],
        "season": df["season"],
        "game_id": df["game_id"],
        "team_id": df["team_id"],
        "team": df["team"],
        "passes": is_pass.astype(int),
        "succ_passes": is_succ_pass.astype(int),
        "open_passes": is_open_pass.astype(int),
        "long_balls": (is_pass & df["q_Longball"].fillna(False)).astype(int),
        "prog_passes": is_prog_pass.astype(int),
        "vert_passes": is_vert_pass.astype(int),
        "pass_len_sum": np.where(is_pass, df["pass_length"].fillna(0.0), 0.0),
        "f3_passes": (is_pass & (df["x"] >= 70.0)).astype(int),
        "own60_passes": (is_pass & (df["x"] <= 63.0)).astype(int),
        "ppda_def_act": is_ppda_def.astype(int),
        "high_rec": is_high_rec.astype(int),
        "box_touches": is_box_touch.astype(int),
        "open_touches": is_open_touch.astype(int),
        "wide_touches": is_wide_touch.astype(int),
        "crosses": (is_pass & df["q_Cross"].fillna(False)).astype(int),
        "throughballs": (is_pass & df["q_Throughball"].fillna(False)).astype(int),
        "shots": df["is_shot"].fillna(False).astype(int),
        "takeons": (df["type"] == "TakeOn").astype(int),
        "aerials": (df["type"] == "Aerial").astype(int),
        "tackles_int": (is_tackle | is_interception).astype(int),
        "clearances": is_clearance.astype(int),
    })
    mc = m_df.groupby(["league", "season", "game_id", "team_id", "team"], as_index=False).sum()

    # Attach opponent match totals (2 teams per game_id)
    game_tot = mc.groupby("game_id", as_index=False)[["passes", "f3_passes", "own60_passes"]].sum().rename(
        columns={"passes": "game_passes", "f3_passes": "game_f3_passes", "own60_passes": "game_own60_passes"}
    )
    mc = mc.merge(game_tot, on="game_id", how="left")
    mc["opp_passes"] = mc["game_passes"] - mc["passes"]
    mc["opp_f3_passes"] = mc["game_f3_passes"] - mc["f3_passes"]
    mc["opp_own60_passes"] = mc["game_own60_passes"] - mc["own60_passes"]
    mc["match_poss"] = safe_div(mc["passes"], mc["game_passes"], fill=0.5)
    mc["match_field_tilt"] = safe_div(mc["f3_passes"], mc["game_f3_passes"], fill=0.5)
    match_club_parts.append(mc)

    # Coordinate distributions per club-season (touch width SD & defensive line median X)
    t_sub = df.loc[is_open_touch, ["league", "season", "team_id", "y"]].copy()
    d_sub = df.loc[is_def_line_ev, ["league", "season", "team_id", "x"]].copy()
    c_width = t_sub.groupby(["league", "season", "team_id"], as_index=False)["y"].std().rename(columns={"y": "touch_width_sd"})
    c_defx = d_sub.groupby(["league", "season", "team_id"], as_index=False)["x"].median().rename(columns={"x": "def_line_height_m"})
    club_coord_parts.append(c_width.merge(c_defx, on=["league", "season", "team_id"], how="outer"))

    # --- B. Player-Season aggregations ---
    valid_p = df["player_id"].notna()
    df_p = df.loc[valid_p].copy()
    df_p["player_id"] = df_p["player_id"].astype(int)

    is_pass_p = is_pass.loc[valid_p]
    is_succ_pass_p = is_succ_pass.loc[valid_p]
    is_open_pass_p = is_open_pass.loc[valid_p]
    is_prog_pass_p = is_prog_pass.loc[valid_p]
    is_touch_p = is_touch.loc[valid_p]
    is_box_touch_p = is_box_touch.loc[valid_p]
    is_shot_p = df_p["is_shot"].fillna(False)
    is_tackle_p = is_tackle.loc[valid_p]
    is_interception_p = is_interception.loc[valid_p]
    is_challenge_p = is_challenge.loc[valid_p]
    is_foul_comm_p = is_foul_comm.loc[valid_p]
    is_recovery_p = is_recovery.loc[valid_p]
    is_clearance_p = is_clearance.loc[valid_p]
    is_def_ev_p = is_def_line_ev.loc[valid_p]

    shot_dist = np.hypot(105.0 - df_p["x"], 34.0 - df_p["y"])

    p_df = pd.DataFrame({
        "league": df_p["league"],
        "season": df_p["season"],
        "team_id": df_p["team_id"],
        "player_id": df_p["player_id"],
        "touches": is_touch_p.astype(int),
        "touch_x_sum": np.where(is_touch_p, df_p["x"].fillna(0.0), 0.0),
        "touch_y_cent_sum": np.where(is_touch_p, np.abs(df_p["y"].fillna(34.0) - 34.0), 0.0),
        "att_third_touches": (is_touch_p & (df_p["x"] >= 70.0)).astype(int),
        "box_touches": is_box_touch_p.astype(int),
        "passes_att": is_pass_p.astype(int),
        "passes_succ": is_succ_pass_p.astype(int),
        "pass_len_sum": np.where(is_pass_p, df_p["pass_length"].fillna(0.0), 0.0),
        "long_balls": (is_pass_p & df_p["q_Longball"].fillna(False)).astype(int),
        "prog_passes": is_prog_pass_p.astype(int),
        "f3_passes_succ": (is_succ_pass_p & (df_p["end_x"] >= 70.0)).astype(int),
        "fwd_passes": (is_pass_p & ((df_p["end_x"] - df_p["x"]) > 2.0)).astype(int),
        "key_passes": (is_pass_p & df_p["q_ShotAssist"].fillna(False)).astype(int),
        "throughballs": (is_pass_p & df_p["q_Throughball"].fillna(False)).astype(int),
        "crosses": (is_pass_p & df_p["q_Cross"].fillna(False)).astype(int),
        "open_crosses": (is_open_pass_p & df_p["q_Cross"].fillna(False)).astype(int),
        "takeons_att": (df_p["type"] == "TakeOn").astype(int),
        "takeons_succ": ((df_p["type"] == "TakeOn") & (df_p["outcome_type"] == "Successful")).astype(int),
        "dispossessed": ((df_p["type"] == "Dispossessed") | ((df_p["type"] == "BallTouch") & (df_p["outcome_type"] == "Unsuccessful"))).astype(int),
        "fouls_won": ((df_p["type"] == "Foul") & (df_p["outcome_type"] == "Successful")).astype(int),
        "shots": is_shot_p.astype(int),
        "shot_dist_sum": np.where(is_shot_p, shot_dist.fillna(0.0), 0.0),
        "headed_shots": (is_shot_p & df_p["q_Head"].fillna(False)).astype(int),
        "tackles_int": (is_tackle_p | is_interception_p).astype(int),
        "ball_recoveries": is_recovery_p.astype(int),
        "high_press_act": ((is_tackle_p | is_interception_p | is_challenge_p | is_recovery_p | is_foul_comm_p) & (df_p["x"] >= 52.5)).astype(int),
        "def_ev_count": is_def_ev_p.astype(int),
        "def_ev_x_sum": np.where(is_def_ev_p, df_p["x"].fillna(0.0), 0.0),
        "clearances": is_clearance_p.astype(int),
        "aerials_att": (df_p["type"] == "Aerial").astype(int),
        "aerials_won": ((df_p["type"] == "Aerial") & (df_p["outcome_type"] == "Successful")).astype(int),
        "gk_sweeper": (df_p["type"] == "KeeperSweeper").astype(int),
        "gk_claims": ((df_p["type"] == "Claim") & (df_p["outcome_type"] == "Successful")).astype(int),
        "gk_saves": (df_p["type"] == "Save").astype(int),
    })
    player_agg_parts.append(p_df.groupby(["league", "season", "team_id", "player_id"], as_index=False).sum())
    print(f"Processed features from {pq.name}")

# --- Combine Club Style Features ---
mc_all = pd.concat(match_club_parts, ignore_index=True)
cc_all = pd.concat(club_coord_parts, ignore_index=True).groupby(["league", "season", "team_id"], as_index=False).mean()

club_season = (
    mc_all.groupby(["league", "season", "team_id", "team"], as_index=False)
    .agg(
        matches=("game_id", "nunique"),
        poss_share=("match_poss", "mean"),
        field_tilt=("match_field_tilt", "mean"),
        passes=("passes", "sum"),
        succ_passes=("succ_passes", "sum"),
        open_passes=("open_passes", "sum"),
        long_balls=("long_balls", "sum"),
        prog_passes=("prog_passes", "sum"),
        vert_passes=("vert_passes", "sum"),
        pass_len_sum=("pass_len_sum", "sum"),
        f3_passes=("f3_passes", "sum"),
        opp_own60_passes=("opp_own60_passes", "sum"),
        ppda_def_act=("ppda_def_act", "sum"),
        high_rec=("high_rec", "sum"),
        box_touches=("box_touches", "sum"),
        open_touches=("open_touches", "sum"),
        wide_touches=("wide_touches", "sum"),
        crosses=("crosses", "sum"),
        throughballs=("throughballs", "sum"),
        shots=("shots", "sum"),
        takeons=("takeons", "sum"),
        aerials=("aerials", "sum"),
        tackles_int=("tackles_int", "sum"),
        clearances=("clearances", "sum"),
    )
    .merge(cc_all, on=["league", "season", "team_id"], how="left")
)

m = club_season["matches"]
club_features = pd.DataFrame({
    "league": club_season["league"],
    "season": club_season["season"],
    "team_id": club_season["team_id"],
    "team": club_season["team"],
    "matches": m,
    "poss_share": club_season["poss_share"].round(4),
    "passes_p90": safe_div(club_season["passes"], m).round(2),
    "pass_acc": safe_div(club_season["succ_passes"], club_season["passes"]).round(4),
    "ppda": safe_div(club_season["opp_own60_passes"], club_season["ppda_def_act"], fill=np.nan).round(2),
    "high_turnovers_p90": safe_div(club_season["high_rec"], m).round(2),
    "field_tilt": club_season["field_tilt"].round(4),
    "f3_pass_share": safe_div(club_season["f3_passes"], club_season["passes"]).round(4),
    "box_touches_p90": safe_div(club_season["box_touches"], m).round(2),
    "long_ball_share": safe_div(club_season["long_balls"], club_season["passes"]).round(4),
    "prog_pass_share": safe_div(club_season["prog_passes"], club_season["open_passes"]).round(4),
    "vert_pass_share": safe_div(club_season["vert_passes"], club_season["open_passes"]).round(4),
    "avg_pass_length": safe_div(club_season["pass_len_sum"], club_season["passes"]).round(2),
    "tempo_passes_per_poss_min": safe_div(safe_div(club_season["passes"], m * 90.0), club_season["poss_share"]).round(2),
    "touch_width_sd": club_season["touch_width_sd"].round(2),
    "wide_touch_share": safe_div(club_season["wide_touches"], club_season["open_touches"]).round(4),
    "crosses_p90": safe_div(club_season["crosses"], m).round(2),
    "def_line_height_m": club_season["def_line_height_m"].round(2),
    "clearance_share": safe_div(club_season["clearances"], club_season["tackles_int"] + club_season["clearances"]).round(4),
    "shots_p90": safe_div(club_season["shots"], m).round(2),
    "throughballs_p90": safe_div(club_season["throughballs"], m).round(2),
    "takeons_p90": safe_div(club_season["takeons"], m).round(2),
    "aerials_p90": safe_div(club_season["aerials"], m).round(2),
})

club_features.to_parquet(PROCESSED / "club_style_features.parquet", index=False)
club_features.to_csv(PROCESSED / "club_style_features.csv", index=False)

# --- Combine Player Style Features ---
p_raw = pd.concat(player_agg_parts, ignore_index=True).groupby(["league", "season", "team_id", "player_id"], as_index=False).sum()
pss = pd.read_parquet(PROCESSED / "player_season_stats.parquet")
pss["player_id"] = pss["player_id"].astype(int)

# Attach TM position/sub_position/age metadata
tm_players = pd.read_csv(TM / "players.csv.gz", usecols=["player_id", "name", "date_of_birth", "position", "sub_position", "last_season"])
tm_players["date_of_birth"] = pd.to_datetime(tm_players["date_of_birth"], errors="coerce")
tm_players = tm_players[tm_players["last_season"] >= 2023].copy()
tm_players["norm_name"] = tm_players["name"].map(norm_text)
tm_unique = tm_players.drop_duplicates("norm_name", keep=False).copy()
tm_unique["age_2025"] = ((pd.Timestamp("2025-07-01") - tm_unique["date_of_birth"]).dt.days / 365.25).round(1)

cohort = pd.read_parquet(PROCESSED / "cohort_matched.parquet")
known_map = cohort.dropna(subset=["ws_player_id"]).drop_duplicates("ws_player_id")[["ws_player_id", "position", "sub_position", "age_years"]].copy()
known_map["ws_player_id"] = known_map["ws_player_id"].astype(int)
known_map["age_2025"] = known_map["age_years"].astype(float)

p_full = pss.merge(p_raw, on=["league", "season", "team_id", "player_id"], how="left")
p_full["norm_name"] = p_full["player"].map(norm_text)
p_full = p_full.merge(tm_unique[["norm_name", "position", "sub_position", "age_2025"]], on="norm_name", how="left")
p_full = p_full.merge(known_map[["ws_player_id", "position", "sub_position", "age_2025"]], left_on="player_id", right_on="ws_player_id", how="left", suffixes=("", "_coh"))
for col in ["position", "sub_position", "age_2025"]:
    p_full[col] = p_full[f"{col}_coh"].combine_first(p_full[col])

p90 = p_full["minutes_90"] / 90.0

player_style = pd.DataFrame({
    "league": p_full["league"],
    "season": p_full["season"],
    "team_id": p_full["team_id"],
    "team": p_full["team"],
    "player_id": p_full["player_id"],
    "player": p_full["player"],
    "position": p_full["position"],
    "sub_position": p_full["sub_position"],
    "age_2025": p_full["age_2025"],
    "minutes": p_full["minutes"],
    "minutes_90": p_full["minutes_90"],
    "apps": p_full["apps"],
    "starts": p_full["starts"],
    "minutes_share_of_season": p_full["minutes_share_of_season"],
    "goals": p_full["goals"],
    "assists": p_full["assists"],
    "ga_p90": p_full["ga_p90"],
    # Spatial / Role Zones
    "touches_p90": safe_div(p_full["touches"], p90).round(2),
    "avg_touch_x": safe_div(p_full["touch_x_sum"], p_full["touches"], fill=np.nan).round(2),
    "avg_touch_y_centrality": safe_div(p_full["touch_y_cent_sum"], p_full["touches"], fill=np.nan).round(2),
    "att_third_touch_share": safe_div(p_full["att_third_touches"], p_full["touches"]).round(4),
    "box_touches_p90": safe_div(p_full["box_touches"], p90).round(2),
    # Passing Volume, Direction & Progression
    "passes_att_p90": safe_div(p_full["passes_att"], p90).round(2),
    "pass_acc": safe_div(p_full["passes_succ"], p_full["passes_att"]).round(4),
    "avg_pass_len": safe_div(p_full["pass_len_sum"], p_full["passes_att"], fill=np.nan).round(2),
    "long_ball_share": safe_div(p_full["long_balls"], p_full["passes_att"]).round(4),
    "prog_passes_p90": safe_div(p_full["prog_passes"], p90).round(2),
    "f3_passes_succ_p90": safe_div(p_full["f3_passes_succ"], p90).round(2),
    "fwd_pass_share": safe_div(p_full["fwd_passes"], p_full["passes_att"]).round(4),
    # Chance Creation & Wide Play
    "key_passes_p90": safe_div(p_full["key_passes"], p90).round(2),
    "throughballs_p90": safe_div(p_full["throughballs"], p90).round(2),
    "crosses_p90": safe_div(p_full["crosses"], p90).round(2),
    "open_cross_share": safe_div(p_full["open_crosses"], p_full["passes_att"]).round(4),
    # Dribbling, Carrying & Retention
    "takeons_att_p90": safe_div(p_full["takeons_att"], p90).round(2),
    "takeon_succ_rate": safe_div(p_full["takeons_succ"], p_full["takeons_att"]).round(4),
    "dispossessed_p90": safe_div(p_full["dispossessed"], p90).round(2),
    "fouls_won_p90": safe_div(p_full["fouls_won"], p90).round(2),
    # Shooting Profile
    "shots_p90": safe_div(p_full["shots"], p90).round(2),
    "avg_shot_dist_m": safe_div(p_full["shot_dist_sum"], p_full["shots"], fill=np.nan).round(2),
    "headed_shot_share": safe_div(p_full["headed_shots"], p_full["shots"]).round(4),
    # Defending, Pressing & Aerials
    "tackles_int_p90": safe_div(p_full["tackles_int"], p90).round(2),
    "ball_recoveries_p90": safe_div(p_full["ball_recoveries"], p90).round(2),
    "high_press_actions_p90": safe_div(p_full["high_press_act"], p90).round(2),
    "avg_def_action_x": safe_div(p_full["def_ev_x_sum"], p_full["def_ev_count"], fill=np.nan).round(2),
    "clearances_p90": safe_div(p_full["clearances"], p90).round(2),
    "aerials_att_p90": safe_div(p_full["aerials_att"], p90).round(2),
    "aerial_win_rate": safe_div(p_full["aerials_won"], p_full["aerials_att"]).round(4),
    # Goalkeeper Profile
    "gk_sweeper_p90": safe_div(p_full["gk_sweeper"], p90).round(2),
    "gk_claims_p90": safe_div(p_full["gk_claims"], p90).round(2),
    "gk_saves_p90": safe_div(p_full["gk_saves"], p90).round(2),
})

player_style.to_parquet(PROCESSED / "player_style_features.parquet", index=False)
player_style.to_csv(PROCESSED / "player_style_features.csv", index=False)

# --- Assemble Master Transfer Dataset (STRICT 2024-25 FEATURES ONLY) ---
labels = pd.read_parquet(PROCESSED / "transfer_labels.parquet")
cmap = pd.read_csv(PROCESSED / "club_name_map.csv")
tm_to_ws_club = dict(zip(cmap["tm_club"], cmap["ws_team"]))

p_feat_cols = [c for c in player_style.columns if c not in {
    "league", "season", "team_id", "team", "player_id", "player",
    "position", "sub_position", "age_2025", "minutes", "minutes_90",
    "apps", "starts", "minutes_share_of_season", "goals", "assists", "ga_p90"
}]

# Minutes-weighted 2024-25 player style across all clubs in 2425
ps24 = player_style[(player_style["season"] == 2425) & (player_style["minutes_90"] > 0)].copy()
p24_rows = []
for pid, grp in ps24.groupby("player_id"):
    w = grp["minutes_90"].values
    row_d = {"ws_player_id": int(pid), "p24_clubs": ",".join(sorted(grp["team"].unique()))}
    for col in p_feat_cols:
        vals = grp[col].values.astype(float)
        valid = ~np.isnan(vals)
        row_d[f"p24_{col}"] = round(float(np.average(vals[valid], weights=w[valid])), 4) if valid.any() else np.nan
    p24_rows.append(row_d)
p24_agg = pd.DataFrame(p24_rows)

c_feat_cols = [c for c in club_features.columns if c not in {"league", "season", "team_id", "team", "matches"}]
c24 = club_features[club_features["season"] == 2425].copy()
dest_c24 = c24[["league", "team"] + c_feat_cols].rename(
    columns={"league": "dest_league_2425", "team": "ws_to_club", **{c: f"dest_c24_{c}" for c in c_feat_cols}}
)
orig_c24 = c24[["league", "team"] + c_feat_cols].rename(
    columns={"league": "orig_league_2425", "team": "ws_from_club", **{c: f"orig_c24_{c}" for c in c_feat_cols}}
)

master = labels.copy()
master["ws_from_club"] = master["tm_from_club"].map(tm_to_ws_club)
master = master.merge(p24_agg, on="ws_player_id", how="left")
master = master.merge(dest_c24, on="ws_to_club", how="left")
master = master.merge(orig_c24, on="ws_from_club", how="left")

# Impute missing avg_shot_dist_m for 0-shot players with position median in 2425
if "p24_avg_shot_dist_m" in master.columns:
    pos_med = master.groupby("position")["p24_avg_shot_dist_m"].transform("median")
    master["p24_avg_shot_dist_m"] = master["p24_avg_shot_dist_m"].fillna(pos_med).fillna(16.0)

master.to_parquet(PROCESSED / "master_transfer_dataset.parquet", index=False)
master.to_csv(PROCESSED / "master_transfer_dataset.csv", index=False)

print("\n=== 1. MASTER DATABASE SUMMARY ===")
print(f"club_style_features.parquet:     {club_features.shape} (seasons: {club_features['season'].value_counts().to_dict()})")
print(f"player_style_features.parquet:   {player_style.shape}")
print(f"master_transfer_dataset.parquet: {master.shape}")
leaked_cols = [c for c in master.columns if ("2526" in c and ("p25" in c or "c25" in c))]
print(f"Leaked 2526 style columns check: {len(leaked_cols)} (must be 0)")

elig_out = master[(master["elig_status"] == "eligible") & (master["position"] != "Goalkeeper")]
p24_cols = [c for c in master.columns if c.startswith("p24_") and c != "p24_clubs"]
dc24_cols = [c for c in master.columns if c.startswith("dest_c24_")]
print(f"Eligible outfield rows: {len(elig_out)}")
print(f"  Missing values in p24_* style features:       {int(elig_out[p24_cols].isna().sum().sum())}")
print(f"  Missing values in dest_c24_* style features:  {int(elig_out[dc24_cols].isna().sum().sum())}")

print("\n=== 2. CLUB STYLE SANITY CHECK (2024-25 PREMIER LEAGUE) ===")
pl24 = club_features[(club_features["league"] == "ENG-Premier League") & (club_features["season"] == 2425)].sort_values("poss_share", ascending=False)
show_c = ["team", "poss_share", "ppda", "field_tilt", "long_ball_share", "prog_pass_share", "def_line_height_m", "tempo_passes_per_poss_min"]
print(pl24[show_c].to_string(index=False))