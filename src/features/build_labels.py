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


# 1. Aggregate Goals (excl. OwnGoal) & Assists across all 10 league-season files
goal_parts = []
assist_parts = []
og_total = 0

for pq in sorted(PROCESSED.glob("events_*.parquet")):
    ev = pd.read_parquet(
        pq,
        columns=["season", "team", "type", "player_id", "related_player_id", "q_Assisted", "qualifiers"],
    )
    g = ev[ev["type"] == "Goal"].copy()
    is_og = g["qualifiers"].astype(str).str.contains("OwnGoal", case=False, na=False)
    og_total += int(is_og.sum())
    g_valid = g[~is_og].dropna(subset=["player_id"]).copy()
    g_valid["player_id"] = g_valid["player_id"].astype(int)
    goal_parts.append(
        g_valid.groupby(["season", "team", "player_id"], as_index=False).size().rename(columns={"size": "goals"})
    )

    a_valid = g_valid[g_valid["q_Assisted"].fillna(False) & g_valid["related_player_id"].notna()].copy()
    a_valid["assister_id"] = a_valid["related_player_id"].astype(int)
    assist_parts.append(
        a_valid.groupby(["season", "team", "assister_id"], as_index=False)
        .size()
        .rename(columns={"assister_id": "player_id", "size": "assists"})
    )

goals_df = pd.concat(goal_parts, ignore_index=True).groupby(["season", "team", "player_id"], as_index=False)["goals"].sum()
assists_df = pd.concat(assist_parts, ignore_index=True).groupby(["season", "team", "player_id"], as_index=False)["assists"].sum()

print(f"=== 1. EVENT GOALS & ASSISTS SUMMARY ===")
print(f"Own goals excluded across all 10 files: {og_total}")
print(f"Total valid goals: {int(goals_df['goals'].sum())} | Total assists: {int(assists_df['assists'].sum())}")

# 2. Merge goals & assists onto player_season_minutes
psm = pd.read_parquet(PROCESSED / "player_season_minutes.parquet")
psm["player_id"] = psm["player_id"].astype(int)
psm = psm.merge(goals_df, on=["season", "team", "player_id"], how="left")
psm = psm.merge(assists_df, on=["season", "team", "player_id"], how="left")
psm["goals"] = psm["goals"].fillna(0).astype(int)
psm["assists"] = psm["assists"].fillna(0).astype(int)
psm["ga"] = psm["goals"] + psm["assists"]
psm["ga_p90"] = np.where(psm["minutes_90"] > 0, psm["ga"] / (psm["minutes_90"] / 90.0), np.nan)
psm.to_parquet(PROCESSED / "player_season_stats.parquet", index=False)

# 3. Map all WhoScored players to Transfermarkt players.csv.gz for position & DOB (needed for stayers)
tm_players = pd.read_csv(TM / "players.csv.gz", usecols=["player_id", "name", "first_name", "last_name", "date_of_birth", "position", "sub_position", "last_season"])
tm_players["date_of_birth"] = pd.to_datetime(tm_players["date_of_birth"], errors="coerce")
tm_players = tm_players[tm_players["last_season"] >= 2023].copy()
tm_players["norm_name"] = tm_players["name"].map(norm_text)

# Keep unique norm_name matches from TM players (plus matched cohort overrides)
cohort = pd.read_parquet(PROCESSED / "cohort_matched.parquet")
known_map = (
    cohort.dropna(subset=["ws_player_id"])
    .drop_duplicates("ws_player_id")[["ws_player_id", "position", "sub_position", "age_years"]]
    .copy()
)
known_map["ws_player_id"] = known_map["ws_player_id"].astype(int)
known_map["age_2025"] = known_map["age_years"].astype(float)

tm_unique = tm_players.drop_duplicates("norm_name", keep=False).copy()
tm_unique["age_2025"] = (pd.Timestamp("2025-07-01") - tm_unique["date_of_birth"]).dt.days / 365.25

ws_meta = psm[["player_id", "player"]].drop_duplicates("player_id").copy()
ws_meta["norm_name"] = ws_meta["player"].map(norm_text)
ws_meta = ws_meta.merge(tm_unique[["norm_name", "position", "sub_position", "age_2025"]], on="norm_name", how="left")
ws_meta = ws_meta.merge(known_map[["ws_player_id", "position", "sub_position", "age_2025"]], left_on="player_id", right_on="ws_player_id", how="left", suffixes=("", "_coh"))
for col in ["position", "sub_position", "age_2025"]:
    ws_meta[col] = ws_meta[f"{col}_coh"].combine_first(ws_meta[col])
ws_meta = ws_meta[["player_id", "position", "sub_position", "age_2025"]]

# 4. Identify Stayers (same club in 2425 & 2526, >=450 min90 both seasons, <=1 flagged match)
p24 = psm[(psm["season"] == 2425) & (psm["minutes_90"] >= 450) & (psm["team_flagged_matches"] <= 1)].copy()
p25 = psm[(psm["season"] == 2526) & (psm["minutes_90"] >= 450) & (psm["team_flagged_matches"] <= 1)].copy()

stayers = p24.merge(p25, on=["player_id", "team"], suffixes=("_2425", "_2526"))
stayers = stayers.merge(ws_meta, on="player_id", how="left")
outfield_stayers = stayers[stayers["position"].isin(["Attack", "Midfield", "Defender"]) & stayers["age_2025"].notna()].copy()

print(f"\n=== 2. STAYER CALIBRATION (>=450 min90 both seasons at same club) ===")
print(f"Total stayers: {len(stayers)} | Matched outfield stayers with position & age: {len(outfield_stayers)}")

calib = {}
for pos in ["Attack", "Midfield", "Defender"]:
    sub = outfield_stayers[outfield_stayers["position"] == pos].copy()
    # Minutes-weighted position mean in 2425
    mu_pos = float(np.average(sub["ga_p90_2425"], weights=sub["minutes_90_2425"]))
    mean_age = float(sub["age_2025"].mean())
    age_centered = sub["age_2025"].values - mean_age
    y_target = sub["ga_p90_2526"].values
    x_raw = sub["ga_p90_2425"].values
    m24 = sub["minutes_90_2425"].values

    # Grid search k in [0, 3000] + OLS age slope via leave-one-out CV MSE
    best_k, best_beta, best_mse = 900.0, 0.0, np.inf
    n = len(sub)
    for k in range(0, 3050, 50):
        w = m24 / (m24 + k) if k > 0 else np.ones_like(m24)
        shrunk = mu_pos + w * (x_raw - mu_pos)
        resid_no_age = y_target - shrunk
        # LOO-CV for simple slope beta on centered age
        loo_sq = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            denom = np.sum(age_centered[mask] ** 2)
            beta = np.sum(age_centered[mask] * resid_no_age[mask]) / denom if denom > 0 else 0.0
            pred_i = shrunk[i] + beta * age_centered[i]
            loo_sq.append((y_target[i] - pred_i) ** 2)
        mse = float(np.mean(loo_sq))
        if mse < best_mse:
            best_mse = mse
            best_k = float(k)
            denom_all = np.sum(age_centered ** 2)
            best_beta = float(np.sum(age_centered * resid_no_age) / denom_all) if denom_all > 0 else 0.0

    w_all = m24 / (m24 + best_k) if best_k > 0 else np.ones_like(m24)
    pred_all = mu_pos + w_all * (x_raw - mu_pos) + best_beta * age_centered
    sd_resid = float(np.std(y_target - pred_all, ddof=1))
    calib[pos] = {
        "n_stayers": n,
        "mu_pos": mu_pos,
        "mean_age": mean_age,
        "k": best_k,
        "beta_age": best_beta,
        "sd_resid": sd_resid,
    }
    print(
        f"  {pos:8s}: n={n:3d} | mu_ga90={mu_pos:.3f} | k={best_k:6.0f} | beta_age={best_beta:+.4f}/yr | sd_resid={sd_resid:.3f}"
    )

# 5. Exact destination-club matches played AFTER transfer_date (from match_qc.parquet)
mq = pd.read_parquet(PROCESSED / "match_qc.parquet")
pm = pd.read_parquet(PROCESSED / "player_match_minutes.parquet", columns=["game_id", "season", "team"]).drop_duplicates()
team_games = pm.merge(mq[["game_id", "game", "flags"]], on="game_id", how="left")
team_games["match_date"] = pd.to_datetime(team_games["game"].str.slice(0, 10), errors="coerce")
team_games_2526 = team_games[team_games["season"] == 2526].copy()

# 6. Compute labels for Summer-2025 Cohort
rows = []
for r in cohort.itertuples():
    elig_reason = "eligible"
    if r.flag_already_at_dest_2425:
        elig_reason = "excl_already_at_dest_2425"
    elif pd.isna(r.ws_player_id):
        elig_reason = "excl_unmatched_ws"

    pid = int(r.ws_player_id) if pd.notna(r.ws_player_id) else None
    dest_club = r.ws_to_club_expected

    # Baseline 2425 stats (across top-5 clubs in 2425)
    b_sub = psm[(psm["season"] == 2425) & (psm["player_id"] == pid)] if pid is not None else pd.DataFrame()
    min90_2425 = float(b_sub["minutes_90"].sum()) if len(b_sub) else 0.0
    ga_2425 = int(b_sub["ga"].sum()) if len(b_sub) else 0
    goals_2425 = int(b_sub["goals"].sum()) if len(b_sub) else 0
    assists_2425 = int(b_sub["assists"].sum()) if len(b_sub) else 0
    ga_p90_2425 = (ga_2425 / (min90_2425 / 90.0)) if min90_2425 > 0 else np.nan
    base_flagged = int(b_sub["team_flagged_matches"].max()) if len(b_sub) else 0

    if elig_reason == "eligible" and min90_2425 < 450:
        elig_reason = "excl_low_baseline_min(<450)"
    elif elig_reason == "eligible" and base_flagged > 1:
        elig_reason = "excl_flagged_club_2425"

    # Outcome 2526 stats at destination club
    o_sub = (
        psm[(psm["season"] == 2526) & (psm["player_id"] == pid) & (psm["team"] == dest_club)]
        if (pid is not None and pd.notna(dest_club))
        else pd.DataFrame()
    )
    min90_dest = float(o_sub["minutes_90"].sum()) if len(o_sub) else 0.0
    goals_dest = int(o_sub["goals"].sum()) if len(o_sub) else 0
    assists_dest = int(o_sub["assists"].sum()) if len(o_sub) else 0
    ga_dest = int(o_sub["ga"].sum()) if len(o_sub) else 0

    # Destination club league matches on/after transfer_date
    tg = team_games_2526[
        (team_games_2526["team"] == dest_club) & (team_games_2526["match_date"] >= pd.Timestamp(r.transfer_date))
    ]
    club_matches_after = int(len(tg))
    dest_flagged = int((tg["flags"].fillna("") != "").sum())
    if elig_reason == "eligible" and dest_flagged > 1:
        elig_reason = "excl_flagged_club_2526"

    ms = (min90_dest / (club_matches_after * 90.0)) if club_matches_after > 0 else np.nan
    actual_ga_p90 = (ga_dest / (min90_dest / 90.0)) if min90_dest > 0 else np.nan

    # Expected G+A/90 & residual Y for outfield players with >=450 min90 at destination
    exp_ga_p90 = np.nan
    resid_R = np.nan
    Y_score = np.nan
    if r.position in calib and elig_reason == "eligible":
        c = calib[r.position]
        w = min90_2425 / (min90_2425 + c["k"]) if c["k"] > 0 else 1.0
        exp_ga_p90 = c["mu_pos"] + w * (ga_p90_2425 - c["mu_pos"]) + c["beta_age"] * (float(r.age_years) - c["mean_age"])
        if min90_dest >= 450:
            resid_R = actual_ga_p90 - exp_ga_p90
            Y_score = resid_R / c["sd_resid"]

    # Binary labels at 3 threshold sets
    def label_outfield(ms_val, y_val, ms_hi, y_hi, ms_lo, y_lo):
        if pd.isna(ms_val):
            return None
        if (ms_val < ms_lo) or (pd.notna(y_val) and y_val <= y_lo):
            return "Failure"
        if pd.notna(y_val) and (ms_val >= ms_hi) and (y_val >= y_hi):
            return "Success"
        return "Ambiguous"

    def label_gk(ms_val, ms_hi, ms_lo):
        if pd.isna(ms_val):
            return None
        if ms_val < ms_lo:
            return "Failure"
        if ms_val >= ms_hi:
            return "Success"
        return "Ambiguous"

    if elig_reason != "eligible":
        lbl_lenient = lbl_base = lbl_strict = None
    elif r.position == "Goalkeeper":
        lbl_lenient = label_gk(ms, 0.40, 0.15)
        lbl_base = label_gk(ms, 0.50, 0.25)
        lbl_strict = label_gk(ms, 0.60, 0.35)
    else:
        lbl_lenient = label_outfield(ms, Y_score, 0.40, -0.25, 0.15, -1.00)
        lbl_base = label_outfield(ms, Y_score, 0.50, 0.00, 0.25, -0.50)
        lbl_strict = label_outfield(ms, Y_score, 0.60, 0.25, 0.35, -0.25)

    rows.append({
        "tm_player_id": r.tm_player_id,
        "ws_player_id": r.ws_player_id,
        "player_name": r.tm_player_name,
        "age_years": r.age_years,
        "position": r.position,
        "sub_position": r.sub_position,
        "tm_from_club": r.tm_from_club,
        "tm_to_club": r.tm_to_club,
        "ws_to_club": dest_club,
        "transfer_date": r.transfer_date,
        "transfer_fee": r.transfer_fee,
        "elig_status": elig_reason,
        "min90_2425": round(min90_2425, 1),
        "goals_2425": goals_2425,
        "assists_2425": assists_2425,
        "ga_p90_2425": round(ga_p90_2425, 3) if pd.notna(ga_p90_2425) else np.nan,
        "club_matches_after": club_matches_after,
        "min90_dest_2526": round(min90_dest, 1),
        "goals_dest_2526": goals_dest,
        "assists_dest_2526": assists_dest,
        "MS": round(ms, 3) if pd.notna(ms) else np.nan,
        "actual_ga_p90": round(actual_ga_p90, 3) if pd.notna(actual_ga_p90) else np.nan,
        "expected_ga_p90": round(exp_ga_p90, 3) if pd.notna(exp_ga_p90) else np.nan,
        "residual_R": round(resid_R, 3) if pd.notna(resid_R) else np.nan,
        "Y_score": round(Y_score, 3) if pd.notna(Y_score) else np.nan,
        "label_lenient": lbl_lenient,
        "label_base": lbl_base,
        "label_strict": lbl_strict,
    })

labels_df = pd.DataFrame(rows)
labels_df.to_parquet(PROCESSED / "transfer_labels.parquet", index=False)
labels_df.to_csv(PROCESSED / "transfer_labels.csv", index=False)

print("\n=== 3. COHORT ELIGIBILITY BREAKDOWN ===")
print(labels_df["elig_status"].value_counts().to_string())

elig = labels_df[labels_df["elig_status"] == "eligible"].copy()
elig_out = elig[elig["position"] != "Goalkeeper"].copy()
elig_gk = elig[elig["position"] == "Goalkeeper"].copy()

print(f"\n=== 4. USABLE COHORT & SMALL-SAMPLE CHECK ===")
print(f"Eligible transfers total:         {len(elig)}")
print(f"  Eligible OUTFIELD total:        {len(elig_out)}")
print(f"    with Y_score (>=450 dest min): {elig_out['Y_score'].notna().sum()}")
print(f"    with <450 dest min (MS-only):  {elig_out['Y_score'].isna().sum()}")
print(f"  Eligible GOALKEEPERS total:     {len(elig_gk)}")
print("\nEligible outfield by position:")
print(elig_out["position"].value_counts().to_string())

for col in ["label_lenient", "label_base", "label_strict"]:
    print(f"\n--- Outfield {col} ---")
    print(elig_out[col].value_counts(dropna=False).to_string())

print("\n=== 5. TOP 15 HIGH-PROFILE ELIGIBLE OUTFIELD TRANSFERS (SANITY CHECK) ===")
cols = ["player_name", "position", "tm_from_club", "tm_to_club", "transfer_fee", "min90_2425", "min90_dest_2526", "MS", "actual_ga_p90", "expected_ga_p90", "Y_score", "label_base"]
print(elig_out.sort_values("transfer_fee", ascending=False)[cols].head(15).to_string(index=False))