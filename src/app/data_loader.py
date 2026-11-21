"""Data loaders, fast FitEngine, human-readable attribute catalogs, and cached rankers."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st
from theme import format_eur, short_league

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

ON_BALL = [
    "touches_p90", "box_touches_p90", "passes_att_p90", "prog_passes_p90",
    "f3_passes_succ_p90", "key_passes_p90", "throughballs_p90", "crosses_p90",
    "takeons_att_p90", "dispossessed_p90", "fouls_won_p90", "shots_p90",
]
DEFENSIVE = [
    "tackles_int_p90", "ball_recoveries_p90", "high_press_actions_p90",
    "clearances_p90", "aerials_att_p90",
]

PILLAR_LABELS = [
    ("Ball Progression", "pillar_progression", "Moving the ball forward into attacking zones"),
    ("Chance Creation", "pillar_creativity", "Key passes, throughballs & final-third delivery"),
    ("Ball Security", "pillar_ball_security", "Pass completion & retaining possession under pressure"),
    ("Dribbling & Width", "pillar_wide_dribbling", "1v1 take-ons and wide crossing threat"),
    ("Goal Threat", "pillar_goal_threat", "Shots, penalty-box touches & goal/assist output"),
    ("Pressing & Defending", "pillar_defending", "High pressing, tackles, interceptions & recoveries"),
]

CLUSTER_ARCHETYPE_MAP = {
    0: "Traditional Stopper Centre-Back",
    1: "Box-to-Box Engine & Presser",
    2: "Creative Advanced Playmaker",
    3: "Creative Wide Winger / Inside Forward",
    4: "Attacking Overlapping Full-Back",
    5: "Penalty-Box Poacher & Target Striker",
    6: "Deep-Lying Metronome & Anchor",
    7: "Two-Way Modern Full-Back",
}

# Clean human-readable scouting dictionaries (eliminates raw snake_case columns in tables)
PLAYER_ATTRIBUTE_CATALOG: list[tuple[str, str, str, str]] = [
    # (column_name, Category, Human Label, format_type)
    ("apps", "Playing Time & Output", "Appearances", "int"),
    ("starts", "Playing Time & Output", "Matches Started", "int"),
    ("minutes_90", "Playing Time & Output", "Total Minutes Played", "int"),
    ("minutes_share_of_season", "Playing Time & Output", "Share of Team Minutes", "pct"),
    ("goals", "Playing Time & Output", "Goals Scored", "int"),
    ("assists", "Playing Time & Output", "Assists", "int"),
    ("ga_p90", "Playing Time & Output", "Goals + Assists per 90", "f2"),
    ("shots_p90", "Attacking & Box Threat", "Shots per 90", "f2"),
    ("avg_shot_dist_m", "Attacking & Box Threat", "Average Shot Distance", "meters"),
    ("headed_shot_share", "Attacking & Box Threat", "Headed Shot Share", "pct"),
    ("box_touches_p90", "Attacking & Box Threat", "Penalty-Box Touches per 90", "f2"),
    ("att_third_touch_share", "Attacking & Box Threat", "Attacking-Third Touch Share", "pct"),
    ("avg_touch_x", "Pitch Positioning", "Average Touch Height (0–105m pitch)", "meters"),
    ("avg_touch_y_centrality", "Pitch Positioning", "Centrality Index (1 = central, 0 = touchline)", "f2"),
    ("touches_p90", "Passing & Progression", "Total Touches per 90", "f1"),
    ("passes_att_p90", "Passing & Progression", "Passes Attempted per 90", "f1"),
    ("pass_acc", "Passing & Progression", "Pass Completion Accuracy", "pct"),
    ("avg_pass_len", "Passing & Progression", "Average Pass Length", "meters"),
    ("prog_passes_p90", "Passing & Progression", "Progressive Passes per 90", "f2"),
    ("f3_passes_succ_p90", "Passing & Progression", "Completed Final-Third Passes per 90", "f2"),
    ("fwd_pass_share", "Passing & Progression", "Forward Pass Share", "pct"),
    ("long_ball_share", "Passing & Progression", "Long-Ball Share (>32m)", "pct"),
    ("key_passes_p90", "Creativity & Wide Play", "Key Passes (Shot Assists) per 90", "f2"),
    ("throughballs_p90", "Creativity & Wide Play", "Throughballs per 90", "f2"),
    ("crosses_p90", "Creativity & Wide Play", "Crosses per 90", "f2"),
    ("open_cross_share", "Creativity & Wide Play", "Open-Play Cross Share", "pct"),
    ("takeons_att_p90", "Dribbling & Ball Retention", "1v1 Dribbles Attempted per 90", "f2"),
    ("takeon_succ_rate", "Dribbling & Ball Retention", "1v1 Dribble Success Rate", "pct"),
    ("dispossessed_p90", "Dribbling & Ball Retention", "Times Dispossessed per 90", "f2"),
    ("fouls_won_p90", "Dribbling & Ball Retention", "Fouls Won per 90", "f2"),
    ("tackles_int_p90", "Defending & Pressing", "Tackles + Interceptions per 90", "f2"),
    ("ball_recoveries_p90", "Defending & Pressing", "Loose-Ball Recoveries per 90", "f2"),
    ("high_press_actions_p90", "Defending & Pressing", "High-Press Defensive Actions per 90", "f2"),
    ("avg_def_action_x", "Defending & Pressing", "Average Defensive Action Height", "meters"),
    ("clearances_p90", "Defending & Pressing", "Clearances per 90", "f2"),
    ("aerials_att_p90", "Aerial Duels", "Aerial Duels Contested per 90", "f2"),
    ("aerial_win_rate", "Aerial Duels", "Aerial Duel Win Rate", "pct"),
    ("gk_saves_p90", "Goalkeeping", "Saves per 90", "f2"),
    ("gk_claims_p90", "Goalkeeping", "Crosses Claimed per 90", "f2"),
    ("gk_sweeper_p90", "Goalkeeping", "Sweeper-Keeper Actions per 90", "f2"),
]

CLUB_ATTRIBUTE_CATALOG: list[tuple[str, str, str, str]] = [
    ("matches", "Squad & Context", "League Matches Played", "int"),
    ("squad_size", "Squad & Context", "First-Team Squad Size", "int"),
    ("average_age", "Squad & Context", "Average Squad Age", "years"),
    ("stadium_seats", "Squad & Context", "Stadium Capacity", "int"),
    ("poss_share", "Possession & Tempo", "Average Ball Possession", "pct"),
    ("passes_p90", "Possession & Tempo", "Passes Attempted per 90", "f1"),
    ("pass_acc", "Possession & Tempo", "Team Pass Accuracy", "pct"),
    ("tempo_passes_per_poss_min", "Possession & Tempo", "Circulation Tempo (Passes / Poss. Min)", "f1"),
    ("avg_pass_length", "Possession & Tempo", "Average Pass Length", "meters"),
    ("ppda", "Pressing & Defensive Line", "Pressing Intensity (PPDA — lower = stronger)", "f2"),
    ("high_turnovers_p90", "Pressing & Defensive Line", "High Turnovers Forced per 90", "f2"),
    ("def_line_height_m", "Pressing & Defensive Line", "Defensive Line Height", "meters"),
    ("clearance_share", "Pressing & Defensive Line", "Clearance Share of Defensive Actions", "pct"),
    ("aerials_p90", "Pressing & Defensive Line", "Aerial Duels per 90", "f1"),
    ("field_tilt", "Territory & Progression", "Final-Third Territory Share (Field Tilt)", "pct"),
    ("f3_pass_share", "Territory & Progression", "Final-Third Share of Team Passes", "pct"),
    ("prog_pass_share", "Territory & Progression", "Progressive Pass Share", "pct"),
    ("vert_pass_share", "Territory & Progression", "Vertical Pass Share", "pct"),
    ("long_ball_share", "Territory & Progression", "Long-Ball Share (>32m)", "pct"),
    ("wide_touch_share", "Width & Attacking Threat", "Wide Touch Share (Flanks)", "pct"),
    ("touch_width_sd", "Width & Attacking Threat", "Horizontal Pitch Spread (Width SD)", "meters"),
    ("crosses_p90", "Width & Attacking Threat", "Crosses per 90", "f2"),
    ("box_touches_p90", "Width & Attacking Threat", "Penalty-Box Touches per 90", "f1"),
    ("shots_p90", "Width & Attacking Threat", "Shots per 90", "f2"),
    ("throughballs_p90", "Width & Attacking Threat", "Throughballs per 90", "f2"),
    ("takeons_p90", "Width & Attacking Threat", "1v1 Dribbles per 90", "f2"),
]


def _format_stat_value(val: Any, fmt: str) -> str:
    num = pd.to_numeric(val, errors="coerce")
    if pd.isna(num):
        return "N/A"
    v = float(num)
    if fmt == "int":
        return f"{int(round(v)):,}"
    if fmt == "pct":
        return f"{v * 100:.1f}%"
    if fmt == "f1":
        return f"{v:.1f}"
    if fmt == "f2":
        return f"{v:.2f}"
    if fmt == "meters":
        return f"{v:.1f} m"
    if fmt == "years":
        return f"{v:.1f} yrs"
    return f"{v:.2f}"


def build_clean_player_table(row: pd.Series) -> pd.DataFrame:
    """Return a clean, non-technical scouting table for a player row (no raw column names)."""
    is_gk = str(row.get("position", "")) == "Goalkeeper"
    records = []
    for col, cat, label, fmt in PLAYER_ATTRIBUTE_CATALOG:
        if col not in row.index:
            continue
        if cat == "Goalkeeping" and not is_gk:
            continue
        if is_gk and col in ("shots_p90", "avg_shot_dist_m", "headed_shot_share", "box_touches_p90", "crosses_p90", "open_cross_share"):
            continue
        pct_col = f"pct_{col}"
        pct_val = f"{int(round(float(row[pct_col])))}th" if (pct_col in row.index and pd.notna(row[pct_col])) else "—"
        records.append({
            "Category": cat,
            "Scouting Metric": label,
            "Value": _format_stat_value(row[col], fmt),
            "Position Peer Percentile": pct_val,
        })
    return pd.DataFrame(records)


def build_clean_club_table(row: pd.Series) -> pd.DataFrame:
    """Return a clean, non-technical scouting table for a club row (no raw column names)."""
    records = []
    for col, cat, label, fmt in CLUB_ATTRIBUTE_CATALOG:
        if col not in row.index:
            continue
        pct_col = f"pct_{col}"
        pct_val = f"{int(round(float(row[pct_col])))}th" if (pct_col in row.index and pd.notna(row[pct_col])) else "—"
        records.append({
            "Category": cat,
            "Tactical Metric": label,
            "Value": _format_stat_value(row[col], fmt),
            "Top-5 League Percentile": pct_val,
        })
    return pd.DataFrame(records)


def build_club_tactical_summary(row: pd.Series) -> str:
    """Generate a plain-English 2-sentence summary of how a club plays."""
    team = str(row.get("team", "This club"))
    poss = float(row.get("poss_share", 0.50)) * 100
    ppda = float(row.get("ppda", 11.0))
    tilt = float(row.get("field_tilt", 0.50)) * 100
    line_m = float(row.get("def_line_height_m", 43.0))
    long_b = float(row.get("long_ball_share", 0.12)) * 100

    poss_desc = "dominates the ball" if poss >= 56 else ("shares possession evenly" if poss >= 46 else "operates primarily without the ball")
    press_desc = "an aggressive high press" if ppda <= 9.8 else ("a selective mid-block press" if ppda <= 12.5 else "a compact low-block shape")
    build_desc = "short, patient build-up play" if long_b <= 11.5 else ("a balanced mix of short and direct passes" if long_b <= 15.0 else "direct vertical passes into space")

    return (
        f"<strong>{team}</strong> {poss_desc} (<strong>{poss:.1f}% possession</strong>) and pairs {build_desc} "
        f"(<strong>{long_b:.1f}% long balls</strong>) with {press_desc} (<strong>{ppda:.2f} PPDA</strong>). "
        f"Their defensive block engages at an average height of <strong>{line_m:.1f} m</strong>, controlling "
        f"<strong>{tilt:.1f}%</strong> of final-third territory."
    )


def _read_parquet(filename: str, expected: list[str]) -> pd.DataFrame:
    path = DATA / filename
    if not path.exists():
        st.warning(f"Processed file not found: `{path}`")
        return pd.DataFrame(columns=expected)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        st.error(f"Could not read `{filename}`: {exc}")
        return pd.DataFrame(columns=expected)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.warning(f"`{filename}` is missing expected columns: {', '.join(missing)}")
    return df


@st.cache_data(show_spinner=False)
def load_club_features() -> pd.DataFrame:
    df = _read_parquet(
        "club_ui_metadata.parquet",
        ["league", "season", "team", "matches", "poss_share", "ppda", "field_tilt"],
    )
    if "season" in df:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    if "tactical_identity" not in df.columns:
        df["tactical_identity"] = "Balanced Hybrid System"
    if "league" in df.columns:
        df["league_clean"] = df["league"].map(short_league)
    return df


@st.cache_data(show_spinner=False)
def load_player_features() -> pd.DataFrame:
    df = _read_parquet(
        "player_ui_metadata.parquet",
        ["league", "season", "team", "player_id", "player", "position", "sub_position", "minutes_90"],
    )
    if "season" in df:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    if "archetype_name" not in df.columns:
        if "style_cluster" in df.columns:
            df["archetype_name"] = df["style_cluster"].map(CLUSTER_ARCHETYPE_MAP).fillna(df["sub_position"])
        else:
            df["archetype_name"] = df["sub_position"]
    if "league" in df.columns:
        df["league_clean"] = df["league"].map(short_league)
    return df


@st.cache_data(show_spinner=False)
def load_role_vectors() -> pd.DataFrame:
    return _read_parquet(
        "player_role_vectors_2425.parquet",
        ["player_id", "player", "team", "league", "position", "sub_position", "minutes_90", "style_cluster"],
    )


@st.cache_data(show_spinner=False)
def load_style_space() -> pd.DataFrame:
    df = _read_parquet(
        "player_style_space_2425.parquet",
        ["player_id", "player", "team", "league", "position", "style_cluster", "umap_1", "umap_2"],
    )
    if "archetype_name" not in df.columns and "style_cluster" in df.columns:
        df["archetype_name"] = (
            pd.to_numeric(df["style_cluster"], errors="coerce")
            .map(CLUSTER_ARCHETYPE_MAP)
            .fillna("Other Role")
        )
    if "league" in df.columns:
        df["league_clean"] = df["league"].map(short_league)
    return df


@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame:
    df = _read_parquet(
        "backtest_results_v3.parquet",
        [
            "player_name", "position", "sub_position", "tm_from_club", "tm_to_club",
            "ws_to_club", "transfer_fee", "fit_score", "role_fit_100", "system_fit_100",
            "env_fit_100", "closest_dest_peer", "closest_peer_sim", "MS", "Y_score", "label_base_sec9",
        ],
    )
    players = load_player_features()
    clubs = load_club_features()
    if not df.empty and not players.empty and "ws_player_id" in df.columns and "image_url" in players.columns:
        p_img = (
            players.dropna(subset=["image_url"])
            .drop_duplicates("player_id")
            .set_index("player_id")["image_url"]
            .to_dict()
        )
        df["image_url"] = df["ws_player_id"].map(p_img)
    if not df.empty and not clubs.empty and "crest_url" in clubs.columns:
        c_img = clubs.drop_duplicates("team").set_index("team")["crest_url"].to_dict()
        df["dest_crest_url"] = df["ws_to_club"].map(c_img)
        if "ws_from_club" in df.columns:
            df["orig_crest_url"] = df["ws_from_club"].map(c_img)
    return df


@st.cache_data(show_spinner=False)
def load_scalers() -> dict[str, Any] | None:
    path = DATA / "fit_scalers_2425.npz"
    if not path.exists():
        st.warning(f"Scaler file not found: `{path}`")
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k].copy() for k in z.files}
        required = ["p_mean", "p_scale", "c_mean", "c_scale", "player_cols", "club_cols"]
        missing = [k for k in required if k not in d]
        if missing:
            st.warning("`fit_scalers_2425.npz` missing arrays: " + ", ".join(missing))
            return None
        d["player_cols"] = [str(x) for x in d["player_cols"].tolist()]
        d["club_cols"] = [str(x) for x in d["club_cols"].tolist()]
        return d
    except Exception as exc:
        st.error(f"Could not load scaler arrays: {exc}")
        return None


def season_label(value: Any) -> str:
    try:
        n = int(value)
        return f"20{n // 100:02d}-{n % 100:02d}"
    except (TypeError, ValueError):
        return str(value)


def available_seasons(*frames: pd.DataFrame) -> list[int]:
    vals: set[int] = set()
    for f in frames:
        if "season" in f:
            vals.update(int(x) for x in f["season"].dropna().unique())
    return sorted(vals)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 else 0.0


def _num(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _club_z(row: pd.Series, s: dict[str, Any]) -> np.ndarray:
    cols = s["club_cols"]
    mean = np.asarray(s["c_mean"], dtype=float)
    scale = np.asarray(s["c_scale"], dtype=float)
    raw = np.array([_num(row.get(c, np.nan), np.nan) for c in cols])
    raw = np.where(np.isfinite(raw), raw, mean)
    scale = np.where(scale == 0, 1.0, scale)
    return (raw - mean) / scale


def _player_z(row: pd.Series, poss: float, s: dict[str, Any]) -> np.ndarray:
    cols = s["player_cols"]
    mean = np.asarray(s["p_mean"], dtype=float)
    scale = np.asarray(s["p_scale"], dtype=float)
    raw = np.array([_num(row.get(c, np.nan), np.nan) for c in cols])
    poss = float(np.clip(poss, 0.30, 0.70))
    for name in ON_BALL:
        if name in cols:
            i = cols.index(name)
            if np.isfinite(raw[i]):
                raw[i] *= 0.50 / poss
    for name in DEFENSIVE:
        if name in cols:
            i = cols.index(name)
            if np.isfinite(raw[i]):
                raw[i] *= 0.50 / (1.0 - poss)
    raw = np.where(np.isfinite(raw), raw, mean)
    scale = np.where(scale == 0, 1.0, scale)
    return (raw - mean) / scale


def _weighted_vector(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    w = pd.to_numeric(frame["minutes_90"], errors="coerce").fillna(0).to_numpy(float)
    x = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    return np.average(x, axis=0, weights=w) if w.sum() > 0 else np.mean(x, axis=0)


def build_plain_english_explanation(
    player_row: pd.Series,
    target_row: pd.Series,
    origin_in_2425: bool,
    res: dict[str, Any],
) -> list[str]:
    """Generate 3 plain-English scouting takeaways for non-technical readers."""
    bullets: list[str] = []
    pname = str(player_row.get("player", "The player"))
    dest = str(target_row.get("team", "the target club"))
    peer = res["closest_dest_peer"]
    sim_pct = int(round(max(0.0, res["closest_peer_sim"]) * 100))

    if res["role_fit_100"] >= 80:
        bullets.append(
            f"**Seamless Role Fit ({res['role_fit_100']:.0f}/100):** {pname}'s on-pitch habits closely mirror "
            f"**{peer}** ({sim_pct}% stylistic match), showing they already play the exact role {dest} asks of that position."
        )
    elif res["role_fit_100"] >= 65:
        bullets.append(
            f"**Solid Role Alignment ({res['role_fit_100']:.0f}/100):** {pname} shares core positional traits with "
            f"**{peer}** ({sim_pct}% stylistic match), requiring only minor role tweaks at {dest}."
        )
    else:
        bullets.append(
            f"**Different Role Profile ({res['role_fit_100']:.0f}/100):** {pname} plays differently from {dest}'s "
            f"existing option **{peer}** ({sim_pct}% match) — signing them would intentionally change how that role is played."
        )

    identity = str(target_row.get("tactical_identity", "Balanced Hybrid System"))
    if res["system_fit_100"] >= 65:
        bullets.append(
            f"**Fits Manager's System ({res['system_fit_100']:.0f}/100):** {dest} operates as a *{identity}*, "
            f"and {pname}'s pressing intensity, ball retention, and progressive passing align naturally with those demands."
        )
    elif res["system_fit_100"] >= 48:
        bullets.append(
            f"**Adaptable to Team Tempo ({res['system_fit_100']:.0f}/100):** {dest}'s *{identity}* approach is "
            f"moderately compatible with {pname}'s current pressing and passing tempo."
        )
    else:
        bullets.append(
            f"**System Adjustment Needed ({res['system_fit_100']:.0f}/100):** {dest}'s *{identity}* style asks for "
            f"different pressing height or build-up tempo than {pname} typically shows."
        )

    if not origin_in_2425:
        bullets.append(
            f"**League Step-Up Context (50/100 Neutral):** {pname}'s current club was outside the 2024-25 Top-5 dataset, "
            f"so environment continuity is held neutral."
        )
    elif res["env_fit_100"] >= 70:
        bullets.append(
            f"**Familiar Tactical Environment ({res['env_fit_100']:.0f}/100):** {player_row.get('team')} and {dest} "
            f"play very similar football overall, lowering adaptation risk."
        )
    else:
        bullets.append(
            f"**New Tactical Environment ({res['env_fit_100']:.0f}/100):** Moving from {player_row.get('team')} to {dest} "
            f"represents a noticeable shift in team possession and territorial dominance."
        )
    return bullets


def compute_fit_reference(
    player_row: pd.Series,
    dest_team: str,
    club_df: pd.DataFrame | None = None,
    role_df: pd.DataFrame | None = None,
    scalers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reference implementation for style compatibility scoring."""
    club_df = club_df if club_df is not None else load_club_features()
    role_df = role_df if role_df is not None else load_role_vectors()
    scalers = scalers if scalers is not None else load_scalers()

    if str(player_row.get("position", "")) == "Goalkeeper":
        return {"is_goalkeeper": True, "note": "The tactical style model covers outfield roles (Attack, Midfield, Defender)."}
    if scalers is None or role_df.empty or club_df.empty:
        return {"error": "Required processed data or scaler is unavailable."}

    c2425 = club_df[pd.to_numeric(club_df["season"], errors="coerce") == 2425]
    target = c2425[c2425["team"].astype(str) == str(dest_team)]
    if target.empty:
        return {"error": f"{dest_team} was not in the 2024-25 Top-5 dataset (newly promoted clubs lack a 2024-25 baseline)."}
    target_row = target.iloc[0]

    p_season = _num(player_row.get("season"), 2425)
    orig_team = str(player_row.get("team", ""))
    orig_same_season = club_df[
        (club_df["team"].astype(str) == orig_team)
        & (pd.to_numeric(club_df["season"], errors="coerce") == int(p_season))
    ]
    orig_2425 = c2425[c2425["team"].astype(str) == orig_team]

    if not orig_same_season.empty:
        orig_poss = _num(orig_same_season.iloc[0].get("poss_share"), 0.50)
    elif not orig_2425.empty:
        orig_poss = _num(orig_2425.iloc[0].get("poss_share"), 0.50)
    else:
        orig_poss = 0.50

    pvec = _player_z(player_row, orig_poss, scalers)
    pcols = scalers["player_cols"]
    pz = dict(zip(pcols, pvec))
    cz = dict(zip(scalers["club_cols"], _club_z(target_row, scalers)))

    pool = role_df[
        (role_df["team"].astype(str) == str(dest_team))
        & (role_df["position"].astype(str) == str(player_row.get("position", "")))
    ].copy()
    if pool.empty:
        return {"error": f"No 2024-25 outfield role peers found at {dest_team} for {player_row.get('position')}."}

    mat = pool[pcols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    pool["__cos"] = [cosine_similarity(pvec, v) for v in mat]

    sub = player_row.get("sub_position")
    matching = pool[pool["sub_position"].astype(str) == str(sub)].copy()
    enough = (
        not pd.isna(sub)
        and not matching.empty
        and pd.to_numeric(matching["minutes_90"], errors="coerce").fillna(0).sum() >= 600
    )
    if enough:
        peer_vec = _weighted_vector(matching, pcols)
        rcos = cosine_similarity(pvec, peer_vec)
        method = f"Matched directly against {dest_team}'s {sub} starters"
    else:
        nearest = pool.nlargest(min(3, len(pool)), "__cos")
        weights = pd.to_numeric(nearest["minutes_90"], errors="coerce").fillna(0).to_numpy(float)
        sims = nearest["__cos"].to_numpy(float)
        rcos = float(np.average(sims, weights=weights)) if weights.sum() else float(np.mean(sims))
        peer_vec = _weighted_vector(nearest, pcols)
        method = f"Matched against top {len(nearest)} closest {player_row.get('position')} peers at {dest_team}"

    closest = pool.loc[pool["__cos"].idxmax()]

    ax1 = ((-cz.get("ppda", 0) + cz.get("def_line_height_m", 0)) / 2) * (
        (pz.get("high_press_actions_p90", 0) + pz.get("avg_def_action_x", 0)) / 2
    )
    ax2 = ((cz.get("poss_share", 0) - cz.get("long_ball_share", 0)) / 2) * (
        (pz.get("pass_acc", 0) - pz.get("long_ball_share", 0) - pz.get("dispossessed_p90", 0)) / 3
    )
    ax3 = ((cz.get("prog_pass_share", 0) + cz.get("vert_pass_share", 0)) / 2) * (
        (pz.get("prog_passes_p90", 0) + pz.get("fwd_pass_share", 0)) / 2
    )
    sysfit = float(np.tanh(np.mean([ax1, ax2, ax3])))

    if not orig_2425.empty:
        envcos = cosine_similarity(_club_z(orig_2425.iloc[0], scalers), _club_z(target_row, scalers))
        origin_in_2425 = True
    else:
        envcos = 0.0
        origin_in_2425 = False

    rcos = float(np.clip(rcos, -1.0, 1.0))
    sysfit = float(np.clip(sysfit, -1.0, 1.0))
    envcos = float(np.clip(envcos, -1.0, 1.0))

    out = {
        "is_goalkeeper": False,
        "fit_score": round(50.0 + 50.0 * (0.80 * rcos + 0.15 * sysfit + 0.05 * envcos), 1),
        "role_fit_100": round(50.0 + 50.0 * rcos, 1),
        "system_fit_100": round(50.0 + 50.0 * sysfit, 1),
        "env_fit_100": round(50.0 + 50.0 * envcos, 1),
        "r_cos": rcos,
        "sys_fit": sysfit,
        "env_cos": envcos,
        "closest_dest_peer": str(closest.get("player", "Unknown")),
        "closest_peer_id": int(closest.get("player_id", 0)),
        "closest_peer_sim": round(float(closest["__cos"]), 3),
        "peer_method": method,
        "player_z": pvec,
        "peer_z": peer_vec,
        "player_cols": pcols,
        "target_row": target_row,
    }
    out["bullets"] = build_plain_english_explanation(player_row, target_row, origin_in_2425, out)
    return out


# ======================================================================================
# FAST ENGINE: precomputes club vectors, role pools and possession lookups once.
# ======================================================================================
class FitEngine:
    """Precomputes club vectors, role pools and possession lookups so one fit costs < 1ms."""

    def __init__(self, clubs: pd.DataFrame, roles: pd.DataFrame, scalers: dict[str, Any]):
        self.s = scalers
        self.pcols = list(scalers["player_cols"])
        self.p_mean = np.asarray(scalers["p_mean"], dtype=float)
        self.p_scale = np.where(np.asarray(scalers["p_scale"], dtype=float) == 0, 1.0, np.asarray(scalers["p_scale"], dtype=float))
        pidx = {c: i for i, c in enumerate(self.pcols)}
        self.on_idx = [pidx[c] for c in ON_BALL if c in pidx]
        self.def_idx = [pidx[c] for c in DEFENSIVE if c in pidx]
        self.pidx = pidx

        c = clubs.copy()
        c["_s"] = pd.to_numeric(c["season"], errors="coerce")
        self.club_rows: dict[tuple[str, int], pd.Series] = {}
        for _, row in c.dropna(subset=["_s"]).iterrows():
            self.club_rows[(str(row["team"]), int(row["_s"]))] = row
        self.club_z = {t: _club_z(r, scalers) for (t, s), r in self.club_rows.items() if s == 2425}

        mat_all = roles[self.pcols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        minutes_all = pd.to_numeric(roles["minutes_90"], errors="coerce").fillna(0).to_numpy(float)
        sub_all = roles["sub_position"].astype(str).to_numpy()
        name_all = roles["player"].astype(str).to_numpy()
        id_all = pd.to_numeric(roles["player_id"], errors="coerce").fillna(0).to_numpy(int)
        key = pd.MultiIndex.from_arrays([roles["team"].astype(str), roles["position"].astype(str)])
        self.pools: dict[tuple[str, str], dict[str, Any]] = {}
        for k, idx in pd.Series(np.arange(len(roles))).groupby(key).indices.items():
            idx = np.asarray(idx)
            mat = mat_all[idx]
            self.pools[k] = dict(
                mat=mat,
                norm=np.linalg.norm(mat, axis=1),
                minutes=minutes_all[idx],
                sub=sub_all[idx],
                names=name_all[idx],
                ids=id_all[idx],
            )

    def _poss(self, team: str, season: float) -> float:
        row = self.club_rows.get((team, int(season)))
        if row is None:
            row = self.club_rows.get((team, 2425))
        return _num(row.get("poss_share"), 0.50) if row is not None else 0.50

    def _pvec(self, player_row, poss: float) -> np.ndarray:
        raw = np.array([_num(player_row.get(c, np.nan), np.nan) for c in self.pcols])
        poss = float(np.clip(poss, 0.30, 0.70))
        for i in self.on_idx:
            if np.isfinite(raw[i]):
                raw[i] *= 0.50 / poss
        for i in self.def_idx:
            if np.isfinite(raw[i]):
                raw[i] *= 0.50 / (1.0 - poss)
        raw = np.where(np.isfinite(raw), raw, self.p_mean)
        return (raw - self.p_mean) / self.p_scale

    @staticmethod
    def _wavg(mat: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.average(mat, axis=0, weights=w) if w.sum() > 0 else np.mean(mat, axis=0)

    def fit(self, player_row, dest_team: str) -> dict[str, Any]:
        if str(player_row.get("position", "")) == "Goalkeeper":
            return {"is_goalkeeper": True, "note": "The tactical style model covers outfield roles (Attack, Midfield, Defender)."}
        dest_team = str(dest_team)
        target_row = self.club_rows.get((dest_team, 2425))
        if target_row is None:
            return {"error": f"{dest_team} was not in the 2024-25 Top-5 dataset (newly promoted clubs lack a 2024-25 baseline)."}
        p_season = _num(player_row.get("season"), 2425)
        orig_team = str(player_row.get("team", ""))
        orig_poss = self._poss(orig_team, p_season)
        pvec = self._pvec(player_row, orig_poss)

        pos = str(player_row.get("position", ""))
        pool = self.pools.get((dest_team, pos))
        if pool is None:
            return {"error": f"No 2024-25 outfield role peers found at {dest_team} for {player_row.get('position')}."}

        denom = np.linalg.norm(pvec) * pool["norm"]
        cos = np.where(denom > 0, pool["mat"] @ pvec / np.where(denom > 0, denom, 1.0), 0.0)

        sub = player_row.get("sub_position")
        matching = (pool["sub"] == str(sub))
        enough = (not pd.isna(sub)) and matching.any() and pool["minutes"][matching].sum() >= 600
        if enough:
            peer_vec = self._wavg(pool["mat"][matching], pool["minutes"][matching])
            rcos = cosine_similarity(pvec, peer_vec)
            method = f"Matched directly against {dest_team}'s {sub} starters"
        else:
            order = np.argsort(-cos, kind="stable")[: min(3, len(cos))]
            w = pool["minutes"][order]
            rcos = float(np.average(cos[order], weights=w)) if w.sum() else float(np.mean(cos[order]))
            peer_vec = self._wavg(pool["mat"][order], w)
            method = f"Matched against top {len(order)} closest {player_row.get('position')} peers at {dest_team}"

        best = int(np.argmax(cos))
        pz = dict(zip(self.pcols, pvec))
        cz = dict(zip(self.s["club_cols"], self.club_z[dest_team]))
        ax1 = ((-cz.get("ppda", 0) + cz.get("def_line_height_m", 0)) / 2) * ((pz.get("high_press_actions_p90", 0) + pz.get("avg_def_action_x", 0)) / 2)
        ax2 = ((cz.get("poss_share", 0) - cz.get("long_ball_share", 0)) / 2) * ((pz.get("pass_acc", 0) - pz.get("long_ball_share", 0) - pz.get("dispossessed_p90", 0)) / 3)
        ax3 = ((cz.get("prog_pass_share", 0) + cz.get("vert_pass_share", 0)) / 2) * ((pz.get("prog_passes_p90", 0) + pz.get("fwd_pass_share", 0)) / 2)
        sysfit = float(np.tanh(np.mean([ax1, ax2, ax3])))

        origin_in_2425 = orig_team in self.club_z
        envcos = cosine_similarity(self.club_z[orig_team], self.club_z[dest_team]) if origin_in_2425 else 0.0

        rcos, sysfit, envcos = (float(np.clip(x, -1.0, 1.0)) for x in (rcos, sysfit, envcos))
        out = {
            "is_goalkeeper": False,
            "fit_score": round(50.0 + 50.0 * (0.80 * rcos + 0.15 * sysfit + 0.05 * envcos), 1),
            "role_fit_100": round(50.0 + 50.0 * rcos, 1),
            "system_fit_100": round(50.0 + 50.0 * sysfit, 1),
            "env_fit_100": round(50.0 + 50.0 * envcos, 1),
            "r_cos": rcos,
            "sys_fit": sysfit,
            "env_cos": envcos,
            "closest_dest_peer": str(pool["names"][best]),
            "closest_peer_id": int(pool["ids"][best]),
            "closest_peer_sim": round(float(cos[best]), 3),
            "peer_method": method,
            "player_z": pvec,
            "peer_z": peer_vec,
            "player_cols": self.pcols,
            "target_row": target_row,
        }
        out["bullets"] = build_plain_english_explanation(player_row, target_row, origin_in_2425, out)
        return out


@st.cache_resource(show_spinner="Loading tactical fit engine...")
def get_engine() -> FitEngine | None:
    clubs, roles, scalers = load_club_features(), load_role_vectors(), load_scalers()
    if clubs.empty or roles.empty or scalers is None:
        return None
    return FitEngine(clubs, roles, scalers)


def compute_fit(player_row, dest_team, club_df=None, role_df=None, scalers=None) -> dict[str, Any]:
    """Fast style compatibility score (matches compute_fit_reference)."""
    eng = get_engine()
    if eng is None:
        return {"error": "Required processed data or scaler is unavailable."}
    return eng.fit(player_row, dest_team)


def _player_record(player_id: int, season: int, team: str) -> dict | None:
    pl = load_player_features()
    m = pl[
        (pd.to_numeric(pl["player_id"], errors="coerce") == player_id)
        & (pd.to_numeric(pl["season"], errors="coerce") == season)
        & (pl["team"].astype(str) == team)
    ]
    return None if m.empty else m.iloc[0].to_dict()


@st.cache_data(show_spinner=False, max_entries=512)
def rank_clubs_for_player(player_id: int, season: int, team: str, top_n: int | None = 10) -> pd.DataFrame:
    """Fit of one player against every 2024-25 Top-5 club."""
    eng, rec = get_engine(), _player_record(int(player_id), int(season), str(team))
    if eng is None or rec is None:
        return pd.DataFrame()
    rows = []
    for (dest, s), crow in eng.club_rows.items():
        if s != 2425 or dest == str(team):
            continue
        res = eng.fit(rec, dest)
        if "error" in res or res.get("is_goalkeeper"):
            continue
        rows.append({
            "team": dest,
            "league": short_league(crow.get("league", "")),
            "crest_url": crow.get("crest_url", ""),
            "coach_name": crow.get("coach_name", ""),
            "tactical_identity": crow.get("tactical_identity", ""),
            "fit_score": res["fit_score"],
            "role_fit_100": res["role_fit_100"],
            "system_fit_100": res["system_fit_100"],
            "env_fit_100": res["env_fit_100"],
            "closest_dest_peer": res["closest_dest_peer"],
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("fit_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["fit_percentile"] = (100.0 * (1.0 - (df["rank"] - 1) / max(len(df) - 1, 1))).round(0)
    return df if top_n is None else df.head(top_n)


@st.cache_data(show_spinner=False, max_entries=128)
def rank_players_for_club(
    dest_team: str,
    position: str = "All",
    season: int = 2425,
    min_minutes: int = 900,
    max_age: float = 35.0,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank outfield candidates for one club."""
    eng, players = get_engine(), load_player_features()
    if eng is None or players.empty:
        return pd.DataFrame()
    cand = players[
        (pd.to_numeric(players["season"], errors="coerce") == season)
        & (pd.to_numeric(players["minutes_90"], errors="coerce").fillna(0) >= min_minutes)
        & (players["team"].astype(str) != str(dest_team))
        & (players["position"].isin(["Attack", "Midfield", "Defender"]))
    ].copy()
    if position != "All":
        cand = cand[cand["position"].astype(str) == position]
    if "age_2025" in cand.columns:
        cand = cand[cand["age_2025"].fillna(25.0) <= max_age]
    keep = ["player", "team", "league", "position", "sub_position", "age_2025", "market_value_eur", "image_url", "club_crest_url", "archetype_name"]
    rows = []
    for rec in cand.to_dict("records"):
        res = eng.fit(rec, dest_team)
        if "error" in res or res.get("is_goalkeeper"):
            continue
        row = {k: rec.get(k, np.nan) for k in keep}
        row["league"] = short_league(row.get("league", ""))
        row["market_value_display"] = format_eur(row.get("market_value_eur"))
        row.update(
            fit_score=res["fit_score"],
            role_fit_100=res["role_fit_100"],
            system_fit_100=res["system_fit_100"],
            closest_dest_peer=res["closest_dest_peer"],
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("fit_score", ascending=False).head(top_n).reset_index(drop=True)