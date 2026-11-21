"""Load WhoScored event CSVs (data/raw/events/<League>/<season>/*.csv) into one clean table.

What this does (and nothing more):
  1. Reads every match CSV of a league-season, keeping the original event order inside each match.
  2. Converts pitch coordinates from WhoScored's 0-100 scale to metres (105 x 68).
  3. Parses the `qualifiers` text into usable columns (flags, pass length/angle, zone).
  4. Infers the pass receiver (receiver_id) because related_player_id is empty on passes.
  5. Prints a quick quality report and saves data/processed/events_<League>_<season>.parquet.

Run from the project root, venv active:
  python src/features/load_events.py --league "ENG-Premier League" --season 2025-2026
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

EVENTS_DIR = Path("data/raw/events")
OUT_DIR = Path("data/processed")

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
COORD_X = ["x", "end_x", "blocked_x"]   # 0-100 -> metres along the pitch
COORD_Y = ["y", "end_y", "blocked_y"]   # 0-100 -> metres across the pitch
# goal_mouth_y / goal_mouth_z are left untouched (different scale; handled later).

# Qualifier names turned into True/False columns named q_<Name>.
FLAG_QUALIFIERS = [
    "Longball", "Chipped", "Throughball", "Cross", "HeadPass", "KeyPass",
    "ShotAssist", "IntentionalAssist", "BigChance", "Blocked", "OutfielderBlock",
    "FromCorner", "ThrowIn", "GoalKick", "CornerTaken", "FreekickTaken",
    "LayOff", "Head", "Assisted", "Volley", "RegularPlay", "IndividualPlay",
    "Offensive", "Defensive",
]
NUMERIC_QUALIFIERS = {"Length": "pass_length", "Angle": "pass_angle"}


def parse_qualifiers(raw) -> dict:
    """'[{type:{displayName:'Zone'}, value:'Back'}, ...]' -> {'Zone': 'Back', ...}"""
    if not isinstance(raw, str) or raw in ("", "[]"):
        return {}
    try:
        items = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {"_unparsed": True}
    out = {}
    for item in items:
        try:
            out[item["type"]["displayName"]] = item.get("value", True)
        except (KeyError, TypeError, AttributeError):
            out["_unparsed"] = True
    return out


def add_qualifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    q = df["qualifiers"].map(parse_qualifiers)
    for name in FLAG_QUALIFIERS:
        df[f"q_{name}"] = q.map(lambda d, n=name: n in d)
    df["zone"] = q.map(lambda d: d.get("Zone"))
    for qname, col in NUMERIC_QUALIFIERS.items():
        df[col] = pd.to_numeric(q.map(lambda d, n=qname: d.get(n)), errors="coerce")
    df["qualifiers_unparsed"] = q.map(lambda d: "_unparsed" in d)
    return df


def add_metres(df: pd.DataFrame) -> pd.DataFrame:
    for c in COORD_X:
        if c in df.columns:
            df[c] = df[c] * PITCH_LENGTH_M / 100.0
    for c in COORD_Y:
        if c in df.columns:
            df[c] = df[c] * PITCH_WIDTH_M / 100.0
    return df


def add_receiver(df: pd.DataFrame) -> pd.DataFrame:
    """INFERRED receiver: the next event in the match, if it belongs to the same team and a
    different player, after a successful pass. An approximation, not WhoScored's own field."""
    g = df.groupby("game_id")
    nxt_player = g["player_id"].shift(-1)
    nxt_team = g["team_id"].shift(-1)
    ok = (
        df["type"].eq("Pass")
        & df["outcome_type"].eq("Successful")
        & nxt_team.eq(df["team_id"])
        & nxt_player.notna()
        & nxt_player.ne(df["player_id"])
    )
    df["receiver_id"] = nxt_player.where(ok)
    return df


def load_events(league: str, season: str, events_dir: Path = EVENTS_DIR) -> pd.DataFrame:
    folder = Path(events_dir) / league.replace(" ", "_") / season
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = add_qualifier_columns(df)
    df = add_metres(df)
    df = add_receiver(df)
    return df


def validate(df: pd.DataFrame) -> None:
    per_match = df.groupby("game_id").size()
    print(f"{df['game_id'].nunique()} matches, {len(df):,} events")
    print(f"events per match: min {per_match.min()}, median {int(per_match.median())}, max {per_match.max()}")
    print(f"duplicate rows: {int(df.duplicated().sum())}")
    print(f"x range (m): {df['x'].min():.1f} to {df['x'].max():.1f}  (expect 0 to 105)")
    print(f"y range (m): {df['y'].min():.1f} to {df['y'].max():.1f}  (expect 0 to 68)")
    print(f"unparsed qualifier rows: {int(df['qualifiers_unparsed'].sum())}")
    passes = df[df["type"].eq("Pass") & df["outcome_type"].eq("Successful")]
    print(f"successful passes with an inferred receiver: {passes['receiver_id'].notna().mean():.1%}")
    t = df["expanded_minute"] * 60 + df["second"]
    back = t.groupby([df["game_id"], df["period"]]).diff().lt(0)
    print(f"events whose time goes backwards inside a period: {back.mean():.2%}  (should be ~0)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True)
    ap.add_argument("--season", required=True)
    args = ap.parse_args()

    df = load_events(args.league, args.season)
    validate(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"events_{args.league.replace(' ', '_')}_{args.season}"
    try:
        out = OUT_DIR / f"{stem}.parquet"
        df.to_parquet(out, index=False)
    except ImportError:
        out = OUT_DIR / f"{stem}.csv.gz"
        df.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()