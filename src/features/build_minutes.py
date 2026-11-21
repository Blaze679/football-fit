"""Build minutes per player per match, and per player per season, from the event files.

Reads data/processed/events_*.parquet (made by load_events.py) and writes
  data/processed/player_match_minutes.parquet   one row per player per match (started, minute on/off, minutes)
  data/processed/match_qc.parquet               one row per match with quality flags
  data/processed/player_season_minutes.parquet  one row per player per club per season

Run from the project root (venv active):  python src/features/build_minutes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_events import parse_qualifiers  # noqa: E402

PROCESSED = Path("data/processed")
COLS = ["league", "season", "game", "game_id", "period", "minute", "second", "type",
        "team_id", "team", "player_id", "player", "card_type", "qualifiers"]
KEEP = ["End", "FormationSet", "SubstitutionOn", "SubstitutionOff", "Card"]
RED = {"Red", "SecondYellow"}
MIN_EVENTS = 1000


def _ids(text) -> list[int]:
    return [int(float(x)) for x in str(text).replace(" ", "").split(",") if x != ""]


def process_match(
    m: pd.DataFrame,
    n_events: int,
    full: pd.DataFrame | None = None,
    name_map: dict[int, str] | None = None,
):
    """m = the lineup/sub/card/end events of ONE match. Returns (player rows, match row)."""
    flags: list[str] = []
    first = m.iloc[0]
    raw = m["minute"] + m["second"].fillna(0) / 60.0
    end1 = m[(m["type"] == "End") & (m["period"] == "FirstHalf")]
    end2 = m[(m["type"] == "End") & (m["period"] == "SecondHalf")]
    if len(end1):
        fh_len = float(raw[end1.index].max())
    else:
        fh_len = 45.0
        flags.append("no_first_half_end")
    t = pd.Series(np.where(m["period"] == "SecondHalf", fh_len + raw - 45.0, raw), index=m.index)
    if len(end2):
        match_len = float(t[end2.index].max())
    else:
        flags.append("no_final_whistle")
        match_len = np.nan
        if full is not None:   # abandoned / incomplete: use the last event of the two halves
            f = full[full["period"].isin(["FirstHalf", "SecondHalf"])]
            fr = f["minute"] + f["second"].fillna(0) / 60.0
            ft = np.where(f["period"] == "SecondHalf", fh_len + fr - 45.0, fr)
            if len(ft):
                match_len = float(np.max(ft))
    if not np.isnan(match_len):
        if match_len < 90:
            flags.append("short_match")
        elif match_len > 110:
            flags.append("long_match")
    if n_events < MIN_EVENTS:
        flags.append("few_events")

    rows = []
    for team_id, tm in m.groupby("team_id"):
        team = tm["team"].iloc[0]
        fs = tm[tm["type"] == "FormationSet"]
        if fs.empty:
            flags.append(f"no_lineup:{team}")
            continue
        q = parse_qualifiers(fs.iloc[0]["qualifiers"])
        try:
            ids, slots = _ids(q["InvolvedPlayers"]), _ids(q["TeamPlayerFormation"])
        except KeyError:
            flags.append(f"lineup_unreadable:{team}")
            continue
        starters = [p for p, s in zip(ids, slots) if s > 0]
        if len(starters) != 11:
            flags.append(f"starters={len(starters)}:{team}")
        rec = {p: dict(started=True, on=0.0, off=match_len, red=False, subbed_off=False) for p in starters}

        on_ev = tm[tm["type"] == "SubstitutionOn"].dropna(subset=["player_id"])
        off_ev = tm[tm["type"] == "SubstitutionOff"].dropna(subset=["player_id"])
        if len(on_ev) != len(off_ev):
            flags.append(f"subs_on_off_mismatch:{team}")
        for idx, r in on_ev.iterrows():
            rec[int(r["player_id"])] = dict(started=False, on=float(t[idx]), off=match_len, red=False, subbed_off=False)
        for idx, r in off_ev.iterrows():
            p = int(r["player_id"])
            if p in rec:
                rec[p]["off"] = float(t[idx])
                rec[p]["subbed_off"] = True
            else:
                flags.append(f"off_without_on:{p}")
        for idx, r in tm[tm["type"] == "Card"].dropna(subset=["player_id"]).iterrows():
            p = int(r["player_id"])
            if r["card_type"] in RED and p in rec:
                rec[p]["off"] = min(rec[p]["off"], float(t[idx])) if not np.isnan(rec[p]["off"]) else float(t[idx])
                rec[p]["red"] = True

        names = tm.dropna(subset=["player_id", "player"]).drop_duplicates("player_id")
        names = dict(zip(names["player_id"].astype(int), names["player"]))
        for p, d in rec.items():
            minutes = max(d["off"] - d["on"], 0.0) if not np.isnan(d["off"]) else np.nan
            scale = min(1.0, 90.0 / match_len) if not np.isnan(match_len) else np.nan
            pname = (name_map.get(p) if name_map else None) or names.get(p)
            rows.append(dict(league=first["league"], season=first["season"], game_id=int(first["game_id"]),
                             team_id=team_id, team=team, player_id=p, player=pname,
                             started=d["started"], minute_on=round(d["on"], 2),
                             minute_off=None if np.isnan(d["off"]) else round(d["off"], 2),
                             minutes=None if np.isnan(minutes) else round(minutes, 2),
                             minutes_90=None if np.isnan(minutes) or np.isnan(scale) else round(minutes * scale, 2),
                             subbed_off=d["subbed_off"], red_card=d["red"]))
    match_row = dict(league=first["league"], season=first["season"], game_id=int(first["game_id"]),
                     game=first["game"], n_events=n_events,
                     match_len=None if np.isnan(match_len) else round(match_len, 2), flags=";".join(flags))
    return rows, match_row


def save(df: pd.DataFrame, stem: str) -> None:
    try:
        df.to_parquet(PROCESSED / f"{stem}.parquet", index=False)
        print(f"saved {PROCESSED / (stem + '.parquet')}  ({len(df):,} rows)")
    except ImportError:
        df.to_csv(PROCESSED / f"{stem}.csv.gz", index=False)
        print(f"saved {PROCESSED / (stem + '.csv.gz')}  ({len(df):,} rows)")


def main() -> None:
    files = sorted(PROCESSED.glob("events_*.parquet"))
    if not files:
        raise SystemExit("No events_*.parquet in data/processed - run load_events.py first.")
    player_rows, match_rows, card_types = [], [], []
    for f in files:
        df = pd.read_parquet(f, columns=COLS)
        valid_p = df.dropna(subset=["player_id", "player"]).drop_duplicates("player_id")
        name_map = dict(zip(valid_p["player_id"].astype(int), valid_p["player"]))
        n_events = df.groupby("game_id").size()
        card_types.append(df.loc[df["type"] == "Card", "card_type"])
        has_end = set(df.loc[(df["type"] == "End") & (df["period"] == "SecondHalf"), "game_id"])
        no_end = df[~df["game_id"].isin(has_end)]
        no_end_groups = {g: x for g, x in no_end.groupby("game_id")}
        df = df[df["type"].isin(KEEP)]
        for gid, m in df.groupby("game_id"):
            rows, mrow = process_match(m, int(n_events[gid]), no_end_groups.get(gid), name_map)
            player_rows += rows
            match_rows.append(mrow)
        print(f"{f.name}: {df['game_id'].nunique()} matches done")

    pm, mq = pd.DataFrame(player_rows), pd.DataFrame(match_rows)
    save(pm, "player_match_minutes")
    save(mq, "match_qc")

    pm = pm.merge(mq[["game_id", "flags"]], on="game_id")
    ps = (pm.groupby(["league", "season", "team_id", "team", "player_id"], as_index=False)
            .agg(player=("player", "first"), minutes=("minutes", "sum"), minutes_90=("minutes_90", "sum"),
                 apps=("game_id", "nunique"), starts=("started", "sum")))
    team_matches = pm.groupby(["league", "season", "team_id"])["game_id"].nunique().rename("team_matches")
    flagged = (pm[pm["flags"] != ""].groupby(["league", "season", "team_id"])["game_id"].nunique()
                 .rename("team_flagged_matches"))
    ps = ps.merge(team_matches, on=["league", "season", "team_id"], how="left")
    ps = ps.merge(flagged, on=["league", "season", "team_id"], how="left")
    ps["team_flagged_matches"] = ps["team_flagged_matches"].fillna(0).astype(int)
    ps["minutes_share_of_season"] = (ps["minutes_90"] / (ps["team_matches"] * 90)).round(3)
    save(ps, "player_season_minutes")

    print("\n== QUALITY REPORT ==")
    print(f"matches: {len(mq)} | player-match rows: {len(pm):,} | player-seasons: {len(ps):,}")
    print(f"missing player names in player_season_minutes: {ps['player'].isna().sum()}")
    print("card_type values:", pd.concat(card_types).value_counts(dropna=False).to_dict())
    print("match length (min): median", mq["match_len"].median(), "| min", mq["match_len"].min(), "| max", mq["match_len"].max())
    print("max minutes in one match for a player:", pm["minutes"].max(), "| max minutes_90:", pm["minutes_90"].max())
    print("max season minutes share:", ps["minutes_share_of_season"].max(), "(should be about 1.0 or below)")


if __name__ == "__main__":
    main()