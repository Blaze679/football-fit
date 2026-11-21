from difflib import SequenceMatcher
from pathlib import Path
import re
import sys
import unicodedata
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TM = Path("data/raw/transfermarkt")
PROCESSED = Path("data/processed")

LEAGUE_TO_COMP = {
    "ENG-Premier League": "GB1",
    "ESP-La Liga": "ES1",
    "ITA-Serie A": "IT1",
    "GER-Bundesliga": "L1",
    "FRA-Ligue 1": "FR1",
}

ALIAS_QUERY = {
    "Man City": "manchester city",
    "Man Utd": "manchester united",
    "PSG": "paris saint germain",
    "RBL": "rb leipzig",
    "Wolves": "wolverhampton",
    "FC Heidenheim": "heidenheim",
    "Atletico": "atletico",
}


def norm_text(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return " ".join(s.split())


clubs_tm = pd.read_csv(TM / "clubs.csv.gz")
clubs_tm["norm_name"] = clubs_tm["name"].map(norm_text)
cs_ui = pd.read_parquet(PROCESSED / "club_ui_metadata.parquet")

team_league = cs_ui.drop_duplicates("team").set_index("team")["league"].to_dict()

for team, lg in team_league.items():
    comp = LEAGUE_TO_COMP.get(lg)
    pool = clubs_tm[clubs_tm["domestic_competition_id"] == comp].reset_index(drop=True)
    q = ALIAS_QUERY.get(team, norm_text(team))

    exact = pool[pool["norm_name"] == q]
    if not exact.empty:
        row = exact.iloc[0]
    else:
        sub = pool[pool["norm_name"].str.contains(r"\b" + re.escape(q) + r"\b", na=False)]
        if not sub.empty:
            row = sub.sort_values("last_season", ascending=False).iloc[0]
        else:
            scores = [SequenceMatcher(None, q, nn).ratio() for nn in pool["norm_name"]]
            best_i = int(pd.Series(scores).idxmax())
            row = pool.iloc[best_i]

    mask = cs_ui["team"] == team
    cid = int(row["club_id"])
    cs_ui.loc[mask, "tm_club_id"] = cid
    cs_ui.loc[mask, "tm_club_name"] = row["name"]
    cs_ui.loc[mask, "stadium_name"] = row["stadium_name"]
    cs_ui.loc[mask, "stadium_seats"] = row["stadium_seats"]
    cs_ui.loc[mask, "coach_name"] = row["coach_name"]
    cs_ui.loc[mask, "average_age"] = row["average_age"]
    cs_ui.loc[mask, "squad_size"] = row["squad_size"]
    cs_ui.loc[mask, "net_transfer_record"] = row["net_transfer_record"]
    cs_ui.loc[mask, "crest_url"] = f"https://tmssl.akamaized.net//images/wappen/head/{cid}.png"

cs_ui.to_parquet(PROCESSED / "club_ui_metadata.parquet", index=False)

club_crest_map = cs_ui.drop_duplicates("team").set_index("team")["crest_url"].to_dict()

ps_ui = pd.read_parquet(PROCESSED / "player_ui_metadata.parquet")
ps_ui["club_crest_url"] = ps_ui["team"].map(club_crest_map)
ps_ui.loc[ps_ui["style_cluster"] == 6, "archetype_name"] = "Deep-Lying Metronome & Anchor"
ps_ui.to_parquet(PROCESSED / "player_ui_metadata.parquet", index=False)

space = pd.read_parquet(PROCESSED / "player_style_space_2425.parquet")
space["club_crest_url"] = space["team"].map(club_crest_map)
space.loc[space["style_cluster"] == 6, "archetype_name"] = "Deep-Lying Metronome & Anchor"
space.to_parquet(PROCESSED / "player_style_space_2425.parquet", index=False)

print(f"Matched WhoScored clubs to TM: {cs_ui['tm_club_id'].notna().sum()} / {len(cs_ui)} club-seasons ({cs_ui['team'].nunique()} unique clubs)")
check_18 = ['Athletic Club', 'Atletico', 'Bayern', 'Como', 'Espanyol', 'FC Heidenheim', 'Genoa', 'Inter', 'Leeds', 'Leverkusen', 'Lille', 'Man City', 'Man Utd', 'PSG', 'RBL', 'Real Betis', 'West Ham', 'Wolves']
print(cs_ui[cs_ui["team"].isin(check_18)].drop_duplicates("team")[["team", "tm_club_name", "coach_name", "crest_url"]].to_string(index=False))