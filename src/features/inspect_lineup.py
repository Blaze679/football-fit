"""Print raw lineup, substitution, card and end-of-period events of one match, so the minutes code
can be written against the real qualifier strings.
Run from the project root (venv active):  python src/features/inspect_lineup.py | Set-Clipboard
Then paste the clipboard into the chat.
"""
import pandas as pd

df = pd.read_parquet("data/processed/events_ENG-Premier_League_2025-2026.parquet")
game_id = df["game_id"].iloc[0]
m = df[df["game_id"] == game_id]
pd.set_option("display.max_colwidth", 700, "display.width", 250)
print("game_id:", game_id, "| events:", len(m), "| periods:", list(m["period"].unique()))

for t, n in [("FormationSet", 2), ("FormationChange", 1), ("SubstitutionOff", 2), ("SubstitutionOn", 2), ("Card", 2), ("End", 4)]:
    sub = m[m["type"] == t]
    print(f"\n== {t} ({len(sub)} in this match)")
    cols = ["period", "minute", "second", "expanded_minute", "team", "player", "player_id", "card_type", "qualifiers"]
    print(sub[cols].head(n).to_string(index=False))