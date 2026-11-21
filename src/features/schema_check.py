import ast, glob
from collections import Counter
import pandas as pd

files = glob.glob("data/raw/events/ENG-Premier_League/2025-2026/*.csv")
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
print(len(files), "matches,", len(df), "events\n")

print("== type x outcome_type ==")
print(df.groupby(["type", "outcome_type"]).size().to_string())

print("\n== sample qualifier ==")
qs = df["qualifiers"].dropna()
print(qs.iloc[0][:300])
keys = Counter()
for q in qs.sample(min(20000, len(qs)), random_state=0):
    try:
        for d in ast.literal_eval(q):
            keys[d["type"]["displayName"]] += 1
    except Exception:
        keys["<unparsed>"] += 1
print(keys.most_common(60))

print("\n== attack direction: mean shot x per team/period ==")
print(df[df["is_shot"] == True].groupby(["game_id", "team", "period"])["x"].mean().describe())

print("\n== fouls and substitutions ==")
m = df["type"].astype(str).str.contains("Foul|Sub", case=False)
print(df[m].groupby(["type", "outcome_type"]).size().to_string())

print("\n== related_player_id on successful passes ==")
p = df[(df["type"] == "Pass") & (df["outcome_type"] == "Successful")]
print("share not null:", p["related_player_id"].notna().mean())
nxt = df.groupby("game_id")["player_id"].shift(-1)
print("share equal to next event's player:", (df.loc[p.index, "related_player_id"] == nxt.loc[p.index]).mean())