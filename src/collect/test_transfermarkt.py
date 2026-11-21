"""Load the public Transfermarkt tables (dcaribou/transfermarkt-datasets) and build a RAW cohort:
summer-2025 transfers from a club that played in a top-5 league in 2024-25 to a club that played in a
top-5 league in 2025-26. League membership comes from the games table (who actually played in which
league that season), NOT from the clubs table, which shows each club's league at the snapshot date.
Run from the project root (venv active):  python src/collect/test_transfermarkt.py

For each table: use a local copy in data/raw/transfermarkt (name.csv or name.csv.gz) if present,
otherwise download it with a browser-like User-Agent header.
"""
from pathlib import Path

import pandas as pd

BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"   # from the dataset's README
ZIP_URL = BASE + "transfermarkt-datasets.zip"                       # manual fallback (browser download)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
OUT = Path("data/raw/transfermarkt")
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    for suffix in (".csv", ".csv.gz"):
        local = OUT / f"{name}{suffix}"
        if local.exists():
            print(f"[local] {name}: {local}")
            return pd.read_csv(local)
    df = pd.read_csv(f"{BASE}{name}.csv.gz", storage_options=HEADERS)
    df.to_csv(OUT / f"{name}.csv.gz", index=False)
    print(f"[download] {name}")
    return df


tables = {}
for name in ["transfers", "games", "players"]:
    try:
        tables[name] = load(name)
    except Exception as e:
        print(f"[fail] {name}: {type(e).__name__}: {str(e)[:150]}")
        continue
    print(f"== {name}: {len(tables[name]):,} rows ==")
    print(list(tables[name].columns))

if not {"transfers", "games"} <= tables.keys():
    raise SystemExit(
        "\nDownload failed. Manual fallback: open this link in your browser, unzip the file, and copy "
        f"transfers.csv, games.csv and players.csv into {OUT}\n{ZIP_URL}\nThen run this script again."
    )

tr, g = tables["transfers"].copy(), tables["games"]
print("\n== transfer_season values (check the label format) ==")
print(tr["transfer_season"].value_counts().head(6))

TOP5 = ["GB1", "ES1", "IT1", "L1", "FR1"]     # my assumption for Transfermarkt's league codes - check below
g5 = g[g["competition_id"].isin(TOP5)]
print("\n== games per top-5 league and season (check the codes and what 'season' means) ==")
print(g5.groupby(["competition_id", "season"]).size().tail(12))


def clubs_in(season):
    s = g5[g5["season"] == season]
    return set(s["home_club_id"]) | set(s["away_club_id"])


from_ok, to_ok = clubs_in(2024), clubs_in(2025)
print(f"\nclubs found in the top 5 leagues: season 2024 = {len(from_ok)}, season 2025 = {len(to_ok)}  (expect 96 each)")

tr["date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
cohort = tr[
    tr["date"].between("2025-06-01", "2025-09-30")
    & tr["from_club_id"].isin(from_ok) & tr["to_club_id"].isin(to_ok)
    & (tr["from_club_id"] != tr["to_club_id"])
]
fee = cohort["transfer_fee"]
print(f"\n== Summer 2025 transfers, top-5 club (2024-25) -> top-5 club (2025-26): {len(cohort)} ==")
print(f"fee > 0: {int((fee > 0).sum())} | fee == 0: {int((fee == 0).sum())} | fee missing: {int(fee.isna().sum())}")
print("(there is no loan column: loans and free moves still need separating)")
print(cohort[["player_name", "from_club_name", "to_club_name", "transfer_date", "transfer_fee"]].head(15).to_string(index=False))
cohort.to_csv(OUT / "cohort_raw_summer2025.csv", index=False)
print(f"\nsaved {OUT / 'cohort_raw_summer2025.csv'}")