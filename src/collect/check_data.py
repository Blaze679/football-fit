"""Count scraped match files per league/season and flag empty or suspiciously small ones.
Run from the project root (venv active):  python src/collect/check_data.py
"""
from pathlib import Path

ROOT = Path("data/raw/events")
FULL_SEASONS = {"2024-2025", "2025-2026"}
EXPECTED = {
    "ENG-Premier_League": 380, "ESP-La_Liga": 380, "ITA-Serie_A": 380,
    "GER-Bundesliga": 306, "FRA-Ligue_1": 306,
}
SMALL_BYTES = 20_000   # a real match is far bigger; smaller files are suspects, not proof

print(f"{'league':<20}{'season':<11}{'files':>6}{'expected':>9}{'missing':>8}{'small':>6}")
problems = []
for league_dir in sorted(p for p in ROOT.glob("*") if p.is_dir()):
    for season_dir in sorted(p for p in league_dir.glob("*") if p.is_dir()):
        files = list(season_dir.glob("*.csv"))
        small = sorted(f for f in files if f.stat().st_size < SMALL_BYTES)
        exp = EXPECTED.get(league_dir.name) if season_dir.name in FULL_SEASONS else None
        missing = "-" if exp is None else str(max(exp - len(files), 0))
        print(f"{league_dir.name:<20}{season_dir.name:<11}{len(files):>6}{(exp or '-'):>9}{missing:>8}{len(small):>6}")
        problems += small

if problems:
    print(f"\n{len(problems)} small files (first 25):")
    for f in problems[:25]:
        print(f"  {f.parent.parent.name}/{f.parent.name}/{f.name}  {f.stat().st_size} bytes")