import soccerdata as sd

LEAGUE = "ENG-Premier League"
SEASON = "2024-2025"

tests = {
    "FBref": lambda: sd.FBref(LEAGUE, SEASON).read_schedule(),
    "FotMob": lambda: sd.FotMob(LEAGUE, SEASON).read_schedule(),
    "Sofascore": lambda: sd.Sofascore(LEAGUE, SEASON).read_schedule(),
    "WhoScored": lambda: sd.WhoScored(LEAGUE, SEASON).read_schedule(),
}

for name, run in tests.items():
    try:
        df = run()
        print(f"[OK] {name}: {df.shape}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:150]}")