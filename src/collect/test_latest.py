import soccerdata as sd

LEAGUE = "ENG-Premier League"

# 1. WhoScored: event data for ONE match from the latest full season
try:
    ws = sd.WhoScored(LEAGUE, "2025-2026")
    sched = ws.read_schedule()
    print("[OK] WhoScored schedule 2025-26:", sched.shape)
    print(sched.columns.tolist())
    gid = int(sched["game_id"].iloc[0])
    events = ws.read_events(match_id=gid)
    print("[OK] WhoScored events for one match:", events.shape)
    print(events.columns.tolist())
    print(events.head())
except Exception as e:
    print("[FAIL] WhoScored:", type(e).__name__, str(e)[:200])

# 2. WhoScored: is the current 2026-27 season available?
try:
    ws2 = sd.WhoScored(LEAGUE, "2026-2027")
    sched2 = ws2.read_schedule()
    print("[OK] WhoScored schedule 2026-27:", sched2.shape)
except Exception as e:
    print("[FAIL] WhoScored 2026-27:", type(e).__name__, str(e)[:200])

# 3. FBref: what basic player stats still work?
try:
    fb = sd.FBref(LEAGUE, "2025-2026")
    players = fb.read_player_season_stats(stat_type="standard")
    print("[OK] FBref player stats:", players.shape)
    print(players.columns.tolist())
except Exception as e:
    print("[FAIL] FBref player stats:", type(e).__name__, str(e)[:200])