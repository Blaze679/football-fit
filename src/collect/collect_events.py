import argparse
import random
import time
from pathlib import Path

import soccerdata as sd

t_script = time.time()

p = argparse.ArgumentParser()
p.add_argument("--league", required=True)            # set in the terminal, not here
p.add_argument("--season", required=True)            # set in the terminal, not here
p.add_argument("--minutes", type=float, default=60)  # stop safely after this long
p.add_argument("--part", default="1/1")              # "2/3" = this PC does the 2nd of 3 shares
p.add_argument("--limit", type=int, default=None)    # for quick tests
args = p.parse_args()

out_dir = Path("data/raw/events") / args.league.replace(" ", "_") / args.season
out_dir.mkdir(parents=True, exist_ok=True)

ws = sd.WhoScored(args.league, args.season, data_dir=Path("data/raw/soccerdata"))
sched = ws.read_schedule().reset_index()
played = sched[sched["home_score"].notna()]          # finished matches only
ids = sorted(int(g) for g in played["game_id"])

k, n = map(int, args.part.split("/"))
ids = [g for i, g in enumerate(ids) if i % n == k - 1]
todo = [g for g in ids if not (out_dir / f"{g}.csv").exists()]
print(f"{len(ids)} matches in this share, {len(todo)} still to download", flush=True)
print(f"Schedule loaded in {time.time() - t_script:.0f} seconds", flush=True)

start, done = time.time(), 0
empty_streak = 0
for gid in todo:
    if (time.time() - start) / 60 > args.minutes:
        print("Time limit reached, stopping safely.", flush=True)
        break
    if args.limit is not None and done >= args.limit:
        break
    try:
        ev = ws.read_events(match_id=gid, force_cache=True).reset_index()
        if ev.empty:
            empty_streak += 1
            print(f"[skip] {gid}: no events returned (blocked?)", flush=True)
            if empty_streak >= 5:
                print("5 empty matches in a row: WhoScored is probably blocking us. Stopping.", flush=True)
                break
            continue
        empty_streak = 0
        tmp = out_dir / f"{gid}.csv.tmp"
        ev.to_csv(tmp, index=False)
        tmp.replace(out_dir / f"{gid}.csv")   # file appears only when fully written
        done += 1
        print(f"[{done}/{len(todo)}] saved match {gid}", flush=True)
    except Exception as e:
        print(f"[skip] {gid}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    time.sleep(random.uniform(1, 3))                 # polite pause between matches

elapsed = time.time() - start
print("-" * 50)
print(f"Session finished: {done} matches saved")
print(f"Download time: {elapsed:.0f} seconds ({elapsed / 60:.1f} minutes)")
if done:
    print(f"Average: {elapsed / done:.1f} seconds per match")
print(f"Total script time (including schedule): {time.time() - t_script:.0f} seconds")