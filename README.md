# Football Fit

Player-to-club playing-style fit for Europe's top five leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1).
University Data Mining & AI project, 2026.

**Question:** does a player's on-pitch style match a club's style, and does that help explain whether a transfer works out?
Motivation: some famous signings flopped (Hazard, Coutinho) while others thrived (Salah), and style fit is one possible reason.

## What it does
- Builds a **style profile** for 5,000+ player-seasons (30 per-90 and ratio features) and 192 club-seasons (22 features) from match event data.
- Scores the **fit** between a player and a target club from 0 to 100:
  `50 + 50 × (0.80 × role similarity + 0.15 × system fit + 0.05 × club continuity)`.
- Tests the score against the **summer 2025 transfer window**: did signings with a better pre-transfer fit get more minutes and more output in 2025-26?
- Provides a **Streamlit app** (player and club profiles, fit analysis, style map, backtest, method page).

## Honest status of the results
The backtest is **exploratory**: about a hundred outfield transfers, several versions of the score and three label thresholds were
compared on the same cohort, and the weights are modelling choices. See the app's *Backtest* and *Method & Limitations* pages for
the numbers with confidence intervals.

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (on macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run src/app/app.py
```
Open http://localhost:8501 and start on the Overview page.

## Data
| Source | Used for | How obtained |
|---|---|---|
| WhoScored match events, 2024-25 and 2025-26, five leagues (3,504 matches) | style features, minutes, goals and assists | scraped with the `soccerdata` library |
| Transfermarkt public tables ([dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)) | transfers, positions, ages, club info | downloaded |

Six small derived tables are included in `data/processed/` so the app runs out of the box. They are built from third-party
sources, so **this repository is private and shared with a few people: please do not redistribute these files.** The raw
event files and the large processed tables are not included. WhoScored can block or change without notice; the collection
script is resumable and polite (random pauses).

## Rebuild the data from scratch (optional)
Only needed if you want to regenerate the tables. Run in this order:
| Step | Script | Output |
|---|---|---|
| 1. Scrape events | `src/collect/collect_events.py --league "ENG-Premier League" --season 2025-2026` (repeat for 5 leagues × 2 seasons) | `data/raw/events/...` |
| 2. Check the scrape | `src/collect/check_data.py` | file counts, empty files |
| 3. Transfermarkt tables | `src/collect/test_transfermarkt.py` | `data/raw/transfermarkt/` |
| 4. Clean events | `src/features/load_events.py` | `data/processed/events_*.parquet` |
| 5. Minutes per player | `src/features/build_minutes.py` | player minutes tables |
| 6. Transfer cohort | `src/features/build_cohort.py` | candidate summer-2025 transfers |
| 7. Link players across sources | `src/features/match_players.py` | matched cohort |
| 8. Outcome labels | `build_labels.py` | transfer labels |
| 9. Style features | `build_features.py` | club and player style features |
| 10. Fit score and backtest | `fit_v3.py` | backtest results, style space, scalers |

## Limitations
- Small cohort; results have wide uncertainty and are not confirmed on separate data.
- Loans are inferred (the Transfermarkt tables have no loan flag); injuries are unknown.
- Event data only: no off-ball movement, physical data or tactical instructions.
- Style similarity is not player quality.
- One 2024-25 match (Holstein Kiel v Bochum) was only partly captured by the source.

## Author
Shayan.