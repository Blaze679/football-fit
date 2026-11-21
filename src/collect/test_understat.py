import soccerdata as sd

understat = sd.Understat(leagues="ENG-Premier League", seasons="2024-2025", no_cache=True)
teams = understat.read_team_match_stats()
print(teams.head())
print(teams.shape)