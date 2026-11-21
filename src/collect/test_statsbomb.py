from statsbombpy import sb

comps = sb.competitions()
print(comps[["competition_name", "season_name"]].drop_duplicates().head(20))
print(comps.shape)