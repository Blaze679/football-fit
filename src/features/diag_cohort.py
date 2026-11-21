import pandas as pd

c = pd.read_csv("data/processed/cohort_flagged_summer2025.csv", parse_dates=["transfer_date"])
flags = ["probable_loan_out", "probable_loan_return", "multi_move_window", "age_le_21"]

print("rows that fail ONLY one flag:")
for f in flags:
    only = c[c[f] & ~c[[x for x in flags if x != f]].any(axis=1)]
    print(f"  only {f}: {len(only)}")

m = c[c.multi_move_window].sort_values(["player_id", "transfer_date"])
print("\nsample multi-move players:")
for pid in m.player_id.drop_duplicates().head(10):
    print()
    for r in m[m.player_id == pid].itertuples():
        print(f"  {r.player_name} | {r.transfer_date.date()} | {r.from_club_name} -> {r.to_club_name} | fee {r.transfer_fee} | out={r.probable_loan_out} ret={r.probable_loan_return}")