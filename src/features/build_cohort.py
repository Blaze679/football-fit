from pathlib import Path
import numpy as np
import pandas as pd

TM = Path("data/raw/transfermarkt")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(TM / "cohort_raw_summer2025.csv")
tr = pd.read_csv(TM / "transfers.csv.gz")
players = pd.read_csv(TM / "players.csv.gz")

raw["transfer_date"] = pd.to_datetime(raw["transfer_date"], errors="coerce")
tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")

# Merge player metadata
p_cols = [
    "player_id", "name", "first_name", "last_name", "date_of_birth",
    "position", "sub_position", "foot", "height_in_cm", "country_of_citizenship"
]
df = raw.merge(players[p_cols], on="player_id", how="left")

# Age at transfer date
df["age_exact"] = (df["transfer_date"] - df["date_of_birth"]).dt.days / 365.25
df["age_years"] = np.floor(df["age_exact"]).astype("Int64")
df["flag_u21"] = df["age_years"] <= 21

# Index all transfers by player_id for fast history checks
tr_valid = tr.dropna(subset=["player_id", "transfer_date"]).copy()
tr_by_player = {pid: grp for pid, grp in tr_valid.groupby("player_id")}

loan_out_flags = []
loan_out_dates = []
loan_return_flags = []
loan_return_prior_dates = []
multi_summer25_flags = []
midseason_2425_flags = []
left_in_2526_flags = []

for row in df.itertuples():
    hist = tr_by_player.get(row.player_id)
    if hist is None:
        loan_out_flags.append(False)
        loan_out_dates.append(pd.NaT)
        loan_return_flags.append(False)
        loan_return_prior_dates.append(pd.NaT)
        multi_summer25_flags.append(False)
        midseason_2425_flags.append(False)
        left_in_2526_flags.append(False)
        continue

    t_date = row.transfer_date
    f_club = row.from_club_id
    t_club = row.to_club_id
    fee = row.transfer_fee

    # 1. Probable loan out: moves back (to_club -> from_club) within 1..730 days after transfer_date
    rev_after = hist[
        (hist["from_club_id"] == t_club)
        & (hist["to_club_id"] == f_club)
        & (hist["transfer_date"] > t_date)
        & (hist["transfer_date"] <= t_date + pd.Timedelta(days=730))
    ].sort_values("transfer_date")
    is_loan_out = len(rev_after) > 0
    loan_out_flags.append(is_loan_out)
    loan_out_dates.append(rev_after["transfer_date"].iloc[0] if is_loan_out else pd.NaT)

    # 2. Probable loan return: moved (to_club -> from_club) within 1..730 days before transfer_date
    # and current move has fee == 0 or NaN
    rev_before = hist[
        (hist["from_club_id"] == t_club)
        & (hist["to_club_id"] == f_club)
        & (hist["transfer_date"] < t_date)
        & (hist["transfer_date"] >= t_date - pd.Timedelta(days=730))
    ].sort_values("transfer_date", ascending=False)
    zero_or_na_fee = pd.isna(fee) or (fee == 0)
    is_loan_ret = (len(rev_before) > 0) and zero_or_na_fee
    loan_return_flags.append(is_loan_ret)
    loan_return_prior_dates.append(rev_before["transfer_date"].iloc[0] if is_loan_ret else pd.NaT)

    # 3. Multiple transfers in summer 2025 window (2025-05-15 to 2025-10-01)
    s25 = hist[hist["transfer_date"].between("2025-05-15", "2025-10-01")]
    multi_summer25_flags.append(len(s25) > 1)

    # 4. Mid-season 2024-25 transfer (split baseline season)
    m2425 = hist[hist["transfer_date"].between("2024-09-05", "2025-05-31")]
    midseason_2425_flags.append(len(m2425) > 0)

    # 5. Left new club during 2025-26 season (2025-10-01 to 2026-05-31)
    left2526 = hist[
        (hist["from_club_id"] == t_club)
        & (hist["transfer_date"].between("2025-10-01", "2026-05-31"))
    ]
    left_in_2526_flags.append(len(left2526) > 0)

df["flag_loan_out"] = loan_out_flags
df["loan_return_date"] = loan_out_dates
df["flag_loan_return"] = loan_return_flags
df["prior_loan_out_date"] = loan_return_prior_dates
df["flag_multi_summer25"] = multi_summer25_flags
df["flag_midseason_2425"] = midseason_2425_flags
df["flag_left_in_2526"] = left_in_2526_flags

# Exclude probable loans out, probable loan returns, and U21 (age <= 21)
df["is_candidate"] = (~df["flag_loan_out"]) & (~df["flag_loan_return"]) & (~df["flag_u21"].fillna(True))

cand = df[df["is_candidate"]].copy()

df.to_parquet(OUT / "cohort_summer2025_annotated.parquet", index=False)
df.to_csv(OUT / "cohort_summer2025_annotated.csv", index=False)
cand.to_parquet(OUT / "cohort_summer2025_candidates.parquet", index=False)
cand.to_csv(OUT / "cohort_summer2025_candidates.csv", index=False)

print("=== COHORT FILTERING SUMMARY ===")
print(f"Raw summer 2025 top-5 -> top-5 moves: {len(df)}")
print(f"  Missing player match in players.csv: {df['name'].isna().sum()}")
print(f"  Missing date_of_birth:               {df['date_of_birth'].isna().sum()}")
print(f"  flag_loan_out:                       {df['flag_loan_out'].sum()}")
print(f"  flag_loan_return:                    {df['flag_loan_return'].sum()}")
print(f"  flag_u21 (age_years <= 21):          {df['flag_u21'].sum()}")
print(f"Remaining senior permanent candidates: {len(cand)}")
print(f"  of which fee > 0:                    {(cand['transfer_fee'] > 0).sum()}")
print(f"  of which fee == 0:                   {(cand['transfer_fee'] == 0).sum()}")
print(f"  of which fee is NaN:                 {cand['transfer_fee'].isna().sum()}")
print(f"  of which flag_multi_summer25:        {cand['flag_multi_summer25'].sum()}")
print(f"  of which flag_midseason_2425:        {cand['flag_midseason_2425'].sum()}")
print(f"  of which flag_left_in_2526:          {cand['flag_left_in_2526'].sum()}")

print("\n=== POSITION BREAKDOWN (CANDIDATES) ===")
print(cand["position"].value_counts(dropna=False).to_string())
print("\n=== SUB-POSITION BREAKDOWN (CANDIDATES) ===")
print(cand["sub_position"].value_counts(dropna=False).to_string())

print("\n=== SAMPLE CANDIDATES (TOP 15 BY FEE) ===")
cols = ["player_id", "player_name", "age_years", "position", "from_club_name", "to_club_name", "transfer_date", "transfer_fee", "flag_multi_summer25", "flag_midseason_2425"]
print(cand.sort_values("transfer_fee", ascending=False)[cols].head(15).to_string(index=False))

print("\n=== SAMPLE CANDIDATES WITH fee == 0 OR NaN (FIRST 15) ===")
free_cand = cand[cand["transfer_fee"].isna() | (cand["transfer_fee"] == 0)]
print(free_cand[cols].head(15).to_string(index=False))