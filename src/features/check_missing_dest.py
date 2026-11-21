import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

master = pd.read_parquet("data/processed/master_transfer_dataset.parquet")
clubs = pd.read_parquet("data/processed/club_style_features.parquet")

c24_teams = set(clubs.loc[clubs["season"] == 2425, "team"])
c25_teams = set(clubs.loc[clubs["season"] == 2526, "team"])
promoted_2526 = sorted(c25_teams - c24_teams)
relegated_2425 = sorted(c24_teams - c25_teams)

print(f"=== 1. PROMOTED / RELEGATED CLUBS BETWEEN 2425 & 2526 ===")
print(f"Promoted into top-5 in 2526 ({len(promoted_2526)}): {promoted_2526}")
print(f"Relegated after 2425 ({len(relegated_2425)}):       {relegated_2425}")

elig_out = master[(master["elig_status"] == "eligible") & (master["position"] != "Goalkeeper")].copy()
miss_dest = elig_out[elig_out["dest_c24_poss_share"].isna()].copy()

print(f"\n=== 2. ELIGIBLE OUTFIELD ROWS MISSING dest_c24_* ({len(miss_dest)} / {len(elig_out)}) ===")
print(miss_dest[["player_name", "position", "tm_from_club", "tm_to_club", "ws_to_club", "transfer_fee"]].to_string(index=False))

non_promoted_missing = miss_dest[~miss_dest["ws_to_club"].isin(promoted_2526)]
print(f"\nNon-promoted clubs missing dest_c24_*: {len(non_promoted_missing)} (must be 0)")

complete_out = elig_out[elig_out["dest_c24_poss_share"].notna()]
print(f"\n=== 3. COMPLETE ELIGIBLE OUTFIELD TRANSFERS (BOTH PLAYER & DEST CLUB IN 2425) ===")
print(f"Total complete outfield transfers: {len(complete_out)} (with Y_score: {complete_out['Y_score'].notna().sum()})")
print(complete_out["position"].value_counts().to_string())