import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_parquet("data/processed/events_ENG-Premier_League_2024-2025.parquet")

non_q = [c for c in df.columns if not c.startswith("q_")]
q_cols = sorted([c for c in df.columns if c.startswith("q_")])

print(f"=== 1. NON-QUALIFIER COLUMNS ({len(non_q)}) ===")
print(non_q)

print(f"\n=== 2. COORDINATE & PASS METRIC RANGES ===")
num_check = [c for c in ["x", "y", "end_x", "end_y", "pass_length", "pass_angle"] if c in df.columns]
print(df[num_check].describe().loc[["min", "mean", "max"]].round(2).to_string())

print(f"\n=== 3. EVENT TYPES & OUTCOME TYPES ===")
print(df.groupby(["type", "outcome_type"]).size().sort_values(ascending=False).to_string())

print(f"\n=== 4. ALL q_* QUALIFIER COLUMNS ({len(q_cols)}) ===")
print(q_cols)