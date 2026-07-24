"""Merge the base 5-seed results_stage4.csv with the sharded expansion CSVs
(results_stage4_s*.csv) into a single de-duplicated 25-seed table."""
import glob
import pandas as pd

frames = [pd.read_csv("results_stage4.csv")]
for f in sorted(glob.glob("results_stage4_s*.csv")):
    frames.append(pd.read_csv(f))
df = pd.concat(frames, ignore_index=True)
before = len(df)
df = df.drop_duplicates(subset=["arm", "rho", "delta", "seed"], keep="first")
df = df.sort_values(["arm", "delta", "rho", "seed"]).reset_index(drop=True)
df.to_csv("results_stage4_n25.csv", index=False)
print(f"merged {before} -> {len(df)} unique rows")
print("seeds per (arm,rho,delta) cell:")
print(df.groupby(["arm", "delta", "rho"]).seed.nunique().describe()[["min", "max"]].to_string())
print("\nby arm:", df.arm.value_counts().to_dict())
print("distinct seeds:", sorted(df.seed.unique()))
