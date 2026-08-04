"""Merge the base 5-seed results_stage4.csv with the sharded expansion CSVs
(results_stage4_s*.csv) into a single de-duplicated 25-seed table."""
import sys as _sys, pathlib as _pathlib
import glob
import pandas as pd
from core.paths import result, RESULTS


def main():

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

    frames = [pd.read_csv(result("main_sweep_pilot.csv"))]
    for f in sorted(glob.glob(str(RESULTS / "main_sweep_s*.csv"))):
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["arm", "rho", "delta", "seed"], keep="first")
    df = df.sort_values(["arm", "delta", "rho", "seed"]).reset_index(drop=True)
    df.to_csv(result("main_sweep.csv"), index=False)
    print(f"merged {before} -> {len(df)} unique rows")
    print("seeds per (arm,rho,delta) cell:")
    print(df.groupby(["arm", "delta", "rho"]).seed.nunique().describe()[["min", "max"]].to_string())
    print("\nby arm:", df.arm.value_counts().to_dict())
    print("distinct seeds:", sorted(df.seed.unique()))


if __name__ == "__main__":
    main()
