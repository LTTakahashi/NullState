#!/usr/bin/env python
"""Find an INDEPENDENT developing-human-cortex reference in CELLxGENE Census for the
second-reference test (reviewer point 3). Must NOT be HDBCA (Braun/Linnarsson). Prints
candidate datasets with cell counts so we can pick one with raw counts + good coverage.
"""
import cellxgene_census
import pandas as pd

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 200)
CENSUS_VERSION = "2025-11-08"

with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
    ds = census["census_info"]["datasets"].read().concat().to_pandas()
    cols = [c for c in ("dataset_id", "collection_name", "dataset_title", "dataset_total_cell_count") if c in ds.columns]
    print("census columns:", list(ds.columns))
    kw = ds["dataset_title"].str.contains("cortic|cortex|prenatal|developing|fetal|neural|brain",
                                          case=False, na=False) | \
         ds["collection_name"].str.contains("cortic|cortex|prenatal|developing|fetal|brain",
                                             case=False, na=False)
    cand = ds[kw][cols].sort_values("dataset_total_cell_count", ascending=False)
    print("\n===== developing-brain/cortex candidates (top 40 by size) =====")
    print(cand.head(40).to_string())

    for name in ["Velmeshev", "Bhaduri", "Kriegstein", "arealization", "Eze",
                 "Nowakowski", "Trevino", "Herring", "first-trimester", "Braun"]:
        m = ds[ds["collection_name"].str.contains(name, case=False, na=False) |
               ds["dataset_title"].str.contains(name, case=False, na=False)]
        if len(m):
            print(f"\n===== match '{name}' =====")
            print(m[cols].to_string())
