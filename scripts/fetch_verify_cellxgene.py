#!/usr/bin/env python
"""Fetch a CELLxGENE dataset and VERIFY it carries integer raw counts (Phase 1 / WS0a step 1).

The pilot used the Zenodo `hnoca_cleanedmeta.h5ad`, which dropped raw counts (only log-norm +
length-normalized), so scArches surgery was disabled. The CELLxGENE Discover schema *requires*
raw counts, so the CELLxGENE copy should carry the integer UMIs we need. This script lists a
collection's datasets, downloads one, and checks where (if anywhere) integer counts live.

Defaults to the HNOCA collection; pass --collection-id b4d13dc2-9b75-401d-9d9a-6d1468c17d90 for
HEOCA. CPU/IO only.

Requirements:
    pip install requests anndata scipy numpy
List + download + verify:
    PYTHONPATH=. python scripts/fetch_verify_cellxgene.py --out-dir data/
Verify an existing file only:
    PYTHONPATH=. python scripts/fetch_verify_cellxgene.py --verify-only data/hnoca.h5ad

If the curation API shape has changed, the script prints what it received; you can also grab the
H5AD 'download' link from the dataset's CELLxGENE Discover page and pass it via --url.
"""

import argparse
import sys
from pathlib import Path

CURATION_API = "https://api.cellxgene.cziscience.com/curation/v1/collections/{cid}"
HNOCA_COLLECTION = "de379e5f-52d0-498c-9801-0f850823c847"


def list_datasets(collection_id: str):
    import requests
    url = CURATION_API.format(cid=collection_id)
    print(f"[api] GET {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    coll = r.json()
    datasets = coll.get("datasets", [])
    rows = []
    for d in datasets:
        h5ad = None
        for asset in d.get("assets", []):
            if str(asset.get("filetype", "")).upper() == "H5AD":
                h5ad = asset.get("url") or asset.get("presigned_url")
                break
        rows.append({
            "dataset_id": d.get("dataset_id") or d.get("id"),
            "title": d.get("title", ""),
            "cell_count": d.get("cell_count") or d.get("primary_cell_count"),
            "h5ad_url": h5ad,
        })
    return rows


def download(url: str, dest: Path, chunk=1 << 23):
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}\n        -> {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                f.write(c)
                done += len(c)
                if total:
                    print(f"\r        {done/1e9:.2f}/{total/1e9:.2f} GB", end="", flush=True)
        print()
    return dest


def verify_counts(h5ad_path: str) -> dict:
    """Report which layer (X / raw.X / layers[*]) holds integer raw counts."""
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp

    def is_integer(X) -> bool:
        d = X.data if sp.issparse(X) else np.asarray(X).ravel()
        if d.size == 0:
            return False
        s = d[:200000]
        return bool(np.all(s >= 0) and np.allclose(s, np.round(s)))

    print(f"[verify] backed read {h5ad_path}")
    a = ad.read_h5ad(h5ad_path, backed="r")
    report = {"path": h5ad_path, "n_obs": int(a.n_obs), "n_vars": int(a.n_vars), "integer_sources": []}
    # X
    try:
        if is_integer(a.X[:5000] if a.isbacked else a.X):
            report["integer_sources"].append("X")
    except Exception as e:  # noqa
        print(f"        (X check skipped: {e})")
    # raw.X
    try:
        if a.raw is not None and is_integer(a.raw.X[:5000]):
            report["integer_sources"].append("raw.X")
    except Exception as e:  # noqa
        print(f"        (raw.X check skipped: {e})")
    # layers
    for lk in list(getattr(a, "layers", {}).keys()):
        try:
            if is_integer(a.layers[lk][:5000]):
                report["integer_sources"].append(f"layers['{lk}']")
        except Exception:  # noqa
            pass
    try:
        a.file.close()
    except Exception:
        pass

    report["has_raw_counts"] = len(report["integer_sources"]) > 0
    print(f"[verify] n_obs={report['n_obs']} n_vars={report['n_vars']}")
    if report["has_raw_counts"]:
        print(f"[verify] ✅ integer raw counts found in: {', '.join(report['integer_sources'])}")
        print("        -> set params.yaml scarches_max_epochs: ~100 and re-run mapping "
              "(_use_query_raw_counts auto-detects integer count layers).")
    else:
        print("[verify] ⚠ no integer counts found in X/raw.X/layers. Try the full Zenodo record "
              "(14160929) or per-study GEO matrices (see docs/ws0a_data_sourcing.md).")
    return report


def main():
    ap = argparse.ArgumentParser(description="Fetch a CELLxGENE dataset and verify raw counts")
    ap.add_argument("--collection-id", default=HNOCA_COLLECTION)
    ap.add_argument("--out-dir", default="data/")
    ap.add_argument("--dataset-index", type=int, default=None,
                    help="which dataset asset to download (default: prompt/largest)")
    ap.add_argument("--url", help="direct H5AD download URL (skip the API)")
    ap.add_argument("--out-name", default="hnoca_cellxgene.h5ad")
    ap.add_argument("--verify-only", help="path to an existing .h5ad to verify (skip download)")
    args = ap.parse_args()

    if args.verify_only:
        verify_counts(args.verify_only)
        return

    out = Path(args.out_dir) / args.out_name
    if args.url:
        download(args.url, out)
        verify_counts(str(out))
        return

    rows = list_datasets(args.collection_id)
    if not rows:
        print("[api] no datasets parsed; response shape may have changed. "
              "Grab the H5AD link from the Discover page and pass --url.")
        sys.exit(1)
    print(f"[api] {len(rows)} dataset(s):")
    for i, r in enumerate(rows):
        print(f"  [{i}] {r['title'][:70]!r}  cells={r['cell_count']}  h5ad={'yes' if r['h5ad_url'] else 'NO'}")

    idx = args.dataset_index
    if idx is None:
        idx = max(range(len(rows)), key=lambda i: rows[i]["cell_count"] or 0)
        print(f"[api] no --dataset-index given; defaulting to the largest: [{idx}]")
    if not rows[idx]["h5ad_url"]:
        print(f"[api] dataset [{idx}] has no H5AD asset url; pass --url manually.")
        sys.exit(1)
    download(rows[idx]["h5ad_url"], out)
    verify_counts(str(out))


if __name__ == "__main__":
    main()
