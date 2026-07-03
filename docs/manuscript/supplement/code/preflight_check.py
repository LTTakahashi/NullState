"""Pre-flight verification for a clean full pilot run.

Run this on the pod BEFORE kicking off the ~4-5h rebuild+retrain. It is read-only
and confirms the assumptions the code cannot otherwise verify, so we don't discover
a blocking data issue 3 hours into training.

Usage:
    python scripts/preflight_check.py --config-dir config/

Exit code 0 = all green; 1 = at least one blocker. Read the printed report regardless.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import h5py
import yaml


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def _h5_index(group):
    idx_key = group.attrs.get("_index", "_index")
    if isinstance(idx_key, bytes):
        idx_key = idx_key.decode()
    return _decode(group[idx_key][:]), idx_key


def _h5_columns(group):
    _, idx_key = _h5_index(group)
    return [k for k in group.keys() if k != idx_key]


def _is_int_counts_sample(h5file, max_sample=300000):
    """Sample /X/data (CSR) and report integer-ness + max."""
    X = h5file["X"]
    if isinstance(X, h5py.Group) and "data" in X:
        data = X["data"]
        n = min(max_sample, data.shape[0])
        d = data[:n].astype("float64")
        return {
            "sparse": True,
            "frac_integer": float(np.mean(d == np.round(d))),
            "max": float(d.max()) if d.size else 0.0,
            "min": float(d.min()) if d.size else 0.0,
        }
    return {"sparse": False, "frac_integer": None, "max": None, "min": None}


def check(label, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ->  {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="config/")
    args = ap.parse_args()

    paths = yaml.safe_load(open(Path(args.config_dir) / "paths.yaml"))
    params = yaml.safe_load(open(Path(args.config_dir) / "params.yaml"))
    ref_path = Path(paths["outputs"]["reference"])
    query_path = Path(paths["datasets"]["hnoca"]["local_path"])
    batch_key = params.get("batch_key")

    blockers = []
    print("=" * 70)
    print("NULLSTATE PRE-FLIGHT CHECK")
    print("=" * 70)

    # ---- 1. Reference -------------------------------------------------
    print("\n[1] REFERENCE", ref_path)
    if not ref_path.exists():
        print("  reference.h5ad not found -> Step 1 will build it. Re-run preflight after.")
    else:
        with h5py.File(ref_path, "r") as f:
            obs = f["obs"]
            cols = list(obs.keys())
            # batch key present + cardinality
            if batch_key:
                has_bk = batch_key in cols
                if has_bk:
                    node = obs[batch_key]
                    ncat = len(node["categories"]) if isinstance(node, h5py.Group) and "categories" in node else len(np.unique(node[:]))
                    ok = check(f"batch_key '{batch_key}' present in reference", True, f"{ncat} levels")
                    if ncat < 2:
                        blockers.append("batch_key has <2 levels -> no integration")
                else:
                    check(f"batch_key '{batch_key}' present in reference", False,
                          f"available: {[c for c in cols if c in ('donor_id','assay','ref_source','suspension_type')]}")
                    blockers.append(f"reference missing batch_key '{batch_key}'")
            # raw counts
            cnt = _is_int_counts_sample(f)
            ok = cnt["frac_integer"] is not None and cnt["frac_integer"] > 0.999
            check("reference X is raw integer counts", ok,
                  f"frac_int={cnt['frac_integer']} max={cnt['max']}")
            if not ok:
                blockers.append("reference X is not raw counts -> ZINB mis-specified")
            # origin coverage
            if "cell_type" in obs:
                ct = obs["cell_type"]
                cats = _decode(ct["categories"][:]) if isinstance(ct, h5py.Group) and "categories" in ct else list(np.unique(_decode(ct[:])))
                sys.path.append(str(Path(__file__).parent.parent))
                from src.utils.origin_map import get_default_origin_map, map_origin
                m = get_default_origin_map()
                unmapped = [c for c in cats if map_origin(c, m) == "unknown"]
                ok = len(unmapped) <= 1  # only generic 'cell' allowed
                check("origin map covers reference cell types", ok,
                      f"{len(cats)-len(unmapped)}/{len(cats)} mapped; unmapped={unmapped}")
                if not ok:
                    blockers.append(f"origin map unmapped types: {unmapped}")

    # ---- 2. Query -----------------------------------------------------
    print("\n[2] QUERY", query_path)
    if not query_path.exists():
        blockers.append("query h5ad not found")
        check("query file exists", False, str(query_path))
    else:
        with h5py.File(query_path, "r") as f:
            # gene namespace: an 'ensembl' var column OR var_names already Ensembl (CELLxGENE schema).
            var = f["var"]
            vcols = _h5_columns(var)
            idx_key = var.attrs.get("_index", "_index")
            if isinstance(idx_key, bytes):
                idx_key = idx_key.decode()
            try:
                vnames = np.asarray(var[idx_key][:500]).astype(str)
                frac_ensg = float(np.mean(np.char.startswith(vnames, "ENSG"))) if vnames.size else 0.0
            except Exception:
                frac_ensg = 0.0
            ensembl_ok = ("ensembl" in vcols) or (frac_ensg > 0.5)
            check("query genes Ensembl-resolvable ('ensembl' col or ENSG var_names)", ensembl_ok,
                  f"ensembl_col={'ensembl' in vcols} ENSG_var_names={frac_ensg:.2f}")
            if not ensembl_ok:
                blockers.append("query genes neither Ensembl-indexed nor have an 'ensembl' column -> remap impossible")

            # raw counts: an integer count layer, OR integer X, OR integer raw.X (CELLxGENE schema).
            layers = list(f["layers"].keys()) if "layers" in f else []
            count_layer = None
            for lk in ("counts", "counts_lengthnorm", "raw_counts", "umi_counts", "X_counts"):
                if lk in layers:
                    node = f["layers"][lk]
                    data = node["data"] if isinstance(node, h5py.Group) and "data" in node else node
                    d = np.asarray(data[:200000]).astype("float64")
                    if d.size and np.all(d >= 0) and np.allclose(d, np.round(d)):
                        count_layer = lk
                        break
            xcnt = _is_int_counts_sample(f)
            x_is_counts = xcnt["frac_integer"] is not None and xcnt["frac_integer"] > 0.999 and (xcnt["max"] or 0) > 30
            raw_is_counts = False
            if "raw" in f and "X" in f["raw"]:
                rx = f["raw"]["X"]
                rdata = rx["data"] if isinstance(rx, h5py.Group) and "data" in rx else rx
                try:
                    rd = np.asarray(rdata[:200000]).astype("float64")
                    raw_is_counts = bool(rd.size and np.all(rd >= 0) and np.allclose(rd, np.round(rd)) and (rd.max() > 30))
                except Exception:
                    raw_is_counts = False
            ok_counts = (count_layer is not None) or x_is_counts or raw_is_counts
            check("query raw counts available (count layer, X, or raw.X)", ok_counts,
                  f"count_layer={count_layer} X_frac_int={xcnt['frac_integer']} raw.X_counts={raw_is_counts}")
            if not ok_counts:
                blockers.append("query has no integer raw counts (layer/X/raw.X) -> ZINB mis-specified")
            elif raw_is_counts and not (count_layer or x_is_counts):
                print("       (counts will be promoted from raw.X by _use_query_raw_counts)")

            # batch column for scArches
            obs = f["obs"]
            ocols = list(obs.keys())
            if batch_key:
                cand = [c for c in (batch_key, 'donor_id','batch','sample','bio_sample','sample_id','dataset','donor') if c in ocols]
                check(f"query has a batch column for '{batch_key}'", len(cand) > 0,
                      f"candidates present: {cand}")
                if not cand:
                    print("       NOTE: will assign a single 'hnoca_query' batch (coarse but valid).")

            # protocol column for the count gate (must NOT be 'batch' -> 395 micro-batches make
            # GREEN unreachable; the CELLxGENE per-study axes are assay_differentiation/publication).
            prot = [c for c in ('assay_differentiation','publication','protocol','assay') if c in ocols]
            check("query has a protocol-like column for count gate (study-level)", len(prot) > 0,
                  f"candidates: {prot}")
            if not prot:
                print("       NOTE: count gate will collapse to 1 'unknown' protocol -> max YELLOW.")

    # ---- verdict ------------------------------------------------------
    print("\n" + "=" * 70)
    if blockers:
        print("RESULT: NOT GREEN -- blockers:")
        for b in blockers:
            print("   -", b)
        print("=" * 70)
        return 1
    print("RESULT: GREEN -- no blockers found. Safe to launch the full run.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
