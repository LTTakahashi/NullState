#!/usr/bin/env python
"""geosketch subsample of an atlas (Phase 1 / WS4 scale lever).

Geometric sketching (Hie et al. 2019) picks a subsample that preserves transcriptomic
heterogeneity -- covering rare populations far better than uniform sampling -- so the heavy
scVI/scANVI/contrastiveVI training can run on ~100-200k representative cells instead of ~1.9M.
That can shrink the GPU run to a small/mid GPU and the RAM to laptop scale.

geosketch operates on a low-dim embedding: pass one via --use-rep (e.g. an existing X_pca or a
trained X_scVI) to skip PCA, otherwise the script computes PCA. Stratify per-protocol/germ-layer
with --within to keep each group represented.

Requirements:
    pip install geosketch scanpy anndata numpy
Run:
    PYTHONPATH=. python scripts/geosketch_subsample.py --input data/reference.h5ad \
        --output data/reference_sketch150k.h5ad --n-cells 150000
With a precomputed embedding (no PCA, low RAM):
    ... --use-rep X_scVI
Stratified (keep every protocol):
    ... --within protocol

NOTE: computing PCA over the full atlas is RAM-heavy (1.9M x HVGs); use --use-rep on a precomputed
embedding, or run on a high-RAM box, if you OOM. This script is data-bound and untested by the assistant.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def _embedding(adata, use_rep, n_pcs, normalize):
    import scanpy as sc
    if use_rep:
        if use_rep not in adata.obsm:
            raise KeyError(f"--use-rep '{use_rep}' not in adata.obsm ({list(adata.obsm)})")
        print(f"[embed] using existing obsm['{use_rep}'] {adata.obsm[use_rep].shape}")
        return np.asarray(adata.obsm[use_rep])
    print(f"[embed] computing PCA (n_pcs={n_pcs}; normalize={normalize}) ...")
    work = adata.copy()
    if normalize:
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
    sc.pp.scale(work, max_value=10)
    sc.tl.pca(work, n_comps=n_pcs)
    return np.asarray(work.obsm["X_pca"])


def _sketch(X, n, seed):
    from geosketch import gs
    n = min(n, X.shape[0])
    idx = gs(X, n, replace=False, seed=seed)
    return np.sort(np.asarray(idx, dtype=int))


def main():
    ap = argparse.ArgumentParser(description="geosketch subsample of an atlas")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-cells", type=int, default=150000)
    ap.add_argument("--use-rep", default=None, help="obsm key of a precomputed embedding (skips PCA)")
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--no-normalize", action="store_true", help="skip normalize+log before PCA (X already log)")
    ap.add_argument("--within", default=None, help="obs column to stratify the sketch within (e.g. protocol)")
    ap.add_argument("--backed", action="store_true",
                    help="low-RAM mode: read backed (REQUIRES --use-rep; cannot PCA backed). Fits a "
                         "16 GB pod for the full atlas. From-scratch PCA needs a high-RAM box.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import anndata as ad

    if args.backed:
        if not args.use_rep:
            raise SystemExit("--backed requires --use-rep (a precomputed embedding in obsm); computing "
                             "PCA needs the full matrix in RAM -- run that on a high-RAM box (e.g. the GPU pod).")
        print(f"[load] {args.input} (backed='r', low-RAM)")
        adata = ad.read_h5ad(args.input, backed="r")
    else:
        print(f"[load] {args.input}")
        adata = ad.read_h5ad(args.input)
    print(f"[load] {adata.shape}")
    X = _embedding(adata, args.use_rep, args.n_pcs, normalize=not args.no_normalize)

    if args.within and args.within in adata.obs:
        groups = adata.obs[args.within].astype(str).to_numpy()
        uniq = np.unique(groups)
        per = max(1, args.n_cells // len(uniq))
        print(f"[sketch] stratified within '{args.within}': {len(uniq)} groups, ~{per} cells each")
        keep = []
        for grp in uniq:
            gidx = np.where(groups == grp)[0]
            sub = _sketch(X[gidx], min(per, len(gidx)), args.seed)
            keep.append(gidx[sub])
        idx = np.sort(np.concatenate(keep))
    else:
        print(f"[sketch] geosketch to {args.n_cells} cells ...")
        idx = _sketch(X, args.n_cells, args.seed)

    out = adata[idx]
    out = out.to_memory() if args.backed else out.copy()   # backed: load ONLY the selected cells
    print(f"[sketch] selected {out.n_obs} / {adata.n_obs} cells "
          f"({100*out.n_obs/adata.n_obs:.1f}%)")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.output, compression="gzip")
    print(f"[done] wrote {args.output}")
    # quick heterogeneity sanity: did we keep all cell types / protocols?
    for col in ("cell_type", "protocol", args.within or ""):
        if col and col in adata.obs:
            kept = out.obs[col].nunique()
            tot = adata.obs[col].nunique()
            print(f"        {col}: kept {kept}/{tot} categories")


if __name__ == "__main__":
    main()
