#!/usr/bin/env python
"""Run the Aim 3 geometry end-to-end (Phase 1 WS2) -- the WS1 -> WS2 bridge.

Two ways to supply the disentangled lineage representation (z_lin):
  (A) precomputed:  --zlin z_lin.npy --labels labels.npy|labels.csv      (no scvi needed)
  (B) from a model: --model <contrastiveVI dir> --adata combined.h5ad --label-key population
                    (lazy-imports scvi; needs the WS1 contrastiveVI model from src.disentangle)

Then: rarefaction -> empirical N_min, all pairwise matched-N tests (+ optional Control-3
baselines via --control3 .npz), BH-FDR, results -> JSON.

SCAFFOLD: mode (B) is not yet runnable until WS1 trains a contrastiveVI model; mode (A) runs
today on any precomputed z_lin (e.g. to exercise the wiring on synthetic data).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from src.geometry.pipeline import run_geometry_over_populations


def _load_labels(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=True)
    import csv
    with open(p) as fh:
        rows = [r[0] for r in csv.reader(fh) if r]
    if rows and rows[0].lower() in ("label", "population", "pop"):
        rows = rows[1:]
    return np.array(rows)


def _zlin_from_model(model_dir: str, adata_path: str, label_key: str):
    import scvi  # lazy: only needed for model mode
    import anndata as ad
    from src.disentangle.contrastive import get_lineage_latent

    adata = ad.read_h5ad(adata_path)
    model = scvi.external.ContrastiveVI.load(str(model_dir), adata=adata)
    z = np.asarray(get_lineage_latent(model, adata))
    labels = np.asarray(adata.obs[label_key].values)
    return z, labels


def _json_safe(result: dict) -> dict:
    def clean(d: dict) -> dict:
        return {k: (None if v is None else (v.item() if isinstance(v, np.generic) else v))
                for k, v in d.items() if not isinstance(v, np.ndarray)}
    fdr = ({f"{a}|{b}": v for (a, b), v in result["fdr"].items()} if result["fdr"] else None)
    return {
        "n_min": result["n_min"],
        "sizes": result["sizes"],
        "fdr": fdr,
        "pairs": {f"{a}|{b}": clean(r) for (a, b), r in result["pairs"].items()},
    }


def main():
    ap = argparse.ArgumentParser(description="NullState Aim 3 geometry (WS2)")
    ap.add_argument("--config-dir", default="config/")
    ap.add_argument("--zlin", help="precomputed z_lin .npy (mode A)")
    ap.add_argument("--labels", help="per-cell population labels .npy/.csv (mode A)")
    ap.add_argument("--model", help="trained contrastiveVI model dir (mode B)")
    ap.add_argument("--adata", help="combined AnnData .h5ad (mode B)")
    ap.add_argument("--label-key", default="population", help="obs column with population labels (mode B)")
    ap.add_argument("--control3", help="optional .npz of maturation baselines, keys 'A|B'")
    ap.add_argument("--out", default="results/pilot/aim3_geometry.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    params = yaml.safe_load(open(Path(args.config_dir) / "params.yaml"))

    if args.zlin:
        z = np.load(args.zlin)
        labels = _load_labels(args.labels)
    elif args.model:
        z, labels = _zlin_from_model(args.model, args.adata, args.label_key)
    else:
        raise SystemExit("provide either --zlin/--labels (mode A) or --model/--adata (mode B)")

    control3 = None
    if args.control3:
        npz = np.load(args.control3)
        control3 = {frozenset(k.split("|")): npz[k] for k in npz.files}

    result = run_geometry_over_populations(
        z, labels,
        ladder=params.get("rarefaction_ladder"),
        control3=control3,
        n_projections=params.get("geometry_n_projections", 200),
        n_boot=params.get("geometry_n_boot", 1000),
        n_repeats=params.get("geometry_n_self_repeats", 200),
        floor_quantile=params.get("geometry_floor_quantile", 95),
        n_permutations=params.get("geometry_n_permutations", 0),
        random_state=0,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(_json_safe(result), fh, indent=2)
    logging.info("N_min=%s; wrote Aim 3 geometry -> %s", result["n_min"], out)
    for (a, b), r in result["pairs"].items():
        logging.info("  %s vs %s -> %s (N*=%s, cross=%.4g, floor_q=%.4g)",
                     a, b, r["verdict"], r["n_star"], r["cross_mean"],
                     r.get(f"floor_q{int(params.get('geometry_floor_quantile', 95))}", float("nan")))


if __name__ == "__main__":
    main()
