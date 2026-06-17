#!/usr/bin/env python
"""WS1 production runner (Aim 2): train contrastiveVI on real data, export z_lin for WS2 (Aim 3).

  background = primary (reference) ON-TARGET cells  (origin in ONTARGET_ORIGINS[germ_layer])
  target     = organoid (HNOCA)    ON-TARGET cells  (pilot_class in {clean_ontarget, poorly_diff_ontarget})

After training, the background (shared / dish-stripped) latent z_lin is exported for the OFF-TARGET
organoid cells (pilot_class == 'true_offtarget') -- the populations Aim 3 compares -- with a matching
populations.csv (same row order) that scripts/run_aim3.py mode A ingests.

Counts: the reference carries integer counts in X; the annotated organoid file has NONE (raw.X was
stripped), so organoid counts are read from the CELLxGENE hnoca_cellxgene.h5ad raw.X and joined to
pilot_class/pred_label by position (the annotated file is derived from it without reordering; an
obs_names assertion guards that). Cells are subsampled (RAM/compute guard) before training.

    PYTHONPATH=. python scripts/run_ws1.py [--config-dir config/] [--population-key pred_label]
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))


def subsample_indices(mask, max_n, seed):
    """Positions where ``mask`` is True, randomly capped to ``max_n`` (None = keep all). Pure."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if max_n is not None and idx.size > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, int(max_n), replace=False))
    return idx


def select_ws1_cells(primary_origin, organoid_pilot_class, ontarget_origins,
                     ontarget_classes=("clean_ontarget", "poorly_diff_ontarget"),
                     offtarget_classes=("true_offtarget",),
                     max_primary=None, max_organoid_ontarget=None, max_offtarget=None, seed=0):
    """Pure selection of WS1 cell positions.

    Returns a dict of int-position arrays: ``primary_bg`` (primary on-target), ``organoid_tg``
    (organoid on-target = contrastiveVI target), ``organoid_off`` (off-target = z_lin export set).
    No scvi/anndata needed, so this is unit-testable on plain arrays.
    """
    po = pd.Series(list(primary_origin)).astype(str)
    pc = pd.Series(list(organoid_pilot_class)).astype(str)
    onto = set(map(str, ontarget_origins))
    return {
        "primary_bg": subsample_indices(po.isin(onto).to_numpy(), max_primary, seed),
        "organoid_tg": subsample_indices(pc.isin(set(ontarget_classes)).to_numpy(), max_organoid_ontarget, seed + 1),
        "organoid_off": subsample_indices(pc.isin(set(offtarget_classes)).to_numpy(), max_offtarget, seed + 2),
    }


def main():
    ap = argparse.ArgumentParser(description="WS1 contrastiveVI runner -> z_lin for WS2")
    ap.add_argument("--config-dir", default="config/")
    ap.add_argument("--population-key", default="pred_label",
                    help="organoid obs column used as the off-target population label for geometry")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    import yaml
    import anndata as ad
    from src.disentangle.contrastive import (
        build_contrastive_adata, select_contrastive_indices, train_contrastive_vi,
        get_lineage_latent, verify_disentanglement,
    )
    from src.mapping.classify import ONTARGET_ORIGINS

    paths = yaml.safe_load(open(Path(args.config_dir) / "paths.yaml"))
    params = yaml.safe_load(open(Path(args.config_dir) / "params.yaml"))
    out_dir = Path(paths.get("results_dir", "results/pilot/"))
    germ = params.get("expected_germ_layer", "neural")
    ontarget_origins = ONTARGET_ORIGINS.get(germ, ONTARGET_ORIGINS["neural"])
    pop_key = args.population_key

    # --- primary (reference): origin -> on-target; counts already in X ---
    ref_path = Path(paths["outputs"]["reference"])
    logging.info(f"Loading reference (primary, backed) {ref_path}")
    primary = ad.read_h5ad(ref_path, backed="r")  # backed: the full 2.6M ref would OOM a small box
    if "origin" not in primary.obs.columns:
        from src.utils.origin_map import get_default_origin_map, map_origin_series
        primary.obs["origin"] = map_origin_series(primary.obs["cell_type"], get_default_origin_map())

    # --- organoid labels from the annotated file (no counts there) ---
    annot_path = out_dir / "hnoca_annotated.h5ad"
    cellx_path = Path(paths["datasets"]["hnoca"]["local_path"])
    logging.info(f"Loading organoid labels {annot_path}")
    _a = ad.read_h5ad(annot_path, backed="r")
    obs = _a.obs.copy()
    try:
        _a.file.close()
    except Exception:
        pass
    if pop_key not in obs.columns:
        logging.warning(f"population key '{pop_key}' not in annotated obs; falling back to 'pred_origin'")
        pop_key = "pred_origin" if "pred_origin" in obs.columns else "pilot_class"

    # --- pure selection ---
    sel = select_ws1_cells(
        primary.obs["origin"], obs["pilot_class"], ontarget_origins,
        max_primary=params.get("contrastive_max_background", 100000),
        max_organoid_ontarget=params.get("contrastive_max_target", 100000),
        max_offtarget=params.get("contrastive_max_offtarget", 100000),
        seed=0,
    )
    logging.info(f"WS1 cells: primary_bg={sel['primary_bg'].size} "
                 f"organoid_tg={sel['organoid_tg'].size} organoid_off={sel['organoid_off'].size}")
    if sel["primary_bg"].size == 0 or sel["organoid_tg"].size == 0:
        raise SystemExit("empty background or target set -- check origin / pilot_class columns.")
    if sel["organoid_off"].size < 4:
        raise SystemExit(f"too few off-target cells ({sel['organoid_off'].size}) for geometry.")

    # --- organoid counts from the CELLxGENE file (raw.X), positions = on-target + off-target ---
    org_rows = np.union1d(sel["organoid_tg"], sel["organoid_off"])
    logging.info(f"Loading organoid counts for {org_rows.size} cells from {cellx_path}")
    _o = ad.read_h5ad(cellx_path, backed="r")
    organoid = _o[org_rows].to_memory()
    try:
        _o.file.close()
    except Exception:
        pass
    # guard: the annotated file is derived from cellx without reordering -> positions must agree
    assert list(map(str, organoid.obs_names)) == list(map(str, obs.index[org_rows])), \
        "organoid (CELLxGENE) and annotated obs_names diverge -- cannot align labels by position."
    organoid.obs["pilot_class"] = obs["pilot_class"].to_numpy()[org_rows]
    organoid.obs[pop_key] = obs[pop_key].to_numpy()[org_rows]
    # promote raw.X -> X (integer counts) so build_contrastive_adata materializes counts
    if organoid.raw is not None:
        rn, vn = list(map(str, organoid.raw.var_names)), list(map(str, organoid.var_names))
        organoid.X = (organoid.raw.X.copy() if rn == vn
                      else organoid.raw[:, organoid.var_names].X.copy())
        del organoid.raw
    org_ontarget = organoid.obs["pilot_class"].isin(["clean_ontarget", "poorly_diff_ontarget"]).to_numpy()
    org_offtarget = (organoid.obs["pilot_class"] == "true_offtarget").to_numpy()

    primary_bg = primary[sel["primary_bg"]].to_memory()  # load only the sampled on-target subset
    primary_bg.obs[pop_key] = primary_bg.obs.get("cell_type", pd.Series(["primary"] * primary_bg.n_obs))

    # --- build combined (B7 materializes counts), select bg/tg, train contrastiveVI ---
    combined = build_contrastive_adata(
        primary_bg, organoid,
        primary_ontarget=np.ones(primary_bg.n_obs, dtype=bool),   # all selected primary are on-target
        organoid_ontarget=org_ontarget,
    )
    # HVG-subset the counts for a tractable contrastiveVI input (full ~35k genes is impractical).
    import scanpy as sc
    n_hvg = params.get("contrastive_n_hvg", 2000)
    sc.pp.highly_variable_genes(combined, n_top_genes=n_hvg, flavor="seurat_v3", layer="counts", subset=True)
    logging.info(f"contrastiveVI input subset to {combined.n_vars} HVGs (counts layer)")
    is_primary = (combined.obs["cs_source"] == "primary").to_numpy()
    is_organoid = (combined.obs["cs_source"] == "organoid").to_numpy()
    is_ontarget = combined.obs["cs_ontarget"].to_numpy()
    bg, tg = select_contrastive_indices(is_primary, is_organoid, is_ontarget)

    # fix-2 (batch-aware): condition the decoder on a batch covariate so the source shift
    # (organoid vs primary) is absorbed there instead of forced into z_lin. Default 'cs_source'
    # (the primary/organoid axis the adversary tests); set contrastive_batch_key: null to disable.
    batch_key = params.get("contrastive_batch_key", "cs_source")
    if batch_key is not None and batch_key not in combined.obs.columns:
        logging.warning(f"contrastive_batch_key '{batch_key}' not in combined.obs; training without batch.")
        batch_key = None
    if batch_key is not None:
        logging.info(f"batch-aware ContrastiveVI: batch_key='{batch_key}' "
                     f"({pd.Series(combined.obs[batch_key]).value_counts().to_dict()})")
    model = train_contrastive_vi(
        combined, bg, tg,
        n_background_latent=params.get("contrastive_n_background_latent", 15),
        n_salient_latent=params.get("contrastive_n_salient_latent", 10),
        batch_key=batch_key,
        layer="counts",
        max_epochs=params.get("contrastive_max_epochs", 200),
        early_stopping=params.get("contrastive_early_stopping", True),
        early_stopping_patience=params.get("contrastive_early_stopping_patience", 15),
    )

    # --- z_lin for OFF-TARGET organoid cells (the rows the geometry will compare) ---
    n_p = primary_bg.n_obs
    off_rows = n_p + np.flatnonzero(org_offtarget)            # positions into combined
    z_off = np.asarray(get_lineage_latent(model, combined, indices=off_rows))
    pops = combined.obs[pop_key].to_numpy()[off_rows]
    logging.info(f"z_lin (off-target) {z_off.shape}; populations: "
                 f"{pd.Series(pops).value_counts().to_dict()}")

    # disentanglement verification on a balanced sample of bg+tg
    samp = np.r_[bg[:2000], tg[:2000]]
    rep = verify_disentanglement(np.asarray(get_lineage_latent(model, combined, indices=samp)),
                                 is_organoid[samp])
    logging.info(f"adversary balanced-acc predicting organoid-vs-primary from z_lin = "
                 f"{rep['adversary_balanced_accuracy_mean']:.3f} (~0.5 good; "
                 f"disentangled_heuristic={rep['disentangled_heuristic']})")

    # --- save artifacts (run_aim3 mode A: --zlin z_lin.npy --labels populations.csv) ---
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_lin.npy", z_off)
    pd.Series(pops, name="population").to_csv(out_dir / "populations.csv", index=False)
    model.save(str(out_dir / "contrastive_model"), overwrite=True)
    logging.info(f"WS1 done -> {out_dir/'z_lin.npy'} ({z_off.shape}), "
                 f"{out_dir/'populations.csv'}, {out_dir/'contrastive_model'}")


if __name__ == "__main__":
    main()
