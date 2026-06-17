"""Lightweight QC for the pilot run.

Two checks that turn "we applied integration / produced labels" into "we confirmed
integration helped / flagged where labels are shaky", with no heavy dependencies
(sklearn + scanpy only):

1. evaluate_integration  -> Aim-1 harmonization check. Compares the SCVI latent
   against a raw-PCA baseline using kNN inverse-Simpson (LISI-style) scores:
   does the latent MIX batches (donors) while KEEPING cell types separable?

2. label_confidence_report -> per-predicted-type mapping-entropy summary, so you can
   see which predicted labels the classifier is confident about vs guessing.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def _knn_inverse_simpson(Z: np.ndarray, labels, n_neighbors: int = 90, seed: int = 0) -> float:
    """Mean effective number of distinct `labels` among each cell's kNN (LISI-style).

    Higher means more label diversity locally. For batch labels: higher = better
    mixing. For cell-type labels: ~1 means each neighborhood is one type (good
    biological separation). This is a kNN approximation of Korsunsky 2019 LISI
    (no Gaussian perplexity kernel), which is sufficient for a relative QC.
    """
    from sklearn.neighbors import NearestNeighbors

    codes, uniq = pd.factorize(np.asarray(labels))
    k = int(min(n_neighbors, Z.shape[0]))
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    _, knn = nn.kneighbors(Z)
    neigh_codes = codes[knn]                       # (n, k)
    n_lab = len(uniq)
    # per-cell Simpson index = sum_l p_l^2 ; effective number = 1 / Simpson
    onehot_counts = np.zeros((Z.shape[0], n_lab), dtype=np.float64)
    for j in range(n_lab):
        onehot_counts[:, j] = (neigh_codes == j).sum(axis=1)
    p = onehot_counts / onehot_counts.sum(axis=1, keepdims=True)
    simpson = np.sum(p ** 2, axis=1)
    return float(np.mean(1.0 / simpson))


def evaluate_integration(ref_model, adata, batch_key: str, label_key: str = "cell_type",
                         sample: int = 20000, n_neighbors: int = 90, seed: int = 0,
                         out_path=None) -> dict:
    """Confirm Aim-1 harmonization on a subsample of the reference.

    Reports batch-iLISI (want HIGHER than the unintegrated PCA baseline -> donors
    mixed) and label-cLISI (want it to stay ~as low as PCA -> cell types preserved).
    Because cell type is partly confounded with donor in this reference, absolute
    iLISI is bounded; the meaningful signal is the integrated-vs-PCA *delta*.
    """
    import scanpy as sc

    if not batch_key or batch_key not in adata.obs.columns:
        logging.warning(f"[Integration QC] batch_key '{batch_key}' unavailable; skipping.")
        return {"skipped": True, "reason": f"batch_key '{batch_key}' not in obs"}

    rng = np.random.default_rng(seed)
    n = adata.n_obs
    idx = np.sort(rng.choice(n, int(min(sample, n)), replace=False))
    sub = adata[idx].copy()
    batch = sub.obs[batch_key].astype(str).values
    label = sub.obs[label_key].astype(str).values

    # Integrated latent: encode ONLY the subsample (~80x cheaper than a full-data
    # forward pass over the whole reference). get_latent_representation preserves
    # sub's row order, which already equals adata[idx], so no re-slicing is needed.
    logging.info(f"[Integration QC] evaluating {len(idx)} cells ({sub.obs[batch_key].nunique()} batches)...")
    Z_int = ref_model.get_latent_representation(sub)

    # Unintegrated baseline: standard log1p-CP10k -> scale -> PCA on the same cells
    base = sub.copy()
    sc.pp.normalize_total(base, target_sum=1e4)
    sc.pp.log1p(base)
    sc.pp.scale(base, max_value=10)
    sc.pp.pca(base, n_comps=int(min(30, base.n_vars - 1)))
    Z_pca = base.obsm["X_pca"]

    res = {"n_cells_evaluated": int(len(idx)), "batch_key": batch_key,
           "n_batches": int(pd.Series(batch).nunique()), "n_labels": int(pd.Series(label).nunique())}
    for name, Z in (("integrated_scvi", Z_int), ("unintegrated_pca", Z_pca)):
        res[name] = {
            "batch_iLISI": _knn_inverse_simpson(Z, batch, n_neighbors, seed),
            "label_cLISI": _knn_inverse_simpson(Z, label, n_neighbors, seed),
        }

    bi, bp = res["integrated_scvi"]["batch_iLISI"], res["unintegrated_pca"]["batch_iLISI"]
    ci, cp = res["integrated_scvi"]["label_cLISI"], res["unintegrated_pca"]["label_cLISI"]
    res["interpretation"] = {
        "batch_mixing_improved": bool(bi > bp),
        "batch_iLISI_gain": float(bi - bp),
        "bio_conservation_preserved": bool(ci <= cp + 0.5),
        "label_cLISI_inflation": float(ci - cp),
    }
    logging.info(f"[Integration QC] batch iLISI integrated={bi:.2f} vs pca={bp:.2f} "
                 f"(higher=better mixing; max possible={res['n_batches']})")
    logging.info(f"[Integration QC] label cLISI integrated={ci:.2f} vs pca={cp:.2f} "
                 f"(~1 = cell types stay separable)")
    logging.info(f"[Integration QC] VERDICT: mixing_improved={res['interpretation']['batch_mixing_improved']}, "
                 f"bio_preserved={res['interpretation']['bio_conservation_preserved']}")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        logging.info(f"[Integration QC] saved -> {out_path}")
    return res


def label_confidence_report(query_obs: pd.DataFrame, soft: pd.DataFrame = None,
                            pred_label_col: str = "pred_label", entropy_col: str = "map_entropy",
                            tau_H: float = None, out_path=None) -> pd.DataFrame:
    """Per-predicted-type confidence: where is the SCANVI classifier guessing?

    High mean mapping-entropy (and low mean top soft-probability) => the classifier
    is uncertain for that predicted type; treat those calls with caution.
    """
    if pred_label_col not in query_obs or entropy_col not in query_obs:
        logging.warning("[Label confidence] required columns missing; skipping.")
        return pd.DataFrame()

    df = query_obs[[pred_label_col, entropy_col]].dropna(subset=[entropy_col]).copy()
    g = df.groupby(pred_label_col, observed=True)[entropy_col]
    rep = pd.DataFrame({"n_cells": g.size(), "mean_entropy": g.mean(), "median_entropy": g.median()})
    if tau_H is not None:
        rep["frac_above_tau_H"] = df.groupby(pred_label_col, observed=True)[entropy_col].apply(
            lambda s: float((s > tau_H).mean()))
    if soft is not None:
        top = soft.max(axis=1).rename("top_prob")
        tmp = pd.concat([query_obs[pred_label_col].reindex(top.index), top], axis=1).dropna()
        rep["mean_top_prob"] = tmp.groupby(pred_label_col, observed=True)["top_prob"].mean()

    rep = rep.sort_values("mean_entropy", ascending=False)
    logging.info("[Label confidence] most-uncertain predicted types (high entropy = guessing):")
    for lab, row in rep.head(8).iterrows():
        extra = f" top_prob={row['mean_top_prob']:.2f}" if "mean_top_prob" in rep.columns else ""
        logging.info(f"   {lab}: n={int(row['n_cells'])} mean_H={row['mean_entropy']:.3f}{extra}")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(out_path)
        logging.info(f"[Label confidence] saved -> {out_path}")
    return rep
