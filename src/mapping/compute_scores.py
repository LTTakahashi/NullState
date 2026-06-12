import numpy as np
import pandas as pd
import scipy.stats
from sklearn.neighbors import NearestNeighbors
import anndata as ad
import scvi
import json
from pathlib import Path
import logging

def build_empirical_label_prior(labels: pd.Series) -> pd.Series:
    """Empirical (non-uniform) class prior from reference label frequencies.

    Returns a Series indexed by label name summing to 1, used to correct scANVI's
    uniform-prior posterior toward the true class frequencies (Phase 1 / WS0b) so that
    rare types are not over-predicted. Labels absent here are floored (never zeroed)
    when the prior is applied; see :func:`_apply_label_prior`.
    """
    return labels.astype(str).value_counts(normalize=True)


def _apply_label_prior(soft: pd.DataFrame, class_prior: pd.Series) -> pd.DataFrame:
    """Reweight soft posteriors by an empirical prior and renormalize per cell.

    posterior_corrected proportional to posterior_uniform * prior (logit-adjustment).
    Any label column missing from the prior gets the smallest observed prior as a
    floor (down-weighted, not dropped); each row is renormalized to sum to 1.
    """
    floor = float(class_prior[class_prior > 0].min()) if (class_prior > 0).any() else 1e-6
    prior = class_prior.reindex(soft.columns).fillna(floor).clip(lower=floor)
    weighted = soft.mul(prior.values, axis=1)
    row_sums = weighted.sum(axis=1).replace(0, np.nan)
    corrected = weighted.div(row_sums, axis=0).fillna(1.0 / soft.shape[1])
    return corrected


def compute_mapping_entropy(query_model: scvi.model.SCANVI, query: ad.AnnData,
                            class_prior: pd.Series = None) -> np.ndarray:
    """Computes mapping entropy from soft label predictions.

    If ``class_prior`` is provided (Phase 1 / WS0b empirical-prior correction), the soft
    posteriors are reweighted by it and renormalized before entropy is computed, so the
    same correction flows into both threshold calibration and query scoring. With
    ``class_prior=None`` the behavior is identical to the validated uniform-prior pilot.
    """
    logging.info("Predicting soft labels for query...")
    soft = query_model.predict(query, soft=True)
    if class_prior is not None:
        logging.info("Applying empirical label-prior correction to soft predictions...")
        soft = _apply_label_prior(soft, class_prior)
    logging.info("Computing Shannon entropy...")
    entropy = soft.apply(lambda r: scipy.stats.entropy(r), axis=1).values
    return entropy, soft

def compute_offmanifold_score(ref_model: scvi.model.SCANVI, ref: ad.AnnData, query_model: scvi.model.SCANVI, query: ad.AnnData, k: int = 15) -> np.ndarray:
    """Computes off-manifold score as k-NN distance to reference latent space."""
    logging.info("Extracting latent representations...")
    Zref = ref_model.get_latent_representation(ref)
    Zq = query_model.get_latent_representation(query)
    
    logging.info(f"Fitting NearestNeighbors (k={k}) on reference latent...")
    nn = NearestNeighbors(n_neighbors=k).fit(Zref)
    
    logging.info("Calculating distances for query...")
    distances = nn.kneighbors(Zq)[0].mean(axis=1)
    return distances

def calibrate_thresholds(ref_model: scvi.model.SCANVI, ref: ad.AnnData, holdout_fraction: float = 0.10, tau_H_percentile: float = 95, tau_R_percentile: float = 99, k: int = 15, class_prior: pd.Series = None) -> dict:
    """Calibrates thresholds on a held-out reference set.

    ``class_prior`` (Phase 1 / WS0b) must be the same prior used for query scoring so
    that tau_H is calibrated on the identically-corrected entropy distribution; pass
    None to keep the validated uniform-prior calibration.
    """
    logging.info(f"Calibrating thresholds using {holdout_fraction*100}% of reference cells...")

    n_holdout = int(ref.n_obs * holdout_fraction)
    # Basic random holdout
    np.random.seed(42)
    holdout_indices = np.random.choice(ref.n_obs, n_holdout, replace=False)
    ref_holdout = ref[holdout_indices].copy()

    # Compute on holdout
    # We use the ref_model for both as this is purely reference data.
    entropy, _ = compute_mapping_entropy(ref_model, ref_holdout, class_prior=class_prior)
    offmanifold = compute_offmanifold_score(ref_model, ref, ref_model, ref_holdout, k=k)
    
    tau_H = np.percentile(entropy, tau_H_percentile)
    tau_R = np.percentile(offmanifold, tau_R_percentile)
    
    logging.info(f"Calibrated tau_H (entropy): {tau_H:.4f}")
    logging.info(f"Calibrated tau_R (off-manifold): {tau_R:.4f}")
    
    return {
        "tau_H": float(tau_H),
        "tau_R": float(tau_R),
        "tau_H_percentile": tau_H_percentile,
        "tau_R_percentile": tau_R_percentile
    }

def annotate_query(query: ad.AnnData, query_model: scvi.model.SCANVI, entropy: np.ndarray, offmanifold: np.ndarray, origin_map: dict, soft: pd.DataFrame) -> ad.AnnData:
    """Adds mapping entropy, off-manifold scores, and predicted origins to the query."""
    logging.info("Annotating query AnnData...")
    query.obs['map_entropy'] = pd.Series(entropy, index=soft.index)
    query.obs['offmanifold'] = pd.Series(offmanifold, index=soft.index)
    
    pred_label = soft.idxmax(axis=1)
    query.obs['pred_label'] = pred_label
    # Case-insensitive origin lookup so every predicted reference label resolves.
    from src.utils.origin_map import map_origin_series
    query.obs['pred_origin'] = map_origin_series(query.obs['pred_label'], origin_map, default='unknown')
    
    return query

def save_thresholds(thresholds: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    logging.info(f"Saved thresholds to {output_path}")
