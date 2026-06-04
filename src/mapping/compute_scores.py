import numpy as np
import pandas as pd
import scipy.stats
from sklearn.neighbors import NearestNeighbors
import anndata as ad
import scvi
import json
from pathlib import Path
import logging

def compute_mapping_entropy(query_model: scvi.model.SCANVI, query: ad.AnnData) -> np.ndarray:
    """Computes mapping entropy from soft label predictions."""
    logging.info("Predicting soft labels for query...")
    soft = query_model.predict(query, soft=True)
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

def calibrate_thresholds(ref_model: scvi.model.SCANVI, ref: ad.AnnData, holdout_fraction: float = 0.10, tau_H_percentile: float = 95, tau_R_percentile: float = 99, k: int = 15) -> dict:
    """Calibrates thresholds on a held-out reference set."""
    logging.info(f"Calibrating thresholds using {holdout_fraction*100}% of reference cells...")
    
    n_holdout = int(ref.n_obs * holdout_fraction)
    # Basic random holdout
    np.random.seed(42)
    holdout_indices = np.random.choice(ref.n_obs, n_holdout, replace=False)
    ref_holdout = ref[holdout_indices].copy()
    
    # Compute on holdout
    # We use the ref_model for both as this is purely reference data.
    entropy, _ = compute_mapping_entropy(ref_model, ref_holdout)
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
