import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import json
import logging
import yaml

def _subsample_idx(labels: np.ndarray, keep_types: set, max_per_type: int, seed: int = 0) -> np.ndarray:
    """Return sorted row indices: up to `max_per_type` cells per label in `keep_types`."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    chunks = []
    for t in keep_types:
        t_idx = np.where(labels == t)[0]
        if len(t_idx) > max_per_type:
            t_idx = rng.choice(t_idx, max_per_type, replace=False)
        chunks.append(t_idx)
    if not chunks:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(chunks))


def compute_dish_vector(query: ad.AnnData, ref: ad.AnnData, cell_type: str, query_class_column: str = 'pilot_class', ref_type_column: str = 'cell_type') -> np.ndarray:
    """Computes Δ_c = mean(organoid clean_ontarget c) - mean(primary c) in log-norm space."""
    # Filter query for clean on-target cells of this type
    q_mask = (query.obs[query_class_column] == 'clean_ontarget') & (query.obs['pred_label'] == cell_type)
    r_mask = ref.obs[ref_type_column] == cell_type
    
    q_cells = query[q_mask]
    r_cells = ref[r_mask]
    
    if q_cells.n_obs < 10 or r_cells.n_obs < 10:
        logging.warning(f"Insufficient cells for {cell_type} (query: {q_cells.n_obs}, ref: {r_cells.n_obs}). Skipping.")
        return None
        
    q_mean = np.asarray(q_cells.X.mean(axis=0)).ravel()
    r_mean = np.asarray(r_cells.X.mean(axis=0)).ravel()
    
    return q_mean - r_mean

def compute_cosine_matrix(dish_vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise cosine similarity between all Δ vectors."""
    types = list(dish_vectors.keys())
    n = len(types)
    matrix = np.zeros((n, n))
    
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types):
            v1, v2 = dish_vectors[t1], dish_vectors[t2]
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            matrix[i, j] = cos_sim
            
    return pd.DataFrame(matrix, index=types, columns=types)

def decompose_by_gene_sets(dish_vectors: dict[str, np.ndarray], var_names: pd.Index, gene_sets: dict[str, set]) -> dict[str, pd.DataFrame]:
    """Computes pairwise cosine similarity restricted to specific gene sets."""
    results = {}
    
    for set_name, genes in gene_sets.items():
        # Find intersection with var_names
        mask = var_names.isin(genes)
        if not mask.any():
            logging.warning(f"No genes found for set {set_name}.")
            continue
            
        subset_vectors = {t: v[mask] for t, v in dish_vectors.items()}
        results[set_name] = compute_cosine_matrix(subset_vectors)
        
    return results

def evaluate_go_nogo(cosine_matrix: pd.DataFrame, threshold: float = 0.70) -> dict:
    """Evaluates the GO/NO-GO gate for single z_iv."""
    # Extract upper triangle excluding diagonal
    upper_tri = cosine_matrix.where(np.triu(np.ones(cosine_matrix.shape), k=1).astype(bool))
    mean_cos = upper_tri.mean().mean()
    
    decision = "GO" if mean_cos > threshold else "NO-GO"
    
    return {
        "gate": "Dish-Vector Test",
        "decision": decision,
        "mean_cosine": float(mean_cos),
        "threshold": threshold,
        "explanation": "Single frozen z_iv justified." if decision == "GO" else "Conditional z_iv | y_id needed.",
        "pairwise": upper_tri.stack().to_dict()
    }

def run_dish_vector_test(query: ad.AnnData, ref: ad.AnnData, gene_sets: dict[str, set], config_path: str = 'config/params.yaml') -> dict:
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
        
    threshold = params.get('cosine_go_threshold', 0.70)

    # MEMORY: only clean_ontarget query cells contribute dish vectors, so subset to them
    # BEFORE any genome-wide copy. Operating on the full 1.77M-cell query x 28k genes (with
    # the repeated .copy() below) transiently doubles memory past the 251 GB container cap.
    query = query[query.obs['pilot_class'] == 'clean_ontarget'].copy()
    logging.info(f"Dish vectors: restricted query to {query.n_obs} clean_ontarget cells.")

    # The annotated query is indexed by gene symbol while the reference uses Ensembl
    # IDs. Remap the query to Ensembl so gene matching recovers ~74% of genes instead
    # of the ~15% that match by symbol coincidence. (Mirrors the model-side remap.)
    if 'ensembl' in query.var.columns:
        ens = query.var['ensembl'].astype(str)
        keep = ens.str.startswith('ENSG').values
        query = query[:, keep].copy()
        query.var_names = query.var['ensembl'].astype(str).values
        query = query[:, ~query.var_names.duplicated()].copy()
        logging.info(f"Remapped query to Ensembl IDs for dish vectors ({query.n_vars} genes).")

    # Identify common types
    q_types = set(query.obs['pred_label'].unique())
    r_types = set(ref.obs['cell_type'].unique())
    common_types = q_types.intersection(r_types)

    # MEMORY: a dish vector is a per-type MEAN, which is statistically stable from a few
    # thousand cells. Subsample up to `max_per_type` cells per common type (selecting
    # indices directly from the full objects, so no giant intermediate copy is made).
    # The 8 common types are the dominant neural types (~1.6M ref cells), so this is the
    # difference between ~250 GB and a few GB.
    max_per_type = int(params.get('dish_max_cells_per_type', 3000))
    ref = ref[_subsample_idx(ref.obs['cell_type'].values, common_types, max_per_type)].copy()
    query = query[_subsample_idx(query.obs['pred_label'].values, common_types, max_per_type)].copy()
    logging.info(f"Dish vectors: subsampled to <= {max_per_type} cells/type "
                 f"-> ref {ref.n_obs}, query {query.n_obs} cells across {len(common_types)} types.")

    common_vars = query.var_names.intersection(ref.var_names)
    logging.info(f"Subsetting query and ref to {len(common_vars)} common genes for dish vectors...")
    query = query[:, common_vars].copy()
    ref = ref[:, common_vars].copy()

    # IMPORTANT: dish vectors are a mean-difference, so query and ref MUST be in the
    # same expression space. The reference X is raw counts; normalize it to log1p-CP10k
    # to match the (log-normalized) query. This assumes the query is CP10k-log1p — verify
    # the query's normalization before trusting the GO/NO-GO decision.
    import scipy.sparse as sp
    ref_is_counts = bool(sp.issparse(ref.X)) and float(ref.X.max()) > 50
    if ref_is_counts:
        logging.warning("Reference X is raw counts; applying log1p-CP10k to match query space "
                        "for dish vectors. Confirm the query uses the same normalization.")
        sc = __import__('scanpy')
        sc.pp.normalize_total(ref, target_sum=1e4)
        sc.pp.log1p(ref)
    
    logging.info(f"Computing dish vectors for {len(common_types)} common types...")
    dish_vectors = {}
    for t in common_types:
        dv = compute_dish_vector(query, ref, t)
        if dv is not None:
            dish_vectors[t] = dv
            
    if not dish_vectors:
        return {"error": "No valid dish vectors could be computed."}
        
    cos_matrix = compute_cosine_matrix(dish_vectors)
    gate_result = evaluate_go_nogo(cos_matrix, threshold)
    
    decomp = decompose_by_gene_sets(dish_vectors, query.var_names, gene_sets)
    
    return {
        "gate_result": gate_result,
        "cosine_matrix": cos_matrix,
        "decomposition": decomp
    }

def save_results(results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "error" in results:
        with open(output_dir / "error.json", 'w') as f:
            json.dump(results, f)
        return
        
    results["cosine_matrix"].to_csv(output_dir / "cosine_matrix.csv")
    
    with open(output_dir / "gate_decision.json", 'w') as f:
        # Convert tuple keys to string for JSON serialization
        gate_res = results["gate_result"].copy()
        if 'pairwise' in gate_res:
            gate_res['pairwise'] = {str(k): v for k, v in gate_res['pairwise'].items()}
        json.dump(gate_res, f, indent=2)
        
    for set_name, matrix in results["decomposition"].items():
        matrix.to_csv(output_dir / f"cosine_matrix_{set_name}.csv")
