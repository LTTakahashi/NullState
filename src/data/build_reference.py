import scanpy as sc
import numpy as np
import scipy.sparse as sp
import pandas as pd
from pathlib import Path
import yaml
import logging
import gc


def load_config(config_path: str = 'config/paths.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _frac_integer(X, n: int = 200000) -> float:
    """Fraction of (a sample of) nonzero entries that are integers."""
    d = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if d.size == 0:
        return 1.0
    s = d[:n]
    return float(np.mean(s == np.round(s)))


def _ensure_ensembl_var_names(adata, name: str = ""):
    """Ensure var_names are Ensembl IDs.

    Census ``get_anndata`` indexes var by integer position ('0','1',...) and stores the Ensembl
    IDs in ``var['feature_id']``. Promote that column to var_names so the reference aligns with
    the Ensembl-indexed CELLxGENE query (otherwise the ref<->query gene intersection is empty).
    """
    frac = float(pd.Index(adata.var_names).astype(str).str.startswith('ENSG').mean())
    if frac >= 0.9:
        return adata
    if 'feature_id' in adata.var.columns and \
       float(adata.var['feature_id'].astype(str).str.startswith('ENSG').mean()) >= 0.9:
        adata.var_names = adata.var['feature_id'].astype(str).values
        adata.var_names_make_unique()
        logging.info(f"[{name}] set var_names from var['feature_id'] (Census -> Ensembl).")
        return adata
    raise ValueError(f"[{name}] var_names are not Ensembl and no usable 'feature_id' column found.")


def _ensure_counts_in_X(adata, name: str = ""):
    """Guarantee ``adata.X`` holds integer raw counts.

    The Census SOMA path (``cellxgene_census.get_anndata``) already returns raw counts in X.
    A CELLxGENE *h5ad* download instead carries log-norm X + integer counts in ``raw.X``;
    in that case we promote ``raw.X`` (aligned to the current var axis) into X. seurat_v3 HVG
    selection and the SCVI/SCANVI NB/ZINB likelihood are mis-specified on anything else, so we
    raise rather than silently train on normalized data.
    """
    if _frac_integer(adata.X) > 0.999 and float(adata.X.max()) > 30:
        return adata
    raw = getattr(adata, 'raw', None)
    if raw is not None:
        raw_names = list(map(str, raw.var_names))
        if raw_names == list(map(str, adata.var_names)):
            raw_X = raw.X
        else:
            pos = {g: i for i, g in enumerate(raw_names)}
            cols = [pos[g] for g in map(str, adata.var_names) if g in pos]
            raw_X = raw.X[:, cols] if len(cols) == adata.n_vars else None
        if raw_X is not None and _frac_integer(raw_X) > 0.999:
            adata.X = raw_X.copy()
            logging.info(f"[{name}] promoted raw.X -> X (integer counts).")
            return adata
    raise ValueError(
        f"[{name}] X is not integer counts (frac_int={_frac_integer(adata.X):.3f}, "
        f"max={float(adata.X.max()):.2f}) and no usable raw.X; seurat_v3 HVG + SCVI ZINB "
        f"require raw counts. Re-fetch via the Census SOMA path (raw X) or supply raw.X."
    )


def load_and_subset_fetal(fetal_path: Path, origin_map: dict,
                          keep_origins=('mesoderm', 'endoderm', 'neural_crest'),
                          cell_type_col: str = 'cell_type', max_cells: int = None,
                          seed: int = 0):
    """Load the fetal atlas (Cao) and subset to the NON-neural off-target anchors.

    The reference's neural core comes from HDBCA; Cao supplies the mesodermal / endodermal /
    neural-crest anchors so that genuine off-target organoid cells have somewhere to map. We
    subset by the CELLxGENE ontology ``cell_type`` via the origin map (NOT the native Cao
    ``Main_cluster_name``, which the Census file does not carry), in BACKED mode, so RAM is
    bounded by the kept subset rather than the full ~4M-cell atlas. ``max_cells`` caps it further.
    """
    from src.utils.origin_map import map_origin_series

    logging.info(f"Loading fetal atlas (backed) from {fetal_path}")
    fetal = sc.read_h5ad(fetal_path, backed='r')
    if cell_type_col not in fetal.obs.columns:
        raise KeyError(
            f"Fetal atlas lacks '{cell_type_col}'; got {list(fetal.obs.columns)[:25]}. "
            f"Fetch via the Census SOMA path so cell_type is the CL ontology label."
        )
    origins = map_origin_series(fetal.obs[cell_type_col], origin_map)
    keep = origins.isin(keep_origins).values
    n_keep = int(keep.sum())
    logging.info(f"Fetal: keeping {n_keep}/{fetal.n_obs} cells with origin in {keep_origins}.")
    if n_keep == 0:
        raise ValueError(
            "No fetal cells map to the requested origins; check origin_map keys vs the atlas "
            f"'{cell_type_col}' labels (examples: {fetal.obs[cell_type_col].astype(str).unique()[:8]})."
        )
    idx = np.flatnonzero(keep)
    if max_cells and n_keep > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, max_cells, replace=False))
        logging.info(f"Capped fetal anchors to {max_cells} cells (RAM guard).")
    fetal = fetal[idx].to_memory()
    fetal = _ensure_ensembl_var_names(fetal, name="fetal")
    return _ensure_counts_in_X(fetal, name="fetal")


def build_combined_reference(hdbca_path: Path, fetal_path: Path, origin_map: dict,
                             cell_type_col: str = 'cell_type',
                             keep_origins=('mesoderm', 'endoderm', 'neural_crest'),
                             fetal_max_cells: int = None, allow_unmapped: bool = False,
                             seed: int = 0, config_path: str = 'config/paths.yaml'):
    """Build the harmonized reference: HDBCA (neural) + Cao non-neural anchors.

    Guarantees (fail loudly, never silently train on bad data):
      * both atlases Ensembl-indexed (>=90% ENSG var_names)            [R4]
      * inner-join keeps >=80% of genes                                [R4]
      * X is integer raw counts (promote raw.X if needed) + asserted   [B3]
      * Cao subset by ontology cell_type via origin map, RAM-capped    [B4]
      * every reference cell_type maps to an origin (except generic 'cell')  [R5]
    """
    logging.info(f"Loading HDBCA (backed) from {hdbca_path}")
    hdbca = sc.read_h5ad(hdbca_path, backed='r')
    hdbca = _ensure_ensembl_var_names(hdbca, name="hdbca")

    fetal = load_and_subset_fetal(fetal_path, origin_map, keep_origins=keep_origins,
                                  cell_type_col=cell_type_col, max_cells=fetal_max_cells, seed=seed)

    # R4: gene-namespace guard (both must be Ensembl-indexed for a meaningful join + query overlap).
    for nm, a in (("hdbca", hdbca), ("fetal", fetal)):
        frac_ensg = float(pd.Index(a.var_names).astype(str).str.startswith('ENSG').mean())
        if frac_ensg < 0.9:
            raise ValueError(f"[{nm}] only {frac_ensg:.0%} ENSG var_names; expected Ensembl-indexed.")

    logging.info(f"HDBCA shape: {hdbca.shape}, Fetal (anchors) shape: {fetal.shape}")
    common_vars = hdbca.var_names.intersection(fetal.var_names)
    drop_frac = 1 - len(common_vars) / max(1, min(len(hdbca.var_names), len(fetal.var_names)))
    logging.info(f"Common genes for inner join: {len(common_vars)} (drop {drop_frac:.0%}).")
    if drop_frac > 0.20:
        raise ValueError(f"Inner join drops {drop_frac:.0%} of genes (>20%); gene-nomenclature "
                         f"mismatch (Ensembl vs symbol, or version suffixes)?")

    hdbca = hdbca[:, common_vars].to_memory()
    hdbca = _ensure_counts_in_X(hdbca, name="hdbca")
    fetal = fetal[:, common_vars].copy()

    # Free heavy metadata before concat (memory).
    hdbca.obsm.clear(); hdbca.varm.clear(); hdbca.uns.clear()
    fetal.obsm.clear(); fetal.varm.clear(); fetal.uns.clear()
    gc.collect()

    logging.info("Concatenating atlases...")
    ref = sc.concat([hdbca, fetal], join='inner', label='ref_source', keys=['hdbca', 'fetal'])
    del hdbca, fetal
    gc.collect()

    # B4: origin from the CELLxGENE ontology cell_type (consistent across both atlases).
    if cell_type_col not in ref.obs.columns:
        raise KeyError(f"Reference lacks '{cell_type_col}' for origin mapping.")
    from src.utils.origin_map import map_origin_series, validate_origin_map
    ref.obs['origin'] = map_origin_series(ref.obs[cell_type_col], origin_map, default='unknown')

    # R5: raising origin-coverage gate (only the generic 'cell' label may be unmapped).
    rep = validate_origin_map(origin_map, ref.obs[cell_type_col])
    bad = [t for t in rep['unmapped_types'] if str(t).strip().lower() != 'cell']
    if bad and not allow_unmapped:
        raise ValueError(
            f"{len(bad)} reference cell types have no origin mapping (e.g. {bad[:8]}). "
            f"Extend src/utils/origin_map.py (extend_origin_map) or pass allow_unmapped=True."
        )
    if bad:
        logging.warning(f"allow_unmapped=True: {len(bad)} cell types -> 'unknown' (e.g. {bad[:8]}).")

    # B3: hard gate -- a log-normalized reference can never reach training.
    fi = _frac_integer(ref.X)
    if fi <= 0.999:
        raise ValueError(f"Reference X is not integer counts (frac_int={fi:.3f}); aborting "
                         f"(seurat_v3 HVG + SCVI ZINB require raw counts).")

    logging.info(f"Combined reference built: {ref.n_obs} cells, {ref.n_vars} genes, integer X confirmed.")
    return ref


def save_reference(ref, output_path: Path):
    """Saves the combined reference with compression."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving combined reference to {output_path}")
    ref.write_h5ad(output_path, compression='gzip')
    logging.info("Save complete.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from src.utils.origin_map import get_default_origin_map
    config = load_config()
    paths = config.get("datasets", {})
    outputs = config.get("outputs", {})

    hdbca_p = Path(paths.get("hdbca", {}).get("local_path", "data/hdbca.h5ad"))
    fetal_p = Path(paths.get("cao_fetal", {}).get("local_path", "data/cao_fetal.h5ad"))
    out_p = Path(outputs.get("reference", "results/pilot/reference.h5ad"))

    if hdbca_p.exists() and fetal_p.exists():
        ref = build_combined_reference(hdbca_p, fetal_p, get_default_origin_map())
        save_reference(ref, out_p)
    else:
        logging.warning("Datasets not found. Run scripts/download_census.py / retrieve_v2.py first.")
