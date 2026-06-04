import scanpy as sc
import pandas as pd
from pathlib import Path
import yaml
import logging

def load_config(config_path: str = 'config/paths.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_subset_fetal(fetal_path: Path, lineage_column: str = 'cell_type', lineages: list[str] = ['mesoderm', 'neural_crest', 'endoderm']) -> sc.AnnData:
    """
    Loads fetal atlas and subsets to specified lineages.
    Note: 'lineage_column' needs to be the one that maps to germ layers, 
    which may require custom logic if the atlas only has raw cell_types.
    """
    logging.info(f"Loading fetal atlas from {fetal_path}")
    # In a real scenario, this would not be backed if we need to modify it heavily
    # or we would read backed, subset, and copy to memory.
    fetal = sc.read_h5ad(fetal_path, backed='r')
    
    # Placeholder: if fetal doesn't have a direct 'lineage' column, we need to map it
    # We will assume for now there's some column or we apply an origin map here as well.
    # To be safe, we will just pass it through and rely on the combined mapping later,
    # OR we assume there's a predefined way to filter.
    # We'll just return the whole fetal object or subset if the column exists.
    if lineage_column in fetal.obs.columns:
        fetal = fetal[fetal.obs[lineage_column].isin(lineages)].to_memory()
        logging.info(f"Subset fetal atlas to {len(fetal)} cells matching lineages.")
    else:
        logging.warning(f"Fetal atlas missing {lineage_column}. Loading full atlas to memory.")
        fetal = fetal.to_memory()
        
    return fetal

def build_combined_reference(hdbca_path: Path, fetal_path: Path, origin_map: dict[str, str], lineage_column: str = 'cell_type', lineages: list[str] = None, config_path: str = 'config/paths.yaml') -> sc.AnnData:
    """
    Builds the combined reference by merging HDBCA and a subset of the fetal atlas.
    
    Biological rationale: Off-target mesenchyme in neural organoids needs a mesodermal
    anchor in the reference. If we only use a neural reference, true off-target cells 
    will look like 'ambiguous/novel' technical artifacts due to high reconstruction error.
    """
    logging.info(f"Loading HDBCA from {hdbca_path}")
    hdbca = sc.read_h5ad(hdbca_path, backed='r')
    
    if lineages is None:
        lineages = ['mesoderm', 'neural_crest', 'endoderm']
        
    fetal = load_and_subset_fetal(fetal_path, lineage_column=lineage_column, lineages=lineages)
    
    logging.info(f"HDBCA shape: {hdbca.shape}, Fetal shape: {fetal.shape}")
    
    # Ensure they have compatible variables before concat
    common_vars = hdbca.var_names.intersection(fetal.var_names)
    logging.info(f"Common genes for inner join: {len(common_vars)}")
    if len(common_vars) < 0.8 * min(len(hdbca.var_names), len(fetal.var_names)):
        logging.warning("Inner join drops >20% of genes. Check gene nomenclature (Ensembl vs Symbols).")
    
    hdbca = hdbca[:, common_vars].to_memory()
    fetal = fetal[:, common_vars].copy()
    
    # CRITICAL MEMORY OPTIMIZATION: Clear all heavy metadata before concat
    hdbca.obsm.clear(); hdbca.varm.clear(); hdbca.uns.clear()
    fetal.obsm.clear(); fetal.varm.clear(); fetal.uns.clear()
    
    import gc
    gc.collect()
    
    logging.info("Concatenating atlases...")
    ref = sc.concat([hdbca, fetal], join='inner', label='ref_source', keys=['hdbca', 'fetal'])
    
    # Free the individual atlases from memory immediately
    del hdbca
    del fetal
    gc.collect()
    
    logging.info("Applying origin map to assign developmental origins...")
    # Assume cell_type is the column containing the detailed annotation
    if 'cell_type' not in ref.obs.columns:
        # Fallback to whatever seems like cell type
        logging.warning("'cell_type' column not found. Using generic mapping strategy placeholder.")
        ref.obs['origin'] = 'unknown'
    else:
        from src.utils.origin_map import map_origin_series
        # Case-insensitive, whitespace-robust mapping so CELLxGENE ontology labels
        # ('radial glial cell', 'Schwann cell', ...) all resolve. Unmapped -> 'unknown'.
        ref.obs['origin'] = map_origin_series(ref.obs['cell_type'], origin_map, default='unknown')

        # Validation
        unmapped = ref.obs.loc[ref.obs['origin'] == 'unknown', 'cell_type'].unique()
        if len(unmapped) > 0:
            logging.error(f"Found {len(unmapped)} cell types with no origin mapping. Examples: {unmapped[:5]}")
            # Depending on strictness, we might raise an error here.
            
    logging.info(f"Combined reference built: {ref.n_obs} cells, {ref.n_vars} genes.")
    return ref

def save_reference(ref: sc.AnnData, output_path: Path):
    """Saves the combined reference with compression."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving combined reference to {output_path}")
    ref.write_h5ad(output_path, compression='gzip')
    logging.info("Save complete.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # This is a placeholder runner. Real execution requires actual origin_map.
    config = load_config()
    paths = config.get("datasets", {})
    outputs = config.get("outputs", {})
    
    hdbca_p = Path(paths.get("hdbca", {}).get("local_path", "data/hdbca.h5ad"))
    fetal_p = Path(paths.get("cao_fetal", {}).get("local_path", "data/cao_fetal.h5ad"))
    out_p = Path(outputs.get("reference", "results/pilot/reference.h5ad"))
    
    # Dummy map for standalone execution test
    dummy_origin_map = {} 
    
    if hdbca_p.exists() and fetal_p.exists():
        ref = build_combined_reference(hdbca_p, fetal_p, dummy_origin_map)
        save_reference(ref, out_p)
    else:
        logging.warning("Datasets not found. Run retrieve.py first.")
