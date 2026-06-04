import pandas as pd
import anndata as ad
from pathlib import Path
import logging

def classify_cell(origin: str, map_entropy: float, offmanifold: float, tau_H: float, tau_R: float, expected_germ_layer: str = 'neural') -> str:
    """
    Pure function implementing the decision tree.
    
    CRITICAL: The neural-crest guard `origin in ('neural', 'neural_crest')`.
    Neural crest is ectoderm-derived but mesenchymal. Without this guard, 
    every neural organoid generates a systematic false-off-target wave because
    neural crest-derived mesenchyme would be classified as an off-target 
    contaminant instead of an on-target ectodermal product.
    """
    if pd.isna(map_entropy) or pd.isna(offmanifold):
        return 'unmapped'

    if offmanifold > tau_R:
        return 'ambiguous_novel'
        
    if origin in ('neural', 'neural_crest'):
        return 'poorly_diff_ontarget' if map_entropy > tau_H else 'clean_ontarget'
        
    if origin == 'mesoderm':
        return 'true_offtarget' if map_entropy < tau_H else 'ambiguous'
        
    if origin == 'endoderm':
        return 'true_offtarget' if map_entropy < tau_H else 'ambiguous'
        
    return 'other'

def classify_all(query: ad.AnnData, tau_H: float, tau_R: float, expected_germ_layer: str = 'neural') -> pd.Series:
    """Vectorized classification of all cells."""
    logging.info("Classifying all query cells...")
    
    def apply_class(row):
        return classify_cell(
            origin=row['pred_origin'],
            map_entropy=row['map_entropy'],
            offmanifold=row['offmanifold'],
            tau_H=tau_H,
            tau_R=tau_R,
            expected_germ_layer=expected_germ_layer
        )
        
    classes = query.obs.apply(apply_class, axis=1)
    
    logging.info("Class distribution:")
    dist = classes.value_counts()
    for k, v in dist.items():
        logging.info(f"  {k}: {v} ({v/len(classes)*100:.1f}%)")
        
    return classes

def validate_classification(query: ad.AnnData, class_column: str = 'pilot_class') -> dict:
    """Sanity checks for classification."""
    logging.info("Validating classification results...")
    
    report = {}
    if class_column not in query.obs.columns:
        return {"error": f"Column {class_column} not found in query."}
        
    # Check 1: clean_ontarget should be dominated by expected (neural) types
    clean = query[query.obs[class_column] == 'clean_ontarget']
    clean_origins = clean.obs['pred_origin'].value_counts()
    report['clean_ontarget_origins'] = clean_origins.to_dict()
    
    # Check 2: true_offtarget should be mostly mesodermal/endodermal
    offtarget = query[query.obs[class_column] == 'true_offtarget']
    offtarget_origins = offtarget.obs['pred_origin'].value_counts()
    report['true_offtarget_origins'] = offtarget_origins.to_dict()
    
    # Log potential issues
    if 'neural' in offtarget_origins:
        logging.warning("Found 'neural' origins in true_offtarget. This violates the decision rules.")
    
    return report

def save_classifications(query: ad.AnnData, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "classifications.csv"
    logging.info(f"Saving classifications to {out_file}")
    query.obs.to_csv(out_file)
