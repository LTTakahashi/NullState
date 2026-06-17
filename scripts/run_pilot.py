import argparse
import logging
from pathlib import Path
import yaml
import anndata as ad
import pandas as pd
import sys

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging, log_step_start, log_step_end, log_gate_decision
from src.utils.origin_map import get_default_origin_map
from src.data.build_reference import build_combined_reference, save_reference
from src.mapping.train_scanvi import run_training_pipeline
from src.mapping.compute_scores import calibrate_thresholds, compute_mapping_entropy, compute_offmanifold_score, annotate_query, build_empirical_label_prior
from src.mapping.classify import classify_all
from src.calibration.dish_vector import run_dish_vector_test
from src.gates.count_check import run_count_check
from src.utils.qc import evaluate_integration, label_confidence_report

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def assert_cellxgene_query_schema(path):
    """Fail loudly if the query is not the CELLxGENE HNOCA copy (Ensembl var_names + raw.X counts).

    Guards against silently running on the legacy Zenodo file (gene-symbol var_names, no usable
    raw.X), which sends scArches surgery down the 283x log-norm fallback (mis-specified ZINB).
    """
    a = ad.read_h5ad(path, backed='r')
    try:
        frac_ensg = float(pd.Index(a.var_names).astype(str).str.startswith('ENSG').mean())
        has_raw = a.raw is not None
    finally:
        try:
            a.file.close()
        except Exception:
            pass
    problems = []
    if frac_ensg < 0.5:
        problems.append(f"var_names not Ensembl ({frac_ensg:.0%} ENSG)")
    if not has_raw:
        problems.append("no .raw (raw integer UMI counts missing)")
    if problems:
        raise SystemExit(
            f"Query at {path} is not the CELLxGENE HNOCA copy: {', '.join(problems)}. "
            f"Fetch it with scripts/fetch_verify_cellxgene.py and point paths.yaml "
            f"datasets.hnoca.local_path at data/hnoca_cellxgene.h5ad."
        )
    logging.info(f"Query schema OK: {frac_ensg:.0%} ENSG var_names, .raw present (CELLxGENE copy).")


def read_annotated_lean(path, obs_only: bool = False) -> ad.AnnData:
    """Read the annotated query WITHOUT its heavy count data.

    Step 3 needs X (log-norm) + obs + var; Step 4 (count gate) is obs-only. The annotated file
    is slimmed at write time (layers/obsm/uns cleared), but the 1.77M-cell X is still large, so
    we backed-read and reconstruct only what is needed -- and for ``obs_only`` we skip X entirely
    (a 1-gene placeholder) so the count gate never materializes the full matrix.
    """
    a = ad.read_h5ad(path, backed='r')
    if obs_only:
        import numpy as np
        import scipy.sparse as sp
        lean = ad.AnnData(X=sp.csr_matrix((a.n_obs, 1), dtype=np.float32), obs=a.obs.copy())
    else:
        lean = ad.AnnData(X=a.X[:], obs=a.obs.copy(), var=a.var.copy())
    try:
        a.file.close()
    except Exception:
        pass
    return lean

def main():
    parser = argparse.ArgumentParser(description="NullState HNOCA Pilot Orchestrator")
    parser.add_argument("--config-dir", type=str, default="config/", help="Path to config directory")
    parser.add_argument("--skip-to-step", type=int, default=1, help="Step to resume from")
    parser.add_argument("--force", action="store_true", help="Force re-run of all steps")
    args = parser.parse_args()

    paths_config = Path(args.config_dir) / "paths.yaml"
    params_config = Path(args.config_dir) / "params.yaml"
    genes_config = Path(args.config_dir) / "gene_sets.yaml"

    paths = load_config(paths_config)
    params = load_config(params_config)
    
    out_dir = Path(paths.get('results_dir', 'results/pilot/'))
    logger = setup_logging(out_dir)
    
    logger.info("Starting NullState HNOCA Pilot")
    
    # --- Step 1: Reference Construction ---
    if args.skip_to_step <= 1:
        log_step_start(logger, 1, "Reference Construction")
        
        hdbca_p = Path(paths['datasets']['hdbca']['local_path'])
        fetal_p = Path(paths['datasets']['cao_fetal']['local_path'])
        ref_out = Path(paths['outputs']['reference'])
        
        if ref_out.exists() and not args.force:
            logger.info(f"Reference already exists at {ref_out}. Skipping build.")
        else:
            origin_map = get_default_origin_map()
            # B4: subset Cao by the CELLxGENE ontology cell_type via the origin map (NOT the
            # native 'Main_cluster_name', which the Census file lacks). Keeps the non-neural
            # off-target anchors; HDBCA supplies the neural core. RAM-capped via fetal_max_cells.
            ref = build_combined_reference(
                hdbca_p, fetal_p, origin_map,
                cell_type_col=params.get('reference_cell_type_col', 'cell_type'),
                keep_origins=tuple(params.get('reference_keep_origins',
                                              ['mesoderm', 'endoderm', 'neural_crest'])),
                fetal_max_cells=params.get('fetal_max_cells'),
                allow_unmapped=params.get('allow_unmapped_origins', False),
            )
            save_reference(ref, ref_out)
            
        log_step_end(logger, 1)

    # --- Step 2: Mapping and Classification ---
    if args.skip_to_step <= 2:
        log_step_start(logger, 2, "scANVI Mapping and Classification")
        
        ref_path = Path(paths['outputs']['reference'])
        query_path = Path(paths['datasets']['hnoca']['local_path'])
        model_out = Path(paths['outputs']['model'])

        # B1 guard: abort now (cheap, metadata-only) if the query is the stale Zenodo file.
        assert_cellxgene_query_schema(query_path)

        # Train
        ref_model, query_model = run_training_pipeline(ref_path, query_path, model_out, params_config)

        # QC #1: confirm Aim-1 harmonization actually worked (donors mixed, cell types preserved)
        evaluate_integration(
            ref_model, ref_model.adata,
            batch_key=params.get('batch_key'),
            sample=params.get('integration_qc_sample', 20000),
            out_path=out_dir / "integration_qc.json",
        )

        # Load query for annotation: X (log-norm) + obs + var only. Skip raw.X (~31GB of integer
        # counts) -- it is never used here, and loading it alongside X + the reference risks the
        # 251GB cgroup (the pilot's OOM ceiling). Backed read + reconstruct loads only what we need.
        _q = ad.read_h5ad(query_path, backed='r')
        query = ad.AnnData(X=_q.X[:], obs=_q.obs.copy(), var=_q.var.copy())
        try:
            _q.file.close()
        except Exception:
            pass

        # Empirical (non-uniform) label prior (Phase 1 / WS0b), off by default. Built from
        # the reference labels and applied identically in calibration and query scoring so
        # the calibrated tau_H stays consistent. Toggle via params.yaml: empirical_label_prior.
        class_prior = None
        if params.get('empirical_label_prior', False):
            class_prior = build_empirical_label_prior(ref_model.adata.obs['cell_type'])
            logger.info(f"Empirical label-prior correction ENABLED ({len(class_prior)} classes).")

        # Calibrate thresholds using the perfectly prepared reference attached to the model
        thresh = calibrate_thresholds(
            ref_model, ref_model.adata,
            holdout_fraction=params.get('ref_holdout_fraction', 0.1),
            tau_H_percentile=params.get('tau_H_percentile', 95),
            tau_R_percentile=params.get('tau_R_percentile', 99),
            k=params.get('k_neighbors_offmanifold', 15),
            class_prior=class_prior,
        )

        # Score using the perfectly prepared query attached to the model
        entropy, soft = compute_mapping_entropy(query_model, query_model.adata, class_prior=class_prior)
        offmanifold = compute_offmanifold_score(ref_model, ref_model.adata, query_model, query_model.adata, k=params.get('k_neighbors_offmanifold', 15))
        
        origin_map = get_default_origin_map() # Assuming soft returns cell_type labels
        query = annotate_query(query, query_model, entropy, offmanifold, origin_map, soft)

        # QC #2: per-predicted-type label confidence (where is the classifier guessing?)
        label_confidence_report(
            query.obs, soft=soft, tau_H=thresh['tau_H'],
            out_path=out_dir / "label_confidence.csv",
        )

        # Classify
        query.obs['pilot_class'] = classify_all(
            query, thresh['tau_H'], thresh['tau_R'],
            expected_germ_layer=params.get('expected_germ_layer', 'neural'),
        )
        
        # Save query with annotations. Drop heavy artifacts (the counts layer, embeddings)
        # so the file stays small and Steps 3/4 don't exceed the container memory limit.
        for _attr in ('layers', 'obsm', 'obsp', 'varm', 'varp'):
            getattr(query, _attr).clear()
        query.uns = {}
        query.raw = None  # never re-attach the ~31GB raw counts -- keeps the annotated file lean
        query_out = out_dir / "hnoca_annotated.h5ad"
        query.write_h5ad(query_out, compression='gzip')
        
        log_step_end(logger, 2)

    # --- Step 3: Dish-Vector Test ---
    if args.skip_to_step <= 3:
        log_step_start(logger, 3, "Dish-Vector Test (GO/NO-GO)")
        
        query_out = out_dir / "hnoca_annotated.h5ad"
        ref_path = Path(paths['outputs']['reference'])
        
        query = read_annotated_lean(query_out)
        ref = ad.read_h5ad(ref_path)

        gene_sets = load_config(genes_config).get('sets', {})
        # Note: In a real scenario, you'd fetch the actual genes from gseapy here
        # Mocking the sets for the scaffold
        mock_sets = {k: set() for k in gene_sets.keys()} 
        
        results = run_dish_vector_test(query, ref, mock_sets, params_config)
        
        if "error" not in results:
            log_gate_decision(logger, "Dish-Vector Test", results["gate_result"]["decision"], {"Mean Cosine": results["gate_result"]["mean_cosine"]})
        else:
            logger.error(results["error"])
            
        log_step_end(logger, 3)

    # --- Step 4: Count Check ---
    if args.skip_to_step <= 4:
        log_step_start(logger, 4, "Count Check (GREEN/YELLOW/RED)")
        
        query_out = out_dir / "hnoca_annotated.h5ad"
        query = read_annotated_lean(query_out, obs_only=True)  # count gate is obs-only

        results = run_count_check(query, params_config)
        
        metrics = {
            "Total candidates": results["counts"]["total_count"],
            "Protocols": len(results["counts"]["by_protocol"])
        }
        
        log_gate_decision(logger, "Count Check", results["gate"]["decision"], metrics)
        
        if results["diagnostic"]:
            logger.info("RED Diagnostic Run:")
            logger.info(results["diagnostic"]["diagnosis"])
            
        log_step_end(logger, 4)

    logger.info("Pilot Execution Complete.")

if __name__ == "__main__":
    main()
