"""scANVI training pipeline with checkpoint saving.

Trains SCVI → SCANVI on the combined reference, then maps the HNOCA query
via scArches surgery. Saves checkpoints every N epochs so training can
resume after pod crashes.
"""
import scvi
import anndata as ad
import scanpy as sc
from pathlib import Path
import yaml
import logging
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import time
import json


def load_params(config_path: str = 'config/params.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"GPU detected: {gpu_name}")
        logging.info(f"VRAM: {vram_gb:.2f} GB")
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.set_float32_matmul_precision('highest')
        logging.info("Set float32 matmul precision to 'highest' (strict FP32, TF32 disabled to prevent ZINB underflow)")
        return True
    else:
        logging.warning("No GPU detected. Training will be extremely slow.")
        return False


def _make_checkpoint_callback(checkpoint_dir: Path, save_every_n_epochs: int = 50):
    """Create a PyTorch Lightning checkpoint callback for scvi-tools."""
    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='epoch_{epoch:03d}',
        save_top_k=-1,  # keep all checkpoints
        every_n_epochs=save_every_n_epochs,
        save_last=True,
        verbose=True,
    )


def _save_progress(output_dir: Path, stage: str, info: dict):
    """Save a progress marker so we know what completed."""
    progress_file = output_dir / 'training_progress.json'
    progress = {}
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
    progress[stage] = info
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    logging.info(f"Progress saved: {stage} -> {info}")


def _check_progress(output_dir: Path) -> dict:
    """Check what training stages have been completed."""
    progress_file = output_dir / 'training_progress.json'
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {}


def train_reference_scvi(
    ref: ad.AnnData,
    output_dir: Path,
    batch_key: str = 'batch',
    layer: str = None,
    n_latent: int = 30,
    max_epochs: int = 400,
    checkpoint_every: int = 50
) -> scvi.model.SCVI:
    """Train SCVI on the reference with periodic checkpointing."""
    logging.info("Setting up AnnData for SCVI...")
    
    if batch_key not in ref.obs.columns:
        logging.warning(f"{batch_key} not found. Skipping batch correction for SCVI setup.")
        scvi.model.SCVI.setup_anndata(ref, layer=layer)
    else:
        scvi.model.SCVI.setup_anndata(ref, batch_key=batch_key, layer=layer)
    
    logging.info(f"Training SCVI (n_latent={n_latent}, max_epochs={max_epochs})...")
    scvi_model = scvi.model.SCVI(ref, n_latent=n_latent)
    
    # Set up checkpoint callback
    ckpt_dir = output_dir / 'scvi_checkpoints'
    ckpt_callback = _make_checkpoint_callback(ckpt_dir, checkpoint_every)
    
    t0 = time.time()
    scvi_model.train(
        max_epochs=max_epochs,
        enable_checkpointing=True,
        batch_size=4096,
        callbacks=[ckpt_callback],
    )
    elapsed = time.time() - t0
    
    # Log convergence
    history = scvi_model.history['elbo_train']
    final_elbo = history.iloc[-1].values[0]
    logging.info(f"SCVI training complete in {elapsed/60:.1f} min. Final ELBO: {final_elbo:.2f}")
    
    # Save the completed model immediately
    model_dir = output_dir / 'scvi_model'
    model_dir.mkdir(parents=True, exist_ok=True)
    scvi_model.save(str(model_dir), overwrite=True)
    logging.info(f"SCVI model saved to {model_dir}")
    
    _save_progress(output_dir, 'scvi', {
        'epochs': max_epochs,
        'final_elbo': float(final_elbo),
        'elapsed_min': round(elapsed / 60, 1),
        'model_dir': str(model_dir),
    })
    
    return scvi_model


def upgrade_to_scanvi(
    scvi_model: scvi.model.SCVI,
    output_dir: Path,
    labels_key: str = 'cell_type',
    unlabeled_category: str = 'Unknown',
    max_epochs: int = 200,
    checkpoint_every: int = 50,
    early_stopping: bool = True,
    early_stopping_patience: int = 15,
) -> scvi.model.SCANVI:
    """Upgrade SCVI to SCANVI with checkpointing.

    Phase 1 / WS0b hardening: when ``early_stopping`` is set, training stops once the
    validation ELBO stops improving for ``early_stopping_patience`` epochs instead of
    always running the full ``max_epochs``. The pilot showed the classifier over-trains
    by epoch 200 (validation ELBO degraded ~15 from its minimum), which is what made
    rare-type labels unreliable; stopping near the validation optimum fixes that.
    ``max_epochs`` becomes the cap, not the fixed budget. This mirrors the early-stopping
    already used in ``map_query_scarches``.
    """
    logging.info(f"Upgrading to SCANVI using labels_key={labels_key}...")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category
    )

    ckpt_dir = output_dir / 'scanvi_checkpoints'
    ckpt_callback = _make_checkpoint_callback(ckpt_dir, checkpoint_every)

    logging.info(f"Training SCANVI (max_epochs={max_epochs}, early_stopping={early_stopping}, "
                 f"patience={early_stopping_patience})...")
    t0 = time.time()
    train_kwargs = dict(max_epochs=max_epochs, batch_size=4096,
                        enable_checkpointing=True, callbacks=[ckpt_callback])
    if early_stopping:
        # Monitor the validation split (scvi-tools default train_size=0.9) every epoch
        # and stop on plateau, so the classifier is taken near its validation optimum
        # rather than the over-trained epoch-200 endpoint documented in the pilot.
        train_kwargs.update(check_val_every_n_epoch=1, early_stopping=True,
                            early_stopping_patience=early_stopping_patience)
    scanvi_model.train(**train_kwargs)
    elapsed = time.time() - t0

    history = scanvi_model.history['elbo_train']
    n_epochs_run = len(history)
    final_elbo = history.iloc[-1].values[0]
    logging.info(f"SCANVI training complete in {elapsed/60:.1f} min "
                 f"({n_epochs_run} epochs run). Final ELBO: {final_elbo:.2f}")

    _save_progress(output_dir, 'scanvi', {
        'epochs': int(n_epochs_run),
        'max_epochs': max_epochs,
        'early_stopping': bool(early_stopping),
        'final_elbo': float(final_elbo),
        'elapsed_min': round(elapsed / 60, 1),
    })

    return scanvi_model


def _remap_query_to_ensembl(query: ad.AnnData) -> ad.AnnData:
    """Remap the HNOCA query from gene symbols to Ensembl IDs.

    The reference is indexed by Ensembl gene IDs, but the HNOCA query is indexed by
    gene symbol, with the Ensembl IDs stored in ``query.var['ensembl']``. Matching on
    symbols recovers only ~15% of the reference HVGs; switching to the Ensembl column
    raises the overlap to ~74%, which is what scArches needs for a usable mapping.
    Genes with a missing or duplicated Ensembl ID are dropped.
    """
    if 'ensembl' not in query.var.columns:
        logging.warning("Query has no 'ensembl' var column; leaving var_names as gene symbols.")
        return query
    ens = query.var['ensembl'].astype(str)
    valid = ens.str.startswith('ENSG').values
    n_invalid = int((~valid).sum())
    query = query[:, valid].copy()
    query.var_names = query.var['ensembl'].astype(str).values
    dup = query.var_names.duplicated()
    n_dup = int(dup.sum())
    if n_dup:
        query = query[:, ~dup].copy()
    logging.info(
        f"Remapped query var_names to Ensembl IDs (dropped {n_invalid} without an Ensembl ID, "
        f"{n_dup} duplicates; {query.n_vars} genes remain)."
    )
    return query


def _use_query_raw_counts(query: ad.AnnData):
    """Prefer genuine raw counts for the query; returns (query, is_raw_counts).

    SCVI/SCANVI model raw counts with an NB/ZINB likelihood, so the query must be raw
    counts too. HNOCA's ``X`` is log-normalized, but integer UMI counts are retained in
    ``layers['counts_lengthnorm']`` (despite the name, the values are integers: 1,2,3...).
    Count layers share the var axis, so they stay aligned after the Ensembl remap. We
    use such a layer when present; otherwise the caller falls back to the approximate
    283x rescale. ``.raw`` is intentionally not used (its var axis differs from the
    remapped query and would need separate alignment).
    """
    import numpy as np
    import scipy.sparse as sp

    def _is_integer(X) -> bool:
        d = X.data if sp.issparse(X) else np.asarray(X).ravel()
        if d.size == 0:
            return False
        s = d[:200000]
        return bool(np.all(s >= 0) and np.allclose(s, np.round(s)))

    # Named integer-count layers (HNOCA: 'counts_lengthnorm').
    for lk in ('counts', 'counts_lengthnorm', 'raw_counts', 'umi_counts', 'X_counts'):
        if lk in getattr(query, 'layers', {}) and _is_integer(query.layers[lk]):
            query.X = query.layers[lk].copy()
            query.layers.clear()  # drop the now-redundant log-norm/layer copies
            logging.info(f"Using query.layers['{lk}'] (integer counts) as raw counts for scVI mapping.")
            return query, True
    # X itself already integer counts? (require a count-like dynamic range, not small ints)
    if _is_integer(query.X) and float(query.X.max()) > 30:
        logging.info("Query X already appears to be raw integer counts.")
        return query, True
    logging.warning("No raw counts found for query (X looks normalized; no integer count layer). "
                    "Falling back to the approximate 283x log-norm rescale.")
    return query, False


def _ensure_query_batch(query: ad.AnnData, batch_key: str) -> ad.AnnData:
    """Ensure the query carries the batch column the reference model expects.

    When SCVI/SCANVI are trained with a ``batch_key`` (e.g. ``donor_id``), scArches
    surgery requires the query to expose the same column so it can learn new batch
    embedding(s) for the organoid data. If the query already has the column we keep
    its (new) categories; otherwise we look for a sensible HNOCA equivalent, and as a
    last resort assign the whole query to a single new batch.
    """
    if not batch_key:
        return query
    if batch_key in query.obs.columns:
        logging.info(f"Query already has batch column '{batch_key}' "
                     f"({query.obs[batch_key].nunique()} levels).")
        return query
    # Prefer 'bio_sample' (clean per-sample id) over 'batch' (which can contain
    # singleton groups). Singleton/tiny query batches give scArches unstable
    # per-batch embeddings and cause NaN divergence during surgery.
    for alt in ('donor_id', 'bio_sample', 'sample_id', 'sample', 'batch', 'dataset', 'donor'):
        if alt in query.obs.columns:
            vals = query.obs[alt].astype(str)
            vc = vals.value_counts()
            rare = set(vc[vc < 10].index)
            if rare:
                vals = vals.where(~vals.isin(rare), other='rare_batch')
                logging.warning(f"Collapsed {len(rare)} rare '{alt}' groups (<10 cells) into 'rare_batch' "
                                f"to keep scArches batch embeddings stable.")
            query.obs[batch_key] = vals.values
            logging.warning(f"Query missing '{batch_key}'; using '{alt}' as the query batch "
                            f"({query.obs[batch_key].nunique()} levels).")
            return query
    query.obs[batch_key] = 'hnoca_query'
    logging.warning(f"Query has no batch-like column; assigning a single new batch "
                    f"'hnoca_query' for '{batch_key}'. Refine to per-sample batches if available.")
    return query


def map_query_scarches(
    query: ad.AnnData,
    ref_scanvi: scvi.model.SCANVI,
    output_dir: Path,
    max_epochs: int = 100,
    weight_decay: float = 0.0,
    checkpoint_every: int = 25
) -> scvi.model.SCANVI:
    """Map query via scArches surgery with checkpointing."""
    logging.info("Preparing query data for scArches surgery...")
    scvi.model.SCANVI.prepare_query_anndata(query, ref_scanvi)
    
    logging.info("Loading query data into SCANVI model...")
    query_model = scvi.model.SCANVI.load_query_data(query, ref_scanvi)
    
    ckpt_dir = output_dir / 'scarches_checkpoints'
    ckpt_callback = _make_checkpoint_callback(ckpt_dir, checkpoint_every)
    
    logging.info(f"Training scArches model (max_epochs={max_epochs}, weight_decay={weight_decay})...")
    t0 = time.time()
    train_kwargs = dict(
        max_epochs=max_epochs,
        plan_kwargs={'weight_decay': weight_decay, 'lr': 5e-5},
        enable_checkpointing=True,
        batch_size=1024,
        accelerator='gpu' if torch.cuda.is_available() else 'auto',
        callbacks=[ckpt_callback],
    )
    if max_epochs > 0:
        # Surgery fine-tuning: stabilizers against encoder NaN on the sparse query.
        # NOTE: with the query lacking true raw UMI counts (only length-normalized),
        # the reference ZINB likelihood is mis-specified and surgery can still diverge;
        # max_epochs=0 falls back to a stable pure reference projection.
        train_kwargs.update(check_val_every_n_epoch=1, early_stopping=True,
                            early_stopping_patience=15, gradient_clip_val=0.5)
    query_model.train(**train_kwargs)
    elapsed = time.time() - t0
    
    if max_epochs > 0 and 'elbo_train' in query_model.history:
        history = query_model.history['elbo_train']
        final_elbo = history.iloc[-1].values[0]
    else:
        final_elbo = 0.0
        
    logging.info(f"scArches training complete in {elapsed/60:.1f} min. Final ELBO: {final_elbo:.2f}")
    
    _save_progress(output_dir, 'scarches', {
        'epochs': max_epochs,
        'final_elbo': float(final_elbo),
        'elapsed_min': round(elapsed / 60, 1),
    })
    
    return query_model


def save_model(model, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving model to {output_dir}")
    model.save(str(output_dir), overwrite=True)


def run_training_pipeline(
    ref_path: Path,
    query_path: Path,
    output_dir: Path,
    config_path: str = 'config/params.yaml'
) -> tuple[scvi.model.SCANVI, scvi.model.SCANVI]:
    """Full training pipeline with checkpointing and resume support."""
    logging.basicConfig(level=logging.INFO)
    scvi.settings.dl_num_workers = 32
    has_gpu = check_gpu()
    params = load_params(config_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    progress = _check_progress(output_dir)
    
    # --- SCVI ---
    # We must compute HVGs to get the exact gene panel, even if already trained, so the
    # reference used for scoring matches the one used for training. Panel size/flavor are
    # config-driven (params.yaml: n_hvg, hvg_flavor); defaults reproduce the pilot's 4000.
    ref = ad.read_h5ad(ref_path)
    n_hvg = params.get('n_hvg', 4000)
    hvg_flavor = params.get('hvg_flavor', 'seurat_v3')
    logging.info(f"Filtering for top {n_hvg} Highly Variable Genes (HVGs, flavor={hvg_flavor})...")
    sc.pp.highly_variable_genes(ref, n_top_genes=n_hvg, flavor=hvg_flavor, subset=True)
    logging.info(f"Reference shape after HVG filtering: {ref.shape}")
    
    needs_ref_training = 'scvi' not in progress or not (output_dir / 'scvi_model').exists()
    if not needs_ref_training:
        logging.info(f"SCVI already trained ({progress.get('scvi')}). Loading saved model...")

        # Load against the FULL HVG-filtered reference (all cells, all categories).
        # Slicing to a handful of cells drops unused `cell_type` categorical levels,
        # which corrupts the SCANVI label registry and triggers a 34-vs-35 state_dict
        # size mismatch on load. The full reference is also required downstream for
        # threshold calibration and off-manifold scoring, and it is already resident
        # in RAM from the read above, so keeping it costs no additional peak memory.
        scvi.model.SCVI.setup_anndata(ref, layer=None)
        scvi_model = scvi.model.SCVI.load(str(output_dir / 'scvi_model'), adata=ref)
    else:
        scvi_model = train_reference_scvi(
            ref, output_dir,
            batch_key=params.get('batch_key') or 'batch',
            n_latent=params.get('n_latent', 30),
            max_epochs=params.get('scvi_max_epochs', 400),
        )
    
    # --- SCANVI ---
    if 'scanvi' in progress and (output_dir / 'ref_scanvi').exists():
        logging.info(f"SCANVI already trained ({progress['scanvi']}). Loading saved model...")
        ref_scanvi = scvi.model.SCANVI.load(str(output_dir / 'ref_scanvi'), adata=ref)
    else:
        ref_scanvi = upgrade_to_scanvi(
            scvi_model, output_dir,
            max_epochs=params.get('scanvi_max_epochs', 200),
            early_stopping=params.get('scanvi_early_stopping', True),
            early_stopping_patience=params.get('scanvi_early_stopping_patience', 15),
        )
        save_model(ref_scanvi, output_dir / "ref_scanvi")
    
    # --- scArches query mapping ---
    if 'scarches' in progress and (output_dir / 'query_scanvi').exists():
        logging.info(f"scArches already trained ({progress['scarches']}). Loading saved model...")
        query = ad.read_h5ad(query_path)
        query = _remap_query_to_ensembl(query)
        query = _ensure_query_batch(query, params.get('batch_key'))
        scvi.model.SCANVI.prepare_query_anndata(query, ref_scanvi)
        query_model = scvi.model.SCANVI.load(str(output_dir / 'query_scanvi'), adata=query)
    else:
        logging.info(f"Loading query from {query_path}...")
        query = ad.read_h5ad(query_path)
        query = _remap_query_to_ensembl(query)
        query = _ensure_query_batch(query, params.get('batch_key'))
        query, query_is_counts = _use_query_raw_counts(query)

        logging.info("Subsetting query to match reference HVGs...")
        # ref has been subset to exactly the 4000 HVGs above
        query = query[:, query.var_names.intersection(ref.var_names)].copy()

        import scipy.sparse as sp
        import numpy as np

        if not query_is_counts:
            # Fallback only: query was log-normalized and no raw counts were found.
            # Rescale log-norm (max ~13.5) toward the reference raw-count scale (max ~3821);
            # 3821 / 13.5 ~= 283. Approximate -- replace with real counts when available.
            logging.info("Scaling query data by 283 to approximate raw counts (fallback)...")
            if sp.issparse(query.X):
                query.X.data = np.round(query.X.data * 283.0)
            else:
                query.X = np.round(query.X * 283.0)


        logging.info("Checking for dead genes (0 expression) to prevent NaN variance explosions...")
        sums = np.array(query.X.sum(axis=0)).flatten()
        zero_genes = np.where(sums == 0)[0]
        if len(zero_genes) > 0:
            logging.warning(f"Found {len(zero_genes)} dead genes. Injecting a single pseudo-count to stabilize PyTorch variance estimators...")
            if sp.issparse(query.X):
                X_lil = query.X.tolil()
                for g in zero_genes:
                    X_lil[0, g] = 1.0
                query.X = X_lil.tocsr()
            else:
                query.X[0, zero_genes] = 1.0

        # Critical QC: the query expresses the reference HVGs only sparsely (median ~190
        # counts across the overlapping HVGs vs ~5000 genome-wide). Cells with very few
        # HVG counts get extreme library-size normalization and blow the scArches encoder
        # to NaN. Drop them (this is also standard mapping QC -- such cells are
        # uninformative for label transfer anyway).
        min_hvg_counts = params.get('query_min_hvg_counts', 50)
        n_before = query.n_obs
        sc.pp.filter_cells(query, min_counts=min_hvg_counts)
        logging.info(f"Filtered query to cells with >= {min_hvg_counts} counts across HVGs: "
                     f"{n_before} -> {query.n_obs} ({100*(n_before-query.n_obs)/max(n_before,1):.1f}% dropped).")

        if 'cell_type' in query.obs.columns:
            query.obs['original_cell_type'] = query.obs['cell_type']
        query.obs['cell_type'] = 'Unknown'
        
        query_model = map_query_scarches(
            query, ref_scanvi, output_dir,
            max_epochs=params.get('scarches_max_epochs', 100),
            weight_decay=params.get('scarches_weight_decay', 0.0),
        )
        save_model(query_model, output_dir / "query_scanvi")
    
    logging.info("=== Full training pipeline complete ===")
    return ref_scanvi, query_model


if __name__ == '__main__':
    pass
