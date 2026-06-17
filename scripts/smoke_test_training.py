#!/usr/bin/env python
"""GPU training smoke test: validate the scvi-tools 1.3.3 API + kwargs BEFORE the multi-hour run.

Exercises the REAL functions -- train_reference_scvi -> upgrade_to_scanvi -> map_query_scarches --
on tiny synthetic raw-count data, with the exact train kwargs the pipeline uses (early_stopping,
early_stopping_patience, plan_kwargs, gradient_clip_val, ModelCheckpoint callbacks, accelerator).
A kwarg/API mismatch here would otherwise crash the real run hours in. NOT a scientific run.

    PYTHONPATH=. python scripts/smoke_test_training.py
"""
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


def _synth(n=600, g=200, seed=0, donors=('d1', 'd2', 'd3')):
    import anndata as ad
    import pandas as pd
    rng = np.random.default_rng(seed)
    X = rng.poisson(0.6, size=(n, g)).astype('float32')          # integer raw counts
    obs = pd.DataFrame({
        'cell_type': rng.choice(['neuron', 'fibroblast', 'radial glial cell'], n),
        'donor_id': rng.choice(list(donors), n),
    })
    a = ad.AnnData(X=X, obs=obs)
    a.var_names = [f'ENSG{i:08d}' for i in range(g)]             # Ensembl-style, ref/query aligned
    return a


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    import scvi
    from src.mapping.train_scanvi import (
        check_gpu, train_reference_scvi, upgrade_to_scanvi, map_query_scarches,
    )
    print(f"[smoke-train] scvi-tools {scvi.__version__}")
    check_gpu()

    ref = _synth(600, 200, seed=0, donors=('d1', 'd2', 'd3'))
    query = _synth(400, 200, seed=1, donors=('q1', 'q2'))         # NEW query donors (scArches surgery)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        scvi_m = train_reference_scvi(ref, out, batch_key='donor_id', max_epochs=2, checkpoint_every=1)
        scanvi_m = upgrade_to_scanvi(scvi_m, out, labels_key='cell_type', max_epochs=2,
                                     early_stopping=True, early_stopping_patience=2, checkpoint_every=1)
        # Query prep: SCANVI expects the unlabeled category and aligned var_names (both ENSG here).
        query.obs['cell_type'] = 'Unknown'
        qm = map_query_scarches(query, scanvi_m, out, max_epochs=2)
        z = qm.get_latent_representation()
        assert z.shape[0] == query.n_obs, "scArches latent row count mismatch"

    print("\n=== TRAINING SMOKE PASSED: scVI -> scANVI -> scArches API valid on scvi-tools "
          f"{scvi.__version__} (latent {z.shape}) ===")


if __name__ == '__main__':
    main()
