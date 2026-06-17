# GPU-pod runbook — H100 80 GB / 251 GB RAM (Phase 1 full run)

The H100 pod does the RAM-heavy reference build + all training. It mounts the **same RunPod network
volume** as the CPU pod (so `data/`, the code, and outputs are shared); the conda env lives on the
per-pod overlay and is rebuilt with **CUDA** torch. Validated end-to-end on 2026-06-15
(scvi-tools 1.3.3, torch 2.12+cu130).

## 0. Environment (overlay; ~10 min)
Miniconda + python 3.10 + CUDA torch + scvi-tools **1.3.3** + scikit-misc + cellxgene-census.
Use `build_env_gpu.sh` (conda-forge to dodge the Anaconda ToS gate; absolute `…/envs/nullstate/bin/python -m pip`).

```bash
bash build_env_gpu.sh           # writes env_build_gpu.log; verifies torch.cuda.is_available()
ENVPY=/root/miniconda3/envs/nullstate/bin/python
```

Hard pins (do NOT loosen): **python=3.10, scvi-tools==1.3.3** (1.4.x needs py≥3.11 and changes the
ContrastiveVI / scANVI train signatures), **scikit-misc** (required by `seurat_v3` HVG), and
**cellxgene-census==1.17.\*** + tiledbsoma≥1.15.3 (matches the hardcoded census `2025-11-08` LTS).

## 1. Stage data (Census streams from S3; ~10–20 min)
The query and reference must both be **Ensembl-indexed with integer raw counts**.

```bash
# Query: the CELLxGENE HNOCA copy (raw.X = integer UMIs; the Zenodo copy lacks them) -> data/hnoca_cellxgene.h5ad
PYTHONPATH=. $ENVPY scripts/fetch_verify_cellxgene.py --out-dir data/

# Reference atlases via the Census (raw counts in X, ontology cell_type):
#   HDBCA all-primary (neural core) + Cao filtered to the non-neural origin_map labels (avoids the 4M pull)
PYTHONPATH=. $ENVPY scripts/download_reference.py
```

Note: Census `get_anndata` indexes var by integer position with the Ensembl IDs in
`var['feature_id']` — `build_reference` / `download_reference` promote that to `var_names`
automatically. `paths.yaml` `hnoca.local_path` must point at `data/hnoca_cellxgene.h5ad`.

## 2. Build the harmonized reference (~30–60 min; needs the 251 GB box)
```bash
PYTHONPATH=. $ENVPY src/data/build_reference.py     # -> results/pilot/reference.h5ad
```
Guards that abort BEFORE the multi-hour train (all validated): both atlases Ensembl-indexed;
inner-join keeps ≥80 % of genes; **X is integer counts** (promoted from `raw.X` if needed); Cao
subset by ontology `cell_type` via `origin_map` (RAM-capped by `fetal_max_cells`); every reference
cell_type maps to an origin (only the generic `'cell'` may be unmapped). Result on 2026-06-15:
2,615,898 cells × 61,497 genes, integer X confirmed.

## 3. Preflight (must be GREEN; ~2 min)
```bash
PYTHONPATH=. $ENVPY scripts/preflight_check.py --config-dir config/
```
GREEN requires: reference raw integer counts + `donor_id` + origin coverage; query Ensembl
var_names + `raw.X` counts + a study-level protocol column (`assay_differentiation`).

## 4. Mapping + gates (Steps 2–4; ~5–7 h)
```bash
PYTHONPATH=. $ENVPY scripts/run_pilot.py --skip-to-step 2 --config-dir config/
```
scVI 400 ep (~2–2.5 h @ ~21 s/epoch on 2.62 M cells) → scANVI ≤200 ep early-stop → **scArches 100 ep**
(`scarches_max_epochs: 100`, valid because the query now feeds real `raw.X` counts) → scoring +
off-manifold kNN + classify (writes `hnoca_annotated.h5ad`) → dish-vector (Step 3) → count gate (Step 4).
**Sanity:** the log must show `Using query.raw.X` (NOT `Scaling query data by 283`) and finite scArches ELBO.

## 5. Aim 2 — WS1 contrastiveVI (~1–2 h)
```bash
PYTHONPATH=. $ENVPY scripts/run_ws1.py --config-dir config/
```
Trains contrastiveVI (background = primary on-target, target = organoid on-target; organoid counts
from `raw.X`), then exports `results/pilot/z_lin.npy` + `populations.csv` for the off-target cells,
plus the `contrastive_model/`. Cell counts are capped by `contrastive_max_*` (RAM/compute guard).

## 6. Aim 3 — WS2 geometry (~0.3–1 h, CPU-ok)
```bash
PYTHONPATH=. $ENVPY scripts/run_aim3.py --config-dir config/ \
    --zlin results/pilot/z_lin.npy --labels results/pilot/populations.csv \
    --out results/pilot/aim3_geometry.json
```
Rarefaction → N_min, all pairwise matched-N sliced-Wasserstein vs the self-distance floor, BH-FDR → JSON.

## Operational notes
- **Resume:** every training stage checkpoints + writes `training_progress.json`; re-running
  `run_pilot.py --skip-to-step 2` reloads completed stages.
- **Background runs:** `nohup flock -n .step2.lock bash -c "... $ENVPY scripts/run_pilot.py ..." > results/pilot/step2.log 2>&1 &`
  survives SSH drops; poll the log. Detect completion via markers, not process presence.
- **Memory:** Step 2 loads the query as X+obs+var (skips the ~31 GB `raw.X`); the count gate reads
  obs-only. The reference (2.62 M) attached to the scvi model is the largest resident object.
- **Total wall-clock:** ~6 h (build → gates), ~7–9 h including WS1+WS2.
