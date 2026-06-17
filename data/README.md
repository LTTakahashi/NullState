# Data

The raw single-cell atlases are **not tracked in git** (they are tens of GB; see
`.gitignore`). Download them into this directory before running the pipeline.

| File | Dataset | Source |
|------|---------|--------|
| `hnoca.h5ad` | Human Neural Organoid Cell Atlas (query) | [Zenodo 14161275](https://zenodo.org/records/14161275) — `hnoca_cleanedmeta.h5ad` |
| `hdbca.h5ad` | Human Developmental Brain Cell Atlas (Braun et al., *Science* 2023) | CELLxGENE collection `4d8fed08-2d6d-4692-b5ea-464f1d072077` |
| `cao_fetal.h5ad` | Human Fetal Cell Atlas (Cao et al., *Science* 2020) | CELLxGENE collection `c114c20f-1ef4-49a5-9c2e-d965787fb90c` |

Local paths are configured in [`config/paths.yaml`](../config/paths.yaml).

## Phase 2 (cross-germ-layer) — not yet sourced

The cross-germ-layer expansion adds two more organoid systems. They are registered as
placeholders in [`config/paths.yaml`](../config/paths.yaml) (`status: phase2_not_sourced`)
with their germ layer but **no collection IDs yet** — locating them is the WS0a task, and
the IDs are deliberately left as `TODO` rather than guessed.

| File | Dataset | Germ layer | Status |
|------|---------|-----------|--------|
| `heoca.h5ad` | Human Endoderm Organoid Cell Atlas (HEOCA) | endoderm | not sourced (ID TODO) |
| `kidney_organoid.h5ad` | Kidney organoid benchmark | mesoderm | not sourced (ID TODO) |

## Download

```bash
# HNOCA query (Zenodo)
wget -O data/hnoca.h5ad https://zenodo.org/records/14161275/files/hnoca_cleanedmeta.h5ad

# Reference atlases (CELLxGENE Census)
python scripts/download_hdbca.py
python scripts/download_census.py
```

Or build the harmonized reference end-to-end:

```bash
python scripts/run_pilot.py --skip-to-step 1
```
