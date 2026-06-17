# NullState

**Convergent or Lineage-Specific? The Geometry of Off-Target Cell States Across Human Organoid Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

NullState is a computational-biology project that asks whether the *off-target* cell
populations produced when human organoid differentiation fails **converge** across germ
layers toward shared "default" states, or fail in **lineage-specific** ways.

## Central Hypothesis
Off-target states form a **structured mixture**: a subset converging across germ layers
toward shared default states, and a subset that is lineage-bound.

## Specific Aims
1. **Construct a harmonized cross-germ-layer reference space and off-target ontology.**
2. **Disentangle the *in-vitro* stress signature from lineage biology** by contrastive subspace separation.
3. **Characterize and validate the geometry of off-target convergence.**

## The HNOCA Pilot
Before executing the full three-aim project, this repository implements a compute pilot on the
Human Neural Organoid Cell Atlas (HNOCA) to resolve three empirical go/no-go gates:

```mermaid
graph TD
    A["Step 2: scANVI Mapping"] --> B{"Definition Sane?"}
    B -- Yes --> C["Step 3: Dish-Vector Test"]
    B -- No --> D["Fix Reference"]
    C --> E{"Consistency > 0.70?"}
    E -- Yes --> F["GO: single z_iv (Aim 2)"]
    E -- No --> G["NO-GO: conditional z_iv|y_id"]
    F --> H["Step 4: Count Check"]
    G --> H
    H --> I{"Count Threshold"}
    I -- ">=1000 total, >=2 protocols" --> J["GREEN: Aim 3 powered"]
    I -- "300-1000 total" --> K["YELLOW: scope-limited"]
    I -- "<300 total" --> L["RED: diagnostic"]
```

### Pilot outcome (all gates passed)
| Gate | Result | Detail |
|------|--------|--------|
| A — Definition sane | **PASS** | coherent on-target neural core; `other` ≈ 0 |
| B — Dish-Vector | **GO** | mean cosine 0.906 (> 0.70) ⇒ single shared `z_iv` |
| C — Count Check | **GREEN** | 7,568 off-target mesenchymal cells; 4 protocols ≥ 300 |

Cross-atlas harmonization independently verified (batch iLISI 8.04 vs. 5.94 unintegrated,
cell-type structure preserved). Full write-up, figures, and honest limitations:
**[`reports/nullstate_pilot_report.pdf`](reports/nullstate_pilot_report.pdf)**.

## Installation
```bash
# Conda (recommended)
conda env create -f env/environment.yml
conda activate nullstate

# or pip
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# 1. Obtain the atlases (see data/README.md), then build the reference
python scripts/run_pilot.py --skip-to-step 1

# 2. Pre-flight data checks before a long run
python scripts/preflight_check.py --config-dir config/

# 3. Run the pilot from the mapping step
python scripts/run_pilot.py --skip-to-step 2 --config-dir config/
```

## Repository Structure
```
NullState/
├── config/            YAML config (paths, params, gene sets)
├── data/              Atlases (gitignored; see data/README.md)
├── docs/              Research strategy, specific aims, execution plan
├── env/               Conda environment definition
├── models/            Trained scVI/scANVI weights (gitignored)
├── reports/           Pilot report (PDF + LaTeX), figures, run artifacts
├── scripts/           Pipeline & data-retrieval scripts
├── src/               Core package
│   ├── data/          Reference construction, retrieval, schema discovery
│   ├── mapping/       scVI/scANVI training, scoring, classification
│   ├── calibration/   Dish-vector test
│   ├── gates/         Count-check gate
│   └── utils/         Origin map, QC, logging, config
└── tests/             Unit tests
```

## Data Sources
- He, Fleck et al. "An integrated transcriptomic cell atlas of human neural organoids." *Nature* 2024.
- Braun et al. "Comprehensive cell atlas of the first-trimester developing human brain." *Science* 2023.
- Cao et al. "A human cell atlas of fetal gene expression." *Science* 2020.

See [`data/README.md`](data/README.md) for download instructions.

## License
Released under the [MIT License](LICENSE).
