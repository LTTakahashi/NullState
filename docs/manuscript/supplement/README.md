# Supplementary Information — NullState

Supporting data, logs, code, and tables for *"NullState: a reference-anchored framework for
detecting, diagnosing, and quantifying the off-target compartment of human neural organoids."*

The purpose of this supplement is **defensibility**: every quantitative claim and every figure in
the main text is traceable here to (i) the derived data behind it, (ii) a run log proving it
executed, and (iii) the exact script that produced it. Files are grouped by the claim they support.

## For preprint / journal upload

**`Supplementary_Information.xlsx`** is the single consolidated file to upload as Supplementary
Information (bioRxiv, or a journal's supplement slot). It contains a Contents sheet (index,
claim→evidence map, data/code-availability statement) plus Tables S1–S6, one per sheet. Upload
alongside the manuscript `main.pdf`.

Everything else in this folder (run logs, latents `z_lin.npy`, `ref2_scores.npz`, `.json` results,
`code/`) is **provenance for the GitHub repository / a Zenodo deposit** — cited in the paper's Data
and Code Availability statement, **not** uploaded to the preprint server.

---

## How to read this supplement

| Folder | Supports | Contents |
|---|---|---|
| `S1_mapping_gates/` | Aim 1 — reference mapping, integration, feasibility gates | run log (ELBO, τ thresholds, gate verdicts, class counts), integration QC, label confidence |
| `S2_offmanifold_diagnostics/` | Aim 2 diagnosis — the "structured mixture" (Fig 3) | full D1–D4 diagnostic output |
| `S3_disentanglement/` | Aim 2 negative — the disentanglement wall (Fig 4) | contrastiveVI adversary run logs (baseline and batch-aware) |
| `S4_geometry/` | Aim 3 — off-target geometry (Fig 5) | pairwise-distance results, the background latent `z_lin` and its labels, run log |
| `S5_second_reference/` | Cross-atlas test (Fig 6) | Velmeshev-2023 projection scores, summary statistics, run log |
| `S6_offset_constancy/` | The Aim-3 cancellation test | per-lineage dish-vector consistency result |
| `tables/` | Supplementary Tables S1–S6 | machine-readable CSVs of every summarized quantity |
| `code/` | Reproducibility | the analysis scripts (self-contained copies) |
| `data/` | Single source of truth | `numbers.json` — every in-text number, keyed |

---

## Claim → evidence map

### Aim 1 — a harmonized reference supports confident mapping
- **scVI ELBO 695.6, scANVI 736.1, scArches-surgery 920.2; τ_H = 0.96, τ_R = 1.93; class distribution; Gate A PASS / B GO (cosine 0.855) / C GREEN** → `S1_mapping_gates/pilot_run.log`, summarized in `tables/tableS1_mapping_gates_summary.csv` and `tables/tableS2_offtarget_classes.csv`.
- **Integration iLISI 6.10→8.10, cLISI 1.31→1.44** → `S1_mapping_gates/integration_qc.json`.
- **Per-predicted-type mapping entropy** → `S1_mapping_gates/label_confidence.csv`.
- Code: `code/run_pilot.py`, `code/preflight_check.py`.

### Aim 2 diagnosis — the off-manifold compartment is a structured mixture (Fig 3)
- **D1 marker scores (NPC 1.11→0.61, hypoxia 0.60→0.96, UPR 0.12→0.56, …); D2 potency 0.13→0.78 & age quintiles; D3 protocol spread 0.4%→96.8% (σ=0.28); D4 30-cluster coherence** → `S2_offmanifold_diagnostics/diagnostics_D1-D4.log`.
- Protocol table → `tables/tableS3_protocol_offmanifold.csv`.
- Code: `code/diagnostics.py` (panels/gene-sets defined inline); figure in `code/make_figures.py` (`fig3`).

### Aim 2 negative — lineage cannot be disentangled from source in a shared latent (Fig 4)
- **Origin adversary balanced accuracy 0.998 (baseline, z_iv=10) and 0.998 (batch-aware)** → `S3_disentanglement/ws1_baseline_adversary0.998.log`, `ws1_batchaware_adversary0.998.log` (grep `adversary balanced-acc`). The salient-capacity variant (z_iv=30 → 0.999) is reported in the main text.
- Code: `code/run_ws1.py`, `code/contrastive.py` (the contrastiveVI wrapper + `verify_disentanglement`).

### Aim 3 — off-target geometry (Fig 5)
- **6 well-powered pairs, all `real_difference`, clearing the matched-N floor 14–105×** → `S4_geometry/aim3_geometry.json` (full 136-pair result), summarized in `tables/tableS4_geometry_pairs.csv`; run log `S4_geometry/ws2_geometry_run.log`.
- **Inputs** (fully reproduce Aim 3): `S4_geometry/z_lin.npy` (background latent, 93,795 off-target cells × 15) + `S4_geometry/populations.csv` (aligned labels). Rerun: `python code/run_aim3.py --zlin z_lin.npy --labels populations.csv`.

### Cross-atlas test — genuine in-vitro divergence beyond the reference gap (Fig 6)
- **Clean-calibrated contrast: AUC 0.65, Cliff's δ 0.30, p<1e-300; 15% of ambiguous-novel beyond clean p95** → `S5_second_reference/ref2_stats.json`; raw per-cell scores `ref2_scores.npz`; verdict `ref2_test.json`; run log `ref2_run.log`. Summary: `tables/tableS6_second_reference.csv`.
- Code: `code/ref2_test.py` (joint scVI integration), `code/find_ref2.py` (Census dataset selection).

### The Aim-3 cancellation assumption — directly tested
- **Per-lineage source offsets are inconsistent (mean cosine 0.15 vs 0.855 on-target neural); geometry-preservation cosine 0.29; organoid↔primary distance-matrix correlation r=0.90** → `S6_offset_constancy/dishvec_consistency.json`; summary and 4×4 cosine matrix in `tables/tableS5_offset_constancy.csv`.
- Code: `code/dishvec_consistency.py`.

---

## Supplementary Tables (`tables/`)
- **Table S1** — mapping, training, integration, and gate summary.
- **Table S2** — six-state off-target classification (n and %).
- **Table S3** — protocol-stratified off-manifold burden (26 protocols).
- **Table S4** — Aim-3 well-powered pairwise distances (with CIs and noise floors).
- **Table S5** — offset-constancy: 4×4 dish-vector cosine matrix + geometry-preservation metrics.
- **Table S6** — cross-atlas (Velmeshev-2023) test statistics.

---

## Data availability

Small derived artifacts (latents, scores, logs, tables) are included here. The **large primary
inputs are not** (size): the harmonized fetal reference (`reference.h5ad`, ~12 GB), the annotated
HNOCA query (`hnoca_annotated.h5ad`, ~9 GB), and the trained models (scVI/scANVI/scArches,
contrastiveVI). These are regenerable from public sources — HNOCA and the fetal atlases are on
**CZ CELLxGENE Discover** (HNOCA dataset `3a805b8d-…`; HDBCA `13149914-…`; Velmeshev-2023
`1a38e762-2465-418f-b81c-6a4bce261c34`) — via the pipeline in the repository, or available from the
authors on request / a Zenodo deposit prior to publication.

## Reproducibility notes
- All in-text numbers are keyed in `data/numbers.json`.
- Figures regenerate with `code/make_figures.py` (Fig 1–5) and `code/make_fig6_ref2.py` (Fig 6).
- Software: scvi-tools 1.3.3, scanpy, anndata (see repository `env/environment.yml`); the pure
  pipeline helpers carry a passing unit-test suite (`tests/`) in the repository.
- Provenance: this supplement corresponds to the manuscript on branch `phase1-aim23-manuscript`.

## Citation / archive DOI

This supplement is archived at Zenodo: **https://doi.org/10.5281/zenodo.21459720**

Code repository: https://github.com/LTTakahashi/NullState (branch `phase1-aim23-manuscript`).
