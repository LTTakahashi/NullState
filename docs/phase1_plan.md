# NullState — Phase 1 Implementation Plan & Readiness Assessment

**Scope:** Phase 1 = the *neural vertical, end-to-end* (per `reports/nullstate_pilot_report.tex` §7), planned now with an explicit bridge to the cross-germ-layer expansion (Phase 2).
**Status of foundation:** Pilot complete; all three gates passed; instrument validated.
**Date:** 2026-06-09 · **Author of record:** L. Takahashi (WSU)

---

## 0. TL;DR — the honest verdict

**The instrument is real and validated. The science of Phase 1 is greenfield.**

The pilot did exactly what a pilot should: it built a harmonized cross-atlas reference, proved the integration *worked* (not just ran), and passed all three go/no-go gates on neural organoids. That foundation — reference, trained scVI/scANVI, a behaving off-target classifier, config-driven pipeline, QC instruments — is a genuine asset you can build on today.

But Phase 1's *load-bearing deliverables* — the disentanglement (contrastiveVI), the maturation control (Control 3), and the entire Aim-3 geometry machinery (sliced-Wasserstein, N_min, matched-N null floors, power simulation) — are **0% implemented**. They are well-specified in the two `docs/research_strategy_*` reports, but specification is not code. So:

- **Readiness to *start*: high.** The runway is clear, the methods are specified, the data foundation exists, the failure modes are already mapped.
- **Readiness as in "most of the work is done": low.** Four of the five Phase 1 items are net-new builds. One is cheap.
- **The single gating dependency** is sourcing true raw UMI counts for the HNOCA query. It blocks the highest-consequence fix (hardening the off-target boundary), and it carries external-data-availability risk that is not fully under your control.

One-line recommendation: **proceed**, but front-load the raw-counts sourcing immediately and in parallel — if it stalls, the rest of Phase 1 inherits a soft-edged off-target definition.

### Progress log

- **2026-06-12 — WS0b classifier hardening: DONE.** scANVI now early-stops on the validation plateau (config `scanvi_early_stopping`, mirrors the scArches pattern), fixing the documented over-training; an empirical (non-uniform) label-prior correction is wired through calibration + scoring, **off by default** (`empirical_label_prior`) pending a retrain to re-validate. (`train_scanvi.py`, `compute_scores.py`, `params.yaml`)
- **2026-06-12 — §9 forward-compat refactors: DONE.** `classify_cell` now honors `expected_germ_layer` (on-target origin set per system; neural default byte-identical, +8 tests). HVG selection is config-driven (`n_hvg`/`hvg_flavor`). Phase 2 data slots (`heoca`, `kidney_organoid`) registered in `paths.yaml`/`data/README.md` with `germ_layer` and `status: phase2_not_sourced`. → adding a germ layer in Phase 2 is now config + data, not code surgery.
- **2026-06-12 — §10 method reconciliation: DONE.** Aim 2 wording aligned to contrastiveVI-primary across `specific_aims`, `research_strategy_foundation`, `significance`, `README` (see `docs/aim2_method_reconciliation.md`).
- **2026-06-12 — packaging fix.** `src/mapping/__init__.py` no longer forces `scvi` on import of the pure classifier; unit tests run with pandas only (18/18 green).
- **2026-06-12 — WS0a sourcing: documented.** `docs/ws0a_data_sourcing.md`. HNOCA raw counts → pull from the CELLxGENE copy (schema requires raw counts) not the Zenodo cleaned file. HEOCA id **confirmed** (`b4d13dc2-…`, ships raw counts) and filled in `paths.yaml`. Kidney = no single atlas; Subramanian 2019 census as the assembly seed.
- **2026-06-12 — WS1 contrastiveVI: scaffolded (not run).** `src/disentangle/contrastive.py` (background=primary on-target → z_lin, salient → z_iv; adversary = verification check), API-checked against scvi-tools docs, pure helper unit-tested. Needs scvi/GPU/data to run.
- **2026-06-12 — WS2 geometry: implemented + unit-tested.** `src/geometry/` — sliced-Wasserstein + debiased Sinkhorn (`distances.py`), rarefaction→N_min + matched-N self-distance floor + bootstrapped cross-distance + verdict incl. Control-3 baseline + BH-FDR (`matched_n.py`), simulation power curves (`power.py`). Pure numpy/scipy, runs on z_lin.
- **2026-06-12 — WS2 topology: H₀ done, H₁ exploratory.** `src/geometry/topology.py` — H₀ persistence via the Euclidean MST (single-linkage; **no TDA dependency**) + Fasy bootstrap confidence band + significant-feature test (confirmatory, unit-tested); H₁ loops via lazy-imported `ripser` (exploratory, optional dep) with threshold/k-NN sensitivity.
- **2026-06-12 — WS1→WS2 wired end-to-end.** `src/geometry/pipeline.py::run_geometry_over_populations` (pure: rarefaction→N_min → all pairwise matched-N tests → Control-3 verdicts → BH-FDR) and `scripts/run_aim3.py` (gets z_lin from a contrastiveVI model [lazy scvi] or a precomputed `.npy`, writes JSON). Pipeline core unit-tested on synthetic 3-population data.
- **2026-06-12 — full suite green (54 passed).** Also decoupled `src/calibration` + `src/gates` from anndata-at-import (pure helpers run without the DL stack). **Bugfix:** `evaluate_go_nogo` averaged column means, not entries → `.stack().mean()` (true pairwise-cosine mean). ⚠ **PENDING:** this is the statistic behind Gate B "0.906" — recompute on the next dish-vector run and update `reports/nullstate_pilot_report.tex` + `README.md` if it shifts (GO decision unaffected; 0.906 ≫ 0.70).
- **2026-06-12 — WS3 interpretation: sketched.** `docs/ws3_sketch.md` — CellOracle in-silico TF perturbation (custom base GRN feasible from the Treutlein-2023 multiome ATAC) + in-vivo program anchoring (decoupler/MSigDB Hallmark; closes the mocked-gene-set gap in `run_pilot`). Heavy compute is data-bound; the pure post-processing helpers are buildable here.
- **2026-06-12 — WS3 pure helpers built.** `src/interpret/{perturbation,signatures}.py` — `rank_perturbation_targets` (CellOracle per-cell rescue shifts → ranked TFs; one-sample t-test + BH) and `differential_program_scores` (cell×program scores + labels → convergent-enriched programs; Mann-Whitney U + AUC + BH). Pure numpy/pandas/scipy, unit-tested. CellOracle/decoupler compute remains data-bound.
- **2026-06-12 — CPU runbook scripts.** `scripts/smoke_test_cpu.py` (synthetic WS1→WS2 end-to-end on CPU in minutes; self-diagnoses the ContrastiveVI `early_stopping` kwarg before any GPU spend), `scripts/fetch_verify_cellxgene.py` (download HNOCA/HEOCA from CELLxGENE + integer raw-count check = WS0a step 1), `scripts/geosketch_subsample.py` (WS4 scale lever: heterogeneity-preserving subsample to fit a small GPU / laptop RAM). Data/scvi-bound → run on the box; untested by the assistant.

---

## 1. What Phase 1 is — and what it is not

Phase 1 deepens the **single neural system** from *feasibility* (pilot) to *mechanism*. It does **not** yet test the cross-germ-layer convergence hypothesis — that is Phase 2. This boundary matters: a reviewer (or you, in six months) should not mistake a polished neural vertical for the cross-system result the project exists to deliver.

The five Phase 1 items, in the report's order of *scientific consequence*:

| # | Item | Type | Why it ranks here |
|---|------|------|-------------------|
| 1 | Source true raw UMI counts → re-enable scArches surgery | **Fix (blocking)** | The 37.7% `ambiguous_novel` is not cosmetic — it likely hides real off-target cells. The off-target boundary stays soft until this is done. |
| 2 | Harden the classifier (early-stop + non-uniform prior) | **Hygiene (cheap)** | ~2h. Cleans up rare-type labels. Not the crux (mesenchyme is abundant, moderate-entropy). |
| 3 | Control 3 (maturation-matched baseline) + contrastiveVI dish-stripping | **Centerpiece** | Converts Gate B from "encouraging" (0.906 cosine proxy) to "defensible." This is the methodological heart. |
| 4 | CellOracle in-silico levers + in-vivo program anchoring | **Interpretation** | Turns a distance into an actionable, mechanistic result. |
| 5 | Build for scale (geosketch, backed/lean I/O) | **Cross-cutting** | The 251 GB container ceiling is real and already bit us once. |

**Out of scope for Phase 1 (→ Phase 2):** HEOCA (endoderm) and kidney (mesoderm) data, the unified cross-germ-layer HVG, the generalized classifier, and the actual cross-germ-layer EMD comparison.

---

## 2. Readiness scorecard

Legend: ✅ done/validated · 🟡 partial/needs work · 🔴 not started · **Blocker** = what must be true to begin.

### Foundation (the pilot's deliverables)

| Component | Status | Evidence | Note |
|-----------|--------|----------|------|
| Harmonized neural reference (1.90M cells, 34 types, 4k HVG, raw counts) | ✅ | Built; `reference.h5ad` (10.8 GB) on compute volume | The Aim-1 instrument. |
| Cross-atlas integration verified | ✅ | iLISI 8.04 vs 5.94; cLISI 1.44 vs 1.39 (`integration_qc.json`) | Donors mix, biology preserved. Genuinely worked. |
| scVI latent (n=30, 400 ep) | ✅ | ELBO 552.2, train–val gap −0.78 (no overfit) | Clean geometric substrate. |
| scANVI classifier (200 ep) | 🟡 | Val ELBO degraded ~15 past min (581→596) | Mildly over-trained → item #2. |
| Off-target classifier + neural-crest guard | 🟡 | `src/mapping/classify.py`, real & tested | `expected_germ_layer` now wired (✅ 2026-06-12, +8 tests). Off-target boundary still soft until the WS0a counts fix (#1). |
| Pipeline, config, retrieval, QC, tests | ✅ | `run_pilot.py`, 3 retrieval strategies, `params.yaml`, 2 test files | Well-engineered; 8 defects already fixed. |
| Gate results (A/B/C) | ✅ | PASS / GO (0.906) / GREEN (7,568, 4 protocols) | Most favorable branch of the decision tree. |

### Phase 1 deliverables (the new work)

| Component | Status | Blocker / dependency | Effort |
|-----------|--------|----------------------|--------|
| **#1** Raw UMI counts for query | 🔴 | Locate per-dataset count matrices behind HNOCA (50+ source studies) | M–L (data-sourcing risk) |
| **#1** Re-enabled scArches surgery | 🔴 | Depends on raw counts; `scarches_max_epochs: 0 → ~100` | M (compute on 1.77M cells) |
| **#2** Classifier hardening | ✅ | done 2026-06-12 (early-stop; empirical prior wired, off by default) | S |
| **#3** contrastiveVI disentanglement | 🔴 | Needs on-target organoid + primary sets (have these) | L (new method) |
| **#3** Control 3 maturation baseline (CytoTRACE2 + velocity latent-time) | 🔴 | Depends on z_lin from contrastiveVI | L |
| **#2 (Aim 3)** Geometry: sliced-Wasserstein + Sinkhorn | ✅ | `src/geometry/distances.py` (2026-06-12, unit-tested) | M |
| Rarefaction → empirical N_min | ✅ | `src/geometry/matched_n.py` (consumes `rarefaction_ladder`) | S–M |
| Matched-N null-floor protocol + bootstrap CI | ✅ | `src/geometry/matched_n.py` (2026-06-12, unit-tested) | M |
| Simulation-based power curves | ✅ | `src/geometry/power.py` (2026-06-12, unit-tested) | M |
| H₀ persistence (confirmatory) / H₁ (exploratory) | ✅ / 🟡 | H₀ ✅ `topology.py` (pure MST, no dep, unit-tested); H₁ 🟡 lazy `ripser` (optional, exploratory) | M |
| **#4** CellOracle + in-vivo anchoring | 🟡 | pure ranking helpers ✅ `src/interpret/` (unit-tested); CellOracle/decoupler compute data-bound (sketch: `docs/ws3_sketch.md`) | M–L |
| **#5** geosketch subsampling / backed I/O | 🟡 | Lean I/O exists; geosketch does not | S–M |
| Compute (H100, 251 GB cgroup) | 🟡 | Available; ceiling is a known, recurring constraint | — |

**Bottom line:** foundation ≈ ready; Phase 1 scientific substance ≈ scaffolding + specs only. This is a build, not a tweak.

---

## 3. The asset base — what Phase 1 stands on

1. **A validated reference + integration.** The hardest, most failure-prone part of Aim 1 (re-integrating from raw counts so the manifold isn't biased) is done and QC-verified for the neural layer. contrastiveVI and the geometry all run *on top of* this latent — they don't require rebuilding it.
2. **A behaving off-target definition.** The classifier's decision tree, the data-derived thresholds (τ_H @95th, τ_R @99th on held-out reference), and the neural-crest guard are implemented and unit-tested. The endoderm branch already exists in code (forward-looking).
3. **Honest, mapped failure modes.** The pilot report's §6 limitations *are* the Phase 1 task list. You are not starting from "what could go wrong?" — you start from a list of the specific things that did.
4. **Engineering discipline.** Config-driven params, three retrieval fallbacks, memory-aware lean readers, QC instruments. The 251 GB ceiling, the gene-namespace trap, the donor_id batch key — all already paid for.

---

## 4. The gap — what is missing, precisely

- **Raw counts.** HNOCA ships only log-norm + length-normalized layers; the ZINB likelihood is mis-specified, so scArches surgery diverges to NaN and was disabled (`scarches_max_epochs: 0`, with a code comment that names this as the Phase 1 fix). Without surgery the query is *projected only*, inflating `ambiguous_novel` to 37.7% and plausibly **undercounting** `true_offtarget` (17.8%).
- **No disentanglement.** The dish-vector cosine (0.906) is a mean-difference *smell test*, not the disentangled z_iv/z_lin the method needs. There is no contrastiveVI, no adversary, no flow-matching code.
- **No maturation control.** No CytoTRACE2, no RNA-velocity latent-time, no matched-primary baseline. Gate B's consistency could partly be shared normalization/technical structure — Control 3 is what rules that out.
- **No Aim-3 geometry.** None of sliced-Wasserstein, debiased Sinkhorn, EMD, persistent homology, rarefaction-driven N_min, the matched-N null floor, or the power simulation exists. The config holds a `rarefaction_ladder` that nothing consumes.
- **No interpretation layer.** No CellOracle, no signature scoring against in-vivo programs.
- **Partial scale infra.** Lean I/O exists; geosketch subsampling and backed-mode AnnData do not.

---

## 5. The Phase 1 plan — sequenced workstreams

Effort sizes are relative T-shirts (S/M/L), not calendar promises. Dependencies, not dates, drive the order.

### WS0 — Front-load in parallel (start immediately)

**WS0a · Raw UMI counts + scArches surgery (highest consequence).**
- Locate the per-dataset raw count matrices behind HNOCA. Path of least resistance: pull the individual source studies from CELLxGENE Census / GEO rather than the integrated HNOCA object (which dropped raw counts). Reconcile to the reference HVG/Ensembl namespace (the pilot already solved symbol→Ensembl; reuse it).
- Validate that a true integer-count layer reproduces a well-behaved ZINB (no NaN on a small surgery test) **before** the full run.
- Re-enable surgery (`scarches_max_epochs ≈ 100`, `weight_decay 0.0`), re-map, re-score, re-classify.
- **Internal gate G1:** measure the migration of cells `ambiguous_novel → true_offtarget`. Report the revised off-target count and boundary stability. *Success = the off-target population is no longer projection-limited.*
- Effort: **M–L.** Risk: data availability is partly external.

**WS0b · Harden the classifier (cheap hygiene).**
- Add best-validation checkpoint / early-stopping to scANVI; replace the uniform class prior with an empirical (cell-frequency) prior.
- Re-emit the label-confidence report; confirm rare-type entropy improves.
- Effort: **S (~2h).** No dependency — do it first, it's upstream of everything and free.

### WS1 — The scientific centerpiece (Control 3 + contrastiveVI)

- **contrastiveVI dish-stripping** as the *primary* disentanglement: background = primary on-target cells, target = organoid on-target cells → an identifiable, stable z_lin (lineage) and z_iv (in-vitro). Because Gate B is **GO**, build the *single shared* z_iv (not the conditional z_iv|y_id). Demote the adversary to a verification check (can it predict organoid-vs-primary from z_lin? it should fail).
- **Control 3 maturation baseline:** score maturation on the *dish-cleaned* z_lin using CytoTRACE2 **and** an RNA-velocity latent-time, keep only concordant-rank cells, and build the maturation-matched primary baseline (same broad class, two primary tissues, subsampled to match maturation).
- **Verification of disentanglement:** confirm z_iv predicts culture conditions and z_lin predicts identity but not organoid origin (`research_strategy_foundation.md` §2).
- **Internal gate G2:** Gate B is re-asserted on the disentangled representation *with* the maturation baseline. *Success = the dish signature is shown to be real biology, not processing structure — the claim a reviewer can no longer dismiss.*
- Effort: **L.** This is the heart; budget accordingly.

### WS2 — Aim-3 geometry machinery (the measurement)

Build on z_lin from WS1. Implement exactly the feasibility doc's protocol — do not interpret raw distances.
- Sliced-Wasserstein (primary, n^(−1/2) convergence) + debiased Sinkhorn divergence (cross-check) on the leading 15–30 z_lin dims.
- Rarefaction on the abundant brain-derived mesenchyme → empirical **N_min** (consume the `rarefaction_ladder` config).
- Matched-N null-floor protocol: self-distance floor, cross-distance, B=1,000 bootstrap CI; a difference is real only if its CI clears the floor.
- Simulation-based power curves (convergent vs divergent synthetic scenarios) → minimum detectable effect at the available N.
- H₀ persistence with Fasy-style confidence bands (confirmatory); H₁ exploratory only.
- **Internal gate G3:** N_min fixed; off-target mesenchyme confirmed above N_min for confirmatory claims. *Success = the measurement is self-calibrating and N-honest.*
- Effort: **M** (well-specified; mostly faithful implementation).

### WS3 — Interpretation

- **CellOracle** in-silico perturbation on the convergent off-target population (which TFs move cells off the default state?).
- **In-vivo program anchoring:** score the off-target state against known developmental/stress signatures so the geometry maps onto biology.
- Effort: **M–L.** Depends on a defined convergent population from WS1–WS2.

### WS4 — Scale & reproducibility (cross-cutting, from day one)

- geosketch subsampling for any whole-atlas operation; backed-mode AnnData; keep the lean-write discipline.
- Respect the 251 GB cgroup ceiling explicitly in every step that touches 1.77M cells (surgery, scoring).
- Effort: **S–M**, amortized across the other workstreams.

---

## 6. Phase 1 internal decision tree

```
WS0b classifier hardening ──► (always proceed; cheap)
WS0a raw counts ─► G1: off-target boundary hardened?
        ├─ YES ─► off-target set finalized for WS1–WS3
        └─ NO (counts unobtainable) ─► proceed with projection-only,
              report off-target count as a *lower bound*, flag soft edge
WS1 contrastiveVI + Control 3 ─► G2: dish signature = real biology?
        ├─ YES ─► geometry runs on defensible z_lin
        └─ NO  ─► report disentanglement limit; geometry becomes exploratory
WS2 geometry ─► G3: mesenchyme ≥ N_min?
        ├─ YES ─► confirmatory convergence claim is licensed
        └─ NO  ─► demote to exploratory; the powered core still stands on pooled mesenchyme
```

No branch produces *nothing* — the same property that made the pilot safe. The worst case (counts unobtainable + thin power) still yields a maturation-controlled, honestly-bounded neural off-target characterization.

---

## 7. Risks & mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Raw HNOCA counts unobtainable / partial | **High** | Source per-study from Census/GEO, not the integrated object; if partial, proceed projection-only and report off-target as a lower bound (G1 branch). |
| Gate B consistency is a normalization artifact | **High** | This is exactly what Control 3 + contrastiveVI tests (WS1/G2). Do not publish disentanglement without it. |
| Off-target mesenchyme concentration (61.7% in one study, 72% top-2) | Medium | Pooled + top-protocol analyses are well powered; report protocol-stratified (Control 2) results with the concentration caveat; treat broad cross-protocol comparison as thin. |
| Small-N geometry inflation (Wasserstein n^(−1/d) bias) | Medium | Never interpret raw distances; matched-N null floor cancels the shared bias (WS2). |
| scANVI over-training contaminates labels | Low–Med | WS0b (best-val checkpoint + empirical prior); mesenchyme sits at moderate entropy, so the load-bearing class is largely unaffected. |
| 251 GB container OOM on 1.77M-cell surgery | Medium | WS4 from the start: backed I/O, geosketch, per-type subsampling. |
| Doc/method drift (adversarial vs contrastiveVI) | Low | ✅ Resolved 2026-06-12 — docs reconciled to contrastiveVI-primary (`docs/aim2_method_reconciliation.md`). |

---

## 8. Compute & data prerequisites

- **Compute:** single H100 sufficed for the pilot (scVI 126 min, scANVI 130 min). scArches surgery on 1.77M cells + contrastiveVI training are heavier; assume multi-GPU-day budgets per major run and respect the **251 GB cgroup cap** (not the 2 TB host).
- **Data on hand:** HNOCA (query), HDBCA + Cao fetal subset (reference) — present and used.
- **Data to source (Phase 1):** per-dataset **raw UMI counts** behind HNOCA (the WS0a blocker).
- **Data to source (Phase 2):** HEOCA (endoderm), kidney organoid benchmarks (mesoderm), broader fetal/adult primary covering all three germ layers.

---

## 9. The bridge to Phase 2 (cross-germ-layer expansion)

Phase 1 is the proving ground; Phase 2 is the hypothesis. To avoid rework, make Phase 1 **forward-compatible** with a few small refactors done *now*:

**Refactor during Phase 1 (cheap insurance):**
- **Wire `expected_germ_layer` through `classify_cell`** — it is currently accepted but ignored, hardcoding the neural-organoid assumption. Parameterize the on-target origin set per system: neural organoid → {neural, neural_crest}; kidney → {mesoderm}; intestinal/HEOCA → {endoderm}.
- **Make HVG selection swappable.** The current 4k HVG is neural-reference-biased. Keep the selection behind config so a *unified ecto/endo/meso* HVG can drop in for Phase 2 without touching the pipeline.
- **Generalize the reference builder** to accept multiple organoid queries and multiple germ-layer fetal anchors (the pilot already concatenates HDBCA + a fetal subset — extend the same pattern).
- **Add HEOCA + kidney dataset slots** to `config/paths.yaml` and `data/README.md` now (even if unfetched), so the data contract is explicit.

**Net-new for Phase 2 (do not attempt in Phase 1):**
- Source HEOCA + kidney atlases and a broader 3-germ-layer primary reference.
- Build the unified cross-germ-layer HVG and re-integrate (the Aim-1 instrument, generalized).
- Run the actual convergence test: EMD between off-target mesenchyme across neural vs kidney vs endodermal systems, read against the Control-3 matched-primary baseline (`research_strategy_foundation.md` §3) — *this* is the structured-mixture hypothesis.

**What carries over for free:** the contrastiveVI "calibration without anchoring" method, the entire WS2 geometry machinery (already germ-layer-agnostic), the matched-N protocol, and the classifier decision tree. Phase 1 is, deliberately, a single-germ-layer dress rehearsal of the full assay.

---

## 10. A note on method drift (read before coding WS1)

> **✅ RESOLVED 2026-06-12.** The reconciliation below was applied across the four docs; see `docs/aim2_method_reconciliation.md`. Retained for context.

The two `docs/research_strategy_*` reports describe Aim 2 as **adversarial subspace separation** (with contrastiveVI/flow-matching as benchmarked alternatives). The pilot report's Phase 1 plan **inverts this**: contrastiveVI becomes the *primary, identifiable, stable* method and the adversary is demoted to a verification check. This is a deliberate de-risking (adversarial training is unstable; contrastiveVI is not). **Recommendation:** before WS1, update `research_strategy_foundation.md` §2 and the `specific_aims` Aim 2 wording to reflect contrastiveVI-primary, so the proposal, the report, and the code agree. Small edit, prevents a reviewer (or future-you) from noticing the three documents disagree.

---

## 11. How to start — the first concrete moves

In rough order, the lowest-regret opening sequence:

1. ✅ **DONE (2026-06-12):** WS0b classifier hardening — early-stopping + empirical prior (off by default). See progress log (§0).
2. ▶ **IN PROGRESS:** scope WS0a — locating the per-study raw counts behind HNOCA (Census/GEO) and the HEOCA/kidney collection IDs. See `docs/ws0a_data_sourcing.md`. The on-GPU surgery NaN-test stays for your environment.
3. ✅ **DONE (2026-06-12):** §9 forward-compatibility refactors (`expected_germ_layer` wired, HVG config-ized, Phase 2 data slots added).
4. ▶ **NEXT:** stand up WS1 (contrastiveVI on the existing on-target sets) — the centerpiece. Scaffolded in `src/disentangle/` (untested; needs the scvi/GPU env).
5. ✅ **DONE (2026-06-12):** docs reconciled to contrastiveVI-primary (§10).

---

## Appendix — source map

- Phase 1 definition: `reports/nullstate_pilot_report.tex` §7.
- Off-target definition, z_iv consistency, Control 3: `docs/research_strategy_foundation.md`.
- Sample size, sliced-Wasserstein, matched-N null floor, power: `docs/research_strategy_feasibility_power.md`.
- Aims & impact: `docs/specific_aims_organoid_offtarget_geometry.md`.
- Significance/innovation, contrastiveVI rationale: `docs/significance_and_innovation.md`.
- Pilot execution (gates, classifier logic): `docs/hnoca_pilot_execution_plan.md`.
- Config of record: `config/params.yaml` (note the `scarches_max_epochs` Phase 1 comment).
- Classifier: `src/mapping/classify.py` (note unused `expected_germ_layer`).
- Pilot artifacts: `reports/pilot_artifacts/{integration_qc,training_progress}.json`, `label_confidence.csv`.
