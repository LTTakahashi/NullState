# Changeset — commit plan for this session (refreshed)

`git` isn't reachable from the assistant's environment, so run these yourself. The plan groups
the session into **7 file-clean commits** (each file in exactly one commit) in dependency order.
A single-commit fallback is at the bottom.

**Before committing — full suite is green (numpy/pandas/scipy; no scvi/GPU needed):**

```bash
cd <repo> && PYTHONPATH=. pytest tests/ -q     # 54 passed
```

> `config/params.yaml` accumulated keys from several concerns (HVG, scANVI early-stopping,
> `expected_germ_layer`, empirical prior, contrastiveVI, geometry, topology). To keep
> one-file-one-commit it is committed once in **C3**; the `contrastive_*` / `geometry_*` /
> `topology_*` keys it also carries belong conceptually to **C6** / **C7**.

---

## C1 — docs: Phase 1 plan + readiness

```bash
git add docs/phase1_plan.md
git commit -m "docs(phase1): implementation plan + readiness assessment

Phase 1 = the neural vertical (pilot report §7): readiness scorecard, sequenced
workstreams, internal decision gates, risks, the Phase 2 bridge, and a dated progress log
tracking WS0b / §9 / WS0a / WS1 / WS2 as completed."
```

## C2 — docs: reconcile Aim 2 to contrastiveVI-primary

```bash
git add docs/specific_aims_organoid_offtarget_geometry.md \
        docs/research_strategy_foundation.md \
        docs/significance_and_innovation.md \
        README.md \
        docs/aim2_method_reconciliation.md
git commit -m "docs(aim2): reconcile to contrastiveVI-primary disentanglement

contrastiveVI is the primary, identifiable method; the origin-predicting adversary is a
verification check; flow-matching kept as a benchmarked alternative. No change to go/no-go
logic or controls. Includes the redline record."
```

## C3 — feat(mapping): WS0b hardening + forward-compat + scvi-free imports

```bash
git add src/mapping/classify.py \
        src/mapping/train_scanvi.py \
        src/mapping/compute_scores.py \
        src/mapping/__init__.py \
        scripts/run_pilot.py \
        tests/test_classify.py \
        config/params.yaml
git commit -m "feat(mapping): Phase 1 WS0b hardening + forward-compat refactors

- classify: honor expected_germ_layer (on-target origin set per system); neural default
  byte-identical; +8 tests; KNOWN_ORIGINS sourced from origin_map.
- train_scanvi: scANVI early-stops on validation plateau (config); HVG selection
  config-driven (n_hvg/hvg_flavor). Defaults reproduce the pilot.
- compute_scores: optional empirical (non-uniform) label prior, OFF by default, applied
  identically in calibration + scoring.
- run_pilot: thread expected_germ_layer + class_prior.
- mapping/__init__ + classify: pure classifier imports without scvi/anndata.
- params.yaml: HVG, scanvi_early_stopping(+patience), expected_germ_layer,
  empirical_label_prior, plus contrastive_*/geometry_*/topology_* keys used by C6/C7.

Only affects future runs/retrains; no existing artifact altered."
```

## C4 — fix(calibration,gates): scvi-free imports + go/no-go mean

```bash
git add src/calibration/dish_vector.py src/gates/count_check.py
git commit -m "fix(calibration,gates): import without anndata; correct Gate B mean

- dish_vector, count_check: anndata is TYPE_CHECKING-only and yaml is lazy, so the pure
  helpers (compute_cosine_matrix, evaluate_go_nogo, evaluate_gate) import with numpy/pandas
  only; full test suite now collects without the DL stack.
- evaluate_go_nogo: BUGFIX — '.mean().mean()' averaged the column means, not the
  upper-triangle entries (wrong for an uneven triangle). Now '.stack().mean()' = true mean
  pairwise cosine. NOTE: this is the statistic behind the pilot's Gate B '0.906'; recompute
  on the next dish-vector run (GO is robust, the value may shift slightly)."
```

## C5 — chore(data): WS0a sourcing — Phase 2 slots + HEOCA id

```bash
git add config/paths.yaml data/README.md docs/ws0a_data_sourcing.md
git commit -m "chore(data): WS0a sourcing — Phase 2 germ-layer slots + HEOCA id

heoca (endoderm) + kidney_organoid (mesoderm) registered with germ_layer + status; HEOCA
CELLxGENE id confirmed (b4d13dc2-…, ships raw counts). Kidney left TODO (Subramanian 2019
census seed). WS0a doc: pull HNOCA raw counts from the CELLxGENE copy, then re-enable
scArches surgery."
```

## C6 — feat(disentangle): contrastiveVI scaffold (Aim 2)

```bash
git add src/disentangle/__init__.py src/disentangle/contrastive.py tests/test_contrastive.py
git commit -m "feat(disentangle): scaffold contrastiveVI dish-stripping (Aim 2)

background = primary on-target -> z_lin (lineage), salient -> z_iv (in-vitro); adversary =
verification check. Heavy deps lazy-imported so the pure index helper unit-tests with numpy
only. NOT YET RUN (needs scvi/GPU/data). Config keys added in C3."
```

## C7 — feat(geometry): Aim 3 measurement layer (WS2)

```bash
git add src/geometry/__init__.py src/geometry/distances.py src/geometry/matched_n.py \
        src/geometry/power.py src/geometry/topology.py src/geometry/pipeline.py \
        scripts/run_aim3.py \
        tests/test_geometry.py tests/test_topology.py tests/test_aim3_pipeline.py
git commit -m "feat(geometry): Aim 3 measurement layer — distances, matched-N, power, topology, wiring

- distances: sliced-Wasserstein (primary) + debiased Sinkhorn (cross-check).
- matched_n: rarefaction->N_min, self-distance floor, bootstrapped cross-distance, verdict
  vs floor + Control-3 baseline, BH-FDR.
- power: simulation-based power curves + min detectable effect.
- topology: H0 persistence via Euclidean MST (no TDA dep) + Fasy band (confirmatory);
  H1 via optional lazy ripser (exploratory).
- pipeline + run_aim3.py: WS1->WS2 end-to-end (z_lin + labels -> per-pair verdicts -> JSON).
- Pure numpy/scipy; 33 unit tests. Runs on z_lin without scvi/GPU."
```

## C8 — feat(interpret): pure WS3 ranking helpers (Aim 3 interpretation)

```bash
git add src/interpret/__init__.py src/interpret/perturbation.py src/interpret/signatures.py \
        tests/test_interpret.py docs/ws3_sketch.md
git commit -m "feat(interpret): pure WS3 ranking helpers + sketch

- perturbation.rank_perturbation_targets: CellOracle per-cell/per-TF rescue shifts -> ranked
  TFs (mean shift, one-sample t-test, BH-FDR).
- signatures.differential_program_scores: cell x program score matrix + labels -> programs
  elevated in the convergent state (Mann-Whitney U, common-language AUC, BH-FDR).
- Pure numpy/pandas/scipy (heavy CellOracle/decoupler compute stays in the data-bound run
  script); unit-tested. docs/ws3_sketch.md is the design. (ws3_alpha added to params in C3.)"
```

## C9 — chore(scripts): CPU runbook (smoke-test, fetch/verify, geosketch)

```bash
git add scripts/smoke_test_cpu.py scripts/fetch_verify_cellxgene.py scripts/geosketch_subsample.py
git commit -m "chore(scripts): CPU runbook — WS1->WS2 smoke-test, CELLxGENE fetch+verify, geosketch

- smoke_test_cpu.py: synthetic end-to-end WS1->WS2 on CPU in minutes; self-diagnoses the
  ContrastiveVI early_stopping kwarg -- validates the untested scvi path before GPU spend.
- fetch_verify_cellxgene.py: download a CELLxGENE collection dataset (HNOCA default) and check
  for integer raw counts (WS0a step 1).
- geosketch_subsample.py: heterogeneity-preserving subsample to fit a small GPU / laptop RAM (WS4).
Data/scvi-bound; run on the box."
```

---

## This session's files

**New:** `docs/phase1_plan.md`, `docs/aim2_method_reconciliation.md`, `docs/ws0a_data_sourcing.md`,
`docs/ws3_sketch.md`, `src/disentangle/{__init__,contrastive}.py`, `tests/test_contrastive.py`,
`src/geometry/{__init__,distances,matched_n,power,topology,pipeline}.py`, `scripts/run_aim3.py`,
`tests/{test_geometry,test_topology,test_aim3_pipeline}.py`,
`src/interpret/{__init__,perturbation,signatures}.py`, `tests/test_interpret.py`,
`scripts/{smoke_test_cpu,fetch_verify_cellxgene,geosketch_subsample}.py`,
`docs/CHANGESET.md` (this file).

**Modified:** `README.md`, `docs/{specific_aims_organoid_offtarget_geometry,research_strategy_foundation,significance_and_innovation}.md`,
`src/mapping/{classify,__init__,train_scanvi,compute_scores}.py`, `scripts/run_pilot.py`,
`tests/test_classify.py`, `config/{params,paths}.yaml`, `data/README.md`,
`src/calibration/dish_vector.py`, `src/gates/count_check.py`.

> `docs/ws3_sketch.md` is a planning doc (commit it with C1-style docs if you want it tracked,
> or leave it). `docs/CHANGESET.md` is scratch — don't commit it; delete after.

---

## Single-commit fallback

```bash
git add -A
git commit -m "feat: Phase 1 WS0b + forward-compat + Aim 2 reconciliation + contrastiveVI scaffold + Aim 3 geometry + WS0a sourcing

WS0b: scANVI early-stopping + empirical prior (off by default).
Forward-compat: expected_germ_layer, config HVG, Phase 2 data slots (+ HEOCA id).
Aim 2: docs reconciled to contrastiveVI-primary; contrastiveVI scaffold.
Aim 3 (WS2): sliced-Wasserstein/Sinkhorn, matched-N null floor, power, H0 topology, end-to-end wiring.
Fixes: pure imports without scvi/anndata across mapping/calibration/gates; evaluate_go_nogo true mean.
Tests: 54 passing (numpy/pandas/scipy)."
```
