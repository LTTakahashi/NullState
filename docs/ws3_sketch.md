# WS3 — Interpretation Layer (sketch)

**Goal.** Turn the *geometric* result of WS2 — "off-target populations converge toward a shared
default state on the dish-stripped `z_lin` manifold" — into something **mechanistic and named**:
(A) which transcription factors, if perturbed, move cells *out* of that default state (candidate
corrective targets), and (B) *what the default state is* in terms of known biological programs.

This is the least-developed, most data-dependent aim. Below is the design, the dependencies, the
seams that are unit-testable here vs. the parts that need GPU/data, and the honest limits. **No
WS3 code is written yet** — this is a sketch to build against, in the WS1/WS2 pattern (heavy
compute lazy-imported; pure post-processing helpers tested).

---

## Inputs (from WS1 + WS2)

- **The convergent population.** From `scripts/run_aim3.py`: the cell sets in any pair whose
  verdict is `convergent_shared_default` (cross-distance below the Control-3 maturation baseline).
  These are the cells that define the shared default state.
- **`z_lin`** (contrastiveVI background latent, WS1) — for neighbor/state definitions.
- **Expression** — the annotated organoid query (raw counts + log-norm) plus matched primary/
  on-target cells. CellOracle and signature scoring run in gene space, not latent space.

---

## Part A — In-silico levers (CellOracle)

CellOracle (Kamimoto et al., *Nature* 2023) infers a context-specific gene-regulatory network
and simulates transcription-factor perturbations, propagating the effect through the GRN to
predict shifted cell states.

1. **Base GRN.** CellOracle ships a human promoter base GRN, but a *custom* base GRN from
   scATAC peaks + motif scanning is stronger. **Opportunity:** the dominant off-target protocol
   in HNOCA — Treutlein 2023, 61.7% of the off-target mesenchyme — is **10x multiome (has ATAC)**,
   so a custom base GRN is feasible for the load-bearing population.
2. **Context GRN.** Fit per-cluster regularized regression on the organoid expression (off-target
   convergent cells + their on-target neighbors).
3. **Perturb.** Simulate KO (TF expression → 0) and OE for candidate TFs; propagate.
4. **Score.** Project the predicted shift onto a state vector pointing *away from* the convergent
   attractor (toward on-target lineage). **Output: ranked TFs whose perturbation collapses or
   rescues the default state** — the "single corrective intervention" target the Significance
   section envisions, screenable once across systems if the state is truly shared.

**Pure, testable seam:** given CellOracle's per-TF, per-cell shift scores, the *aggregation and
ranking* (mean shift along the rescue vector, FDR over TFs, stability across clusters) is
numpy/pandas → unit-testable. The simulation itself needs CellOracle + data.

---

## Part B — In-vivo program anchoring (signature scoring)

The convergent attractor is, so far, a *region of `z_lin`*. Part B gives it a biological identity
by scoring the convergent cells against curated gene programs and asking which are elevated
relative to on-target and primary cells.

- **Programs to score:** MSigDB **Hallmark** (the repo already names `HALLMARK_GLYCOLYSIS`,
  `HALLMARK_HYPOXIA`, `HALLMARK_UNFOLDED_PROTEIN_RESPONSE` in `config/gene_sets.yaml`), plus
  developmental/identity programs — primitive streak / mesendoderm, EMT, neural crest,
  fibroblast/stromal identity, senescence, generic stress.
- **Method:** `decoupler` (AUCell / ORA / run_ulm) or `scanpy.tl.score_genes` per cell; compare
  convergent vs on-target vs maturation-matched primary.
- **Output:** a named identity, e.g. "the convergent off-target default is a stressed
  stromal/fibroblast-like state (high EMT + glycolysis, low lineage-specific identity)" — and a
  direct tie back to the Control-2/Control-3 stress logic.

**Closes a known gap:** `scripts/run_pilot.py` currently *mocks* the gene sets
(`mock_sets = {k: set() for k in ...}`). WS3 is where real gene-set fetching (gseapy/decoupler/
MSigDB) must land; the `gene_sets.yaml` scaffold is already there to point at.

**Pure, testable seam:** given a cell×program score matrix + population labels, the
per-population mean, the differential ranking (convergent − reference), and significance are
pure pandas/numpy → unit-testable. The score computation needs expression + the gene sets.

---

## Dependencies & data

| Need | Source | Notes |
|------|--------|-------|
| CellOracle | `pip install celloracle` | base GRN ships; custom base GRN needs scATAC |
| scATAC for custom base GRN | HNOCA Treutlein 2023 (10x multiome) | the largest off-target protocol carries ATAC |
| Signature scoring | `decoupler` (+ MSigDB via `decoupler.get_resource`) | or scanpy `score_genes` |
| Gene sets | MSigDB Hallmark + curated dev/stress sets | replaces the mocked sets in run_pilot |
| Inputs | WS1 `z_lin` + WS2 convergent labels + organoid/primary expression | from run_aim3 output |

---

## What's buildable here vs. data-bound

- **Buildable + testable now (numpy/pandas):** the post-processing helpers — `rank_perturbation_targets()`
  (CellOracle shift scores → ranked TFs) and `differential_program_scores()` (score matrix + labels
  → convergent-vs-reference ranking). These can be written and unit-tested in the WS1/WS2 pattern
  without CellOracle/data.
- **Data/GPU-bound (lazy-imported, run on the box):** the CellOracle GRN inference + perturbation
  simulation, and the AUCell/decoupler score computation. These need the expression data, the gene
  sets, and (for the custom base GRN) the multiome ATAC.

Suggested module layout (mirrors `src/geometry`): `src/interpret/{perturbation,signatures}.py`
with lazy heavy imports + pure helpers, `scripts/run_ws3.py` consuming `run_aim3.py`'s convergent
populations.

---

## Honest limitations

- **GRN inference is noisy.** CellOracle TF rankings are *hypotheses* for wet-lab follow-up, not
  validated targets; report them as such.
- **Signature identity is interpretive.** The named program depends on the gene-set choice; report
  the scores and let the identity follow the evidence, not a hoped-for label.
- **Depends on a positive WS2 result.** Part A/B are most meaningful when WS2 actually returns
  `convergent_shared_default`. If WS2 says lineage-bound/divergent, WS3 instead characterizes each
  germ layer's distinct failure program — still useful, but a different framing.
- **Single-system caveat carries over.** On the neural vertical alone, WS3 describes the neural
  off-target default; the cross-system "shared target" claim needs Phase 2.

---

## Suggested first step

Once WS2 has run on real `z_lin` and a convergent population exists, start with **Part B signature
scoring** (cheaper, no GRN, immediately interpretable) to name the state, *then* CellOracle to find
the levers. Build the two pure post-processing helpers + tests first (doable here), so the heavy
compute slots into a verified harness.
