# Reproducing the paper

Entry point for reviewers. Every headline number in the paper maps to one script
below. **All result CSVs are committed**, so each analysis can be re-run in
seconds without repeating the sweeps that produced them; the sweep commands are
given for anyone who wants to regenerate the data from scratch.

```bash
pip install -r requirements.txt      # numpy, scipy, scikit-learn, torch, pandas, matplotlib
```

Everything is CPU-only and synthetic. No data download, no GPU, no credentials.

---

## Start here (about 5 minutes, no training)

These three reproduce the paper's core argument and need no model fitting at all.

| command | reproduces | runtime |
|---|---|---|
| `python -m analysis.taxonomy` | **Fig. 1** — the estimand × failure-mode table, *and* the identity gap ≡ CCA−MCC (max dev 1e-4) | ~3 min |
| `python -m analysis.geometry_constant` | §4 — the v1 metric on an **oracle** embedding follows 1−4(1−ρ)³, crossing at 0.377 ≈ 1−4^(−1/3) | ~2 min |
| `python -m analysis.random_frame_null` | §7 — the closed-form random-frame law (E=0.0997, SD=0.0880) and the KS tests that no arm rejects | ~5 s |

If you read only one output, read the first: the complementary blindness in that
table is the paper's thesis, and it is pure metric geometry — no DGP, no model.

---

## Claim → script

### §3 Taxonomy of invariance classes
- `analysis/taxonomy.py` — six estimands × corruptions isolating one failure each.
  Includes the counterexamples (common-mode noise, collapse of a *rotated* axis)
  that correct "blind to information loss" → **blind to axis-factorised maps**.

### §4 Demonstration 1 — the manufactured threshold
- `analysis/geometry_constant.py` — the receipt: the artifact on an oracle embedding.
- `analysis/closed_form.py` (~10 min) — the recipe. V1 the closed form; V2 width
  invariance; V3 per-axis **scale**, not dimension count, moves ρ\* (0.377→0.808);
  V4 finite-k correction; V5 compact support required (Gaussian has no stable
  threshold); V6 rotation invariance is **exact** (|Δ|=0).

### §5 Demonstration 2 — the gap identity
- `analysis/taxonomy.py::verify_gap_identity` — gap = CCA−MCC to 1e-4 on every row.
- `metrics2.entanglement_gap` — the estimand as reported (oracle demoted).

### §6 Basis-dependence
- `core/dgp_rotated.py` — shifted direction placed at an angle to the ground-truth basis.
- `experiments/run_basis_rotation.py` (~40 min, 128 runs) → `analysis/basis_rotation.py`
  — the attack is closed (alignment excess ≤0), and the larger limitation:
  on a perfectly recovering model, single-run gaps span [0.009, 0.290].

### §7 The floor as an instrument
- `analysis/random_frame_null.py` — the closed-form null and all KS tests.
- `analysis/identifiability_floor.py` — per-arm floor, implied frame angle,
  the structural check that the environment varies only the unshifted axis,
  and the SESOI restated in gap units.
- `experiments/run_variability_condition.py` (~40 min, 48 runs) — n_env swept across nk+1.
- `experiments/run_sigma_min.py` (~50 min, 75 runs) → `analysis/sigma_min.py`
  — σ_min(L) at fixed n_env=9; monotone floor, still no rejection.

### §8 Application
- `gates/gate1_dgp.py` (~20 min) — **the Stage-1 gate**: ρ and δ decoupled. The
  decisive check is removability (affine map removes 100% of δ, ~0% of ρ).
- `gates/gate2_metrics.py` (~15 min) — metric ceiling flat in ρ.
- `gates/gate3_models.py` (~20 min) — models train as VAEs; conditioning is live.
- `experiments/run_main_sweep.py` (~2.5 h, 900 runs) → `analysis/main_sweep.py`, `analysis/power.py`,
  `analysis/equivalence.py results/main_sweep.csv 0.8` — the pre-registered,
  equivalence-tested null.
- `experiments/run_alignment.py` (~2 h, 600 runs) → `analysis/alignment.py`,
  `analysis/alignment_shape.py` — the alignment collapse and its **shape**
  (monotonicity, segmented-vs-linear BIC; no threshold claimed).

### Figures
`python ../docs/tmlr/make_figs_v3.py` regenerates all four from the committed CSVs.

---

## The gates

The pipeline is gated: nothing downstream was run until the stage below passed.
Each `verify_*.py` exits non-zero on failure and prints per-check PASS/FAIL.

```bash
python -m gates.gate1_dgp && python -m gates.gate2_metrics && python -m gates.gate3_models
```

Passing all three is the precondition for §8 being interpretable at all.

---

## Sharding the long sweeps

The three campaigns are resumable and shard by seed via environment variables:

```bash
STAGE4_SEEDS="0,1,2,3,4" STAGE4_OUT="results_stage4_s1.csv" python -m experiments.run_main_sweep
BR_SEEDS="0,1"           BR_OUT="results_basis_rotation_s1.csv" python -m experiments.run_basis_rotation
SM_SEEDS="0,1,2,3,4"     SM_OUT="results_sigma_min_s1.csv" python -m experiments.run_sigma_min
```

Each writes a checkpoint after every run and skips completed cells on restart.

---

## A note on `archive_v1/`

The retired first version of this probe is kept, with
[`archive_v1/RETRACTED.md`](archive_v1/RETRACTED.md) documenting why every number
in it is void. It is retained because §4 and §5 of the paper are about how those
numbers were produced, and because the retraction is checkable: the oracle receipt
reproduces the retracted threshold from the retired code.
