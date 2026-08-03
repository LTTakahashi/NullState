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
| `python estimand_taxonomy.py` | **Fig. 1** — the estimand × failure-mode table, *and* the identity gap ≡ CCA−MCC (max dev 1e-4) | ~3 min |
| `python verify_geometry_constant.py` | §4 — the v1 metric on an **oracle** embedding follows 1−4(1−ρ)³, crossing at 0.377 ≈ 1−4^(−1/3) | ~2 min |
| `python analyze_random_frame_null.py` | §7 — the closed-form random-frame law (E=0.0997, SD=0.0880) and the KS tests that no arm rejects | ~5 s |

If you read only one output, read the first: the complementary blindness in that
table is the paper's thesis, and it is pure metric geometry — no DGP, no model.

---

## Claim → script

### §3 Taxonomy of invariance classes
- `estimand_taxonomy.py` — six estimands × corruptions isolating one failure each.
  Includes the counterexamples (common-mode noise, collapse of a *rotated* axis)
  that correct "blind to information loss" → **blind to axis-factorised maps**.

### §4 Demonstration 1 — the manufactured threshold
- `verify_geometry_constant.py` — the receipt: the artifact on an oracle embedding.
- `verify_closed_form.py` (~10 min) — the recipe. V1 the closed form; V2 width
  invariance; V3 per-axis **scale**, not dimension count, moves ρ\* (0.377→0.808);
  V4 finite-k correction; V5 compact support required (Gaussian has no stable
  threshold); V6 rotation invariance is **exact** (|Δ|=0).

### §5 Demonstration 2 — the gap identity
- `estimand_taxonomy.py::verify_gap_identity` — gap = CCA−MCC to 1e-4 on every row.
- `metrics2.entanglement_gap` — the estimand as reported (oracle demoted).

### §6 Basis-dependence
- `dgp2_rotbasis.py` — shifted direction placed at an angle to the ground-truth basis.
- `run_basis_rotation.py` (~40 min, 128 runs) → `analyze_basis_rotation.py`
  — the attack is closed (alignment excess ≤0), and the larger limitation:
  on a perfectly recovering model, single-run gaps span [0.009, 0.290].

### §7 The floor as an instrument
- `analyze_random_frame_null.py` — the closed-form null and all KS tests.
- `analyze_identifiability_floor.py` — per-arm floor, implied frame angle,
  the structural check that the environment varies only the unshifted axis,
  and the SESOI restated in gap units.
- `run_variability_condition.py` (~40 min, 48 runs) — n_env swept across nk+1.
- `run_sigma_min_sweep.py` (~50 min, 75 runs) → `analyze_sigma_min.py`
  — σ_min(L) at fixed n_env=9; monotone floor, still no rejection.

### §8 Application
- `verify_dgp2.py` (~20 min) — **the Stage-1 gate**: ρ and δ decoupled. The
  decisive check is removability (affine map removes 100% of δ, ~0% of ρ).
- `verify_stage2.py` (~15 min) — metric ceiling flat in ρ.
- `verify_stage3.py` (~20 min) — models train as VAEs; conditioning is live.
- `run_stage4.py` (~2.5 h, 900 runs) → `analyze_stage4.py`, `power_analysis.py`,
  `equivalence_test.py results_stage4_n25.csv 0.8` — the pre-registered,
  equivalence-tested null.
- `run_alignment_campaign.py` (~2 h, 600 runs) → `analyze_alignment.py`,
  `analyze_alignment_shape.py` — the alignment collapse and its **shape**
  (monotonicity, segmented-vs-linear BIC; no threshold claimed).

### Figures
`python ../docs/tmlr/make_figs_v3.py` regenerates all four from the committed CSVs.

---

## The gates

The pipeline is gated: nothing downstream was run until the stage below passed.
Each `verify_*.py` exits non-zero on failure and prints per-check PASS/FAIL.

```bash
python verify_dgp2.py && python verify_stage2.py && python verify_stage3.py
```

Passing all three is the precondition for §8 being interpretable at all.

---

## Sharding the long sweeps

The three campaigns are resumable and shard by seed via environment variables:

```bash
STAGE4_SEEDS="0,1,2,3,4" STAGE4_OUT="results_stage4_s1.csv" python run_stage4.py
BR_SEEDS="0,1"           BR_OUT="results_basis_rotation_s1.csv" python run_basis_rotation.py
SM_SEEDS="0,1,2,3,4"     SM_OUT="results_sigma_min_s1.csv" python run_sigma_min_sweep.py
```

Each writes a checkpoint after every run and skips completed cells on restart.

---

## A note on `archive_v1/`

The retired first version of this probe is kept, with
[`archive_v1/RETRACTED.md`](archive_v1/RETRACTED.md) documenting why every number
in it is void. It is retained because §4 and §5 of the paper are about how those
numbers were produced, and because the retraction is checkable: the oracle receipt
reproduces the retracted threshold from the retired code.
