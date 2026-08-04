# Every recovery metric has an invariance class

Code and results for a study of **what estimands of latent recovery cannot see**.

> A recovery metric is defined by what it cannot see. Every estimand has an
> invariance class, and a reported result is an artifact whenever the failure
> mode under test lies inside that class.

Everything here is synthetic, CPU-only, and needs no data download.

## Where to start

| you want to… | read / run |
|---|---|
| check the central claim in ~5 minutes | [`REPRODUCE.md`](REPRODUCE.md) → "Start here" |
| see all results and how they were reached | [`notes/FINDINGS.md`](notes/FINDINGS.md) |
| see the paper framing | [`notes/PAPER_ABSTRACT.md`](notes/PAPER_ABSTRACT.md) |
| see a design that was tried and abandoned | [`notes/DESIGN_NOTES.md`](notes/DESIGN_NOTES.md) |
| see what was retracted, and why | [`archive_v1/RETRACTED.md`](archive_v1/RETRACTED.md) |

```bash
pip install -r requirements.txt
python -m analysis.taxonomy          # the thesis, as a table (~3 min)
python -m analysis.geometry_constant   # a "critical threshold" that is metric geometry (~2 min)
python -m analysis.random_frame_null  # the closed-form null nothing rejects (~5 s)
```

## The two demonstrations

**1. A manufactured threshold.** kNN recovery scores depend on the embedding only
through its neighbour sets, so they are bit-identically invariant to rotation. A
separate failure of the same metric produces a clean "critical support overlap"
ρ\* ≈ 0.37, which is exactly **1 − 4^(−1/3)**, the zero-crossing of the metric's
own extrapolation geometry — reproduced on a *perfectly identified oracle
embedding*. `analysis/geometry_constant.py` runs the **retracted** v1 metric from
`archive_v1/` to show this, so the retraction is itself checkable.

**2. A hidden collapse, in an estimand we built.** The rotation-sensitive
matched-information oracle gap we introduced to fix (1) equals **CCA − MCC
identically**, so its zero set is the axis-factorised maps — and it reported no
effect on a real alignment-induced collapse (CCA 0.99 → 0.70). Its blind region is
moreover defined relative to a chosen ground-truth basis, so an invariance class
is a property of the metric **plus a coordinate choice**.

That same quantity then becomes an instrument: with the subspace recovered,
CCA − MCC = 1 − cos φ reads out a frame angle, and "no rotational preference" has
the closed-form null φ ~ U[0°, 45°] (E = 0.0997, SD = 0.0880). **No arm we test
rejects that law**, including conditional priors satisfying the iVAE variability
condition with a well-conditioned natural-parameter matrix.

## Layout

```
analysis/taxonomy.py       the thesis: 6 estimands x corruptions, + the gap identity
analysis/geometry_constant.py  demo 1 receipt (imports the retracted v1 metric)
analysis/closed_form.py      the recipe behind the constant (marginal / scale / k)
analysis/random_frame_null.py  the closed-form null and its KS tests

core/dgp.py  core/metrics.py  core/models.py        the generative process, estimands, models
gates/gate1_dgp.py  gates/gate2_metrics.py  gates/gate3_models.py     the gates (PASS/FAIL, exit non-zero)

experiments/run_main_sweep.py / analysis/main_sweep.py       pre-registered null (+ power_analysis, equivalence_test)
experiments/run_alignment.py / analyze_alignment*.py      the alignment collapse and its shape
experiments/run_basis_rotation.py / analysis/basis_rotation.py      basis-dependence
experiments/run_variability_condition.py, experiments/run_sigma_min.py   the iVAE condition, measured
core/certificate.py                          a pre-integration diagnostic (synthetic-only)

results_*.csv                           every headline result, committed
archive_v1/                             the retired first version + why it is void
```

## How this repository is meant to be read

The pipeline is **gated**: `gates/gate1_dgp.py`, `gates/gate2_metrics.py` and
`gates/gate3_models.py` each print per-check PASS/FAIL and exit non-zero on failure,
and no downstream stage was run until the stage below it passed.

Two things are kept that a tidier repository would delete, deliberately. The
retired v1 code remains in `archive_v1/` with a document recording why each of its
numbers is void, because the paper's first demonstration is about how those
numbers were produced. And `notes/DESIGN_NOTES.md` records a mechanism that was designed,
built, de-risked and then **abandoned** when the evidence did not support it.
Both are part of the result.

## Limitations, stated up front

- The corruption taxonomy is **illustrative, not exhaustive**.
- The geometry constant is **not universal**: it is fixed only at a given marginal
  shape, embedding dimension, per-axis scale and k, and is stable only for
  compact-support marginals.
- CCA − MCC sits on an identifiability floor (~0.1), is **not interpretable
  per-run**, and that floor is **not subtractable**.
- The certificate in `core/certificate.py` is **synthetic-only**; no transfer to real
  data is claimed.
- All experiments are synthetic by design.
