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
| see all results and how they were reached | [`FINDINGS_v2.md`](FINDINGS_v2.md) |
| see the paper framing | [`ABSTRACT_tmlr.md`](ABSTRACT_tmlr.md) |
| see a design that was tried and abandoned | [`DESIGN_v3.md`](DESIGN_v3.md) |
| see what was retracted, and why | [`archive_v1/RETRACTED.md`](archive_v1/RETRACTED.md) |

```bash
pip install -r requirements.txt
python estimand_taxonomy.py          # the thesis, as a table (~3 min)
python verify_geometry_constant.py   # a "critical threshold" that is metric geometry (~2 min)
python analyze_random_frame_null.py  # the closed-form null nothing rejects (~5 s)
```

## The two demonstrations

**1. A manufactured threshold.** kNN recovery scores depend on the embedding only
through its neighbour sets, so they are bit-identically invariant to rotation. A
separate failure of the same metric produces a clean "critical support overlap"
ρ\* ≈ 0.37, which is exactly **1 − 4^(−1/3)**, the zero-crossing of the metric's
own extrapolation geometry — reproduced on a *perfectly identified oracle
embedding*. `verify_geometry_constant.py` runs the **retracted** v1 metric from
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
estimand_taxonomy.py       the thesis: 6 estimands x corruptions, + the gap identity
verify_geometry_constant.py  demo 1 receipt (imports the retracted v1 metric)
verify_closed_form.py      the recipe behind the constant (marginal / scale / k)
analyze_random_frame_null.py  the closed-form null and its KS tests

dgp2.py  metrics2.py  models2.py        the generative process, estimands, models
verify_dgp2.py  verify_stage2.py  verify_stage3.py     the gates (PASS/FAIL, exit non-zero)

run_stage4.py / analyze_stage4.py       pre-registered null (+ power_analysis, equivalence_test)
run_alignment_campaign.py / analyze_alignment*.py      the alignment collapse and its shape
run_basis_rotation.py / analyze_basis_rotation.py      basis-dependence
run_variability_condition.py, run_sigma_min_sweep.py   the iVAE condition, measured
certificate.py                          a pre-integration diagnostic (synthetic-only)

results_*.csv                           every headline result, committed
archive_v1/                             the retired first version + why it is void
```

## How this repository is meant to be read

The pipeline is **gated**: `verify_dgp2.py`, `verify_stage2.py` and
`verify_stage3.py` each print per-check PASS/FAIL and exit non-zero on failure,
and no downstream stage was run until the stage below it passed.

Two things are kept that a tidier repository would delete, deliberately. The
retired v1 code remains in `archive_v1/` with a document recording why each of its
numbers is void, because the paper's first demonstration is about how those
numbers were produced. And `DESIGN_v3.md` records a mechanism that was designed,
built, de-risked and then **abandoned** when the evidence did not support it.
Both are part of the result.

## Limitations, stated up front

- The corruption taxonomy is **illustrative, not exhaustive**.
- The geometry constant is **not universal**: it is fixed only at a given marginal
  shape, embedding dimension, per-axis scale and k, and is stable only for
  compact-support marginals.
- CCA − MCC sits on an identifiability floor (~0.1), is **not interpretable
  per-run**, and that floor is **not subtractable**.
- The certificate in `certificate.py` is **synthetic-only**; no transfer to real
  data is claimed.
- All experiments are synthetic by design.
