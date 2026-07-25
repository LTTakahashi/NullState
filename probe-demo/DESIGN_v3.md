# Design: the hard regime — support overlap that governs *recovery*, not just removal

> **Outcome (de-risked).** The domain-gauge mechanism (§ mechanism) gives a real
> but *weak* effect, capped by an escape hatch — a faithful encoder can absorb a
> domain-specific gauge. The clean, strong result is **objective-induced** and
> needs no gauge: on the plain v2 DGP, an **alignment objective** collapses
> recovery as overlap falls (CCA 0.99 → 0.60–0.73) while a faithful encoder stays
> flat at 0.99. See "Option B confirmed" below. The failure is information loss,
> so the correct metric is raw CCA, not the matched-oracle gap.

## Why v3 exists

The v2 result is honest but its recovery-invariance null is **trivial for the
regime it was run in**. In v2 the shifted factor is *within-domain decodable*: a
shared encoder recovers `t_shift` from each cell's own expression to R²≈0.95
regardless of overlap, because recovery never requires comparing domains. Support
overlap therefore cannot touch recovery — it bounds only *removal* (and that bound
is a metric-definition identity). The paper's general contribution is the metric
critique; the recovery null is a clean-regime footnote.

v3 removes exactly that triviality: a regime where recovering the shared factor
**requires cross-domain alignment**, so support overlap genuinely governs
recovery. If it does, the field's intuition ("you need overlapping populations to
integrate") becomes a *characterised, metric-verified* phenomenon rather than the
geometry artifact v1 mistook it for — and the honest contrast is sharp: overlap
is irrelevant to a faithful per-cell encoder (v2) but decisive once recovery must
go through alignment (v3).

## The mechanism: a domain-specific gauge only overlap can pin

The shared biology is a 2-D latent `t = (t_shift, t_free)`; `t_shift` carries the
overlap knob ρ (symmetric, pooled-invariant as in v2), `t_free` fully overlaps.
Each domain renders the biology through a **domain-specific rotation** `R(θ_s)`
that mixes `t_shift` into `t_free` *before* the shared nonlinear mixing `g`:

```
x = counts( g( R(θ_s) · t ) + δ·batch ) ,   θ_A = −θ_g/2,  θ_B = +θ_g/2.
```

`θ_g = 0` is exactly v2 (no gauge). The gauge is a faithful abstraction of two
labs measuring the same biology in instrument-specific coordinate frames, and it
is **not removable by an affine batch map** (a rotation mixes the biological axes;
undoing it requires knowing `t`), so it is a genuine identifiability obstruction,
categorically different from the removable offset δ.

**The identifiability argument (why overlap controls recovery here).** A real
integration model uses a **shared encoder** (scVI-style: only the decoder is
domain-conditional). A shared encoder cannot apply a domain-specific inverse; it
must find *one* map `E` sending both domains to a common latent. For the same true
`t`, domain A presents `g(R(θ_A)t)` and B presents `g(R(θ_B)t)` — different
points — and `E` must send both to the same `z`. That constraint exists **only
where both domains have support**: in the overlap band the shared `t_shift` values
force `E` to a consistent common frame; in each domain's exclusive region there is
no cross-domain counterpart, so the frame is un-pinned and the recovered axes
rotate freely per domain. As ρ falls, more of the latent is un-pinned →
domain-inconsistent axes → the rotation-sensitive matched-oracle MCC gap **grows**.
Equivalently: the common-frame `t_shift` is *not* within-domain decodable (within
one domain you only ever see `R(θ_s)t`, and `θ_s` relative to the other domain is
unknown) — the exact property v2 lacked.

## The escape hatch, and why it does not dominate (the thing that could sink this)

The failure mode to watch, and the one that would make this another v1: with an
isotropic prior the model can represent the **domain-gauged** latent `z = R(θ_s)t`
and *ignore* the domain covariate, reconstructing perfectly with an MCC degraded
by θ but **constant in ρ** — no phenomenon. The de-risk rules this out
empirically for the tested settings: at θ=π/4, ρ=1.0 the gap is **0.006** (the
model recovers the *common* frame, not the gauged one — a hatched solution would
be θ-degraded at every ρ), and the gap grows to **0.097** at ρ=0.1. The
domain-conditional decoder plus reconstruction over the overlap band evidently
pushes the optimum toward the common frame where overlap allows, and loses it
where it does not. **This must be reported at every run** (the ρ=1 gap is the
hatch monitor); if a configuration shows a large, ρ-flat gap, it took the hatch.

## De-risk result — a real but WEAK effect, capped by the escape hatch

Two runs, and they disagree in a way that matters. A 2-seed run
([`derisk_dgp3.py`](derisk_dgp3.py)) looked strong (θ=0 change +0.007, θ=π/4
change +0.092, ~16×). But the 4-seed **dose-response**
([`derisk_dgp3_dose.py`](derisk_dgp3_dose.py)) is the honest picture:

| θ | gap @ ρ=1 | gap @ ρ=0.1 | change |
|---|---|---|---|
| 0 | 0.010 | 0.052 | +0.041 |
| π/6 | 0.022 | 0.092 | +0.069 |
| π/4 | 0.028 | 0.100 | +0.072 |
| π/3 | 0.035 | 0.080 | +0.045 |

Three corrections to the optimistic read:
1. **The θ=0 baseline change is +0.041, not +0.007** — the 2-seed number was
   noise. The genuine gauge effect *over baseline* is only ~**+0.03**, comparable
   to the per-cell noise at 4 seeds.
2. **Not monotone in θ** — the effect peaks at π/4 and *falls* at π/3.
3. **The escape hatch is visibly active**: `gap @ ρ=1` rises with θ
   (0.010 → 0.035), i.e. at larger angles the model increasingly encodes the
   *gauged* latent even at full overlap. That degrades the ceiling and caps the
   overlap-dependent effect — which is why the dose-response turns over.

**Diagnosis (why it is weak).** A domain-specific gauge is, by construction,
absorbable by domain-conditional model components: the domain-conditional decoder
can represent `D(z,s)=g(R(θ_s)z)` (recovering common-frame `z=t`) *or* `D(z,s)=g(z)`
(with a gauged `z=R(θ_s)t`), and both reconstruct. The isotropic prior does not
choose between them, so overlap acts only as a **soft tie-breaker**, not a hard
identifiability constraint — hence the small, hatch-limited effect. The gauge
never *strictly* requires overlap.

**What this means for the design.** As stated (faithful encoder + iso prior), v3
gives a directionally-correct but weak and confounded effect. Getting a *clean,
strong* overlap-dependence of recovery needs one of:
- **(A) close the hatch structurally** so the common frame is the unique optimum
  — e.g. a *shared* (non-domain-conditional) decoder forces both domains through
  one map, but then the model is forced to gauged latents and recovery becomes
  θ-degraded and ρ-flat (overlap-independent — the opposite failure). An
  anisotropic iVAE prior pins the frame, but likely *per domain* (also
  overlap-independent). Neither obviously yields "recovery strictly needs
  overlap"; this is the open design question.
- **(B) an alignment objective** (adversarial domain-invariance / mixing at
  moderate strength) that actively penalises the domain-separable solution. This
  *does* make the non-overlap regions mix incorrectly and should give a strong,
  clean overlap-dependence — but it is **objective-induced**, which is precisely
  the v2 thesis, not a new "recovery is intrinsically overlap-bound" claim.

Honest current verdict: the domain-gauge is not, on this evidence, a clean
strong probe for a *faithful* encoder; the phenomenon most likely lives in the
alignment-objective regime (B), consistent with v2. The next experiment is the
alignment arm, not more gauge seeds.

## Option B confirmed (the clean, strong result) — no gauge needed

[`derisk_alignment.py`](derisk_alignment.py): standard v2 DGP (no gauge), a
moment-matching alignment penalty at strength λ, 4 seeds. **CCA (subspace
recovery):**

| ρ | λ=0 (faithful) | λ=200 | λ=1000 |
|---|---|---|---|
| 1.0 | 0.993 | 0.993 | 0.992 |
| 0.6 | 0.994 | 0.984 | 0.850 |
| 0.3 | 0.993 | 0.769 | 0.725 |
| 0.1 | 0.993 | 0.597 | 0.707 |

The faithful encoder is **flat at 0.993 across all ρ** (the v2 null); the
alignment objective is free at full overlap but **collapses recovery as ρ falls**.
The MCC drop (λ=1000 − λ=0) is overlap-dependent: −0.084 at ρ=1 (alignment even
helps slightly), +0.21/+0.27/+0.18 at ρ=0.6/0.3/0.1.

**Two methodological points, both borne out:**
- The failure is **information loss (folding the shifted axis), not entanglement**:
  when MCC collapses, CCA collapses with it (0.55/0.55), and the **matched-oracle
  gap stays ~flat and noisy** — it is designed to subtract information loss, so it
  is BLIND to this failure mode. The correct metric here is **raw CCA/MCC**, not
  the entanglement gap. (This is itself a reportable point: the v2 estimand and
  the v3 estimand must differ, because the two failure modes differ.)
- The effect is **stochastic**: at low ρ some seeds fold and some do not (ρ=0.3,
  λ=200 MCC ranged 0.55–0.98), so the mean CCA drop reflects a *rising probability
  of collapse* as ρ falls. A proper build-out needs enough seeds to characterise
  that distribution.

**Conclusion.** "Support overlap governs recovery" is real and clean — but it is
**objective-induced**: irrelevant to a faithful per-cell encoder (v2 null,
equivalence-tested), decisive for the alignment-based integration objectives
practitioners actually use (MNN/Harmony/scANVI-style). This completes the arc:
whether overlap matters for recovery is a property of the *objective*, not the
data or the representation. The domain gauge is unnecessary; the plain DGP plus
an alignment objective is the cleaner probe.

### Build-out (if pursued)
Power to ≥25 seeds; add an adversarial (DANN) alignment arm alongside
moment-matching; TOST the faithful arm as *equivalent* (flat) and test the
alignment arm's CCA-drop slope as significantly negative; report the
collapse-probability vs ρ. Metric = raw CCA (primary), MCC (secondary); the
matched-oracle gap is explicitly the WRONG tool here and that contrast is worth
stating.

## The DGP ([`dgp3.py`](dgp3.py), built and smoke-tested)

Extends `dgp2` (symmetric pooled-invariant overlap, fixed reference-grid
standardisation on the *pooled rotated* prior, non-saturating mixing, depth drawn
independently of ρ/δ/s/t, NB counts, environment structure) with the single new
`gauge_angle` knob. Returns the **common-frame** `t` as the recovery target.

## Models / arms

1. **shared-encoder conditional VAE** (scVI-style; the realistic integrator) —
   *predict:* gap grows as ρ falls.
2. **iVAE, 5-env vs 2-env** — the env-anchored prior breaks rotation symmetry.
   *Open question the probe answers:* does a within-domain env anchor pin each
   frame independently (→ overlap-independent), or is cross-domain overlap still
   required? Either answer is informative and ties directly to the iVAE
   variability condition the v2 paper already invokes.
3. **domain-conditional-ENCODER control** — give the encoder the domain too. It
   can now apply a domain-specific inverse, so recovery should become
   overlap-*independent* again. This isolates that the phenomenon is a property of
   the *shared-encoder alignment constraint*, not of the DGP.
4. **gauge-angle dose-response** (θ ∈ {0, π/6, π/4, π/3}) — the effect must scale
   with θ; θ=0 must be flat.

Metric: the v2 matched-noise-oracle MCC gap (rotation-sensitive), plus a
**cross-domain MCC** (fit the Hungarian assignment on domain A, score on B) that
directly measures frame consistency — and, unlike v1's kNN transfer, is rotation-
sensitive rather than an extrapolation-geometry artifact.

## Gates (nothing runs downstream until these pass)

- **G-decouple:** ρ and δ still decoupled *under the gauge* — the v2
  `verify_dgp2` checks (depth, per-gene stats, library size ρ-invariant;
  δ affine-removable) must still pass on `dgp3` at θ>0. The gauge must also be
  shown **not** affine-removable (distinguishing it from δ).
- **G-metric:** the matched-oracle ceiling is flat in ρ for fixed embedding
  quality (carries over from v2 Stage-2).
- **G-null recovery:** at θ=0 the gap is flat in ρ (recovers v2) — the internal
  negative control.
- **G-hatch:** the ρ=1 gap is small (not θ-degraded) at every configuration used
  to claim the phenomenon.
- **G-oracle-ceiling:** an *identifiability* ceiling — the best achievable
  cross-domain-consistent recovery, computed from the generative model
  independent of any trained network, must itself fall with ρ. This proves the
  learned failure is identifiability, not optimisation (the control v1 never had).

## Predictions, pre-registered

- **Supported:** shared-encoder gap grows with falling ρ (Cohen's d ≥ 0.8, ρ=1
  vs ρ=0.1, at ≥25 seeds), monotone in the gauge angle, while the
  domain-conditional-encoder control and θ=0 stay flat.
- **Refuted / hatch:** the gap is flat in ρ, or large-and-flat (escape hatch), or
  moves with δ rather than ρ. Report honestly per the v2 Stage-F template.
- Power: reuse [`power_analysis.py`](power_analysis.py) / [`equivalence_test.py`](equivalence_test.py);
  target ≥25 seeds so the *positive* claim clears MDE d≈0.8, and TOST the θ=0 and
  domain-conditional-encoder controls as *equivalent* (flat).

## Theory tie-in

This formalises the **alignment / anchoring** non-identifiability that makes
MNN/Harmony/STACAS need shared populations: the frame reconciliation is Ben-David's
best-joint-hypothesis term (λ), inflated by non-overlap and irreducible by any
alignment. It also makes the v2 **anchor certificate** load-bearing rather than
peripheral: anchors in the overlap band are exactly what pins the gauge, so the
certificate now predicts *recovery* failure, not just removal — closing the loop
between the metric critique, the null, and the diagnostic.

## Honest risks

- **The gauge is a specific abstraction** (a pure latent rotation). Real
  instrument effects are richer; the claim is a controlled existence proof of
  overlap-governed recovery, not that every batch effect is a rotation.
- **The escape hatch is a live threat** at every setting — hence G-hatch as a
  hard gate, not an afterthought.
- **The env-prior may pin frames within-domain**, making the iVAE arm
  overlap-independent. That is a genuine open question, not a bug; report it.
- **The effect is modest at θ=π/4** (~0.09); it must be powered (≥25 seeds) and
  may need a larger gauge angle. Do not tune θ to manufacture size — report the
  dose-response and let it speak.
