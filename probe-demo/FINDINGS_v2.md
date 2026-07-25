# When a critical threshold is a geometry constant

## Headline

**Rotation-invariant recovery metrics cannot detect entanglement, and a
"critical threshold" derived from one can be a constant of the metric's own
geometry rather than a property of the data or the model.** We demonstrate this
with a case where a clean, published-looking critical overlap ρ\* ≈ 0.37 turned
out to be **exactly 1 − 4^(−1/3)**, the zero-crossing of a k-nearest-neighbour
transfer score's extrapolation geometry — a number no model, and no amount of
data, can move. When the metric is replaced with a rotation-sensitive
matched-oracle estimand and the data-generating process is rebuilt so that
support overlap (ρ) and removable batch magnitude (δ) are provably decoupled,
the "cliff" disappears entirely, and the actual structure is an analytic one.

This generalises past the probe. kNN-R² and kNN-transfer recovery scores are
common in the integration and disentanglement literature; any critical-threshold
claim built on one, in fixed embedding dimension, is exposed to reading a
geometry constant as a phenomenon, and almost nobody checks.

## The geometry constant, in detail

The retired v1 probe (see [`archive_v1/RETRACTED.md`](archive_v1/RETRACTED.md))
scored cross-domain recovery with a kNN map fit on one domain and evaluated on
the other. Its "critical overlap" fell out as a clean threshold near ρ ≈ 0.37,
below which recovery went negative. But the whole curve is closed form:

> cross-domain transfer = 1 − 4(1 − ρ)³,  zero at (1 − ρ)³ = ¼,  i.e.
> **ρ\* = 1 − 4^(−1/3) ≈ 0.370.**

That is the point at which a kNN regressor asked to extrapolate ρ-controlled
distance beyond its training support crosses from "better than the mean" to
"worse than the mean". It is a fact about nearest-neighbour extrapolation in the
embedding, present for an **oracle embedding** (z = the true latent) just as much
as for any learned one — which is the receipt: run the v1 metric on the true
latent and it follows 1 − 4(1 − ρ)³ closely (mean abs error 0.016 at n=8000,
0.023 at n=4000) and crosses zero at ≈0.377
([`verify_geometry_constant.py`](verify_geometry_constant.py),
[`verify_closed_form.py`](verify_closed_form.py)). The small offset between the
measured 0.377 and the analytic 0.370 is a **finite-k smoothing bias** — the
saturated boundary prediction is w − O(k/n), not exactly w — and it shrinks to
≈0.372 as k/n → 0 (V4). An oracle embedding is perfectly identified by
construction, so a "critical threshold" that appears for it cannot be about
(non-)identifiability. The threshold was the metric, wearing a phenomenon's
clothes.

### The constant is a recipe, not a universal

Three independent derivations (an adversarial workflow) and a direct simulation
agree on where every factor comes from, and on where the constant moves:

- **1 − 4(1 − ρ)³** decomposes as: exponent 3 = 1 (fraction of the target domain
  that must extrapolate, linear in 1−ρ) + 2 (squared extrapolation error, whose
  scale is linear in 1−ρ); constant 4 = (1/Var of the target) × (second-moment
  factor of the error law) = 12 × ⅓ for a unit uniform. The target width
  **cancels** (both error and variance scale as w²), so it is scale-free in w.
- **Marginal shape sets the constant** — but only for **compact-support**
  marginals. Uniform pins ρ\*≈0.377 across a 200× range of k/n; a triangular
  marginal gives a *different but stable* constant (≈0.44). A boundary-free
  Gaussian has **no stable threshold at all**: with no support edge, R²→1 for
  every fixed ρ>0 and the crossing drifts toward 0 as n grows (0.14→0.09→<0.05
  at n=2k→8k→32k). Report a Gaussian crossing as a finite-sample artifact, not a
  marginal analogue of 4.
- **Embedding geometry moves it**, and the driver is **per-axis scale, not
  dimension count**: at fixed d=10, shrinking the shared axes to std 0.001
  returns ρ\* to the 1-D 0.377, and enlarging them to std 1.0 drives it to 0.81.
  So "no model or data can move ρ\*" holds only at *fixed marginal, embedding
  dimension, and per-axis scale*.

**The transferable check:** compute (1/Var_shape)×(second-moment factor) for
your marginal, dimension, and k, and compare it to your reported "critical
overlap" before reading the threshold as a phenomenon. Every v1 headline metric
was, likewise, a closed-form function of ρ that an oracle embedding also
followed.

## Why the metric could not have caught it, and the fix

kNN-based recovery is **exactly rotation-invariant** (not merely approximately):
it depends on the embedding only through its neighbour graph, and an orthogonal
map preserves every pairwise distance, hence the entire graph, hence every
prediction and R² — bit-identical, up to the measure-zero set of exact
k-th-neighbour ties ([`verify_closed_form.py`](verify_closed_form.py) V6:
|Δ|=0). Its invariance class (all isometries and global rescalings) *contains*
the orthogonal group O(d), which is exactly the family that leaves an
isotropic-Gaussian prior invariant while scrambling its axes — the rotational
non-identifiability of Locatello et al. (2019). So the metric provably cannot
distinguish an axis-aligned latent from a rotated one, and cannot certify the
axis-level identifiability disentanglement targets. (Its invariance class is
neither a subset nor a superset of the full non-identifiability family, which
also includes nonlinear reparametrisations kNN-R² *is* sensitive to; the
relevant overlap is the rotational part.) Replace it with:

- **MCC** (Hungarian matching, train/test split) — rotation- and
  permutation-sensitive; scores a 45° rotation **0.707** (=cos 45°) where
  kNN-R² scores ~1.0.
- **CCA** — the weak (∼_A) notion; scores that same rotation **1.000**, so
  MCC-vs-CCA reads out "right subspace, wrong axes".
- a **matched-noise oracle**: isotropic noise added to the true latent until it
  carries the same information (CCA) as the learned embedding. Isotropic noise
  cannot mix axes, so the **learned − oracle MCC gap** is attributable to
  entanglement, never to the learned embedding simply being noisier.

The gate ([`verify_stage2.py`](verify_stage2.py), PASS) proves this metric does
**not** bend with ρ on its own: for a fixed-quality embedding the spread across
the ρ sweep is **0.003** (v1's metric varied by ~3.5 — three orders of magnitude
worse).

## The worked example: the "cliff" was never there

With a rotation-sensitive metric and a DGP where ρ and δ are genuinely
separable, the support-overlap experiment gives a clean, boring, correct answer.
We flag its scope up front: this is a **deliberately clean regime** — a
within-domain-decodable 1-D shift, where a faithful per-cell encoder recovers the
latent whatever the overlap — so the recovery-invariance null below may be
*trivial for this regime*, and is not the paper's general contribution (the
metric result above is). We report it because it is the honest answer to the
question v1 got wrong, and because pre-registering and equivalence-testing a null
is worth more than another over-claimed positive.

**The DGP actually decouples the knobs** ([`verify_dgp2.py`](verify_dgp2.py),
PASS). A symmetric, pooled-invariant overlap construction (`w=(1−ρ)/2`; domain A
mass `2w` on `[0,w)`, B mass `2w` on `(1−w,1]`; `p_A+p_B≡2` so the pooled
marginal is *exactly* Uniform at every ρ while A has strictly zero density above
`1−w`), fixed reference-grid standardisation, non-saturating mixing, and depth
drawn independently of ρ/δ/s/t. The decisive check is **removability**
(Ben-David's divergence-vs-λ split): an affine batch map removes **100.0%** of
δ's expression gap and **~0%** of ρ's. (*The design brief asked to pin each
domain's marginal AND set ρ=OVL; that is unsatisfiable — OVL=1 iff the marginals
coincide. Pooled invariance is the achievable form and is what makes the
observable statistics ρ-invariant.*)

**Recovery does not degrade with overlap** ([`run_stage4.py`](run_stage4.py) →
[`analyze_stage4.py`](analyze_stage4.py); 900 runs at 25 seeds/cell). The
matched-information MCC gap at δ=0:

| ρ | conditional | ivae_5env | ivae_2env |
|---|---|---|---|
| 1.00 | 0.097 | 0.079 | 0.093 |
| 0.60 | 0.137 | 0.067 | 0.079 |
| 0.20 | 0.084 | 0.112 | 0.093 |
| 0.05 | 0.080 | 0.093 | 0.074 |

The gap is small everywhere (~0.05–0.14), non-monotone, and BIC-linear in every
arm; the
learned latent stays **within a ρ-independent gap (~0.10) of the matched-oracle
ceiling at every ρ** (learned MCC 0.80–0.94). Satisfying the iVAE variability
condition (ivae_5env, 5 = 2n+1) vs violating it (ivae_2env) made no clean
difference, because identifiability of a within-domain-decodable 1-D shift was
never at stake.

*Power and equivalence.* At the pre-registered 5 seeds/cell the pairwise test
had minimum detectable effect **d≈2.0** (power 0.20 for d=0.8) — underpowered, so
"d<0.8" was weak evidence ([`power_analysis.py`](power_analysis.py)). We treat
the 5-seed run as a pilot whose *a-priori* power analysis (not its p-value)
motivated expanding to **25 seeds/cell** (MDE d≈0.8), and report the confirmatory
analysis as an **equivalence test** (TOST) against the pre-registered SESOI of
d=0.8, not a non-significant p ([`equivalence_test.py`](equivalence_test.py)).
Adding seeds to a null reduces Type II error and cannot manufacture a false
positive. **At n=25 all three arms are statistically equivalent** — the gap change
from ρ=1 to ρ=0.05 is significantly smaller than the pre-registered d=0.8 SESOI in
every arm:

| arm | n | D (gap) | d | 90% CI | verdict |
|---|---|---|---|---|---|
| conditional | 25 | −0.017 | −0.23 | [−0.053, +0.018] | EQUIVALENT |
| ivae_2env | 25 | −0.018 | −0.21 | [−0.060, +0.023] | EQUIVALENT |
| ivae_5env | 25 | +0.014 | +0.15 | [−0.029, +0.056] | EQUIVALENT |

A *positive* null: recovery is overlap-invariant within a pre-specified bound, not
merely "not significantly different". (Signs vary — the gap slightly *decreases*
with falling ρ in two arms — underscoring there is no overlap-driven recovery
loss.)

**What overlap bounds is removal — as a recovery-preserving frontier, not a hard
bound** ([`run_frontier.py`](run_frontier.py), [`run_mixing_bound.py`](run_mixing_bound.py)).
Recording both axes for the faithful encoder:

| ρ | recovery (MCC) | removal (batch_removed) | joint |
|---|---|---|---|
| 1.00 | 0.897 | 0.996 | 0.897 |
| 0.60 | 0.805 | 0.612 | 0.612 |
| 0.20 | 0.851 | 0.205 | 0.205 |
| 0.05 | 0.873 | 0.065 | 0.065 |

Two honest caveats make this a frontier rather than a law. (i) **removal ≈ ρ is
algebra, not a measured effect**: `batch_removed = clip(1 − 2(balacc − 0.5))`,
and the Bayes-optimal domain classifier on two overlap-ρ uniform marginals has
balanced accuracy exactly 1 − ρ/2, so `batch_removed ≡ ρ` for *any* latent-
preserving encoder (verified to 3 decimals). (ii) removal is **not** hard-bounded
by ρ: an objective explicitly maximising cross-domain mixing pushes removal to
0.40 at ρ=0.2 and 0.27 at ρ=0.05 — *above* overlap — but only by collapsing
recovery point-for-point (MCC 0.86 → 0.67). So the one non-trivial statement is:
the **recovery-preserving optimum** sits at removal ≈ ρ, and buying removal beyond
that costs recovery. A frontier, not a ceiling.

### The frontier, in full (the mixing sweep)

The λ sweep behind the caveats above ([`run_mixing_bound.py`](run_mixing_bound.py),
54 runs) — batch_removed and the recovery cost:

| ρ | λ=0 (faithful) | λ=200 | λ=1000 | MCC λ=0 → λ=1000 |
|---|---|---|---|---|
| 1.00 | 0.998 | 0.997 | 0.996 | 0.90 → 0.89 |
| 0.60 | 0.610 | 0.592 | 0.617 | 0.81 → 0.61 |
| 0.20 | 0.208 | 0.210 | 0.302 | 0.82 → 0.64 |
| 0.05 | 0.062 | 0.059 | 0.180 | 0.86 → 0.67 |

Up to λ=200 the objective sits at removed ≈ ρ with recovery intact; at λ=1000 it
forces removal above ρ at low overlap (ρ=0.05: 0.06 → 0.18) only by collapsing
recovery (MCC 0.86 → 0.67). At ρ=0.05 no setting achieves both MCC > 0.8 and
removed > 0.1 — the recovery-preserving frontier. Note this refutes any "removal
≤ overlap" reading: removal *can* exceed overlap; what it cannot do is exceed it
while preserving recovery.

## The certificate

The diagnostic that separates removable δ from support mismatch ρ *before*
integration. An **unsupervised** version is **provably limited** — Ben-David &
Luu (2010) prove unlabeled data cannot in general distinguish an adaptable from a
non-adaptable covariate shift — and the whole unsupervised family is shown
failing. The **anchor-based** version (a few known-shared cells fix the removable
affine map; the *irreducible per-gene mean-gap* is tested against a
permutation-null floor) resolves it, correctly calling all validation cases
including the harmless ρ=1/δ=2 case v1 wrongly flagged. This is the semi-
supervised-integration setting (STACAS anchors), not the unsupervised-batch-
metric one.

## The corrected scientific claim

> **The general, correctness-solid result:** a rotation-invariant recovery
> metric is provably blind to the rotational non-identifiability of an
> isotropic-prior latent model, and a "critical overlap threshold" derived from
> one can be the zero-crossing of the metric's own extrapolation geometry
> (1 − 4^(−1/3) for uniform marginals) — a constant computable from the marginal
> shape, embedding scale, and k, present even for a perfectly identified oracle
> embedding. **The worked example:** with the metric fixed (rotation-sensitive,
> matched-oracle) and ρ, δ decoupled, the shared latent is recovered within a
> ρ-independent gap of the ceiling at all overlaps (a pre-registered,
> equivalence-tested null in a deliberately clean regime); what overlap bounds is
> only domain *removal*, and even that is a recovery-preserving frontier (removal
> ≈ ρ is a metric-definition identity; mixing can exceed it only by destroying
> recovery), not a hard bound.

## Honest limitations

- **The recovery-invariance null may be trivial for this regime.** A within-
  domain-decodable 1-D shift is recovered by any faithful encoder; the null is
  the honest answer to v1's question but is *not* the paper's general
  contribution (the metric result is). A higher-dimensional latent whose shifted
  axis is not locally decodable is the natural next probe.
- **removal ≈ ρ is algebra**, not a measured law (balacc = 1 − ρ/2), and removal
  is not hard-bounded by ρ; only the recovery-preserving optimum is.
- **The geometry constant is not universal**: marginal-, scale-, and k-dependent,
  and it exists as a stable constant only for compact-support marginals.
- **A ρ×δ interaction is intrinsic to compositional data** (≤7% variance at
  δ=1), so the primary estimand is run at δ=0, and the decoupling is certified
  only for δ ≤ 2.0 — both published rather than hidden.

## Reproduce

```bash
python verify_geometry_constant.py                     # headline receipt (oracle -> 1-4^(-1/3))
python verify_closed_form.py                            # the recipe: marginal / scale / k / rotation
python verify_dgp2.py && python verify_stage2.py && python verify_stage3.py
python run_stage4.py   && python analyze_stage4.py     # main result (5-seed pilot)
python power_analysis.py                                # MDE: n=5 d~2.0, n=25 d~0.8
python run_frontier.py && python run_mixing_bound.py   # recovery-preserving frontier, both directions
python equivalence_test.py results_stage4_n25.csv 0.8  # TOST vs pre-registered SESOI (needs merged n=25)
python certificate.py                                  # certificate validation
```
