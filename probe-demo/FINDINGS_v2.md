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
latent and it follows 1 − 4(1 − ρ)³ to within mean error **0.023** across the ρ
range and crosses zero at **0.378**, matching 1 − 4^(−1/3) = 0.370
([`verify_geometry_constant.py`](verify_geometry_constant.py)). An oracle
embedding is perfectly identified by construction, so a "critical threshold" that
appears for it cannot be about (non-)identifiability. An audit confirmed every v1
headline metric was likewise a closed-form function of ρ, and that the learned
model sat *on* that analytic ceiling everywhere. The threshold was the metric,
wearing a phenomenon's clothes.

## Why the metric could not have caught it, and the fix

kNN-based recovery is **near rotation-invariant**: it uses only the neighbourhood
graph, so a 45° rotated (entangled) latent scores the same as the identity. A
metric blind to rotation is blind to entanglement, which is the thing
identifiability is *about*. Replace it with:

- **MCC** (Hungarian matching, train/test split) — rotation- and
  permutation-sensitive; scores a 45° rotation **0.707** where kNN-R² scores
  ~1.0.
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
[`analyze_stage4.py`](analyze_stage4.py); 180 runs). The matched-information MCC
gap at δ=0:

| ρ | conditional | ivae_5env | ivae_2env |
|---|---|---|---|
| 1.00 | 0.096 | 0.083 | 0.114 |
| 0.60 | 0.188 | 0.047 | 0.094 |
| 0.20 | 0.141 | 0.109 | 0.135 |
| 0.05 | 0.118 | 0.127 | 0.101 |

Small everywhere, non-monotone, Cohen's d (ρ=1 vs 0.05) = 0.24 / 0.58 / −0.11 —
none reaching the pre-registered 0.8, BIC-linear in every arm. **Learned MCC
itself stays 0.80–0.94 at every ρ.** Satisfying the iVAE variability condition
(ivae_5env, 5 = 2n+1) vs violating it (ivae_2env) made no clean difference,
because identifiability of a within-domain-decodable 1-D shift was never at
stake.

**The only thing overlap bounds is removal, and that bound is analytic**
([`run_frontier.py`](run_frontier.py)). Recording both axes for the faithful
encoder:

| ρ | recovery (MCC) | removal (batch_removed) | joint |
|---|---|---|---|
| 1.00 | 0.897 | 0.996 | 0.897 |
| 0.80 | 0.883 | 0.806 | 0.806 |
| 0.60 | 0.805 | 0.612 | 0.612 |
| 0.40 | 0.854 | 0.407 | 0.407 |
| 0.20 | 0.851 | 0.205 | 0.205 |
| 0.05 | 0.873 | 0.065 | 0.065 |

Recovery flat; **removal ≈ ρ** to two decimals; the joint min tracks ρ *entirely
through removal*, whose bound (removal ≤ overlap) is a geometric identity — you
cannot remove a domain distinction where the biology does not overlap, because
the shared latent's domain-informative direction *is* the non-overlapping
biological axis.

## The removal bound, confirmed from the mixing direction

The obvious objection — "a faithful encoder is not *trying* to remove the
domain" — is answered by running an objective that is: a moment-matching
integration penalty (mean + covariance of the shared latent matched across
domains, MNN/Harmony-style) at increasing strength λ
([`run_mixing_bound.py`](run_mixing_bound.py), 54 runs). batch_removed:

| ρ | λ=0 (faithful) | λ=200 | λ=1000 | MCC λ=0 → λ=1000 |
|---|---|---|---|---|
| 1.00 | 0.998 | 0.997 | 0.996 | 0.90 → 0.89 |
| 0.60 | 0.610 | 0.592 | 0.617 | 0.81 → 0.61 |
| 0.20 | 0.208 | 0.210 | 0.302 | 0.82 → 0.64 |
| 0.05 | 0.062 | 0.059 | 0.180 | 0.86 → 0.67 |

Up to λ=200 the objective sits at **removed ≈ ρ** at every overlap while recovery
stays intact — an objective explicitly maximising cross-domain mixing cannot beat
the overlap bound, because at low overlap there is not enough shared support to
mix *through*. Pushed to the extreme (λ=1000), it *can* force removal a little
above ρ at low overlap (ρ=0.05: 0.06 → 0.18) — but **only by collapsing
recovery** (MCC 0.86 → 0.67). At ρ=0.05 no setting achieves both MCC > 0.8 and
removed > 0.1. That is the feasibility frontier stated exactly: removal ≤ overlap
bounds the *recovery-preserving* region, and exceeding it costs recovery point
for point. The analytic bound, demonstrated from the mixing side rather than
asserted — and, honestly, it is a frontier tradeoff, not a hard ceiling.

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

> A "critical overlap threshold" for latent recovery, obtained from a rotation-
> invariant recovery metric, can be an artifact of the metric's geometry rather
> than a property of the representation. Under a rotation-sensitive matched-
> oracle estimand and a DGP where support overlap and removable batch magnitude
> are decoupled, the shared biological latent is recovered up to the metric
> ceiling at all overlap levels; what overlap bounds is domain *removal*.
> removal ≤ overlap bounds the recovery-preserving region — a faithful encoder
> and moderate mixing both sit at removal ≈ ρ with recovery intact, and only
> extreme mixing exceeds it, point-for-point at the cost of recovery. Support-
> mismatch non-identifiability, in this regime, is a property of the integration
> objective, not of the representation.

## Honest limitations

- **removal ≈ ρ is partly algebraic.** It is reported as such; the non-trivial
  content is that recovery does not *additionally* degrade, so the joint
  shortfall is the analytic removal bound and nothing more.
- **The regime is deliberately clean** (2-D latent, 1-D shift, NB counts). The
  null says the phenomenon does not arise *here*; a higher-dimensional latent
  whose shifted axis is not locally decodable is the natural next probe.
- **A ρ×δ interaction is intrinsic to compositional data** (≤7% variance at
  δ=1), so the primary estimand is run at δ=0, and the decoupling is certified
  only for δ ≤ 2.0 — both published rather than hidden.

## Reproduce

```bash
python verify_geometry_constant.py                     # the headline receipt
python verify_dgp2.py && python verify_stage2.py && python verify_stage3.py
python run_stage4.py   && python analyze_stage4.py     # main result
python run_frontier.py && python run_mixing_bound.py   # removal bound, both directions
python certificate.py                                  # certificate validation
```
