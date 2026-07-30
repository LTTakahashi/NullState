# Every recovery estimand has an invariance class

## The thesis

**A recovery metric is defined by what it cannot see. Every estimand has an
invariance class, and a reported result is an artifact whenever the failure mode
under test lives inside that class.** This is not a claim about bad metrics; it
is a claim about matching the estimand to the failure mode, and we demonstrate it
twice — once on a standard metric, and once on a metric we built ourselves,
deliberately, with a correctness argument and a passing gate.

**Demonstration 1 — a standard metric, blind to rotation.** kNN-R² and
kNN-transfer scores depend on the embedding only through its neighbour graph,
which an orthogonal map preserves exactly; their invariance class therefore
contains the rotations that leave an isotropic-Gaussian prior invariant while
scrambling its axes — i.e. precisely the non-identifiability they are being used
to test for. The consequence is not merely insensitivity: a clean,
published-looking "critical support overlap" ρ\* ≈ 0.37 turns out to be **exactly
1 − 4^(−1/3)**, the zero-crossing of the metric's own extrapolation geometry,
reproduced on a *perfectly identified oracle embedding*.

**Demonstration 2 — our own metric, blind to information loss.** To fix
Demonstration 1 we built a rotation-sensitive matched-information oracle gap:
noise the true latent until it carries the same information (CCA) as the learned
embedding, then difference MCC, so the residual is attributable to entanglement
rather than to a noisier embedding. It is correct for what it was built for — and
it is **blind to information loss**. When alignment objectives destroy recovery by
folding the shifted axis — information loss, the failure mode practitioners
actually care about — the gap reads ≈0 while raw CCA collapses 0.99 → 0.70. Had
we reported only our own estimand, we would have concluded "no effect."

Adversarial verification then sharpened this into something better than a blind
spot: **the gap is not an independent estimand at all — it is identically
CCA − MCC** (verified to max |Δ| = 0.0001 across every corruption below). The
reason is structural: the oracle family is isotropic noise on the *true* latent,
so its canonical directions are the coordinate axes and every oracle satisfies
MCC = CCA; matching the oracle's information therefore forces
MCC(oracle) = CCA(z), giving gap = CCA(z) − MCC(z). That *explains* the blindness
— information loss lowers CCA and MCC together and cancels in the difference —
and says exactly what the estimand does measure: the excess of linear-subspace
recovery over axis-wise recovery, i.e. "right subspace, wrong axes." The
bisection machinery we built was unnecessary.

The second demonstration is the more persuasive one, because the blind spot
survived a correctness argument, a gate, and our own scrutiny. The lesson is
general: **choose the estimand whose invariance class excludes the failure mode
under test, and state the class explicitly.**

## The taxonomy

Measured directly ([`estimand_taxonomy.py`](estimand_taxonomy.py)): corrupt the
*true* latent in exactly one way, score every estimand. No training and no DGP —
these are properties of the estimands themselves. A cell at its identity value
means that estimand is blind to that corruption.

| corruption | failure mode | kNN-R² | CCA | MCC-P | MCC-S | DCI-D | oracle gap |
|---|---|---|---|---|---|---|---|
| identity | — | 0.996 | 1.000 | 1.000 | 1.000 | 0.997 | −0.000 |
| permute axes | nuisance | 0.996 | 1.000 | 1.000 | 1.000 | 0.998 | −0.000 |
| per-axis rescale | nuisance | 0.999 | 1.000 | 1.000 | 1.000 | 0.997 | −0.000 |
| **rotate 45°** | **entanglement** | **0.996** | **1.000** | 0.707 | 0.685 | 0.000 | 0.293 |
| **shear** | **entanglement** | **0.996** | **1.000** | 0.887 | 0.878 | 0.508 | 0.113 |
| monotone nonlinear | reparametrisation | 0.987 | 0.885 | 0.885 | **1.000** | **0.997** | **−0.000** |
| **isotropic noise** | **information loss** | 0.725 | 0.857 | 0.857 | 0.846 | 0.838 | **0.000** |
| **fold axis** | **information loss** | −0.062 | 0.531 | 0.531 | 0.507 | 0.594 | **0.000** |
| **collapse axis** | **information loss** | −0.070 | 0.502 | 0.500 | 0.502 | 0.801 | **0.001** |

The blindness is complementary, which is the point: **kNN-R² and CCA cannot see
entanglement; the matched-oracle gap cannot see information loss; MCC-Spearman
and DCI cannot see monotone reparametrisation.** No single estimand covers the
space, so "we measured recovery" is not a well-formed claim without naming the
estimand and its invariance class.

The three corruption classes are **illustrative, not a partition** of recovery
failure. Each row is an existence proof — *for this corruption, an estimand
sitting at its identity value could not have detected it* — not a coverage map.
Two of the three are genuinely orthogonal axes of the map from true latent to
embedding: whether it is **injective** (information) and whether it lies in the
**equivalence group** modulo which recovery is claimed (alignment). The third,
monotone reparametrisation, is not a separate mode but the *boundary* of that
group.

Invariance classes, stated:

| estimand | provably cannot see |
|---|---|
| kNN-R² / kNN transfer | any map preserving the k-NN **sets**. Distance-rank preservation is *sufficient but not necessary*; the class contains the similarity group Sim(d) = translations ⋊ (O(d) × ℝ₊) |
| CCA | any affine map with **injective** linear part (⊋ GL(d): also injective lifts to higher dimension, appended noise coordinates, duplicated coordinates, translations). Blind to *linear* entanglement only — information-preserving *nonlinear* axis mixing is visible |
| MCC-Pearson | permutation + per-axis affine |
| MCC-Spearman | permutation + per-axis *monotone* reparametrisation |
| matched-oracle gap | = CCA − MCC identically; blind to information loss because it cancels in the difference |

This generalises past this probe: kNN-R² and kNN-transfer recovery scores are
common in the integration and disentanglement literature, and any
critical-threshold claim built on one is exposed to reading a geometry constant
as a phenomenon.

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

## When overlap *does* govern recovery: the objective, not the data

The null above says a faithful per-cell encoder recovers the biology at every
overlap. The complementary question — is there *any* setting where overlap
governs recovery? — has a clean, powered answer: **yes, when the objective aligns
domains.** The integration objectives practitioners actually use (MNN/Harmony/
scANVI-style) do not encode each cell independently; they force the domains'
latent distributions together. On the *same* v2 DGP, adding a moment-matching
alignment penalty makes recovery overlap-dependent
([`run_alignment_campaign.py`](run_alignment_campaign.py) →
[`analyze_alignment.py`](analyze_alignment.py); 25 seeds/cell). CCA (subspace
recovery), the primary metric:

*Pre-registration.* The analysis reported here — raw CCA as the primary metric
and MCC secondary, TOST for the faithful arm, a slope test for the alignment
arms, and collapse-probability vs ρ — was fixed in
[`DESIGN_v3.md`](DESIGN_v3.md) after a 4-seed pilot and **before** the 25-seed
confirmatory run; the pilot's role was to motivate the design, not to select the
test.

The full curve (CCA, 25 seeds/cell) — reported in full because the endpoint
contrast alone would hide the shape:

| ρ | faithful (λ=0) | mix (λ=200) | mix (λ=1000) | adversarial (DANN) |
|---|---|---|---|---|
| 1.00 | 0.993 | 0.993 | 0.993 | 0.991 |
| 0.80 | 0.993 | 0.989 | 0.868 | 0.981 |
| 0.60 | 0.993 | 0.910 | 0.739 | 0.913 |
| 0.40 | 0.992 | 0.887 | 0.728 | 0.927 |
| 0.20 | 0.992 | 0.848 | 0.754 | 0.938 |
| 0.05 | 0.992 | 0.795 | 0.702 | 0.890 |

- **Faithful:** CCA drop (ρ=1→0.05) = +0.000 [−0.000, +0.001] — **TOST-equivalent
  to flat** (SESOI 0.05): recovery is overlap-invariant, confirming the null on
  the recovery axis too. Collapse probability P(CCA<0.85) = 0 at every ρ.
- **Alignment collapses recovery once overlap is reduced**: CCA drop +0.199
  (d=1.67, slope p=3.6×10⁻¹¹) at moment-matching λ=200 and +0.291 (d=2.37,
  p=1.9×10⁻¹⁰) at λ=1000. The effect is **stochastic** — collapse probability
  rises from 0 at ρ=1 to 0.52 (λ=200) and 0.76 (λ=1000) at ρ=0.05, i.e. a rising
  *fraction* of runs fold the axis rather than a uniform degradation.

**The shape, reported honestly** ([`analyze_alignment_shape.py`](analyze_alignment_shape.py)).
The v2 rule — no threshold language unless a segmented fit beats linear on BIC —
applies to our own positive result too, and the interior is not uniformly
monotone:

| arm | Spearman(CCA,ρ) | monotone? | BIC | drop (ρ≥0.6 vs ρ≤0.4) |
|---|---|---|---|---|
| faithful | +0.23 | inversions ≤0.001 (noise) | linear | +0.001 |
| mix_200 | +0.73 | **monotone** | linear | +0.121 |
| mix_1000 | +0.57 | ρ=0.20 above ρ=0.40 by 0.026 | *segmented* (bp 0.6) | +0.138 |
| adv_100 | +0.53 | ρ=0.20 above ρ=0.60 by 0.025 | linear | +0.043 |

Only `mix_200` is strictly monotone; `mix_1000` and `adv_100` each contain one
inversion at low ρ. Only one arm of four licenses segmented-over-linear, and it
is also an arm with an inversion, so **we do not claim a threshold**. The
defensible statement is: *recovery degrades substantially once overlap is
reduced and then saturates at low overlap* — a large but saturating effect, not a
clean monotone dose-response in ρ, and not a cliff. **The dose-response that is
clean is in the alignment strength λ, not in ρ.**

**Two mechanisms, but not equally.** The adversarial (DANN) arm also degrades
(drop +0.101, d=1.08, p=2.7×10⁻⁶), which supports the effect being a property of
*aligning* rather than of one penalty — but it is roughly **a third the size** of
the moment-matching effect (+0.043 vs +0.121/+0.138 on the half-range contrast)
and carries the clearest non-monotonicity. The honest claim is that the effect
replicates in direction and significance across two alignment mechanisms, with
magnitude strongly mechanism-dependent; a single-mechanism claim would be
overreach in the other direction.

So "support overlap governs recovery" is real, but it is a property of the
**objective**, not of the data or the representation: irrelevant to a faithful
encoder, decisive for an aligner. This is the field's over-correction intuition,
placed on a controlled ρ knob. *(The domain-gauge probe that motivated this is in
[`DESIGN_v3.md`](DESIGN_v3.md); it gave only a weak effect — a faithful encoder
absorbs a domain-specific gauge — which is why the clean result is
objective-induced and needs no gauge.)*

## The certificate (synthetic-only — scope stated)

A diagnostic that separates removable δ from support mismatch ρ *before*
integration. An **unsupervised** version is **provably limited** — Ben-David &
Luu (2010) prove unlabeled data cannot in general distinguish an adaptable from a
non-adaptable covariate shift — and the whole unsupervised family is shown
failing. The **anchor-based** version (a few known-shared cells fix the removable
affine map; the *irreducible per-gene mean-gap* is tested against a
permutation-null floor) resolves it on synthetic data, correctly calling all
validation cases including the harmless ρ=1/δ=2 case v1 wrongly flagged. This is
the semi-supervised-integration setting (STACAS anchors), not the
unsupervised-batch-metric one.

> **Scope.** This result is **synthetic-only**. It is reported as a section, not
> a headline claim, and no transfer to real data is asserted. Real-data
> validation is specified but not run: controlled cell-line mixtures with
> construction-level ground truth (CellBench/sc_mixology GSE118767; Zheng
> Jurkat:293T) to instantiate the ρ and δ sweeps, with anchors taken **only** from
> channels orthogonal to the RNA being tested (genotype demultiplexing, sort
> gates, spike-ins) to avoid circularity, and a permutation null resampled at the
> **pseudobulk/replicate** level rather than the cell level (a cell-level null
> would make everything significant — cells within a sample are not exchangeable,
> Squair et al. 2021). The two gates that decide it are a negative control (two
> random splits of one library must not fire) and the harmless high-δ/high-ρ case
> (same populations across very different chemistries must not fire while an
> adversarial source-detector reaches ≈1.0). Until those run, the certificate's
> real-data operating envelope is unknown.

## The corrected scientific claim

> **The thesis:** every recovery estimand has an invariance class, and a result
> is an artifact whenever the failure mode under test lives inside it. Shown
> twice: a standard kNN recovery score is *exactly* invariant to rotation — the
> non-identifiability it is used to test for — and manufactures a critical
> overlap threshold equal to the zero-crossing of its own extrapolation geometry
> (1 − 4^(−1/3) at fixed uniform marginal, dimension, per-axis scale and k),
> reproduced on a perfectly identified oracle embedding; and our own
> matched-information oracle gap, built to fix that, is invariant to *information
> loss* by construction and would have reported "no effect" on a real
> alignment-induced collapse.
>
> **What that buys, once the estimands are matched to the failure modes:** for a
> faithful per-cell encoder, latent recovery is **overlap-invariant** — a
> pre-registered, equivalence-tested null across three model families (TOST, 25
> seeds). For the alignment objectives practitioners actually run, recovery
> **collapses once overlap is reduced** (CCA 0.99 → 0.70, d up to 2.4, replicated
> in direction across two alignment mechanisms with strongly mechanism-dependent
> magnitude, saturating rather than monotone in ρ). Whether support overlap
> matters for recovery is set by the **objective**, not by the data or the
> representation — and which of these two facts you observe is set by the
> estimand you chose.

## Honest limitations

- **The recovery-invariance null is regime-specific — by design, and now with its
  complement.** A within-domain-decodable shift is recovered by any faithful
  encoder, so the null is the honest answer to v1's question for that encoder; the
  paired alignment result shows the *other* side (overlap collapses recovery once
  the objective aligns), so together they bound the phenomenon rather than leaving
  it open. The metric result remains the domain-general contribution.
- **The alignment collapse is not monotone in ρ.** Only one of four arms is
  strictly monotone and only one licenses a segmented fit; the honest description
  is *large but saturating*, and we do not claim a threshold. The clean
  dose-response is in alignment strength λ, not in ρ.
- **The two alignment mechanisms are not equal.** DANN reproduces the direction
  and significance but at roughly a third the magnitude, and carries the clearest
  non-monotonicity.
- **removal ≈ ρ is algebra**, not a measured law (balacc = 1 − ρ/2), and removal
  is not hard-bounded by ρ; only the recovery-preserving optimum is.
- **The geometry constant is not universal**: it is a constant only at a *fixed*
  marginal shape, embedding dimension, per-axis scale and k, and only for
  compact-support marginals (a Gaussian marginal has no stable threshold at all —
  its crossing drifts toward 0 as n grows).
- **The taxonomy is illustrative, not exhaustive.** Three failure modes
  (entanglement, information loss, monotone reparametrisation) with six
  estimands; other failure modes (support mismatch itself, seed-dependent
  bimodality, axis-subset recovery) are not covered by the table.
- **The certificate is synthetic-only** (see its scope box).
- **A ρ×δ interaction is intrinsic to compositional data** (≤7% variance at
  δ=1), so the primary estimand is run at δ=0, and the decoupling is certified
  only for δ ≤ 2.0 — both published rather than hidden.

## Reproduce

```bash
python estimand_taxonomy.py                            # THE THESIS: invariance-class table
python verify_geometry_constant.py                     # demo 1 receipt (oracle -> 1-4^(-1/3))
python verify_closed_form.py                            # the recipe: marginal / scale / k / rotation
python verify_dgp2.py && python verify_stage2.py && python verify_stage3.py
python run_stage4.py   && python analyze_stage4.py     # main result (5-seed pilot)
python power_analysis.py                                # MDE: n=5 d~2.0, n=25 d~0.8
python run_frontier.py && python run_mixing_bound.py   # recovery-preserving frontier, both directions
python equivalence_test.py results_stage4_n25.csv 0.8  # TOST vs pre-registered SESOI (needs merged n=25)
python run_alignment_campaign.py && python analyze_alignment.py  # objective-induced overlap-dependence
python analyze_alignment_shape.py                      # curve shape: monotonicity + segmented-vs-linear BIC
python certificate.py                                  # certificate validation
```
