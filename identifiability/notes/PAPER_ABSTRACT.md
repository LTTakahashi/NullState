# TMLR draft — title + abstract

## Title

**Every Recovery Metric Has an Invariance Class: Manufactured Thresholds and
Hidden Collapses in Latent-Variable Identifiability**

*(ML/identifiability vocabulary throughout — "recovery metric", "invariance
class", "entanglement", "latent recovery". The single-cell integration setting is
a worked example in the experiments, not the frame. The colon clause names the
two demonstrations; no marginal-specific constant appears in the title, since the
constant is not universal.)*

## Abstract

A recovery metric is defined by what it cannot see. We argue — and demonstrate
twice — that **every estimand of latent recovery has an invariance class, and a
reported result is an artifact whenever the failure mode under test lies inside
that class.**

**A manufactured threshold.** k-nearest-neighbour recovery scores depend on the
embedding only through its k-nearest-neighbour sets, so they are *exactly*
(bit-identically) invariant to the similarity group — including the orthogonal
gauge freedom O(d) of an isotropic-Gaussian prior, enough to void any result whose
failure mode under test is a rotation. (The two classes are non-nested, not
identical, and cross-domain transfer is invariant only to a *common* gauge.) A
second, independent failure of the same metric is an extrapolation artifact: in a
controlled support-overlap study it produces a clean, published-looking "critical
overlap" ρ\* ≈ 0.37 that we derive in closed form: for a one-dimensional uniform
shift the large-sample cross-domain transfer is R² = 1 − 4(1 − ρ)³, so the
apparent threshold is ρ\* = 1 − 4^(−1/3) — the zero-crossing of the metric's own
extrapolation geometry, which we reproduce on a **perfectly identified oracle
embedding**. The constant is a recipe, not a universal: it equals (1/target
variance) × (second moment of the extrapolation-error law), so it is fixed only
once the marginal shape, embedding dimension, per-axis scale and k are fixed.
Vary any of them and it moves — holding d = 10 and changing only the added axes'
scale drags ρ\* from 0.377 to 0.808, and a non-compact (Gaussian) marginal has no
stable threshold at all, its crossing drifting toward 0 as n grows.

**A hidden collapse — in a metric we built to fix the first problem.** We
introduced a rotation-sensitive *matched-information oracle gap*: noise the true
latent until it matches the learned embedding's information (CCA), then difference
MCC, so the residual is attributable to entanglement rather than to a noisier
embedding. It passed a correctness argument and a gate — and it still concealed a real
effect. Adversarial verification explained why: the gap is not an independent
estimand at all, but equals **CCA − MCC identically** (max deviation 1×10⁻⁴),
because isotropic-noise oracles are coordinate-aligned and so satisfy MCC = CCA.
Its exact zero set is therefore the **axis-factorised** maps — each recovered
coordinate a function of one true coordinate. (We first mis-stated this as
"blind to information loss"; that is false, and we report the counterexamples:
non-axis-factorised information loss such as common-mode noise scores as high as
the 45° rotation we present as the entanglement signature.) We therefore report
CCA − MCC directly and demote the oracle construction to a validation.

**The blind spot is basis-dependent.** {MCC = CCA} is the axis-factorised set
*relative to the ground-truth basis one chose*: CCA is basis-free, Hungarian
matching is not, so the gap inherits basis-dependence entirely through MCC, and
rotating the ground-truth basis moves its blind region. On real data no privileged
basis for "the true latent" exists, so the blind region is positioned by a choice
with no observable counterpart. An invariance class is thus a property of the
metric **plus a coordinate choice** — and one of those is often arbitrary. Testing
this (128 runs) closes the obvious attack on our own demonstration — placing the
shifted direction at 45° to the ground-truth basis does *not* make the gap fire for
the alignment arm (excess over control ≤ 0 at every overlap) — but exposes a larger
limitation: on a *perfectly recovering* model at full overlap, single-run gaps span
[0.009, 0.290] across seeds, reaching the value a genuine 45° rotation produces.
That floor is not estimator noise: with the subspace already recovered, CCA − MCC
reads out the angle between the frame a run converged to and the chosen basis
(gap = 1 − cos φ), so it **measures how much axis-level identifiability a model
class attains**. Used as an instrument it answers a question the field states but rarely measures.
If the optimiser has no rotational preference the frame angle is uniform on
[0°, 45°] (Hungarian matching folds it there), giving a closed-form null:
E[gap] = 1 − 2√2/π = 0.0997, SD = 0.0880, range [0, 0.2929]. **Every floor we
measure sits on that law**, and one-sample KS rejects it nowhere — not for an
isotropic prior, and not for conditional priors satisfying the iVAE variability
condition of Khemakhem et al. (2020), whether the environment count merely meets
nk+1 or exceeds it with a well-conditioned natural-parameter difference matrix
(σ_min(L) = 1.65, κ = 9.3). The conclusion is stronger than "the condition does
not bite as a threshold": **satisfying it leaves the recovered frame statistically
indistinguishable from a uniformly random one.** Because the theorem concerns the
population limit and does not promise that a finite-sample optimiser finds the
identified solution, we report this as a measurement rather than a refutation —
but it is the kind of direct evidence this widely invoked condition mostly lacks,
Sweeping σ_min(L) directly at fixed environment count — constructing the spectrum
of L with total prior variability held constant — makes the floor decline
monotonically (0.115 → 0.076, with MCC genuinely improving) where the environment
count did not, identifying σ_min as the right axis and giving the theorem's binary
condition a continuous practical analogue. But the effect is bounded at ≈21% of the
full rotation signature, and no conditioning level rejects the random-frame law. Crucially the floor is **not subtractable**: its height
depends on the angle between the data's shift direction and the latent basis
(0.087 → 0.151 when we rotate it), which on real data is precisely unobservable.
Our own equivalence result is therefore best stated in gap units: no ρ-dependent
effect larger than 0.070, about a quarter of the full rotation signature.

**What the estimands, once matched to the failure modes, actually show.** In a
data-generating process where biological support overlap (ρ) and removable batch
magnitude (δ) are provably decoupled (an affine map removes 100% of δ and 0% of
ρ): for a **faithful per-cell encoder**, latent recovery is *overlap-invariant* —
a pre-registered null, equivalence-tested (TOST) against the pre-registered effect
size across three model families at 25 seeds per cell, after a power analysis
showed the 5-seed pilot could only detect d ≈ 2. For the **alignment objectives
practitioners actually run**, recovery collapses once overlap is reduced: CCA
0.99 → 0.70, Cohen's d up to 2.4 — and the failure is approximately
axis-factorised, so it is exactly what the oracle gap cannot see. We report the shape honestly — only one of four arms
is strictly monotone in ρ and only one licenses a segmented-over-linear fit, so
we claim a *large but saturating* degradation, not a threshold; the clean
dose-response is in alignment strength, not in overlap. An adversarial (DANN)
objective reproduces the collapse in direction and significance but at roughly a
third the magnitude, so the effect is a property of aligning rather than of one
penalty, with strongly mechanism-dependent size. The analysis plan — CCA primary,
TOST for the control arm, slope tests for the alignment arms, collapse-probability
versus ρ — was fixed after a four-seed pilot and before the confirmatory run.

**The practical message** is a check with a receipt: before reading a recovery
result as a phenomenon, state your estimand's invariance class and verify the
failure mode you care about is outside it — for a critical-threshold claim,
compare it against the metric's own geometry at your marginal, dimension, scale
and k.

## Scope and non-claims

- The recovery-invariance null is for a within-domain-decodable shift; the paired
  alignment result supplies the other side, so together they bound the phenomenon
  rather than leaving it open.
- The corruption taxonomy is **illustrative, not exhaustive**: three failure modes
  and six estimands, each row an existence proof rather than a coverage map.
- The pre-integration **certificate is synthetic-only** and is reported as a
  section, not a headline claim; real-data validation on controlled cell-line
  mixtures (with orthogonal, non-circular anchors and a pseudobulk-level
  permutation null) is specified as future work, not asserted.

## Why TMLR

TMLR judges technical correctness and evidence over perceived significance, with
rolling double-blind review. A metric-invariance thesis with two demonstrations —
one of which is a self-correction of our own estimand — plus an equivalence-tested
null and a powered positive result, is exactly the correctness-first contribution
its bar accepts.

## Positioning (one paragraph for the intro)

Over-correction in single-cell integration is a recognised failure mode with
methods (STACAS) and metrics (RBET, scIB) built for it, and the disentanglement
literature has long formalised recovery up to a group of transformations (Higgins
et al. 2018; Eastwood & Williams 2018; Locatello et al. 2019). Our contribution is
neither "non-shared populations break integration" nor a new metric: it is that
the *invariance class of the estimand* determines which failures are observable at
all, demonstrated by a critical threshold that is a metric-geometry constant, and
by an estimand of our own construction that reduces to CCA − MCC and therefore
conceals an entire real effect. The iVAE variability condition (Khemakhem et al.
2020) defines our satisfied/violated arms; we report frankly that it does not bite
in this regime.
