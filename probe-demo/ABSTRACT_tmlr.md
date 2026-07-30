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
embedding only through its neighbour sets, so they are *exactly* invariant to the
similarity group — including the rotations that leave an isotropic-Gaussian prior
invariant while scrambling its axes, i.e. precisely the non-identifiability they
are used to test for. The consequence is not mere insensitivity. In a controlled
support-overlap study, such a score produces a clean, published-looking "critical
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
embedding. It passed a correctness argument and a gate — and it is blind to
**information loss**. Adversarial verification sharpened this: the gap is not an
independent estimand at all, but equals **CCA − MCC identically** (max deviation
1×10⁻⁴), because isotropic-noise oracles are coordinate-aligned and so satisfy
MCC = CCA. That explains the blindness — information loss lowers both terms and
cancels — and it means an entire real effect can be invisible to it.

**What the estimands, once matched to the failure modes, actually show.** In a
data-generating process where biological support overlap (ρ) and removable batch
magnitude (δ) are provably decoupled (an affine map removes 100% of δ and 0% of
ρ): for a **faithful per-cell encoder**, latent recovery is *overlap-invariant* —
a pre-registered null, equivalence-tested (TOST) against the pre-registered effect
size across three model families at 25 seeds per cell, after a power analysis
showed the 5-seed pilot could only detect d ≈ 2. For the **alignment objectives
practitioners actually run**, recovery collapses once overlap is reduced: CCA
0.99 → 0.70, Cohen's d up to 2.4, and the failure is exactly the information loss
the oracle gap cannot see. We report the shape honestly — only one of four arms
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
