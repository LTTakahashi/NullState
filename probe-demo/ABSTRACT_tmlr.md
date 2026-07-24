# TMLR draft — title + abstract

## Title

**Rotation-Invariant Recovery Metrics Cannot Detect Entanglement: When a
Critical Overlap Threshold Is a Geometry Constant**

(ML/identifiability vocabulary throughout — "recovery metric", "entanglement",
"critical overlap", "geometry constant". The single-cell integration setting is a
worked example in the experiments, not the frame. "a geometry constant" is
deliberately hedged: the specific value is marginal-dependent, so the *number*
does not go in the title.)

## Abstract

Recovery metrics based on k-nearest-neighbour regression or transfer are widely
used to judge whether a latent factor has been recovered. We show they cannot
serve that purpose against the transformation identifiability is about. Because
such a metric depends on the embedding only through its neighbour graph, it is
**invariant to every orthogonal transformation of the embedding** — exactly (up
to the measure-zero set of exact k-th-neighbour ties), since an orthogonal map
preserves all pairwise distances. That invariance class contains precisely the
rotations that leave an isotropic-Gaussian prior invariant while scrambling its
axes, so the metric is provably blind to the rotational non-identifiability of an
isotropic-prior latent-variable model (in our runs a 45° rotation leaves
kNN-R² unchanged to machine precision while MCC drops from 1.00 to 0.76).

This blindness is not benign. In a controlled support-overlap probe, an
oracle embedding — z set to the *true* latent, hence perfectly identified by
construction — produces a clean, published-looking "critical overlap" ρ\* below
which a cross-domain kNN transfer score goes negative. We derive that curve in
closed form: for a 1-D uniform shift of overlap ρ, the large-sample transfer is
**R²(ρ) = 1 − 4(1 − ρ)³**, so the apparent threshold is **ρ\* = 1 − 4^(−1/3) ≈
0.370** — the zero of the metric's own extrapolation geometry, appearing for a
*perfectly identified* embedding and therefore not a fact about identifiability.
The constant is a recipe, not a universal: it equals (1/Var of the target) ×
(second moment of the extrapolation-error law), so it is set by the domain
**marginal shape** and the transfer construction (a different compact marginal
gives a different but stable constant), and it exists at all only for
**compact-support** marginals — a boundary-free Gaussian has no stable threshold,
its crossing drifting to 0 as n grows. It shifts with **embedding geometry**
(per-axis scale, not dimension count: padding the oracle with shared axes on the
signal's scale moves ρ\* from 0.38 to 0.81), and the small offset between the
measured crossing (≈0.377) and the analytic 0.370 is a finite-k smoothing bias
that vanishes as k/n → 0. The practical message is a check with a receipt: **a
critical-overlap threshold from a rotation-invariant recovery metric should be
compared against the metric's own extrapolation geometry — for the reader's
marginal, embedding dimension, per-axis scale, and k — before it is read as a
phenomenon.**

As a worked example we then ask the underlying question honestly, with the metric
fixed (a rotation-sensitive MCC gap against a matched-noise oracle, so the
residual is entanglement and not merely a noisier embedding) and a data-
generating process in which biological support overlap (ρ) and removable batch
magnitude (δ) are provably decoupled (an affine map removes 100% of δ and 0% of
ρ). Across a conditional VAE and an iVAE (variability condition satisfied and
violated), the learned latent stays **within a ρ-independent gap (~0.10) of the
matched-oracle ceiling at every overlap** — a pre-registered null, which we make
a claim rather than an absence with an equivalence test (TOST) against the
pre-registered effect size after expanding the design from 5 to 25 seeds per cell
(the pilot's minimum detectable effect was d≈2.0; the confirmatory design reaches
d≈0.8). We are explicit that this recovery-invariance null is for a deliberately
clean regime — a within-domain-decodable 1-D shift, where a faithful per-cell
encoder recovers the latent whatever the overlap — and that the domain-general,
correctness-solid contribution is the metric result above. What overlap does bound
is domain *removal*, but "removal ≈ ρ" is an identity of the metric's definition
(the Bayes-optimal domain classifier has balanced accuracy 1 − ρ/2), and it is
not a hard bound: an objective explicitly maximising cross-domain mixing can push
removal above ρ, at a point-for-point cost in recovery (MCC 0.86 → 0.67 at
ρ=0.05). The honest statement is a recovery-preserving frontier, not a ceiling.

## Why TMLR

TMLR judges technical correctness and evidence over perceived significance, with
rolling double-blind review. The metric result — a provable invariance, a derived
closed form, and a reproducible recipe — is exactly the kind of correctness-first
contribution its bar accepts; the null and the frontier are the worked example.
Lead the submission on the metric result.

## Positioning (one paragraph for the intro)

Over-correction in single-cell integration is a recognised failure mode with
methods (STACAS) and metrics (RBET, scIB) built for it; our contribution is not
"non-shared populations break integration" but (i) a *provable* blindness of
rotation-invariant recovery metrics to the identifiability transformation, (ii) a
*derived* closed form showing a critical overlap threshold can be a metric-
geometry constant (1 − 4^(−1/3) for uniform marginals) with an explicit recipe
for computing it in any setting, and (iii) with the metric and DGP fixed, a
pre-registered, equivalence-tested recovery-invariance null and an honest
recovery-preserving removal frontier. The iVAE variability condition (Khemakhem
et al. 2020) is used to define the satisfied/violated arms; we report frankly
that it does not bite in this clean regime.
