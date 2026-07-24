# TMLR draft — title + abstract

## Title

**When a Critical Threshold Is a Geometry Constant: Rotation-Invariant Recovery
Metrics and Spurious Identifiability Thresholds in Domain Integration**

## Abstract

Recovery metrics based on k-nearest-neighbour regression or cross-domain transfer
are widely used to assess whether a latent representation has been recovered, in
disentanglement and in single-cell data integration. We show that such metrics
are near **rotation-invariant** — they score an entangled (rotated) latent the
same as a disentangled one — and that this is not a benign approximation but a
failure mode that can manufacture results. In a controlled support-overlap probe,
a kNN-transfer recovery score produced a clean "critical overlap" ρ\* ≈ 0.37
below which recovery appeared to collapse. That number is exactly **1 − 4^(−1/3)**,
the zero-crossing of the metric's own extrapolation geometry (transfer =
1 − 4(1 − ρ)³); it is present for an oracle embedding as much as a learned one,
and no model or quantity of data can move it. The threshold was the metric, not
the phenomenon.

We then rebuild the experiment to test the underlying question honestly. We
introduce a data-generating process in which biological support overlap (ρ) and
removable batch magnitude (δ) are **provably decoupled** — verified by
removability: an affine batch map removes 100% of δ's effect and 0% of ρ's — and
a **matched-noise oracle** estimand: isotropic noise is added to the true latent
until it carries the same information as the learned embedding, so the learned −
oracle MCC gap is attributable to entanglement rather than to a noisier
embedding. Under this rotation-sensitive, information-matched metric the collapse
disappears: across three model families (a conditional VAE and iVAE with the
variability condition satisfied and violated) the shared latent is recovered up
to the metric ceiling at **every** overlap level (Cohen's d < 0.8, non-monotone,
BIC-linear); we pre-registered the null as an outcome.

What overlap bounds is not recovery but domain **removal**. This is a feasibility
frontier we characterise from both directions: a faithful per-cell encoder sits
at removal ≈ ρ with recovery intact, and an integration objective explicitly
maximising cross-domain mixing likewise cannot beat removal ≈ ρ up to strong
regularisation — and can exceed it at low overlap only by collapsing recovery
point for point (at ρ=0.05, no setting achieves both MCC > 0.8 and removal >
0.1). removal ≤ overlap thus bounds the recovery-preserving region. We give a
diagnostic that separates removable batch from support mismatch before
integration (an anchor-calibrated irreducible-gap certificate; we show the
unsupervised version is provably limited via Ben-David & Luu's domain-adaptation
impossibility). The practical message is a warning with a receipt: **any
critical-threshold result derived from a rotation-invariant recovery metric in
fixed embedding dimension should be checked against the metric's own geometry
before it is read as a phenomenon.**

## Why TMLR

TMLR judges technical correctness and evidence, not perceived significance or
novelty, and runs a rolling double-blind review. A rigorous null paired with a
metric critique and a reusable diagnostic is exactly the kind of contribution its
bar is designed to accept. Lead the submission on the metric finding; the null
and the analytic removal bound are the worked example that makes it concrete.

## Positioning (one paragraph for the intro)

Over-correction in single-cell integration is a recognised failure mode, with
methods (STACAS) and metrics (RBET, scIB) built for it; the novelty here is not
"non-shared populations break integration" but a controlled, identifiability-
theoretic dissection showing (i) a rotation-invariant metric can report a
critical threshold that is a geometry constant, (ii) with the metric and DGP
fixed, recovery is overlap-invariant and only removal is overlap-bounded, and
(iii) that bound is analytic and confirmed from the mixing direction. The iVAE
variability condition (Khemakhem et al. 2020) and the Ben-David/Luu impossibility
results are used as the theoretical spine.
