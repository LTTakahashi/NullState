# Support-overlap identifiability probe — v2

## The finding, first

**Rotation-invariant recovery metrics cannot detect entanglement, and a
"critical threshold" derived from one can be a constant of the metric's own
geometry.** kNN-R²/transfer scores are *exactly* invariant to any orthogonal
transform of the embedding (up to measure-zero neighbour ties) — the same
rotations that leave an isotropic-Gaussian prior invariant — so they are blind to
the identifiability transformation. The retired v1 probe reported a clean
critical overlap ρ\* ≈ 0.37; that is exactly **1 − 4^(−1/3)**, the zero-crossing
of the kNN transfer curve (transfer = 1 − 4(1 − ρ)³), present for a perfectly
identified *oracle* embedding. The constant is a recipe — marginal-, scale-, and
k-dependent, stable only for compact-support marginals — not a universal. With a
rotation-sensitive matched-oracle metric and a DGP where support overlap (ρ) and
removable batch magnitude (δ) are provably decoupled, the "cliff" vanishes:
recovery stays within a ρ-independent gap of the ceiling. What overlap bounds is
only *removal*, and that is a recovery-preserving frontier (removal ≈ ρ is a
metric-definition identity; a mixing objective can exceed it only by collapsing
recovery), not a hard bound. See [`FINDINGS_v2.md`](FINDINGS_v2.md) for the full
argument and [`ABSTRACT_tmlr.md`](ABSTRACT_tmlr.md) for the paper framing.

This is the load-bearing, generalisable result: kNN-R² and kNN-transfer recovery
scores are common in the integration and disentanglement literature, and any
critical-threshold claim built on one should be checked against the metric's own
geometry (for the reader's marginal, dimension, per-axis scale, and k) before it
is read as a phenomenon.

## The rebuild

v1 was retired after an adversarial audit (28 agents, 33 findings, 17 confirmed;
verdict: *"confounded at every measured axis"*; retired artifacts in
[`archive_v1/`](archive_v1/), documented in
[`RETRACTED.md`](archive_v1/RETRACTED.md)). v2 is a ground-up rebuild whose
discipline is a chain of **hard gates**: no stage runs until the stage below has
provably passed. It answers the underlying question — is recovery of the shared
latent governed by support overlap ρ, or only by removable batch magnitude δ? —
honestly, with the null pre-registered as a publishable outcome. The short
answer: recovery is governed by *neither* here; overlap bounds only removal.

## Pipeline and gates

| stage | file | gate | status |
|---|---|---|---|
| 1. DGP | [`dgp2.py`](dgp2.py) | [`verify_dgp2.py`](verify_dgp2.py) — ρ and δ decoupled | **PASS** |
| 2. metrics | [`metrics2.py`](metrics2.py) | [`verify_stage2.py`](verify_stage2.py) — metric ceiling flat in ρ | **PASS** |
| 3. models | [`models2.py`](models2.py) | [`verify_stage3.py`](verify_stage3.py) — trains as a VAE; conditioning live | **PASS** |
| 4. main sweep | [`run_stage4.py`](run_stage4.py) | [`analyze_stage4.py`](analyze_stage4.py) — pre-registered decision rule | see results |
| 5. certificate | [`certificate.py`](certificate.py) | validated in `__main__` | anchor-based version passes |

### Stage 1 — the DGP is the whole ballgame (and is where v1 died)

v1's ρ was not a pure knob: within-dataset per-gene standardisation pushed the
shifted domain into decoder-tanh saturation, so lowering ρ *also* compressed the
signal 3.4×, opened a mean gap equivalent to δ≈36 (δ swept only to 4), and cut
depth 30%. Every headline metric then became a closed-form function of ρ.

v2 fixes each cause **by construction** and proves it in `verify_dgp2.py`:

- **Symmetric, pooled-invariant overlap.** With `w=(1-ρ)/2`, domain A puts mass
  `2w` on `[0,w)` and B puts mass `2w` on `(1-w,1]`, sharing `[w,1-w]`. Then
  `p_A+p_B≡2`, so the **pooled** marginal is *exactly* Uniform at every ρ while A
  has strictly zero density above `1-w` (real support mismatch). `OVL = ρ`
  exactly. *(The brief's "pin each domain's marginal AND set ρ=OVL" is
  unsatisfiable — OVL=1 iff the marginals are equal; pooled invariance is the
  achievable form.)*
- **Fixed reference-grid standardisation**, **non-saturating** smooth-leaky-ReLU
  mixing (derivative bounded in (α,1)), **depth drawn independently** of ρ/δ/s/t
  and multiplied after simplex normalisation, a **2-D latent** (shifted +
  unshifted axes, so rotations are non-trivial), and **n_env = 2n+1 = 5**
  environments so the iVAE variability condition is *satisfiable*.

Verified invariances (all pass): per-gene mean/variance/dispersion, library
size, dynamic range, and within-domain decodability are flat across ρ; domain
symmetry `std_B/std_A` and `depth_B/depth_A` are ~1 (v1: 0.344 and 0.699).

**The decisive Stage-1 result — removability (Ben-David's divergence-vs-λ
split).** An affine batch map fit on the shared band removes **100.0%** of δ's
expression gap (residual 0.0000) and **~0%** of ρ's (residual ≈ raw gap at
ρ=0.6/0.4/0.2). δ and ρ are separable *by construction*, not by assertion.

**Two limits, published rather than hidden.** (i) A ρ×δ interaction is intrinsic
to compositional data (variance ≤7%, dispersion ≤16% at δ=1) — so the primary
estimand is run at δ=0 where it is zero. (ii) Decoupling holds for **δ ≤ 2.0**;
worst within-domain decodability ratio is 0.983 at δ=2 but 0.964/0.925/0.837 at
δ=2.5/3/4, so those are flagged an explicit extreme arm outside the certificate.

### Stage 2 — a metric that can see entanglement

v1 scored with kNN-regression R², which is ~rotation-invariant: it gives a
45°-rotated (entangled) latent the same score as the identity, so it could
*never* detect the phenomenon. Worse, its cross-domain transfer measured kNN
extrapolation — its "critical ρ*=0.370" is exactly `1-4^(-1/3)`, a geometry
constant.

v2 uses **MCC** (Hungarian matching, train/test split — rotation- and
permutation-sensitive), **CCA** (the weak ∼_A notion), and **DCI**, each also
computed on a **matched-noise oracle** for the ceiling. The gate proves the
metric does not bend with ρ on its own: for a fixed-quality embedding the score
spread across the ρ sweep is **0.003** (v1 varied by ~3.5). A 45° rotation is
scored **MCC 0.707 / CCA 1.000** at every ρ — "right subspace, wrong axes", the
exact discrimination v1 lacked.

### Stage 3 — a real count VAE, with live knobs

v1's MSE-over-genes reconstruction against β=1 was an effective β≈0.003 — a
deterministic autoencoder, so the Locatello/Khemakhem framing did not apply and
its adversarial λ was inert. v2 uses a genuine **NB likelihood** on raw counts
with per-gene dispersion and the observed library size as size factor. The gate
confirms it trains as a VAE (recon ~1280 nats, KL ~6 over 2 dims, 2 active
units, no collapse, not deterministic) and that the **domain-conditioning knob
v2 actually uses is live**: at ρ=1/δ=2, domain-predictability of the shared
latent is vanilla 0.999 → conditional (scVI decoder covariate) **0.573**. *(The
iVAE conditional prior conditions on environment, not domain, so it leaves the
removable batch in the latent — a real mechanistic distinction, not a bug. The
adversarial variant is vestigial in v2 and structurally weak in the saturated
2-D-latent regime; documented, not used.)*

### Stage 4 — the claim, tested against a pre-registered rule

Primary estimand: the **matched-information MCC gap** = MCC(oracle matched to the
learned embedding's CCA) − MCC(learned). Isotropic noise cannot mix axes, so at
matched information the gap is attributable to entanglement alone — never to the
model simply having a noisier embedding (the confound a raw score hides, and
what audit finding B4 demanded be fixed). Arms: `ivae_5env` (condition
satisfiable), `ivae_2env` (provably violates it — where v1 unknowingly sat),
`conditional` (isotropic control). Decision rule fixed before results:
*supported* iff gap monotone in ρ with Cohen's d ≥ 0.8 (ρ=1 vs 0.2) and flat in
δ; *refuted/null* otherwise; "threshold" language only if a segmented fit beats
linear on BIC. Default expectation is **smooth degradation** — no theory
predicts a discontinuity in identifiability as a function of overlap.

Results and verdict: see `analyze_stage4.py` output.

### Stage 5 — the certificate

An unsupervised certificate that separates removable δ from support mismatch ρ
is **provably limited**: Ben-David & Luu (2010) prove unlabeled data cannot in
general distinguish an adaptable covariate shift from a non-adaptable one. The
module shows the whole unsupervised family failing (raw adversary, rank
projections, cross-fitted gene-wise alignments, UOT). The **anchor-based**
version — a few known-shared cells fix the removable affine map, then the
*irreducible per-gene mean-gap* is measured against a permutation-null floor —
resolves it, correctly calling all validation cells including the harmless
ρ=1/δ=2 case that v1 wrongly flagged as un-integrable. This connects the
certificate to semi-supervised integration (STACAS anchors) rather than to
unsupervised batch metrics.

## Reproduce

```bash
python verify_geometry_constant.py  # the headline receipt (v1 threshold = 1-4^(-1/3))
python verify_dgp2.py     # Stage-1 gate (must pass first)
python verify_stage2.py   # Stage-2 gate
python verify_stage3.py   # Stage-3 gate
python run_stage4.py      # main sweep -> results_stage4.csv  (resumable)
python analyze_stage4.py  # verdict against the pre-registered rule
python run_frontier.py    # recovery-vs-removal frontier (removal ~ rho)
python run_mixing_bound.py# removal bound confirmed from the mixing direction
python certificate.py     # Stage-5 validation table
```

## Documents

- [`FINDINGS_v2.md`](FINDINGS_v2.md) — the full argument, metric finding first.
- [`ABSTRACT_tmlr.md`](ABSTRACT_tmlr.md) — TMLR title + abstract (metric-led).
- [`archive_v1/RETRACTED.md`](archive_v1/RETRACTED.md) — why v1 was retired.
