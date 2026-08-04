# support-overlap-identifiability

**TL;DR.** On single-cell-like data with *known* ground truth, the identifiability
of a shared biological latent under a domain shift is governed by the **support
overlap** of the domains in the biological factor, not by the size of the batch
effect. Below a critical overlap ρ\*, **no method can jointly recover biology and
remove the domain** — the objectives become infeasible — and this failure is
**certifiable in advance**, from the data, by a capacity-stable source adversary
plus an off-manifold overlap score. Structural priors (mechanism sparsity, prior
knowledge, anchors) shift ρ\* but do not remove it.

This turns the single anecdote in NullState (one organoid→reference mapping where
disentanglement provably failed) into a **characterized phase transition with a
sound, pre-hoc certificate** — the negative-side counterpart to Uhler's positive
identifiability guarantees, the precondition Sridhar's/Ding's methods assume, and
the structural-vs-practical distinction Stumpf draws.

---

## The claim (falsifiable — commit to it, then try to break it)

> For a shared biological factor observed in two domains with support overlap ρ:
> 1. **Frontier collapse.** There is a critical ρ\* below which the achievable
>    (biology-recovered, domain-removed) frontier cannot reach the top-right
>    corner. `max over correction-strength of min(recovery, removal)` drops
>    sharply at ρ\*.
> 2. **Certifiability.** A raw source-adversary (stable across capacity) plus an
>    off-manifold score predicts ρ\* **before** any integration is run.
> 3. **Structural, not practical.** Below ρ\*, recovery does not improve with n.
> 4. **Priors relax, don't remove.** Mechanism sparsity / prior structure /
>    anchors move ρ\* rightward but a support-overlap floor remains.

If (1)–(4) hold, this is a genuine finding, not a demo.

---

## Design decisions that actually matter

**1. The generative model is the whole ballgame (`dgp.py`).** A shared factor `t`,
a shared nonlinear decoder `g(t)`, and two independent knobs:
`rho` = biological support overlap; `delta` = magnitude of a *removable,
t-independent* batch offset. Because `g` is shared, any failure to recover the
cross-domain biological axis is attributable to support mismatch alone. Counts via
Poisson/NB so the result transfers to real scVI-style settings.

**2. Measure a FEASIBILITY FRONTIER, not recovery at one setting.** A vanilla VAE
with an isotropic prior is *provably non-identifiable* (Khemakhem 2020; Locatello
2019), so "the VAE fails to recover t" would be a confounded claim. Worse, our own
smoke test shows that at *fixed* correction strength, recovery of `t` stays high
even at ρ=0 while domain-removal collapses — the failure is in the **joint**
objective. So the headline is traced by **sweeping the domain-removal strength**
(`sweep_frontier`, adversarial `adv_lambda`) and showing the top-right corner
(both high) is reachable above ρ\* and unreachable below. Joint infeasibility *is*
the operational definition of non-identifiability, and it sidesteps the vanilla-VAE
confound cleanly.

**3. The ρ-vs-δ dissociation is the rigor (`sweep_grid`).** The naive story — "an
adversary that detects the batch predicts failure" — is *wrong*, and proving it is
the contribution. Recovery must collapse along **ρ** and stay flat along **δ**,
while a raw source-adversary is high whenever *either* ρ is low *or* δ is high. So
source-detectability alone conflates the harmless case (big removable batch, full
overlap) with the fatal case (support mismatch). That is exactly why NullState
measured off-manifold + entropy *and* the adversary, not the adversary alone — and
this proves that was necessary.

**4. Use the field's standard metrics, and position against prior work
(`metrics.py`).** Overcorrection (batch correction erasing biology) is *known*:
there are methods (STACAS) and metrics (RBET; scIB's AvgBIO/AvgBATCH; iLISI) for
it. The contribution here is **not** a new metric — it is showing these axes become
**jointly infeasible past a support-overlap threshold**, with a **pre-integration**
certificate, bridged to ML identifiability theory. `bio_recovery` / `batch_removed`
mirror scIB's AvgBIO / AvgBATCH by design, so the frontier is legible to the
single-cell community. **Cite and contrast:** Luecken 2022 (scIB), Korsunsky 2019
(LISI), Andreatta 2024 (STACAS), Wang 2025 (RBET). State the delta explicitly or a
reviewer will say "known."

**5. The certificate is a-priori (`metrics.raw_certificate`).** RBET/scIB evaluate
*after* integration; our certificate is computed from raw x *before*, and its
signature is being **high AND stable across adversary capacity** (NullState's
"≥0.998 regardless of latent capacity"). High-and-stable ⇒ domain perfectly
predictable ⇒ support mismatch ⇒ integration will extrapolate, not adjust.

**6. Structural vs practical (`sweep_n`) — the Stumpf cut.** At ρ below threshold,
sweep n and show recovery stays flat and the certificate stays ~1.0: the
non-identifiability is a property of the population, not the sample. Contrast with
ρ just above threshold, where more data helps.

**7. Align to Uhler's theorem (the moonshot hook).** The synthetic threshold is an
*empirical* boundary. Read *Identifiability Guarantees for Causal Disentanglement
from Purely Observational Data* (arXiv 2410.23620), extract the specific
observational condition the guarantee rests on, and mark on the same ρ axis where
that condition breaks (`plot.make_figure(uhler_threshold=...)`). **Do the empirical
collapse and her theoretical condition coincide?** Coincide ⇒ your adversary is a
cheap operational test for her condition. Gap ⇒ a regime her guarantees don't cover.
Either answer is publishable and is exactly what her email asks. Mapping her
condition is *your* deep-read step — and that read is what makes the outreach
credible.

---

## Repo

| file | what |
|------|------|
| `dgp.py`      | data-generating process; the `rho`/`delta` knobs |
| `models.py`   | VAE variants: vanilla · conditional (scVI-style) · contrastive (contrastiveVI-style) · adversarial (DANN); extension stubs for sparsity/prior/anchor and a scvi-tools spot-check |
| `metrics.py`  | `bio_recovery`, `batch_removed`/`source_leakage`, `ilisi`, `raw_certificate`, `off_manifold` |
| `sweep.py`    | `sweep_frontier` (headline), `sweep_rho`, `sweep_grid`, `sweep_n` |
| `plot.py`     | `panel_frontier`, `panel_joint_vs_rho`, `panel_a/b/c`, `make_figure` |

### Run
```bash
pip install -r requirements.txt

# tiny end-to-end sanity check (seconds):
python sweep.py

# the headline frontier (below vs above threshold), then plot:
python -c "from sweep import sweep_frontier; from plot import panel_joint_vs_rho; \
import matplotlib.pyplot as plt; \
df=sweep_frontier(rhos=(0.05,0.15,0.25,0.35,0.6,1.0), adv_lambdas=(0,0.3,1,3,10,30,100), seeds=(0,1,2)); \
panel_joint_vs_rho(df); plt.savefig('joint.png',bbox_inches='tight')"
```

---

## The one figure

- **Panel A / frontier**: (domain removed) vs (biology recovered) as correction
  strength sweeps, one curve per ρ; top-right reachable only above ρ\*. Companion
  scalar: `max_λ min(recovery, removal)` vs ρ, collapsing at ρ\*.
- **Panel B**: ρ×δ heatmap of recovery — collapses along ρ only.
- **Panel C**: recovery vs n at a below- vs just-above-threshold ρ — flat below.
- **Overlay** on Panel A: the raw certificate (twin axis) and, once mapped, Uhler's
  theoretical threshold line.

A stranger reads the finding in ten seconds: recovery falls off a cliff at a
specific overlap, the cheap certificate calls the cliff in advance, and more data
does not save you.

---

## Execution — tiered, with a hard gate

1. **Core (~1 week).** DGP + adversarial `sweep_frontier` (Panel A) + `sweep_grid`
   (ρ/δ dissociation) + `raw_certificate` + `sweep_n`. **This alone ships.**
   **Gate G2 @ day 10:** if the frontier collapse isn't clean, descope to
   `sweep_rho` with the certificate and one method. A small true result ships; an
   ambitious half-result doesn't.
2. **Methods** — fill `models.py` sparsity/prior/anchor stubs; show each shifts ρ\*
   but leaves a floor. (Sridhar/Lachapelle, Ding.)
3. **Real data** — run the identical `raw_certificate` + off-manifold on the
   NullState organoid↔reference data; show it predicts where integration is
   trustworthy vs extrapolating. Closes the loop: NullState proposed the
   certificate, the synthetic core proved it sound, the real arm shows it working.
4. **Uhler condition** — the theorem alignment (§7).

---

## Honest risks (keep the repo truthful)

- **Transition may be gradual, not a sharp cliff.** Then characterize the
  transition; report it as-is. *Do not tune the DGP to manufacture a cliff.*
- **Certificate may not perfectly predict ρ\*.** The *gap* is a finding (and the
  Uhler question), not a failure. Report it.
- **Nonlinearity + count noise can make the VAE fail for optimization reasons.**
  Keep a linear-Gaussian easy setting as a sanity anchor where theory is cleanest,
  then add realism and show the phenomenon survives; run multiple seeds and check
  convergence so you measure identifiability, not training luck.
- Keep negative/null results in the repo. For these PIs, the honesty *is* the
  evidence.

---

## Why this is the right bet for the cycle

One artifact, quadruple duty: it upgrades every artifact-anchored email from "my
related work lands on your question" to "I ran the exact regime, here is the repo
and the figure"; it gives the SOP its second paragraph (a clean arc from *finding a
boundary* to *characterizing the phenomenon and building a sound certificate*); it
is novel enough to become a short paper regardless of any single application; and it
is the one thing that converts the Uhler moonshot from a lottery ticket into a real
shot, because a verifiable test of her own theorem is the currency that makes a PI
at her level look twice.
