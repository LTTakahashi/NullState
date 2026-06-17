# Aim 2 Method Reconciliation — draft for review

**Purpose.** Bring the proposal documents into agreement with the pilot report's Phase 1 decision to make **contrastiveVI the primary disentanglement method**, with an origin-predicting adversary demoted to a *verification check*. This is a wording reconciliation, not a science change — the pilot already adopted this stance (`reports/nullstate_pilot_report.tex` §7, item 3).

**Why it matters.** Three documents currently imply three different primary methods for Aim 2:
- `specific_aims_organoid_offtarget_geometry.md` — names **adversarial training** as the primary mechanism (contrastiveVI a benchmarked alternative).
- `significance_and_innovation.md` — describes a **freeze-from-on-target** method (contrastiveVI in all but name) and warns that a "standard adversarial approach" causes negative transfer.
- `reports/nullstate_pilot_report.tex` §7 — states **contrastiveVI is primary**, adversary is a check.

A reviewer reading the package back-to-back will notice Aim 2's headline method (adversarial) sits awkwardly beside the Innovation section's warning against adversarial alignment. Switching the primary to contrastiveVI removes the ambiguity and matches what the pilot actually validated. **Nothing about the go/no-go logic, the single-vs-conditional z_iv split, or the controls changes.**

**Status:** proposed redlines below — *not yet applied*. Approve and I'll make the in-place edits.

---

## Edit 1 — `specific_aims_organoid_offtarget_geometry.md` (Aim 2) · PRIMARY

The only load-bearing change. Replace the entire Aim 2 paragraph.

### Current
> **Aim 2. Disentangle the in-vitro stress signature from lineage biology by adversarial subspace separation.** We will learn a representation that partitions variation into a shared *in-vitro* subspace, common to all organoid cells regardless of lineage, and a *lineage* subspace. We will use adversarial training in which an adversary attempts to predict organoid-versus-primary origin from the lineage subspace; successful disentanglement is reached when it cannot. This will be benchmarked against contrastive variational autoencoder and conditional flow-matching alternatives. Disentanglement will be verified by confirming that the in-vitro subspace predicts culture conditions while the lineage subspace predicts cell identity but not organoid origin. *Outcome:* a representation in which off-target states can be compared on lineage signal alone, with the trivial "grown-in-a-dish" confound removed with mathematical equity across all three germ layers.

### Proposed
> **Aim 2. Disentangle the in-vitro stress signature from lineage biology by contrastive subspace separation.** We will learn a representation that partitions variation into a shared *in-vitro* subspace (z_iv), common to all organoid cells regardless of lineage, and a *lineage* subspace (z_lin). The primary method is a contrastive latent-variable model (contrastiveVI) trained with primary on-target cells as background and organoid on-target cells as target, so that the salient latent absorbs the organoid-specific in-vitro signature while the shared latent retains lineage identity; the in-vitro axis is thus learned from identity-matched on-target cells and applied to off-target cells without anchoring them to a primary identity (the calibration-without-anchoring strategy of the Innovation section). We adopt the contrastive formulation rather than adversarial subspace separation because its objective is stable and its shared/salient partition is identifiable, whereas adversarial minimax training is unstable and its solution is not; an adversary that attempts to predict organoid-versus-primary origin from z_lin is retained only as a post-hoc verification check (disentanglement is confirmed when it cannot), and conditional flow-matching is retained as a benchmarked alternative. Disentanglement will be verified by confirming that z_iv predicts culture conditions while z_lin predicts cell identity but not organoid origin. *Outcome:* a representation in which off-target states can be compared on lineage signal alone, with the trivial "grown-in-a-dish" confound removed with mathematical equity across all three germ layers.

### What changed and why
- Title: "adversarial subspace separation" → "contrastive subspace separation."
- Primary method is now contrastiveVI, with the background/target construction stated explicitly (ties to the Innovation section's freeze-and-apply language).
- The adversary is kept — but only as verification. Flow-matching is kept as a benchmarked alternative, preserving the original breadth.
- Adds z_iv / z_lin notation, matching `research_strategy_foundation.md` and the README.
- One added clause states *why* (stability + identifiability vs. unstable, non-identifiable minimax) — the defensible reason the pilot made the switch.

---

## Edit 2 — `research_strategy_foundation.md` (§2) · SECONDARY

§2 is already method-neutral (it specifies the go/no-go, not the architecture), so this is an **append**, not a rewrite. Add one paragraph at the end of §2, immediately after the "hard case" paragraph that ends "...reported as a limit on the cleanliness of z_lin for that population, not silently left in."

### Proposed addition
> **Implementing architecture (and the pilot's verdict).** The in-vitro subspace is realized with a contrastive latent-variable model (contrastiveVI): primary on-target cells serve as background and organoid on-target cells as target, so the salient latent captures the in-vitro signature and the shared latent retains lineage identity — a stable, identifiable replacement for adversarial subspace separation, with an origin-predicting adversary on z_lin retained only as a verification check. In the HNOCA pilot the dish-vector test returned GO (mean pairwise cosine 0.906 > 0.70), so the single frozen z_iv of the Go path is the active design; the conditional z_iv | y_id remains the specified fallback for any later germ layer that fails the same test.

### What changed and why
- Names the implementing architecture (none was named before) and ties in the pilot's GO result, so the document records that the single-z_iv Go path is the *active* design, not just one branch.
- Leaves the entire go/no-go and conditional-model logic untouched.

---

## Edit 3 — `significance_and_innovation.md` (Methodological paragraph) · TERTIARY

A one-sentence insertion to name contrastiveVI and disambiguate it from the "standard adversarial approach" the same paragraph (correctly) warns against. Insert after "...removing the culture artifact without imposing a primary anchor."

### Current (excerpt)
> Our solution learns the in-vitro culture signature exclusively from identity-matched on-target cells, then freezes and applies it to off-target cells, removing the culture artifact without imposing a primary anchor. This converts the on-target cells from a passive control into the calibration instrument for the entire assay, and the strategy generalizes to any setting requiring removal of a technical confound from a population that has no clean reference.

### Proposed (excerpt)
> Our solution learns the in-vitro culture signature exclusively from identity-matched on-target cells, then freezes and applies it to off-target cells, removing the culture artifact without imposing a primary anchor. We realize this with a contrastive latent-variable model (contrastiveVI), whose shared/salient latent partition performs the freeze-and-apply separation under a stable, identifiable objective; an origin-predicting adversary, if used, serves only to verify the result rather than to perform the separation. This converts the on-target cells from a passive control into the calibration instrument for the entire assay, and the strategy generalizes to any setting requiring removal of a technical confound from a population that has no clean reference.

### What changed and why
- Names contrastiveVI so the term is consistent across all docs.
- The existing critique of the "standard adversarial approach" (which is about *alignment/domain-adaptation* forcing collapse, a different use of "adversarial") is left intact; the new sentence makes clear the only adversary in *our* method is a verifier.

---

## Considered, leaving unchanged (so you know it was checked)

| Location | Text | Verdict |
|----------|------|---------|
| `README.md` line 17 | "...by subspace separation." | Already method-neutral and consistent. *Optional* alignment: change to "by contrastive subspace separation" to mirror the new Aim 2 title. |
| `README.md` lines 30–31, 44 | `z_iv` GO / conditional in the mermaid + gate table | Consistent with the Go path; no change. |
| `research_strategy_feasibility_power.md` §6 | "The micro-cluster case, handled **adversarially**" | Different sense — adversarial *stress-testing* of a rare-class comparison, not adversarial training. **Do not change** (changing it would introduce an error). |
| `specific_aims...` Aim 3 + Impact | "the disentanglement method developed in Aim 2..." | Generic reference; remains valid once Aim 2 is contrastive. No change. |
| `reports/nullstate_pilot_report.tex` §7 | contrastiveVI-primary | This is the source of truth; no change. |

---

## Application checklist (once approved)

1. `specific_aims_organoid_offtarget_geometry.md` — replace the Aim 2 paragraph (Edit 1). *Required.*
2. `research_strategy_foundation.md` — append the "Implementing architecture" paragraph to §2 (Edit 2). *Required.*
3. `significance_and_innovation.md` — insert the contrastiveVI sentence (Edit 3). *Required.*
4. `README.md` — optional one-line title alignment (Edit-4 optional).
5. Leave `research_strategy_feasibility_power.md` §6 and all `z_iv` go/no-go logic untouched.

Net effect: one paragraph rewritten, one paragraph added, one sentence inserted — the three documents and the pilot report then describe the same Aim 2.
