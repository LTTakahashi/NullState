# Research Strategy — Feasibility, Sample Size, and Statistical Power

We treat sample size as the most likely point of failure and design against it explicitly. Headline atlas counts are irrelevant to this project; what matters is the cell count surviving the full subsetting funnel — germ layer → organoid origin → off-target classification → per-protocol batch → high-QC, low-mapping-entropy cells. We state below what survives that funnel, where the geometry becomes unstable, and how the project yields a complete result even under the worst realistic counts.

## 1. The cell-count reality after subsetting

The funnel produces a sharply bimodal availability of off-target populations.

**The mesenchymal surplus (powered).** Every organoid protocol surveyed — neural (HNOCA), kidney, and endodermal (HEOCA) — reproducibly generates off-target mesenchymal/stromal cells, typically 10–30% of total cells. After the full funnel this leaves on the order of 10⁴ off-target mesenchymal cells per germ layer, distributed across multiple protocols. These comparisons are well above any plausible stability threshold and form the confirmatory core of the project.

**The rare-class asymmetry (likely underpowered).** Non-mesenchymal off-target states — off-target neural cells in kidney organoids, off-target endodermal cells in neural organoids — exist but are typically <1% of a dataset, frequently leaving N in the low hundreds or below, and often dominated by a single batch or donor. A direct comparison between a 15,000-cell brain-derived mesenchymal population and a 120-cell kidney-derived neural population is not a valid measurement, for reasons detailed next. We treat these comparisons as exploratory throughout and never allow a confirmatory claim to rest on them.

## 2. Why the geometry is N-sensitive, and the asymmetry trap

The empirical Wasserstein-1 distance between two finite samples carries a bias that, for exact optimal transport in dimension *d* > 2, decays only as *n*^(−1/d). In a 15–30 dimensional latent space this is severe: two samples drawn from the *same* distribution will show a substantial nonzero empirical distance, and that spurious distance grows as N shrinks. A 120-cell sample therefore looks "far" from everything, including a second draw of itself. Comparing populations of grossly unequal size compounds this: the small sample's intrinsic inflation, not biology, dominates the result.

This yields the single non-negotiable rule of the analysis: **raw distances are never interpreted.** Every distance is read only relative to a matched-N self-distance floor and the matched-N Control 3 baseline, with all three computed at identical N so the *n*^(−1/d) bias is shared and cancels in the comparison.

## 3. Estimator choices that mitigate the dimensionality problem

We do not use exact W₁. We use estimators with sample complexity that is robust to dimension:

- **Sliced-Wasserstein distance** as the primary metric. By averaging one-dimensional Wasserstein distances over random projections, it achieves an *n*^(−1/2) convergence rate independent of *d*, removing the curse-of-dimensionality penalty that cripples exact OT in latent space.
- **Debiased Sinkhorn divergence** as a cross-check, with entropic regularization tuned to the per-comparison N; it interpolates between OT and MMD and is markedly more sample-stable than exact OT at small N.
- Optimal transport is computed on a reduced z_lin representation (the leading 15–30 disentangled dimensions), not the full gene space, to keep the effective dimensionality tractable.
- For topology, **H₀ persistence with subsampling-based confidence bands** (Fasy-style bootstrap confidence sets for persistence diagrams) is confirmatory; **H₁ is exploratory only**, with sensitivity testing across k-NN parameters, exactly because its features are sampling-density artifacts at the N available here.

## 4. Empirical determination of the minimum N (rarefaction)

We do not assert a stability threshold; we measure it. Taking the largest available population (brain-derived mesenchyme), we subsample to an N-ladder (50, 100, 200, 500, 1,000, 2,000, 5,000) and compute the split-half self-distance (sliced-Wasserstein between two disjoint subsamples) at each N. The curve saturates where additional cells no longer reduce the self-distance; that saturation point defines N_min, the minimum cell count at which a population's geometry is stable. Based on the *n*^(−1/2) behavior of sliced-Wasserstein in this dimensionality, we anticipate N_min in the low hundreds, but the pilot fixes it empirically. Any population below N_min is barred from confirmatory geometric claims by rule, not by judgment after seeing results.

## 5. The matched-N null-floor protocol

For every pairwise comparison:

1. Set N* = min(N_P, N_Q), capped so the comparison only proceeds confirmatorily if N* ≥ N_min.
2. **Self-distance floor (same-distribution null):** within each population, repeatedly split into two disjoint N*-cell subsamples and compute the sliced-Wasserstein distance; the distribution of these is the noise floor expected when nothing differs, at exactly this N.
3. **Cross-distance:** subsample both populations to N* and compute the cross-population distance, bootstrapped B = 1,000 times for a confidence interval.
4. A difference is declared real only if the cross-distance CI lies above the self-distance floor; convergence (Aim 3) is declared only if it additionally lies below the Control 3 maturation-matched baseline, by permutation test with FDR correction across all pairs.

This makes the test self-calibrating: each comparison carries its own N-specific null, so density differences and small-sample inflation cannot masquerade as signal.

## 6. The micro-cluster case, handled adversarially

For a rare population at N ≈ 120 compared against an abundant one: we subsample the abundant population down to 120, bootstrap, and build the N = 120 self-distance floor from disjoint 120-cell splits of the abundant population. Three honest consequences follow.

First, bootstrapping a 120-cell sample does not manufacture power — it characterizes the (wide) sampling variability of that one observed sample, and the resulting confidence intervals will be correspondingly wide. We display that width rather than suppress it.

Second, at N = 120 in this dimensionality the self-distance floor is high, so only very large effects are detectable. When a rare-class comparison's CI overlaps its self-distance floor, the result is reported as **inconclusive — underpowered**, never as evidence of convergence. Absence of a detectable difference at N = 120 is not evidence of a shared state.

Third, rare-class comparisons are formally demoted to a labeled exploratory secondary analysis, presented with their achieved power stated alongside each estimate. They generate hypotheses for future targeted sequencing; they do not support conclusions.

## 7. Simulation-based power analysis

No closed-form power exists for a permutation test on sliced-Wasserstein distances, so power is established by simulation. From the abundant mesenchymal data we construct synthetic scenarios with known ground truth — a "convergent" scenario (two populations drawn closer by a controlled amount) and a "divergent" scenario (held apart) — across a grid of effect sizes δ and sample sizes N. For each cell we compute the probability of correctly rejecting the null at the target FDR, yielding power curves N(δ): the minimum N to detect a convergence effect of size δ. Before interpreting any real comparison, we report the minimum detectable effect size at its available N, so each result is accompanied by what it could and could not have found.

## 8. Graceful degradation: the result does not depend on the rare classes

The project is structured so that its central, publishable claim is delivered entirely by the mesenchymal populations, which are abundant in every germ layer. The worst-case outcome — that *only* mesenchymal off-target states clear N_min — still produces a complete and high-value paper: a definitive, maturation-controlled test of whether the most ubiquitous organoid contaminant, off-target mesenchyme, converges across germ layers toward a shared default state or retains lineage-specific structure. Because off-target mesenchyme appears in essentially every organoid protocol, an answer about it has the broadest possible relevance to the stem-cell field, and it carries a direct translational corollary: quantifying the convergent-contaminant fraction of each organoid model's total mass — a fidelity metric drug developers can use immediately.

In this framing the rare non-mesenchymal off-target states are upside, not load-bearing. If they clear N_min, they extend the convergence map; if they do not, the central result stands unaffected. The project therefore has no single point of sample-size failure: the abundant class guarantees the core deliverable, and the design refuses to let the scarce classes contaminate it.

## 9. Bottom line for the reviewer

The powered core of this project is off-target mesenchyme, where counts exceed any stability threshold by orders of magnitude. The geometric machinery is built on dimension-robust estimators (sliced-Wasserstein, Sinkhorn divergence), interpreted only against matched-N null floors, with the minimum stable N fixed empirically by rarefaction and confirmed by simulated power curves. Rare off-target classes are surveyed honestly, reported with their achieved power, and walled off from the confirmatory claim. The result is a project whose main finding is robust to the cell counts that actually exist in the 2025–2026 atlases, not the cell counts a headline number implies.
