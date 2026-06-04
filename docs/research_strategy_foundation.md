# Research Strategy — Foundation: Off-Target Definition, Calibration Validity, and the Maturation Control

This section anchors the three assumptions on which Aims 2 and 3 depend: that "off-target" can be defined operationally without circularity, that the in-vitro signature learned from on-target cells transfers to off-target cells, and that any measured convergence reflects a shared default state rather than shared immaturity. Each is specified as a concrete, testable procedure with explicit failure handling.

## 1. Operational Definition of Off-Target Identity (Aim 1)

**The discrimination problem.** The central difficulty is separating a true off-target cell (correct germ layer of origin, wrong lineage — e.g., mesodermal mesenchyme in a brain organoid) from a poorly differentiated on-target cell (correct lineage, immature or stressed — e.g., a badly patterned neuro-epithelial cell). The first must enter the Aim 2/3 analysis set; the second must enter the on-target calibration set. Misassignment is the most dangerous error in the pipeline because it contaminates the calibration instrument itself.

**Projection and the two-axis decision space.** Each organoid cell *i* from a known germ layer *G* is projected onto the integrated HCA fetal/adult reference using a reference-mapping tool that returns both a soft lineage assignment and an out-of-reference signal (Symphony or scVI-hub; we will benchmark both for assignment stability). For each cell we compute three quantities:

- **Germ-layer concordance:** the germ layer *GL_i* of the dominant assigned reference lineage, compared to the organoid's expected *G*.
- **Mapping entropy *H_i*:** the Shannon entropy of the soft lineage-assignment distribution over reference neighbors. Low entropy = confident assignment to one lineage; high entropy = confusion across several.
- **Reconstruction / off-manifold error *R_i*:** the distance from the reference manifold (latent reconstruction error under the reference model), measuring whether the cell resembles *any* real reference cell at all.

The decision rule operates on the joint space:

- **True off-target:** *GL_i ≠ G* (maps to a different germ layer), *H_i* low (confident), and *R_i* below the off-manifold gate. The cell confidently resembles a real cell type it should not be.
- **Poorly differentiated on-target:** *GL_i = G* (correct germ layer) but *H_i* high — ambiguous across early nodes of the correct lineage. This is immaturity, not off-target identity. Excluded from both the calibration set and the off-target analysis set; tracked separately.
- **Clean on-target (calibration set):** *GL_i = G* and *H_i* low.
- **Ambiguous / novel:** *R_i* above the off-manifold gate — the cell resembles nothing in the reference. Held out and reported; not forced into a category.

**Threshold calibration, not arbitrary cutoffs.** The entropy threshold τ_H is set at the 95th percentile of mapping entropy observed for held-out *primary* cells of the matching germ layer (i.e., the confusion level expected for genuine, well-differentiated cells). The off-manifold gate τ_R is set at the 99th percentile of reconstruction error for held-out primary cells. Both are data-derived floors, not hand-set values.

**The neural crest caveat.** Germ-layer concordance is not naïve. Neural crest is ectoderm-derived but mesenchymal; a "mesenchymal" cell in a neural organoid may be legitimate neural crest rather than off-target mesoderm. The reference lineage ontology will therefore encode developmental origin, not just morphology, so that neural-crest-derived mesenchyme in an ectodermal organoid is scored as on-target. This single distinction prevents a large, systematic false-off-target call in neural systems.

**Bounding annotation error.** We treat the off-target call as probabilistic, not binary, and propagate that uncertainty downstream. Three mechanisms: (a) a labeled benchmark — primary cells with known germ layer, plus organoid off-target populations independently validated by immunostaining in their source publications, used to estimate the false-off-target and false-on-target rates of the rule; (b) confidence-weighting — cells enter Aim 2/3 weighted by call confidence rather than as hard members, so borderline cells contribute proportionally; (c) full-pipeline sensitivity analysis — every downstream convergence result is recomputed across a strict-to-liberal range of τ_H. A result that holds across the threshold range is reported as robust; a result that flips with the threshold is reported as threshold-dependent rather than concealed.

## 2. The z_iv Consistency Check — Go/No-Go for the Calibration Strategy (Aim 2)

The on-target calibration assumes the in-vitro signature is a shared axis across on-target cell types. The stress literature predicts this is only partly true: glycolytic/hypoxic shifts are pan-organoid, but ER stress and the unfolded protein response are lineage-specific, scaling with a cell type's secretory demand. We test the assumption before relying on it.

**The dish-vector test.** For each on-target cell type *c* present in both primary and organoid data, compute the dish vector Δ_c = mean(organoid *c*) − mean(primary *c*) in the shared HVG space after within-type batch correction. Compute all pairwise cosine similarities between Δ_c vectors, and the fraction of total Δ variance captured by their shared first principal component.

**Go path (universal axis).** If mean pairwise cosine similarity exceeds 0.7 and the shared PC explains the majority of dish-vector variance, a single frozen z_iv is justified; Aim 2 proceeds as written.

**No-Go path (lineage-specific stress).** If similarities are low, the framework adapts rather than dies. We decompose Δ_c = Δ_shared + Δ_c^specific, where Δ_shared is the common glycolytic/hypoxic component and Δ_c^specific is the lineage-specific residual (ER/UPR). The encoder then models a conditional in-vitro subspace, z_iv | y_id — a generic dish axis modulated by an affine transformation keyed to broad lineage class — so that each lineage's characteristic culture stress is removed with its own coefficients.

**The hard case, stated honestly.** Off-target cells lack a clean *y_id*, so for the conditional model we apply the dish-signal conditional of the broad class the off-target cell most resembles (off-target mesenchyme uses the mesenchymal conditional). This requires that on-target cells of that broad class exist somewhere in the matched calibration set to estimate the conditional. Where they do not — an off-target lineage with no on-target counterpart in any germ layer — only the shared glycolytic component can be removed; the lineage-specific stress residual cannot be calibrated and will be explicitly bounded (by its magnitude in the nearest available class) and reported as a limit on the cleanliness of z_lin for that population, not silently left in.

## 3. The Maturation Confound Control — Third Tier of the Control Structure (Aim 3)

Measured convergence of off-target cells across germ layers has a deflating alternative explanation: organoids are developmentally immature, and immature progenitors of a broad class (mesenchymal-like, neuro-epithelial-like) are generically similar because canalization has not yet diversified them. Off-target cells may simply revert toward a shared primitive node (early mesendoderm, primitive streak) rather than a specific failure-default attractor. The third control isolates genuine convergence from shared immaturity.

**Maturation scoring, lineage-independent and dish-robust.** Each cell receives a differentiation/maturation score computed without lineage labels, using CytoTRACE2 and an RNA-velocity-derived latent-time estimate as orthogonal measures; only cells where the two agree (concordant rank) are used for matched comparisons, so the control does not rest on a single estimator's assumptions. Critically, maturation is scored on the dish-cleaned z_lin representation (after Aim 2), because culture stress perturbs total gene counts and would otherwise bias count-based maturation estimates.

**Control 3 — the maturation-matched primary baseline.** For each cross-germ-layer off-target pair *P* (e.g., neural-derived mesenchyme) and *Q* (kidney-derived mesenchyme), we construct a primary comparison: mesenchymal cells of the same broad class drawn from two different primary tissues, subsampled to match the maturation-score distributions of *P* and *Q*. The EMD between these maturation-matched primary populations measures how different same-class cells of different origin are *in vivo at matched developmental stage* — the convergence expected from immaturity alone.

**The discriminating test.** We compare EMD(*P*, *Q*) against the maturation-matched primary baseline:

- EMD(*P*, *Q*) **significantly below** the matched-primary baseline → off-target cells have converged *more* than their origin and maturation predict → genuine shared default state. This is the interpretable, interesting result.
- EMD(*P*, *Q*) **indistinguishable from** the matched-primary baseline → the convergence is fully explained by immaturity → the default-state hypothesis is not supported, and we report this honestly.
- EMD(*P*, *Q*) **above** the baseline → off-target cells retain *more* origin-specific structure than maturation predicts → lineage-bound, divergent failure modes.

Significance is assessed by permutation testing with FDR correction across all cross-germ-layer pairs. Because the primary baseline already encodes the real residual differences between same-class cells of different origin (lung vs. kidney fibroblasts differ even at matched maturation), this control sets the correct, non-trivial bar: convergence counts only when it exceeds what in-vivo biology shows for comparable cells.

## Residual Limitations (stated, not hidden)

Even with these three mechanisms, three risks remain and will be reported rather than obscured. First, the off-target definition inherits the error of the underlying reference-mapping tool; the labeled benchmark bounds this error but does not eliminate it. Second, if the dish signature proves strongly lineage-specific and the relevant off-target lineages lack on-target matches, z_lin for those populations carries an uncalibrated stress residual whose magnitude we can bound but not remove. Third, off-target populations are frequently a minority of organoid cells; per-population sample sizes after germ-layer stratification may limit the stability of EMD and H₀ estimates, and underpowered comparisons will be flagged rather than reported as null convergence. These are the genuine floors of the assay. The framework is designed so that each becomes a stated limit on interpretation, not a hidden source of false confidence.
