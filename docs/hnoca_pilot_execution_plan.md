# HNOCA Pilot — Execution Plan

**Objective.** In ~48 hours of compute on a single GPU, decide whether the full proposal is viable by running four gates on one germ layer (neural). The pilot answers three empirical questions the prose cannot: (1) does the off-target definition behave sanely on real cells, (2) is the in-vitro signature consistent enough across neural cell types to justify a single frozen z_iv, and (3) do enough off-target mesenchymal cells survive the filters to power Aim 3. Each gate ends in a hard threshold.

A standing caveat: exact `.obs` keys, layer names, and label strings below depend on the released HNOCA/HDBCA object structure and must be checked against the actual files; the logic is fixed, the field names are placeholders.

---

## Step 1 — Retrieval and the combined reference

The reference choice is the single most consequential decision in the pilot, because of a subtlety that determines whether the count check even works. Off-target mesenchyme in a neural organoid is *aberrant relative to brain* but *normal relative to mesoderm*. If we map it against a brain-only reference (HDBCA alone), it will look off-manifold — high reconstruction error — and be wrongly discarded as technical garbage. It must be mapped against a reference that **contains mesoderm and neural crest**, so it maps cleanly to mesoderm (low error, low entropy) and is correctly called off-target. We therefore build a combined reference.

```python
import scanpy as sc, scvi, numpy as np, pandas as pd
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

# Query: Human Neural Organoid Cell Atlas (organoid)
hnoca = sc.read_h5ad("HNOCA.h5ad")                      # from CELLxGENE

# Reference part A: Human Developmental Brain Cell Atlas (fine neural + maturation)
hdbca = sc.read_h5ad("HDBCA.h5ad")

# Reference part B: non-neural fetal lineages (mesoderm + neural crest) pulled
# from a broad human fetal atlas so off-target & neural-crest cells have anchors
fetal = sc.read_h5ad("fetal_broad.h5ad")
fetal = fetal[fetal.obs.lineage.isin(["mesoderm", "neural_crest", "endoderm"])]

ref = sc.concat([hdbca, fetal], join="inner", label="ref_source")

# Annotate every reference cell with developmental ORIGIN (not morphology).
# This is where the neural-crest guard is seeded: neural-crest mesenchyme is
# tagged 'neural_crest' (ectodermal), distinct from 'mesoderm'.
origin_map = {... per-celltype dict ...}              # built from atlas ontology
ref.obs["origin"] = ref.obs["cell_type"].map(origin_map)
```

---

## Step 2 — Mapping, entropy, off-manifold score, and the neural-crest-guarded classifier

Train scANVI on the combined reference, then map HNOCA in via scArches surgery. From the mapped query we extract a soft label distribution (→ entropy τ_H) and a latent-space off-manifold score (→ τ_R, computed as distance to the *k* nearest reference cells, which is robust and avoids per-cell-likelihood API ambiguity).

```python
scvi.model.SCVI.setup_anndata(ref, batch_key="batch", layer="counts")
ref_scvi   = scvi.model.SCVI(ref, n_latent=30)
ref_scvi.train()
ref_scanvi = scvi.model.SCANVI.from_scvi_model(ref_scvi, labels_key="cell_type",
                                               unlabeled_category="Unknown")
ref_scanvi.train()

q = scvi.model.SCANVI.load_query_data(hnoca, ref_scanvi)   # scArches surgery
q.train(max_epochs=100, plan_kwargs={"weight_decay": 0.0})

# (a) Mapping entropy
soft = q.predict(soft=True)                                # cells x labels
hnoca.obs["map_entropy"] = soft.apply(lambda r: entropy(r), axis=1).values
hnoca.obs["pred_label"]  = soft.idxmax(axis=1)
hnoca.obs["pred_origin"] = hnoca.obs["pred_label"].map(origin_map)

# (b) Off-manifold score = mean distance to k nearest REFERENCE cells in latent
Zref = ref_scanvi.get_latent_representation(ref)
Zq   = q.get_latent_representation(hnoca)
nn   = NearestNeighbors(n_neighbors=15).fit(Zref)
hnoca.obs["offmanifold"] = nn.kneighbors(Zq)[0].mean(axis=1)
```

Thresholds are calibrated on held-out **reference** cells (the confusion and off-manifold levels expected for genuine, well-differentiated cells), never hand-set:

```python
ref_hold = ...   # 10% of reference held out, mapped back through q
tau_H = np.percentile(ref_hold.obs["map_entropy"], 95)
tau_R = np.percentile(ref_hold.obs["offmanifold"], 99)
```

The classifier, with the neural-crest guard written explicitly:

```python
def classify(r):
    if r.offmanifold > tau_R:
        return "ambiguous_novel"                  # resembles nothing real -> set aside
    # NEURAL-CREST GUARD: neural crest is ectoderm-derived -> NOT off-target
    if r.pred_origin in ("neural", "neural_crest"):
        return "poorly_diff_ontarget" if r.map_entropy > tau_H else "clean_ontarget"
    if r.pred_origin == "mesoderm":
        return "true_offtarget" if r.map_entropy < tau_H else "ambiguous"
    return "other"

hnoca.obs["pilot_class"] = hnoca.obs.apply(classify, axis=1)
```

The single line `if r.pred_origin in ("neural", "neural_crest")` is the code-level defense against the neural-crest trap: a confidently neural-crest-derived mesenchymal cell is classed as on-target ectoderm, not counted as an off-target contaminant. Without it, every neural organoid would generate a large systematic false-off-target wave.

---

## Step 3 — Go/No-Go: is the in-vitro signature one axis or many?

For each on-target neural type present in both organoid (`clean_ontarget`) and reference, compute the dish vector Δ = mean(organoid) − mean(primary) in log-normalized HVG space, then test pairwise cosine similarity.

```python
def delta(celltype):
    org = hnoca[(hnoca.obs.pilot_class=="clean_ontarget") &
                (hnoca.obs.pred_label==celltype)]
    pri = ref[ref.obs.cell_type==celltype]
    return np.asarray(org.X.mean(0)).ravel() - np.asarray(pri.X.mean(0)).ravel()

d_neuron = delta("Neuron")
d_glia   = delta("Astrocyte")
d_prog   = delta("Radial_glia")

cos = lambda a,b: float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
mean_cos = np.mean([cos(d_neuron,d_glia), cos(d_neuron,d_prog), cos(d_glia,d_prog)])
```

**The threshold.** `mean_cos > 0.70` → **GO**: a single frozen z_iv is justified; Aim 2 proceeds as written. `mean_cos ≤ 0.70` → **NO-GO**, which is *not* fatal — it triggers the pre-specified conditional model z_iv | y_id (dish signal modulated by broad lineage class). The pilot's job is to tell us which Aim 2 we are building, not whether Aim 2 is possible.

We additionally decompose Δ to test the literature prediction directly — glycolysis shared, ER stress lineage-specific:

```python
glyco = ref.var_names.isin(HALLMARK_GLYCOLYSIS | HALLMARK_HYPOXIA)
upr   = ref.var_names.isin(HALLMARK_UNFOLDED_PROTEIN_RESPONSE)
cos_glyco = cos(d_neuron[glyco], d_glia[glyco])   # expect HIGH (shared dish signal)
cos_upr   = cos(d_neuron[upr],   d_glia[upr])     # expect LOWER (lineage-specific)
```

A high `cos_glyco` with a low `cos_upr` would confirm the mechanism and tell us the conditional model only needs to modulate the secretory-stress component — a clean, defensible design narrowing.

---

## Step 4 — The count check and the green light

Count the surviving true off-target mesenchymal cells, stratified by protocol (the Control 2 in-vitro batch baseline needs ≥2 protocols).

```python
mes = hnoca[(hnoca.obs.pilot_class=="true_offtarget") &
            (hnoca.obs.pred_label.str.contains("mesench|stroma|fibro", case=False))]
N_total = mes.n_obs
by_protocol = mes.obs.groupby("protocol").size().sort_values(ascending=False)
print(N_total); print(by_protocol)
```

**Hard decision thresholds** (anticipated N_min from rarefaction is low hundreds; these build in margin and the Control 2 protocol-split requirement):

- **GREEN — proceed to Aim 3:** `N_total ≥ 1000` AND `≥2 protocols each ≥ 300` off-target mesenchymal cells. Protocol-stratified Control 2 is feasible; the powered core exists.
- **YELLOW — proceed, scope-limited:** `300 ≤ N_total < 1000`, or only one protocol clears 300. Pooled matched-N comparisons only; Control 2 weakened; report accordingly.
- **RED — stop and diagnose:** `N_total < 300`.

**The critical RED diagnostic.** A RED count has two very different causes that the pilot must distinguish, because one is a dead end and one is a one-line fix:

```python
# Where did the candidate mesenchyme actually go?
cand = hnoca[hnoca.obs.pred_label.str.contains("mesench|stroma|fibro", case=False)]
print(cand.obs.pilot_class.value_counts())
```

- If most candidate mesenchyme landed in `ambiguous_novel` (high off-manifold) → the **reference lacked adequate mesodermal representation**; off-target cells couldn't find an anchor. Fixable by broadening the fetal reference (Step 1, part B). Not a biology dead end.
- If candidate mesenchyme is genuinely scarce in the raw data → the biology is the limit, and the proposal's premise about the "mesenchyme tax" must be revisited honestly.

Distinguishing these two is itself a valuable pilot result; it converts a scary RED into either a pipeline fix or a real, reportable finding about the data.

---

## Decision tree out of the pilot

- **Step 2 sane** (clean_ontarget dominated by neurons/glia, true_offtarget dominated by confident mesoderm, neural crest correctly spared) → definition works.
- **Step 3 GO** → build single-axis Aim 2. **Step 3 NO-GO** → build conditional z_iv | y_id Aim 2. Either way, proceed.
- **Step 4 GREEN** → the full Aim 3 geometry is powered on neural mesenchyme; commit to the project. **YELLOW** → commit with stated scope limits. **RED** → run the diagnostic; broaden reference and re-run, or revisit the premise.

The only outcome that kills the project is RED *with* the diagnostic showing genuinely scarce raw mesenchyme — and given that every neural organoid protocol reports 10–30% mesenchymal contamination, that outcome would itself be a surprising, publishable correction to the field's assumptions. There is no result of this 48-hour pilot that produces nothing.
