"""Diagnose the 38.4% ambiguous_novel (mostly neural): reference/maturation gap vs in-vitro aberrance.
Four diagnostics (D1 marker coherence, D2 maturation, D3 cross-protocol, D4 cluster coherence).
Runs on CPU; subsamples where it needs X / the embedding. Output -> results/pilot/diagnostics.log.
"""
import json
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp

ANNOT = "results/pilot/hnoca_annotated.h5ad"
HNOCA = "data/hnoca_cellxgene.h5ad"
rng = np.random.default_rng(0)

a = ad.read_h5ad(ANNOT, backed="r")
obs = a.obs.copy()
n = a.n_obs
h = ad.read_h5ad(HNOCA, backed="r")
assert list(map(str, h.obs_names[:50])) == list(map(str, obs.index[:50])), "HNOCA/annotated order mismatch"
for c in ("organoid_age_days", "assay_differentiation", "publication", "n_genes_by_counts",
          "Hallmark_Glycolysis", "annot_level_2", "annot_level_3_rev2", "cell_type_original"):
    if c in h.obs.columns:
        obs[c] = h.obs[c].values
obsm_keys = list(h.obsm.keys())
print("HNOCA obsm keys:", obsm_keys)

neural = obs["pred_origin"].isin(["neural", "neural_crest"]).to_numpy()
ambnov = (obs["pilot_class"] == "ambiguous_novel").to_numpy()
clean = (obs["pilot_class"] == "clean_ontarget").to_numpy()
ambnov_neural = ambnov & neural
clean_neural = clean & neural
om = pd.to_numeric(obs["offmanifold"], errors="coerce").to_numpy()
print("ambnov_neural=%d  clean_neural=%d" % (int(ambnov_neural.sum()), int(clean_neural.sum())))

# ===== D3 cross-protocol reproducibility (obs-only, full data) =====
print("\n===== D3 CROSS-PROTOCOL: ambiguous_novel fraction among NEURAL cells per protocol =====")
protocol_col = "assay_differentiation" if "assay_differentiation" in obs.columns else "publication"
df = obs.loc[neural, [protocol_col]].copy()
df["ambnov"] = ambnov[neural]
g = df.groupby(protocol_col)["ambnov"].agg(["mean", "size"]).query("size >= 200").sort_values("mean")
g.columns = ["ambnov_frac", "n_neural"]
print(g.round(3).to_string())
print("  -> spread: min=%.2f max=%.2f std=%.3f  (consistent=general property; concentrated=protocol-specific)"
      % (g["ambnov_frac"].min(), g["ambnov_frac"].max(), g["ambnov_frac"].std()))

# ===== D2 maturation stratification (organoid_age + n_genes potency proxy) =====
print("\n===== D2 MATURATION: off-manifold vs maturation axes =====")
if "organoid_age_days" in obs.columns:
    age = pd.to_numeric(obs["organoid_age_days"], errors="coerce")
    bins = pd.qcut(age[neural], 5, duplicates="drop")
    t = pd.DataFrame({"bin": bins, "ambnov": ambnov[neural], "om": om[neural]}).groupby("bin").agg(
        ambnov_frac=("ambnov", "mean"), median_offmanifold=("om", "median"), n=("ambnov", "size"))
    print("  by organoid_age_days quintile (low->high age):")
    print(t.round(3).to_string())
if "n_genes_by_counts" in obs.columns:
    ng = pd.to_numeric(obs["n_genes_by_counts"], errors="coerce")  # CytoTRACE-style potency proxy
    bins = pd.qcut(ng[neural], 5, duplicates="drop")
    t = pd.DataFrame({"bin": bins, "ambnov": ambnov[neural]}).groupby("bin")["ambnov"].mean()
    print("  ambnov_frac by n_genes_by_counts quintile (CytoTRACE potency proxy, low->high):")
    print("   ", t.round(3).to_dict())

# ===== D1 marker coherence (needs X; subsample 30k ambnov_neural + 30k clean_neural) =====
print("\n===== D1 MARKER COHERENCE: neural identity vs stress signatures =====")
def samp(mask, k):
    idx = np.flatnonzero(mask)
    return np.sort(rng.choice(idx, min(k, idx.size), replace=False))
i_amb, i_cln = samp(ambnov_neural, 30000), samp(clean_neural, 30000)
idx = np.sort(np.concatenate([i_amb, i_cln]))
sub = a[idx].to_memory()
sub.obs["grp"] = np.where(np.isin(idx, i_amb), "ambnov_neural", "clean_neural")
sym2ens = {}
if "feature_name" in sub.var.columns:
    for e, s in zip(sub.var_names, sub.var["feature_name"].astype(str)):
        sym2ens.setdefault(s, e)
def ens(syms):
    return [sym2ens[s] for s in syms if s in sym2ens]
panels = {
    "NPC_progenitor": ["SOX2", "NES", "PAX6", "VIM", "HES1", "HES5", "NOTCH1"],
    "Neuron": ["RBFOX3", "MAP2", "DCX", "STMN2", "SYT1", "TUBB3", "NEFL", "SNAP25"],
    "Glia_astro": ["GFAP", "AQP4", "S100B", "SLC1A3", "OLIG1", "OLIG2", "SOX10"],
    "Hypoxia": ["VEGFA", "SLC2A1", "LDHA", "PGK1", "HK2", "BNIP3", "CA9", "ALDOA", "PDK1"],
    "UPR_ERstress": ["HSPA5", "DDIT3", "ATF4", "XBP1", "ERN1", "ATF6", "HERPUD1", "DNAJB9"],
    "Glycolysis": ["PKM", "LDHA", "ENO1", "GAPDH", "ALDOA", "PGK1", "TPI1", "PFKL"],
}
scored = []
for name, syms in panels.items():
    g = ens(syms)
    if len(g) >= 3:
        sc.tl.score_genes(sub, g, score_name=name, use_raw=False)
        scored.append(name)
print("  scored panels:", scored)
print(sub.obs.groupby("grp")[scored].mean().round(3).to_string())
if "Hallmark_Glycolysis" in obs.columns:
    hg = pd.to_numeric(obs["Hallmark_Glycolysis"], errors="coerce")
    print("  HNOCA Hallmark_Glycolysis (precomputed) mean: ambnov=%.3f clean=%.3f"
          % (np.nanmean(hg[ambnov_neural]), np.nanmean(hg[clean_neural])))

# ===== D4 cluster coherence (Leiden on HNOCA embedding; per-cluster ambnov fraction) =====
print("\n===== D4 CLUSTER COHERENCE: do ambnov_neural cells form coherent clusters? =====")
emb_key = next((k for k in ("X_scpoli", "X_scANVI", "X_pca", "X_umap") if k in obsm_keys), None)
if emb_key is None and obsm_keys:
    emb_key = obsm_keys[0]
if emb_key:
    both = np.flatnonzero(neural & (ambnov | clean))
    s = np.sort(rng.choice(both, min(60000, both.size), replace=False))
    E = np.asarray(h.obsm[emb_key][s])
    cl = ad.AnnData(np.zeros((len(s), 1), dtype="float32"))
    cl.obsm["X_emb"] = E
    cl.obs["ambnov"] = ambnov[s].astype(int)
    sc.pp.neighbors(cl, use_rep="X_emb", n_neighbors=15)
    sc.tl.leiden(cl, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)
    per = cl.obs.groupby("leiden")["ambnov"].agg(["mean", "size"])
    per.columns = ["ambnov_frac", "n"]
    per = per.sort_values("ambnov_frac")
    overall = float(cl.obs["ambnov"].mean())
    pure_hi = per.query("ambnov_frac >= 0.8")["n"].sum()
    pure_lo = per.query("ambnov_frac <= 0.2")["n"].sum()
    print("  emb=%s  n_clusters=%d  overall ambnov_frac=%.2f" % (emb_key, len(per), overall))
    print("  cluster ambnov_frac spread: min=%.2f max=%.2f std=%.3f" %
          (per["ambnov_frac"].min(), per["ambnov_frac"].max(), per["ambnov_frac"].std()))
    print("  cells in ~pure-ambnov clusters(>=0.8): %d (%.0f%%); ~pure-clean(<=0.2): %d (%.0f%%)"
          % (pure_hi, 100*pure_hi/len(s), pure_lo, 100*pure_lo/len(s)))
    print("  -> high std / large pure fractions = COHERENT population; uniform ~overall = SCATTER")
else:
    print("  no usable embedding in HNOCA obsm:", obsm_keys)

print("\n=== DIAGNOSTICS DONE ===")
