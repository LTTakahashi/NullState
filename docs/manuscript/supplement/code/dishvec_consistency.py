#!/usr/bin/env python
"""Per-lineage dish-vector consistency check (reviewer point 3).

Aim 3's offset cancellation assumes the organoid-versus-primary source offset is the SAME
vector across the off-target populations compared. Gate B verified offset-consistency on
on-target NEURAL cells (cosine 0.855); this checks the assumption that actually matters --
the OFF-TARGET NON-NEURAL populations. For each off-target lineage we estimate the dish
vector = mean(organoid) - mean(primary) in log-normalized HVG space, then report the mutual
cosine across lineages. High mutual cosine => offset ~constant => cancellation valid; low =>
the recovered Aim-3 distances are an upper bound (partly differential-stress distance).

    PYTHONPATH=. python scripts/dishvec_consistency.py
"""
import json, logging
from pathlib import Path
import numpy as np
import pandas as pd

# organoid pred_label  ->  matching primary cell_type label(s) in reference.h5ad
# (FILLED from scripts/inspect_ct.py output)
LINEAGE_MAP = {
    "cell of skeletal muscle":        ["cell of skeletal muscle"],
    "cardiac muscle cell":            ["cardiac muscle cell"],
    "fibroblast":                     ["fibroblast"],
    "cortical cell of adrenal gland": ["cortical cell of adrenal gland"],
}


def strip_ver(ids):
    return pd.Index([str(g).split(".")[0] for g in ids])


def main():
    import anndata as ad, scanpy as sc
    import scipy.sparse as sp
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ROOT = Path("/workspace/NullState"); RES = ROOT / "results/pilot"
    n_per, n_hvg = 2500, 2000
    rng = np.random.default_rng(0)

    # --- 1. organoid off-target cells per lineage (labels from annotated, counts from CELLxGENE) ---
    a = ad.read_h5ad(RES / "hnoca_annotated.h5ad", backed="r"); obs = a.obs
    off = (obs["pilot_class"] == "true_offtarget").to_numpy()
    pred = obs["pred_label"].astype(str).to_numpy()
    org_pos = {}
    for lin in LINEAGE_MAP:
        idx = np.flatnonzero(off & (pred == lin))
        if idx.size > n_per:
            idx = np.sort(rng.choice(idx, n_per, replace=False))
        org_pos[lin] = idx
        logging.info(f"organoid {lin}: {idx.size}")
    all_org = np.sort(np.concatenate([org_pos[l] for l in LINEAGE_MAP]))
    lin_of = {p: l for l in LINEAGE_MAP for p in org_pos[l]}
    o = ad.read_h5ad(ROOT / "data/hnoca_cellxgene.h5ad", backed="r")
    org = o[all_org].to_memory()
    try: o.file.close()
    except Exception: pass
    if org.raw is not None:
        rn, vn = list(map(str, org.raw.var_names)), list(map(str, org.var_names))
        org.X = org.raw.X.copy() if rn == vn else org.raw[:, org.var_names].X.copy()
        del org.raw
    org.var_names = strip_ver(org.var_names); org.var_names_make_unique()
    org.obs = pd.DataFrame({"lin": [lin_of[p] for p in all_org], "source": "organoid"},
                           index=org.obs_names.astype(str))

    # --- 2. matched primary cells per lineage (raw counts in X) ---
    r = ad.read_h5ad(RES / "reference.h5ad", backed="r")
    rct = r.obs["cell_type"].astype(str).to_numpy()
    parts = []
    for lin, cts in LINEAGE_MAP.items():
        idx = np.flatnonzero(np.isin(rct, cts))
        if idx.size == 0:
            raise SystemExit(f"no primary cells for {lin} ({cts})")
        if idx.size > n_per:
            idx = np.sort(rng.choice(idx, n_per, replace=False))
        part = r[idx].to_memory(); part.obs = pd.DataFrame({"lin": lin, "source": "primary"},
                                                           index=part.obs_names.astype(str))
        parts.append(part); logging.info(f"primary {lin}: {idx.size}")
    try: r.file.close()
    except Exception: pass
    prim = ad.concat(parts); prim.var_names = strip_ver(prim.var_names); prim.var_names_make_unique()

    # --- 3. shared genes, HVG on combined raw, log-normalize ---
    common = org.var_names.intersection(prim.var_names)
    logging.info(f"shared genes: {len(common)}")
    comb = ad.concat([org[:, common], prim[:, common]], join="inner")
    comb.obs_names_make_unique()
    sc.pp.highly_variable_genes(comb, n_top_genes=n_hvg, flavor="seurat_v3", subset=True)
    sc.pp.normalize_total(comb, target_sum=1e4); sc.pp.log1p(comb)

    # --- 4. dish vectors + the decisive geometry-preservation test ---
    X = comb.X.toarray() if sp.issparse(comb.X) else np.asarray(comb.X)
    src = comb.obs["source"].to_numpy(); lab = comb.obs["lin"].to_numpy()
    lins = list(LINEAGE_MAP)
    org_c = {l: X[(src == "organoid") & (lab == l)].mean(0) for l in lins}   # organoid centroids
    prim_c = {l: X[(src == "primary") & (lab == l)].mean(0) for l in lins}   # primary centroids
    raw_off = {l: org_c[l] - prim_c[l] for l in lins}                        # per-lineage source offset

    def cos(u, v): return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    # (a) reviewer's literal check: are the per-lineage offsets mutually consistent?
    M = np.array([[cos(raw_off[a_], raw_off[b_]) for b_ in lins] for a_ in lins])
    offdiag = [M[i, j] for i in range(len(lins)) for j in range(i + 1, len(lins))]
    # (b) DECISIVE: does organoid inter-lineage geometry recover PRIMARY inter-lineage geometry?
    #     cos( org_i-org_j , prim_i-prim_j ) -- if ~1, inconsistent offsets do NOT distort geometry.
    geom_cos, dist_org, dist_prim, contam = [], [], [], []
    for i in range(len(lins)):
        for j in range(i + 1, len(lins)):
            vo = org_c[lins[i]] - org_c[lins[j]]; vp = prim_c[lins[i]] - prim_c[lins[j]]
            geom_cos.append(cos(vo, vp))
            dist_org.append(float(np.linalg.norm(vo))); dist_prim.append(float(np.linalg.norm(vp)))
            contam.append(float(np.linalg.norm(raw_off[lins[i]] - raw_off[lins[j]])))
    # distance-matrix correlation (organoid vs primary pairwise distances)
    dmat_corr = float(np.corrcoef(dist_org, dist_prim)[0, 1])
    res = {
        "lineages": lins, "n_hvg": int(comb.n_vars), "n_shared_genes": int(len(common)),
        "offset_cosine_matrix": [[round(v, 3) for v in row] for row in M.tolist()],
        "mean_offset_cosine": round(float(np.mean(offdiag)), 3),
        "gate_b_neural_cosine": 0.855,
        "geometry_preservation_cos_mean": round(float(np.mean(geom_cos)), 3),
        "geometry_preservation_cos_min": round(float(np.min(geom_cos)), 3),
        "organoid_vs_primary_distmatrix_corr": round(dmat_corr, 3),
        "mean_contamination_over_lineagedist": round(float(np.mean(np.array(contam) / np.array(dist_prim))), 3),
        "n_per_lineage_organoid": {l: int(org_pos[l].size) for l in LINEAGE_MAP},
    }
    res["interpretation"] = (
        f"Per-lineage source offsets are directionally INconsistent (mean cosine {res['mean_offset_cosine']} "
        f"vs 0.855 on-target neural). The decisive test is whether that distorts geometry: organoid "
        f"inter-lineage vectors align with PRIMARY inter-lineage vectors at mean cosine "
        f"{res['geometry_preservation_cos_mean']} (min {res['geometry_preservation_cos_min']}), and the "
        f"organoid vs primary pairwise-distance matrices correlate r={res['organoid_vs_primary_distmatrix_corr']}. "
        f"{'Geometry is preserved despite inconsistent offsets -> Aim 3 ordering is genuine (offsets small vs lineage separation).' if res['geometry_preservation_cos_mean'] >= 0.7 else 'Offsets materially distort the geometry -> Aim 3 distances are confounded.'}")
    json.dump(res, open(RES / "dishvec_consistency.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
