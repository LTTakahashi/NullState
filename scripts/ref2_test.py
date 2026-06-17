#!/usr/bin/env python
"""Second-reference test (reviewer point 3) -- v2, JOINT-INTEGRATION design.

Question: is the ambiguous-novel NEURAL organoid compartment a REFERENCE-COVERAGE GAP
(cells that co-embed with an independent developing-cortex atlas HDBCA lacked) or GENUINE
IN-VITRO DIVERGENCE (cells off-manifold against every fetal reference)?

v1 trained scVI on the reference alone and scArches-projected the organoid; that left even
the clean-on-target positive control ~96% off-manifold, i.e. the projection failed to bridge
the organoid<->primary technical gap and the metric was confounded. v2 instead JOINTLY
integrates organoid + the PRENATAL Velmeshev 2023 cortex with scVI (batch_key=dataset), the
standard way to map a query onto a reference, and validates with the clean control: if clean
fetal organoid neural cells co-embed with prenatal cortex (on-manifold) while ambiguous-novel
cells do not, the divergence signal is real. The clean-vs-ambnov contrast is the readout.

    PYTHONPATH=. python scripts/ref2_test.py
"""
import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd


def strip_ver(ids):
    return pd.Index([str(g).split(".")[0] for g in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", default="1a38e762-2465-418f-b81c-6a4bce261c34")  # Velmeshev 2023
    ap.add_argument("--census-version", default="2025-11-08")
    ap.add_argument("--n-ref", type=int, default=100000)
    ap.add_argument("--n-ambnov", type=int, default=30000)
    ap.add_argument("--n-clean", type=int, default=15000)
    ap.add_argument("--n-hvg", type=int, default=3000)
    ap.add_argument("--n-latent", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--max-count-per-cell", type=int, default=1000)
    ap.add_argument("--config-dir", default="config/")
    ap.add_argument("--out", default="results/pilot/ref2_test.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    import yaml, anndata as ad, scanpy as sc, scvi, cellxgene_census
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    scvi.settings.seed = 0
    paths = yaml.safe_load(open(Path(args.config_dir) / "paths.yaml"))
    out_dir = Path("results/pilot"); out_dir.mkdir(parents=True, exist_ok=True)
    annot_path = out_dir / "hnoca_annotated.h5ad"
    cellx_path = Path(paths["datasets"]["hnoca"]["local_path"])
    rng = np.random.default_rng(0)

    # ---------------- 1. independent 2nd reference: PRENATAL Velmeshev 2023 ----------------
    logging.info(f"Opening Census {args.census_version}; sampling PRENATAL dataset {args.dataset_id}")
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        obs_df = cellxgene_census.get_obs(
            census, "Homo sapiens",
            value_filter=f"dataset_id == '{args.dataset_id}'",
            column_names=["soma_joinid", "development_stage", "cell_type"],
        )
        prenatal = obs_df["development_stage"].str.contains("LMP month", case=False, na=False)
        logging.info(f"dataset {len(obs_df)} cells; prenatal (LMP month) = {int(prenatal.sum())}")
        joinids = obs_df.loc[prenatal, "soma_joinid"].to_numpy()
        if joinids.size > args.n_ref:
            joinids = np.sort(rng.choice(joinids, args.n_ref, replace=False))
        ref = cellxgene_census.get_anndata(
            census, "Homo sapiens", obs_coords=joinids.tolist(),
            column_names={"obs": ["soma_joinid", "development_stage", "cell_type"]},
        )
    if "feature_id" in ref.var.columns:
        ref.var_names = ref.var["feature_id"].astype(str)
    ref.var_names = strip_ver(ref.var_names); ref.var_names_make_unique()
    ref.obs = pd.DataFrame({"dataset": "velmeshev", "grp": "ref2"}, index=ref.obs_names.astype(str))
    logging.info(f"prenatal 2nd ref {ref.shape}")

    # ---------------- 2. organoid query: ambnov-neural + clean-neural ---------------------
    a = ad.read_h5ad(annot_path, backed="r"); obs = a.obs
    neural = obs["pred_origin"].isin(["neural", "neural_crest"]).to_numpy()
    amb = (obs["pilot_class"] == "ambiguous_novel").to_numpy()
    cln = (obs["pilot_class"] == "clean_ontarget").to_numpy()
    amb_sel = np.sort(rng.choice(np.flatnonzero(neural & amb), min(args.n_ambnov, int((neural & amb).sum())), replace=False))
    cln_sel = np.sort(rng.choice(np.flatnonzero(neural & cln), min(args.n_clean, int((neural & cln).sum())), replace=False))
    q_pos = np.union1d(amb_sel, cln_sel)
    grp = np.where(np.isin(q_pos, amb_sel), "ambnov_neural", "clean_neural")
    logging.info(f"query: ambnov_neural={amb_sel.size}, clean_neural={cln_sel.size}; loading counts")
    _o = ad.read_h5ad(cellx_path, backed="r"); q = _o[q_pos].to_memory()
    try: _o.file.close()
    except Exception: pass
    if q.raw is not None:
        rn, vn = list(map(str, q.raw.var_names)), list(map(str, q.var_names))
        q.X = q.raw.X.copy() if rn == vn else q.raw[:, q.var_names].X.copy()
        del q.raw
    q.var_names = strip_ver(q.var_names); q.var_names_make_unique()
    mx = (q.X.max(axis=1).toarray().ravel() if sp.issparse(q.X) else np.asarray(q.X).max(1))
    keep = mx <= args.max_count_per_cell
    logging.info(f"dropping {int((~keep).sum())} non-UMI query cells (max>{args.max_count_per_cell})")
    q = q[keep].copy(); grp = grp[keep]
    q.obs = pd.DataFrame({"dataset": "organoid", "grp": grp}, index=q.obs_names.astype(str))

    # ---------------- 3. JOINT scVI integration (batch_key=dataset) -----------------------
    common = ref.var_names.intersection(q.var_names)
    logging.info(f"common Ensembl genes: {len(common)}")
    comb = ad.concat([ref[:, common], q[:, common]], join="inner", label="src", keys=["ref2", "organoid"])
    comb.obs_names_make_unique()
    sc.pp.highly_variable_genes(comb, n_top_genes=args.n_hvg, flavor="seurat_v3",
                                batch_key="dataset", subset=True)
    logging.info(f"joint HVG -> {comb.n_vars}; scVI ({args.epochs} ep, batch=dataset) on {comb.n_obs} cells")
    scvi.model.SCVI.setup_anndata(comb, batch_key="dataset")
    model = scvi.model.SCVI(comb, n_latent=args.n_latent, n_layers=2, gene_likelihood="nb")
    model.train(max_epochs=args.epochs, early_stopping=True, early_stopping_patience=8,
                check_val_every_n_epoch=1)
    Z = model.get_latent_representation()
    is_ref = (comb.obs["dataset"] == "velmeshev").to_numpy()
    g = comb.obs["grp"].to_numpy()
    z_ref, z_amb, z_cln = Z[is_ref], Z[g == "ambnov_neural"], Z[g == "clean_neural"]

    # ---------------- 4. off-manifold vs the integrated 2nd reference ---------------------
    k = 15
    n = z_ref.shape[0]; perm = rng.permutation(n); cal = perm[: n // 10]; man = perm[n // 10:]
    nn = NearestNeighbors(n_neighbors=k).fit(z_ref[man])
    ref_self = nn.kneighbors(z_ref[cal])[0].mean(1)
    tau_R = float(np.quantile(ref_self, 0.95))
    amb_off = nn.kneighbors(z_amb)[0].mean(1)
    cln_off = nn.kneighbors(z_cln)[0].mean(1)

    res = {
        "second_reference": "Velmeshev et al. 2023 (PRENATAL human cortex, LMP months)",
        "dataset_id": args.dataset_id, "design": "joint scVI integration (batch=dataset)",
        "n_ref_prenatal_trained": int(is_ref.sum()), "n_common_genes": int(len(common)),
        "n_hvg": int(comb.n_vars), "tau_R_ref2": tau_R, "k": k,
        "ref2_self_onmanifold_frac": float(np.mean(ref_self <= tau_R)),
        "clean_neural": {"n": int(z_cln.shape[0]), "on_manifold_frac": float(np.mean(cln_off <= tau_R)),
                         "median_offscore": float(np.median(cln_off))},
        "ambnov_neural": {"n": int(z_amb.shape[0]), "on_manifold_frac": float(np.mean(amb_off <= tau_R)),
                          "median_offscore": float(np.median(amb_off))},
        "ref2_self_median_offscore": float(np.median(ref_self)),
    }
    cl_on, amb_on = res["clean_neural"]["on_manifold_frac"], res["ambnov_neural"]["on_manifold_frac"]
    res["positive_control_ok"] = bool(cl_on >= 0.5)
    res["interpretation"] = (
        f"POSITIVE CONTROL clean-neural on-manifold = {cl_on*100:.0f}% "
        f"({'VALID' if cl_on >= 0.5 else 'STILL CONFOUNDED -- integration did not co-embed clean cells'}). "
        f"ambiguous-novel on-manifold = {amb_on*100:.0f}%. "
        f"Reading the clean baseline as the recoverable ceiling, the ambiguous-novel coverage-gap "
        f"fraction is ~{amb_on*100:.0f}% and the genuine-divergence fraction ~{(1-amb_on)*100:.0f}%; "
        f"relative to clean, ambiguous-novel cells are {(1-amb_on)/(max(1-cl_on,1e-6)):.1f}x more off-manifold.")
    json.dump(res, open(args.out, "w"), indent=2)
    np.savez(out_dir / "ref2_scores.npz", ref_self=ref_self, amb_off=amb_off, cln_off=cln_off, tau_R=tau_R)
    logging.info("RESULT:\n" + json.dumps(res, indent=2))
    logging.info(f"wrote {args.out} + ref2_scores.npz")


if __name__ == "__main__":
    main()
