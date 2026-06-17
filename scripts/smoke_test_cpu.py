#!/usr/bin/env python
"""CPU smoke-test for the WS1 -> WS2 path (Phase 1) -- validate the code before any GPU spend.

Runs the FULL contrastiveVI -> z_lin -> Aim 3 geometry path on a tiny SYNTHETIC dataset, on CPU,
in a couple of minutes. The point is to shake out the parts the assistant could NOT execute --
chiefly the scvi-tools ContrastiveVI API (setup_anndata / constructor / train(background_indices,
target_indices, ...) / get_latent_representation(representation_kind=...)) and the wiring through
`src.disentangle` + `src.geometry`. It is NOT a scientific run.

Self-diagnosing: it tries the real-run training kwargs (early_stopping=True) first; if that raises,
it reports the exact error and retries without early-stopping, so you learn the correct API cheaply.

Requirements (CPU is fine):
    pip install "scvi-tools>=1.0" torch --index-url https://download.pytorch.org/whl/cpu
    pip install anndata scanpy scikit-learn
Run:
    PYTHONPATH=. python scripts/smoke_test_cpu.py
Optional (use a real subsample instead of synthetic):
    PYTHONPATH=. python scripts/smoke_test_cpu.py --adata my_small_combined.h5ad \
        --source-key cs_source --ontarget-key cs_ontarget --pop-key population
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


def _synthetic_combined(n_genes=200, seed=0):
    """Tiny synthetic AnnData with raw counts and a known primary/organoid + on/off-target design.

    Lineages A/B share a gene-module structure; organoid cells carry an extra 'dish' signal
    (present only in organoid) that contrastiveVI should route into the salient (z_iv) space,
    leaving lineage in the shared (z_lin) space. Off-target organoid cells form 2 populations
    for the geometry step.
    """
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    g = n_genes
    # gene-module mean-expression programs (log space)
    base = rng.normal(1.0, 0.3, g)
    lin_A = base + np.r_[rng.normal(1.5, 0.2, g // 4), np.zeros(g - g // 4)]
    lin_B = base + np.r_[np.zeros(g // 4), rng.normal(1.5, 0.2, g // 4), np.zeros(g - g // 2)]
    lin_C = base + np.r_[np.zeros(g // 2), rng.normal(1.5, 0.2, g // 4), np.zeros(g - 3 * (g // 4))]
    dish = np.r_[np.zeros(3 * (g // 4)), rng.normal(1.2, 0.2, g - 3 * (g // 4))]  # organoid-only

    def counts(mean_log, n):
        lam = np.exp(mean_log)[None, :] * rng.uniform(0.5, 1.5, (n, 1))
        return rng.poisson(lam).astype(np.float32)

    blocks, obs = [], []
    def add(mat, source, ontarget, pop):
        blocks.append(mat)
        obs.extend([(source, ontarget, pop)] * mat.shape[0])

    # primary on-target (background): lineages A,B, NO dish
    add(counts(lin_A, 400), "primary", True, "primary_A")
    add(counts(lin_B, 400), "primary", True, "primary_B")
    # organoid on-target (target): lineages A,B + dish
    add(counts(lin_A + dish, 400), "organoid", True, "org_onA")
    add(counts(lin_B + dish, 400), "organoid", True, "org_onB")
    # organoid OFF-target: lineage C + dish, two populations for geometry
    add(counts(lin_C + dish, 300), "organoid", False, "offtarget_1")
    add(counts(lin_C + dish + rng.normal(0, 0.1, g), 300), "organoid", False, "offtarget_2")

    X = np.vstack(blocks)
    obs = pd.DataFrame(obs, columns=["cs_source", "cs_ontarget", "population"])
    a = ad.AnnData(X=X.copy(), obs=obs)
    a.layers["counts"] = X.copy()
    a.var_names = [f"g{i}" for i in range(g)]
    return a


def main():
    ap = argparse.ArgumentParser(description="CPU smoke-test for WS1->WS2")
    ap.add_argument("--adata", help="optional real (small) combined .h5ad; else synthetic")
    ap.add_argument("--source-key", default="cs_source")
    ap.add_argument("--ontarget-key", default="cs_ontarget")
    ap.add_argument("--pop-key", default="population")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    import anndata as ad
    from src.disentangle.contrastive import (
        select_contrastive_indices, train_contrastive_vi, get_lineage_latent, verify_disentanglement,
    )
    from src.geometry.pipeline import run_geometry_over_populations

    print("=== NullState CPU smoke-test (WS1 -> WS2) ===")
    if args.adata:
        adata = ad.read_h5ad(args.adata)
        print(f"[data] loaded {args.adata}: {adata.shape}")
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
    else:
        adata = _synthetic_combined()
        print(f"[data] synthetic combined: {adata.shape} "
              f"({(adata.obs[args.source_key]=='primary').sum()} primary / "
              f"{(adata.obs[args.source_key]=='organoid').sum()} organoid)")

    is_primary = (adata.obs[args.source_key] == "primary").to_numpy()
    is_organoid = (adata.obs[args.source_key] == "organoid").to_numpy()
    is_ontarget = adata.obs[args.ontarget_key].astype(bool).to_numpy()
    bg, tg = select_contrastive_indices(is_primary, is_organoid, is_ontarget)
    print(f"[contrastive] background(primary on-target)={len(bg)}  target(organoid on-target)={len(tg)}")

    # --- train contrastiveVI; self-diagnose the early_stopping kwarg ---
    model = None
    for es in (True, False):
        try:
            print(f"[train] ContrastiveVI max_epochs={args.epochs}, early_stopping={es} ...")
            model = train_contrastive_vi(adata, bg, tg, n_background_latent=10, n_salient_latent=5,
                                         layer="counts", max_epochs=args.epochs, early_stopping=es)
            print(f"[train] OK with early_stopping={es}"
                  + ("" if es else "  <-- NOTE: early_stopping=True FAILED; adjust train_contrastive_vi"))
            break
        except Exception as e:  # noqa
            print(f"[train] early_stopping={es} raised {type(e).__name__}: {e}")
            if es:
                print("       (retrying without early_stopping to isolate the API issue...)")
            else:
                traceback.print_exc()
                raise SystemExit("contrastiveVI training failed on CPU smoke data -- fix before GPU.")

    z_lin = np.asarray(get_lineage_latent(model, adata))
    print(f"[z_lin] background/shared latent: {z_lin.shape}")
    assert z_lin.shape[0] == adata.n_obs, "z_lin row count != n_obs"

    rep = verify_disentanglement(z_lin, is_organoid)
    print(f"[verify] adversary balanced-acc predicting organoid-vs-primary from z_lin = "
          f"{rep['adversary_balanced_accuracy_mean']:.3f} "
          f"(~0.5 good; disentangled_heuristic={rep['disentangled_heuristic']})")

    # --- WS2 geometry on the OFF-target populations' z_lin ---
    off = ~is_ontarget & is_organoid
    z_off = z_lin[off]
    labels_off = adata.obs[args.pop_key].to_numpy()[off]
    uniq = np.unique(labels_off)
    print(f"[geometry] off-target populations: {dict(zip(*np.unique(labels_off, return_counts=True)))}")
    if len(uniq) >= 2:
        res = run_geometry_over_populations(
            z_off, labels_off, ladder=[20, 40, 80], n_repeats=20, n_boot=80,
            n_projections=40, n_rarefaction_repeats=10, random_state=0,
        )
        print(f"[geometry] N_min={res['n_min']}")
        for pair, r in res["pairs"].items():
            print(f"           {pair[0]} vs {pair[1]} -> {r['verdict']} (N*={r['n_star']})")
    else:
        print("[geometry] <2 off-target populations; skipped (synthetic provides 2).")

    print("\n=== SMOKE TEST PASSED: WS1->WS2 code path runs end-to-end on CPU ===")


if __name__ == "__main__":
    main()
