"""
Sweep harness. Runs the grid and writes a tidy results table (one row per
(rho, delta, method, seed, n)). Keep grids small while developing; scale up once
Panel A is clean (see README, Gate G2).

Three sweeps you will actually run:
  sweep_rho   : the headline -- recovery & certificate vs support overlap.
  sweep_grid  : the rho x delta dissociation (recovery collapses along rho only).
  sweep_n     : structural vs practical -- fix rho below threshold, vary n.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from dgp import DGPConfig, generate
from models import ModelConfig, train_model
from metrics import (raw_certificate, evaluate_embedding)


def _one_run(rho, delta, method, seed, n, base_dgp, base_model) -> dict:
    dcfg = DGPConfig(**{**base_dgp, "rho": rho, "delta": delta,
                        "n_per_domain": n, "seed": seed})
    data = generate(dcfg)
    X, t, s, mask = data["X"], data["t"], data["s"], data["overlap_region"]

    # certificate is model-free -> compute once per dataset
    cert = raw_certificate(X, s)

    mcfg = ModelConfig(**{**base_model, "mode": method, "seed": seed})
    model, _ = train_model(X, s, mcfg)
    z = model.embed(model_input(X, model))          # shared latent
    m = evaluate_embedding(z, t, s, mask=mask)

    return dict(rho=rho, delta=delta, method=method, seed=seed, n=n,
                overlap=data["overlap"],
                cert_max=cert["max"], cert_spread=cert["spread"],
                **{f"cert_{kk}": vv for kk, vv in cert.items()
                   if kk.startswith("cap_")},
                **m)


def model_input(X, model):
    """Recompute the exact preprocessing the model was trained on."""
    from models import _prep
    import torch
    Xn = _prep(X)
    return torch.tensor(Xn, device=next(model.parameters()).device)


def run_sweep(rhos, deltas, methods, seeds, ns, base_dgp=None, base_model=None,
              out_csv="results.csv", verbose=True) -> pd.DataFrame:
    base_dgp = base_dgp or {}
    base_model = base_model or {}
    rows = []
    combos = list(itertools.product(rhos, deltas, methods, seeds, ns))
    for i, (rho, delta, method, seed, n) in enumerate(combos, 1):
        if verbose:
            print(f"[{i}/{len(combos)}] rho={rho:.2f} delta={delta:.2f} "
                  f"method={method} seed={seed} n={n}", flush=True)
        rows.append(_one_run(rho, delta, method, seed, n, base_dgp, base_model))
        pd.DataFrame(rows).to_csv(out_csv, index=False)     # checkpoint each run
    return pd.DataFrame(rows)


# --- three canonical sweeps ------------------------------------------------- #
def sweep_rho(**kw):
    return run_sweep(
        rhos=np.round(np.linspace(1.0, 0.0, 11), 2),
        deltas=[kw.pop("delta", 1.0)],
        methods=kw.pop("methods", ["vanilla", "conditional", "contrastive", "adversarial"]),
        seeds=kw.pop("seeds", [0, 1, 2]),
        ns=[kw.pop("n", 3000)],
        out_csv=kw.pop("out_csv", "results_rho.csv"), **kw)


def sweep_grid(**kw):
    return run_sweep(
        rhos=kw.pop("rhos", np.round(np.linspace(1.0, 0.0, 6), 2)),
        deltas=kw.pop("deltas", [0.0, 0.5, 1.0, 2.0, 4.0]),
        methods=kw.pop("methods", ["conditional"]),
        seeds=kw.pop("seeds", [0, 1]),
        ns=[kw.pop("n", 3000)],
        out_csv=kw.pop("out_csv", "results_grid.csv"), **kw)


def sweep_n(**kw):
    return run_sweep(
        rhos=kw.pop("rhos", [0.15, 0.35]),         # one below, one just-above threshold
        deltas=[kw.pop("delta", 1.0)],
        methods=kw.pop("methods", ["conditional"]),
        seeds=kw.pop("seeds", [0, 1, 2]),
        ns=kw.pop("ns", [500, 1000, 2000, 4000, 8000]),
        out_csv=kw.pop("out_csv", "results_n.csv"), **kw)


def sweep_frontier(rhos=(0.15, 0.35), adv_lambdas=(0.0, 0.3, 1.0, 3.0, 10.0),
                   seeds=(0, 1), n=3000, delta=1.0,
                   base_dgp=None, base_model=None, out_csv="results_frontier.csv",
                   verbose=True):
    """THE HEADLINE sweep. For a below-threshold and an above-threshold rho, trace
    the achievable (batch_removed, bio_recovery) frontier by sweeping the
    domain-removal strength (adv_lambda) of the adversarial model. Above rho* the
    frontier reaches the top-right corner (both high); below rho* the top-right is
    unreachable -- you can buy domain removal only by destroying biology. That
    unreachability *is* the identifiability limit."""
    base_dgp = base_dgp or {}; base_model = base_model or {}
    rows = []
    for rho in rhos:
        for seed in seeds:
            dcfg = DGPConfig(**{**base_dgp, "rho": rho, "delta": delta,
                                "n_per_domain": n, "seed": seed})
            data = generate(dcfg)
            X, t, s, mask = data["X"], data["t"], data["s"], data["overlap_region"]
            for lam in adv_lambdas:
                if verbose:
                    print(f"rho={rho:.2f} lambda={lam:.2f} seed={seed}", flush=True)
                mcfg = ModelConfig(**{**base_model, "mode": "adversarial",
                                      "adv_lambda": lam, "seed": seed})
                model, _ = train_model(X, s, mcfg)
                z = model.embed(model_input(X, model))
                m = evaluate_embedding(z, t, s, mask=mask)
                rows.append(dict(rho=rho, adv_lambda=lam, seed=seed,
                                 overlap=data["overlap"], **m))
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    # scalar summary: best achievable balance = max over lambda of min(bio, batch)
    df["joint"] = df[["bio_recovery", "batch_removed"]].min(axis=1)
    return df


if __name__ == "__main__":
    # tiny demo run (fast). Real runs: call sweep_rho()/sweep_grid()/sweep_n().
    df = run_sweep(rhos=[1.0, 0.5, 0.0], deltas=[1.0],
                   methods=["conditional", "contrastive"], seeds=[0], ns=[800],
                   base_model={"epochs": 60}, out_csv="results_demo.csv")
    cols = ["rho", "method", "bio_recovery", "batch_removed", "cert_max", "off_manifold_latent"]
    print(df[cols].to_string(index=False))
