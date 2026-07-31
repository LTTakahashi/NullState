"""
DOES THE iVAE VARIABILITY CONDITION BITE?  -- measured with the floor instrument.

The Stage-4 arm contrast was inconclusive (ivae_5env floor 0.079 vs conditional
0.097, d=+0.23, p=0.43) AND structurally confounded: in dgp2 the environment
varies only the UNSHIFTED axis (across-environment spread of the mean: 1.63 on
axis 1, 0.066 on axis 0), so the condition had purchase on half the latent. A
condition satisfiable on paper can be weak on one coordinate.

This removes both problems:
  * every latent axis gets environment-dependent location AND scale, so the
    condition applies to the whole latent;
  * no support shift at all (rho = 1, both domains identical), so the only thing
    under test is axis-level identifiability;
  * n_env is swept ACROSS the theoretical boundary. Khemakhem et al. (2020)
    Thm 1 assumption (iv) needs nk+1 distinct auxiliary values; for a Gaussian
    location-scale prior (k=2) at latent dim n=2 that is 2n+1 = 5. So
    n_env in {2, 3} cannot satisfy it, n_env = 5 exactly meets it, n_env = 9
    over-satisfies.

Readout: the identifiability floor gap = CCA - MCC, with implied frame angle
phi = arccos(1 - gap). If the condition bites, the floor should drop as n_env
crosses 5. If it does not, that is a negative result about a widely invoked
theorem, measured directly rather than inferred from downstream accuracy.
"""
from __future__ import annotations
import os, sys, time, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from dgp2 import DGP2Config, _cached_reference, FIXED_STRUCTURE_SEED
from models2 import Model2Config, train_model2
from metrics2 import cca_score, mcc

N_ENVS = [2, 3, 5, 9]
SEEDS = [int(x) for x in os.environ.get("VC_SEEDS", ",".join(map(str, range(12)))).split(",")]
N, EPOCHS = 1500, 150
OUT = os.environ.get("VC_OUT", "results_variability_condition.csv")


def env_params_all_axes(n_env, k):
    """Environment-dependent location AND scale for EVERY latent axis."""
    rng = np.random.default_rng(FIXED_STRUCTURE_SEED + 500 + n_env)
    mu = rng.normal(0.0, 0.9, size=(n_env, k))
    sd = rng.uniform(0.4, 1.2, size=(n_env, k))
    pm = mu.mean(0)
    pv = (sd ** 2 + mu ** 2).mean(0) - pm ** 2
    return mu, sd, pm, np.sqrt(pv)


def generate_fullenv(n_env, n_per_domain, seed, n_genes=500, k=2):
    """dgp2's observation model, but every axis is environment-structured and
    there is no support shift. Reference-grid standardisation is recomputed for
    the matching prior so per-gene statistics stay comparable across n_env."""
    cfg = DGP2Config(n_genes=n_genes, latent_dim=k, n_env=n_env)
    g, _, _, direction = _cached_reference(cfg)
    mu, sd, pm, ps = env_params_all_axes(n_env, k)

    # fixed reference grid on THIS prior
    rg = np.random.default_rng(FIXED_STRUCTURE_SEED + 600 + n_env)
    e_ref = rg.integers(0, n_env, size=20000)
    t_ref = (mu[e_ref] + sd[e_ref] * rg.normal(size=(20000, k)) - pm) / ps
    G_ref = g(t_ref)
    ref_mean, ref_std = G_ref.mean(0), G_ref.std(0) + 1e-8

    rng = np.random.default_rng(seed)
    n = 2 * n_per_domain
    e = rng.permutation(np.arange(n) % n_env)
    t = (mu[e] + sd[e] * rng.normal(size=(n, k)) - pm) / ps
    Gt = (g(t) - ref_mean) / ref_std
    lr = Gt - Gt.max(1, keepdims=True)
    comp = np.exp(lr); comp /= comp.sum(1, keepdims=True)
    lib = np.exp(rng.normal(cfg.lib_log_mean, cfg.lib_log_sd, size=(n, 1)))
    rate = comp * lib
    X = rng.negative_binomial(cfg.nb_dispersion,
                              cfg.nb_dispersion / (cfg.nb_dispersion + rate)).astype(np.float32)
    s = (rng.random(n) < 0.5).astype(int)          # dummy domain: no shift at all
    return dict(X=X, t=t.astype(np.float32), s=s, e=e)


def main():
    rows, done = [], set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT); rows = prev.to_dict("records")
        done = {(r["n_env"], r["seed"]) for r in rows}
    combos = list(itertools.product(N_ENVS, SEEDS))
    print(f"{len(combos)} runs, {len(combos)-len(done)} to do   "
          f"(condition needs n_env >= 2n+1 = 5)", flush=True)
    print(f"{'n_env':>6} {'sd':>3} {'CCA':>6} {'MCC':>6} {'gap':>6} {'phi_deg':>8} {'cond?':>6}",
          flush=True)
    for n_env, seed in combos:
        if (n_env, seed) in done:
            continue
        d = generate_fullenv(n_env, N, seed)
        cfg = Model2Config(mode="ivae", prior="cond", latent_dim=2,
                           n_env=n_env, epochs=EPOCHS, seed=seed)
        m_, xn, _ = train_model2(d["X"], d["s"], d["e"], cfg)
        z = m_.embed(xn); t = np.asarray(d["t"], float)
        c = cca_score(t, z, seed=seed)["cca_mean"]
        mm = mcc(t, z, "pearson", seed=seed)["mcc_pearson"]
        gap = c - mm
        phi = float(np.degrees(np.arccos(np.clip(1 - gap, -1, 1))))
        rows.append(dict(n_env=n_env, seed=seed, cca=c, mcc=mm, gap=gap, phi_deg=phi,
                         satisfies=bool(n_env >= 5)))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{n_env:>6} {seed:>3} {c:>6.3f} {mm:>6.3f} {gap:>6.3f} {phi:>8.1f} "
              f"{'YES' if n_env >= 5 else 'no':>6}", flush=True)
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
