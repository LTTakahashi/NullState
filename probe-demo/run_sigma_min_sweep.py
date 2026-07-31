"""
sigma_min(L) SWEEP AT FIXED n_env -- the continuous analogue of assumption (iv).

Khemakhem et al. (2020) Thm 1 (iv) asks that the nk x nk matrix L of
natural-parameter differences be invertible. That is binary on paper and
continuous in practice, and our environment-count sweep confounded the two: at
n_env = 5 (nominally "satisfies") L was invertible but near-singular
(sigma_min 0.027, kappa 451), while n_env = 9 was well conditioned
(sigma_min 1.65, kappa 9.3) -- yet both floors sat on the random-frame law.

This isolates conditioning. n_env is FIXED at 9, so the environment count never
changes and the condition is nominally satisfied throughout. What varies is the
SPECTRUM of L, constructed directly:

  * draw a random orthonormal frame for the nk x (n_env-1) difference matrix,
  * SET its singular values to [1, 1, 1, s] and rescale to a fixed Frobenius
    norm, so TOTAL prior variability is held constant and only the conditioning
    moves,
  * invert the natural parameters back to (mu, sigma) per environment per axis:
        sigma^2 = -1/(2 eta2),   mu = -eta1 / (2 eta2).

s is the knob (kappa = 1/s); the x-axis we REPORT is the sigma_min actually
achieved on the standardised prior the model sees, not the nominal one.

Readout: the identifiability floor gap = CCA - MCC, and a one-sample KS test
against the uniform-random-frame law (E = 0.0997, SD = 0.0880). If the theorem's
condition has a continuous practical analogue, the floor should fall -- and
eventually REJECT the random-frame null -- as sigma_min grows. If it stays pinned
on the law at every conditioning level, then within this regime axis-level
identifiability is undetectable however well conditioned L is.
"""
from __future__ import annotations
import os, sys, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from dgp2 import DGP2Config, _cached_reference, FIXED_STRUCTURE_SEED
from models2 import Model2Config, train_model2
from metrics2 import cca_score, mcc

N_ENV = 9
K_LAT = 2
NK = K_LAT * 2                     # n * k, with k = 2 sufficient statistics
S_LEVELS = [0.02, 0.1, 0.3, 0.6, 1.0]      # kappa = 50, 10, 3.3, 1.67, 1
SEEDS = [int(x) for x in os.environ.get("SM_SEEDS", ",".join(map(str, range(15)))).split(",")]
FRO = 2.0                          # fixed total variability
N, EPOCHS = 1500, 150
OUT = os.environ.get("SM_OUT", "results_sigma_min.csv")


def env_params_for_s(s_level, design_seed=0):
    """Environment (mu, sigma) whose natural-parameter difference matrix has
    singular values proportional to [1,1,1,s], at fixed Frobenius norm."""
    rng = np.random.default_rng(FIXED_STRUCTURE_SEED + 900 + design_seed)
    U, _ = np.linalg.qr(rng.normal(size=(NK, NK)))
    V, _ = np.linalg.qr(rng.normal(size=(N_ENV - 1, N_ENV - 1)))
    sv = np.array([1.0, 1.0, 1.0, s_level])
    sv = sv * (FRO / np.linalg.norm(sv))          # fix total variability
    D = U @ np.diag(sv) @ V[:NK]                  # [NK, n_env-1]

    lam0 = np.concatenate([np.zeros(K_LAT), np.full(K_LAT, -0.5)])   # sigma=1, mu=0
    lam = np.vstack([lam0, lam0 + D.T])           # [n_env, NK]
    eta1, eta2 = lam[:, :K_LAT], lam[:, K_LAT:]
    eta2 = np.minimum(eta2, -0.05)                # keep sigma^2 finite and positive
    sd = np.sqrt(-1.0 / (2.0 * eta2))
    mu = -eta1 / (2.0 * eta2)
    pm = mu.mean(0)
    pv = (sd ** 2 + mu ** 2).mean(0) - pm ** 2
    return mu, sd, pm, np.sqrt(pv)


def achieved_sigma_min(mu, sd, pm, ps):
    """sigma_min of L on the STANDARDISED prior the model actually sees."""
    m, s = (mu - pm) / ps, sd / ps
    lam = np.concatenate([m / s ** 2, -1.0 / (2 * s ** 2)], axis=1)
    L = (lam[1:] - lam[0]).T
    sv = np.linalg.svd(L, compute_uv=False)
    smin = sv[NK - 1] if len(sv) >= NK else 0.0
    return float(smin), float(sv[0] / smin if smin > 1e-12 else np.inf)


def generate(s_level, n_per_domain, seed):
    cfg = DGP2Config(n_genes=500, latent_dim=K_LAT, n_env=N_ENV)
    g, _, _, _ = _cached_reference(cfg)
    mu, sd, pm, ps = env_params_for_s(s_level)

    rg = np.random.default_rng(FIXED_STRUCTURE_SEED + 950)
    e_ref = rg.integers(0, N_ENV, size=20000)
    t_ref = (mu[e_ref] + sd[e_ref] * rg.normal(size=(20000, K_LAT)) - pm) / ps
    G_ref = g(t_ref)
    ref_mean, ref_std = G_ref.mean(0), G_ref.std(0) + 1e-8

    rng = np.random.default_rng(seed)
    n = 2 * n_per_domain
    e = rng.permutation(np.arange(n) % N_ENV)
    t = (mu[e] + sd[e] * rng.normal(size=(n, K_LAT)) - pm) / ps
    Gt = (g(t) - ref_mean) / ref_std
    lr = Gt - Gt.max(1, keepdims=True)
    comp = np.exp(lr); comp /= comp.sum(1, keepdims=True)
    lib = np.exp(rng.normal(cfg.lib_log_mean, cfg.lib_log_sd, size=(n, 1)))
    rate = comp * lib
    X = rng.negative_binomial(cfg.nb_dispersion,
                              cfg.nb_dispersion / (cfg.nb_dispersion + rate)).astype(np.float32)
    s = (rng.random(n) < 0.5).astype(int)
    return dict(X=X, t=t.astype(np.float32), s=s, e=e), (mu, sd, pm, ps)


def main():
    rows, done = [], set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT); rows = prev.to_dict("records")
        done = {(round(r["s_level"], 4), r["seed"]) for r in rows}
    combos = list(itertools.product(S_LEVELS, SEEDS))
    print(f"n_env fixed at {N_ENV} (condition nominally satisfied throughout); "
          f"{len(combos)} runs, {len(combos)-len(done)} to do", flush=True)
    print(f"{'s':>6} {'sig_min':>8} {'kappa':>8} {'sd':>3} {'CCA':>6} {'gap':>7} {'phi':>7}",
          flush=True)
    for s_level, seed in combos:
        if (round(s_level, 4), seed) in done:
            continue
        d, (mu, sd_, pm, ps) = generate(s_level, N, seed)
        smin, kappa = achieved_sigma_min(mu, sd_, pm, ps)
        cfg = Model2Config(mode="ivae", prior="cond", latent_dim=K_LAT,
                           n_env=N_ENV, epochs=EPOCHS, seed=seed)
        m_, xn, _ = train_model2(d["X"], d["s"], d["e"], cfg)
        z = m_.embed(xn); t = np.asarray(d["t"], float)
        c = cca_score(t, z, seed=seed)["cca_mean"]
        mm = mcc(t, z, "pearson", seed=seed)["mcc_pearson"]
        gap = c - mm
        phi = float(np.degrees(np.arccos(np.clip(1 - gap, -1, 1))))
        rows.append(dict(s_level=s_level, sigma_min=smin, kappa=kappa, seed=seed,
                         cca=c, mcc=mm, gap=gap, phi_deg=phi))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{s_level:>6} {smin:>8.3f} {kappa:>8.1f} {seed:>3} {c:>6.3f} "
              f"{gap:>7.3f} {phi:>7.1f}", flush=True)
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
