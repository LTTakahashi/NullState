"""
DGP variant: put the SHIFTED DIRECTION at an angle to the ground-truth basis.

WHY. In dgp2 the support-mismatch direction *is* latent coordinate 0. An
alignment objective that folds that direction therefore produces an
AXIS-FACTORISED corruption by construction -- and the matched-oracle gap
(= CCA - MCC) has exactly the axis-factorised maps as its zero set. So our
headline demonstration ("the gap misses the alignment collapse") may be an
artifact of the shifted direction coinciding with a latent coordinate: the table
predicts collapse-of-an-axis -> 0.000 but collapse-of-a-ROTATED-axis -> 0.142.

THIS MODULE removes that coincidence. Sample (a, b) exactly as dgp2 does -- `a`
carries the symmetric pooled-invariant overlap structure, `b` is the free,
environment-structured axis -- then declare the GROUND-TRUTH latent to be

        s = (a, b) R(theta)^T

so the support mismatch lies along the direction R(theta) e_0, at angle theta to
s's own coordinate axes. theta = 0 reproduces dgp2 exactly. The observation is
x = counts(g(s) + delta * batch) with the same shared mixing map and a reference
grid recomputed on the pooled ROTATED prior, so per-gene statistics stay
rho-invariant as in dgp2.

INTERPRETATION CAVEAT (why the faithful arm is a required control). At theta > 0
the declared basis s is no longer the independent-component basis (a, b). A model
with any preference for independent axes may recover (a, b), scoring MCC ~ cos
theta against s even with perfect recovery -- a BASIS MISMATCH, not folding. So
the faithful arm must be run at the same theta, and the quantity of interest is
the DIFFERENCE (alignment gap - faithful gap) at matched theta: that is the part
attributable to the alignment objective rather than to the basis choice.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from dgp2 import (DGP2Config, _smooth_leaky, _env_params, sample_shifted_axis,
                  FIXED_STRUCTURE_SEED, REF_GRID_N)


@dataclass(frozen=True)
class RotBasisConfig(DGP2Config):
    basis_angle: float = 0.0        # theta; 0 == dgp2 exactly


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(2)
    R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    return R


def _build_mixing(cfg):
    rng = np.random.default_rng(FIXED_STRUCTURE_SEED + 200)
    n, G = cfg.latent_dim, cfg.n_genes
    Ws, bs = [], []
    for _ in range(cfg.mix_layers):
        Q, R = np.linalg.qr(rng.normal(size=(n, n)))
        Ws.append(Q * np.sign(np.diag(R)))
        bs.append(rng.normal(0, 0.3, size=n))
    E = rng.normal(0, 1.0 / np.sqrt(n), size=(n, G))

    def g(y):
        h = np.atleast_2d(np.asarray(y, float))
        for W, b in zip(Ws, bs):
            h = _smooth_leaky(h @ W + b, cfg.leaky_alpha)
        return h @ E
    return g


_REF: dict = {}


def _cached_reference(cfg):
    key = (cfg.n_genes, cfg.latent_dim, cfg.n_env, cfg.mix_layers,
           round(cfg.leaky_alpha, 4), round(cfg.basis_angle, 5))
    if key in _REF:
        return _REF[key]
    g = _build_mixing(cfg)
    rng = np.random.default_rng(FIXED_STRUCTURE_SEED + 201)
    n = cfg.latent_dim
    ab = np.empty((REF_GRID_N, n))
    ab[:, 0] = (rng.random(REF_GRID_N) - 0.5) * np.sqrt(12.0)
    mu, sd, pm, ps = _env_params(cfg)
    if n > 1:
        e = rng.integers(0, cfg.n_env, size=REF_GRID_N)
        ab[:, 1:] = (mu[e] + sd[e] * rng.normal(size=(REF_GRID_N, n - 1)) - pm) / ps
    s = ab @ _rot(cfg.basis_angle).T
    Gt = g(s)
    mean, std = Gt.mean(0), Gt.std(0) + 1e-8
    d = rng.normal(size=cfg.n_genes); d = (d - d.mean()) / d.std()
    _REF[key] = (g, mean, std, d)
    return _REF[key]


def generate(cfg: RotBasisConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    n, k, G = cfg.n_per_domain, cfg.latent_dim, cfg.n_genes
    g, ref_mean, ref_std, direction = _cached_reference(cfg)

    dom = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    ab = np.empty((2 * n, k))
    raw0 = np.concatenate([sample_shifted_axis(rng, n, cfg.rho, 0),
                           sample_shifted_axis(rng, n, cfg.rho, 1)])
    ab[:, 0] = (raw0 - 0.5) * np.sqrt(12.0)          # the shift-carrying variable
    e = np.concatenate([rng.permutation(np.arange(n) % cfg.n_env),
                        rng.permutation(np.arange(n) % cfg.n_env)])
    mu, sd, pm, ps = _env_params(cfg)
    if k > 1:
        ab[:, 1:] = (mu[e] + sd[e] * rng.normal(size=(2 * n, k - 1)) - pm) / ps

    # GROUND TRUTH: rotate, so the shift direction sits at `basis_angle`
    s_true = ab @ _rot(cfg.basis_angle).T

    Gt = (g(s_true) - ref_mean) / ref_std
    log_rate = cfg.bio_scale * Gt
    log_rate[dom == 1] += cfg.delta * direction
    lr = log_rate - log_rate.max(1, keepdims=True)
    comp = np.exp(lr); comp /= comp.sum(1, keepdims=True)
    lib = np.exp(rng.normal(cfg.lib_log_mean, cfg.lib_log_sd, size=(2 * n, 1)))
    rate = comp * lib
    if cfg.noise == "poisson":
        X = rng.poisson(rate).astype(np.float32)
    else:
        r = cfg.nb_dispersion
        X = rng.negative_binomial(r, r / (r + rate)).astype(np.float32)

    w = 0.5 * (1.0 - float(np.clip(cfg.rho, 0.0, 1.0)))
    lo, hi = (w - 0.5) * np.sqrt(12.0), (1.0 - w - 0.5) * np.sqrt(12.0)
    return dict(
        X=X, t=s_true.astype(np.float32), s=dom, e=e,
        ab=ab.astype(np.float32),                    # the independent-component basis
        overlap=float(np.clip(cfg.rho, 0.0, 1.0)), cfg=cfg,
        overlap_region=(ab[:, 0] >= lo) & (ab[:, 0] <= hi),
        basis_angle=cfg.basis_angle,
    )


if __name__ == "__main__":
    for th in (0.0, np.pi / 4):
        d = generate(RotBasisConfig(rho=0.5, basis_angle=th, n_per_domain=1000, seed=0))
        X, dom = d["X"], d["s"]
        print(f"theta={th:.3f} depth_B/A={X[dom==1].sum(1).mean()/X[dom==0].sum(1).mean():.3f} "
              f"gene_mu={X.mean(0).mean():.3f} band={d['overlap_region'].mean():.3f}")
