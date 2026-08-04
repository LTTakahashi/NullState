"""
DGP v3 -- the HARD regime: recovery that genuinely depends on support overlap.

WHY v3. The v2 recovery-invariance null is honest but likely trivial: the shifted
factor is WITHIN-DOMAIN decodable, so a shared encoder recovers it per-cell
without any cross-domain alignment, and overlap therefore cannot touch recovery
(only removal). v3 removes exactly that triviality.

THE MECHANISM (a domain-specific gauge that only overlap can pin). The shared
biology t = (t_shift, t_free) is rendered through a DOMAIN-SPECIFIC ROTATION
R(theta_s) that mixes t_shift into t_free BEFORE the shared nonlinear mixing g:

        x = counts( g( R(theta_s) . t ) + removable_batch )

with theta_A = -theta_g/2, theta_B = +theta_g/2 (symmetric). Because a real
integration encoder is SHARED (scVI-style: only the decoder sees the domain), the
model cannot apply a domain-specific inverse; it must find ONE encoder that maps
both domains to a common latent. That reconciliation -- undoing R(theta_A) vs
R(theta_B) into a common frame -- is constrained ONLY where both domains have
support (the overlap band on t_shift). As overlap rho falls, the exclusive
regions have no cross-domain counterpart to align against, the common frame is
un-pinned there, and the recovered axes rotate domain-specifically -> the
matched-oracle MCC gap (rotation-sensitive) GROWS.

The common-frame t_shift is therefore NOT within-domain decodable (within one
domain you only ever see R(theta_s) t, and theta_s relative to the other domain
is unknown), which is the property v2 lacked. Set theta_g = 0 to recover exactly
the v2 regime (no gauge -> the null); theta_g > 0 turns on the phenomenon.

Everything else (symmetric pooled-invariant overlap on t_shift, fixed reference-
grid standardisation on the POOLED ROTATED distribution, non-saturating mixing,
depth decoupled from rho/delta, NB counts, environment structure for an iVAE
prior) follows v2 so that rho and delta stay decoupled and the metric stays fair.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
from dataclasses import dataclass, replace
import numpy as np

from core.dgp import (DGP2Config, _smooth_leaky, _env_params, sample_shifted_axis,
                  FIXED_STRUCTURE_SEED, REF_GRID_N)


@dataclass(frozen=True)
class DGP3Config(DGP2Config):
    gauge_angle: float = 0.7854   # theta_g (radians); 0 == v2 regime. pi/4 ~ 0.785


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(2)                                   # rotate the (0,1) plane only
    R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    return R


def _build_mixing3(cfg: DGP3Config):
    """Shared smooth-leaky mixing g: R^n -> R^G (same construction as v2, keyed
    to the structural config so it is identical across rho/delta/seed)."""
    rng = np.random.default_rng(FIXED_STRUCTURE_SEED + 100)
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


_REF3: dict = {}


def _cached_reference3(cfg: DGP3Config):
    """Mixing map, per-gene standardisation (on the POOLED ROTATED prior so gene
    statistics stay rho-invariant), and the removable batch direction."""
    key = (cfg.n_genes, cfg.latent_dim, cfg.n_env, cfg.mix_layers,
           round(cfg.leaky_alpha, 4), round(cfg.gauge_angle, 5))
    if key in _REF3:
        return _REF3[key]
    g = _build_mixing3(cfg)
    rng = np.random.default_rng(FIXED_STRUCTURE_SEED + 101)
    n = cfg.latent_dim
    # pooled prior: axis0 ~ Uniform(0,1)->(0,1), axes1.. ~ standardised env prior,
    # half rotated by +theta/2 (domain B) and half by -theta/2 (domain A).
    t = np.empty((REF_GRID_N, n))
    t[:, 0] = (rng.random(REF_GRID_N) - 0.5) * np.sqrt(12.0)
    mu, sd, pm, ps = _env_params(cfg)
    if n > 1:
        e = rng.integers(0, cfg.n_env, size=REF_GRID_N)
        t[:, 1:] = (mu[e] + sd[e] * rng.normal(size=(REF_GRID_N, n - 1)) - pm) / ps
    RA, RB = _rot(-cfg.gauge_angle / 2), _rot(+cfg.gauge_angle / 2)
    half = REF_GRID_N // 2
    y = t.copy()
    y[:half] = t[:half] @ RA.T
    y[half:] = t[half:] @ RB.T
    Gt = g(y)
    mean, std = Gt.mean(0), Gt.std(0) + 1e-8
    d = rng.normal(size=cfg.n_genes); d = (d - d.mean()) / d.std()
    _REF3[key] = (g, mean, std, d)
    return _REF3[key]


def generate(cfg: DGP3Config) -> dict:
    if cfg.latent_dim < 2:
        raise ValueError("v3 requires latent_dim >= 2 (gauge mixes axes 0 and 1)")
    rng = np.random.default_rng(cfg.seed)
    n, k, G = cfg.n_per_domain, cfg.latent_dim, cfg.n_genes
    g, ref_mean, ref_std, direction = _cached_reference3(cfg)

    s = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    t = np.empty((2 * n, k))
    raw0 = np.concatenate([sample_shifted_axis(rng, n, cfg.rho, 0),
                           sample_shifted_axis(rng, n, cfg.rho, 1)])
    t[:, 0] = (raw0 - 0.5) * np.sqrt(12.0)               # common-frame t_shift
    e = np.concatenate([rng.permutation(np.arange(n) % cfg.n_env),
                        rng.permutation(np.arange(n) % cfg.n_env)])
    mu, sd, pm, ps = _env_params(cfg)
    if k > 1:
        t[:, 1:] = (mu[e] + sd[e] * rng.normal(size=(2 * n, k - 1)) - pm) / ps

    # DOMAIN GAUGE: rotate the common-frame latent per domain before mixing.
    RA, RB = _rot(-cfg.gauge_angle / 2), _rot(+cfg.gauge_angle / 2)
    y = t.copy()
    y[s == 0] = t[s == 0] @ RA.T
    y[s == 1] = t[s == 1] @ RB.T

    Gt = (g(y) - ref_mean) / ref_std
    log_rate = cfg.bio_scale * Gt
    log_rate[s == 1] += cfg.delta * direction

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
        X=X, t=t.astype(np.float32), s=s, e=e,      # t is the COMMON-frame target
        rate=rate.astype(np.float32), lib=lib.ravel().astype(np.float32),
        overlap=float(np.clip(cfg.rho, 0.0, 1.0)), cfg=cfg,
        overlap_region=(t[:, 0] >= lo) & (t[:, 0] <= hi),
        shifted_axis=0, gauge_angle=cfg.gauge_angle,
    )


if __name__ == "__main__":
    for th in (0.0, np.pi / 4):
        d = generate(DGP3Config(rho=0.5, gauge_angle=th, n_per_domain=1000, seed=0))
        X, s = d["X"], d["s"]
        print(f"theta={th:.3f}  depth_B/A={X[s==1].sum(1).mean()/X[s==0].sum(1).mean():.3f} "
              f"gene_mu={X.mean(0).mean():.3f}  band={d['overlap_region'].mean():.3f}")
