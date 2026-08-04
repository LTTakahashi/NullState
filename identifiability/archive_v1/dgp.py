"""
Data-generating process (DGP) for the support-overlap identifiability probe.

Design goal (the whole ballgame): instantiate exactly what a disentanglement /
batch-correction model assumes -- a SHARED biological factor `t` plus a
domain-specific effect -- with TWO independently controllable knobs:

  rho   : biological support overlap of the two domains in `t`  (in [0, 1])
  delta : magnitude of an orthogonal, REMOVABLE batch offset    (>= 0)

Keeping rho and delta separable is the scientific point. Support mismatch (low
rho) confounds domain with biology and is the identifiability killer; a large
removable offset (high delta) makes the domain easy to *detect* but does not, on
its own, destroy identifiability. The probe shows recovery collapses along rho,
not delta -- which is why a raw source-detecting adversary alone is NOT a
sufficient trust signal (see metrics.raw_certificate + off-manifold overlap).

Ground truth returned: X (counts), t (true latent), s (domain label 0=reference,
1=in-vitro). Because g(t) is SHARED across domains, any failure to recover the
cross-domain biological axis is attributable to support mismatch alone.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DGPConfig:
    n_per_domain: int = 4000
    rho: float = 1.0            # biological support overlap in [0, 1]
    delta: float = 1.0          # magnitude of the removable, t-independent batch offset
    n_genes: int = 500
    latent_dim: int = 1         # shift is applied to dim 0 (the "developmental axis")
    decoder_hidden: int = 64
    decoder_layers: int = 2     # >=1 ; number of nonlinear layers in the fixed decoder
    base_log_rate: float = 1.2  # sets baseline count scale (exp(base) ~ counts)
    bio_scale: float = 1.0      # how strongly biology modulates log-rate
    noise: str = "poisson"      # "poisson" | "nb" | "gaussian"
    nb_dispersion: float = 5.0  # NB inverse-dispersion (higher = closer to Poisson)
    seed: int = 0


def _fixed_decoder(rng: np.random.Generator, in_dim: int, out_dim: int,
                   hidden: int, layers: int):
    """Return a callable g: R^{n x in_dim} -> R^{n x out_dim} with fixed random
    smooth (tanh) weights. This is the *shared* ground-truth generative map."""
    dims = [in_dim] + [hidden] * max(layers - 1, 0) + [out_dim]
    Ws, bs = [], []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        # scaled init keeps activations well-conditioned
        Ws.append(rng.normal(0, 1.0 / np.sqrt(d_in), size=(d_in, d_out)))
        bs.append(rng.normal(0, 0.1, size=d_out))

    def g(t: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(t).astype(float)
        for i, (W, b) in enumerate(zip(Ws, bs)):
            h = h @ W + b
            if i < len(Ws) - 1:
                h = np.tanh(h)
        return h

    return g


def overlap_coefficient(rho: float) -> float:
    """Analytical overlap coefficient of U[0,1] and U[1-rho, 2-rho] is rho."""
    return float(np.clip(rho, 0.0, 1.0))


def generate(cfg: DGPConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    n, k, G = cfg.n_per_domain, cfg.latent_dim, cfg.n_genes

    # --- sample the shared biological factor t, per domain ------------------
    # Reference (s=0): t[:,0] ~ U[0,1].  In-vitro (s=1): t[:,0] ~ U[c, c+1],
    # c = 1 - rho, so support overlap along dim 0 is exactly rho.
    c = 1.0 - float(np.clip(cfg.rho, 0.0, 1.0))
    tA = rng.uniform(0.0, 1.0, size=(n, k))
    tB = rng.uniform(0.0, 1.0, size=(n, k))
    tB[:, 0] = rng.uniform(c, c + 1.0, size=n)          # shift only the primary axis
    t = np.vstack([tA, tB])
    s = np.concatenate([np.zeros(n, int), np.ones(n, int)])

    # --- shared nonlinear decoder g(t) --------------------------------------
    g = _fixed_decoder(rng, k, G, cfg.decoder_hidden, cfg.decoder_layers)
    G_t = g(t)
    # standardize per gene across the whole sampled range so bio_scale is meaningful
    G_t = (G_t - G_t.mean(0)) / (G_t.std(0) + 1e-8)

    # --- removable, t-INDEPENDENT batch offset (a batch effect) --------------
    direction = rng.normal(size=G)
    direction /= (np.linalg.norm(direction) + 1e-8)
    offset = np.zeros((2 * n, G))
    offset[s == 1] = cfg.delta * direction                # only in-vitro is shifted

    log_rate = cfg.base_log_rate + cfg.bio_scale * G_t + offset
    rate = np.exp(np.clip(log_rate, -8, 8))

    # --- observation noise ---------------------------------------------------
    if cfg.noise == "poisson":
        X = rng.poisson(rate).astype(np.float32)
    elif cfg.noise == "nb":
        r = cfg.nb_dispersion
        p = r / (r + rate)
        X = rng.negative_binomial(r, p).astype(np.float32)
    elif cfg.noise == "gaussian":
        X = (log_rate + rng.normal(0, 0.3, size=log_rate.shape)).astype(np.float32)
    else:
        raise ValueError(f"unknown noise {cfg.noise!r}")

    return dict(
        X=X, t=t.astype(np.float32), s=s, rate=rate.astype(np.float32),
        overlap=overlap_coefficient(cfg.rho), cfg=cfg,
        overlap_region=(t[:, 0] >= c) & (t[:, 0] <= 1.0),  # cells in the shared band
    )


if __name__ == "__main__":
    for rho in (1.0, 0.5, 0.0):
        d = generate(DGPConfig(n_per_domain=500, rho=rho, n_genes=50, seed=0))
        frac = d["overlap_region"].mean()
        print(f"rho={rho:.2f}  overlap={d['overlap']:.2f}  "
              f"X.shape={d['X'].shape}  mean_count={d['X'].mean():.2f}  "
              f"cells_in_shared_band={frac:.2f}")
