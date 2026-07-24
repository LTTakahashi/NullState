"""
DGP v2 for the support-overlap identifiability probe.

WHY v2 EXISTS. The v1 DGP (dgp.py) failed an adversarial audit: rho was not a
pure knob. Lowering rho moved domain B's t-range outside the range used for
*pooled* per-gene standardization, and the decoder's tanh then saturated, so
lowering rho also (a) compressed domain B's signal ~3.4x, (b) opened a per-gene
mean gap equivalent to delta ~ 36 (while the delta sweep only reached 4), and
(c) created a ~30% sequencing-depth gap. Every headline metric turned out to be
a closed-form function of rho, and the learned model sat ON that analytic
ceiling everywhere -- i.e. there was no measurable non-identifiability at all.

v2 fixes each cause by construction:

  F1  SYMMETRIC, POOLED-INVARIANT OVERLAP.  With w = (1-rho)/2,

          p_A = 2 on [0,w)   1 on [w,1-w]   0 on (1-w,1]
          p_B = 0 on [0,w)   1 on [w,1-w]   2 on (1-w,1]

      so p_A + p_B == 2 everywhere: the POOLED marginal of the shifted axis is
      exactly Uniform[0,1] at EVERY rho, while domain A has strictly zero
      density above 1-w (genuine support mismatch, not reweighting).
      Overlap coefficient OVL = int min(p_A,p_B) = 1 - 2w = rho, exactly.
      NOTE: the brief's "pin each domain's marginal to a canonical shape AND set
      rho = OVL" is unsatisfiable -- OVL(p_A,p_B) = 1 iff p_A = p_B. Pooled
      invariance is the achievable (and sufficient) form: it is what makes the
      observable count statistics rho-invariant.

  F2  FIXED REFERENCE-GRID STANDARDIZATION.  Per-gene centering/scaling
      constants are computed ONCE from the pooled prior with a fixed seed, and
      are independent of rho, delta and the data seed. v1 standardized within
      each dataset, which is what coupled rho to the gene-level statistics.

  F3  NO SATURATION.  The mixing MLP uses the smooth leaky-ReLU
      s_a(x) = a*x + (1-a)*softplus(x) (Gresele et al. 2021), whose derivative
      lies in (a, 1): strictly increasing, asymptotically linear, never
      saturating. Square hidden layers keep it invertible; a fixed full-column-
      rank linear expansion lifts n -> n_genes (injective).

  F4  DEPTH IS EXACTLY DECOUPLED.  Library sizes are drawn from a fixed
      lognormal independent of rho, delta, s and t; rates are row-normalised to
      a simplex BEFORE being multiplied by the library size. delta is therefore
      a purely COMPOSITIONAL, removable offset with no depth side-effect.

  F5  >= 2 LATENT DIMS, shifted vs unshifted separated.  Axis 0 carries rho.
      Axes 1.. are identically distributed in both domains at every rho, so a
      rotation-sensitive metric (MCC) can detect mixing between them -- v1's
      1-D latent made every rotation trivial and every metric rotation-blind.

  F6  ENOUGH ENVIRONMENTS FOR THE THEORY TO APPLY.  iVAE (Khemakhem 2020,
      Thm 1 assumption (iv)) needs nk+1 distinct auxiliary values; for a
      Gaussian location-scale prior (k=2) at latent dim n that is 2n+1. A
      2-domain design cannot satisfy this for any n >= 1, which voided v1's
      identifiability framing outright. Here an environment index
      e in {0..n_env-1} carries location-scale variation on the UNSHIFTED axes
      (default n_env = 2n+1 = 5), balanced across domains so it cannot leak
      into the pooled statistics. The SHIFTED axis deliberately has only two
      distinct prior values (s = A/B), so it is exactly the coordinate the
      variability condition does NOT cover -- which is the regime rho probes.
      Set n_env = 2 to obtain the deliberately-violating arm.

Ground truth returned: X (counts), t (true latent), s (domain), e (environment).
Because the mixing map g is SHARED across domains and environments, any failure
to recover the cross-domain biological axes is attributable to support mismatch.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Seed for everything that must be invariant to rho/delta/data-seed: the mixing
# map, the batch direction, and the per-gene standardisation constants.
FIXED_STRUCTURE_SEED = 12345
REF_GRID_N = 20000


# --------------------------------------------------------------------------- #
#  config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DGP2Config:
    n_per_domain: int = 3000
    rho: float = 1.0             # support overlap of the SHIFTED axis, == OVL, in [0,1]
    delta: float = 1.0           # magnitude of the removable compositional batch offset
    n_genes: int = 500
    latent_dim: int = 2          # >= 2 : axis 0 shifted, axes 1.. unshifted
    n_env: int = 5               # 2*latent_dim+1 satisfies iVAE (iv); 2 violates it
    mix_layers: int = 3          # square smooth-leaky-ReLU layers before expansion
    leaky_alpha: float = 0.2     # slope of the smooth leaky-ReLU
    bio_scale: float = 1.0       # log-rate modulation per unit standardised gene signal
    lib_log_mean: float = 8.5    # log library size ~ N(lib_log_mean, lib_log_sd)
    lib_log_sd: float = 0.25
    noise: str = "nb"            # "nb" | "poisson"
    nb_dispersion: float = 5.0
    seed: int = 0


# --------------------------------------------------------------------------- #
#  the fixed, shared mixing map  g : R^n -> R^G
# --------------------------------------------------------------------------- #
def _smooth_leaky(x, a):
    """s_a(x) = a x + (1-a) softplus(x).  Derivative in (a, 1): strictly
    increasing and asymptotically linear, so it never saturates and is
    invertible elementwise (Gresele et al. 2021)."""
    # numerically stable softplus
    sp = np.logaddexp(0.0, x)
    return a * x + (1.0 - a) * sp


def _build_mixing(cfg: DGP2Config):
    """Fixed random mixing map, identical for every rho/delta/seed.

    Square layers (dim n) with orthogonal weights keep the nonlinear part
    exactly invertible and perfectly conditioned; a fixed G x n expansion with
    full column rank lifts to gene space injectively.
    """
    rng = np.random.default_rng(FIXED_STRUCTURE_SEED)
    n, G = cfg.latent_dim, cfg.n_genes

    Ws, bs = [], []
    for _ in range(cfg.mix_layers):
        A = rng.normal(size=(n, n))
        Q, R = np.linalg.qr(A)                       # orthogonal -> cond(W) = 1
        Q = Q * np.sign(np.diag(R))
        Ws.append(Q)
        bs.append(rng.normal(0, 0.3, size=n))

    E = rng.normal(0, 1.0 / np.sqrt(n), size=(n, G))  # expansion, full column rank

    def g(t: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(np.asarray(t, dtype=float))
        for W, b in zip(Ws, bs):
            h = _smooth_leaky(h @ W + b, cfg.leaky_alpha)
        return h @ E

    g.Ws, g.bs, g.E = Ws, bs, E
    return g


def mixing_conditioning(cfg: DGP2Config) -> dict:
    """Diagnostics proving the mixing map is invertible and well conditioned."""
    g = _build_mixing(cfg)
    return dict(
        square_conds=[float(np.linalg.cond(W)) for W in g.Ws],
        expansion_cond=float(np.linalg.cond(g.E)),
        expansion_rank=int(np.linalg.matrix_rank(g.E)),
        latent_dim=cfg.latent_dim,
        # derivative of the elementwise nonlinearity is bounded in (alpha, 1)
        nonlinearity_deriv_bounds=(cfg.leaky_alpha, 1.0),
    )


# --------------------------------------------------------------------------- #
#  latent prior
# --------------------------------------------------------------------------- #
def _env_params(cfg: DGP2Config):
    """Location-scale prior parameters per (unshifted axis, environment).

    Fixed across rho/delta/seed. Returns mu, sd of shape [n_env, latent_dim-1]
    plus the pooled mean/sd used to standardise those axes to (0, 1), so the
    unshifted axes' pooled marginal is a fixed constant at every rho.
    """
    rng = np.random.default_rng(FIXED_STRUCTURE_SEED + 1)
    m = cfg.latent_dim - 1
    if m == 0:
        return np.zeros((cfg.n_env, 0)), np.ones((cfg.n_env, 0)), np.zeros(0), np.ones(0)
    mu = rng.normal(0.0, 0.8, size=(cfg.n_env, m))
    sd = rng.uniform(0.4, 1.0, size=(cfg.n_env, m))
    pooled_mean = mu.mean(0)
    pooled_var = (sd ** 2 + mu ** 2).mean(0) - pooled_mean ** 2
    return mu, sd, pooled_mean, np.sqrt(pooled_var)


def sample_shifted_axis(rng, n, rho, domain):
    """Sample the rho-carrying axis for one domain.

    w = (1-rho)/2.  Domain A: mass 2w on [0,w) and 1-2w on [w,1-w].
                    Domain B: mass 1-2w on [w,1-w] and 2w on (1-w,1].
    Pooled over equal-sized domains this is exactly Uniform[0,1] for every rho.
    """
    w = 0.5 * (1.0 - float(np.clip(rho, 0.0, 1.0)))
    excl = rng.random(n) < (2.0 * w)                 # in the domain-exclusive region
    u = rng.random(n)
    x = np.empty(n)
    if domain == 0:                                   # A: exclusive band is [0, w)
        x[excl] = u[excl] * w
        x[~excl] = w + u[~excl] * (1.0 - 2.0 * w)
    else:                                             # B: exclusive band is (1-w, 1]
        x[excl] = 1.0 - u[excl] * w
        x[~excl] = w + u[~excl] * (1.0 - 2.0 * w)
    return x


def _reference_grid(cfg: DGP2Config, g) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene standardisation constants from the POOLED prior, computed once
    with a fixed seed. Independent of rho, delta and the data seed (F2)."""
    rng = np.random.default_rng(FIXED_STRUCTURE_SEED + 2)
    t = np.empty((REF_GRID_N, cfg.latent_dim))
    t[:, 0] = (rng.random(REF_GRID_N) - 0.5) * np.sqrt(12.0)   # pooled Unif[0,1] -> (0,1)
    mu, sd, pm, ps = _env_params(cfg)
    if cfg.latent_dim > 1:
        e = rng.integers(0, cfg.n_env, size=REF_GRID_N)
        raw = mu[e] + sd[e] * rng.normal(size=(REF_GRID_N, cfg.latent_dim - 1))
        t[:, 1:] = (raw - pm) / ps
    Gt = g(t)
    return Gt.mean(0), Gt.std(0) + 1e-8


_REF_CACHE: dict = {}


def _cached_reference(cfg: DGP2Config):
    """Mixing map + standardisation constants depend only on the STRUCTURAL
    config (not on rho/delta/seed/n), so cache on that key."""
    key = (cfg.n_genes, cfg.latent_dim, cfg.n_env, cfg.mix_layers, cfg.leaky_alpha)
    if key not in _REF_CACHE:
        g = _build_mixing(cfg)
        mean, std = _reference_grid(cfg, g)
        rng = np.random.default_rng(FIXED_STRUCTURE_SEED + 3)
        # PER-GENE denomination: unit standard deviation across genes, NOT unit
        # L2 norm. With unit L2 norm the per-gene shift is delta/sqrt(G) ~ 0.002
        # at G=500, so delta was inert -- the same class of bug as v1's inert
        # adv_lambda. Denominated this way, delta = 1 puts the removable offset
        # on the same log-rate scale as bio_scale = 1, so the delta sweep spans
        # (and exceeds) the expression gap that rho itself induces.
        d = rng.normal(size=cfg.n_genes)
        d = (d - d.mean()) / d.std()
        _REF_CACHE[key] = (g, mean, std, d)
    return _REF_CACHE[key]


# --------------------------------------------------------------------------- #
#  generate
# --------------------------------------------------------------------------- #
def generate(cfg: DGP2Config) -> dict:
    if cfg.latent_dim < 2:
        raise ValueError("v2 requires latent_dim >= 2 (shifted + unshifted axes)")
    rng = np.random.default_rng(cfg.seed)
    n, k, G = cfg.n_per_domain, cfg.latent_dim, cfg.n_genes
    g, ref_mean, ref_std, direction = _cached_reference(cfg)

    # --- latents -----------------------------------------------------------
    s = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    t = np.empty((2 * n, k))
    raw0 = np.concatenate([sample_shifted_axis(rng, n, cfg.rho, 0),
                           sample_shifted_axis(rng, n, cfg.rho, 1)])
    t[:, 0] = (raw0 - 0.5) * np.sqrt(12.0)            # pooled -> mean 0, var 1, all rho

    # environment: balanced across domains, so it cannot leak into pooled stats
    e = np.concatenate([rng.permutation(np.arange(n) % cfg.n_env),
                        rng.permutation(np.arange(n) % cfg.n_env)])
    mu, sd, pm, ps = _env_params(cfg)
    if k > 1:
        t[:, 1:] = (mu[e] + sd[e] * rng.normal(size=(2 * n, k - 1)) - pm) / ps

    # --- shared mixing + FIXED-grid standardisation (F2, F3) ---------------
    Gt = (g(t) - ref_mean) / ref_std

    # --- removable compositional batch offset (F4) -------------------------
    log_rate = cfg.bio_scale * Gt
    log_rate[s == 1] += cfg.delta * direction

    # softmax over genes -> composition; depth is supplied separately and is
    # independent of rho, delta, s and t, so delta cannot move sequencing depth.
    lr = log_rate - log_rate.max(1, keepdims=True)
    comp = np.exp(lr)
    comp /= comp.sum(1, keepdims=True)
    lib = np.exp(rng.normal(cfg.lib_log_mean, cfg.lib_log_sd, size=(2 * n, 1)))
    rate = comp * lib

    if cfg.noise == "poisson":
        X = rng.poisson(rate).astype(np.float32)
    elif cfg.noise == "nb":
        r = cfg.nb_dispersion
        X = rng.negative_binomial(r, r / (r + rate)).astype(np.float32)
    else:
        raise ValueError(f"unknown noise {cfg.noise!r}")

    w = 0.5 * (1.0 - float(np.clip(cfg.rho, 0.0, 1.0)))
    lo, hi = (w - 0.5) * np.sqrt(12.0), (1.0 - w - 0.5) * np.sqrt(12.0)
    return dict(
        X=X, t=t.astype(np.float32), s=s, e=e,
        rate=rate.astype(np.float32), comp=comp.astype(np.float32),
        lib=lib.ravel().astype(np.float32),
        overlap=float(np.clip(cfg.rho, 0.0, 1.0)), cfg=cfg,
        # cells inside the shared band of the shifted axis
        overlap_region=(t[:, 0] >= lo) & (t[:, 0] <= hi),
        shifted_axis=0,
    )


def true_overlap_coefficient(rho: float) -> float:
    """OVL = int min(p_A, p_B) = 1 - 2w = rho, exactly, by construction."""
    return float(np.clip(rho, 0.0, 1.0))


if __name__ == "__main__":
    cfg0 = DGP2Config()
    print("mixing conditioning:", mixing_conditioning(cfg0))
    print(f"\n{'rho':>5} {'band':>6} {'depth':>9} {'gene_mu':>9} {'gene_sd':>9} {'t0_sdA':>8} {'t0_sdB':>8}")
    for rho in (1.0, 0.75, 0.5, 0.25, 0.0):
        d = generate(DGP2Config(rho=rho, n_per_domain=1500, seed=0))
        X, s, t = d["X"], d["s"], d["t"]
        print(f"{rho:>5} {d['overlap_region'].mean():>6.3f} {X.sum(1).mean():>9.1f} "
              f"{X.mean(0).mean():>9.4f} {X.mean(0).std():>9.4f} "
              f"{t[s==0,0].std():>8.3f} {t[s==1,0].std():>8.3f}")
