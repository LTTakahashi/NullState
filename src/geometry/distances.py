"""Dimension-robust optimal-transport distances for Aim 3 geometry (Phase 1 WS2).

Implements the two estimators specified in `docs/research_strategy_feasibility_power.md` §3,
chosen because exact Wasserstein in a 15-30-dim latent has an n^(-1/d) bias that is fatal at
the cell counts available:

  * `sliced_wasserstein`  -- PRIMARY metric. Averages 1-D Wasserstein-1 over random
    projections; converges at n^(-1/2) independent of dimension, removing the curse of
    dimensionality.
  * `sinkhorn_divergence` -- debiased entropic-OT CROSS-CHECK. Interpolates between OT and
    MMD; more sample-stable than exact OT at small N. Debiased so S(X, X) = 0.

Both operate on the reduced z_lin representation (the leading 15-30 disentangled dims), not
gene space. Pure numpy + scipy -- no GPU, no scvi. **Raw distances from these are never
interpreted on their own**: always read against the matched-N self-distance floor in
`matched_n.py` (feasibility_power §2, §5).
"""

from __future__ import annotations

import numpy as np


def _as2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D (n_cells, n_dims) array, got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("empty point cloud")
    return X


def _w1d(a: np.ndarray, b: np.ndarray, p: int) -> float:
    """1-D Wasserstein-p between samples a, b (unequal sizes allowed)."""
    if p == 1:
        from scipy.stats import wasserstein_distance  # exact 1-D W1, any sizes
        return float(wasserstein_distance(a, b))
    # p >= 2: quantile-grid approximation of (∫|F_a^{-1}-F_b^{-1}|^p)^(1/p)
    q = np.linspace(0.0, 1.0, max(len(a), len(b)), endpoint=False) + 0.5 / max(len(a), len(b))
    qa = np.quantile(a, q)
    qb = np.quantile(b, q)
    return float((np.mean(np.abs(qa - qb) ** p)) ** (1.0 / p))


def sliced_wasserstein(X, Y, n_projections: int = 200, p: int = 1,
                       random_state: int | None = None, return_components: bool = False):
    """Sliced-Wasserstein-``p`` distance between point clouds ``X`` (n,d) and ``Y`` (m,d).

    SW_p = (E_u[ W_p(<u,X>, <u,Y>)^p ])^(1/p) over random unit directions u. Default p=1
    (the natural choice for the project's W1 framing) uses exact 1-D W1 per projection.
    n and m may differ. Deterministic given ``random_state``.
    """
    X = _as2d(X)
    Y = _as2d(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"dimension mismatch: X has {X.shape[1]}, Y has {Y.shape[1]}")
    if n_projections < 1:
        raise ValueError("n_projections must be >= 1")
    d = X.shape[1]
    rng = np.random.default_rng(random_state)
    dirs = rng.standard_normal((n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    Xp = X @ dirs.T  # (n, n_proj)
    Yp = Y @ dirs.T  # (m, n_proj)
    vals = np.array([_w1d(Xp[:, i], Yp[:, i], p) for i in range(n_projections)])
    if p == 1:
        sw = float(vals.mean())
    else:
        sw = float((np.mean(vals ** p)) ** (1.0 / p))
    if return_components:
        return sw, vals
    return sw


def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    M = a2 + b2 - 2.0 * (A @ B.T)
    return np.maximum(M, 0.0)


def _ot_eps(A: np.ndarray, B: np.ndarray, eps: float, n_iter: int) -> float:
    """Entropic-OT transport cost between uniform-weighted clouds, log-domain Sinkhorn.

    Marginals a_i=1/n, b_j=1/m; cost = squared Euclidean. Fixed point:
        f_i = -eps * logsumexp_j( log b_j + (g_j - M_ij)/eps )
        g_j = -eps * logsumexp_i( log a_i + (f_i - M_ij)/eps )
    Returns sum_ij P_ij M_ij with P_ij = a_i b_j exp((f_i+g_j-M_ij)/eps).
    """
    from scipy.special import logsumexp
    A = _as2d(A)
    B = _as2d(B)
    n, m = A.shape[0], B.shape[0]
    log_a = -np.log(n)
    log_b = -np.log(m)
    M = _pairwise_sq_dists(A, B)
    f = np.zeros(n)
    g = np.zeros(m)
    for _ in range(n_iter):
        f = -eps * logsumexp((g[None, :] - M) / eps + log_b, axis=1)
        g = -eps * logsumexp((f[:, None] - M) / eps + log_a, axis=0)
    logP = (f[:, None] + g[None, :] - M) / eps + log_a + log_b
    P = np.exp(logP)
    return float((P * M).sum())


def sinkhorn_divergence(X, Y, eps: float | None = None, eps_scale: float = 0.1,
                        n_iter: int = 200) -> float:
    """Debiased Sinkhorn divergence S(X,Y) = OT_e(X,Y) - ½OT_e(X,X) - ½OT_e(Y,Y).

    Debiasing makes S(X,X)=0 and S>=0 (approximately, for finite iterations). If ``eps`` is
    None it is set to ``eps_scale * median`` of the pooled pairwise squared distances and the
    SAME absolute eps is used for all three OT terms (required for the debiasing to cancel).
    Cross-check for sliced-Wasserstein; O(N^2), intended for N in the hundreds-to-low-thousands
    (which is where the matched-N protocol operates).
    """
    X = _as2d(X)
    Y = _as2d(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"dimension mismatch: X has {X.shape[1]}, Y has {Y.shape[1]}")
    if eps is None:
        pooled = np.vstack([X, Y])
        med = np.median(_pairwise_sq_dists(pooled, pooled))
        eps = float(eps_scale * med) if med > 0 else eps_scale
    return (_ot_eps(X, Y, eps, n_iter)
            - 0.5 * _ot_eps(X, X, eps, n_iter)
            - 0.5 * _ot_eps(Y, Y, eps, n_iter))
