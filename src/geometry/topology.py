"""Persistent homology for Aim 3 geometry (Phase 1 WS2).

Per `docs/research_strategy_feasibility_power.md` §3: **H0 persistence with Fasy-style
bootstrap confidence bands is CONFIRMATORY; H1 is EXPLORATORY only** (its features are
sampling-density artifacts at the available N, so it is sensitivity-tested, not trusted).

Key implementation choice: H0 of a Vietoris-Rips filtration is exactly single-linkage
merging, so the H0 persistence diagram = the Euclidean **minimum spanning tree edge weights**
(births at 0, deaths at the merge distances). That means the confirmatory H0 path needs **no
TDA dependency** -- pure numpy/scipy -- and is fully unit-tested here. Only the exploratory H1
path imports `ripser`, lazily, so the module imports without it.

Fasy et al. (2014) confidence band: bootstrap the cloud, compute a diagram-distance to the
original, take the (1-alpha) quantile c; features with persistence > 2c lie above the band and
are significant (everything within 2c of the diagonal is noise). For H0 (births=0) we use the
sorted-persistence L-infinity matching as the diagram distance.
"""

from __future__ import annotations

import numpy as np

from .distances import _as2d


def h0_persistence(X) -> np.ndarray:
    """H0 persistence (finite features) = Euclidean MST edge weights, ascending.

    Births are all 0, so each value is also the feature's persistence. There are n-1 finite
    features (plus one essential component that never dies, omitted). Pure scipy. O(n^2) memory.
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree

    X = _as2d(X)
    n = X.shape[0]
    if n < 2:
        return np.empty(0)
    D = squareform(pdist(X))
    mst = minimum_spanning_tree(D)          # zeros treated as no-edge; diagonal ignored
    deaths = np.asarray(mst.data, dtype=float)
    return np.sort(deaths)


def _h0_diagram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L-infinity distance between two H0 diagrams (births=0): match sorted persistences,
    pad the shorter with zeros (the diagonal), take the max absolute difference."""
    a = np.sort(np.asarray(a, dtype=float))[::-1]
    b = np.sort(np.asarray(b, dtype=float))[::-1]
    L = max(a.size, b.size)
    aa = np.zeros(L)
    bb = np.zeros(L)
    aa[:a.size] = a
    bb[:b.size] = b
    return float(np.max(np.abs(aa - bb))) if L else 0.0


def fasy_confidence_band(X, alpha: float = 0.05, n_bootstrap: int = 1000,
                         subsample: int | None = None, random_state: int | None = None) -> float:
    """Fasy-style bootstrap confidence value c for the H0 diagram.

    Resamples the cloud (size `subsample` or n, with replacement) `n_bootstrap` times, measures
    the H0 diagram distance to the original each time, and returns the (1-alpha) quantile c.
    The significance band is 2c (see `significant_h0_features`).
    """
    X = _as2d(X)
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    base = h0_persistence(X)
    m = int(subsample) if subsample else n
    stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, m, replace=True)
        stats[b] = _h0_diagram_distance(base, h0_persistence(X[idx]))
    return float(np.quantile(stats, 1.0 - alpha))


def significant_h0_features(persistences, c: float, multiplier: float = 2.0) -> dict:
    """Features whose persistence exceeds the band (``multiplier * c``) are significant clusters."""
    p = np.asarray(persistences, dtype=float)
    band = multiplier * c
    sig = np.sort(p[p > band])[::-1]
    return {"band": band, "n_significant": int(sig.size),
            "significant_persistences": sig}


def h0_confirmatory_test(X, alpha: float = 0.05, n_bootstrap: int = 1000,
                         subsample: int | None = None, random_state: int | None = None) -> dict:
    """Confirmatory H0 analysis: persistence diagram + Fasy band + significant-feature count.

    `n_significant` is the number of robust connected-component features (e.g. distinct cluster
    cores) that survive the bootstrap band -- the topological signal that is robust to batch
    noise (feasibility_power §3).
    """
    pers = h0_persistence(X)
    c = fasy_confidence_band(X, alpha=alpha, n_bootstrap=n_bootstrap,
                             subsample=subsample, random_state=random_state)
    sig = significant_h0_features(pers, c)
    return {
        "n_points": int(_as2d(X).shape[0]),
        "n_features": int(pers.size),
        "max_persistence": float(pers.max()) if pers.size else 0.0,
        "confidence_c": c,
        "alpha": alpha,
        **sig,
    }


# ----------------------- H1: EXPLORATORY ONLY (needs ripser) -----------------------

def _require_ripser():
    try:
        from ripser import ripser
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "H1 persistence is exploratory and needs the optional 'ripser' package "
            "(`pip install ripser`). H0 (the confirmatory path) needs no extra dependency."
        ) from e
    return ripser


def h1_persistence(X, thresh: float = np.inf, coeff: int = 2) -> np.ndarray:
    """EXPLORATORY H1 (loops) birth-death pairs via Vietoris-Rips (ripser). Treat features as
    hypotheses only -- at the available N they may be sampling-density artifacts."""
    ripser = _require_ripser()
    X = _as2d(X)
    dgms = ripser(X, maxdim=1, thresh=thresh, coeff=coeff)["dgms"]
    return dgms[1] if len(dgms) > 1 else np.empty((0, 2))


def h1_knn_sensitivity(X, thresholds, coeff: int = 2) -> dict:
    """EXPLORATORY: H1 feature count across Rips thresholds (a proxy for k-NN/density
    sensitivity). Stable counts across thresholds are less likely to be artifacts; volatile
    counts flag the feature as a sampling artifact. Returns {threshold: n_H1_features}."""
    out = {}
    for t in thresholds:
        out[float(t)] = int(h1_persistence(X, thresh=float(t), coeff=coeff).shape[0])
    return out
