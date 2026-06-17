"""Simulation-based power analysis for Aim 3 geometry (Phase 1 WS2).

No closed-form power exists for a permutation test on sliced-Wasserstein distances
(`docs/research_strategy_feasibility_power.md` §7), so power is established by simulation.
From an abundant population (e.g. brain-derived mesenchyme z_lin) we synthesize pairs with a
KNOWN separation delta at sample size N, then measure how often the matched-N rule
(cross-distance exceeding the self-distance floor's upper quantile) correctly fires. This
yields power curves N(delta) and the minimum detectable effect at each available N -- so every
real comparison can be reported alongside what it could and could not have found.

Detection rule used here = cross-distance point estimate > floor upper-quantile (a fast,
standard approximation of the CI-vs-floor decision in `matched_n.matched_n_test`). At delta=0
the firing rate equals 1 - floor_quantile/100 (the calibrated false-positive rate). Pure
numpy/scipy.
"""

from __future__ import annotations

import numpy as np

from .distances import _as2d, sliced_wasserstein
from .matched_n import self_distance_floor


def simulate_shifted_pair(base, delta: float, n: int, rng: np.random.Generator,
                          direction=None):
    """Two N-cell samples from `base`; the second is mean-shifted by `delta` along a unit
    direction, giving a known separation (delta=0 => same distribution)."""
    base = _as2d(base)
    d = base.shape[1]
    if direction is None:
        u = rng.standard_normal(d)
    else:
        u = np.asarray(direction, dtype=float)
    u = u / (np.linalg.norm(u) + 1e-12)
    P = base[rng.choice(base.shape[0], n, replace=True)]
    Q = base[rng.choice(base.shape[0], n, replace=True)] + delta * u
    return P, Q


def detection_power(base, delta: float, n: int, n_trials: int = 100,
                    floor_quantile: float = 95.0, n_floor_repeats: int = 100,
                    n_projections: int = 100, random_state: int | None = None) -> float:
    """Fraction of trials in which a separation of size `delta` at sample size `n` clears the
    matched-N self-distance floor built from `base`. delta=0 estimates the false-positive rate
    (~ 1 - floor_quantile/100)."""
    base = _as2d(base)
    rng = np.random.default_rng(random_state)
    if 2 * n > base.shape[0]:
        raise ValueError(f"need >= 2*n = {2 * n} cells in `base` to build the floor at N={n}")
    floor = self_distance_floor(base, n, n_repeats=n_floor_repeats,
                                n_projections=n_projections, random_state=int(rng.integers(1 << 31)))
    thr = float(np.percentile(floor, floor_quantile))
    hits = 0
    for _ in range(n_trials):
        P, Q = simulate_shifted_pair(base, delta, n, rng)
        d = sliced_wasserstein(P, Q, n_projections=n_projections,
                               random_state=int(rng.integers(1 << 31)))
        if d > thr:
            hits += 1
    return hits / n_trials


def power_curve(base, deltas, sample_sizes, n_trials: int = 100, floor_quantile: float = 95.0,
                n_floor_repeats: int = 100, n_projections: int = 100,
                random_state: int | None = None) -> dict:
    """Detection power over the (N, delta) grid. Returns {(n, delta): power}."""
    base = _as2d(base)
    rng = np.random.default_rng(random_state)
    grid = {}
    for n in sample_sizes:
        for delta in deltas:
            grid[(int(n), float(delta))] = detection_power(
                base, float(delta), int(n), n_trials=n_trials, floor_quantile=floor_quantile,
                n_floor_repeats=n_floor_repeats, n_projections=n_projections,
                random_state=int(rng.integers(1 << 31)),
            )
    return grid


def min_detectable_effect(grid: dict, sample_sizes, deltas, target_power: float = 0.8) -> dict:
    """For each N, the smallest delta whose power reaches `target_power` (None if none does)."""
    out = {}
    for n in sample_sizes:
        mde = None
        for delta in sorted(deltas):
            if grid.get((int(n), float(delta)), 0.0) >= target_power:
                mde = float(delta)
                break
        out[int(n)] = mde
    return out
