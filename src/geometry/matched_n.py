"""Matched-N null-floor protocol for Aim 3 geometry (Phase 1 WS2).

Implements `docs/research_strategy_feasibility_power.md` §4-§6: the rule that **raw distances
are never interpreted** -- every distance is read only relative to a matched-N self-distance
floor (the noise expected when nothing differs, at exactly this N), so the n^(-1/d) small-sample
inflation cancels.

Pipeline:
  1. `rarefaction_curve` + `estimate_n_min` -- empirically fix N_min (where a population's
     geometry stops changing as you add cells).
  2. `self_distance_floor` -- the same-distribution null at N* (disjoint split-half distances).
  3. `cross_distance_ci` -- bootstrapped cross-population distance at N*.
  4. `matched_n_test` -- the verdict: a difference is real only if the cross-distance CI clears
     the floor; convergence (Aim 3) is declared only if it ALSO lies below the Control-3
     maturation-matched baseline. Underpowered / micro-cluster cases (§6) are reported as
     `inconclusive_underpowered`, never as evidence of convergence.
  5. `benjamini_hochberg` -- FDR control across all cross-germ-layer pairs.

Pure numpy/scipy. Distances come from `distances.sliced_wasserstein` (the primary metric).
"""

from __future__ import annotations

import logging

import numpy as np

from .distances import _as2d, sliced_wasserstein


def _two_disjoint(rng: np.random.Generator, n_total: int, n: int):
    if n_total < 2 * n:
        raise ValueError(f"need >= 2*n = {2 * n} cells for disjoint splits, have {n_total}")
    idx = rng.permutation(n_total)
    return idx[:n], idx[n:2 * n]


def split_half_self_distance(X, n: int, n_repeats: int = 200, n_projections: int = 200,
                             random_state: int | None = None) -> np.ndarray:
    """Sliced-W between two DISJOINT n-cell subsamples of X, repeated `n_repeats` times.

    This is the same-distribution null: both halves are drawn from one population, so any
    distance is pure finite-sample noise at this N. Requires len(X) >= 2*n.
    """
    X = _as2d(X)
    rng = np.random.default_rng(random_state)
    out = np.empty(n_repeats)
    for r in range(n_repeats):
        i, j = _two_disjoint(rng, X.shape[0], n)
        out[r] = sliced_wasserstein(X[i], X[j], n_projections=n_projections,
                                    random_state=int(rng.integers(1 << 31)))
    return out


def rarefaction_curve(X, ladder, n_repeats: int = 50, n_projections: int = 200,
                      random_state: int | None = None) -> dict:
    """Mean split-half self-distance at each N in `ladder` (only N with 2*N <= len(X)).

    The curve saturates where adding cells no longer reduces the self-distance; that knee is
    N_min (`estimate_n_min`). Returns {n: {"mean":…, "std":…, "n_repeats":…}}.
    """
    X = _as2d(X)
    N = X.shape[0]
    rng = np.random.default_rng(random_state)
    curve = {}
    for n in sorted(ladder):
        if 2 * n > N:
            continue
        d = split_half_self_distance(X, n, n_repeats=n_repeats, n_projections=n_projections,
                                     random_state=int(rng.integers(1 << 31)))
        curve[int(n)] = {"mean": float(d.mean()), "std": float(d.std()), "n_repeats": n_repeats}
    if not curve:
        raise ValueError(f"no ladder rung satisfies 2*n <= len(X)={N}; population too small")
    return curve


def estimate_n_min(curve_means: dict, rel_tol: float = 0.1) -> int:
    """Smallest ladder N at which the self-distance has saturated.

    `curve_means` maps N -> mean self-distance (e.g. {n: v["mean"] for n,v in curve.items()}).
    Saturation = the first N whose relative drop to the next rung is below `rel_tol` (adding
    cells barely lowers the noise). Falls back to the largest rung if it never saturates.
    """
    ns = sorted(curve_means)
    if len(ns) == 1:
        return ns[0]
    for k in range(len(ns) - 1):
        m0 = curve_means[ns[k]]
        m1 = curve_means[ns[k + 1]]
        if m0 <= 0:
            return ns[k]
        if (m0 - m1) / m0 < rel_tol:
            return ns[k]
    return ns[-1]


def self_distance_floor(X, n_star: int, n_repeats: int = 200, n_projections: int = 200,
                        random_state: int | None = None) -> np.ndarray:
    """The matched-N noise floor at N* -- disjoint split-half distances of X at size n_star."""
    return split_half_self_distance(X, n_star, n_repeats=n_repeats,
                                    n_projections=n_projections, random_state=random_state)


def cross_distance_ci(P, Q, n_star: int, n_boot: int = 1000, ci=(2.5, 97.5),
                      n_projections: int = 200, random_state: int | None = None) -> dict:
    """Bootstrapped cross-population sliced-W at matched N*.

    Each bootstrap draws N* cells WITH replacement from each population (standard bootstrap)
    and computes the distance; returns mean and the `ci` percentile interval. Bootstrapping a
    small population yields a correspondingly WIDE interval -- that width is displayed, not
    suppressed (feasibility_power §6).
    """
    P = _as2d(P)
    Q = _as2d(Q)
    rng = np.random.default_rng(random_state)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pi = rng.choice(P.shape[0], n_star, replace=True)
        qi = rng.choice(Q.shape[0], n_star, replace=True)
        out[b] = sliced_wasserstein(P[pi], Q[qi], n_projections=n_projections,
                                    random_state=int(rng.integers(1 << 31)))
    lo, hi = np.percentile(out, ci)
    return {"mean": float(out.mean()), "lo": float(lo), "hi": float(hi),
            "ci": tuple(ci), "distances": out}


def _perm_pvalue(P, Q, n_star, observed, n_perm, n_projections, rng) -> float:
    """One-sided permutation p-value for 'cross-distance larger than under the same-dist null'."""
    pooled = np.vstack([P, Q])
    nP = P.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled.shape[0])
        A = pooled[perm[:nP]]
        B = pooled[perm[nP:]]
        ai = rng.choice(A.shape[0], n_star, replace=A.shape[0] < n_star)
        bi = rng.choice(B.shape[0], n_star, replace=B.shape[0] < n_star)
        d = sliced_wasserstein(A[ai], B[bi], n_projections=n_projections,
                               random_state=int(rng.integers(1 << 31)))
        if d >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def matched_n_test(P, Q, n_min: int | None = None, n_repeats: int = 200, n_boot: int = 1000,
                   floor_quantile: float = 95.0, n_projections: int = 200,
                   control3_baseline=None, n_permutations: int = 0,
                   random_state: int | None = None) -> dict:
    """Full matched-N verdict for one population pair (feasibility_power §5-§6).

    N* = min(|P|, |Q|). A difference is REAL only if the cross-distance CI lower bound clears
    the self-distance floor; CONVERGENCE (Aim 3) is declared only if the cross-distance CI ALSO
    lies below the Control-3 maturation-matched baseline (passed as an array of distances).

    Verdicts: ``inconclusive_underpowered`` (N* < N_min, or CI overlaps floor) ·
    ``real_difference`` (clears floor; no baseline given) · ``convergent_shared_default``
    (clears floor AND below maturation baseline) · ``lineage_bound_divergent`` (above baseline)
    · ``explained_by_maturation`` (clears floor but indistinguishable from baseline).

    The self-distance floor is built from disjoint splits of the ABUNDANT population at N*
    (§6); if neither population has >= 2*N* cells, the floor subsample shrinks and `floor_n`
    is reported below N* with `floor_reduced=True` (a conservative, slightly inflated floor).
    """
    P = _as2d(P)
    Q = _as2d(Q)
    rng = np.random.default_rng(random_state)
    nP, nQ = P.shape[0], Q.shape[0]
    n_star = min(nP, nQ)
    larger = P if nP >= nQ else Q

    underpowered = (n_min is not None) and (n_star < n_min)

    # Self-distance floor at N* from the abundant population's disjoint splits.
    if larger.shape[0] >= 2 * n_star:
        floor_n = n_star
        floor_reduced = False
    else:
        floor_n = larger.shape[0] // 2
        floor_reduced = True
    floor = self_distance_floor(larger, floor_n, n_repeats=n_repeats,
                                n_projections=n_projections, random_state=int(rng.integers(1 << 31)))
    floor_q = float(np.percentile(floor, floor_quantile))

    cross = cross_distance_ci(P, Q, n_star, n_boot=n_boot, n_projections=n_projections,
                              random_state=int(rng.integers(1 << 31)))

    p_value = None
    if n_permutations > 0:
        p_value = _perm_pvalue(P, Q, n_star, cross["mean"], n_permutations, n_projections, rng)

    clears_floor = cross["lo"] > floor_q
    result = {
        "n_p": nP, "n_q": nQ, "n_star": n_star, "n_min": n_min, "underpowered": underpowered,
        "floor_n": floor_n, "floor_reduced": floor_reduced,
        "floor_mean": float(floor.mean()), f"floor_q{int(floor_quantile)}": floor_q,
        "cross_mean": cross["mean"], "cross_lo": cross["lo"], "cross_hi": cross["hi"],
        "clears_floor": bool(clears_floor), "p_value": p_value,
    }

    if underpowered or not clears_floor:
        result["verdict"] = "inconclusive_underpowered"
        return result

    if control3_baseline is None:
        result["verdict"] = "real_difference"
        return result

    base = np.asarray(control3_baseline, dtype=float)
    base_lo, base_hi = np.percentile(base, [2.5, 97.5])
    result["baseline_lo"], result["baseline_hi"] = float(base_lo), float(base_hi)
    if cross["hi"] < base_lo:
        result["verdict"] = "convergent_shared_default"      # below maturation baseline
    elif cross["lo"] > base_hi:
        result["verdict"] = "lineage_bound_divergent"        # above maturation baseline
    else:
        result["verdict"] = "explained_by_maturation"
    return result


def benjamini_hochberg(pvalues, alpha: float = 0.05):
    """Benjamini-Hochberg FDR across all cross-germ-layer pairs. Returns (rejected, qvalues)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.shape[0]
    if m == 0:
        return np.array([], dtype=bool), np.array([])
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qvalues = np.empty(m)
    qvalues[order] = np.clip(ranked, 0.0, 1.0)
    return qvalues <= alpha, qvalues
