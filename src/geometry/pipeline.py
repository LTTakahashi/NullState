"""End-to-end Aim 3 geometry over labeled populations (Phase 1 WS2 wiring).

`run_geometry_over_populations` is the pure (numpy/scipy) core that turns a disentangled
z_lin matrix + per-cell population labels into the full Aim 3 result: empirical N_min from
rarefaction, every pairwise matched-N test, optional Control-3 verdicts, and BH-FDR across
pairs. `scripts/run_aim3.py` is the thin I/O wrapper that obtains z_lin (from a trained
contrastiveVI model, or a precomputed .npy) and calls this.
"""

from __future__ import annotations

import logging

import numpy as np

from .distances import _as2d
from .matched_n import rarefaction_curve, estimate_n_min, matched_n_test, benjamini_hochberg


def run_geometry_over_populations(z, labels, n_min=None, ladder=None, control3=None,
                                  n_repeats: int = 200, n_boot: int = 1000,
                                  n_projections: int = 200, floor_quantile: float = 95.0,
                                  n_permutations: int = 0, n_rarefaction_repeats: int = 50,
                                  random_state: int | None = None) -> dict:
    """Run the matched-N geometry across all pairs of labeled populations in z_lin.

    Parameters
    ----------
    z : (n_cells, n_dims) disentangled lineage representation (contrastiveVI background latent).
    labels : (n_cells,) population id per cell (e.g. germ-layer x off-target-class).
    n_min : fixed N_min; if None and `ladder` is given, it is estimated by rarefaction on the
            largest population.
    control3 : optional maturation baselines (Control 3) keyed by `frozenset({a, b})` or `(a, b)`;
               each value is an array of maturation-matched-primary distances. When present, a
               cleared-floor pair can be called convergent / lineage-bound / maturation-explained.

    Returns {n_min, sizes, pairs: {(a,b): matched_n_test result}, fdr: {(a,b): {rejected, q}}}.
    """
    z = _as2d(z)
    labels = np.asarray(labels)
    if labels.shape[0] != z.shape[0]:
        raise ValueError(f"labels ({labels.shape[0]}) and z ({z.shape[0]}) length mismatch")

    uniq = list(np.unique(labels))
    if len(uniq) < 2:
        raise ValueError("need >= 2 distinct populations to compare")
    pops = {lab: z[labels == lab] for lab in uniq}
    sizes = {str(lab): int(pops[lab].shape[0]) for lab in uniq}
    rng = np.random.default_rng(random_state)

    if n_min is None and ladder is not None:
        largest = max(uniq, key=lambda l: pops[l].shape[0])
        curve = rarefaction_curve(pops[largest], ladder, n_repeats=n_rarefaction_repeats,
                                  n_projections=n_projections,
                                  random_state=int(rng.integers(1 << 31)))
        n_min = estimate_n_min({n: v["mean"] for n, v in curve.items()})
        logging.info("Empirical N_min from rarefaction on '%s': %s", largest, n_min)

    def _baseline(a, b):
        if control3 is None:
            return None
        # Explicit lookup (no `or` chaining: the values are arrays, whose truth value is ambiguous).
        for key in (frozenset({a, b}), (a, b), (b, a)):
            if key in control3:
                return control3[key]
        return None

    results = {}
    pvals, keys = [], []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            a, b = uniq[i], uniq[j]
            res = matched_n_test(
                pops[a], pops[b], n_min=n_min, n_repeats=n_repeats, n_boot=n_boot,
                floor_quantile=floor_quantile, n_projections=n_projections,
                control3_baseline=_baseline(a, b), n_permutations=n_permutations,
                random_state=int(rng.integers(1 << 31)),
            )
            results[(str(a), str(b))] = res
            if res.get("p_value") is not None:
                pvals.append(res["p_value"])
                keys.append((str(a), str(b)))

    fdr = None
    if pvals:
        rejected, q = benjamini_hochberg(pvals)
        fdr = {keys[k]: {"rejected": bool(rejected[k]), "q": float(q[k])} for k in range(len(keys))}

    return {"n_min": n_min, "sizes": sizes, "pairs": results, "fdr": fdr}
