"""Rank in-silico perturbation targets (Aim 3 / WS3, Part A) -- pure post-processing.

This is the testable seam of the CellOracle step (`docs/ws3_sketch.md`): CellOracle does the
heavy GRN inference + perturbation simulation (data/GPU-bound, lazy-imported elsewhere); this
module takes its per-cell, per-TF shift scores and ranks the TFs whose perturbation moves the
convergent off-target cells OUT of the default state (the candidate corrective levers).

Sign convention: a shift score is the cell's predicted movement under perturbing one TF,
projected onto the "rescue" direction (positive = away from the convergent default, toward
on-target). Pure numpy/pandas/scipy. No CellOracle dependency here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.geometry.matched_n import benjamini_hochberg


def _as_frame(shift_scores) -> pd.DataFrame:
    """Accept a (cells x TFs) DataFrame or a {tf: per-cell 1-D array} dict -> DataFrame."""
    if isinstance(shift_scores, pd.DataFrame):
        return shift_scores
    if isinstance(shift_scores, dict):
        if not shift_scores:
            raise ValueError("shift_scores dict is empty")
        lengths = {len(np.asarray(v).ravel()) for v in shift_scores.values()}
        if len(lengths) != 1:
            raise ValueError(f"per-TF score arrays have unequal lengths: {lengths}")
        return pd.DataFrame({tf: np.asarray(v, dtype=float).ravel() for tf, v in shift_scores.items()})
    raise TypeError("shift_scores must be a DataFrame or {tf: array} dict")


def _bh_with_nan(pvals: np.ndarray) -> np.ndarray:
    """BH q-values, propagating NaN p-values (e.g. TFs with too few cells) as NaN."""
    pvals = np.asarray(pvals, dtype=float)
    q = np.full(pvals.shape, np.nan)
    mask = ~np.isnan(pvals)
    if mask.any():
        _, qv = benjamini_hochberg(pvals[mask])
        q[mask] = qv
    return q


def rank_perturbation_targets(shift_scores, ascending: bool = False, alpha: float = 0.05) -> pd.DataFrame:
    """Rank candidate TFs by their mean per-cell rescue shift across the convergent cells.

    Parameters
    ----------
    shift_scores : (n_cells x n_TFs) DataFrame, or {tf: per-cell 1-D array}. Entries are
        rescue-projected shift scores (positive = moves the cell out of the convergent default).
    ascending : if False (default) rank by descending mean shift (most-rescuing TFs first);
        set True to surface TFs whose perturbation drives cells deeper into the default.
    alpha : FDR level for the `significant` flag.

    Returns
    -------
    DataFrame indexed by TF with columns: mean_shift, std_shift, frac_positive, n_cells,
    t_stat, p_value (one-sample t-test of per-cell shifts vs 0), q_value (BH), significant,
    rank. Sorted by mean_shift (per `ascending`).
    """
    from scipy.stats import ttest_1samp

    df = _as_frame(shift_scores)
    rows = []
    for tf in df.columns:
        x = df[tf].to_numpy(dtype=float)
        x = x[~np.isnan(x)]
        if x.size >= 2 and np.any(x != x[0]):
            res = ttest_1samp(x, 0.0)
            t_stat, p_value = float(res.statistic), float(res.pvalue)
        else:
            t_stat, p_value = np.nan, np.nan
        rows.append({
            "tf": tf,
            "mean_shift": float(np.mean(x)) if x.size else np.nan,
            "std_shift": float(np.std(x, ddof=1)) if x.size >= 2 else np.nan,
            "frac_positive": float(np.mean(x > 0)) if x.size else np.nan,
            "n_cells": int(x.size),
            "t_stat": t_stat,
            "p_value": p_value,
        })
    out = pd.DataFrame(rows).set_index("tf")
    out["q_value"] = _bh_with_nan(out["p_value"].to_numpy())
    out["significant"] = out["q_value"] <= alpha
    out = out.sort_values("mean_shift", ascending=ascending, kind="mergesort")
    out["rank"] = np.arange(1, len(out) + 1)
    return out
