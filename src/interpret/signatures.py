"""Differential program scoring (Aim 3 / WS3, Part B) -- pure post-processing.

The testable seam of the in-vivo program anchoring step (`docs/ws3_sketch.md`): the heavy AUCell/
decoupler scoring of cells against gene programs (MSigDB Hallmark, dev/stress signatures) is
data-bound and lives in the run script; this module takes the resulting cell x program score
matrix and ranks which programs are elevated in the convergent off-target cells versus a
reference population -- giving the abstract default state a biological identity.

Pure numpy/pandas/scipy. No decoupler/expression dependency here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .perturbation import _bh_with_nan


def differential_program_scores(scores, labels, group, reference=None, alpha: float = 0.05) -> pd.DataFrame:
    """Rank gene programs by enrichment in the `group` (convergent) cells vs a `reference`.

    Parameters
    ----------
    scores : (n_cells x n_programs) DataFrame of per-cell program scores (e.g. AUCell/decoupler).
    labels : 1-D per-cell labels aligned to `scores` rows (positional).
    group : label of the convergent population.
    reference : label, list of labels, or None (default = all cells not in `group`).
    alpha : FDR level for the `significant` flag.

    Returns
    -------
    DataFrame indexed by program with: mean_group, mean_reference, mean_diff (group-reference),
    auc (common-language effect = P(group > reference)), n_group, n_reference, p_value
    (Mann-Whitney U, two-sided), q_value (BH), significant, rank. Sorted by mean_diff descending
    (programs most elevated in the convergent state first).
    """
    from scipy.stats import mannwhitneyu

    if not isinstance(scores, pd.DataFrame):
        raise TypeError("scores must be a (cells x programs) DataFrame")
    labels = np.asarray(labels)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(f"labels ({labels.shape[0]}) and scores rows ({scores.shape[0]}) mismatch")

    group_mask = labels == group
    if reference is None:
        ref_mask = ~group_mask
    elif np.ndim(reference) == 0:
        ref_mask = labels == reference
    else:
        ref_mask = np.isin(labels, list(reference))
    if int(group_mask.sum()) == 0 or int(ref_mask.sum()) == 0:
        raise ValueError(f"empty group ({int(group_mask.sum())}) or reference ({int(ref_mask.sum())}) set")

    X = scores.to_numpy(dtype=float)
    rows = []
    for j, prog in enumerate(scores.columns):
        a = X[group_mask, j]
        b = X[ref_mask, j]
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if a.size == 0 or b.size == 0:
            p_value, auc, mean_diff = np.nan, np.nan, np.nan
        else:
            try:
                res = mannwhitneyu(a, b, alternative="two-sided")
                p_value = float(res.pvalue)
                auc = float(res.statistic) / (a.size * b.size)   # P(group > reference)
            except ValueError:                                    # all values identical
                p_value, auc = 1.0, 0.5
            mean_diff = float(np.mean(a) - np.mean(b))
        rows.append({
            "program": prog,
            "mean_group": float(np.mean(a)) if a.size else np.nan,
            "mean_reference": float(np.mean(b)) if b.size else np.nan,
            "mean_diff": mean_diff,
            "auc": auc,
            "n_group": int(a.size),
            "n_reference": int(b.size),
            "p_value": p_value,
        })

    out = pd.DataFrame(rows).set_index("program")
    out["q_value"] = _bh_with_nan(out["p_value"].to_numpy())
    out["significant"] = out["q_value"] <= alpha
    out = out.sort_values("mean_diff", ascending=False, kind="mergesort")
    out["rank"] = np.arange(1, len(out) + 1)
    return out
