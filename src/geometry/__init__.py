"""Aim 3 geometry (Phase 1 WS2): the measurement layer.

Dimension-robust OT distances (`distances`), the matched-N null-floor protocol that makes
every distance self-calibrating (`matched_n`), and simulation-based power (`power`). Pure
numpy/scipy -- no scvi, no GPU. Runs on the disentangled z_lin from `src.disentangle`.

The one non-negotiable rule (feasibility_power §2): raw distances are never interpreted on
their own -- only relative to the matched-N self-distance floor and the Control-3 baseline.
"""

from .distances import sliced_wasserstein, sinkhorn_divergence
from .matched_n import (
    split_half_self_distance,
    rarefaction_curve,
    estimate_n_min,
    self_distance_floor,
    cross_distance_ci,
    matched_n_test,
    benjamini_hochberg,
)
from .power import (
    simulate_shifted_pair,
    detection_power,
    power_curve,
    min_detectable_effect,
)
from .topology import (
    h0_persistence,
    fasy_confidence_band,
    significant_h0_features,
    h0_confirmatory_test,
    h1_persistence,
    h1_knn_sensitivity,
)
from .pipeline import run_geometry_over_populations

__all__ = [
    "sliced_wasserstein",
    "sinkhorn_divergence",
    "split_half_self_distance",
    "rarefaction_curve",
    "estimate_n_min",
    "self_distance_floor",
    "cross_distance_ci",
    "matched_n_test",
    "benjamini_hochberg",
    "simulate_shifted_pair",
    "detection_power",
    "power_curve",
    "min_detectable_effect",
    "h0_persistence",
    "fasy_confidence_band",
    "significant_h0_features",
    "h0_confirmatory_test",
    "h1_persistence",
    "h1_knn_sensitivity",
    "run_geometry_over_populations",
]
