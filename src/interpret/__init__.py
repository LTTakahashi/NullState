"""Interpretation layer (Aim 3 / WS3): pure post-processing helpers.

The heavy compute (CellOracle GRN perturbation; AUCell/decoupler program scoring) is data/GPU-bound
and belongs in the run scripts; these helpers rank its outputs. Import with numpy/pandas/scipy only
(see `docs/ws3_sketch.md`).
"""

from .perturbation import rank_perturbation_targets
from .signatures import differential_program_scores

__all__ = ["rank_perturbation_targets", "differential_program_scores"]
