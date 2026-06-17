"""scANVI mapping, scArches surgery, and off-target classification modules.

The pure-Python classifier is imported eagerly so that
``from src.mapping.classify import classify_cell`` (and the unit tests) work without the
deep-learning stack installed. The scvi-tools-backed entry points are re-exported only
when ``scvi`` is importable, so reaching the pure classifier never *requires* scvi/torch.
"""

from .classify import classify_all, classify_cell, validate_classification

__all__ = ["classify_all", "classify_cell", "validate_classification"]

try:  # scvi-tools-backed entry points are optional at import time (e.g. CPU/test env)
    from .train_scanvi import run_training_pipeline
    from .compute_scores import annotate_query, build_empirical_label_prior
    __all__ += ["run_training_pipeline", "annotate_query", "build_empirical_label_prior"]
except ModuleNotFoundError:
    # scvi/torch not installed: pure classification + tests still import fine.
    pass
