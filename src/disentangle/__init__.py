"""Disentanglement (Aim 2): contrastiveVI dish-stripping of the in-vitro signature.

contrastive.py imports its heavy deps (scvi/anndata/sklearn) lazily inside function
bodies, so importing this package -- and the pure `select_contrastive_indices` helper --
works with only numpy installed (mirrors the scvi-free import contract used in src.mapping).
"""

from .contrastive import (
    select_contrastive_indices,
    build_contrastive_adata,
    train_contrastive_vi,
    get_lineage_latent,
    get_invitro_latent,
    verify_disentanglement,
)

__all__ = [
    "select_contrastive_indices",
    "build_contrastive_adata",
    "train_contrastive_vi",
    "get_lineage_latent",
    "get_invitro_latent",
    "verify_disentanglement",
]
