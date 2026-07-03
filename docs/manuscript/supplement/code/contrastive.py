"""contrastiveVI dish-stripping (Aim 2 / Phase 1 WS1) -- SCAFFOLD.

PRIMARY disentanglement method for Aim 2 (per the reconciled wording; see
`docs/aim2_method_reconciliation.md`). A contrastive latent-variable model separates an
in-vitro (dish) signature from lineage biology WITHOUT anchoring off-target cells to a
primary identity ("calibration without anchoring").

Mapping to contrastiveVI's two latent spaces (Weinberger et al., Nat Methods 2023):

    background cells = PRIMARY on-target cells   (the "control" lacking the dish artifact)
    target cells     = ORGANOID on-target cells  (carry the extra in-vitro variation)

      -> background (shared) latent = z_lin : lineage biology, dish-stripped
                                              [this is what Aim 3 geometry runs on]
      -> salient latent             = z_iv  : the in-vitro / culture signature

For an off-target ORGANOID cell we read the BACKGROUND representation -> z_lin, with the
culture component routed into the (target-only) salient space. The in-vitro axis is thus
learned from identity-matched on-target cells and applied to off-target cells without
imposing a primary anchor. The origin-predicting adversary is retained ONLY as a
verification check (`verify_disentanglement`), never as the training mechanism.

Because the pilot's Gate B was GO (mean dish-vector cosine 0.906 > 0.70), a SINGLE shared
z_iv is justified; the conditional z_iv|y_id remains the documented fallback.

=========================  STATUS: SCAFFOLD -- NOT YET RUN  =========================
This module has NOT been executed. It requires scvi-tools + a GPU + the harmonized
reference (primary) and the HNOCA query (organoid). Treat thresholds/dims as design
choices, not validated results. `scvi`/`anndata`/`scanpy`/`sklearn` are imported lazily
inside the functions that need them, so the pure helpers (`select_contrastive_indices`)
import -- and unit-test -- with only numpy installed.

Intended call sequence (on a GPU box):

    from src.disentangle.contrastive import (
        build_contrastive_adata, select_contrastive_indices,
        train_contrastive_vi, get_lineage_latent, verify_disentanglement,
    )
    adata = build_contrastive_adata(primary, organoid)            # one AnnData, shared genes
    bg, tg = select_contrastive_indices(                          # numpy-only, pure
        is_primary=adata.obs["cs_source"].eq("primary").values,
        is_organoid=adata.obs["cs_source"].eq("organoid").values,
        is_ontarget=adata.obs["cs_ontarget"].values,
    )
    model = train_contrastive_vi(adata, bg, tg, n_background_latent=15)
    z_lin = get_lineage_latent(model, adata)                      # -> Aim 3 geometry
    report = verify_disentanglement(z_lin, adata.obs["cs_source"].eq("organoid").values)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # heavy deps are lazy-imported in function bodies, not at import time
    import anndata as ad


def select_contrastive_indices(is_primary, is_organoid, is_ontarget):
    """Return ``(background_indices, target_indices)`` as int position arrays.

    background = primary AND on-target cells; target = organoid AND on-target cells.
    Pure (numpy only) so it is unit-testable without scvi/anndata.

    Parameters
    ----------
    is_primary, is_organoid, is_ontarget : 1-D boolean array-likes of equal length,
        aligned to the rows of the combined contrastive AnnData.

    Raises
    ------
    ValueError if lengths mismatch, masks overlap (a cell marked both primary and
    organoid), or either resulting set is empty.
    """
    is_primary = np.asarray(is_primary, dtype=bool)
    is_organoid = np.asarray(is_organoid, dtype=bool)
    is_ontarget = np.asarray(is_ontarget, dtype=bool)

    if not (is_primary.shape == is_organoid.shape == is_ontarget.shape):
        raise ValueError(
            f"mask length mismatch: primary={is_primary.shape}, "
            f"organoid={is_organoid.shape}, ontarget={is_ontarget.shape}"
        )
    if np.any(is_primary & is_organoid):
        raise ValueError("a cell is marked both primary and organoid; sources must be disjoint")

    background = np.flatnonzero(is_primary & is_ontarget)
    target = np.flatnonzero(is_organoid & is_ontarget)

    if background.size == 0 or target.size == 0:
        raise ValueError(
            f"empty contrastive set: background(primary on-target)={background.size}, "
            f"target(organoid on-target)={target.size}. contrastiveVI needs both."
        )
    return background, target


def _materialize_counts(adata, counts_layer: str = "counts", name: str = ""):
    """Ensure ``adata.layers[counts_layer]`` holds integer raw counts.

    contrastiveVI's NB/ZINB likelihood is mis-specified on normalized data, so we source the
    counts (in order) from an existing integer count layer, an integer ``X``, or ``raw.X``
    (reindexed onto the current var axis -- the CELLxGENE organoid keeps counts there). Raise
    if none are integer rather than silently training on log-norm values.
    """
    import numpy as np
    import scipy.sparse as sp

    def _intfrac(X):
        d = X.data if sp.issparse(X) else np.asarray(X).ravel()
        d = d[:200000]
        return float(np.mean(d == np.round(d))) if d.size else 1.0

    layers = getattr(adata, "layers", {})
    if counts_layer in layers and _intfrac(adata.layers[counts_layer]) > 0.999:
        return adata
    if _intfrac(adata.X) > 0.999 and float(adata.X.max()) > 30:
        adata.layers[counts_layer] = adata.X.copy()
        return adata
    raw = getattr(adata, "raw", None)
    if raw is not None:
        rn = list(map(str, raw.var_names))
        vn = list(map(str, adata.var_names))
        if rn == vn:
            rx = raw.X
        else:
            pos = {g: i for i, g in enumerate(rn)}
            cols = [pos[g] for g in vn if g in pos]
            rx = raw.X[:, cols] if len(cols) == adata.n_vars else None
        if rx is not None and _intfrac(rx) > 0.999:
            adata.layers[counts_layer] = rx.copy()
            return adata
    raise ValueError(f"[{name}] no integer raw counts (layer/X/raw.X) for contrastiveVI; "
                     f"the NB/ZINB likelihood requires counts, not normalized data.")


def build_contrastive_adata(primary, organoid, counts_layer: str = "counts",
                            source_key: str = "cs_source", ontarget_key: str = "cs_ontarget",
                            primary_ontarget=None, organoid_ontarget=None):
    """Concatenate primary (background) + organoid (target) into one AnnData for contrastiveVI.

    Lazy-imports anndata. Genes are inner-joined; both inputs must carry raw integer counts
    in ``layers[counts_layer]`` (HEOCA and the CELLxGENE HNOCA copy do; the Zenodo cleaned
    HNOCA does not -- see ``docs/ws0a_data_sourcing.md``). Adds:
      obs[source_key]   : "primary" / "organoid"
      obs[ontarget_key] : bool on-target flag (drives select_contrastive_indices)

    ``primary_ontarget`` / ``organoid_ontarget`` are boolean masks (or obs-column names) marking
    on-target cells in each input. For the neural pilot, organoid on-target = pilot_class in
    {clean_ontarget, poorly_diff_ontarget}; primary on-target = the matched primary lineages.
    """
    import anndata as ad

    def _ontarget_mask(adata, spec):
        if spec is None:
            return np.ones(adata.n_obs, dtype=bool)
        if isinstance(spec, str):
            return np.asarray(adata.obs[spec].values, dtype=bool)
        return np.asarray(spec, dtype=bool)

    primary = primary.copy()
    organoid = organoid.copy()
    primary.obs[source_key] = "primary"
    organoid.obs[source_key] = "organoid"
    primary.obs[ontarget_key] = _ontarget_mask(primary, primary_ontarget)
    organoid.obs[ontarget_key] = _ontarget_mask(organoid, organoid_ontarget)

    # Materialize integer counts per input BEFORE concat (organoid counts live in raw.X; primary
    # counts in X). join="inner" then aligns the counts layer to the shared genes.
    primary = _materialize_counts(primary, counts_layer, name="primary")
    organoid = _materialize_counts(organoid, counts_layer, name="organoid")

    combined = ad.concat([primary, organoid], join="inner", label="cs_concat",
                         keys=["primary", "organoid"])
    logging.info(
        "Built contrastive AnnData: %d cells (%d primary / %d organoid), %d shared genes.",
        combined.n_obs, primary.n_obs, organoid.n_obs, combined.n_vars,
    )
    if counts_layer not in combined.layers:
        raise ValueError(
            f"layers['{counts_layer}'] missing after concat -- contrastiveVI needs raw counts. "
            f"(Both inputs had counts materialized; a join mismatch dropped the layer.)")
    return combined


def train_contrastive_vi(adata, background_indices, target_indices, *,
                         n_background_latent: int = 15, n_salient_latent: int = 10,
                         batch_key: str | None = None, layer: str = "counts",
                         max_epochs: int = 200, early_stopping: bool = True,
                         early_stopping_patience: int = 15):
    """Train scvi.external.ContrastiveVI. Lazy-imports scvi.

    background_indices = primary on-target (from select_contrastive_indices),
    target_indices     = organoid on-target. n_background_latent is z_lin's dimensionality
    (15 sits in the 15-30 band the Aim-3 geometry expects); n_salient_latent is z_iv's.
    """
    import scvi

    scvi.external.ContrastiveVI.setup_anndata(adata, layer=layer, batch_key=batch_key)
    model = scvi.external.ContrastiveVI(
        adata, n_background_latent=n_background_latent, n_salient_latent=n_salient_latent,
    )
    train_kwargs = {"max_epochs": max_epochs}
    if early_stopping:
        # Standard scvi train kwargs; mirrors the scANVI/scArches early-stopping in this repo.
        train_kwargs.update(early_stopping=True, early_stopping_patience=early_stopping_patience,
                            check_val_every_n_epoch=1)
    logging.info("Training ContrastiveVI (bg=%d, target=%d, z_lin=%d, z_iv=%d, max_epochs=%d)...",
                 len(background_indices), len(target_indices), n_background_latent,
                 n_salient_latent, max_epochs)
    model.train(background_indices, target_indices, **train_kwargs)
    return model


def get_lineage_latent(model, adata=None, indices=None):
    """z_lin -- the dish-stripped lineage representation = contrastiveVI BACKGROUND (shared)
    latent. This is the representation Aim 3's geometry (sliced-Wasserstein etc.) runs on."""
    return model.get_latent_representation(adata=adata, indices=indices,
                                           representation_kind="background")


def get_invitro_latent(model, adata=None, indices=None):
    """z_iv -- the in-vitro / culture signature = contrastiveVI SALIENT latent."""
    return model.get_latent_representation(adata=adata, indices=indices,
                                           representation_kind="salient")


def verify_disentanglement(z_lin, is_organoid, n_splits: int = 5, random_state: int = 0,
                           pass_threshold: float = 0.6):
    """Adversary VERIFICATION check (NOT a training mechanism).

    Can a classifier recover organoid-vs-primary origin from z_lin? If balanced accuracy is
    near chance (0.5), the dish signal has been removed from the lineage subspace and
    disentanglement is confirmed; if it is high, z_lin still carries the in-vitro confound.
    Lazy-imports sklearn. This is the demoted adversary from the reconciled Aim 2.

    Returns a dict with the cross-validated balanced accuracy and a heuristic pass flag
    (``< pass_threshold`` ~= disentangled). Complementary checks to run alongside: z_iv
    SHOULD predict origin/culture well, and z_lin SHOULD predict cell identity.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    z = np.asarray(z_lin)
    y = np.asarray(is_organoid).astype(int)
    if z.ndim != 2 or z.shape[0] != y.shape[0]:
        raise ValueError(f"z_lin {z.shape} and is_organoid {y.shape} are not aligned 2-D/1-D")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    scores = cross_val_score(clf, z, y, cv=n_splits, scoring="balanced_accuracy")
    mean = float(scores.mean())
    return {
        "adversary_balanced_accuracy_mean": mean,
        "adversary_balanced_accuracy_std": float(scores.std()),
        "n_splits": n_splits,
        "disentangled_heuristic": bool(mean < pass_threshold),
        "note": "balanced accuracy ~0.5 => origin not recoverable from z_lin (good).",
    }
