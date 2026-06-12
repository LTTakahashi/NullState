from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.utils.origin_map import ORIGIN_CATEGORIES

if TYPE_CHECKING:  # anndata is only referenced in type hints, never at runtime,
    import anndata as ad  # so the pure classifier imports with just pandas installed.

# On-target developmental origins for each organoid germ-layer system.
# The set names which mapped origins count as the intended product of that
# system; every *other* known origin (see KNOWN_ORIGINS) is an off-target
# contaminant for it.
#
#   'neural'   -> ectodermal neural organoids (e.g. HNOCA): neural AND neural_crest
#                 are on-target. The neural_crest entry IS the critical guard --
#                 neural crest is ectoderm-derived but mesenchymal, so without it
#                 every neural organoid would generate a systematic
#                 false-off-target wave.
#   'mesoderm' -> e.g. kidney organoids: mesoderm is on-target.
#   'endoderm' -> e.g. intestinal / HEOCA organoids: endoderm is on-target.
#
# Adding a system to this dict is all that the cross-germ-layer expansion
# (Phase 2) needs to reuse the identical decision logic on another germ layer.
ONTARGET_ORIGINS = {
    'neural':   frozenset({'neural', 'neural_crest'}),
    'mesoderm': frozenset({'mesoderm'}),
    'endoderm': frozenset({'endoderm'}),
}

# Origins the reference ontology can assign (single source of truth: origin_map).
# A mapped origin that is not on-target for the system but IS a known lineage is a
# genuine off-target contaminant; an origin outside this set (e.g. 'unknown') is
# unresolved and returns 'other'.
KNOWN_ORIGINS = frozenset(ORIGIN_CATEGORIES)


def classify_cell(origin: str, map_entropy: float, offmanifold: float, tau_H: float, tau_R: float, expected_germ_layer: str = 'neural') -> str:
    """
    Pure function implementing the off-target decision tree for one cell.

    ``expected_germ_layer`` selects the on-target origin set for the organoid
    system under analysis (see ``ONTARGET_ORIGINS``). For the neural pilot this is
    ``'neural'`` -> {neural, neural_crest} on-target, {mesoderm, endoderm} off-target,
    which is byte-for-byte the original neural-only behavior. Passing ``'mesoderm'``
    (kidney) or ``'endoderm'`` (HEOCA) generalizes the same logic to other germ
    layers with no code change -- the forward-compatibility the cross-germ-layer
    expansion (Phase 2) requires. An unrecognized ``expected_germ_layer`` falls back
    to the neural definition.

    CRITICAL: the neural-crest guard now lives in ``ONTARGET_ORIGINS['neural']`` =
    {'neural', 'neural_crest'}. Neural crest is ectoderm-derived but mesenchymal;
    without treating it as on-target in neural systems, every neural organoid would
    generate a systematic false-off-target wave because neural-crest-derived
    mesenchyme would be classified as an off-target contaminant instead of an
    on-target ectodermal product.
    """
    if pd.isna(map_entropy) or pd.isna(offmanifold):
        return 'unmapped'

    if offmanifold > tau_R:
        return 'ambiguous_novel'                  # resembles nothing real -> set aside

    ontarget = ONTARGET_ORIGINS.get(expected_germ_layer, ONTARGET_ORIGINS['neural'])

    if origin in ontarget:
        return 'poorly_diff_ontarget' if map_entropy > tau_H else 'clean_ontarget'

    if origin in KNOWN_ORIGINS:
        # Known lineage, but not the intended one for this system -> off-target.
        return 'true_offtarget' if map_entropy < tau_H else 'ambiguous'

    return 'other'


def classify_all(query: ad.AnnData, tau_H: float, tau_R: float, expected_germ_layer: str = 'neural') -> pd.Series:
    """Vectorized classification of all cells."""
    logging.info(f"Classifying all query cells (expected_germ_layer={expected_germ_layer})...")

    def apply_class(row):
        return classify_cell(
            origin=row['pred_origin'],
            map_entropy=row['map_entropy'],
            offmanifold=row['offmanifold'],
            tau_H=tau_H,
            tau_R=tau_R,
            expected_germ_layer=expected_germ_layer
        )

    classes = query.obs.apply(apply_class, axis=1)

    logging.info("Class distribution:")
    dist = classes.value_counts()
    for k, v in dist.items():
        logging.info(f"  {k}: {v} ({v/len(classes)*100:.1f}%)")

    return classes

def validate_classification(query: ad.AnnData, class_column: str = 'pilot_class') -> dict:
    """Sanity checks for classification."""
    logging.info("Validating classification results...")

    report = {}
    if class_column not in query.obs.columns:
        return {"error": f"Column {class_column} not found in query."}

    # Check 1: clean_ontarget should be dominated by expected (neural) types
    clean = query[query.obs[class_column] == 'clean_ontarget']
    clean_origins = clean.obs['pred_origin'].value_counts()
    report['clean_ontarget_origins'] = clean_origins.to_dict()

    # Check 2: true_offtarget should be mostly mesodermal/endodermal
    offtarget = query[query.obs[class_column] == 'true_offtarget']
    offtarget_origins = offtarget.obs['pred_origin'].value_counts()
    report['true_offtarget_origins'] = offtarget_origins.to_dict()

    # Log potential issues
    if 'neural' in offtarget_origins:
        logging.warning("Found 'neural' origins in true_offtarget. This violates the decision rules.")

    return report

def save_classifications(query: ad.AnnData, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "classifications.csv"
    logging.info(f"Saving classifications to {out_file}")
    query.obs.to_csv(out_file)
