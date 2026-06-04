import pandas as pd
import logging

# Germ-layer / lineage categories used throughout the pilot.
ORIGIN_CATEGORIES = ('neural', 'neural_crest', 'mesoderm', 'endoderm')


def get_default_origin_map() -> dict[str, str]:
    """Map every reference cell-type label to a developmental origin.

    Categories: 'neural', 'neural_crest', 'mesoderm', 'endoderm'.

    The KEYS here must match the *exact* CELLxGENE ontology labels found in the
    harmonized reference's ``obs['cell_type']`` (e.g. "radial glial cell", not the
    informal "radial glia"). Lookups are performed case-insensitively via
    :func:`map_origin`, but the canonical strings are kept verbatim so that a
    direct ``Series.map(get_default_origin_map())`` also works.

    Biological rationale for neural crest: NC cells are ectoderm-derived but undergo
    EMT and generate mesenchymal/peripheral derivatives (Schwann cells, sympathetic
    and enteric neurons, chromaffin cells). In a NEURAL organoid these are ON-TARGET
    ectodermal products, NOT off-target mesodermal contaminants, so they are kept
    distinct from 'mesoderm'.

    Several assignments are developmental-biology judgment calls and are tagged
    "[REVIEW]" below — confirm or override them for your germ-layer definitions.
    """
    origin_map = {
        # === Neural (CNS, ectoderm) ===
        "neuron": "neural",
        "oligodendrocyte": "neural",
        "radial glial cell": "neural",
        "glioblast": "neural",                                  # glial progenitor (CNS)
        "glial cell": "neural",                                 # generic CNS glia
        "neuroblast (sensu Vertebrata)": "neural",              # CNS neuronal progenitor
        "neuroblast (sensu Nematoda and Protostomia)": "neural",  # [REVIEW] ontology artifact; treated as CNS neuroblast

        # === Neural crest (ectoderm-derived, peripheral / mesenchymal) ===
        "Schwann cell": "neural_crest",
        "neural crest cell": "neural_crest",
        "enteric neuron": "neural_crest",
        "sympathetic neuron": "neural_crest",
        "chromaffin cell": "neural_crest",                      # adrenal medulla, NC-derived
        "neuroplacodal cell": "neural_crest",                   # [REVIEW] placodal ectoderm; grouped with NC as on-target peripheral

        # === Mesoderm ===
        "fibroblast": "mesoderm",
        "stromal cell": "mesoderm",
        "smooth muscle cell": "mesoderm",                       # (vascular SMC can be NC-derived; bulk -> mesoderm)
        "cardiac muscle cell": "mesoderm",
        "cell of skeletal muscle": "mesoderm",
        "blood vessel endothelial cell": "mesoderm",
        "endothelial cell of vascular tree": "mesoderm",
        "mesothelial cell": "mesoderm",
        "cortical cell of adrenal gland": "mesoderm",           # adrenal cortex = intermediate mesoderm
        "hepatic stellate cell": "mesoderm",                    # mesenchymal (septum transversum)
        "macrophage": "mesoderm",                               # [REVIEW] yolk-sac / hematopoietic -> mesoderm
        "myeloid cell": "mesoderm",
        "innate lymphoid cell": "mesoderm",
        "erythroblast": "mesoderm",
        "erythrocyte": "mesoderm",
        "megakaryocyte": "mesoderm",

        # === Endoderm ===
        "hepatoblast": "endoderm",
        "intestinal epithelial cell": "endoderm",
        "epithelial cell of lower respiratory tract": "endoderm",
        "ciliated epithelial cell": "endoderm",                 # [REVIEW] assumed airway/endodermal in this fetal context

        # 'cell' (generic / unannotated catch-all) is intentionally left unmapped
        # -> resolves to 'unknown' so it is never counted as on- or off-target.
    }
    return origin_map


def map_origin(cell_type, origin_map: dict, default: str = 'unknown') -> str:
    """Case-insensitive, whitespace-robust lookup of a single label's origin."""
    if cell_type is None or (isinstance(cell_type, float) and pd.isna(cell_type)):
        return default
    key = str(cell_type).strip().lower()
    lower_map = {k.strip().lower(): v for k, v in origin_map.items()}
    return lower_map.get(key, default)


def map_origin_series(cell_types: pd.Series, origin_map: dict, default: str = 'unknown') -> pd.Series:
    """Vectorized case-insensitive origin assignment for a column of labels."""
    lower_map = {k.strip().lower(): v for k, v in origin_map.items()}
    norm = cell_types.astype(str).str.strip().str.lower()
    return norm.map(lower_map).fillna(default)


def validate_origin_map(origin_map: dict, cell_types: pd.Series) -> dict:
    """Checks that every unique cell_type in the data has a mapping (case-insensitive)."""
    unique_types = cell_types.dropna().unique()
    lower_map = {k.strip().lower(): v for k, v in origin_map.items()}

    mapped_count = 0
    unmapped_types = []
    for ct in unique_types:
        if str(ct).strip().lower() in lower_map:
            mapped_count += 1
        else:
            unmapped_types.append(ct)

    coverage_pct = (mapped_count / len(unique_types)) * 100 if len(unique_types) > 0 else 0

    if unmapped_types:
        logging.error(f"Found {len(unmapped_types)} unmapped cell types ({coverage_pct:.1f}% coverage).")
        logging.error(f"Examples of unmapped: {unmapped_types[:10]}")
    else:
        logging.info("100% cell type origin coverage.")

    return {
        "mapped_count": mapped_count,
        "unmapped_types": unmapped_types,
        "coverage_pct": coverage_pct,
    }


def extend_origin_map(base_map: dict, extensions: dict[str, str]) -> dict:
    """Merges user-provided extensions into the base map."""
    new_map = base_map.copy()
    for k, v in extensions.items():
        if k in new_map and new_map[k] != v:
            logging.warning(f"Overriding {k}: {new_map[k]} -> {v}")
        new_map[k] = v
    return new_map
