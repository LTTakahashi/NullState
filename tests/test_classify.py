import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.mapping.classify import classify_cell

# ---------------------------------------------------------------------------
# Neural system (default expected_germ_layer='neural') -- original behavior.
# These must remain green: the refactor preserves the neural-only logic exactly.
# ---------------------------------------------------------------------------

def test_classify_clean_ontarget():
    # Neural cell, low entropy, low offmanifold
    assert classify_cell('neural', 0.5, 0.5, 1.0, 1.0) == 'clean_ontarget'

def test_classify_poorly_diff_ontarget():
    # Neural cell, high entropy, low offmanifold
    assert classify_cell('neural', 1.5, 0.5, 1.0, 1.0) == 'poorly_diff_ontarget'

def test_classify_true_offtarget_mesoderm():
    # Mesoderm cell, low entropy, low offmanifold
    assert classify_cell('mesoderm', 0.5, 0.5, 1.0, 1.0) == 'true_offtarget'

def test_classify_ambiguous_mesoderm():
    # Mesoderm cell, high entropy, low offmanifold
    assert classify_cell('mesoderm', 1.5, 0.5, 1.0, 1.0) == 'ambiguous'

def test_classify_true_offtarget_endoderm():
    # Endoderm cell, low entropy
    assert classify_cell('endoderm', 0.5, 0.5, 1.0, 1.0) == 'true_offtarget'

def test_classify_ambiguous_novel():
    # Any cell, high offmanifold
    assert classify_cell('neural', 0.5, 1.5, 1.0, 1.0) == 'ambiguous_novel'
    assert classify_cell('mesoderm', 0.5, 1.5, 1.0, 1.0) == 'ambiguous_novel'

def test_neural_crest_guard():
    # CRITICAL: Neural crest cell, low entropy, low offmanifold -> clean_ontarget
    assert classify_cell('neural_crest', 0.5, 0.5, 1.0, 1.0) == 'clean_ontarget'
    # NOT true_offtarget!

def test_classify_boundary_entropy():
    # Exactly at tau_H. Logic says > tau_H or < tau_H.
    # Usually we use <= or >=, but let's check what we implemented.
    # We implemented `> tau_H`. So exact == tau_H should fall into the `else` of that branch.
    # In `classify_cell`: `if origin in ontarget: return 'poorly_diff_ontarget' if map_entropy > tau_H else 'clean_ontarget'`
    assert classify_cell('neural', 1.0, 0.5, 1.0, 1.0) == 'clean_ontarget'

def test_classify_boundary_offmanifold():
    # offmanifold > tau_R -> ambiguous_novel. So exact == tau_R -> proceeds.
    assert classify_cell('neural', 0.5, 1.0, 1.0, 1.0) == 'clean_ontarget'

def test_unknown_origin_is_other():
    # An origin outside the known germ-layer categories resolves to 'other',
    # never on- or off-target. (origin_map assigns 'unknown' to unmapped types.)
    assert classify_cell('unknown', 0.5, 0.5, 1.0, 1.0) == 'other'

def test_default_germ_layer_is_neural():
    # With no expected_germ_layer, mesoderm is off-target (neural system default).
    assert classify_cell('mesoderm', 0.5, 0.5, 1.0, 1.0) == 'true_offtarget'
    assert classify_cell('mesoderm', 0.5, 0.5, 1.0, 1.0, expected_germ_layer='neural') == 'true_offtarget'

# ---------------------------------------------------------------------------
# Generalization (Phase 2 forward-compat): expected_germ_layer drives the
# on-target set, so the same logic works for kidney (mesoderm) and HEOCA
# (endoderm) systems without code changes.
# ---------------------------------------------------------------------------

def test_mesoderm_system_mesoderm_is_ontarget():
    # Kidney organoid: mesoderm is now the intended product, not a contaminant.
    assert classify_cell('mesoderm', 0.5, 0.5, 1.0, 1.0, expected_germ_layer='mesoderm') == 'clean_ontarget'
    assert classify_cell('mesoderm', 1.5, 0.5, 1.0, 1.0, expected_germ_layer='mesoderm') == 'poorly_diff_ontarget'

def test_mesoderm_system_neural_is_offtarget():
    # In a kidney organoid, neural cells are off-target contamination.
    assert classify_cell('neural', 0.5, 0.5, 1.0, 1.0, expected_germ_layer='mesoderm') == 'true_offtarget'

def test_mesoderm_system_neural_crest_is_offtarget():
    # The neural-crest guard is neural-system-specific: in a mesoderm organoid,
    # neural crest is ectodermal contamination -> off-target (NOT spared).
    assert classify_cell('neural_crest', 0.5, 0.5, 1.0, 1.0, expected_germ_layer='mesoderm') == 'true_offtarget'

def test_endoderm_system_endoderm_is_ontarget():
    # HEOCA / intestinal organoid: endoderm is on-target.
    assert classify_cell('endoderm', 0.5, 0.5, 1.0, 1.0, expected_germ_layer='endoderm') == 'clean_ontarget'

def test_endoderm_system_mesoderm_is_offtarget():
    # Off-target mesenchyme in an endodermal organoid is still off-target.
    assert classify_cell('mesoderm', 0.5, 0.5, 1.0, 1.0, expected_germ_layer='endoderm') == 'true_offtarget'

def test_unrecognized_germ_layer_falls_back_to_neural():
    # An unknown system name falls back to the neural on-target definition.
    assert classify_cell('neural', 0.5, 0.5, 1.0, 1.0, expected_germ_layer='nonsense') == 'clean_ontarget'
    assert classify_cell('mesoderm', 0.5, 0.5, 1.0, 1.0, expected_germ_layer='nonsense') == 'true_offtarget'
