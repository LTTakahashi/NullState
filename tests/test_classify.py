import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.mapping.classify import classify_cell

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
    # In `classify_cell`: `if origin in ('neural', 'neural_crest'): return 'poorly_diff_ontarget' if map_entropy > tau_H else 'clean_ontarget'`
    assert classify_cell('neural', 1.0, 0.5, 1.0, 1.0) == 'clean_ontarget'

def test_classify_boundary_offmanifold():
    # offmanifold > tau_R -> ambiguous_novel. So exact == tau_R -> proceeds.
    assert classify_cell('neural', 0.5, 1.0, 1.0, 1.0) == 'clean_ontarget'
