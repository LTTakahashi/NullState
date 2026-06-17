"""Unit tests for the pure WS1 cell-selection helpers (scripts/run_ws1.py).

These cover the deterministic selection/subsampling logic that decides which cells become the
contrastiveVI background (primary on-target), target (organoid on-target), and the off-target
export set -- without needing scvi/anndata or any data on disk.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from scripts.run_ws1 import subsample_indices, select_ws1_cells


def test_subsample_indices_selects_true_positions():
    mask = np.array([True, False, True, True, False])
    out = subsample_indices(mask, max_n=None, seed=0)
    assert out.tolist() == [0, 2, 3]


def test_subsample_indices_caps_and_is_sorted_and_deterministic():
    mask = np.ones(1000, dtype=bool)
    a = subsample_indices(mask, max_n=50, seed=7)
    b = subsample_indices(mask, max_n=50, seed=7)
    assert a.size == 50
    assert np.array_equal(a, b)                     # deterministic for a fixed seed
    assert np.all(np.diff(a) > 0)                   # sorted, unique
    assert a.max() < 1000


def test_subsample_indices_no_cap_when_below_max():
    mask = np.array([True, True, False])
    out = subsample_indices(mask, max_n=100, seed=0)
    assert out.tolist() == [0, 1]


def test_select_ws1_cells_groups_by_origin_and_class():
    # 6 primary cells (origins), 6 organoid cells (pilot classes)
    primary_origin = ["neural", "neural_crest", "mesoderm", "neural", "endoderm", "unknown"]
    organoid_class = ["clean_ontarget", "poorly_diff_ontarget", "true_offtarget",
                      "true_offtarget", "ambiguous_novel", "qc_dropped"]
    ontarget_origins = frozenset({"neural", "neural_crest"})

    sel = select_ws1_cells(primary_origin, organoid_class, ontarget_origins, seed=0)

    # primary on-target = neural / neural_crest positions
    assert sel["primary_bg"].tolist() == [0, 1, 3]
    # organoid target = clean_ontarget / poorly_diff_ontarget
    assert sel["organoid_tg"].tolist() == [0, 1]
    # off-target = true_offtarget
    assert sel["organoid_off"].tolist() == [2, 3]


def test_select_ws1_cells_respects_caps():
    primary_origin = ["neural"] * 500
    organoid_class = ["clean_ontarget"] * 300 + ["true_offtarget"] * 200
    sel = select_ws1_cells(
        primary_origin, organoid_class, frozenset({"neural"}),
        max_primary=100, max_organoid_ontarget=50, max_offtarget=25, seed=1,
    )
    assert sel["primary_bg"].size == 100
    assert sel["organoid_tg"].size == 50
    assert sel["organoid_off"].size == 25
    # off-target positions all fall in the true_offtarget block [300, 500)
    assert sel["organoid_off"].min() >= 300


def test_select_ws1_cells_empty_when_no_match():
    sel = select_ws1_cells(["mesoderm", "endoderm"], ["ambiguous", "unmapped"],
                           frozenset({"neural"}), seed=0)
    assert sel["primary_bg"].size == 0
    assert sel["organoid_tg"].size == 0
    assert sel["organoid_off"].size == 0
