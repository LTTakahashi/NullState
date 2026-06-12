import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

# Pure helper: imports with numpy only (heavy deps in contrastive.py are lazy).
from src.disentangle.contrastive import select_contrastive_indices


def test_basic_split():
    # rows: 0 primary/on, 1 primary/off, 2 organoid/on, 3 organoid/off, 4 primary/on
    is_primary  = [True,  True,  False, False, True]
    is_organoid = [False, False, True,  True,  False]
    is_ontarget = [True,  False, True,  False, True]
    bg, tg = select_contrastive_indices(is_primary, is_organoid, is_ontarget)
    assert bg.tolist() == [0, 4]      # primary AND on-target
    assert tg.tolist() == [2]         # organoid AND on-target


def test_empty_background_raises():
    # no primary on-target cells
    with pytest.raises(ValueError):
        select_contrastive_indices(
            is_primary=[True, False], is_organoid=[False, True], is_ontarget=[False, True]
        )


def test_empty_target_raises():
    # no organoid on-target cells
    with pytest.raises(ValueError):
        select_contrastive_indices(
            is_primary=[True, False], is_organoid=[False, True], is_ontarget=[True, False]
        )


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        select_contrastive_indices(
            is_primary=[True, False], is_organoid=[False], is_ontarget=[True, True]
        )


def test_overlapping_sources_raise():
    # a cell flagged both primary and organoid is invalid
    with pytest.raises(ValueError):
        select_contrastive_indices(
            is_primary=[True, True], is_organoid=[True, False], is_ontarget=[True, True]
        )


def test_returns_integer_arrays():
    bg, tg = select_contrastive_indices(
        is_primary=[True, False], is_organoid=[False, True], is_ontarget=[True, True]
    )
    assert bg.dtype.kind == "i" and tg.dtype.kind == "i"
    assert bg.tolist() == [0] and tg.tolist() == [1]
