import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.geometry.pipeline import run_geometry_over_populations


def test_pipeline_three_populations():
    rng = np.random.default_rng(0)
    d = 3
    A = rng.standard_normal((600, d))                 # reference
    B = rng.standard_normal((250, d)) + 6.0           # well separated from A
    C = rng.standard_normal((250, d))                 # same distribution as A (a true null pair)
    z = np.vstack([A, B, C])
    labels = np.array(["A"] * 600 + ["B"] * 250 + ["C"] * 250)

    res = run_geometry_over_populations(
        z, labels, ladder=[50, 100, 200], n_repeats=40, n_boot=120,
        n_projections=60, n_rarefaction_repeats=20, random_state=1,
    )
    assert res["n_min"] is not None
    assert res["sizes"] == {"A": 600, "B": 250, "C": 250}
    assert res["pairs"][("A", "B")]["verdict"] == "real_difference"
    assert res["pairs"][("B", "C")]["verdict"] == "real_difference"
    assert res["pairs"][("A", "C")]["verdict"] == "inconclusive_underpowered"


def test_pipeline_convergent_with_baseline():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((600, 3))
    B = rng.standard_normal((250, 3)) + 6.0
    z = np.vstack([A, B])
    labels = np.array(["A"] * 600 + ["B"] * 250)
    # maturation baseline predicts a much larger distance than observed -> convergent
    control3 = {frozenset({"A", "B"}): np.full(200, 100.0) + rng.standard_normal(200) * 0.1}
    res = run_geometry_over_populations(
        z, labels, n_min=50, control3=control3, n_repeats=40, n_boot=120,
        n_projections=60, random_state=1,
    )
    assert res["pairs"][("A", "B")]["verdict"] == "convergent_shared_default"


def test_pipeline_requires_two_populations():
    z = np.random.default_rng(0).standard_normal((10, 2))
    with pytest.raises(ValueError):
        run_geometry_over_populations(z, np.array(["X"] * 10))


def test_pipeline_length_mismatch():
    z = np.random.default_rng(0).standard_normal((10, 2))
    with pytest.raises(ValueError):
        run_geometry_over_populations(z, np.array(["X"] * 9))
