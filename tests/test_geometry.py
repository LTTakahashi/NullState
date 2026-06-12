import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.geometry.distances import sliced_wasserstein, sinkhorn_divergence
from src.geometry.matched_n import (
    estimate_n_min, self_distance_floor, cross_distance_ci, matched_n_test, benjamini_hochberg,
)
from src.geometry.power import detection_power


# --------------------------- sliced-Wasserstein ---------------------------

def test_sw_identical_is_zero():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    assert sliced_wasserstein(X, X, n_projections=50, random_state=1) == pytest.approx(0.0, abs=1e-9)


def test_sw_increases_with_separation():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    shift = np.array([1.0, 0.0, 0.0])
    d1 = sliced_wasserstein(X, X + shift, n_projections=200, random_state=2)
    d2 = sliced_wasserstein(X, X + 3 * shift, n_projections=200, random_state=2)
    assert 0 < d1 < d2


def test_sw_symmetric_with_same_seed():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 3))
    Y = rng.standard_normal((90, 3)) + 0.5
    a = sliced_wasserstein(X, Y, n_projections=100, random_state=7)
    b = sliced_wasserstein(Y, X, n_projections=100, random_state=7)
    assert a == pytest.approx(b, rel=1e-9)


# --------------------------- Sinkhorn divergence ---------------------------

def test_sinkhorn_self_is_zero():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    assert sinkhorn_divergence(X, X, n_iter=100) == pytest.approx(0.0, abs=1e-9)


def test_sinkhorn_positive_for_separated():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    Y = rng.standard_normal((60, 3)) + 4.0
    assert sinkhorn_divergence(X, Y, n_iter=150) > 0.0


# --------------------------- matched-N protocol ---------------------------

def test_estimate_n_min_saturation():
    curve = {50: 1.0, 100: 0.6, 200: 0.55, 500: 0.54}  # saturates between 100 and 200
    assert estimate_n_min(curve, rel_tol=0.1) == 100


def test_floor_below_cross_for_separated():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((400, 3))
    Q = rng.standard_normal((400, 3)) + 3.0
    floor = self_distance_floor(P, 100, n_repeats=50, n_projections=80, random_state=1)
    cross = cross_distance_ci(P, Q, 100, n_boot=100, n_projections=80, random_state=2)
    assert float(np.percentile(floor, 95)) < cross["lo"]   # cross clears the floor


def test_matched_n_real_difference():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((400, 3))
    Q = rng.standard_normal((150, 3)) + 3.0
    res = matched_n_test(P, Q, n_min=50, n_repeats=50, n_boot=150, n_projections=80,
                         random_state=3)
    assert res["n_star"] == 150
    assert res["clears_floor"] is True
    assert res["verdict"] == "real_difference"


def test_matched_n_inconclusive_for_same_distribution():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((400, 3))
    Q = rng.standard_normal((150, 3))          # same distribution as P
    res = matched_n_test(P, Q, n_min=50, n_repeats=50, n_boot=150, n_projections=80,
                         random_state=4)
    assert res["verdict"] == "inconclusive_underpowered"


def test_matched_n_underpowered_below_nmin():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((400, 3))
    Q = rng.standard_normal((40, 3)) + 5.0     # big effect but N* = 40 < n_min
    res = matched_n_test(P, Q, n_min=100, n_repeats=40, n_boot=100, n_projections=60,
                         random_state=5)
    assert res["underpowered"] is True
    assert res["verdict"] == "inconclusive_underpowered"


def test_matched_n_convergent_below_baseline():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((400, 3))
    Q = rng.standard_normal((150, 3)) + 3.0
    baseline = np.full(200, 100.0) + rng.standard_normal(200) * 0.1   # maturation predicts huge distance
    res = matched_n_test(P, Q, n_min=50, n_repeats=50, n_boot=150, n_projections=80,
                         control3_baseline=baseline, random_state=6)
    assert res["verdict"] == "convergent_shared_default"   # clears floor AND below baseline


def test_benjamini_hochberg():
    pvals = [0.001, 0.2, 0.03, 0.5]
    rejected, q = benjamini_hochberg(pvals, alpha=0.05)
    assert rejected.tolist() == [True, False, False, False]
    assert q[0] == pytest.approx(0.004, rel=1e-6)


# --------------------------- power simulation ---------------------------

def test_power_increases_with_effect():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((600, 4))
    p0 = detection_power(base, delta=0.0, n=60, n_trials=40, n_floor_repeats=40,
                         n_projections=50, random_state=1)
    p_big = detection_power(base, delta=5.0, n=60, n_trials=40, n_floor_repeats=40,
                            n_projections=50, random_state=1)
    assert p0 < 0.3            # ~false-positive rate near 1 - 0.95
    assert p_big > 0.8         # large effect is detected
    assert p_big > p0
