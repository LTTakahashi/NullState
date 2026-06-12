import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.geometry.topology import (
    h0_persistence, _h0_diagram_distance, fasy_confidence_band,
    significant_h0_features, h0_confirmatory_test,
)


def test_h0_persistence_count_and_order():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    p = h0_persistence(X)
    assert p.shape[0] == 19              # n-1 finite features
    assert np.all(np.diff(p) >= 0)       # ascending death times


def test_h0_two_clusters_big_gap():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 2)) * 0.1
    B = rng.standard_normal((50, 2)) * 0.1 + np.array([10.0, 0.0])
    p = h0_persistence(np.vstack([A, B]))
    assert p.max() > 5.0                 # the single inter-cluster bridge (~10)
    assert np.median(p) < 1.0            # within-cluster edges are tiny


def test_diagram_distance_zero_and_positive():
    a = np.array([0.1, 0.2, 5.0])
    assert _h0_diagram_distance(a, a) == pytest.approx(0.0)
    assert _h0_diagram_distance(a, np.array([0.1, 0.2, 8.0])) == pytest.approx(3.0)


def test_fasy_band_positive():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    c = fasy_confidence_band(X, alpha=0.05, n_bootstrap=80, random_state=1)
    assert c > 0


def test_significant_features_signature():
    out = significant_h0_features(np.array([0.1, 0.2, 5.0]), c=0.5, multiplier=2.0)
    assert out["band"] == pytest.approx(1.0)
    assert out["n_significant"] == 1     # only 5.0 exceeds band=1.0


def test_confirmatory_detects_clusters_not_blob():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((60, 2)) * 0.1
    B = rng.standard_normal((60, 2)) * 0.1 + np.array([10.0, 0.0])
    two = h0_confirmatory_test(np.vstack([A, B]), n_bootstrap=80, random_state=1)
    blob = h0_confirmatory_test(rng.standard_normal((120, 2)), n_bootstrap=80, random_state=1)
    assert two["n_significant"] >= 1                       # the cluster gap survives the band
    assert two["max_persistence"] > blob["max_persistence"]
