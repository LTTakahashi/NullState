import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.interpret.perturbation import rank_perturbation_targets
from src.interpret.signatures import differential_program_scores


# --------------------------- perturbation ranking ---------------------------

def test_rank_perturbation_orders_and_flags():
    rng = np.random.default_rng(0)
    n = 80
    null = rng.normal(0.0, 0.2, n)
    null = null - null.mean()                      # exactly mean 0 -> robustly non-significant
    shift = {
        "TF_rescue": rng.normal(1.0, 0.2, n),      # strong positive (rescues)
        "TF_null": null,
        "TF_collapse": rng.normal(-1.0, 0.2, n),   # strong negative (drives into default)
    }
    res = rank_perturbation_targets(shift, alpha=0.05)
    assert list(res.index) == ["TF_rescue", "TF_null", "TF_collapse"]   # mean_shift descending
    assert res.loc["TF_rescue", "mean_shift"] > 0
    assert res.loc["TF_collapse", "mean_shift"] < 0
    assert bool(res.loc["TF_rescue", "significant"]) is True
    assert bool(res.loc["TF_null", "significant"]) is False
    assert res.loc["TF_rescue", "rank"] == 1
    assert res.loc["TF_rescue", "frac_positive"] > 0.9


def test_rank_perturbation_ascending():
    rng = np.random.default_rng(1)
    shift = {"A": rng.normal(1.0, 0.2, 50), "B": rng.normal(-1.0, 0.2, 50)}
    res = rank_perturbation_targets(shift, ascending=True)
    assert list(res.index) == ["B", "A"]            # most-collapsing first


def test_rank_perturbation_accepts_dataframe():
    df = pd.DataFrame({"X": [1.0, 1.1, 0.9, 1.05], "Y": [0.0, 0.1, -0.1, 0.0]})
    res = rank_perturbation_targets(df)
    assert set(res.index) == {"X", "Y"}
    assert res.loc["X", "mean_shift"] > res.loc["Y", "mean_shift"]


def test_rank_perturbation_bad_input():
    with pytest.raises(TypeError):
        rank_perturbation_targets([1, 2, 3])


# --------------------------- differential program scores ---------------------------

def test_differential_program_scores():
    rng = np.random.default_rng(0)
    ng = nr = 60
    labels = np.array(["conv"] * ng + ["ref"] * nr)
    scores = pd.DataFrame({
        "EMT":    np.concatenate([rng.normal(0.8, 0.1, ng), rng.normal(0.2, 0.1, nr)]),
        "Neuron": np.concatenate([rng.normal(0.2, 0.1, ng), rng.normal(0.8, 0.1, nr)]),
        "Noise":  np.concatenate([rng.normal(0.5, 0.1, ng), rng.normal(0.5, 0.1, nr)]),
    })
    res = differential_program_scores(scores, labels, group="conv", reference="ref")
    assert res.index[0] == "EMT"                                  # most elevated in conv
    assert res.loc["EMT", "mean_diff"] > 0
    assert bool(res.loc["EMT", "significant"]) is True
    assert res.loc["EMT", "auc"] > 0.9                            # group >> reference
    assert res.loc["Neuron", "mean_diff"] < 0
    assert 0.35 < res.loc["Noise", "auc"] < 0.65                  # null program ~ chance


def test_differential_reference_none_is_rest():
    labels = np.array(["conv"] * 30 + ["other"] * 40)
    scores = pd.DataFrame({"P": np.concatenate([np.ones(30), np.zeros(40)])})
    res = differential_program_scores(scores, labels, group="conv")  # reference=None -> the rest
    assert res.loc["P", "mean_diff"] == pytest.approx(1.0)
    assert res.loc["P", "n_reference"] == 40


def test_differential_empty_group_raises():
    labels = np.array(["a"] * 10)
    scores = pd.DataFrame({"P": np.arange(10.0)})
    with pytest.raises(ValueError):
        differential_program_scores(scores, labels, group="missing")
