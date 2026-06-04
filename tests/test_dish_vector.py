import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.calibration.dish_vector import compute_cosine_matrix, evaluate_go_nogo
from src.gates.count_check import evaluate_gate

def test_cosine_matrix_identical():
    v1 = np.array([1, 2, 3])
    v2 = np.array([1, 2, 3])
    dv = {"type1": v1, "type2": v2}
    mat = compute_cosine_matrix(dv)
    assert np.isclose(mat.loc["type1", "type2"], 1.0)

def test_cosine_matrix_orthogonal():
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    dv = {"type1": v1, "type2": v2}
    mat = compute_cosine_matrix(dv)
    assert np.isclose(mat.loc["type1", "type2"], 0.0)

def test_cosine_matrix_antiparallel():
    v1 = np.array([1, 1])
    v2 = np.array([-1, -1])
    dv = {"type1": v1, "type2": v2}
    mat = compute_cosine_matrix(dv)
    assert np.isclose(mat.loc["type1", "type2"], -1.0)

def test_evaluate_go_nogo_go():
    mat = pd.DataFrame([
        [1.0, 0.8, 0.9],
        [0.8, 1.0, 0.75],
        [0.9, 0.75, 1.0]
    ], index=["t1", "t2", "t3"], columns=["t1", "t2", "t3"])
    res = evaluate_go_nogo(mat, threshold=0.70)
    assert res["decision"] == "GO"
    assert np.isclose(res["mean_cosine"], (0.8 + 0.9 + 0.75) / 3)

def test_evaluate_go_nogo_nogo():
    mat = pd.DataFrame([
        [1.0, 0.5, 0.6],
        [0.5, 1.0, 0.4],
        [0.6, 0.4, 1.0]
    ], index=["t1", "t2", "t3"], columns=["t1", "t2", "t3"])
    res = evaluate_go_nogo(mat, threshold=0.70)
    assert res["decision"] == "NO-GO"

def test_count_gate_green():
    counts = {
        "total_count": 1500,
        "by_protocol": {"p1": 400, "p2": 400, "p3": 100}
    }
    gate = evaluate_gate(counts, green_total=1000, green_per_protocol=300, yellow_total=300)
    assert gate["decision"] == "GREEN"

def test_count_gate_yellow():
    counts = {
        "total_count": 500,
        "by_protocol": {"p1": 400, "p2": 100}
    }
    gate = evaluate_gate(counts, green_total=1000, green_per_protocol=300, yellow_total=300)
    assert gate["decision"] == "YELLOW"

def test_count_gate_red():
    counts = {
        "total_count": 200,
        "by_protocol": {"p1": 200}
    }
    gate = evaluate_gate(counts, green_total=1000, green_per_protocol=300, yellow_total=300)
    assert gate["decision"] == "RED"
