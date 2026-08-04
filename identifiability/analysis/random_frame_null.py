"""
THE NULL WE NEVER SPECIFIED: is the recovered frame better than a random one?

If a run recovers the subspace (CCA ~ 1) but converges to a frame at angle phi to
the ground-truth basis, then gap = 1 - cos(phi). Hungarian matching folds phi into
[0, 45 deg], because a 90-degree rotation is a permutation and 90 - phi is a swap.
So "the optimiser picks a frame with no rotational preference" means phi ~ U[0, pi/4],
and the pushforward of gap has closed form:

    F(x)    = (4/pi) * arccos(1 - x)          for x in [0, 1 - cos(pi/4)]
    E[gap]  = 1 - (4/pi) sin(pi/4) = 1 - 2*sqrt(2)/pi = 0.09968
    SD[gap] = sqrt(1/2 + 1/pi - 2*sqrt(2)/pi ... )                = 0.08800
    range   = [0, 0.29289]

Every floor we measured sits on those numbers. This script tests the pooled
per-run gaps against the analytic law directly (one-sample KS), per arm and per
environment count. If we cannot reject, the honest statement is far stronger than
"the variability condition does not bite as a threshold": it is that the recovered
frame is **statistically indistinguishable from a uniformly random one** -- not
weak axis-level identifiability, none detectable.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from core.paths import result, RESULTS

GAP_MAX = 1 - np.cos(np.pi / 4)


def analytic_moments():
    mean = 1 - 2 * np.sqrt(2) / np.pi
    e2 = 1 - 2 * (2 * np.sqrt(2) / np.pi) + (0.5 + 1 / np.pi)
    return mean, np.sqrt(e2 - mean ** 2)


def cdf(x):
    """P(gap <= x) under phi ~ U[0, pi/4]."""
    x = np.clip(np.asarray(x, float), 0.0, GAP_MAX)
    return (4 / np.pi) * np.arccos(1 - x)


def ks_test(sample):
    s = np.sort(np.clip(np.asarray(sample, float), 0.0, GAP_MAX))
    n = len(s)
    F = cdf(s)
    d = max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(0, n) / n))
    try:
        from scipy.stats import kstwo
        p = float(kstwo.sf(d, n))
    except Exception:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
        p = float(2 * sum((-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam)
                          for k in range(1, 101)))
    return float(d), min(max(p, 0.0), 1.0)


def report(name, g):
    g = np.asarray(g, float)
    d, p = ks_test(g)
    m, sd = analytic_moments()
    print(f"  {name:>26}  n={len(g):>3}  mean {g.mean():.4f}  sd {g.std(ddof=1):.4f}"
          f"  KS D={d:.3f} p={p:.3f}  -> "
          f"{'CANNOT reject random-frame' if p > 0.05 else 'REJECTS random-frame'}")
    return p


def main():
    m, sd = analytic_moments()
    print("=" * 88)
    print("ANALYTIC RANDOM-FRAME LAW   phi ~ U[0, 45deg],  gap = 1 - cos(phi)")
    print("=" * 88)
    print(f"  E[gap] = {m:.4f}   SD = {sd:.4f}   range = [0, {GAP_MAX:.4f}]\n")

    print("=" * 88)
    print("VARIABILITY-CONDITION SWEEP (env on every axis, no shift)")
    print("=" * 88)
    vc = pd.read_csv(result("variability_condition.csv"))
    ps = []
    for ne in sorted(vc.n_env.unique()):
        g = vc[vc.n_env == ne].gap.values
        ps.append(report(f"n_env={ne} ({'satisfies' if ne >= 5 else 'violates'})", g))
    report("POOLED", vc.gap.values)
    report("satisfying only (n_env>=5)", vc[vc.n_env >= 5].gap.values)

    print("\n" + "=" * 88)
    print("STAGE-4 ARMS (faithful encoders, rho=1, delta=0)")
    print("=" * 88)
    s4 = pd.read_csv(result("main_sweep.csv"))
    s4 = s4[(s4.rho == 1.0) & (s4.delta == 0.0)].assign(
        gap=lambda x: x.learned_cca - x.learned_mcc)
    for arm in ["conditional", "ivae_2env", "ivae_5env"]:
        g = s4[s4.arm == arm].gap.values
        if len(g):
            report(arm, g)

    print("\n" + "=" * 88)
    print("BASIS-ROTATION, faithful arm")
    print("=" * 88)
    try:
        br = pd.read_csv(result("basis_rotation.csv"))
        for th, lab in [(0.0, "theta=0"), (np.pi / 4, "theta=45deg")]:
            g = br[(br.arm == "faithful") & np.isclose(br.theta, th)].gap_s.values
            if len(g):
                report(lab, g)
    except FileNotFoundError:
        print("  (results_basis_rotation.csv not found)")

    print("\n" + "=" * 88)
    print("WHAT THIS MEANS")
    print("=" * 88)
    print("""  If the tests above do not reject, the recovered frame is statistically
  indistinguishable from uniformly random -- in EVERY arm, including those that
  satisfy nk+1. The correct claim is then not "the variability condition does not
  bite as a threshold" but "satisfying it leaves axis-level identifiability
  undetectable". Chasing the apparent decline with more seeds would buy a tighter
  estimate of nothing: the random-frame law already predicts the mean, the SD and
  the range.""")


if __name__ == "__main__":
    main()
