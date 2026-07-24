"""
POWER ANALYSIS for the pre-registered null.

A null is only meaningful if the design could have detected the effect it failed
to find. This computes, by simulation under the OBSERVED per-cell noise, the
minimum detectable effect (MDE) at 80% power for:

  (A) the pairwise test the pre-registration used: gap(rho=1) vs gap(rho=0.05),
      two-sample t at alpha=0.05 two-sided, n seeds per group; and
  (B) the trend test that pools all seeds x all rho levels: OLS slope of gap on
      rho, which is far more powerful because it uses every point.

Reported at n=5 (the pre-registered design) and n=25 (the expanded design), so
the reader can see exactly what each could and could not rule out. Simulation
(not a formula) so the small-sample noncentral-t behaviour is exact.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20240524)
NSIM = 4000
RHOS = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.05])


def two_sample_t(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    t = (a.mean() - b.mean()) / (sp * np.sqrt(1 / na + 1 / nb) + 1e-12)
    return t, na + nb - 2


def t_crit(df, alpha=0.05):
    # two-sided critical value via scipy if available, else a stored table
    try:
        from scipy.stats import t as tdist
        return float(tdist.ppf(1 - alpha / 2, df))
    except Exception:
        table = {8: 2.306, 18: 2.101, 48: 2.011, 98: 1.984}
        return table[min(table, key=lambda k: abs(k - df))]


def pairwise_power(sigma, n, d, nsim=NSIM):
    """Power to detect a standardized effect d = mean_gap_diff / sigma at n/group."""
    tc = t_crit(2 * n - 2)
    rej = 0
    for _ in range(nsim):
        a = RNG.normal(0.0, sigma, n)
        b = RNG.normal(d * sigma, sigma, n)
        t, df = two_sample_t(a, b)
        if abs(t) > tc:
            rej += 1
    return rej / nsim


def pairwise_mde(sigma, n, target_power=0.80):
    lo, hi = 0.0, 4.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if pairwise_power(sigma, n, mid) < target_power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def trend_power(sigma, n_per_rho, slope, nsim=NSIM):
    """Power of the OLS slope test of gap ~ rho, pooling n_per_rho seeds per rho."""
    x = np.repeat(RHOS, n_per_rho)
    xc = x - x.mean()
    Sxx = (xc ** 2).sum()
    dfree = len(x) - 2
    tc = t_crit(dfree)
    rej = 0
    for _ in range(nsim):
        y = slope * x + RNG.normal(0, sigma, len(x))
        b = (xc * (y - y.mean())).sum() / Sxx
        resid = y - (y.mean() + b * xc)
        se = np.sqrt((resid ** 2).sum() / dfree / Sxx)
        if abs(b / (se + 1e-12)) > tc:
            rej += 1
    return rej / nsim


def trend_mde(sigma, n_per_rho, target_power=0.80):
    lo, hi = 0.0, 2.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if trend_power(sigma, n_per_rho, mid) < target_power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "results_stage4.csv"
    df = pd.read_csv(csv)
    print(f"data: {csv}  ({len(df)} rows, {df.seed.nunique()} seeds)\n")

    # observed per-cell residual SD of the gap, at delta=0, per arm
    print("observed per-cell gap SD (delta=0):")
    sigmas = {}
    for arm in sorted(df.arm.unique()):
        s = df[(df.arm == arm) & (df.delta == 0)].groupby("rho").gap_mcc.std().mean()
        sigmas[arm] = float(s)
        print(f"   {arm:>12}: sigma = {s:.4f}")
    sigma = float(np.mean(list(sigmas.values())))
    n_now = df.seed.nunique()
    print(f"   pooled sigma used below = {sigma:.4f}\n")

    print("=" * 66)
    print("(A) PAIRWISE test  gap(rho=1) vs gap(rho=0.05), alpha=0.05 two-sided")
    print("=" * 66)
    for n in sorted({5, n_now, 25}):
        mde = pairwise_mde(sigma, n)
        p08 = pairwise_power(sigma, n, 0.8)
        print(f"   n={n:<3} seeds/group:  MDE(d) at 80% power = {mde:.2f}"
              f"   |  power to detect d=0.8 = {p08:.2f}"
              f"   |  MDE in gap units = {mde*sigma:.3f}")

    print("\n" + "=" * 66)
    print("(B) TREND test  OLS slope of gap on rho (pools all seeds x all rho)")
    print("=" * 66)
    for n in sorted({5, n_now, 25}):
        mde_slope = trend_mde(sigma, n)
        # translate slope MDE into the gap change across the rho range [0.05,1]
        span = mde_slope * (RHOS.max() - RHOS.min())
        print(f"   n={n:<3} seeds/rho:  MDE(slope) at 80% = {mde_slope:.3f}"
              f"   |  = gap change of {span:.3f} across rho in [0.05,1]"
              f"   (d={span/sigma:.2f})")

    print("\nInterpretation:")
    print("  At n=5 the pairwise MDE is ~d=2.0 -- the pre-registered design could")
    print("  only have ruled out enormous effects, so 'd<0.8 => null' was under-")
    print("  powered. At n=25 the pairwise MDE reaches ~d=0.8, matching the pre-")
    print("  registered threshold, and the trend test is far stronger still. The")
    print("  expanded sweep is what makes the null a claim rather than an absence.")


if __name__ == "__main__":
    main()
