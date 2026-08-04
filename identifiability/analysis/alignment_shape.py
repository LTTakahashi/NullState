"""
SHAPE of the alignment collapse -- applying the v2 discipline to the v3 result.

In v2 we forbade "threshold"/"cliff" language unless a segmented fit beat a linear
one on BIC. The v3 alignment result must be held to the same standard, and it has
a specific problem the endpoint contrast hides: the INTERIOR IS NOT MONOTONE.

    mix_1000 : 0.99  0.87  0.74  0.73  0.75  0.70   (rho = 1.0 .. 0.05)
    adv_100  : 0.99  0.98  0.91  0.93  0.94  0.89

rho=0.20 sits ABOVE rho=0.60 in two of three alignment arms. Endpoint effect sizes
are large and slope p-values are tiny, but non-monotone interiors are exactly the
shape a reviewer reads as endpoint-picking. So we report:

  1. the full per-rho curve with CIs (no endpoint-only summary),
  2. Spearman rank correlation of CCA with rho (monotonicity, all points),
  3. the largest monotonicity VIOLATION (worst inversion) per arm,
  4. linear vs segmented (continuous two-piece) fit by BIC -- and we only permit
     threshold language where segmented wins,
  5. an explicit saturation test: is the curve better described as
     "drop then plateau" than as a continuous decline?
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import glob
import numpy as np
import pandas as pd
from core.paths import result, RESULTS

ARM_ORDER = ["faithful", "mix_200", "mix_1000", "adv_100"]


def load(path=None):
    if path:
        return pd.read_csv(path)
    fs = sorted(glob.glob(str(RESULTS / "alignment_s*.csv")))
    if fs:
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) \
                 .drop_duplicates(["arm", "rho", "seed"])
    return pd.read_csv(result("alignment.csv"))


def bic(resid, k, n):
    rss = float(np.sum(resid ** 2))
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def linear_vs_segmented(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    b1 = bic(y - np.polyval(np.polyfit(x, y, 1), x), 3, n)
    best = (np.inf, None, None)
    for bp in np.unique(x)[1:-1]:
        h = np.maximum(x - bp, 0.0)
        A = np.column_stack([np.ones(n), x, h])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        b2 = bic(y - A @ coef, 4, n)
        if b2 < best[0]:
            best = (b2, bp, coef)
    b2, bp, coef = best
    ratio = (abs((coef[1] + coef[2]) / coef[1])
             if coef is not None and abs(coef[1]) > 1e-9 else np.nan)
    return dict(bic_linear=b1, bic_segmented=b2, breakpoint=bp,
                slope_ratio=float(ratio), segmented_wins=bool(b2 < b1))


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float)))
    ry = np.argsort(np.argsort(np.asarray(y, float)))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-12))


def boot_ci(v, n=5000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    m = r.choice(v, (n, len(v)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    df = load()
    arms = [a for a in ARM_ORDER if a in set(df.arm)]
    rhos = sorted(df.rho.unique(), reverse=True)

    print("=" * 78)
    print("SHAPE OF THE ALIGNMENT COLLAPSE (CCA vs rho) -- full curve, not endpoints")
    print("=" * 78)
    for a in arms:
        s = df[df.arm == a]
        means = [s[s.rho == r].cca.mean() for r in rhos]
        print(f"\n  {a}")
        print("    rho :  " + "  ".join(f"{r:>5.2f}" for r in rhos))
        print("    CCA :  " + "  ".join(f"{m:>5.3f}" for m in means))
        cis = [boot_ci(s[s.rho == r].cca.values) for r in rhos]
        print("    lo  :  " + "  ".join(f"{c[0]:>5.3f}" for c in cis))
        print("    hi  :  " + "  ".join(f"{c[1]:>5.3f}" for c in cis))

        # monotonicity over ALL pairs, not just adjacent ones: an inversion
        # between rho=0.20 and rho=0.60 is what a reviewer reads as
        # endpoint-picking, and adjacent-only checks miss it.
        m_of = dict(zip(rhos, means))
        inv = [(rl, rh, m_of[rl] - m_of[rh])
               for i, rh in enumerate(rhos) for rl in rhos[i + 1:]
               if m_of[rl] > m_of[rh] + 1e-12]          # lower rho scoring HIGHER
        rho_s = spearman(s.rho.values, s.cca.values)
        print(f"    spearman(CCA, rho) = {rho_s:+.3f} (all {len(s)} runs)")
        if inv:
            w = max(inv, key=lambda v: v[2])
            print(f"    NON-MONOTONE: {len(inv)} inversion(s) over all pairs; worst "
                  f"rho={w[0]:.2f} ABOVE rho={w[1]:.2f} by {w[2]:.3f}")
        else:
            print("    monotone in rho (no inversions among the means)")

        sh = linear_vs_segmented(s.rho.values, s.cca.values)
        winner = "SEGMENTED" if sh["segmented_wins"] else "LINEAR"
        print(f"    BIC linear {sh['bic_linear']:.1f} vs segmented "
              f"{sh['bic_segmented']:.1f} -> {winner} "
              f"(bp={sh['breakpoint']}, slope ratio {sh['slope_ratio']:.2f})")
        if sh["segmented_wins"] and sh["slope_ratio"] > 3:
            print("    -> threshold language LICENSED for this arm")
        else:
            print("    -> report as smooth/saturating; threshold language NOT licensed")

        # saturation: is 'drop then plateau' a fair description?
        hi_half = np.mean([m for r, m in zip(rhos, means) if r >= 0.6])
        lo_half = np.mean([m for r, m in zip(rhos, means) if r <= 0.4])
        print(f"    mean CCA rho>=0.6: {hi_half:.3f} | rho<=0.4: {lo_half:.3f} "
              f"| drop {hi_half - lo_half:+.3f}")

    print("\n" + "=" * 78)
    print("HONEST SUMMARY")
    print("=" * 78)
    print("""  The endpoint contrast (rho=1 vs rho=0.05) is large and the slope tests are
  highly significant, but the INTERIOR of the alignment curves is not monotone:
  rho=0.20 sits above rho=0.60 in two arms. The defensible claim is therefore
  'recovery degrades substantially once overlap is reduced, saturating at low
  overlap', NOT a clean monotone dose-response in rho and NOT a threshold.
  The dose-response that IS clean is in the ALIGNMENT STRENGTH (lambda), not rho.""")


if __name__ == "__main__":
    main()
