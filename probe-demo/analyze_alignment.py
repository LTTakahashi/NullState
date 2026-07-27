"""
Analysis for the alignment build-out. Merges shards and reports:

  1. CCA by arm x rho (mean, 95% CI) -- the primary recovery metric.
  2. faithful arm: TOST that the CCA drop (rho=1 -> rho=0.05) is EQUIVALENT to
     zero within a SESOI (recovery is overlap-invariant) -- the control.
  3. alignment arms: the CCA drop with Cohen's d, and a slope test of CCA on rho
     (a significantly positive slope == recovery falls as overlap falls).
  4. COLLAPSE PROBABILITY: P(CCA < 0.85 | arm, rho) -- the effect is stochastic
     (some seeds fold the axis), so we characterise the rising collapse rate.
  5. matched-oracle GAP by arm x rho -- to show it is BLIND to this failure
     (flat while CCA collapses): the v2 estimand is the wrong tool here.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd

SESOI_CCA = 0.05        # smallest CCA drop worth caring about
COLLAPSE = 0.85         # CCA below this == recovery collapsed


def load():
    fs = glob.glob("results_alignment_s*.csv")
    if not fs:
        fs = ["results_alignment.csv"]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return df.drop_duplicates(["arm", "rho", "seed"])


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return float((a.mean()-b.mean())/(sp+1e-12))


def t_crit(dfree, alpha):
    try:
        from scipy.stats import t
        return float(t.ppf(1-alpha, dfree))
    except Exception:
        return 1.68


def slope_test(rhos, y):
    """OLS slope of y on rho; return slope and two-sided p (t)."""
    x = np.asarray(rhos, float); y = np.asarray(y, float)
    xc = x - x.mean(); Sxx = (xc**2).sum(); dfree = len(x)-2
    b = (xc*(y-y.mean())).sum()/Sxx
    resid = y - (y.mean()+b*xc)
    se = np.sqrt((resid**2).sum()/dfree/Sxx)
    tval = b/(se+1e-12)
    try:
        from scipy.stats import t
        p = 2*(1-t.cdf(abs(tval), dfree))
    except Exception:
        p = float("nan")
    return float(b), float(p)


def tost(a, b, sesoi, alpha=0.05):
    """Equivalence of mean(a)-mean(b) within +/- sesoi (raw units)."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2))
    se = sp*np.sqrt(1/na+1/nb); D = a.mean()-b.mean(); dfree = na+nb-2
    tc = t_crit(dfree, alpha)
    equiv = ((D+sesoi)/se > tc) and ((D-sesoi)/se < -tc)
    return D, (D-tc*se, D+tc*se), bool(equiv)


def main():
    df = load()
    arms = [a for a in ["faithful","mix_200","mix_1000","adv_100"] if a in df.arm.unique()]
    rhos = sorted(df.rho.unique(), reverse=True)
    print(f"{len(df)} rows | arms {arms} | "
          f"seeds/cell {df.groupby(['arm','rho']).seed.nunique().min()}-"
          f"{df.groupby(['arm','rho']).seed.nunique().max()}\n")

    print("=== CCA by arm x rho (mean [95% CI]) ===")
    print(f"{'rho':>5} " + " ".join(f"{a:>18}" for a in arms))
    for rho in rhos:
        cells = []
        for a in arms:
            v = df[(df.arm==a)&(df.rho==rho)].cca.values
            if len(v) == 0:
                cells.append("--"); continue
            lo, hi = boot_ci(v)
            cells.append(f"{v.mean():.2f}[{lo:.2f},{hi:.2f}]")
        print(f"{rho:>5} " + " ".join(f"{c:>18}" for c in cells))

    print("\n=== CCA drop (rho=1.0 -> rho=0.05), slope, collapse ===")
    hi_r, lo_r = max(rhos), min(rhos)
    for a in arms:
        s = df[df.arm==a]
        hi = s[s.rho==hi_r].cca.values; lo = s[s.rho==lo_r].cca.values
        if len(hi) < 3 or len(lo) < 3:
            print(f"  {a:>9}: incomplete ({len(hi)},{len(lo)} at endpoints)"); continue
        D, ci, equiv = tost(hi, lo, SESOI_CCA)      # drop = cca(hi)-cca(lo)
        d = cohens_d(hi, lo)
        sl, p = slope_test(s.rho.values, s.cca.values)
        verdict = ("EQUIVALENT(flat)" if equiv else
                   f"slope+={sl:.3f} p={p:.1e}" if sl > 0 else "n.s.")
        print(f"  {a:>9}: drop={D:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  d={d:+.2f}  "
              f"slope(cca~rho)={sl:+.3f} (p={p:.1e})  -> {verdict}")

    print("\n=== collapse probability P(CCA < 0.85) by arm x rho ===")
    print(f"{'rho':>5} " + " ".join(f"{a:>9}" for a in arms))
    for rho in rhos:
        cells = [f"{(df[(df.arm==a)&(df.rho==rho)].cca < COLLAPSE).mean():.2f}" for a in arms]
        print(f"{rho:>5} " + " ".join(f"{c:>9}" for c in cells))

    print("\n=== matched-oracle GAP by arm x rho (should be flat/blind) ===")
    print(f"{'rho':>5} " + " ".join(f"{a:>9}" for a in arms))
    for rho in rhos:
        cells = [f"{df[(df.arm==a)&(df.rho==rho)].gap.mean():.3f}" for a in arms]
        print(f"{rho:>5} " + " ".join(f"{c:>9}" for c in cells))

    print("\nRead: faithful CCA flat + EQUIVALENT = overlap-invariant recovery (v2).")
    print("Alignment arms: CCA drop large, slope(cca~rho) significantly +, collapse")
    print("probability rising as rho falls. GAP flat throughout = wrong metric for")
    print("this information-loss failure; raw CCA is the right one.")


if __name__ == "__main__":
    main()
