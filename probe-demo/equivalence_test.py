"""
EQUIVALENCE TEST (TOST) for the recovery-invariance null.

A non-significant p is not evidence of absence. To claim the matched-oracle MCC
gap does NOT grow with falling overlap, we need an equivalence test: reject the
presence of an effect at least as large as a pre-specified smallest effect of
interest (SESOI). The pre-registration named d = 0.8 as the threshold that would
count as "supported", so d = 0.8 is exactly the right SESOI: TOST against it asks
"is the true effect smaller than the effect we pre-declared we cared about?"

Two one-sided tests (Schuirmann): for the difference D = gap(rho=0.05) - gap(rho=1),
with SESOI bound Delta = 0.8 * pooled_SD,
  reject D <= -Delta   if  (D + Delta)/SE >  t_crit
  reject D >= +Delta   if  (D - Delta)/SE < -t_crit
Equivalence (|effect| < SESOI) is concluded when BOTH reject at alpha, i.e. the
90% CI of D lies entirely inside (-Delta, +Delta).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd


def t_crit(df, alpha=0.05):
    try:
        from scipy.stats import t
        return float(t.ppf(1 - alpha, df))          # one-sided
    except Exception:
        return 1.70 if df > 40 else 1.75


def tost(a, b, sesoi_d, alpha=0.05):
    """a = gap at rho=0.05 (n), b = gap at rho=1 (n). Equivalence vs +/- sesoi_d
    (in Cohen's d units). Returns dict with the verdict and the 90% CI."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    se = sp * np.sqrt(1 / na + 1 / nb)
    D = a.mean() - b.mean()
    Delta = sesoi_d * sp                              # SESOI in gap units
    df = na + nb - 2
    tc = t_crit(df, alpha)
    t1 = (D + Delta) / se                             # test D <= -Delta
    t2 = (D - Delta) / se                             # test D >= +Delta
    equiv = (t1 > tc) and (t2 < -tc)
    ci = (D - tc * se, D + tc * se)                   # 90% CI (1-2alpha)
    return dict(D=float(D), d=float(D / sp), se=float(se), sp=float(sp),
                Delta=float(Delta), sesoi_d=sesoi_d, ci90=(float(ci[0]), float(ci[1])),
                equivalent=bool(equiv), t1=float(t1), t2=float(t2))


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "results_stage4_n25.csv"
    sesoi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8   # pre-registered d
    df = pd.read_csv(csv)
    lo, hi = df.rho.min(), df.rho.max()
    print(f"data: {csv}  ({df.seed.nunique()} seeds)   SESOI = d {sesoi}")
    print(f"TOST on D = gap(rho={lo}) - gap(rho={hi}), delta=0, per arm\n")
    print(f"{'arm':>12} {'n':>3} {'D(gap)':>8} {'d':>6} {'90% CI':>20} "
          f"{'Delta':>7} {'verdict':>12}")
    for arm in sorted(df.arm.unique()):
        s = df[(df.arm == arm) & (df.delta == 0)]
        a = s[s.rho == lo].gap_mcc.values
        b = s[s.rho == hi].gap_mcc.values
        if len(a) < 3 or len(b) < 3:
            print(f"{arm:>12}  insufficient rows"); continue
        r = tost(a, b, sesoi)
        v = "EQUIVALENT" if r["equivalent"] else "inconclusive"
        print(f"{arm:>12} {len(a):>3} {r['D']:>+8.3f} {r['d']:>+6.2f} "
              f"[{r['ci90'][0]:>+7.3f},{r['ci90'][1]:>+7.3f}] {r['Delta']:>7.3f} {v:>12}")
    print(f"\nEQUIVALENT => the gap change from rho={hi} to rho={lo} is significantly")
    print(f"SMALLER than the pre-registered d={sesoi} effect: a positive null, not an absence.")


if __name__ == "__main__":
    main()
