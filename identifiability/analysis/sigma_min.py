"""
Does the identifiability floor respond to sigma_min(L) at fixed n_env?

Three readouts, in increasing strength:
  1. floor vs sigma_min: trend and Spearman.
  2. one-sample KS against the uniform-random-frame law at each level -- the only
     test that can convert "lower floor" into "better than a random frame".
  3. the same for the best-conditioned level alone, which is the single point with
     the most power to reject.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import glob
import numpy as np
import pandas as pd
from core.paths import result, RESULTS
from analysis.random_frame_null import analytic_moments, ks_test, GAP_MAX


def load():
    fs = sorted(glob.glob(str(RESULTS / "sigma_min_s*.csv")))
    d = (pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
         if fs else pd.read_csv(result("sigma_min.csv")))
    return d.drop_duplicates(["s_level", "seed"])


def main():
    d = load()
    d.to_csv(result("sigma_min.csv"), index=False)
    m, sd = analytic_moments()
    print(f"{len(d)} runs at n_env = 9 (condition nominally satisfied throughout)")
    print(f"random-frame law: E[gap] = {m:.4f}, SD = {sd:.4f}, "
          f"range [0, {GAP_MAX:.4f}]\n")

    print("=" * 84)
    print("FLOOR vs sigma_min(L)   -- total prior variability held constant")
    print("=" * 84)
    print(f"{'s':>6} {'sigma_min':>10} {'kappa':>8} {'n':>3} {'CCA':>6} {'gap':>7} "
          f"{'sd':>6} {'phi':>7} {'KS p':>7}")
    for s_level in sorted(d.s_level.unique()):
        g = d[d.s_level == s_level]
        _, p = ks_test(g.gap.values)
        print(f"{s_level:>6} {g.sigma_min.mean():>10.3f} {g.kappa.mean():>8.1f} "
              f"{len(g):>3} {g.cca.mean():>6.3f} {g.gap.mean():>7.3f} "
              f"{g.gap.std(ddof=1):>6.3f} {g.phi_deg.mean():>7.1f} {p:>7.3f}")

    try:
        from scipy.stats import spearmanr, ttest_ind
        rs = spearmanr(d.sigma_min, d.gap)
        print(f"\nspearman(sigma_min, gap) = {rs.statistic:+.3f}  (p = {rs.pvalue:.3f})")
        lo = d[d.s_level <= 0.1].gap.values
        hi = d[d.s_level >= 0.6].gap.values
        t = ttest_ind(lo, hi, equal_var=False)
        sp = np.sqrt(((len(lo) - 1) * lo.var(ddof=1) + (len(hi) - 1) * hi.var(ddof=1))
                     / (len(lo) + len(hi) - 2))
        print(f"ill-conditioned {lo.mean():.3f} vs well-conditioned {hi.mean():.3f}: "
              f"d = {(lo.mean() - hi.mean()) / (sp + 1e-12):+.2f}, p = {t.pvalue:.3f}")
    except Exception:
        pass

    print("\n" + "=" * 84)
    print("THE DECISIVE TEST: does ANY conditioning level beat a random frame?")
    print("=" * 84)
    best = d[d.s_level == max(d.s_level.unique())]
    _, p_best = ks_test(best.gap.values)
    _, p_pool = ks_test(d[d.s_level >= 0.6].gap.values)
    print(f"  best-conditioned level (s = {max(d.s_level.unique())}, "
          f"sigma_min ~ {best.sigma_min.mean():.2f}, kappa ~ {best.kappa.mean():.1f}):")
    print(f"      n = {len(best)}, mean gap {best.gap.mean():.4f}, KS p = {p_best:.4f}")
    print(f"  well-conditioned pooled (s >= 0.6): n = {len(d[d.s_level >= 0.6])}, "
          f"KS p = {p_pool:.4f}")
    verdict = ("REJECTS the random-frame law -- conditioning buys real axis-level "
               "identifiability" if p_best < 0.05 or p_pool < 0.05 else
               "does NOT reject -- even a well-conditioned L leaves the frame "
               "indistinguishable from random")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
