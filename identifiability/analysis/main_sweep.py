"""
STAGE-4 ANALYSIS, against the decision rule fixed BEFORE the results existed.

  supported : gap monotone in rho AND Cohen's d >= 0.8 between rho=1 and rho=0.2
              AND flat in delta (d < 0.2 across delta at rho=1)
  refuted   : gap flat in rho, or gap moves with delta
  threshold : "cliff"/"phase transition" language permitted ONLY if a two-segment
              fit beats a linear one on BIC AND the slope ratio exceeds 3.
              Default expectation is SMOOTH degradation -- no theory predicts a
              discontinuity in identifiability as a function of support overlap.

The BIC test is included because the v1 write-up claimed a phase transition that
its own data contradicted: BIC preferred linear (-107.24) over every two-segment
fit (-104.72), with piecewise slopes 0.949 vs 0.964.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import sys
import numpy as np
import pandas as pd
from core.paths import result, RESULTS

CSV = sys.argv[1] if len(sys.argv) > 1 else result("main_sweep_pilot.csv")
METRIC = "gap_mcc"


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bic(resid, k, n):
    rss = float(np.sum(resid ** 2))
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def linear_vs_segmented(x, y):
    """Compare a linear fit to the best two-segment (continuous) fit by BIC."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    b1 = bic(y - np.polyval(np.polyfit(x, y, 1), x), 3, n)
    best = (np.inf, None, None)
    for bp in np.unique(x)[1:-1]:
        h = np.maximum(x - bp, 0.0)                # continuous hinge basis
        A = np.column_stack([np.ones(n), x, h])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        b2 = bic(y - A @ coef, 4, n)
        if b2 < best[0]:
            best = (b2, bp, coef)
    b2, bp, coef = best
    ratio = abs((coef[1] + coef[2]) / coef[1]) if coef is not None and abs(coef[1]) > 1e-9 else np.nan
    return dict(bic_linear=b1, bic_segmented=b2, breakpoint=bp,
                slope_ratio=float(ratio), segmented_wins=bool(b2 < b1))


def main():
    df = pd.read_csv(CSV)
    print(f"{len(df)} rows | arms: {sorted(df.arm.unique())} | "
          f"rho: {sorted(df.rho.unique())} | delta: {sorted(df.delta.unique())}\n")

    bad = df[(df.posterior_collapse) | (df.deterministic_ae)]
    if len(bad):
        print(f"!! {len(bad)} run(s) at a degenerate VAE corner -- excluded:")
        print(bad[["arm", "rho", "delta", "seed", "kl_total", "active_units"]].to_string(index=False))
        df = df.drop(bad.index)
    print()

    for arm in sorted(df.arm.unique()):
        A = df[df.arm == arm]
        print("=" * 72)
        print(f"ARM: {arm}")
        print("=" * 72)

        for delta in sorted(A.delta.unique()):
            S = A[A.delta == delta]
            print(f"\n  --- delta = {delta} ---")
            print(f"  {'rho':>5} {'n':>3} {'gap':>8} {'95% CI':>18} "
                  f"{'L_mcc':>7} {'O_mcc':>7} {'L_cca':>7}")
            for rho in sorted(S.rho.unique(), reverse=True):
                g = S[S.rho == rho]
                lo, hi = boot_ci(g[METRIC].values)
                print(f"  {rho:>5} {len(g):>3} {g[METRIC].mean():>8.4f} "
                      f"[{lo:>7.4f},{hi:>7.4f}] {g.learned_mcc.mean():>7.4f} "
                      f"{g.oracle_mcc.mean():>7.4f} {g.learned_cca.mean():>7.4f}")

            hi_r = S[S.rho == S.rho.max()][METRIC].values
            lo_r = S[S.rho == S.rho.min()][METRIC].values
            d = cohens_d(lo_r, hi_r)
            means = [S[S.rho == r][METRIC].mean() for r in sorted(S.rho.unique(), reverse=True)]
            mono = all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1))
            print(f"\n  H1 gap grows as rho falls: monotone={mono}, "
                  f"Cohen's d(rho_min vs rho_max) = {d:.3f} "
                  f"{'>= 0.8 OK' if d >= 0.8 else '< 0.8'}")
            seg = linear_vs_segmented(S.rho.values, S[METRIC].values)
            print(f"  shape: BIC linear {seg['bic_linear']:.2f} vs segmented "
                  f"{seg['bic_segmented']:.2f} -> "
                  f"{'SEGMENTED' if seg['segmented_wins'] else 'LINEAR'} "
                  f"(bp={seg['breakpoint']}, slope ratio {seg['slope_ratio']:.2f})")
            if seg["segmented_wins"] and seg["slope_ratio"] > 3:
                print("  -> 'threshold' language is licensed for this arm/delta.")
            else:
                print("  -> report as SMOOTH degradation; 'cliff' language NOT licensed.")

        # H2: delta-invariance at full overlap
        if len(A.delta.unique()) > 1:
            print(f"\n  --- H2: gap flat in delta (at rho = 1.0) ---")
            F = A[A.rho == 1.0]
            ds = sorted(F.delta.unique())
            for dd in ds:
                v = F[F.delta == dd][METRIC].values
                lo, hi = boot_ci(v)
                print(f"    delta={dd}: gap {v.mean():>8.4f}  [{lo:.4f},{hi:.4f}]  n={len(v)}")
            d2 = cohens_d(F[F.delta == ds[-1]][METRIC].values,
                          F[F.delta == ds[0]][METRIC].values)
            print(f"    Cohen's d across delta = {d2:.3f} "
                  f"{'< 0.2 -> FLAT, H2 supported' if abs(d2) < 0.2 else '>= 0.2 -> NOT flat, H2 violated'}")

        # verdict
        S0 = A[A.delta == A.delta.min()]
        d_rho = cohens_d(S0[S0.rho == S0.rho.min()][METRIC].values,
                         S0[S0.rho == S0.rho.max()][METRIC].values)
        F = A[A.rho == 1.0]
        ds = sorted(F.delta.unique())
        d_del = (cohens_d(F[F.delta == ds[-1]][METRIC].values,
                          F[F.delta == ds[0]][METRIC].values) if len(ds) > 1 else 0.0)
        print(f"\n  VERDICT [{arm}]: ", end="")
        if d_rho >= 0.8 and abs(d_del) < 0.2:
            print("SUPPORTED (gap grows with falling rho, flat in delta)")
        elif d_rho < 0.8:
            print(f"NOT SUPPORTED -- gap does not grow with falling rho "
                  f"(d = {d_rho:.2f} < 0.8). Report the honest null.")
        else:
            print(f"CONFOUNDED -- gap grows with rho (d={d_rho:.2f}) but ALSO "
                  f"moves with delta (d={d_del:.2f}). Not attributable to overlap.")
        print()

    # arm contrast: does satisfying the iVAE variability condition help?
    if {"ivae_5env", "ivae_2env"} <= set(df.arm.unique()):
        print("=" * 72)
        print("ARM CONTRAST: does satisfying iVAE assumption (iv) matter?")
        print("  5 env = 2n+1 satisfies it; 2 env provably cannot (v1 sat here).")
        print("=" * 72)
        for delta in sorted(df.delta.unique()):
            print(f"\n  delta={delta}")
            print(f"  {'rho':>5} {'5env gap':>10} {'2env gap':>10} {'diff':>8} {'d':>7}")
            for rho in sorted(df.rho.unique(), reverse=True):
                a = df[(df.arm == "ivae_5env") & (df.rho == rho) & (df.delta == delta)][METRIC].values
                b = df[(df.arm == "ivae_2env") & (df.rho == rho) & (df.delta == delta)][METRIC].values
                if len(a) and len(b):
                    print(f"  {rho:>5} {a.mean():>10.4f} {b.mean():>10.4f} "
                          f"{b.mean()-a.mean():>8.4f} {cohens_d(b, a):>7.2f}")


if __name__ == "__main__":
    main()
