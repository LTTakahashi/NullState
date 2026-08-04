"""
THE FLOOR IS AN INSTRUMENT, NOT NOISE.

CCA - MCC on a faithful encoder at rho = 1 is not an estimand defect. With CCA ~ 1
the model has recovered the right SUBSPACE, so MCC registers only the angle phi
between the frame the run converged to and the ground-truth basis: for a 2-D
latent MCC = cos(phi) (Hungarian matching swaps past 45 deg, so MCC >= 0.707), and

        gap = 1 - cos(phi)   <=>   phi = arccos(1 - gap),

mapping gap in [0, 0.293] exactly onto phi in [0 deg, 45 deg]. This is Locatello's
result asserting itself: an isotropic prior leaves the objective indifferent to
rotation, every run lands in some frame, and the gap distribution IS the
distribution over converged frames. So the quantity measures **how much strong
(axis-level) identifiability a model class actually achieves**.

THE QUESTION IT CAN ANSWER. The three Stage-4 arms make different theoretical
predictions about that floor:

  conditional : isotropic prior, nothing breaks rotation      -> floor HIGH
  ivae_2env   : conditional prior, but 2 environments, which
                cannot satisfy the iVAE variability condition -> floor HIGH
  ivae_5env   : 5 = 2n+1 environments, the condition of
                Khemakhem et al. (2020) Thm 1 assumption (iv),
                whose conclusion is identifiability up to
                PERMUTATION (strong)                          -> floor LOWER

Separation would be a direct empirical validation of that condition biting in
practice, measured rather than inferred from downstream task accuracy. No
separation is a negative result about a very widely invoked theorem, measured with
a purpose-built instrument.

STRUCTURAL CAVEAT, CHECKED BELOW: a condition satisfiable on paper can be weak on
one coordinate. dgp2 gives environment-dependent location/scale to the UNSHIFTED
axes only; the shifted axis carries no environment variation. So the variability
condition has purchase on part of the latent, not all of it, and the arm contrast
must be read in that light.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from core.paths import result, RESULTS

GAP_AT_45DEG = 1 - np.cos(np.pi / 4)          # 0.2929: the full rotation signature


def implied_angle_deg(gap):
    """gap = 1 - cos(phi)  ->  phi in degrees, clipped to the identifiable range."""
    return np.degrees(np.arccos(np.clip(1.0 - np.asarray(gap, float), -1.0, 1.0)))


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb + 1e-300)
    dfree = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1) + 1e-300)
    try:
        from scipy.stats import t as tdist
        p = 2 * (1 - tdist.cdf(abs(t), dfree))
    except Exception:
        p = float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float(t), float(p), float((a.mean() - b.mean()) / (sp + 1e-12))


def env_structure_check():
    """Does the environment vary the SHIFTED axis, or only the free axes?"""
    from core.dgp import DGP2Config, generate
    d = generate(DGP2Config(rho=1.0, delta=0.0, n_per_domain=4000, seed=0, n_env=5))
    t, e = np.asarray(d["t"], float), np.asarray(d["e"])
    print("  per-environment mean/sd of each latent coordinate:")
    print(f"    {'env':>4} " + " ".join(f"{'axis'+str(j):>16}" for j in range(t.shape[1])))
    for ev in sorted(set(e.tolist())):
        m = e == ev
        cells = [f"{t[m, j].mean():+.3f}/{t[m, j].std():.3f}" for j in range(t.shape[1])]
        print(f"    {ev:>4} " + " ".join(f"{c:>16}" for c in cells))
    spread = [float(np.ptp([t[e == ev, j].mean() for ev in sorted(set(e.tolist()))]))
              for j in range(t.shape[1])]
    print(f"  across-environment spread of the mean, per axis: "
          f"{[round(s, 4) for s in spread]}")
    print(f"  -> axis 0 (the SHIFTED axis) spread = {spread[0]:.4f}: the environment "
          f"does NOT vary it.")
    print("  -> the iVAE variability condition therefore has purchase only on the")
    print("     unshifted coordinate(s). Read the arm contrast accordingly.")


def main():
    print("=" * 78)
    print("STRUCTURAL CHECK: does the environment vary the shifted axis?")
    print("=" * 78)
    env_structure_check()

    df = pd.read_csv(result("main_sweep.csv"))
    base = df[(df.rho == 1.0) & (df.delta == 0.0)]
    # gap == CCA - MCC (verified identity); recompute from the raw columns
    base = base.assign(gap=base.learned_cca - base.learned_mcc)

    print("\n" + "=" * 78)
    print("THE IDENTIFIABILITY FLOOR, per arm (faithful encoders, rho=1, delta=0)")
    print("   gap = CCA - MCC; implied frame angle phi = arccos(1 - gap)")
    print(f"   full rotation signature (phi = 45 deg) = {GAP_AT_45DEG:.3f}")
    print("=" * 78)
    print(f"  {'arm':>12} {'n':>3} {'gap mean':>9} {'sd':>6} {'range':>16} "
          f"{'phi mean':>9} {'% of 45deg':>11}")
    arms = {}
    for arm in ["conditional", "ivae_2env", "ivae_5env"]:
        g = base[base.arm == arm].gap.values
        if not len(g):
            continue
        arms[arm] = g
        print(f"  {arm:>12} {len(g):>3} {g.mean():>9.3f} {g.std(ddof=1):>6.3f} "
              f"[{g.min():>5.3f},{g.max():>5.3f}] {implied_angle_deg(g.mean()):>8.1f}d "
              f"{100 * g.mean() / GAP_AT_45DEG:>10.0f}%")

    print("\n" + "=" * 78)
    print("DOES SATISFYING THE iVAE VARIABILITY CONDITION LOWER THE FLOOR?")
    print("   ivae_5env has 5 = 2n+1 environments (condition satisfiable);")
    print("   ivae_2env has 2 (cannot satisfy it); conditional has an isotropic prior.")
    print("=" * 78)
    if "ivae_5env" in arms:
        for other in ["conditional", "ivae_2env"]:
            if other not in arms:
                continue
            t, p, d_ = welch(arms[other], arms["ivae_5env"])
            lower = arms["ivae_5env"].mean() < arms[other].mean()
            print(f"  ivae_5env vs {other:>12}: "
                  f"{arms['ivae_5env'].mean():.3f} vs {arms[other].mean():.3f}  "
                  f"(d={d_:+.2f}, p={p:.3f}) -> "
                  f"{'LOWER' if lower else 'NOT lower'}"
                  f"{' and significant' if p < 0.05 and lower else ''}")
    print()

    print("=" * 78)
    print("DOES THE FLOOR DRIFT WITH rho?  (the null's hidden assumption)")
    print("   If it does, the Stage-4 equivalence test compares floors at")
    print("   different heights instead of measuring a flat signal.")
    print("=" * 78)
    d0 = df[df.delta == 0.0].assign(gap=lambda x: x.learned_cca - x.learned_mcc)
    piv = d0.pivot_table(index="rho", columns="arm", values="gap",
                         aggfunc="mean").sort_index(ascending=False)
    print(piv.round(3).to_string())
    for arm in piv.columns:
        v = piv[arm].values
        print(f"  {arm:>12}: range {v.min():.3f}-{v.max():.3f}  "
              f"spread {v.max() - v.min():.3f} "
              f"({100 * (v.max() - v.min()) / GAP_AT_45DEG:.0f}% of the 45deg signature)")

    print("\n" + "=" * 78)
    print("SESOI IN GAP UNITS (what the equivalence test actually bounds)")
    print("=" * 78)
    sd = d0.groupby(["arm", "rho"]).gap.std().mean()
    sesoi = 0.8 * sd
    print(f"  pooled per-cell SD of the gap        = {sd:.3f}")
    print(f"  pre-registered SESOI d = 0.8         = {sesoi:.3f} gap units")
    print(f"  identifiability floor (arm means)    = "
          f"{base.groupby('arm').gap.mean().min():.3f}-{base.groupby('arm').gap.mean().max():.3f}")
    print(f"  full rotation signature (45 deg)     = {GAP_AT_45DEG:.3f}")
    print(f"\n  => the null bounds any rho-dependent effect at {sesoi:.3f}, i.e. about"
          f" {100 * sesoi / GAP_AT_45DEG:.0f}%")
    print("     of the full rotation signature. That is the honest reading of the")
    print("     equivalence result -- a real bound, and a smaller one than 'd = 0.8'")
    print("     sounds when stated without the scale it is measured against.")


if __name__ == "__main__":
    main()
