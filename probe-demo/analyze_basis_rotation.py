"""
Analysis of the basis-dependence test.

Q1  Does the entanglement gap FIRE for an alignment arm once the shifted
    direction is rotated off the ground-truth coordinate axis? (The taxonomy
    predicts ~0.14 for a collapse of a rotated axis; if so, our headline
    Demonstration 2 would be an artifact of the DGP's basis choice.)
Q2  How much of any gap is basis/seed ARBITRARINESS rather than entanglement?
    Measured on the faithful arm at rho=1, where the true gap is 0.
Q3  Is the theta>0 excess specifically basis MISMATCH? Score each embedding
    against both candidate ground truths -- the declared rotated basis and the
    independent-component basis -- and take the better.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd

TAXONOMY_ROT45 = 0.293      # the gap a genuine 45-degree rotation produces


def load():
    fs = sorted(glob.glob("results_basis_rotation_s*.csv"))
    d = (pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
         if fs else pd.read_csv("results_basis_rotation.csv"))
    return d.drop_duplicates(["arm", "theta", "rho", "seed"])


def main():
    d = load()
    d["gap_best"] = d[["gap_s", "gap_ab"]].min(axis=1)
    th45 = np.isclose(d.theta, np.pi / 4)
    print(f"{len(d)} runs, {d.seed.nunique()} seeds\n")

    print("=" * 74)
    print("Q1  does the gap fire for the alignment arm at theta = 45 deg?")
    print("=" * 74)
    print(f"    {'rho':>5} {'faithful':>10} {'mix_1000':>10} {'excess':>9}")
    exc = []
    for rho in sorted(d.rho.unique(), reverse=True):
        f = d[(d.arm == "faithful") & th45 & (d.rho == rho)].gap_s.mean()
        m = d[(d.arm == "mix_1000") & th45 & (d.rho == rho)].gap_s.mean()
        exc.append(m - f)
        print(f"    {rho:>5} {f:>10.3f} {m:>10.3f} {m - f:>+9.3f}")
    print(f"\n    verdict: {'FIRES' if max(exc) > 0.05 else 'DOES NOT FIRE'} "
          f"(max excess {max(exc):+.3f}); the predicted ~0.14 does not appear.")
    print("    => the most obvious attack on Demonstration 2 is CLOSED: the gap's")
    print("       blind spot is not merely an artifact of the DGP putting the")
    print("       shifted direction on a coordinate axis.\n")

    print("=" * 74)
    print("Q2  how much of the gap is basis/seed arbitrariness?")
    print("    faithful arm, rho = 1.0: perfect overlap, no folding, TRUE gap = 0")
    print("=" * 74)
    for th, lab in [(0.0, "theta= 0 (shift ON an axis)"),
                    (np.pi / 4, "theta=45 (shift OFF axis)")]:
        s = d[(d.arm == "faithful") & np.isclose(d.theta, th) & (d.rho == 1.0)].gap_s.values
        print(f"    {lab}: mean {s.mean():.3f}  range [{s.min():.3f}, {s.max():.3f}]"
              f"  {100 * (s > 0.15).mean():.0f}% of runs > 0.15")
    s45 = d[(d.arm == "faithful") & th45 & (d.rho == 1.0)].gap_s.values
    print(f"\n    a genuine 45-deg rotation scores {TAXONOMY_ROT45:.3f}")
    print(f"    a PERFECTLY RECOVERING model reaches  {s45.max():.3f} on some seeds")
    print("    => a SINGLE-RUN gap cannot be distinguished from the entanglement")
    print("       signature. The estimand is interpretable only in aggregate, and")
    print("       even in aggregate its null baseline is ~0.09-0.15, not 0.\n")

    print("=" * 74)
    print("Q3  is the theta>0 excess basis MISMATCH? (score vs the basis found)")
    print("=" * 74)
    for arm in ["faithful", "mix_1000"]:
        a = d[(d.arm == arm) & th45]
        print(f"    {arm:>9}: declared-basis gap {a.gap_s.mean():.3f}"
              f"  ->  best-of-two-bases {a.gap_best.mean():.3f}")
    print("    => most of it is which basis the run happened to land in.\n")

    print("=" * 74)
    print("WHAT THIS MEANS")
    print("=" * 74)
    print("""  The predicted failure did not occur, so Demonstration 2 survives its most
  obvious attack. A larger limitation surfaced instead, and it is ours: the
  entanglement gap = CCA - MCC inherits basis-dependence entirely through MCC
  (CCA is basis-free; Hungarian matching is not). When the ground-truth basis is
  not the basis a run happens to converge to, the reading is dominated by that
  arbitrary mismatch -- reaching the value of a genuine 45-degree rotation on a
  model that is recovering perfectly.

  This is the sharpest form of the thesis: an invariance class is not a property
  of the metric alone, but of the metric PLUS a coordinate choice -- and on real
  data, where no privileged basis for "the true latent" exists, that choice has
  no observable counterpart. It also re-reads our own Stage-4 result: the
  0.05-0.14 band reported there is the estimand's noise floor, not a signal. The
  null (no trend in rho) is unaffected; the baseline is simply not zero.""")


if __name__ == "__main__":
    main()
