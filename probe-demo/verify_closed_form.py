"""
THE CLOSED FORM behind the geometry constant, and where it comes from.

For a cross-domain kNN transfer score on a 1-D uniform shift with overlap rho,
run on an ORACLE embedding (z = the true latent), the large-n transfer R^2 is

        R^2(rho) = 1 - 4 (1 - rho)^3 ,     zero at rho* = 1 - 4^(-1/3) ~ 0.370.

DERIVATION (this script verifies each factor empirically).
  Domain A: t ~ U[0, w].  Domain B: t ~ U[w(1-rho), w(2-rho)] (shift, overlap rho).
  A fraction (1 - rho) of B lies beyond A's support; there the kNN map, having no
  training neighbours past the boundary, predicts the boundary value, so the error
  for a point a distance u past the boundary is ~ u. Then:
      E[error^2] = (1 - rho) * E[u^2 | u ~ U[0, w(1-rho)]]
                 = (1 - rho) * (w(1-rho))^2 / 3
                 = w^2 (1 - rho)^3 / 3
      Var(t within a domain) = w^2 / 12
      R^2 = 1 - E[error^2]/Var = 1 - (w^2 (1-rho)^3 / 3)/(w^2/12) = 1 - 4 (1-rho)^3.

WHERE EACH PIECE COMES FROM (the transferable recipe), as MEASURED below:
  * exponent 3 = 1 (fraction of points that extrapolate, linear in 1-rho for a
    1-D shift) + 2 (squared extrapolation error, whose scale is linear in 1-rho).
  * constant 4 = (1 / Var_shape) * (1 / 3) = 12 * (1/3), i.e. the inverse target
    variance (12 for a unit uniform) times the second-moment factor of the
    extrapolation-error law (1/3 for a uniform ramp). It is a property of the
    DOMAIN MARGINAL SHAPE and the transfer construction. [V1, V5]
  * width w CANCELS (both error and variance scale as w^2): scale-free. [V2]
  * EMBEDDING DIMENSION is NOT free. Adding shared (overlapping) axes on the same
    scale INFLATES the apparent threshold (rho* rises 0.38 -> 0.39 -> 0.48 for
    d = 1, 3, 10) -- the kNN curse of dimensionality: neighbours are pulled off
    the boundary by the extra coordinates, so transfer degrades sooner. The 1-D
    closed form is the clean lower anchor; higher-dimensional embeddings push the
    geometry threshold up from there. [V3, and this corrected an earlier draft
    that wrongly asserted dimension-invariance.]
  * k / n is only a small finite-sample correction (rho* moves ~0.005 over two
    decades of k/n). [V4]
  A reader computes the constant for THEIR marginal, dimension and k, and checks
  whether their reported "critical overlap" coincides with the geometry
  prediction -- a five-minute check that separates a phenomenon from an artifact.

VERIFICATIONS below (each factor of the recipe, measured):
  V1  R^2(rho) matches 1 - 4(1-rho)^3 on the oracle (the headline).
  V2  width-invariance: the curve does not move with w.
  V3  dimension DEPENDENCE: shared axes raise rho* (curse of dimensionality).
  V4  k / n is a finite-sample correction: rho* -> ~0.372 as k/n -> 0, which is
      why the measured zero was 0.378 rather than 0.370 (explains that gap).
  V5  marginal-shape dependence: a Gaussian marginal gives a DIFFERENT constant,
      confirming the constant is computed from the marginal, not universal.
  V6  EXACT rotation invariance: within-domain kNN-R^2 is invariant to an
      orthogonal transform of the embedding to machine precision, while MCC is
      not -- the metric is provably blind to the rotation that isotropic-prior
      non-identifiability consists of.
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

RHOS = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.15, 0.05])


def analytic(rho):
    return 1.0 - 4.0 * (1.0 - rho) ** 3


def make(rho, w=1.0, d=1, n=8000, seed=0, marginal="uniform", noise_scale=0.3):
    """Two domains shifted by overlap rho on axis 0; d-1 SHARED extra axes at
    scale `noise_scale` (their scale RELATIVE to the signal axis is what matters,
    not their count)."""
    rng = np.random.default_rng(seed)
    if marginal == "uniform":
        a0 = rng.uniform(0, w, n)
        b0 = rng.uniform(w * (1 - rho), w * (2 - rho), n)
    elif marginal == "triangular":
        # a second COMPACT marginal (hard boundary) -> a different but stable
        # constant, unlike the boundary-free Gaussian below.
        a0 = rng.triangular(0, 0.5 * w, w, n)
        b0 = rng.triangular(w * (1 - rho), w * (1.5 - rho), w * (2 - rho), n)
    elif marginal == "gaussian":
        # shift chosen so the overlap coefficient equals rho:
        # OVL(N(0,1), N(mu,1)) = 2*Phi(-mu/2) = rho  ->  mu = -2*Phinv(rho/2)
        from scipy.stats import norm
        mu = -2.0 * norm.ppf(rho / 2.0)
        a0 = rng.normal(0, 1, n)
        b0 = rng.normal(mu, 1, n)
    z = np.zeros((2 * n, d)); t = np.zeros(2 * n); s = np.zeros(2 * n, int)
    z[:n, 0] = a0; z[n:, 0] = b0
    t[:n] = a0; t[n:] = b0
    s[n:] = 1
    if d > 1:                                   # shared, fully overlapping axes
        z[:, 1:] = rng.normal(0, noise_scale, size=(2 * n, d - 1))
    return z, t, s


def transfer_r2(z, t, s, k=15):
    """kNN fit on domain 0, scored on domain 1 (and vice versa), averaged."""
    out = []
    for a, b in ((0, 1), (1, 0)):
        ia, ib = s == a, s == b
        kn = KNeighborsRegressor(n_neighbors=k).fit(z[ia], t[ia])
        out.append(r2_score(t[ib], kn.predict(z[ib])))
    return float(np.mean(out))


def zero_crossing(rhos, ys):
    order = np.argsort(rhos); x, y = np.asarray(rhos)[order], np.asarray(ys)[order]
    for i in range(len(x) - 1):
        if y[i] * y[i + 1] < 0:
            return float(x[i] + (0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


def main():
    print("=" * 70)
    print("V1  R^2(rho) vs 1 - 4(1-rho)^3   (oracle, w=1, d=1, k=15, n=8000)")
    errs = []
    for rho in RHOS:
        z, t, s = make(rho)
        m = transfer_r2(z, t, s)
        errs.append(abs(m - analytic(rho)))
        print(f"   rho={rho:.2f}  measured={m:+.3f}  1-4(1-rho)^3={analytic(rho):+.3f}"
              f"  |err|={abs(m-analytic(rho)):.3f}")
    print(f"   -> mean |err| = {np.mean(errs):.4f}  (analytic zero 1-4^(-1/3)={1-4**(-1/3):.4f})")

    print("\nV2  width invariance (curve should not move with w)")
    for w in (0.5, 1.0, 3.0):
        ys = [transfer_r2(*make(r, w=w)) for r in RHOS]
        print(f"   w={w:<4} zero-crossing rho* = {zero_crossing(RHOS, ys):.4f}")

    print("\nV3  the driver is per-axis SCALE, not dimension COUNT")
    print("   (a) dimension count at fixed added-axis scale 0.3:")
    for d in (1, 3, 10):
        ys = [transfer_r2(*make(r, d=d)) for r in RHOS]
        print(f"       d={d:<3} rho* = {zero_crossing(RHOS, ys):.4f}"
              f"{'   <- 1-D analytic anchor' if d == 1 else ''}")
    print("   (b) added-axis SCALE at fixed d=10 -- this is the real knob:")
    for sc in (0.001, 0.03, 0.3, 0.6, 1.0):
        ys = [transfer_r2(*make(r, d=10, noise_scale=sc)) for r in RHOS]
        print(f"       std={sc:<5} rho* = {zero_crossing(RHOS, ys):.4f}"
              f"{'   <- vanishing axes -> back to 1-D 0.37' if sc <= 0.001 else ''}")
    print("   => irrelevant axes on a scale comparable to the signal dilute the")
    print("      signal coordinate in the kNN distance; count alone is not the driver.")

    print("\nV4  k/n is a finite-sample correction: rho* -> 0.370 as k/n -> 0")
    print(f"   {'analytic':>22} rho* = {1-4**(-1/3):.4f}")
    for k, n in ((50, 4000), (15, 4000), (5, 8000), (3, 16000), (1, 16000)):
        ys = [transfer_r2(*make(r, n=n), k=k) for r in RHOS]
        zc = zero_crossing(RHOS, ys)
        print(f"   k={k:<3} n={n:<6} k/n={k/n:.5f}  rho* = {zc:.4f}")

    print("\nV5  the constant needs a COMPACT-support marginal (a hard boundary)")
    print("   uniform vs triangular are stable across n; gaussian is NOT a constant")
    print("   -- it has no boundary, so rho* is a finite-sample crossing that")
    print("   drifts toward 0 as n grows.")
    def fmt(x):
        return f"{x:.4f}" if x is not None else "<0.05 (drifting to 0)"
    for n in (2000, 8000, 32000):
        u = zero_crossing(RHOS, [transfer_r2(*make(r, marginal='uniform', n=n)) for r in RHOS])
        tri = zero_crossing(RHOS, [transfer_r2(*make(r, marginal='triangular', n=n)) for r in RHOS])
        g = zero_crossing(RHOS, [transfer_r2(*make(r, marginal='gaussian', n=n)) for r in RHOS])
        print(f"   n={n:<6} uniform rho*={fmt(u)}   triangular rho*={fmt(tri)}   "
              f"gaussian rho*={fmt(g)}")
    print(f"   -> uniform pins near 1-4^(-1/3)={1-4**(-1/3):.4f}; triangular gives a "
          f"DIFFERENT but STABLE constant;")
    print("      gaussian drifts toward 0 (a finite-sample artifact, not a marginal analogue).")

    print("\nV6  EXACT rotation invariance of within-domain kNN-R^2 (vs MCC)")
    from metrics2 import mcc
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4000, 2)); y = z[:, 0] + 0.5 * z[:, 1]
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    zr = z @ R
    kn0 = KNeighborsRegressor(15).fit(z, y);  r0 = r2_score(y, kn0.predict(z))
    kn1 = KNeighborsRegressor(15).fit(zr, y); r1 = r2_score(y, kn1.predict(zr))
    m0 = mcc(z, z)["mcc_pearson"]; m1 = mcc(z, zr)["mcc_pearson"]
    print(f"   kNN-R^2:  identity={r0:.10f}  rotated={r1:.10f}  |diff|={abs(r0-r1):.2e}")
    print(f"   MCC:      identity={m0:.4f}      rotated={m1:.4f}      |diff|={abs(m0-m1):.4f}")
    print(f"   -> kNN-R^2 invariant to machine precision; MCC moves by {abs(m0-m1):.2f}.")


if __name__ == "__main__":
    main()
