"""
RECEIPT FOR THE HEADLINE: v1's "critical overlap" was a geometry constant.

v1 reported a critical support overlap rho* ~ 0.37 below which its kNN cross-
domain transfer recovery score collapsed. This script shows that number is a
constant of the metric's extrapolation geometry, not a property of
identifiability, by running the SAME metric on an ORACLE embedding: z := the true
latent t, which is perfectly identified by construction. If the collapse and its
threshold appear even for the oracle, they cannot be about (non-)identifiability.

Result: the oracle transfer follows the closed form 1 - 4(1-rho)^3 to within
~0.02 across the whole range, and crosses zero at rho* = 1 - 4^(-1/3) = 0.370.

Generalisation (the paper's practical message): kNN-R^2 and kNN-transfer recovery
scores are EXACTLY rotation-invariant (up to almost-surely-unique neighbour sets),
so they cannot detect entanglement, and any critical-threshold claim built on one
is exposed to reading a geometry constant as a phenomenon. The constant depends on
the marginal shape, embedding scale, and k (see verify_closed_form.py). Check the
oracle first.
"""
from __future__ import annotations
import os
import sys
import numpy as np

# The v1 code is retired and lives in archive_v1/. We import it deliberately:
# the point of this script is that the RETRACTED metric, run on a perfectly
# identified oracle embedding, still produces the retracted threshold.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive_v1"))
from dgp import DGPConfig, generate            # noqa: E402  v1 (retired) DGP
from metrics import cross_domain_transfer      # noqa: E402  v1 (retired) metric

RHOS = [1.0, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.15, 0.05]


def analytic(rho):
    return 1.0 - 4.0 * (1.0 - rho) ** 3


def oracle_transfer(rho, seed=0, n=4000):
    """v1 kNN transfer with the embedding set to the TRUE latent."""
    d = generate(DGPConfig(rho=rho, delta=1.0, n_per_domain=n, seed=seed))
    t = np.asarray(d["t"])
    s = np.asarray(d["s"]).ravel()
    return cross_domain_transfer(t.copy(), t, s)["xdom_mean"]


def main():
    print("v1 kNN transfer on the ORACLE embedding (z = true latent):")
    print(f"{'rho':>6} {'oracle':>9} {'1-4(1-rho)^3':>14} {'|err|':>7}")
    xs, ys, errs = [], [], []
    for rho in RHOS:
        m = oracle_transfer(rho)
        a = analytic(rho)
        xs.append(rho); ys.append(m); errs.append(abs(m - a))
        print(f"{rho:>6} {m:>9.3f} {a:>14.3f} {abs(m - a):>7.3f}")

    xs, ys = np.array(xs), np.array(ys)
    order = np.argsort(xs); xs, ys = xs[order], ys[order]
    zc = None
    for i in range(len(xs) - 1):
        if ys[i] * ys[i + 1] < 0:
            zc = xs[i] + (0 - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
            break

    print(f"\nmean |err| vs 1-4(1-rho)^3          = {np.mean(errs):.4f}")
    print(f"analytic zero  rho* = 1 - 4^(-1/3)  = {1 - 4 ** (-1 / 3):.4f}")
    print(f"empirical zero of the ORACLE curve  = {zc:.4f}"
          if zc else "no zero crossing found")
    ok = np.mean(errs) < 0.05 and zc is not None and abs(zc - (1 - 4 ** (-1 / 3))) < 0.05
    print("\n" + ("RECEIPT CONFIRMED: the threshold is metric geometry, not "
                  "identifiability." if ok else "RECEIPT NOT CONFIRMED (investigate)."))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
