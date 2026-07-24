"""
STAGE-2 GATE: the metric must not bend with rho on its own.

The v1 probe's fatal metric property was that its scores were closed-form
functions of rho REGARDLESS of the embedding: `cross_domain_transfer` equalled
1 - 4(1-rho)^3 with a "critical" zero crossing at 1 - 4^(-1/3) = 0.370, a
constant of kNN extrapolation geometry. Any model scored against that is being
scored against the DGP's geometry, not its own identifiability.

So before any model is trained we check the metric itself, on embeddings whose
quality is fixed BY CONSTRUCTION and independent of rho:

  noiseless oracle   z := t            -- must be 1.0 everywhere (trivially)
  matched-noise      z := t + sigma*eps -- must be FLAT in rho at each sigma
  rotated oracle     z := R(theta) t    -- must be flat in rho, and must be
                                          DETECTED (MCC < 1) while CCA stays 1

The matched-noise family is the one that matters, and is what audit finding B4
demanded: a noiseless oracle returns ~1.0 at every rho by construction and is
therefore blind to exactly the distortions we need to rule out.

PASS: for every sigma, the spread of each metric across the rho sweep is below
tolerance. Any metric that bends with rho here is unusable downstream, because
a model's score could not be separated from that bend.
"""
from __future__ import annotations
import sys
import numpy as np

from dgp2 import DGP2Config, generate
from metrics2 import score_embedding

RHOS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.05]
DELTAS = [0.0, 1.0, 2.0]
SIGMAS = [0.0, 0.25, 0.5, 1.0]
SEEDS = [0, 1, 2]
N = 3000
TOL = 0.05

FAILURES = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def main():
    print("=" * 74)
    print("STAGE-2 GATE  |  metric ceiling must be flat in rho")
    print("=" * 74)

    # cache latents once per (rho, delta, seed): the metric only needs t
    lat = {}
    for rho in RHOS:
        for delta in DELTAS:
            for seed in SEEDS:
                d = generate(DGP2Config(rho=rho, delta=delta,
                                        n_per_domain=N, seed=seed))
                lat[(rho, delta, seed)] = (d["t"], d["overlap_region"])

    for delta in DELTAS:
        print(f"\n--- delta = {delta} ---")
        print(f"  {'sigma':>6} " + " ".join(f"{'r=' + str(r):>8}" for r in RHOS)
              + f" {'spread':>8}")
        for metric in ("mcc_pearson", "cca_mean"):
            print(f"  metric: {metric}")
            for sigma in SIGMAS:
                row = []
                for rho in RHOS:
                    vals = []
                    for seed in SEEDS:
                        t, mask = lat[(rho, delta, seed)]
                        rng = np.random.default_rng(1000 + seed)
                        z = t + sigma * t.std(0, keepdims=True) * rng.normal(size=t.shape)
                        vals.append(score_embedding(t, z, with_dci=False,
                                                    seed=seed)[metric])
                    row.append(float(np.mean(vals)))
                spread = max(row) - min(row)
                print(f"  {sigma:>6} " + " ".join(f"{v:>8.4f}" for v in row)
                      + f" {spread:>8.4f}")
                check(f"{metric} flat across rho (delta={delta}, sigma={sigma})",
                      spread < TOL, f"spread {spread:.4f} (tol {TOL})")

    # rotation must be DETECTED, and detected equally at every rho
    print("\n--- rotated oracle: MCC must drop, CCA must not, both flat in rho ---")
    th = np.pi / 4
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    mccs, ccas = [], []
    for rho in RHOS:
        m, c = [], []
        for seed in SEEDS:
            t, _ = lat[(rho, 1.0, seed)]
            s = score_embedding(t, t @ R, with_dci=False, seed=seed)
            m.append(s["mcc_pearson"]); c.append(s["cca_mean"])
        mccs.append(np.mean(m)); ccas.append(np.mean(c))
    print(f"  {'rho':>6} " + " ".join(f"{r:>8}" for r in RHOS))
    print(f"  {'MCC':>6} " + " ".join(f"{v:>8.4f}" for v in mccs))
    print(f"  {'CCA':>6} " + " ".join(f"{v:>8.4f}" for v in ccas))
    check("rotation is detected by MCC", max(mccs) < 0.95,
          f"max MCC on a 45deg rotation = {max(mccs):.4f}")
    check("rotation is NOT penalised by CCA", min(ccas) > 0.95,
          f"min CCA = {min(ccas):.4f}")
    check("rotation detection flat in rho", max(mccs) - min(mccs) < TOL,
          f"spread {max(mccs) - min(mccs):.4f}")

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"GATE FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print("   -", f)
        return 1
    print("GATE PASSED: the metric ceiling is flat in rho. Stage 3/4 may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
