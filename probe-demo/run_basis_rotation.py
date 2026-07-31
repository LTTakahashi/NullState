"""
THE BASIS-DEPENDENCE TEST -- the most obvious attack on Demonstration 2.

Claim under attack: "the matched-oracle gap missed the alignment collapse" may be
an artifact, because in dgp2 the shifted direction IS latent coordinate 0, so a
folding of it is axis-factorised BY CONSTRUCTION -- and axis-factorised maps are
exactly the gap's zero set. The taxonomy predicts collapse-of-an-axis -> 0.000 vs
collapse-of-a-ROTATED-axis -> 0.142.

Test: put the shifted direction at theta = 45 degrees to the ground-truth basis
(dgp2_rotbasis) and rerun one alignment arm. Does the gap now fire near ~0.14?

CONTROLS. At theta > 0 the declared basis is no longer the independent-component
basis, so a model that prefers independent axes scores MCC ~ cos(theta) even with
perfect recovery -- basis mismatch, not folding. The faithful arm is therefore run
at the SAME theta, and the estimand is the DIFFERENCE:

    excess_gap(theta) = gap[alignment](theta) - gap[faithful](theta)

We also score every embedding against BOTH candidate ground truths -- the declared
rotated basis `s` and the independent basis `ab` -- which tells us directly which
one the model actually recovered, and hence whether a gap is basis mismatch or
genuine axis mixing.

Both outcomes are publishable:
  fires  -> whether an estimand is blind to a failure mode depends on an alignment
            between the failure's geometry and the ground-truth basis, which in
            real data nobody controls or observes. Strongest form of the thesis.
  quiet  -> the most obvious attack on Demonstration 2 is closed; say so.
"""
from __future__ import annotations
import os, sys, time, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from dgp2_rotbasis import RotBasisConfig, generate
from models2 import Model2Config, train_model2
from metrics2 import score_embedding, cca_score, mcc

THETAS = [0.0, np.pi / 4]
RHOS = [1.0, 0.6, 0.2, 0.05]
ARMS = {
    "faithful": dict(mode="conditional", prior="iso", adv_lambda=0.0),
    "mix_1000": dict(mode="mixing",      prior="iso", adv_lambda=1000.0),
}
SEEDS = [int(x) for x in os.environ.get("BR_SEEDS", "0,1,2,3,4").split(",")]
N, EPOCHS = 1500, 120
OUT = os.environ.get("BR_OUT", "results_basis_rotation.csv")


def scores(t_true, z, seed):
    """CCA, MCC and the entanglement gap (= CCA - MCC) against a given basis."""
    c = cca_score(t_true, z, seed=seed)["cca_mean"]
    m = mcc(t_true, z, "pearson", seed=seed)["mcc_pearson"]
    return c, m, c - m


def main():
    rows, done = [], set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT); rows = prev.to_dict("records")
        done = {(r["arm"], round(r["theta"], 4), r["rho"], r["seed"]) for r in rows}
    combos = list(itertools.product(ARMS, THETAS, RHOS, SEEDS))
    print(f"{len(combos)} runs, {len(combos)-len(done)} to do", flush=True)
    print(f"{'arm':>9} {'theta':>6} {'rho':>5} {'sd':>3} | "
          f"{'CCA_s':>6} {'MCC_s':>6} {'gap_s':>6} | {'MCC_ab':>7} {'gap_ab':>7}", flush=True)
    for arm, th, rho, seed in combos:
        if (arm, round(th, 4), rho, seed) in done:
            continue
        d = generate(RotBasisConfig(rho=rho, delta=0.0, n_per_domain=N,
                                    seed=seed, n_env=5, basis_angle=th))
        cfg = Model2Config(**ARMS[arm], latent_dim=2, n_env=5, epochs=EPOCHS, seed=seed)
        m_, xn, _ = train_model2(d["X"], d["s"], d["e"], cfg)
        z = m_.embed(xn)
        # score against the DECLARED (rotated) basis and the INDEPENDENT basis
        c_s, m_s, g_s = scores(np.asarray(d["t"], float), z, seed)
        c_ab, m_ab, g_ab = scores(np.asarray(d["ab"], float), z, seed)
        rows.append(dict(arm=arm, theta=th, rho=rho, seed=seed,
                         cca_s=c_s, mcc_s=m_s, gap_s=g_s,
                         cca_ab=c_ab, mcc_ab=m_ab, gap_ab=g_ab))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{arm:>9} {th:>6.3f} {rho:>5} {seed:>3} | {c_s:>6.3f} {m_s:>6.3f} "
              f"{g_s:>6.3f} | {m_ab:>7.3f} {g_ab:>7.3f}", flush=True)
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
