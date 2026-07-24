"""
THE REMOVAL BOUND, CONFIRMED FROM THE MIXING DIRECTION.

The faithful-encoder frontier (run_frontier.py) shows removal ~ rho: a per-cell
encoder removes the domain only as far as the biology overlaps. The obvious
objection is that a faithful encoder simply is not TRYING to remove the domain.
This script answers it by running an objective that IS: a moment-matching
integration penalty (mean + covariance of the shared latent matched across
domains) at increasing strength lambda, and asking whether it can push removal
PAST the overlap ceiling.

If removal stays ~ rho as lambda grows -- i.e. an objective explicitly maximising
cross-domain mixing still cannot remove more domain than the biology shares --
then removal <= overlap is not an artifact of a passive encoder; it is the bound
biting from the opposite direction. That converts the "failed positive control"
(a mixing proxy that would not mix) into a confirmation: at low overlap there is
not enough shared support to mix THROUGH, no matter how hard the objective pushes.

Recorded per run: MCC (recovery of the true latent), domain balanced-accuracy and
batch_removed (removal), so the recovery cost of pushing lambda is visible too.
"""
from __future__ import annotations
import sys, time, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from dgp2 import DGP2Config, generate
from models2 import Model2Config, train_model2, _domain_predictability
from metrics2 import score_embedding

RHOS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.05]
LAMBDAS = [0.0, 200.0, 1000.0]        # 0 = faithful baseline; 1000 = very hard push
SEEDS = [0, 1, 2]
N = 2000
EPOCHS = 150
OUT = "results_mixing_bound.csv"


def main():
    rows, t0 = [], time.time()
    combos = list(itertools.product(LAMBDAS, RHOS, SEEDS))
    print(f"{len(combos)} runs\n")
    print(f"{'lambda':>7} {'rho':>5} {'sd':>3} | {'MCC':>6} {'domBA':>6} "
          f"{'removed':>7} {'rmv-rho':>7}", flush=True)
    for lam, rho, seed in combos:
        d = generate(DGP2Config(rho=rho, delta=0.0, n_per_domain=N, seed=seed, n_env=5))
        mode = "conditional" if lam == 0 else "mixing"
        cfg = Model2Config(mode=mode, prior="iso", adv_lambda=lam,
                           latent_dim=2, n_env=5, epochs=EPOCHS, seed=seed)
        model, xn, _ = train_model2(d["X"], d["s"], d["e"], cfg)
        z = model.embed(xn)
        mcc = score_embedding(np.asarray(d["t"], float), z, with_dci=False,
                              seed=seed)["mcc_pearson"]
        ba = _domain_predictability(z, d["s"])
        removed = float(np.clip(1.0 - 2.0 * (ba - 0.5), 0.0, 1.0))
        rows.append(dict(lam=lam, rho=rho, seed=seed, mcc=mcc,
                         domain_balacc=ba, batch_removed=removed,
                         removed_minus_rho=removed - rho))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        r = rows[-1]
        print(f"{lam:>7} {rho:>5} {seed:>3} | {mcc:>6.3f} {ba:>6.3f} "
              f"{removed:>7.3f} {removed-rho:>+7.3f}", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== batch_removed by lambda x rho (mean) -- does mixing beat rho? ===")
    print(df.pivot_table(index="rho", columns="lam", values="batch_removed",
                         aggfunc="mean").round(3).sort_index(ascending=False).to_string())
    print("\n=== MCC by lambda x rho (mean) -- recovery cost of pushing lambda ===")
    print(df.pivot_table(index="rho", columns="lam", values="mcc",
                         aggfunc="mean").round(3).sort_index(ascending=False).to_string())
    # headline number: max removal achievable ABOVE rho, over all lambda, per rho
    print("\n=== can maximal mixing push removal above overlap? ===")
    for rho in RHOS:
        best = df[df.rho == rho].batch_removed.max()
        print(f"  rho={rho:<5} best removed over all lambda = {best:.3f}  "
              f"(excess over rho = {best-rho:+.3f})")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
