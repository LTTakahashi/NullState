"""
THE RECOVERY-PRESERVING FRONTIER, from the mixing direction.

The faithful-encoder frontier (run_frontier.py) sits at removal ~ rho -- but that
is a metric-definition identity (batch_removed = clip(1-2(balacc-0.5)) and the
Bayes-optimal domain classifier on two overlap-rho uniforms has balacc = 1-rho/2,
so batch_removed == rho for ANY latent-preserving encoder). The real question is
whether an objective that ACTIVELY tries to mix can push removal PAST rho, and at
what cost. This script runs a moment-matching integration penalty (mean +
covariance of the shared latent matched across domains) at increasing strength
lambda and records both removal and recovery.

Finding: removal CAN exceed rho at high lambda (0.40 at rho=0.2, 0.27 at rho=0.05)
-- so "removal <= overlap" is NOT a hard bound -- but only by collapsing recovery
point-for-point (MCC 0.86 -> 0.67 at rho=0.05). At rho=0.05 no setting achieves
both MCC > 0.8 and removed > 0.1. The honest statement is therefore a recovery-
preserving frontier: the recovery-preserving optimum sits at removal ~ rho, and
buying removal beyond it costs recovery.

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
