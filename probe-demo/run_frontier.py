"""
POSITIVE CONTROL + FEASIBILITY FRONTIER.

The Stage-4 null (matched-oracle MCC gap flat in rho for faithful per-cell
encoders) is only interpretable if the metric COULD have seen a support-overlap
effect. This script supplies that control by contrasting two objectives across
the rho sweep at delta = 0, and recording BOTH axes of the feasibility frontier:

  RECOVERY  : matched-information MCC gap (entanglement of the shared latent)
  REMOVAL   : batch_removed = 1 - 2*(domain balanced-accuracy - 0.5) of the
              shared latent  (1 = domain fully removed, 0 = fully present)

Arms:
  faithful (conditional, lambda=0)  -- encodes each cell independently. Expected:
      recovery high and flat in rho; removal achieved only when biology overlaps.
  mixing  (moment-matching integration, lambda>0) -- forces the two domains'
      shared-latent distributions to coincide, as MNN/Harmony do. Expected:
      at full overlap, matching is free (no recovery cost); as rho falls,
      matching REQUIRES misaligning biology, so the MCC gap GROWS. If it does,
      the metric demonstrably detects support-overlap non-identifiability when
      an integration objective induces it -- which is the whole point.

The frontier claim, made precise: below some rho you cannot get BOTH high
recovery AND high removal. Faithful encoders buy recovery by NOT removing;
mixing objectives buy removal by destroying recovery; neither reaches the
top-right corner once overlap is low.
"""
from __future__ import annotations
import sys, time, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from dgp2 import DGP2Config, generate
from models2 import Model2Config, train_model2, _domain_predictability
from metrics2 import score_embedding, cca_score
from run_stage4 import matched_oracle

RHOS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.05]
SEEDS = [0, 1, 2, 3, 4]
N = 2000
EPOCHS = 150
# Faithful per-cell encoder only. The mixing arm was DROPPED after calibration:
# a moment-matching penalty on the 2-D latent does NOT force domain removal at
# low overlap -- the model satisfies the mean/covariance match by ROTATING the
# latent while preserving biology (measured: at rho=0.2, domain balanced-accuracy
# stayed 0.90 at lambda in {0, 50, 200}, and MCC even recovered to 0.96 at
# lambda=200). Forcing genuine pointwise cross-domain mixing needs a real
# integration objective (MNN/Harmony/a strong characteristic-kernel MMD), which
# is out of scope. So this script establishes the two frontier axes for the
# faithful encoder and does not claim a mixing positive control it cannot
# cleanly produce. (The rotation behaviour is itself an honest observation:
# moment-matching integration does not over-correct in this geometry.)
ARMS = {"faithful": dict(mode="conditional", prior="iso", adv_lambda=0.0)}
OUT = "results_frontier_v2.csv"


def main():
    rows, t0 = [], time.time()
    combos = list(itertools.product(ARMS, RHOS, SEEDS))
    print(f"{len(combos)} runs\n")
    print(f"{'arm':>11} {'rho':>5} {'sd':>3} | {'L_mcc':>6} {'gap':>6} "
          f"{'removed':>7} {'domBA':>6} {'L_cca':>6}", flush=True)
    for arm, rho, seed in combos:
        d = generate(DGP2Config(rho=rho, delta=0.0, n_per_domain=N, seed=seed, n_env=5))
        cfg = Model2Config(**ARMS[arm], latent_dim=2, n_env=5, epochs=EPOCHS, seed=seed)
        model, xn, _ = train_model2(d["X"], d["s"], d["e"], cfg)
        z = model.embed(xn)
        t = np.asarray(d["t"], float)

        learned = score_embedding(t, z, with_dci=False, seed=seed)
        _, _, orc = matched_oracle(t, z, seed=seed)
        dom_ba = _domain_predictability(z, d["s"])
        removed = float(np.clip(1.0 - 2.0 * (dom_ba - 0.5), 0.0, 1.0))

        row = dict(arm=arm, rho=rho, seed=seed,
                   learned_mcc=learned["mcc_pearson"], oracle_mcc=orc["mcc_pearson"],
                   gap_mcc=orc["mcc_pearson"] - learned["mcc_pearson"],
                   learned_cca=learned["cca_mean"],
                   domain_balacc=dom_ba, batch_removed=removed,
                   joint=min(learned["mcc_pearson"], removed))
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{arm:>11} {rho:>5} {seed:>3} | {row['learned_mcc']:>6.3f} "
              f"{row['gap_mcc']:>6.3f} {removed:>7.3f} {dom_ba:>6.3f} "
              f"{row['learned_cca']:>6.3f}", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== MCC gap by arm x rho (mean) ===", flush=True)
    print(df.pivot_table(index="rho", columns="arm", values="gap_mcc",
                         aggfunc="mean").round(3).sort_index(ascending=False).to_string())
    print("\n=== batch_removed by arm x rho (mean) ===", flush=True)
    print(df.pivot_table(index="rho", columns="arm", values="batch_removed",
                         aggfunc="mean").round(3).sort_index(ascending=False).to_string())
    print("\n=== joint feasibility min(recovery, removal) by arm x rho ===", flush=True)
    print(df.pivot_table(index="rho", columns="arm", values="joint",
                         aggfunc="mean").round(3).sort_index(ascending=False).to_string())

    # the RECOVERY-PRESERVING FRONTIER (not a bound). Two honest caveats:
    #  * removal ~ rho is ALGEBRA, not a measured law: batch_removed is defined
    #    as clip(1 - 2*(domain_balacc - 0.5)), and for the Bayes-optimal domain
    #    classifier on two overlap-rho uniform marginals balanced accuracy is
    #    exactly 1 - rho/2, so batch_removed == rho identically for ANY latent-
    #    preserving encoder. (Verified in the module receipt.)
    #  * removal is NOT hard-bounded by rho: run_mixing_bound.py pushes removal
    #    to 0.40 at rho=0.2 and 0.27 at rho=0.05 -- ABOVE overlap -- but only by
    #    collapsing recovery point-for-point (MCC 0.86 -> 0.67). So the frontier
    #    is: the RECOVERY-PRESERVING optimum sits at removal ~ rho; exceeding it
    #    costs recovery. It is a tradeoff curve, not "removal <= overlap".
    print("\n=== RECOVERY-PRESERVING FRONTIER (faithful encoder) ===")
    g = df.groupby("rho").agg(recovery=("learned_mcc", "mean"),
                              gap=("gap_mcc", "mean"),
                              removal=("batch_removed", "mean"),
                              joint=("joint", "mean")).sort_index(ascending=False)
    print(g.round(3).to_string())
    print("\n  recovery (MCC) is flat in rho; removal ~ rho is a metric-definition")
    print("  identity (balacc=1-rho/2). The non-trivial content is that recovery")
    print("  does NOT additionally degrade: the recovery-preserving optimum sits at")
    print("  removal~rho, and (run_mixing_bound.py) an objective can push removal")
    print("  ABOVE rho only by collapsing recovery point-for-point. Frontier, not bound.")
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
