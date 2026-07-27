"""
BUILD-OUT: overlap-governed recovery is objective-induced.

Powered campaign (25 seeds) confirming that an ALIGNMENT objective makes recovery
overlap-dependent while a faithful encoder is overlap-invariant, on the standard
v2 DGP. Records raw CCA (primary; sees information-loss folding), MCC (secondary),
the matched-oracle gap (to show it is BLIND to this failure mode), and
batch_removed (the removal axis).

Arms:
  faithful   : conditional VAE, no alignment      -> predict CCA flat in rho
  mix_200    : moment-matching alignment, moderate -> collapse as rho falls
  mix_1000   : moment-matching alignment, strong   -> stronger collapse (dose)
  adv_100    : adversarial (DANN) alignment        -> robustness: a different
               alignment mechanism (may be structurally weaker)

Sharded via env vars STAGE4_SEEDS / ALIGN_OUT (see run_stage4).
"""
from __future__ import annotations
import os, sys, time, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from dgp2 import DGP2Config, generate
from models2 import Model2Config, train_model2, _domain_predictability
from metrics2 import score_embedding
from derisk_dgp3 import matched_oracle_gap

RHOS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.05]
SEEDS = [int(s) for s in os.environ.get("STAGE4_SEEDS",
         ",".join(str(i) for i in range(25))).split(",")]
N, EPOCHS = 1500, 120   # matches derisk_alignment (which established the effect)
ARMS = {
    "faithful": dict(mode="conditional", prior="iso", adv_lambda=0.0),
    "mix_200":  dict(mode="mixing",      prior="iso", adv_lambda=200.0),
    "mix_1000": dict(mode="mixing",      prior="iso", adv_lambda=1000.0),
    "adv_100":  dict(mode="adversarial", prior="iso", adv_lambda=100.0),
}
OUT = os.environ.get("ALIGN_OUT", "results_alignment.csv")


def main():
    rows = []
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = {(r["arm"], r["rho"], r["seed"]) for r in rows}
    combos = list(itertools.product(ARMS, RHOS, SEEDS))
    print(f"{len(combos)} runs, {len(combos)-len(done)} to do", flush=True)
    print(f"{'arm':>9} {'rho':>5} {'sd':>3} {'cca':>6} {'mcc':>6} {'gap':>6} {'rmv':>6}",
          flush=True)
    for arm, rho, seed in combos:
        if (arm, rho, seed) in done:
            continue
        d = generate(DGP2Config(rho=rho, delta=0.0, n_per_domain=N, seed=seed, n_env=5))
        cfg = Model2Config(**ARMS[arm], latent_dim=2, n_env=5, epochs=EPOCHS, seed=seed)
        m, xn, _ = train_model2(d["X"], d["s"], d["e"], cfg)
        z = m.embed(xn); t = np.asarray(d["t"], float)
        sc = score_embedding(t, z, with_dci=False, seed=seed)
        ba = _domain_predictability(z, d["s"])
        rows.append(dict(arm=arm, rho=rho, seed=seed,
                         cca=sc["cca_mean"], mcc=sc["mcc_pearson"],
                         gap=matched_oracle_gap(t, z, seed),
                         batch_removed=float(np.clip(1 - 2*(ba-0.5), 0, 1))))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        r = rows[-1]
        print(f"{arm:>9} {rho:>5} {seed:>3} {r['cca']:>6.3f} {r['mcc']:>6.3f} "
              f"{r['gap']:>6.3f} {r['batch_removed']:>6.3f}", flush=True)
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
