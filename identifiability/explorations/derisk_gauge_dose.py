"""
DE-RISK part 2: dose-response in the gauge angle.

If the gauge is the cause of overlap-dependent recovery, the gap change from full
to low overlap should INCREASE with the gauge angle theta. Test theta in
{0, pi/6, pi/4, pi/3} at rho in {1.0, 0.1}, several seeds. A monotone dose-response
is strong evidence the mechanism -- not noise -- drives the effect.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import sys, time
import numpy as np
from core.dgp_gauge import DGP3Config, generate
from core.models import Model2Config, train_model2
from core.metrics import score_embedding, cca_score
from explorations.derisk_gauge import matched_oracle_gap

THETAS = [0.0, np.pi / 6, np.pi / 4, np.pi / 3]
RHOS = [1.0, 0.1]
SEEDS = [0, 1, 2, 3]
N, EP = 1500, 120


def main():
    t0 = time.time()
    res = {}
    print(f"{'theta':>6} {'rho':>5} {'sd':>3} {'gap':>6}", flush=True)
    for th in THETAS:
        for rho in RHOS:
            gs = []
            for seed in SEEDS:
                d = generate(DGP3Config(rho=rho, gauge_angle=th, delta=0.0,
                                        n_per_domain=N, seed=seed, n_env=5))
                m, xn, _ = train_model2(d["X"], d["s"], d["e"],
                    Model2Config(mode="conditional", prior="iso", latent_dim=2,
                                 n_env=5, epochs=EP, seed=seed))
                g = matched_oracle_gap(np.asarray(d["t"], float), m.embed(xn), seed)
                gs.append(g)
                print(f"{th:>6.3f} {rho:>5} {seed:>3} {g:>6.3f}", flush=True)
            res[(th, rho)] = (np.mean(gs), np.std(gs) / np.sqrt(len(gs)))
    print("\n=== DOSE-RESPONSE: gap change (rho 1.0 -> 0.1) vs gauge angle ===")
    print(f"{'theta':>8} {'gap@rho=1':>12} {'gap@rho=0.1':>12} {'change':>10}")
    changes = []
    for th in THETAS:
        hi, lo = res[(th, 1.0)][0], res[(th, 0.1)][0]
        changes.append(lo - hi)
        print(f"{th:>8.3f} {hi:>12.3f} {lo:>12.3f} {lo-hi:>+10.3f}")
    mono = all(changes[i] <= changes[i+1] + 0.02 for i in range(len(changes)-1))
    print(f"\nmonotone in theta: {mono}   change at theta=0: {changes[0]:+.3f} -> "
          f"theta=pi/3: {changes[-1]:+.3f}")
    print("DOSE-RESPONSE CONFIRMED (gauge drives it)" if mono and changes[-1] > 0.10
          else "weak/non-monotone", f"[{(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    main()
