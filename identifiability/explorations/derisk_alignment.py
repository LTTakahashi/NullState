"""
DE-RISK (option B): does an ALIGNMENT objective make recovery overlap-dependent,
on the STANDARD v2 DGP (no gauge)?

Hypothesis: a moment-matching integration penalty (mean+cov of the shared latent
matched across domains, MNN/Harmony-style) forces the domains together; at low
overlap this maps A's exclusive region onto B's, folding/mixing the biological
axis -> recovery degrades. At full overlap, aligning through the shared region is
free. So recovery should fall as rho falls under alignment, and stay flat for a
faithful encoder (lambda=0).

Metric subtlety, handled: alignment likely destroys recovery by INFORMATION LOSS
(folding the axis), not axis-rotation. The matched-oracle GAP subtracts
information loss by construction (it isolates entanglement at matched CCA), so it
can read ~0 even when recovery collapsed. We therefore report RAW learned MCC and
CCA (which see information loss) alongside the gap. The headline effect, if real,
lives in raw MCC/CCA vs rho.

Design: dgp2 (theta=0, standard), mode='mixing' at lambda in {0, 200, 1000},
rho in {1.0, 0.6, 0.3, 0.1}, seeds 0..3.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import sys, time
import numpy as np
from core.dgp import DGP2Config, generate
from core.models import Model2Config, train_model2
from core.metrics import score_embedding, cca_score
from explorations.derisk_gauge import matched_oracle_gap

RHOS = [1.0, 0.6, 0.3, 0.1]
LAMBDAS = [0.0, 200.0, 1000.0]
SEEDS = [0, 1, 2, 3]
N, EP = 1500, 120


def main():
    t0 = time.time()
    res = {}
    print(f"{'lam':>7} {'rho':>5} {'sd':>3} {'mcc':>6} {'cca':>6} {'gap':>6}", flush=True)
    for lam in LAMBDAS:
        for rho in RHOS:
            mccs, ccas, gaps = [], [], []
            for seed in SEEDS:
                d = generate(DGP2Config(rho=rho, delta=0.0, n_per_domain=N,
                                        seed=seed, n_env=5))
                mode = "conditional" if lam == 0 else "mixing"
                m, xn, _ = train_model2(d["X"], d["s"], d["e"],
                    Model2Config(mode=mode, prior="iso", adv_lambda=lam,
                                 latent_dim=2, n_env=5, epochs=EP, seed=seed))
                z = m.embed(xn); t = np.asarray(d["t"], float)
                sc = score_embedding(t, z, with_dci=False, seed=seed)
                mccs.append(sc["mcc_pearson"]); ccas.append(sc["cca_mean"])
                gaps.append(matched_oracle_gap(t, z, seed))
                print(f"{lam:>7} {rho:>5} {seed:>3} {mccs[-1]:>6.3f} "
                      f"{ccas[-1]:>6.3f} {gaps[-1]:>6.3f}", flush=True)
            res[(lam, rho)] = (np.mean(mccs), np.mean(ccas), np.mean(gaps))

    for name, idx in (("raw MCC", 0), ("CCA", 1), ("matched-oracle gap", 2)):
        print(f"\n=== {name} by lambda x rho (mean) ===")
        print(f"{'rho':>6} " + " ".join(f"{'lam=' + str(int(l)):>10}" for l in LAMBDAS))
        for rho in RHOS:
            print(f"{rho:>6} " + " ".join(f"{res[(l,rho)][idx]:>10.3f}" for l in LAMBDAS))

    # the effect: alignment-induced RECOVERY DROP (MCC) as a function of rho
    print("\n=== alignment-induced MCC drop (lam=1000 minus lam=0) vs rho ===")
    print("   (overlap-dependent if the drop GROWS as rho falls)")
    drops = []
    for rho in RHOS:
        drop = res[(0.0, rho)][0] - res[(1000.0, rho)][0]
        drops.append(drop)
        print(f"   rho={rho:<5} drop = {drop:+.3f}")
    grows = drops[-1] > drops[0] + 0.05
    print(f"\ndrop at rho=1.0: {drops[0]:+.3f}  ->  rho=0.1: {drops[-1]:+.3f}")
    print("OVERLAP-DEPENDENT RECOVERY LOSS CONFIRMED" if grows
          else "NOT overlap-dependent (rethink)", f"[{(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    main()
