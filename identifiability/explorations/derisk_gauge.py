"""
DE-RISK: does the domain gauge make recovery genuinely overlap-dependent?

Predict: matched-oracle MCC gap is FLAT in rho at theta=0 (the v2 null) and GROWS
as rho falls at theta=pi/4 (the phenomenon). If so, the hard-regime design works.
Reuses the v2 model (shared encoder, domain-conditional decoder) and metric.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import sys, time
import numpy as np
from core.dgp_gauge import DGP3Config, generate
from core.models import Model2Config, train_model2
from core.metrics import score_embedding, cca_score

RHOS = [1.0, 0.6, 0.3, 0.1]
THETAS = [0.0, np.pi / 4]
SEEDS = [0, 1]
N, EP = 1500, 120


def matched_oracle_gap(t, z, seed):
    target = cca_score(t, z, seed=seed)["cca_mean"]
    lo, hi = 0.0, 8.0
    sd = t.std(0, keepdims=True)
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        rng = np.random.default_rng(seed + 991)
        zz = t + mid * sd * rng.normal(size=t.shape)
        if cca_score(t, zz, seed=seed)["cca_mean"] > target:
            lo = mid
        else:
            hi = mid
    rng = np.random.default_rng(seed + 991)
    zz = t + 0.5 * (lo + hi) * sd * rng.normal(size=t.shape)
    return (score_embedding(t, zz, with_dci=False, seed=seed)["mcc_pearson"]
            - score_embedding(t, z, with_dci=False, seed=seed)["mcc_pearson"])


def main():
    t0 = time.time()
    print(f"{'theta':>6} {'rho':>5} {'sd':>3} {'L_mcc':>6} {'gap':>6}", flush=True)
    res = {}
    for th in THETAS:
        for rho in RHOS:
            gaps, mccs = [], []
            for seed in SEEDS:
                d = generate(DGP3Config(rho=rho, gauge_angle=th, delta=0.0,
                                        n_per_domain=N, seed=seed, n_env=5))
                m, xn, _ = train_model2(d["X"], d["s"], d["e"],
                    Model2Config(mode="conditional", prior="iso", latent_dim=2,
                                 n_env=5, epochs=EP, seed=seed))
                z = m.embed(xn); t = np.asarray(d["t"], float)
                g = matched_oracle_gap(t, z, seed)
                mcc = score_embedding(t, z, with_dci=False, seed=seed)["mcc_pearson"]
                gaps.append(g); mccs.append(mcc)
                print(f"{th:>6.3f} {rho:>5} {seed:>3} {mcc:>6.3f} {g:>6.3f}", flush=True)
            res[(th, rho)] = (np.mean(mccs), np.mean(gaps))
    print("\n=== matched-oracle MCC gap vs rho ===")
    print(f"{'rho':>6} {'theta=0 (v2)':>14} {'theta=pi/4 (gauge)':>20}")
    for rho in RHOS:
        print(f"{rho:>6} {res[(0.0,rho)][1]:>14.3f} {res[(np.pi/4,rho)][1]:>20.3f}")
    g0 = res[(0.0, RHOS[-1])][1] - res[(0.0, RHOS[0])][1]
    gg = res[(np.pi/4, RHOS[-1])][1] - res[(np.pi/4, RHOS[0])][1]
    print(f"\ngap change (rho {RHOS[0]} -> {RHOS[-1]}):  theta=0: {g0:+.3f}   "
          f"theta=pi/4: {gg:+.3f}")
    print("PHENOMENON PRESENT" if gg > 0.10 and gg > g0 + 0.05 else "NOT (rethink)",
          f"[{(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    main()
