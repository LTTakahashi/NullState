"""DECISIVE TEST (Gate G2 on a VALID metric).

Question: does the LEARNED shared latent lose the cross-domain axis as support overlap
falls, over and above what the DGP's range shift explains?

Design: for each rho, compute cross-domain transfer RESTRICTED TO THE OVERLAP BAND for
  (a) the LEARNED embedding, and
  (b) an ORACLE embedding (z := true t), which is identified by construction.
The oracle is the control: it is flat at ~1.0 for every rho (verified), so it absorbs the
extrapolation artifact that invalidated the unrestricted metric. Any gap between the two
curves is attributable to non-identifiability of the learned axis.

Reports n_overlap at every point, because the overlap band shrinks as rho -> 0 and the
metric loses power exactly where the effect should be largest. Small-n points are
UNDERPOWERED, not null.
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd

from dgp import DGPConfig, generate
from models import ModelConfig, train_model
from metrics import cross_domain_transfer_overlap, bio_recovery, batch_removed, raw_certificate
from sweep import model_input

RHOS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
SEEDS = [0, 1, 2]
N = 3000

rows, t0 = [], time.time()
print(f"{'rho':>5} {'seed':>4} {'n_ovl':>6} {'LEARNED':>8} {'ORACLE':>8} {'gap':>7} "
      f"{'bio_ovl':>8} {'cert':>6}", flush=True)
for rho in RHOS:
    for seed in SEEDS:
        d = generate(DGPConfig(rho=rho, delta=1.0, n_per_domain=N, seed=seed))
        X, t, s, mask = d["X"], np.asarray(d["t"]), np.asarray(d["s"]).ravel(), np.asarray(d["overlap_region"])
        cert = raw_certificate(X, s)
        m, _ = train_model(X, s, ModelConfig(mode="conditional", epochs=150, seed=seed))
        z = np.asarray(m.embed(model_input(X, m)))

        learned = cross_domain_transfer_overlap(z, t, s, mask)
        oracle = cross_domain_transfer_overlap(t.copy(), t, s, mask)   # control
        gap = oracle["xdom_ovl_mean"] - learned["xdom_ovl_mean"]

        rows.append(dict(rho=rho, seed=seed, n_overlap=learned["n_overlap"],
                         learned=learned["xdom_ovl_mean"], oracle=oracle["xdom_ovl_mean"],
                         gap=gap, bio_overlap=bio_recovery(z, t, mask=mask),
                         batch_removed=batch_removed(z, s),
                         cert_max=cert["max"], cert_spread=cert["spread"]))
        pd.DataFrame(rows).to_csv("results_decisive.csv", index=False)
        r = rows[-1]
        print(f"{rho:>5} {seed:>4} {r['n_overlap']:>6} {r['learned']:>8.3f} {r['oracle']:>8.3f} "
              f"{r['gap']:>7.3f} {r['bio_overlap']:>8.3f} {r['cert_max']:>6.3f}", flush=True)

df = pd.DataFrame(rows)
print("\n=== GATE G2: learned vs oracle (mean over seeds), overlap-restricted ===", flush=True)
g = df.groupby("rho").agg(n_ovl=("n_overlap", "mean"), learned=("learned", "mean"),
                          learned_sd=("learned", "std"), oracle=("oracle", "mean"),
                          gap=("gap", "mean"), cert=("cert_max", "mean")).sort_index(ascending=False)
print(g.round(3).to_string(), flush=True)
print(f"\nDONE in {(time.time()-t0)/60:.1f} min", flush=True)
