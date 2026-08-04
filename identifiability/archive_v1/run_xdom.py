"""Corrected headline sweep: cross-domain transfer (identifiability of the shared axis)
vs support overlap rho, on a fine grid to locate the knee. Also records the model-free
a-priori certificate at each rho so we can ask whether it predicts the collapse."""
import sys, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from dgp import DGPConfig, generate
from models import ModelConfig, train_model
from metrics import evaluate_embedding, raw_certificate
from sweep import model_input

RHOS = [1.0, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.15, 0.05, 0.0]
SEEDS = [0, 1, 2]
rows = []
t0 = time.time()
for rho in RHOS:
    for seed in SEEDS:
        d = generate(DGPConfig(rho=rho, delta=1.0, n_per_domain=3000, seed=seed))
        X, t, s, mask = d["X"], d["t"], d["s"], d["overlap_region"]
        cert = raw_certificate(X, s)          # model-free, a-priori
        m, _ = train_model(X, s, ModelConfig(mode="conditional", epochs=150, seed=seed))
        z = m.embed(model_input(X, m))
        ev = evaluate_embedding(z, t, s, mask=mask)
        rows.append(dict(rho=rho, seed=seed, overlap=d["overlap"],
                         cert_max=cert["max"], cert_spread=cert["spread"], **ev))
        pd.DataFrame(rows).to_csv("results_xdom.csv", index=False)
        print(f"[{len(rows)}/{len(RHOS)*len(SEEDS)}] rho={rho:.2f} seed={seed} "
              f"xdom={ev['xdom_mean']:+.3f} bio={ev['bio_recovery']:.3f} "
              f"cert={cert['max']:.3f} ({(time.time()-t0)/60:.1f}m)", flush=True)
print("DONE", (time.time()-t0)/60, "min")
