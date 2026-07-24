"""
STAGE-4: the main sweep. THE claim, and the only one the probe exists to test.

PRIMARY ESTIMAND -- the MATCHED-INFORMATION MCC GAP.

  Never the raw learned MCC. A raw score confounds two different failures:
  the model lost information (its embedding is noisier), and the model kept the
  information but on the wrong axes (entanglement). Only the second is
  non-identifiability. So we compare the learned embedding against an oracle
  built from the TRUE latent corrupted by isotropic noise, with the noise level
  chosen so the oracle carries the SAME amount of information as the learned
  embedding (matched CCA). Isotropic noise cannot mix axes, so at matched
  information the oracle's MCC is the best achievable, and

      gap = MCC(matched oracle) - MCC(learned)

  is attributable to axis mixing alone.

  This is what audit finding B4 demanded. run_decisive.py used a NOISELESS
  oracle (z := t), which returns ~1.0 at every rho by construction and is blind
  to distortion at any realistic operating point.

HYPOTHESES, pre-registered:
  H1  gap grows as rho falls   (support mismatch destroys identifiability)
  H2  gap is flat in delta     (a removable batch effect does not)
  H0  gap is flat in rho       -- the honest null, and publishable as such

DECISION RULE (from the design brief, fixed before looking at results):
  supported : gap monotone in rho AND Cohen's d >= 0.8 between rho=1 and rho=0.2
              AND flat in delta (d < 0.2 across delta at rho=1)
  refuted   : gap flat in rho, or gap moves with delta
  "threshold" language is permitted ONLY if segmented regression beats linear
  (Davies p < 0.05, slope ratio > 3). Default expectation is SMOOTH degradation:
  no theory predicts a discontinuity in identifiability as a function of overlap.
  Ben-David's bounds are linear in divergence, and weak-overlap causal inference
  gives continuously degrading rates, not a cliff.

ARMS:
  ivae_5env  : conditional prior over 5 environments = 2n+1, so the iVAE
               variability condition (Khemakhem 2020 Thm 1 (iv)) is SATISFIABLE
  ivae_2env  : the same model with 2 environments -- provably CANNOT satisfy it.
               This is the deliberately-violating arm; v1's entire design sat
               here without knowing it.
  conditional: scVI-style, isotropic prior, domain as a decoder covariate.
               Provably non-identifiable (Locatello 2019) -- the control.

Everything is confined to the CERTIFIED region delta <= 2.0 established by
verify_dgp2.py. Checkpoints after every run.
"""
from __future__ import annotations
import sys, time, itertools
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from dgp2 import DGP2Config, generate
from models2 import Model2Config, train_model2, vae_diagnostics
from metrics2 import score_embedding, mcc, cca_score

RHOS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.05]
DELTAS = [0.0, 2.0]                  # both inside the certified region
SEEDS = [0, 1, 2, 3, 4]
N = 2000
EPOCHS = 150
ARMS = {
    "ivae_5env":   dict(mode="ivae", prior="cond", n_env=5),
    "ivae_2env":   dict(mode="ivae", prior="cond", n_env=2),
    "conditional": dict(mode="conditional", prior="iso", n_env=5),
}
OUT = "results_stage4.csv"


def matched_oracle(t, z_learned, seed=0):
    """Isotropic-noise oracle matched to the learned embedding's information.

    Bisect the noise level until the oracle's CCA equals the learned CCA. CCA
    measures recovery up to an invertible LINEAR map, so it is insensitive to
    axis mixing -- which is exactly why it is the right thing to match on: it
    equalises information while leaving MCC free to expose entanglement.
    """
    target = cca_score(t, z_learned, seed=seed)["cca_mean"]
    lo, hi = 0.0, 8.0
    sd = t.std(0, keepdims=True)
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        rng = np.random.default_rng(seed + 991)
        zz = t + mid * sd * rng.normal(size=t.shape)
        if cca_score(t, zz, seed=seed)["cca_mean"] > target:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    rng = np.random.default_rng(seed + 991)
    zz = t + sigma * sd * rng.normal(size=t.shape)
    s = score_embedding(t, zz, with_dci=False, seed=seed)
    return sigma, target, s


def main():
    import os
    rows, t0 = [], time.time()
    combos = list(itertools.product(ARMS, RHOS, DELTAS, SEEDS))

    # RESUMABLE: keep any valid rows already on disk (the 60 ivae_5env runs that
    # completed before the n_env one-hot crash), recompute only what is missing.
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = {(r["arm"], r["rho"], r["delta"], r["seed"]) for r in rows}
        print(f"resuming: {len(done)} rows already on disk", flush=True)

    print(f"{len(combos)} runs total, {len(combos)-len(done)} to do\n")
    print(f"{'arm':>12} {'rho':>5} {'d':>4} {'sd':>3} | {'L_mcc':>6} {'O_mcc':>6} "
          f"{'GAP':>6} | {'L_cca':>6} {'sigma':>6} {'KL':>6} {'AU':>3}", flush=True)

    for arm, rho, delta, seed in combos:
        if (arm, rho, delta, seed) in done:
            continue
        # BUGFIX: the DGP environment index e ranges over the arm's n_env, so the
        # data must be generated with the SAME n_env the model one-hots against.
        # The 2-env violating arm therefore uses genuinely 2-environment data
        # (the iVAE variability condition 2n+1=5 is unsatisfiable), while the
        # 5-env satisfiable arm and the isotropic control both use 5-env data.
        n_env = ARMS[arm]["n_env"]
        d = generate(DGP2Config(rho=rho, delta=delta, n_per_domain=N, seed=seed,
                                n_env=n_env))
        cfg = Model2Config(**ARMS[arm], latent_dim=2, epochs=EPOCHS, seed=seed)
        model, xn, _ = train_model2(d["X"], d["s"], d["e"], cfg)
        z = model.embed(xn)
        t = np.asarray(d["t"], float)

        learned = score_embedding(t, z, mask=d["overlap_region"],
                                  with_dci=False, seed=seed)
        sigma, cca_t, orc = matched_oracle(t, z, seed=seed)

        import torch.nn.functional as F
        xr = torch.tensor(np.asarray(d["X"], np.float32))
        uh = F.one_hot(torch.tensor(d["e"]), cfg.n_env).float()
        diag = vae_diagnostics(model, xn, xr, torch.tensor(d["s"]), uh)

        row = dict(arm=arm, rho=rho, delta=delta, seed=seed, n=N,
                   learned_mcc=learned["mcc_pearson"],
                   learned_mcc_spearman=learned["mcc_spearman"],
                   learned_cca=learned["cca_mean"],
                   learned_mcc_band=learned.get("mcc_band"),
                   n_band=learned.get("n_band"),
                   oracle_sigma=sigma,
                   oracle_mcc=orc["mcc_pearson"],
                   oracle_mcc_spearman=orc["mcc_spearman"],
                   oracle_cca=orc["cca_mean"],
                   gap_mcc=orc["mcc_pearson"] - learned["mcc_pearson"],
                   gap_mcc_spearman=orc["mcc_spearman"] - learned["mcc_spearman"],
                   kl_total=diag["kl_total"], kl_min=diag["kl_per_dim_min"],
                   active_units=diag["active_units"],
                   recon_nats=diag["recon_nats"],
                   posterior_collapse=diag["posterior_collapse"],
                   deterministic_ae=diag["deterministic_ae"])
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{arm:>12} {rho:>5} {delta:>4} {seed:>3} | "
              f"{row['learned_mcc']:>6.3f} {row['oracle_mcc']:>6.3f} "
              f"{row['gap_mcc']:>6.3f} | {row['learned_cca']:>6.3f} "
              f"{sigma:>6.3f} {diag['kl_total']:>6.2f} {diag['active_units']:>3}",
              flush=True)

    df = pd.DataFrame(rows)
    print(f"\n=== gap_mcc by arm x rho (mean over seeds, delta=0) ===", flush=True)
    print(df[df.delta == 0].pivot_table(index="rho", columns="arm",
                                        values="gap_mcc").round(4).to_string(), flush=True)
    print(f"\n=== gap_mcc by arm x delta (mean over seeds, rho=1.0) ===", flush=True)
    print(df[df.rho == 1.0].pivot_table(index="delta", columns="arm",
                                        values="gap_mcc").round(4).to_string(), flush=True)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
