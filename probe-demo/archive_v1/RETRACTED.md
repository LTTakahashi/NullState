# v1 results: RETRACTED, do not cite

Every CSV in this directory was produced by `dgp.py` (v1), which failed an
adversarial audit (28 agents, 33 findings, 17 confirmed). Verdict:

> Unproven, and currently confounded at every measured axis.
> Do not report the central claim.

## Why every number here is void

1. **rho was not a pure knob.** `dgp.py:88` standardises per gene *within each
   dataset*, so lowering rho moved domain B outside the standardisation range
   and the decoder's tanh saturated. Lowering rho therefore also compressed
   domain B's signal 3.4x, opened a per-gene mean gap equivalent to delta ~ 36
   (the delta sweep only reached 4, a 9x mismatch), and created a 30% depth gap.

2. **Every headline metric is a closed-form function of rho.**
   - `batch_removed` = rho identically (max err 0.012)
   - `joint = min(bio, batch)` = rho in 90/90 rows (corr 0.99991)
   - `bio_recovery` floor = 3(1-rho)^2 / (1 + 3(1-rho)^2), and RISES as overlap
     is destroyed; `z = s` alone scores 0.724
   - `bio_recovery_shared`: Var(t | band) = rho^2/12
   - `cross_domain_transfer` = 1 - 4(1-rho)^3, zero crossing at
     rho* = 1 - 4^(-1/3) = 0.370 -- a constant of kNN extrapolation geometry,
     not a threshold of identifiability

3. **The learned model sits ON the analytic ceiling everywhere** (gap <= 0.011).
   There is no measurable non-identifiability anywhere data exists.

4. **The metric cannot order its own hypotheses.** A genuinely non-identified
   oracle at rho=1.0 scores -2.93; a perfectly identified latent at rho=0.05
   scores -2.46. Same reading, opposite ground truth.

5. **The "phase transition" is not supported.** BIC prefers a linear fit
   (-107.24) over every two-segment fit (-104.72); piecewise slopes 0.949 vs
   0.964. `plot._empirical_threshold` returns None on every real file and
   raises KeyError on two of them.

6. **The metrics were rotation-invariant.** kNN-regression R^2 scores a 45deg
   rotated latent the same as an identity one, so no v1 metric could detect
   entanglement -- the phenomenon the probe exists to measure.

7. **The identifiability framing was void by construction.** iVAE (Khemakhem
   2020, Thm 1 (iv)) requires nk+1 distinct auxiliary values -- 2n+1 = 3 for a
   Gaussian prior at n=1. A 2-domain design cannot satisfy this for any n.

8. **`results_decisive.csv` is additionally invalid** on its own terms: audit
   finding B4 shows `run_decisive.py:42` used a NOISELESS oracle (z := t), which
   returns ~1.0 at every rho by construction and is blind to variance-
   normalisation artifacts. Only 4 rows completed before the job was killed.

## Also retracted

Numbers circulated in conversation as "rho=0.25 -> -6.0" and "rho=0.05 -> -14.7"
came from an ad-hoc script that never wrote to disk. They are not reproducible
from any file in this repository and should be treated as withdrawn.

## Replacement

`dgp2.py` + `verify_dgp2.py` (Stage-1 decoupling gate), `metrics2.py`
(rotation-sensitive MCC/CCA/DCI + oracle ceiling), `models2.py` (genuine NB
ELBO + VAE-behaviour diagnostics).
