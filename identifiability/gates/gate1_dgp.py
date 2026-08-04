"""
STAGE-1 GATE: prove that rho and delta are decoupled knobs in dgp2.

This script exists because the v1 DGP failed exactly here. It is a HARD GATE:
nothing downstream (metrics, models, sweeps) may be run until every check
passes. Each check is stated as a falsifiable assertion with an explicit
tolerance, and the script exits non-zero on any failure.

The invariance we require, precisely:

  (I1) POOLED OBSERVABLE STATISTICS ARE rho-INVARIANT.
       Per-gene mean, per-gene variance, per-gene NB-style dispersion, the
       library-size distribution, and the dynamic range of the log-normalised
       matrix must be statistically indistinguishable across the rho sweep.
       This is what v1 violated (delta-equivalent ~36 mean gap, 30% depth gap).

  (I2) THE TWO DOMAINS ARE SYMMETRIC.  std_B/std_A of the shifted latent, and
       depth_B/depth_A, must both be ~1 at every rho. v1 gave 0.344 and 0.699.

  (I3) THE UNSHIFTED AXES DO NOT MOVE WITH rho.  Their per-domain marginals
       must be indistinguishable from each other and across rho: only the
       shifted axis may respond to rho.

  (I4) THE MEASURED OVERLAP EQUALS rho.  The empirical overlap coefficient
       between the two domains' shifted-axis samples must equal rho.

  (I5) delta DOES NOT MOVE ANY OF THE ABOVE.  Sweeping delta at fixed rho must
       leave depth, the latent marginals and the measured overlap unchanged --
       delta may only move the gene composition. Conversely rho at fixed delta
       must not change the size of the removable offset.

  (I6) THE MIXING MAP IS INVERTIBLE AND WELL CONDITIONED.

Statistical convention: two-sample KS between the rho=1 reference sample and
each other rho, on POOLED quantities. We require p > 0.01 (not 0.1) because at
n = 2*3000 cells x 500 genes KS is powerful enough to reject on differences far
below anything that could confound a metric; the effect size (normalised mean
difference / ratio) is reported alongside and carries the real tolerance.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import sys
import numpy as np
from scipy import stats

from core.dgp import DGP2Config, generate, mixing_conditioning, sample_shifted_axis

RHOS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.05]

# CERTIFIED delta range. Beyond delta ~ 2 the knobs stop being decoupled: a very
# large compositional offset pushes genes into the count-noise floor, which costs
# domain B decodability on its own, and that cost compounds as B's biology
# concentrates at low rho. Measured worst-case within-domain decodability ratio
# B/A over the rho sweep:
#     delta   1.0    1.5    2.0    2.5    3.0    4.0
#     B/A    0.995  1.001  0.983  0.964  0.925  0.837
# delta = 2 already induces a mean log-expression gap of 1.30, which exceeds the
# largest gap rho itself can induce (0.99), so the delta axis SPANS the rho axis
# inside the certified region -- the requirement v1 failed (its delta reached 4
# against a rho-induced gap equivalent to ~36). Larger deltas remain available as
# an explicitly declared EXTREME arm, outside this certificate.
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0]
DELTA_EXTREME = [2.5, 3.0, 4.0]
SEEDS = [0, 1, 2]
N = 3000

FAILURES: list[str] = []
NOTES: list[str] = []


def check(name: str, ok: bool, detail: str):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def lognorm(X):
    lib = X.sum(1, keepdims=True)
    return np.log1p(X / (lib + 1e-8) * np.median(lib))


def gene_stats(X):
    """Pooled per-gene statistics on RAW counts + the log-normalised range."""
    mu = X.mean(0)
    var = X.var(0)
    # method-of-moments NB dispersion phi:  var = mu + phi * mu^2
    phi = np.clip((var - mu) / np.maximum(mu ** 2, 1e-12), 0, None)
    Ln = lognorm(X)
    return dict(mean=mu, var=var, disp=phi, lib=X.sum(1),
                dyn_range=float(np.percentile(Ln, 99.5) - np.percentile(Ln, 0.5)))


def empirical_ovl(a, b, bins=60, lo=None, hi=None):
    """Overlap coefficient int min(p_A,p_B) via a common histogram."""
    lo = min(a.min(), b.min()) if lo is None else lo
    hi = max(a.max(), b.max()) if hi is None else hi
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, edges, density=True)
    pb, _ = np.histogram(b, edges, density=True)
    return float(np.minimum(pa, pb).sum() * np.diff(edges)[0])


# ---------------------------------------------------------------------------
def main():
    cfg0 = DGP2Config(n_per_domain=N)
    print("=" * 78)
    print("STAGE-1 GATE  |  dgp2 decoupling verification")
    print("=" * 78)

    # ---- I6: mixing map ---------------------------------------------------
    print("\n[I6] mixing map invertibility / conditioning")
    mc = mixing_conditioning(cfg0)
    check("square layers orthogonal", max(mc["square_conds"]) < 1.01,
          f"max cond = {max(mc['square_conds']):.4f} (need < 1.01)")
    check("expansion full column rank",
          mc["expansion_rank"] == cfg0.latent_dim,
          f"rank {mc['expansion_rank']} / {cfg0.latent_dim}")
    check("expansion well conditioned", mc["expansion_cond"] < 5.0,
          f"cond = {mc['expansion_cond']:.3f} (need < 5)")
    # injectivity: no two distinct latents collide in gene space
    from core.dgp import _cached_reference
    g, _, _, _ = _cached_reference(cfg0)
    rng = np.random.default_rng(0)
    tt = rng.normal(size=(2000, cfg0.latent_dim))
    Gt = g(tt)
    dt = stats.spearmanr(
        np.linalg.norm(tt[:1000] - tt[1000:], axis=1),
        np.linalg.norm(Gt[:1000] - Gt[1000:], axis=1)).statistic
    check("distance monotonicity (injective in practice)", dt > 0.9,
          f"spearman(|dt|, |dg|) = {dt:.4f} (need > 0.9)")

    # ---- I4: measured overlap == rho -------------------------------------
    print("\n[I4] empirical overlap coefficient == rho")
    for rho in RHOS:
        rng = np.random.default_rng(7)
        a = sample_shifted_axis(rng, 200000, rho, 0)
        b = sample_shifted_axis(rng, 200000, rho, 1)
        ovl = empirical_ovl(a, b, bins=200, lo=0.0, hi=1.0)
        check(f"OVL(rho={rho})", abs(ovl - rho) < 0.02,
              f"measured {ovl:.4f} vs target {rho:.2f} (|err| {abs(ovl-rho):.4f})")

    # ---- generate the rho sweep ------------------------------------------
    # Run the tight invariance test at delta = 0. This is the PRIMARY-ESTIMAND
    # arm (gap vs rho at fixed delta), and it is free of the rho x delta
    # interaction quantified in I1c below. Running it at delta > 0 would test
    # the DGP and that interaction simultaneously and confound the two.
    print("\n[I1] pooled observable statistics are rho-invariant  (delta = 0)")
    data = {}
    for rho in RHOS:
        for seed in SEEDS:
            data[(rho, seed)] = generate(
                DGP2Config(rho=rho, delta=0.0, n_per_domain=N, seed=seed))

    ref = [gene_stats(data[(1.0, s)]["X"]) for s in SEEDS]
    ref_mean = np.concatenate([r["mean"] for r in ref])
    ref_var = np.concatenate([r["var"] for r in ref])
    ref_disp = np.concatenate([r["disp"] for r in ref])
    ref_lib = np.concatenate([r["lib"] for r in ref])
    ref_dyn = np.mean([r["dyn_range"] for r in ref])

    # seed-to-seed spread at FIXED rho = the irreducible noise floor. Any
    # rho effect must be judged against this, not against zero.
    floor_mean = np.mean([abs(ref[i]["mean"].mean() - ref[j]["mean"].mean())
                          for i in range(3) for j in range(i + 1, 3)])
    NOTES.append(f"seed-to-seed |d mean| floor at fixed rho = {floor_mean:.5f}")

    print(f"       {'rho':>5} {'mean_ratio':>11} {'var_ratio':>10} {'disp_ratio':>11} "
          f"{'depth_ratio':>12} {'dyn_ratio':>10} {'KS_p(mean)':>11}")
    for rho in RHOS:
        gs = [gene_stats(data[(rho, s)]["X"]) for s in SEEDS]
        m = np.concatenate([x["mean"] for x in gs])
        v = np.concatenate([x["var"] for x in gs])
        d = np.concatenate([x["disp"] for x in gs])
        lb = np.concatenate([x["lib"] for x in gs])
        dy = np.mean([x["dyn_range"] for x in gs])
        r_m, r_v = m.mean() / ref_mean.mean(), v.mean() / ref_var.mean()
        r_d = d.mean() / ref_disp.mean()
        r_l, r_y = lb.mean() / ref_lib.mean(), dy / ref_dyn
        p = stats.ks_2samp(m, ref_mean).pvalue
        print(f"       {rho:>5} {r_m:>11.4f} {r_v:>10.4f} {r_d:>11.4f} "
              f"{r_l:>12.4f} {r_y:>10.4f} {p:>11.3g}")
        check(f"per-gene mean rho={rho}", abs(r_m - 1) < 0.02, f"ratio {r_m:.4f}")
        check(f"per-gene var  rho={rho}", abs(r_v - 1) < 0.05, f"ratio {r_v:.4f}")
        check(f"dispersion    rho={rho}", abs(r_d - 1) < 0.10, f"ratio {r_d:.4f}")
        check(f"library size  rho={rho}", abs(r_l - 1) < 0.02, f"ratio {r_l:.4f}")
        check(f"dynamic range rho={rho}", abs(r_y - 1) < 0.05, f"ratio {r_y:.4f}")

    # ---- I1c: the rho x delta INTERACTION, quantified and declared -------
    # Not zero, and not removable. A multiplicative batch offset on the simplex
    # is renormalised per cell (softmax), and the normaliser
    # Z(t) = sum_g comp_g(t) e^{delta d_g} depends on that cell's biology. When
    # rho falls, domain B's biology concentrates, so the distribution of Z
    # shifts and pooled per-gene variance moves slightly. This is intrinsic to
    # compositional data with a multiplicative batch effect, i.e. a property of
    # scRNA-seq, not a pipeline artifact. (Aitchison/CLR centring does NOT fix
    # it: subtracting a per-cell scalar is a no-op under softmax renormalisation
    # -- verified, identical to 4 decimals.)
    # It is bounded and declared here rather than hidden. Compare v1, whose
    # analogous coupling was a 3.4x SNR compression and a 30% depth gap.
    print("\n[I1c] rho x delta interaction (intrinsic to compositional data)")
    print(f"       {'rho':>5} {'var ratio':>10} {'disp ratio':>11}   (vs rho=1, delta=1)")
    ref1 = None
    worst_v = worst_d = 0.0
    for rho in RHOS:
        acc = []
        for seed in SEEDS:
            gs = gene_stats(generate(DGP2Config(rho=rho, delta=1.0,
                                                n_per_domain=N, seed=seed))["X"])
            acc.append([gs["mean"].mean(), gs["var"].mean(), gs["disp"].mean()])
        cur = np.array(acc).mean(0)
        if ref1 is None:
            ref1 = cur
        rv, rd = cur[1] / ref1[1], cur[2] / ref1[2]
        worst_v, worst_d = max(worst_v, abs(rv - 1)), max(worst_d, abs(rd - 1))
        print(f"       {rho:>5} {rv:>10.4f} {rd:>11.4f}")
    check("rho x delta interaction is bounded", worst_v < 0.10 and worst_d < 0.20,
          f"max |var dev| {worst_v:.3f} (<0.10), max |disp dev| {worst_d:.3f} (<0.20)")
    NOTES.append(f"declared rho x delta interaction at delta=1: variance up to "
                 f"{100*worst_v:.1f}%, dispersion up to {100*worst_d:.1f}%")

    # ---- I1b: WITHIN-DOMAIN DECODABILITY -- the direct SNR probe ----------
    # This is the check that matters most. v1's fatal defect was that rho
    # silently compressed domain B's signal 3.4x, i.e. rho secretly controlled
    # SNR. Per-gene variance/dispersion are only proxies for that; the direct
    # test is whether the true latent is equally decodable from the counts
    # WITHIN a single domain at every rho and delta. Fit and score inside one
    # domain, so cross-domain support mismatch cannot contribute.
    print("\n[I1b] within-domain decodability of t (direct SNR probe; must be flat)")
    print(f"       {'rho':>5} {'delta':>6} {'A:t0':>7} {'B:t0':>7} {'B/A':>6} "
          f"{'A:t1':>7} {'B:t1':>7}")

    def within_r2(d, dom, axis, alpha=10.0):
        m = d["s"] == dom
        L = lognorm(d["X"])[m]
        y = d["t"][m, axis]
        L = L - L.mean(0)
        ntr = len(y) // 2
        # closed-form ridge on the train half, scored on the held-out half
        Xtr, ytr = L[:ntr], y[:ntr] - y[:ntr].mean()
        A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
        wgt = np.linalg.solve(A, Xtr.T @ ytr)
        pred = L[ntr:] @ wgt + y[:ntr].mean()
        return float(1 - ((y[ntr:] - pred) ** 2).sum() /
                     ((y[ntr:] - y[ntr:].mean()) ** 2).sum())

    snr = {}
    for delta in (0.0, 1.0, 2.0):
        for rho in RHOS:
            d = generate(DGP2Config(rho=rho, delta=delta, n_per_domain=N, seed=0))
            a0, b0 = within_r2(d, 0, 0), within_r2(d, 1, 0)
            a1, b1 = within_r2(d, 0, 1), within_r2(d, 1, 1)
            snr[(rho, delta)] = (a0, b0, a1, b1)
            print(f"       {rho:>5} {delta:>6} {a0:>7.4f} {b0:>7.4f} "
                  f"{b0/max(a0,1e-9):>6.3f} {a1:>7.4f} {b1:>7.4f}")
            check(f"A/B decodability symmetry rho={rho} d={delta}",
                  abs(b0 / max(a0, 1e-9) - 1) < 0.15, f"B/A = {b0/max(a0,1e-9):.3f}")
    # flatness across rho at each delta (this is the v1 3.4x-compression test)
    for delta in (0.0, 1.0, 2.0):
        vals = [snr[(r, delta)][0] for r in RHOS] + [snr[(r, delta)][1] for r in RHOS]
        spread = max(vals) - min(vals)
        check(f"decodability flat across rho (delta={delta})", spread < 0.10,
              f"range {min(vals):.3f}-{max(vals):.3f}, spread {spread:.3f}")

    # ---- I1d: the VALIDITY BOUNDARY, published rather than hidden --------
    # Where does the decoupling stop holding? Reporting this is what makes the
    # certified region meaningful; v1 had no such boundary and was used far
    # outside the region where its knobs were separable.
    print("\n[I1d] validity boundary: worst B/A decodability outside the certified range")
    for delta in DELTA_EXTREME:
        worst = min(within_r2(generate(DGP2Config(rho=r, delta=delta,
                                                  n_per_domain=N, seed=0)), 1, 0)
                    / within_r2(generate(DGP2Config(rho=r, delta=delta,
                                                    n_per_domain=N, seed=0)), 0, 0)
                    for r in RHOS)
        print(f"       delta={delta}: worst B/A = {worst:.3f}  "
              f"{'(OUTSIDE certificate)' if worst < 0.98 else ''}")
        NOTES.append(f"delta={delta} is OUTSIDE the decoupling certificate "
                     f"(worst B/A = {worst:.3f})")

    # ---- I2: domain symmetry ---------------------------------------------
    print("\n[I2] domain symmetry  (v1: std_B/std_A = 0.344, depth_B/depth_A = 0.699)")
    print(f"       {'rho':>5} {'std_B/std_A':>12} {'depth_B/depth_A':>16} {'meanExpr_B/A':>13}")
    for rho in RHOS:
        rs, rd, rm = [], [], []
        for seed in SEEDS:
            d = data[(rho, seed)]
            t0, s, X = d["t"][:, 0], d["s"], d["X"]
            rs.append(t0[s == 1].std() / t0[s == 0].std())
            rd.append(X[s == 1].sum(1).mean() / X[s == 0].sum(1).mean())
            rm.append(X[s == 1].mean() / X[s == 0].mean())
        rs, rd, rm = np.mean(rs), np.mean(rd), np.mean(rm)
        print(f"       {rho:>5} {rs:>12.4f} {rd:>16.4f} {rm:>13.4f}")
        check(f"latent std symmetry rho={rho}", abs(rs - 1) < 0.05, f"{rs:.4f}")
        check(f"depth symmetry rho={rho}", abs(rd - 1) < 0.02, f"{rd:.4f}")

    # ---- I3: unshifted axes untouched by rho -----------------------------
    print("\n[I3] unshifted axes do not respond to rho")
    base = np.concatenate([data[(1.0, s)]["t"][:, 1:].ravel() for s in SEEDS])
    for rho in RHOS:
        cur = np.concatenate([data[(rho, s)]["t"][:, 1:].ravel() for s in SEEDS])
        p = stats.ks_2samp(cur, base).pvalue
        # also: are the two DOMAINS' unshifted marginals identical?
        pa = np.concatenate([data[(rho, s)]["t"][data[(rho, s)]["s"] == 0, 1:].ravel()
                             for s in SEEDS])
        pb = np.concatenate([data[(rho, s)]["t"][data[(rho, s)]["s"] == 1, 1:].ravel()
                             for s in SEEDS])
        p_ab = stats.ks_2samp(pa, pb).pvalue
        check(f"unshifted vs rho=1 (rho={rho})", p > 0.01, f"KS p = {p:.3g}")
        check(f"unshifted A vs B (rho={rho})", p_ab > 0.01, f"KS p = {p_ab:.3g}")

    # ---- I5: delta moves composition ONLY --------------------------------
    # NOTE on the OVL column: empirical_ovl is a histogram estimator and is
    # DOWNWARD biased at finite n (0.354 vs a true 0.40 at n=3000/domain,
    # 120 bins). I4 above verifies it converges to rho at n=2e5. So the check
    # here is INVARIANCE ACROSS delta at fixed estimator settings, not equality
    # to the nominal rho -- comparing a biased estimate to the nominal value
    # would be testing the estimator, not the DGP.
    print("\n[I5] delta is orthogonal to rho and to depth  (rho = 0.4)")
    print(f"       {'delta':>6} {'depth_B/depth_A':>16} {'OVL_meas':>9} "
          f"{'lib_ratio_vs_d0':>16} {'log-expr gap':>13}")
    d0 = generate(DGP2Config(rho=0.4, delta=0.0, n_per_domain=N, seed=0))
    lib0 = d0["X"].sum(1).mean()
    ovl0 = empirical_ovl(d0["t"][d0["s"] == 0, 0], d0["t"][d0["s"] == 1, 0], bins=120)
    sigs = []
    for delta in DELTAS:
        dd = generate(DGP2Config(rho=0.4, delta=delta, n_per_domain=N, seed=0))
        X, s, t0 = dd["X"], dd["s"], dd["t"][:, 0]
        rd = X[s == 1].sum(1).mean() / X[s == 0].sum(1).mean()
        ovl = empirical_ovl(t0[s == 0], t0[s == 1], bins=120)
        rl = X.sum(1).mean() / lib0
        Ln = lognorm(X)
        sig = float(np.abs(Ln[s == 1].mean(0) - Ln[s == 0].mean(0)).mean())
        sigs.append(sig)
        print(f"       {delta:>6} {rd:>16.4f} {ovl:>9.4f} {rl:>16.4f} {sig:>13.4f}")
        check(f"depth invariant to delta={delta}", abs(rd - 1) < 0.02, f"{rd:.4f}")
        check(f"library invariant to delta={delta}", abs(rl - 1) < 0.02, f"{rl:.4f}")
        check(f"latent overlap invariant to delta={delta}", abs(ovl - ovl0) < 0.005,
              f"OVL {ovl:.4f} vs delta=0 {ovl0:.4f}")

    # delta must actually DO something. An inert knob is also a bug -- v1's
    # adv_lambda was inert, and the first draft of v2 made delta inert by
    # normalising the batch direction to unit L2 norm (delta/sqrt(G) per gene).
    # Measure liveness at rho = 1, where the gap is attributable to delta ALONE;
    # at rho = 0.4 the rho-induced gap is a large additive baseline that would
    # mask the ratio.
    check("delta is a live knob, monotone at fixed rho=0.4",
          all(sigs[i] < sigs[i + 1] for i in range(len(sigs) - 1)),
          f"gap per delta = {[round(x, 4) for x in sigs]}")
    pure = []
    for delta in DELTAS:
        dd = generate(DGP2Config(rho=1.0, delta=delta, n_per_domain=N, seed=0))
        Ln = lognorm(dd["X"]); s = dd["s"]
        pure.append(float(np.abs(Ln[s == 1].mean(0) - Ln[s == 0].mean(0)).mean()))
    check("delta spans a non-trivial range at rho=1",
          all(pure[i] < pure[i + 1] for i in range(len(pure) - 1)) and pure[-1] > 10 * pure[0],
          f"gap per delta = {[round(x, 4) for x in pure]}")

    # AUDIT FINDING B1 REDUX: in v1 the rho-induced gap was equivalent to
    # delta ~ 36 while the delta sweep only reached 4 -- a 9x mismatch that made
    # the "rho vs delta dissociation" a comparison against a crippled axis.
    # Require the delta sweep to SPAN the gap that rho itself can induce.
    rho_max_gap = 0.0
    for rho in RHOS:
        dd = generate(DGP2Config(rho=rho, delta=0.0, n_per_domain=N, seed=0))
        Ln = lognorm(dd["X"]); s = dd["s"]
        rho_max_gap = max(rho_max_gap,
                          float(np.abs(Ln[s == 1].mean(0) - Ln[s == 0].mean(0)).mean()))
    check("delta axis spans the rho-induced gap range",
          pure[-1] > rho_max_gap,
          f"max delta gap {pure[-1]:.4f} vs max rho-induced gap {rho_max_gap:.4f} "
          f"(v1 was 4 vs ~36)")

    # ---- I5b: REMOVABILITY -- the property that actually separates rho and
    # delta.  rho SHOULD induce a mean-expression gap: that is what biological
    # support mismatch IS. (The v1 defect was not the existence of a gap but
    # that a *pipeline artifact* -- pooled standardisation into tanh saturation
    # -- manufactured one on top of it.)  The discriminator is Ben-David's
    # decomposition: a divergence term that an alignment can remove, versus an
    # irreducible lambda.  So: estimate the best per-gene affine batch map on
    # the SHARED OVERLAP BAND (where both domains carry the same biology by
    # construction) and subtract it. delta's gap must collapse to ~0; rho's
    # must survive.
    print("\n[I5b] removability: delta's gap is affine-removable, rho's is not")
    print(f"       {'setting':>18} {'raw gap':>9} {'residual':>9} {'removed %':>10}")

    def gap_and_residual(dd):
        Ln = lognorm(dd["X"]); s = dd["s"]; band = dd["overlap_region"]
        raw = float(np.abs(Ln[s == 1].mean(0) - Ln[s == 0].mean(0)).mean())
        a, b = band & (s == 0), band & (s == 1)
        if a.sum() < 50 or b.sum() < 50:
            return raw, float("nan"), float("nan")
        shift = Ln[b].mean(0) - Ln[a].mean(0)          # affine map fit on the band
        Lc = Ln.copy(); Lc[s == 1] -= shift
        res = float(np.abs(Lc[s == 1].mean(0) - Lc[s == 0].mean(0)).mean())
        return raw, res, 100.0 * (1 - res / max(raw, 1e-12))

    # pure delta (full overlap): must be ~fully removable
    for delta in (1.0, 4.0):
        raw, res, pct = gap_and_residual(
            generate(DGP2Config(rho=1.0, delta=delta, n_per_domain=N, seed=0)))
        print(f"       {'rho=1.0 d=' + str(delta):>18} {raw:>9.4f} {res:>9.4f} {pct:>9.1f}%")
        check(f"delta={delta} gap is removable (rho=1)", pct > 90,
              f"{pct:.1f}% removed, residual {res:.4f}")
    # pure rho (no batch effect): must NOT be removable
    for rho in (0.6, 0.4, 0.2):
        raw, res, pct = gap_and_residual(
            generate(DGP2Config(rho=rho, delta=0.0, n_per_domain=N, seed=0)))
        print(f"       {'rho=' + str(rho) + ' d=0.0':>18} {raw:>9.4f} {res:>9.4f} {pct:>9.1f}%")
        check(f"rho={rho} gap is NOT removable", pct < 60,
              f"{pct:.1f}% removed, residual {res:.4f}")
        NOTES.append(f"rho={rho}: raw gap {raw:.4f}, irreducible residual {res:.4f}")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 78)
    for n in NOTES:
        print("  note:", n)
    if FAILURES:
        print(f"\nGATE FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print("   -", f)
        return 1
    print("\nGATE PASSED: rho and delta are decoupled. Stage 2 may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
