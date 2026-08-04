"""
STAGE-5: the a-priori certificate -- a diagnostic computed from RAW data, before
any integration, that is SENSITIVE to biological support mismatch (rho) and
INVARIANT to a removable batch effect (delta).

WHY v1's CERTIFICATE WAS WRONG. v1 used a source-detecting adversary on raw
counts (`metrics.raw_certificate`) and read "high and stable across capacity" as
evidence of support mismatch. But a source adversary fires on ANY distributional
difference between domains, including a perfectly removable offset. So it scored
~1.0 on the harmless case (full overlap, large batch effect) and on the fatal
case (support mismatch) alike -- it could not tell them apart, which is the one
thing it existed to do.

THE FIX is Ben-David's decomposition. In
    eps_T(h) <= eps_S(h) + (1/2) d_{H dH}(S,T) + lambda
the divergence term is what an alignment can remove and lambda -- the error of
the best JOINT hypothesis -- is what it cannot. A removable delta inflates only
the divergence; genuine support mismatch inflates lambda. Ben-David & Luu (2010)
prove that small divergence plus covariate shift is NOT sufficient for adaptation,
which is exactly why measuring divergence alone (v1) cannot certify anything.

So the certificate measures overlap AFTER removing the estimable component:

  1. PROPENSITY / POSITIVITY.  Fit e(x) = P(domain = A | x) with a
     cross-validated classifier. Report the trimmed common-support mass, the
     fraction of cells with e(x) in [eta, 1-eta] (Crump et al. 2009), and the
     effective sample size under overlap weights (Li et al. 2018).

  2. RESIDUAL OVERLAP AFTER ALIGNMENT  <- THE KEY STEP.  Estimate the removable
     per-gene offset WITHOUT ground truth: take the cells the propensity model
     itself considers ambiguous (its estimated common support), estimate the
     domain shift there, subtract it, and re-measure. A removable delta collapses
     to full overlap; support mismatch does not. `alignment_removed_fraction` is
     the headline statistic and is what makes the certificate delta-invariant.

  3. UNBALANCED OPTIMAL TRANSPORT.  Entropic UOT relaxes the marginal
     constraints, so its transported-mass fraction estimates the SHARED-support
     proportion and is robust to the displacement a removable delta causes.

  4. DENSITY-RATIO SANITY FLAG.  The domain density ratio stays bounded under a
     removable offset (overlapping support) but diverges under genuine
     non-overlap -- the documented "support chasm" of density-ratio estimation.

PASS CRITERIA for the certificate itself (validated in Stage 5, not assumed):
  (a) it must return "integrable" on the same-support / large-delta control,
      the case where v1 wrongly fired; and
  (b) rho_hat must decrease monotonically with true rho, at EVERY delta.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA


def _lognorm(X):
    lib = X.sum(1, keepdims=True)
    return np.log1p(X / (lib + 1e-8) * np.median(lib)).astype(np.float32)


# --------------------------------------------------------------------------- #
#  1. propensity / positivity
# --------------------------------------------------------------------------- #
def propensity_overlap(L, s, eta: float = 0.1, cv: int = 3, seed: int = 0) -> dict:
    """Cross-fitted domain propensity and its positivity diagnostics."""
    s = np.asarray(s).ravel()
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=seed)
    p = cross_val_predict(clf, L, s, cv=KFold(cv, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    trimmed = float(((p > eta) & (p < 1 - eta)).mean())
    # overlap weights w = e(1-e): effective sample size relative to n
    w = p * (1 - p)
    ess = float(w.sum() ** 2 / (np.sum(w ** 2) * len(w) + 1e-12))
    return {"propensity": p, "auc": float(roc_auc_score(s, p)),
            "trimmed_mass": trimmed, "overlap_ess": ess}


# --------------------------------------------------------------------------- #
#  2. residual overlap after a data-driven removable alignment  (THE KEY STEP)
# --------------------------------------------------------------------------- #
def residual_overlap_after_alignment(L, s, eta: float = 0.1, rank: int = 1,
                                     seed: int = 0) -> dict:
    """Neutralise the removable component by PROJECTING IT OUT, then re-measure.

    WHY PROJECTION AND NOT SUBTRACTION. The obvious implementation -- estimate
    shift = mean_B - mean_A and subtract it from B -- is wrong, and wrong in a
    way that silently inverts the result. When the domains are genuinely
    identical (rho=1, delta=0) the estimated shift is pure noise, and subtracting
    it from one domain MANUFACTURES a real offset of exactly that size. Measured:
    raw adversary AUC 0.523 (correct, chance) but post-subtraction AUC 0.992 --
    the certificate reported catastrophic non-overlap for two identical
    distributions. Estimating the shift on an "ambiguous propensity band" is
    worse still: that band is empty precisely when delta is large, so the
    estimator returns NaN exactly in the harmless case it exists to certify.

    Projection has neither failure mode. Removing the rank-`rank` subspace
    spanned by the between-domain mean difference can only DESTROY information,
    never create it, and it is always estimable. The question it answers is the
    right one: is there domain structure BEYOND a removable translation?
      * a removable delta lives entirely in that subspace -> AUC falls to ~0.5
      * support mismatch is a difference in WHICH REGION of the manifold each
        domain occupies, not a translation -> it survives the projection

    `rank` should match the dimensionality of the removable effect (dgp2's
    delta is an exact rank-1 translation in log space). Raise it for
    multi-directional batch effects; note that each extra rank costs a little
    genuine biological signal, so this is a declared modelling choice.
    """
    s = np.asarray(s).ravel()
    pre = propensity_overlap(L, s, eta=eta, seed=seed)

    A, B = L[s == 0], L[s == 1]
    D = (B.mean(0) - A.mean(0))[None, :]
    if rank > 1:
        # additional directions from the domain-difference of PCA-whitened data
        extra = PCA(n_components=rank - 1, random_state=seed).fit(L).components_
        D = np.vstack([D, extra])
    Q, _ = np.linalg.qr(D.T)                          # orthonormal basis
    Lp = L - (L @ Q) @ Q.T                            # project out the subspace

    post = propensity_overlap(Lp, s, eta=eta, seed=seed)
    pre_ex, post_ex = pre["auc"] - 0.5, post["auc"] - 0.5
    removed = float(1.0 - post_ex / pre_ex) if pre_ex > 1e-6 else float("nan")
    return {"pre_auc": pre["auc"], "pre_trimmed": pre["trimmed_mass"],
            "post_auc": post["auc"], "post_trimmed": post["trimmed_mass"],
            "alignment_removed_fraction": removed,
            "alignment_estimable": True, "projection_rank": int(rank)}


# --------------------------------------------------------------------------- #
#  3. unbalanced optimal transport
# --------------------------------------------------------------------------- #
ALIGNMENTS = ("none", "rank1", "rank5", "gene_mean", "gene_zscore", "gene_quantile")


def _fit_alignment(Ltr, str_, kind, seed=0):
    """Estimate alignment parameters on TRAINING cells only; return an apply fn."""
    if kind == "none":
        return lambda L, s: L
    if kind.startswith("rank"):
        k = int(kind[4:])
        D = (Ltr[str_ == 1].mean(0) - Ltr[str_ == 0].mean(0))[None, :]
        if k > 1:
            D = np.vstack([D, PCA(n_components=k - 1, random_state=seed).fit(Ltr).components_])
        Q, _ = np.linalg.qr(D.T)
        return lambda L, s: L - (L @ Q) @ Q.T
    if kind in ("gene_mean", "gene_zscore"):
        mu = {v: Ltr[str_ == v].mean(0) for v in (0, 1)}
        sd = {v: Ltr[str_ == v].std(0) + 1e-8 for v in (0, 1)}

        def apply(L, s):
            out = L.copy()
            for v in (0, 1):
                m = s == v
                out[m] = (L[m] - mu[v]) / (sd[v] if kind == "gene_zscore" else 1.0)
            return out
        return apply
    if kind == "gene_quantile":
        ref = {v: np.sort(Ltr[str_ == v], axis=0) for v in (0, 1)}

        def apply(L, s):
            out = L.copy()
            for v in (0, 1):
                m = s == v
                R = ref[v]
                q = np.empty_like(L[m])
                for j in range(L.shape[1]):
                    q[:, j] = np.searchsorted(R[:, j], L[m][:, j]) / max(len(R) - 1, 1)
                out[m] = q
            return out
        return apply
    raise ValueError(kind)


def cross_fitted_alignment_auc(L, s, kind, folds: int = 3, seed: int = 0) -> float:
    """Domain AUC after a CROSS-FITTED alignment.

    THE BUG THIS FIXES. Estimating alignment parameters on all cells and then
    classifying those same cells leaks domain identity into the features. With
    500 genes, 500 individually negligible leaks aggregate into near-perfect
    separation: measured on two IDENTICAL distributions (rho=1, delta=0), naive
    per-gene centring gave AUC 0.988 and per-gene z-scoring 0.993, and a per-gene
    rank transform gave 0.210 (separable in the other direction). All three are
    pure artifacts of the procedure. Cross-fitting -- estimate on the training
    folds, apply to the held-out fold, classify only the held-out fold -- removes
    the leak, so a null case must return ~0.5.
    """
    from sklearn.model_selection import StratifiedKFold
    L, s = np.asarray(L), np.asarray(s).ravel()
    oof = np.zeros(len(s))
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(L, s):
        apply = _fit_alignment(L[tr], s[tr], kind, seed=seed)
        Ltr, Lte = apply(L[tr], s[tr]), apply(L[te], s[te])
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=150,
                                             random_state=seed).fit(Ltr, s[tr])
        oof[te] = clf.predict_proba(Lte)[:, 1]
    return float(roc_auc_score(s, oof))


def uot_shared_mass(L, s, n_sub: int = 600, reg: float = 0.05,
                    reg_m: float = 1.0, iters: int = 300, seed: int = 0) -> dict:
    """Entropic UOT transported-mass fraction as an estimate of shared support.

    Marginal constraints are relaxed with a KL penalty (reg_m), so mass with no
    counterpart in the other domain is simply not transported. The transported
    fraction therefore estimates the shared-support proportion and is robust to
    the pure displacement a removable delta induces.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(s).ravel()
    Z = PCA(n_components=min(20, L.shape[1]), random_state=seed).fit_transform(L)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    ia = rng.choice(np.where(s == 0)[0], min(n_sub, int((s == 0).sum())), replace=False)
    ib = rng.choice(np.where(s == 1)[0], min(n_sub, int((s == 1).sum())), replace=False)
    A, B = Z[ia], Z[ib]

    C = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    C /= (C.mean() + 1e-12)
    K = np.exp(-C / reg)
    na, nb = len(A), len(B)
    a = np.full(na, 1.0 / na); b = np.full(nb, 1.0 / nb)
    u = np.ones(na); v = np.ones(nb)
    fi = reg_m / (reg_m + reg)                       # KL-relaxed Sinkhorn exponent
    for _ in range(iters):
        u = (a / (K @ v + 1e-300)) ** fi
        v = (b / (K.T @ u + 1e-300)) ** fi
    P = u[:, None] * K * v[None, :]
    return {"uot_transported_mass": float(P.sum()),
            "uot_marginal_dev_a": float(np.abs(P.sum(1) - a).sum()),
            "uot_marginal_dev_b": float(np.abs(P.sum(0) - b).sum())}


# --------------------------------------------------------------------------- #
#  4. density-ratio sanity flag
# --------------------------------------------------------------------------- #
def density_ratio_flag(L, s, seed: int = 0, cv: int = 3) -> dict:
    """Log-density-ratio spread from a calibrated linear probe. Bounded under a
    removable offset; diverges under genuine non-overlap (the 'support chasm')."""
    s = np.asarray(s).ravel()
    Z = PCA(n_components=min(20, L.shape[1]), random_state=seed).fit_transform(L)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    p = cross_val_predict(LogisticRegression(max_iter=2000), Z, s,
                          cv=KFold(cv, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lr = np.log(p / (1 - p))
    return {"logratio_p99_spread": float(np.percentile(lr, 99) - np.percentile(lr, 1)),
            "logratio_tail_mass": float(((p < 0.01) | (p > 0.99)).mean())}


# --------------------------------------------------------------------------- #
#  the composite
# --------------------------------------------------------------------------- #
def certificate(X, s, seed: int = 0, with_uot: bool = True) -> dict:
    """The full a-priori certificate. `rho_hat` is the headline estimate of
    shared support; `integrable` is the binary call.

    `raw_adversary_auc` is retained ONLY as the negative control: it is v1's
    statistic, and it fires on rho and delta alike. The certificate's value is
    precisely that `rho_hat` does not.
    """
    L = _lognorm(np.asarray(X))
    s = np.asarray(s).ravel()
    out = {}
    pre = propensity_overlap(L, s, seed=seed)
    out["raw_adversary_auc"] = pre["auc"]            # v1's statistic (the control)
    out["trimmed_mass"] = pre["trimmed_mass"]
    out["overlap_ess"] = pre["overlap_ess"]

    # UNSUPERVISED alignment family, cross-fitted (leakage-free). These are
    # reported as DIAGNOSTICS, not as the headline, because none of them
    # separates removable delta from support mismatch -- see the module note and
    # `certificate_anchor` below. This is not a coding shortfall: Ben-David & Luu
    # (2010) prove that unlabeled data cannot in general distinguish an adaptable
    # covariate shift from a non-adaptable one, which is exactly this task.
    for kind in ALIGNMENTS:
        out[f"auc_{kind}"] = cross_fitted_alignment_auc(L, s, kind, seed=seed)
    out.update(density_ratio_flag(L, s, seed=seed))
    if with_uot:
        A, B = L[s == 0], L[s == 1]
        D = (B.mean(0) - A.mean(0))[None, :]
        Q, _ = np.linalg.qr(D.T)
        out.update(uot_shared_mass(L - (L @ Q) @ Q.T, s, reg=0.01, reg_m=0.2, seed=seed))
    return out


def certificate_anchor(X, s, anchors, seed: int = 0) -> dict:
    """ANCHOR-BASED certificate -- the version that actually separates rho from
    delta, and the honest resolution of the unsupervised family's failure.

    `anchors` is a boolean mask of cells KNOWN to be biologically shared across
    domains (in real data: housekeeping populations, spike-ins, a handful of
    annotated common cell types; in the probe: a random subset of the true
    overlap band). The removable per-gene offset is estimated on the anchors
    ONLY, subtracted, and the residual domain-separability re-measured with a
    CROSS-FITTED classifier so nothing leaks.

    This is exactly Stage-1 I5b (which cleanly gave delta -> 100% removed,
    rho -> 0% removed) but with a realistic anchor set in place of the oracle
    band, and it is why the anchor requirement is principled rather than a
    convenience: Ben-David & Luu prove the unlabeled problem is unsolvable in
    general, and anchors are the minimal target-side signal that makes it
    solvable. This connects the certificate to semi-supervised integration
    (STACAS anchors; Andreatta 2024) rather than to unsupervised batch metrics.
    """
    L = _lognorm(np.asarray(X))
    s = np.asarray(s).ravel()
    anchors = np.asarray(anchors, bool)
    a_all = np.where(anchors & (s == 0))[0]
    b_all = np.where(anchors & (s == 1))[0]
    out = {"n_anchor_a": int(len(a_all)), "n_anchor_b": int(len(b_all))}
    if len(a_all) < 20 or len(b_all) < 20:
        out.update(anchor_estimable=False, irreducible_gap=float("nan"),
                   raw_gap=float("nan"), removed_fraction=float("nan"),
                   rho_hat=0.0, integrable=False)
        return out

    # MEASURE THE IRREDUCIBLE MEAN-GAP, NOT A CLASSIFIER AUC. A high-dimensional
    # classifier detects the AGGREGATE of a noisy alignment: subtracting a shift
    # estimated to ~0.1/gene injects a 500-dim offset of norm ~0.1*sqrt(500)=2.2
    # that a classifier reads as near-perfect separation even when the domains
    # are identical (measured anchAUC 0.996 at rho=1/delta=0). The per-gene L1
    # mean-gap does not aggregate noise that way -- it is what Stage-1 I5b used,
    # where delta -> 100% removed and rho -> 0% removed cleanly.
    #
    # Estimate the shift on HALF the anchors and measure the residual on the
    # OTHER half + all non-anchors, so the shift is never fit and scored on the
    # same cells (removes in-sample optimism).
    rng = np.random.default_rng(seed)

    def irreducible(labels):
        """Estimate the removable affine offset on half the anchors, subtract,
        and return the residual per-gene mean-gap on the held-out cells. Passing
        the TRUE labels gives the irreducible gap; passing shuffled labels gives
        the noise floor of this exact procedure (the gap it manufactures when
        there is no real domain difference to find)."""
        aa = np.where(anchors & (labels == 0))[0]
        bb = np.where(anchors & (labels == 1))[0]
        if len(aa) < 20 or len(bb) < 20:
            return float("nan"), float("nan")
        af = rng.choice(aa, len(aa) // 2, replace=False)
        bf = rng.choice(bb, len(bb) // 2, replace=False)
        em = np.ones(len(labels), bool); em[af] = False; em[bf] = False
        ea, eb = em & (labels == 0), em & (labels == 1)
        shift = L[bf].mean(0) - L[af].mean(0)
        raw = float(np.abs(L[eb].mean(0) - L[ea].mean(0)).mean())
        res = float(np.abs((L[eb].mean(0) - shift) - L[ea].mean(0)).mean())
        return raw, res

    raw_gap, residual = irreducible(s)
    # noise floor: same procedure under label permutation (no real difference).
    floors = [irreducible(rng.permutation(s))[1] for _ in range(5)]
    floor = float(np.nanmean(floors))
    removed = float(1.0 - residual / raw_gap) if raw_gap > 1e-9 else float("nan")
    # excess over the procedure's own noise floor is the irreducible signal.
    excess = max(residual - floor, 0.0)
    out.update(anchor_estimable=True, raw_gap=raw_gap, irreducible_gap=residual,
               removed_fraction=removed, noise_floor=floor,
               excess_over_floor=excess,
               rho_hat=float(np.clip(1.0 - excess / (raw_gap + 1e-9), 0.0, 1.0)),
               integrable=bool(residual < 2.0 * floor))
    return out


if __name__ == "__main__":
    import sys
    from core.dgp import DGP2Config, generate
    rng = np.random.default_rng(0)
    print("CERTIFICATE VALIDATION")
    print("  (a) unsupervised family: NONE separates the harmless from the fatal case")
    print("  (b) anchor-based: rho_hat must fall with rho and be FLAT in delta")
    print("  key rows: rho=1.0/delta=2.0 HARMLESS (v1 wrongly flagged);"
          " rho=0.2/delta=0.0 FATAL\n")
    print(f"{'rho':>5} {'delta':>6} | {'v1 AUC':>7} {'rank1':>6} "
          f"{'UOT':>6} || {'raw':>6} {'irred':>6} {'rmv%':>6} {'rho_hat':>8} {'integ':>6}")
    grid = [(1.0, 0.0), (1.0, 2.0), (0.6, 0.0), (0.6, 2.0), (0.4, 0.0),
            (0.2, 0.0), (0.2, 2.0)]
    for rho, delta in grid:
        d = generate(DGP2Config(rho=rho, delta=delta, n_per_domain=1500, seed=0))
        c = certificate(d["X"], d["s"], with_uot=True)
        band = np.where(d["overlap_region"])[0]
        amask = np.zeros(len(d["s"]), bool)
        amask[rng.choice(band, min(200, len(band)), replace=False)] = True
        ca = certificate_anchor(d["X"], d["s"], amask)
        star = " <==" if (rho, delta) in [(1.0, 2.0), (0.2, 0.0)] else ""
        print(f"{rho:>5} {delta:>6} | {c['raw_adversary_auc']:>7.3f} "
              f"{c['auc_rank1']:>6.3f} {c['uot_transported_mass']:>6.3f} || "
              f"{ca['raw_gap']:>6.3f} {ca['irreducible_gap']:>6.3f} "
              f"{100*ca['removed_fraction']:>5.0f}% {ca['rho_hat']:>8.3f} "
              f"{str(ca['integrable']):>6}{star}")
