"""
Metrics. Two families:

  (1) The two axes of the FEASIBILITY FRONTIER (scIB-style):
        bio_recovery  : how well the shared latent recovers the true factor t
        batch_removed : how well the domain has been removed from the latent
      The claim: below a critical overlap, no model achieves high on BOTH.

  (2) The A-PRIORI CERTIFICATE (NullState-style), computed from the DATA, before
      trusting any integration:
        raw_certificate : source-detecting adversary on raw x, swept over
                          capacity. Stability across capacity is what makes it a
                          *certificate* rather than an overfitting artifact.
        off_manifold    : how far in-vitro cells sit from the reference manifold.
      Together these estimate whether the biological supports overlap at all.

Positioning vs prior work: bio_recovery / batch_removed mirror scIB's
AvgBIO / AvgBATCH; iLISI is the standard batch-mixing metric; RBET/STACAS study
overcorrection. Our contribution is not a new metric -- it is showing these axes
become *jointly infeasible* past a support-overlap threshold, and that the
certificate predicts the threshold in advance. Cite: Luecken 2022 (scIB),
Korsunsky 2019 (LISI/Harmony), Andreatta 2024 (STACAS), Wang 2025 (RBET).
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import balanced_accuracy_score, r2_score


# --------------------------------------------------------------------------- #
#  axis 1a: biological recovery  (R^2 of true t from the shared latent)
# --------------------------------------------------------------------------- #
def bio_recovery(z: np.ndarray, t: np.ndarray, mask: np.ndarray | None = None,
                 k: int = 15, cv: int = 3) -> float:
    """kNN-regression R^2 predicting t (dim 0) from the shared latent z.
    If `mask` is given (e.g. the shared support band), restrict to those cells,
    which isolates recovery of the biology both domains actually share."""
    z = np.asarray(z); y = np.asarray(t)[:, 0]
    if mask is not None:
        z, y = z[mask], y[mask]
    if len(y) < k + 5 or np.std(y) < 1e-6:
        return float("nan")
    reg = KNeighborsRegressor(n_neighbors=k)
    pred = cross_val_predict(reg, z, y, cv=KFold(cv, shuffle=True, random_state=0))
    return float(r2_score(y, pred))


# --------------------------------------------------------------------------- #
#  axis 1b: domain removal  (low leakage == domain removed)
# --------------------------------------------------------------------------- #
def source_leakage(z: np.ndarray, s: np.ndarray, cv: int = 3) -> float:
    """Balanced accuracy of predicting domain s from the shared latent z.
    0.5 == domain fully removed (good); 1.0 == domain fully present (bad)."""
    z, s = np.asarray(z), np.asarray(s)
    clf = HistGradientBoostingClassifier(max_depth=4, random_state=0)
    pred = cross_val_predict(clf, z, s, cv=KFold(cv, shuffle=True, random_state=0))
    return float(balanced_accuracy_score(s, pred))


def batch_removed(z, s, cv: int = 3) -> float:
    """Rescale leakage to a [0,1] 'removed' score: 1 == fully removed."""
    la = source_leakage(z, s, cv)
    return float(np.clip(1.0 - 2.0 * (la - 0.5), 0.0, 1.0))


# --------------------------------------------------------------------------- #
#  axis 2 (mixing, scIB-style): integration LISI on the latent
# --------------------------------------------------------------------------- #
def ilisi(z: np.ndarray, s: np.ndarray, k: int = 30) -> float:
    """Integration LISI: mean effective number of domains in kNN neighborhoods,
    normalized to [0,1] (1 == perfectly mixed across the 2 domains)."""
    z, s = np.asarray(z), np.asarray(s)
    nn = NearestNeighbors(n_neighbors=min(k, len(z))).fit(z)
    _, idx = nn.kneighbors(z)
    neigh = s[idx]                                   # [n, k]
    p1 = neigh.mean(1)                               # fraction domain 1
    inv_simpson = 1.0 / (p1 ** 2 + (1 - p1) ** 2)    # in [1, 2] for 2 domains
    return float((inv_simpson.mean() - 1.0))         # -> [0, 1]


# --------------------------------------------------------------------------- #
#  the a-priori certificate (computed from RAW data, no model)
# --------------------------------------------------------------------------- #
def raw_certificate(X: np.ndarray, s: np.ndarray,
                    capacities=(2, 4, 8, None), cv: int = 3) -> dict:
    """Source-detecting adversary on raw (log1p) x, swept over capacity.
    Returns per-capacity balanced accuracy and the max. The KEY diagnostic is
    that this is high AND STABLE across capacity when supports are disjoint
    (NullState's '>=0.998 regardless of latent capacity'). High-and-stable ->
    domain is perfectly predictable -> support mismatch -> integration will
    extrapolate, not adjust."""
    Xn = np.log1p(X).astype(np.float32)
    out = {}
    for cap in capacities:
        clf = HistGradientBoostingClassifier(
            max_depth=(cap if cap else None), max_iter=200, random_state=0)
        pred = cross_val_predict(clf, Xn, s, cv=KFold(cv, shuffle=True, random_state=0))
        out[f"cap_{cap}"] = float(balanced_accuracy_score(s, pred))
    vals = np.array(list(out.values()))
    out["max"] = float(vals.max())
    out["spread"] = float(vals.max() - vals.min())   # small spread == stable == certificate
    return out


def off_manifold(z: np.ndarray, s: np.ndarray, k: int = 15) -> float:
    """NullState-style: median distance from each in-vitro (s=1) cell to its
    k-th nearest reference (s=0) cell in latent space, normalized by the
    reference's own k-NN scale. High == in-vitro sits off the reference manifold
    == biological supports do not overlap."""
    z = np.asarray(z)
    ref, tgt = z[s == 0], z[s == 1]
    if len(ref) < k + 1 or len(tgt) == 0:
        return float("nan")
    nn_ref = NearestNeighbors(n_neighbors=k).fit(ref)
    d_self, _ = nn_ref.kneighbors(ref)               # reference internal scale
    scale = np.median(d_self[:, -1]) + 1e-8
    d_tgt, _ = nn_ref.kneighbors(tgt)
    return float(np.median(d_tgt[:, -1]) / scale)


# --------------------------------------------------------------------------- #
#  convenience: one full row of metrics for a fitted embedding
# --------------------------------------------------------------------------- #
def cross_domain_transfer_overlap(z, t, s, mask, k: int = 15) -> dict:
    """Cross-domain transfer RESTRICTED TO THE OVERLAP BAND. THIS is the valid metric.

    WHY THE RESTRICTION IS REQUIRED (validated by an oracle control, do not remove):
    the unrestricted version is confounded by extrapolation. With overlap rho the two
    domains' t-ranges only partly coincide (e.g. at rho=0.05 domain A spans [0.00,1.00]
    while domain B spans [0.95,1.95]), so a kNN map fit on A is asked to predict t values
    it has never seen. Running the metric on an ORACLE embedding (z := the TRUE t, i.e. a
    perfectly identified shared axis by construction) still produces a collapse:

        rho      1.00    0.85    0.60    0.40    0.25    0.05
        oracle  1.000   0.986   0.735   0.112  -0.726  -2.483     <- unrestricted (INVALID)
        oracle  1.000   1.000   1.000   1.000   1.000   0.992     <- overlap-restricted (VALID)

    The unrestricted curve therefore measures the DGP's range shift, not identifiability.
    Restricted to the overlap band the oracle stays pinned at ~1.0 for every rho, so any
    collapse the LEARNED embedding shows against that flat baseline is attributable to
    non-identifiability of the shared axis rather than to extrapolation.

    CAVEAT (report it): the overlap band shrinks as rho -> 0, so this metric loses
    statistical power exactly where the effect is expected to be largest. Always report
    n_overlap alongside it, and treat small-n points as underpowered rather than null.
    """
    z = np.asarray(z); y = np.asarray(t)[:, 0]
    s = np.asarray(s).ravel(); mask = np.asarray(mask).astype(bool)
    out, n_ovl = [], int(mask.sum())
    for a, b in ((0, 1), (1, 0)):
        ia, ib = (s == a) & mask, (s == b) & mask
        if ia.sum() < k + 5 or ib.sum() < k + 5 or np.std(y[ib]) < 1e-9:
            out.append(float("nan")); continue
        kn = KNeighborsRegressor(n_neighbors=min(k, int(ia.sum()) - 1)).fit(z[ia], y[ia])
        out.append(float(r2_score(y[ib], kn.predict(z[ib]))))
    vals = [v for v in out if np.isfinite(v)]
    return {"xdom_ovl_a2b": out[0], "xdom_ovl_b2a": out[1],
            "xdom_ovl_mean": float(np.mean(vals)) if vals else float("nan"),
            "n_overlap": n_ovl}


def cross_domain_transfer(z, t, s, k: int = 15) -> dict:
    """THE identifiability metric: is the shared factor recovered on a COMMON axis?

    Fit a kNN map z -> t using ONLY domain A, then score it on domain B (and vice
    versa). If the two domains' latent axes are aligned, the map transfers (R^2 -> 1).
    If each domain received its own arbitrary axis -- which is exactly what
    non-identifiability means -- the map does not transfer and R^2 collapses, going
    negative once predictions are worse than the target mean.

    This is deliberately NOT `batch_removed`: with overlap fraction rho, a source
    adversary scores ~1 - rho/2 by construction, so `batch_removed ~= rho` is an
    algebraic identity of the design rather than a property of the estimator. Within
    -domain recovery is likewise uninformative here (it saturates ~0.999 at every
    rho). Cross-domain transfer is the quantity that actually distinguishes an
    identified shared axis from two unrelated ones.
    """
    z = np.asarray(z); y = np.asarray(t)[:, 0]; s = np.asarray(s).ravel()
    out = {}
    for a, b, name in ((0, 1, "xdom_a2b"), (1, 0, "xdom_b2a")):
        ia, ib = (s == a), (s == b)
        if ia.sum() < k + 5 or ib.sum() < k + 5 or np.std(y[ib]) < 1e-6:
            out[name] = float("nan"); continue
        kn = KNeighborsRegressor(n_neighbors=min(k, int(ia.sum()) - 1)).fit(z[ia], y[ia])
        out[name] = float(r2_score(y[ib], kn.predict(z[ib])))
    vals = [v for v in out.values() if np.isfinite(v)]
    # symmetric summary; clipped at 0 so "no transfer" and "catastrophically wrong"
    # both read as 0 on the headline axis (the raw signed values are kept too).
    out["xdom_mean"] = float(np.mean(vals)) if vals else float("nan")
    out["xdom_min"] = float(np.min(vals)) if vals else float("nan")
    out["xdom_score"] = float(np.clip(out["xdom_mean"], 0.0, 1.0)) if vals else float("nan")
    return out


def evaluate_embedding(z, t, s, mask=None) -> dict:
    return dict(
        bio_recovery=bio_recovery(z, t, mask=None),        # overall (always defined)
        bio_recovery_shared=bio_recovery(z, t, mask=mask),  # on the shared band (NaN if empty)
        batch_removed=batch_removed(z, s),
        source_leakage=source_leakage(z, s),
        ilisi=ilisi(z, s),
        off_manifold_latent=off_manifold(z, s),
        # identifiability of the shared axis.
        # xdom_* (unrestricted) is retained for diagnosis ONLY -- it is confounded by
        # extrapolation (see cross_domain_transfer_overlap). Use xdom_ovl_mean as the
        # headline; compare it against the oracle baseline, which is flat at ~1.0.
        **cross_domain_transfer(z, t, s),
        **(cross_domain_transfer_overlap(z, t, s, mask) if mask is not None else {}),
    )
