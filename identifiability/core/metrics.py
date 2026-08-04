"""
STAGE-2 METRICS: rotation-sensitive latent recovery, plus the oracle ceiling.

WHY v1's METRICS WERE VOID. The v1 probe scored recovery with kNN-regression
R^2 (`bio_recovery`) and a kNN cross-domain transfer. Both are (approximately)
ROTATION-INVARIANT: a kNN regressor built on a rotated/entangled latent
predicts just as well as on a disentangled one, because it only uses the
neighbourhood graph. So the metric could not see entanglement at all -- which
is why every score turned out to be a closed-form function of rho and the
learned model sat exactly on that analytic ceiling. Worse, the unrestricted
cross-domain metric measured kNN EXTRAPOLATION: its "critical rho* = 0.370" was
exactly 1 - 4^(-1/3), a constant of kNN geometry.

What replaces them:

  MCC   -- mean correlation coefficient with HUNGARIAN matching. Sensitive to
           rotation and to permutation. Pearson variant tests linear (~_A)
           identifiability; Spearman variant is blind to component-wise
           nonlinear reparameterisation and so tests the ~_P notion.
           The matching is solved on a TRAIN split and scored on a HELD-OUT
           split, so a lucky assignment cannot inflate the score.

  CCA   -- mean canonical correlation: the WEAK (~_A, up to invertible linear
           map) notion. Reporting MCC and CCA together prevents the classic
           misattribution in which a model that is only linearly identifiable
           by construction (iVAE Prop. 1: Gaussian, fixed variance, k=1) is
           scored against a permutation criterion it can never meet, and the
           shortfall is then blamed on support mismatch.

  DCI   -- Disentanglement / Completeness / Informativeness (Eastwood &
           Williams 2018) from a gradient-boosted importance matrix.

  ORACLE -- every metric is also computed on z := the TRUE latent. This is the
           metric CEILING. The primary estimand of the whole probe is the
           LEARNED-MINUS-ORACLE gap, never the raw learned score: only the gap
           separates "the model failed" from "the metric cannot go higher here".
           Stage-2 gate: the oracle must be ~1.0 and FLAT in rho and delta. If
           the oracle itself bends with rho, the metric is artifact-laden and
           must be fixed before any model is compared.

  MATCHED-NOISE ORACLE -- audit finding B4 killed run_decisive.py's noiseless
           oracle: z := t returns ~1.0 at every rho by construction and is blind
           to variance-normalisation artifacts. `oracle_scores(..., noise=sigma)`
           adds calibrated isotropic noise so the control sits at a realistic
           operating point rather than at a degenerate one.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score


# --------------------------------------------------------------------------- #
#  MCC  (rotation- and permutation-sensitive)
# --------------------------------------------------------------------------- #
def _corr_matrix(a: np.ndarray, b: np.ndarray, kind: str) -> np.ndarray:
    """|corr| between every column of a and every column of b."""
    if kind == "spearman":
        a = np.apply_along_axis(lambda v: np.argsort(np.argsort(v)), 0, a).astype(float)
        b = np.apply_along_axis(lambda v: np.argsort(np.argsort(v)), 0, b).astype(float)
    a = (a - a.mean(0)) / (a.std(0) + 1e-12)
    b = (b - b.mean(0)) / (b.std(0) + 1e-12)
    return np.abs(a.T @ b) / len(a)


def mcc(t_true: np.ndarray, z: np.ndarray, kind: str = "pearson",
        train_frac: float = 0.5, seed: int = 0) -> dict:
    """Mean correlation coefficient under the optimal one-to-one matching.

    The assignment is solved on a train split and the reported score is
    computed on the held-out split under that FIXED assignment (out-of-sample
    MCC, as in ICE-BeeM and the CRL literature). Rectangular assignment is
    supported, so latent dimensionalities need not agree; the score averages
    over min(dim_true, dim_z) matched pairs.
    """
    t_true, z = np.asarray(t_true, float), np.asarray(z, float)
    n = len(t_true)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(train_frac * n)
    tr, te = idx[:cut], idx[cut:]

    C_tr = _corr_matrix(t_true[tr], z[tr], kind)
    rows, cols = linear_sum_assignment(-C_tr)              # maximise |corr|
    C_te = _corr_matrix(t_true[te], z[te], kind)
    matched = C_te[rows, cols]
    return {
        f"mcc_{kind}": float(np.mean(matched)),
        f"mcc_{kind}_min": float(np.min(matched)),
        f"mcc_{kind}_per_axis": [float(x) for x in matched],
        f"mcc_{kind}_assignment": [int(c) for c in cols],
        f"mcc_{kind}_insample": float(np.mean(C_tr[rows, cols])),
    }


# --------------------------------------------------------------------------- #
#  CCA  (weak / ~_A identifiability)
# --------------------------------------------------------------------------- #
def cca_score(t_true: np.ndarray, z: np.ndarray, train_frac: float = 0.5,
              seed: int = 0) -> dict:
    """Mean canonical correlation, fit on train and evaluated out of sample.

    This is the WEAK notion: identifiability up to an invertible linear map.
    A model can score ~1 here while scoring poorly on MCC -- that combination
    means 'recovered the right subspace, wrong axes', which is exactly the
    distinction v1's rotation-invariant metrics could not make.
    """
    t_true, z = np.asarray(t_true, float), np.asarray(z, float)
    k = min(t_true.shape[1], z.shape[1])
    n = len(t_true)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(train_frac * n)
    tr, te = idx[:cut], idx[cut:]
    try:
        model = CCA(n_components=k, max_iter=1000).fit(t_true[tr], z[tr])
        U, V = model.transform(t_true[te], z[te])
    except Exception:
        return {"cca_mean": float("nan"), "cca_min": float("nan")}
    cors = []
    for j in range(k):
        u, v = U[:, j], V[:, j]
        sd = u.std() * v.std()
        cors.append(abs(float(np.mean((u - u.mean()) * (v - v.mean())) / (sd + 1e-12))))
    return {"cca_mean": float(np.mean(cors)), "cca_min": float(np.min(cors)),
            "cca_per_component": [float(c) for c in cors]}


# --------------------------------------------------------------------------- #
#  DCI  (Eastwood & Williams 2018)
# --------------------------------------------------------------------------- #
def dci(t_true: np.ndarray, z: np.ndarray, cv: int = 3, seed: int = 0) -> dict:
    """Disentanglement / Completeness / Informativeness from a GBT importance
    matrix R[i, j] = importance of latent i for factor j.

    D_i = 1 - H_{n_factors}(P_{i.}) weighted by latent i's total importance;
    C_j = 1 - H_{n_latents}(P_{.j});  I = mean CV R^2 of the factor predictors.
    """
    t_true, z = np.asarray(t_true, float), np.asarray(z, float)
    nz, nf = z.shape[1], t_true.shape[1]
    R = np.zeros((nz, nf))
    r2s = []
    for j in range(nf):
        est = HistGradientBoostingRegressor(max_depth=4, max_iter=200,
                                            random_state=seed)
        pred = cross_val_predict(est, z, t_true[:, j],
                                 cv=KFold(cv, shuffle=True, random_state=seed))
        r2s.append(r2_score(t_true[:, j], pred))
        # permutation importance (HistGBR exposes no feature_importances_)
        est.fit(z, t_true[:, j])
        base = r2_score(t_true[:, j], est.predict(z))
        rng = np.random.default_rng(seed)
        for i in range(nz):
            zp = z.copy()
            zp[:, i] = rng.permutation(zp[:, i])
            R[i, j] = max(base - r2_score(t_true[:, j], est.predict(zp)), 0.0)

    def _ent(p, base):
        p = p / (p.sum() + 1e-12)
        p = p[p > 0]
        return float(-(p * (np.log(p) / np.log(base))).sum()) if base > 1 else 0.0

    D_i = np.array([1.0 - _ent(R[i, :], nf) for i in range(nz)])
    w = R.sum(1) / (R.sum() + 1e-12)
    C_j = np.array([1.0 - _ent(R[:, j], nz) for j in range(nf)])
    return {"dci_disentanglement": float((w * D_i).sum()),
            "dci_completeness": float(C_j.mean()),
            "dci_informativeness": float(np.mean(r2s)),
            "dci_importance": R.tolist()}


# --------------------------------------------------------------------------- #
#  the full scorecard + the oracle ceiling
# --------------------------------------------------------------------------- #
def score_embedding(t_true, z, mask=None, with_dci: bool = True,
                    seed: int = 0) -> dict:
    """All identifiability scores for one embedding.

    `mask` (the shared overlap band) yields the band-restricted MCC. Report it
    ALONGSIDE the full-sample score and ALWAYS with n_band: the band shrinks as
    rho -> 0, so band-restricted numbers lose power exactly where the effect is
    expected to be largest. Small-n points are UNDERPOWERED, not null.
    """
    t_true, z = np.asarray(t_true, float), np.asarray(z, float)
    out = {}
    out.update(mcc(t_true, z, "pearson", seed=seed))
    out.update(mcc(t_true, z, "spearman", seed=seed))
    out.update(cca_score(t_true, z, seed=seed))
    if with_dci:
        out.update(dci(t_true, z, seed=seed))
    if mask is not None:
        m = np.asarray(mask, bool)
        out["n_band"] = int(m.sum())
        if m.sum() > 200:
            b = mcc(t_true[m], z[m], "pearson", seed=seed)
            out["mcc_band"] = b["mcc_pearson"]
            out["cca_band"] = cca_score(t_true[m], z[m], seed=seed)["cca_mean"]
        else:
            out["mcc_band"] = float("nan")
            out["cca_band"] = float("nan")
    return out


def oracle_scores(t_true, mask=None, noise: float = 0.0, seed: int = 0,
                  with_dci: bool = True) -> dict:
    """The metric CEILING: score the TRUE latent under the same pipeline.

    noise > 0 gives the MATCHED-NOISE oracle demanded by audit finding B4. A
    noiseless oracle (z := t) is degenerate -- it returns ~1.0 at every rho by
    construction and therefore cannot reveal whether the metric is distorted at
    a realistic operating point. Set `noise` to the residual scale of the
    learned embedding (see `calibrate_oracle_noise`) so the control sits where
    the learned model actually sits.
    """
    t_true = np.asarray(t_true, float)
    z = t_true.copy()
    if noise > 0:
        rng = np.random.default_rng(seed + 991)
        z = z + noise * z.std(0, keepdims=True) * rng.normal(size=z.shape)
    out = score_embedding(t_true, z, mask=mask, with_dci=with_dci, seed=seed)
    return {f"oracle_{k}": v for k, v in out.items()}


def calibrate_oracle_noise(t_true, z_learned, seed: int = 0) -> float:
    """Noise level sigma such that a noisy oracle attains the SAME informative-
    ness as the learned embedding. This puts the control at the learned model's
    operating point, so any remaining MCC gap is attributable to axis mixing
    (entanglement) rather than to a difference in raw signal quality.
    """
    t_true = np.asarray(t_true, float)
    target = dci(t_true, np.asarray(z_learned, float), seed=seed)["dci_informativeness"]
    lo, hi = 0.0, 5.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        rng = np.random.default_rng(seed + 991)
        zz = t_true + mid * t_true.std(0, keepdims=True) * rng.normal(size=t_true.shape)
        got = dci(t_true, zz, seed=seed)["dci_informativeness"]
        if got > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gap(learned: dict, oracle: dict, keys=("mcc_pearson", "mcc_spearman",
                                           "cca_mean", "dci_disentanglement")) -> dict:
    """THE primary estimand: oracle minus learned, per metric."""
    out = {}
    for k in keys:
        o, l = oracle.get(f"oracle_{k}"), learned.get(k)
        if o is not None and l is not None:
            out[f"gap_{k}"] = float(o - l)
    return out


if __name__ == "__main__":
    # self-test: the metrics must SEE a rotation. v1's kNN metrics did not.
    rng = np.random.default_rng(0)
    t = rng.normal(size=(4000, 2))
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for name, z in [("identity", t.copy()),
                    ("permuted", t[:, ::-1].copy()),
                    ("rotated 45deg", t @ R),
                    ("scaled", t * np.array([3.0, 0.2])),
                    ("noisy 0.5", t + 0.5 * rng.normal(size=t.shape))]:
        s = score_embedding(t, z, with_dci=False)
        print(f"{name:>15}: MCC(p) {s['mcc_pearson']:.3f}  "
              f"MCC(s) {s['mcc_spearman']:.3f}  CCA {s['cca_mean']:.3f}")


def entanglement_gap(t_true, z, seed: int = 0) -> float:
    """THE estimand for axis mixing: CCA - MCC.

    Interpretation is immediate: CCA is recovery up to an invertible linear map
    (was the right SUBSPACE found?), MCC is recovery up to permutation and
    per-axis affine (were the right AXES found?). Their difference is exactly
    "subspace recovered, axes scrambled".

    This replaces the matched-noise-oracle construction, which was shown to be
    identically equal to this quantity (see estimand_taxonomy.verify_gap_identity):
    the oracle family is isotropic noise on the true latent, hence axis-factorised,
    hence MCC(oracle) = CCA(oracle); matching information forces
    MCC(oracle) = CCA(z) and the oracle cancels out. Reporting CCA - MCC directly
    removes a construction a reader would otherwise have to audit, and makes the
    zero set self-evident: {MCC = CCA}, the AXIS-FACTORISED maps.

    BASIS-DEPENDENCE (state this whenever the estimand is used). CCA is basis-free;
    MCC is not, because Hungarian matching asks whether recovered coordinate i
    tracks true coordinate j. The gap therefore inherits basis-dependence entirely
    through MCC, and its blind set -- the axis-factorised maps -- is defined
    RELATIVE TO THE GROUND-TRUTH BASIS one chose. Rotate that basis and the blind
    region moves. In synthetic work the basis is a modelling choice; on real data
    there is no privileged basis for "the true latent", so the estimand's blind
    region is positioned by a choice with no observable counterpart.
    """
    c = cca_score(t_true, z, seed=seed)["cca_mean"]
    m = mcc(t_true, z, "pearson", seed=seed)["mcc_pearson"]
    return float(c - m)
