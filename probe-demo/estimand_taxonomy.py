"""
THE THESIS, MADE CONCRETE: every recovery estimand has an invariance class, and a
result is an artifact whenever the failure mode lives inside that class.

This project produced two independent demonstrations of the same trap:

  v1: kNN-R^2 / kNN transfer are EXACTLY invariant to rotation (they depend on the
      embedding only through its neighbour graph, which an orthogonal map
      preserves). Rotation is precisely the non-identifiability of an isotropic
      prior -- so the metric could not see the phenomenon, and the "critical
      overlap" it reported was the zero-crossing of its own extrapolation
      geometry (1 - 4^(-1/3)).

  v3: the MATCHED-INFORMATION ORACLE GAP -- which we built deliberately, argued
      for, and gated -- is structurally invariant to INFORMATION LOSS, because it
      matches the oracle to the learned embedding's information (CCA) before
      differencing. Alignment objectives destroy recovery by folding the shifted
      axis, which IS information loss, so the gap reads ~0 while raw CCA collapses
      0.99 -> 0.70. Our own metric has a blind spot that conceals the failure
      mode practitioners care about most.

The second demonstration is the more persuasive one: the blind spot survived a
correctness argument and a passing gate. So the general claim is not "kNN metrics
are bad" but:

    CHOOSE THE ESTIMAND BY THE FAILURE MODE. An estimand whose invariance class
    contains the failure mode reports "no effect" regardless of the truth.

This script measures the full table: each estimand scored on embeddings corrupted
in exactly one way, so each column is an invariance class read off directly. No
training and no DGP -- pure metric geometry, so the table is a property of the
estimands themselves.
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

from metrics2 import mcc, cca_score, dci

N = 4000
SEED = 0


def _rot(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


# --------------------------------------------------------------------------- #
#  corruptions: each isolates ONE failure mode (or one nuisance)
# --------------------------------------------------------------------------- #
CORRUPTIONS = {
    # -- nuisances: a good estimand SHOULD be invariant to these --------------
    "identity":             lambda t, r: t.copy(),
    "permute axes":         lambda t, r: t[:, ::-1].copy(),
    "per-axis rescale":     lambda t, r: t * np.array([3.0, 0.2]),
    # -- ENTANGLEMENT: axis mixing. the v1/disentanglement failure mode -------
    "rotate 45deg":         lambda t, r: t @ _rot(np.pi / 4),
    "shear (linear mix)":   lambda t, r: t @ np.array([[1.0, 0.8], [0.0, 1.0]]),
    # -- nonlinear reparametrisation (monotone, per-axis) ---------------------
    "monotone nonlinear":   lambda t, r: np.c_[np.sign(t[:, 0]) * np.abs(t[:, 0]) ** 3,
                                               t[:, 1]],
    # -- INFORMATION LOSS: the v3 / alignment failure mode --------------------
    "isotropic noise":      lambda t, r: t + 0.6 * r.normal(size=t.shape),
    "fold axis 0":          lambda t, r: np.c_[np.abs(t[:, 0]), t[:, 1]],
    "collapse axis 0":      lambda t, r: np.c_[np.zeros(len(t)), t[:, 1]],
}

# what each corruption IS, for the verdict column
KIND = {
    "identity": "-", "permute axes": "nuisance", "per-axis rescale": "nuisance",
    "rotate 45deg": "ENTANGLEMENT", "shear (linear mix)": "ENTANGLEMENT",
    "monotone nonlinear": "reparametrisation",
    "isotropic noise": "INFORMATION LOSS", "fold axis 0": "INFORMATION LOSS",
    "collapse axis 0": "INFORMATION LOSS",
}


# --------------------------------------------------------------------------- #
#  estimands
# --------------------------------------------------------------------------- #
def knn_r2(t, z, k=15, cv=3, seed=SEED):
    """kNN-regression R^2 predicting the true factor from the embedding.
    Invariance class: every map preserving the ranking of pairwise distances --
    all isometries (incl. the orthogonal group O(d)) and global rescalings."""
    pred = cross_val_predict(KNeighborsRegressor(n_neighbors=k), z, t[:, 0],
                             cv=KFold(cv, shuffle=True, random_state=seed))
    return float(r2_score(t[:, 0], pred))


def matched_oracle_gap(t, z, seed=SEED):
    """MCC(oracle matched to z's information) - MCC(z).
    Invariance class: INFORMATION LOSS -- by construction, since the oracle is
    noised until its CCA equals z's before differencing."""
    target = cca_score(t, z, seed=seed)["cca_mean"]
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
    rng = np.random.default_rng(seed + 991)
    zz = t + 0.5 * (lo + hi) * sd * rng.normal(size=t.shape)
    return float(mcc(t, zz, seed=seed)["mcc_pearson"] - mcc(t, z, seed=seed)["mcc_pearson"])


def score_all(t, z, with_dci=True, seed=SEED):
    out = {
        "kNN-R2": knn_r2(t, z, seed=seed),
        "CCA": cca_score(t, z, seed=seed)["cca_mean"],
        "MCC-P": mcc(t, z, "pearson", seed=seed)["mcc_pearson"],
        "MCC-S": mcc(t, z, "spearman", seed=seed)["mcc_spearman"],
        "oracle gap": matched_oracle_gap(t, z, seed=seed),
    }
    if with_dci:
        out["DCI-D"] = dci(t, z, seed=seed)["dci_disentanglement"]
    return out


def verify_gap_identity(t, corruptions):
    """The matched-oracle gap is NOT an independent estimand: it equals CCA - MCC.

    Why: the oracle family is isotropic noise added to the TRUE latent, so its
    canonical directions are the coordinate axes and every oracle satisfies
    MCC(oracle) = CCA(oracle). Matching the oracle's CCA to the learned
    embedding's therefore forces MCC(oracle) = CCA(z), and

        gap = MCC(oracle) - MCC(z) = CCA(z) - MCC(z).

    This is a sharper statement than "blind to information loss" -- it EXPLAINS
    the blindness: information loss lowers CCA and MCC together, so the
    difference cancels. It also says what the gap does measure: the excess of
    linear-subspace recovery over axis-wise recovery, i.e. 'right subspace,
    wrong axes' = entanglement.
    """
    print("\n" + "=" * 92)
    print("IDENTITY: the matched-oracle gap == CCA - MCC (it is not independent)")
    print("=" * 92)
    print(f"{'corruption':>20} {'gap':>9} {'CCA-MCC':>9} {'|diff|':>8}")
    worst = 0.0
    for name, fn in corruptions.items():
        z = np.asarray(fn(t, np.random.default_rng(SEED + 7)), float)
        g = matched_oracle_gap(t, z)
        d = cca_score(t, z)["cca_mean"] - mcc(t, z)["mcc_pearson"]
        worst = max(worst, abs(g - d))
        print(f"{name:>20} {g:>9.4f} {d:>9.4f} {abs(g - d):>8.4f}")
    print(f"  max deviation = {worst:.4f}  ->  "
          f"{'IDENTITY HOLDS' if worst < 0.02 else 'NOT an identity'}")
    return worst


def main():
    rng = np.random.default_rng(SEED)
    t = rng.normal(size=(N, 2))

    cols = ["kNN-R2", "CCA", "MCC-P", "MCC-S", "DCI-D", "oracle gap"]
    print("=" * 92)
    print("ESTIMAND x FAILURE MODE.  Each row corrupts the TRUE latent in exactly")
    print("one way; each column is one estimand. A cell near the identity value")
    print("means that estimand is BLIND to that corruption.")
    print("=" * 92)
    header = f"{'corruption':>20} {'kind':>18} " + " ".join(f"{c:>10}" for c in cols)
    print(header)
    print("-" * len(header))

    rows = {}
    for name, fn in CORRUPTIONS.items():
        z = np.asarray(fn(t, np.random.default_rng(SEED + 7)), float)
        s = score_all(t, z)
        rows[name] = s
        print(f"{name:>20} {KIND[name]:>18} " + " ".join(f"{s[c]:>10.3f}" for c in cols))

    ident = rows["identity"]
    print("\n" + "=" * 92)
    print("BLIND SPOTS (estimand within 0.05 of its identity value on a REAL failure)")
    print("=" * 92)
    for name in CORRUPTIONS:
        if KIND[name] in ("-", "nuisance"):
            continue
        blind = [c for c in cols
                 if abs(rows[name][c] - ident[c]) < 0.05]
        if blind:
            print(f"  {name:>20} [{KIND[name]}] -> BLIND: {', '.join(blind)}")

    print("\n" + "=" * 92)
    print("INVARIANCE CLASSES (what each estimand provably cannot see)")
    print("=" * 92)
    verify_gap_identity(t, CORRUPTIONS)

    print("\n" + "=" * 92)
    print("INVARIANCE CLASSES (what each estimand provably cannot see)")
    print("=" * 92)
    print("""  kNN-R2 / kNN transfer : exactly invariant to any map preserving the k-NN
                          SETS. Distance-rank preservation is sufficient but not
                          necessary; the class contains the similarity group
                          Sim(d) = translations x (O(d) x R_{>0}). CONTAINS
                          rotation = the isotropic-prior non-identifiability.
                          Blind to ENTANGLEMENT.
  CCA                   : any affine map whose linear part is INJECTIVE on the
                          latent (z -> zA + b, rank(A) = d) -- strictly larger
                          than GL(d): also injective maps into higher dimension,
                          appended noise coordinates, duplicated coordinates and
                          translations. Blind to LINEAR entanglement only --
                          information-preserving NONLINEAR axis mixing IS visible.
                          SEES information loss. (The 'weak'/~_A notion.)
  MCC-Pearson           : permutation + per-axis affine. SEES rotation, shear,
                          information loss; partially sees nonlinear reparam.
  MCC-Spearman          : permutation + per-axis MONOTONE reparam. SEES rotation
                          and information loss; blind to monotone reparam.
  matched-oracle gap    : NOT an independent estimand -- it equals CCA - MCC
                          exactly (verified above). Hence blind to information
                          loss, which lowers both terms and cancels; what it
                          measures is the excess of linear-subspace recovery over
                          axis-wise recovery, i.e. ENTANGLEMENT.

  => Pick the estimand whose invariance class EXCLUDES the failure mode under
     test. Rotation-type failure -> MCC/DCI (not kNN-R2, not CCA).
     Information-loss-type failure -> raw CCA/MCC (not the oracle gap).
     Reporting the wrong one returns 'no effect' regardless of the truth.""")


if __name__ == "__main__":
    main()
