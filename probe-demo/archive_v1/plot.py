"""
The one figure (the whole point). A stranger should read the finding in 10s:
recovery falls off a cliff at a critical overlap, the cheap certificate calls
the cliff in advance, and more data does not save you.

  Panel A (headline): bio_recovery vs rho, one line per method, with the raw
                      certificate overlaid on a twin axis, and vertical lines at
                      (i) the empirical collapse and (ii) [TODO] Uhler's
                      theoretical identifiability threshold.
  Panel B          : rho x delta heatmap of bio_recovery (collapses along rho only).
  Panel C          : bio_recovery vs n at a below-threshold rho vs a just-above rho.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _agg(df, x, y, by=None):
    g = df.groupby(([by] if by else []) + [x])[y].agg(["mean", "std"]).reset_index()
    return g


def panel_a(df: pd.DataFrame, ax=None, uhler_threshold: float | None = None):
    ax = ax or plt.gca()
    for method, sub in df.groupby("method"):
        a = _agg(sub, "rho", "bio_recovery")
        ax.plot(a["rho"], a["mean"], marker="o", label=method)
        ax.fill_between(a["rho"], a["mean"] - a["std"], a["mean"] + a["std"], alpha=0.15)
    ax.set_xlabel("biological support overlap  \u03c1")
    ax.set_ylabel("biological recovery  (R\u00b2 of true t)")
    ax.set_xlim(1.02, -0.02)                       # high overlap on the left
    ax.axhline(0, color="k", lw=0.6, ls=":")

    # certificate on a twin axis (model-free)
    if "cert_max" in df:
        ax2 = ax.twinx()
        c = _agg(df.drop_duplicates(["rho", "seed"]), "rho", "cert_max")
        ax2.plot(c["rho"], c["mean"], color="crimson", ls="--", marker="s",
                 label="raw certificate (adversary)")
        ax2.set_ylabel("source detectability (balanced acc.)", color="crimson")
        ax2.set_ylim(0.45, 1.02)
        ax2.tick_params(axis="y", labelcolor="crimson")

    # empirical collapse point: largest rho where mean recovery drops below half
    coll = _empirical_threshold(df)
    if coll is not None:
        ax.axvline(coll, color="gray", ls="-", lw=1.2,
                   label=f"empirical collapse \u03c1*\u2248{coll:.2f}")
    if uhler_threshold is not None:                # TODO: fill from arXiv 2410.23620
        ax.axvline(uhler_threshold, color="green", ls="-.", lw=1.2,
                   label=f"Uhler condition breaks \u2248{uhler_threshold:.2f}")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("A. Recovery collapses at a critical overlap; the certificate calls it in advance")
    return ax


def _empirical_threshold(df, frac=0.5):
    ref = df[df["method"] == "conditional"] if "conditional" in df["method"].values else df
    a = _agg(ref, "rho", "bio_recovery").sort_values("rho")
    hi = a["mean"].max()
    below = a[a["mean"] < frac * hi]
    return float(below["rho"].max()) if len(below) else None


def panel_b(df: pd.DataFrame, ax=None, value="bio_recovery"):
    ax = ax or plt.gca()
    piv = df.pivot_table(index="delta", columns="rho", values=value, aggfunc="mean")
    piv = piv.sort_index(ascending=False)[sorted(piv.columns, reverse=True)]
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:.1f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels([f"{r:.1f}" for r in piv.index])
    ax.set_xlabel("support overlap  \u03c1"); ax.set_ylabel("batch offset  \u03b4")
    ax.set_title("B. Recovery collapses along \u03c1, not \u03b4")
    plt.colorbar(im, ax=ax, label=value)
    return ax


def panel_c(df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    for rho, sub in df.groupby("rho"):
        a = _agg(sub, "n", "bio_recovery")
        ax.plot(a["n"], a["mean"], marker="o",
                label=f"\u03c1={rho:.2f} ({'below' if rho < 0.3 else 'above'} threshold)")
        ax.fill_between(a["n"], a["mean"] - a["std"], a["mean"] + a["std"], alpha=0.15)
    ax.set_xscale("log"); ax.set_xlabel("cells per domain  n")
    ax.set_ylabel("biological recovery  (R\u00b2)")
    ax.set_title("C. Structural vs practical: below threshold, more data does not help")
    ax.legend(fontsize=8)
    return ax


def panel_frontier(df_frontier: pd.DataFrame, ax=None):
    """The feasibility frontier: (domain removed) vs (biology recovered) as the
    correction strength sweeps, one curve per rho. Above rho* the curve reaches
    the top-right; below rho* the top-right corner is unreachable."""
    ax = ax or plt.gca()
    for rho, sub in df_frontier.groupby("rho"):
        a = (sub.groupby("adv_lambda")[["batch_removed", "bio_recovery"]]
             .mean().reset_index().sort_values("adv_lambda"))
        ax.plot(a["batch_removed"], a["bio_recovery"], marker="o",
                label=f"\u03c1={rho:.2f}")
    ax.scatter([1], [1], marker="*", s=200, color="gold", edgecolor="k",
               zorder=5, label="ideal (both = 1)")
    ax.set_xlabel("domain removed  (1 = fully removed)")
    ax.set_ylabel("biological recovery  (R\u00b2)")
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    ax.set_title("Feasibility frontier: top-right reachable only above \u03c1*")
    ax.legend(fontsize=8)
    return ax


def panel_joint_vs_rho(df_frontier: pd.DataFrame, ax=None):
    """Scalar headline: best achievable balance = max_lambda min(bio, batch),
    plotted vs rho. Collapses at rho*."""
    ax = ax or plt.gca()
    df = df_frontier.copy()
    df["joint"] = df[["bio_recovery", "batch_removed"]].min(axis=1)
    best = (df.groupby(["rho", "seed"])["joint"].max().reset_index()
            .groupby("rho")["joint"].agg(["mean", "std"]).reset_index())
    ax.plot(best["rho"], best["mean"], marker="o", color="navy")
    ax.fill_between(best["rho"], best["mean"] - best["std"],
                    best["mean"] + best["std"], alpha=0.2)
    ax.set_xlim(1.02, -0.02); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("biological support overlap  \u03c1")
    ax.set_ylabel("best achievable  min(recovery, removal)")
    ax.set_title("Joint feasibility collapses at a critical overlap")
    return ax


def make_figure(df_rho, df_grid=None, df_n=None, out="figure.png",
                uhler_threshold=None):
    ncols = 1 + (df_grid is not None) + (df_n is not None)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5))
    axes = np.atleast_1d(axes)
    i = 0
    panel_a(df_rho, ax=axes[i], uhler_threshold=uhler_threshold); i += 1
    if df_grid is not None: panel_b(df_grid, ax=axes[i]); i += 1
    if df_n is not None:    panel_c(df_n, ax=axes[i]); i += 1
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    return fig
