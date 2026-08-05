"""Figures for the invariance-class paper, generated from the identifiability results."""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# The study now lives in its own repository:
#     https://github.com/LTTakahashi/identifiability
# Point IDENTIFIABILITY_REPO at a checkout, or clone it beside this one.
_here = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.environ.get("IDENTIFIABILITY_REPO"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_here))), "identifiability"),
]
PROBE = next((c for c in _candidates if c and os.path.isdir(os.path.join(c, "results"))), None)
if PROBE is None:
    raise SystemExit(
        "Cannot find the identifiability study.\n"
        "  git clone https://github.com/LTTakahashi/identifiability.git\n"
        "  export IDENTIFIABILITY_REPO=/path/to/identifiability\n"
        "Then re-run this script."
    )
sys.path.insert(0, PROBE)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
C_BLIND, C_SEE, C_MID = "#c0392b", "#2c6fbb", "#f0f0f0"


# ---------------------------------------------------------------- FIGURE 1
def fig1_taxonomy():
    """Estimand x failure mode: the invariance-class table."""
    rows = [
        ("identity",            "—",                 [0.996, 1.000, 1.000, 1.000, 0.997, -0.000]),
        ("permute axes",        "nuisance",          [0.996, 1.000, 1.000, 1.000, 0.998, -0.000]),
        ("per-axis rescale",    "nuisance",          [0.999, 1.000, 1.000, 1.000, 0.997, -0.000]),
        ("rotate 45°",          "entanglement",      [0.996, 1.000, 0.707, 0.685, 0.000, 0.293]),
        ("shear",               "entanglement",      [0.996, 1.000, 0.887, 0.878, 0.508, 0.113]),
        ("monotone nonlinear",  "reparametrisation", [0.987, 0.885, 0.885, 1.000, 0.997, -0.000]),
        ("isotropic noise",     "info loss (axis-f.)", [0.725, 0.857, 0.857, 0.846, 0.838, 0.000]),
        ("fold axis",           "info loss (axis-f.)", [-0.062, 0.531, 0.531, 0.507, 0.594, 0.000]),
        ("collapse axis",       "info loss (axis-f.)", [-0.070, 0.502, 0.500, 0.502, 0.801, 0.001]),
        ("common-mode noise",   "info loss (mixed)",  [0.537, 0.658, 0.442, 0.428, 0.038, 0.216]),
        ("collapse rotated ax.", "info loss (mixed)", [0.482, 0.500, 0.358, 0.356, 0.001, 0.142]),
    ]
    cols = ["kNN-$R^2$", "CCA", "MCC-P", "MCC-S", "DCI-D", "CCA$-$MCC"]
    ident = np.array(rows[0][2], float)
    M = np.array([r[2] for r in rows], float)
    # "blind" = within 0.05 of the identity value on a genuine failure
    blind = np.abs(M - ident) < 0.05
    is_failure = np.array([r[1] not in ("—", "nuisance") for r in rows])

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(rows)); ax.invert_yaxis()
    for i, (name, kind, vals) in enumerate(rows):
        for j, v in enumerate(vals):
            b = blind[i, j] and is_failure[i]
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                       facecolor=(C_BLIND if b else
                                                  (C_MID if not is_failure[i] else "#eaf2fb")),
                                       edgecolor="white", lw=1.2,
                                       alpha=0.85 if b else 1.0))
            ax.text(j + 0.5, i + 0.5, f"{v:.3f}", ha="center", va="center",
                    fontsize=6.8, color="white" if b else "#222",
                    fontweight="bold" if b else "normal")
    ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([f"{n}" for n, _, _ in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # right-hand annotation of the failure class
    for i, (_, kind, _) in enumerate(rows):
        ax.text(len(cols) + 0.12, i + 0.5, kind, fontsize=6.3, va="center",
                color="#555", style="italic")
    ax.add_patch(plt.Rectangle((0, 0), len(cols), 3, fill=False,
                               edgecolor="#999", lw=1.0, ls=":"))
    ax.text(-0.15, 1.5, "nuisances", rotation=90, ha="center", va="center",
            fontsize=6.5, color="#777")
    ax.set_title("Each estimand is blind to a different failure "
                 "(red = within 0.05 of its identity value on a genuine failure)",
                 pad=8)
    fig.savefig(f"{OUT}/fig1.pdf"); plt.close(fig)
    print("fig1 done")


# ---------------------------------------------------------------- FIGURE 2
def fig2_geometry_constant():
    """The manufactured threshold, and why it is a recipe not a universal."""
    from analysis.closed_form import make, transfer_r2, analytic, zero_crossing, RHOS
    from matplotlib.ticker import NullLocator, NullFormatter
    fig = plt.figure(figsize=(6.9, 2.35))
    gs = gridspec.GridSpec(1, 3, wspace=0.46)

    ax = fig.add_subplot(gs[0])
    meas = [transfer_r2(*make(r)) for r in RHOS]
    rr = np.linspace(0.02, 1.0, 200)
    ax.plot(rr, analytic(rr), "-", color="#333", lw=1.6,
            label=r"$1-4(1-\rho)^3$")
    ax.plot(RHOS, meas, "o", color=C_SEE, ms=4.5, label="oracle embedding")
    ax.axhline(0, color="#aaa", lw=0.8, ls=":")
    zc = 1 - 4 ** (-1 / 3)
    ax.axvline(zc, color=C_BLIND, lw=1.2, ls="--")
    ax.text(zc + 0.03, -2.0, r"$\rho^*=1-4^{-1/3}$" "\n" r"$=0.370$",
            fontsize=6.5, color=C_BLIND)
    ax.set_xlabel(r"support overlap $\rho$"); ax.set_ylabel("kNN transfer $R^2$")
    ax.set_title("(a) a threshold with no model in it", fontsize=7.5)
    ax.legend(frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[1])
    scales = [0.001, 0.03, 0.3, 0.6, 1.0]
    zs = [zero_crossing(RHOS, [transfer_r2(*make(r, d=10, noise_scale=s)) for r in RHOS])
          for s in scales]
    ax.semilogx(scales, zs, "o-", color=C_SEE, ms=4)
    ax.axhline(zc, color="#333", ls="--", lw=1.0)
    ax.text(0.0015, zc + 0.02, "1-D value 0.370", fontsize=6.3)
    ax.set_xlabel("scale of irrelevant axes ($d{=}10$)")
    ax.set_ylabel(r"apparent $\rho^*$")
    ax.set_title("(b) it moves with embedding scale", fontsize=7.5)

    ax = fig.add_subplot(gs[2])
    ns = [2000, 8000, 32000]
    for marg, col in (("uniform", C_SEE), ("triangular", "#e08a1e"), ("gaussian", C_BLIND)):
        ys = []
        for n in ns:
            z = zero_crossing(RHOS, [transfer_r2(*make(r, marginal=marg, n=n)) for r in RHOS])
            ys.append(z if z is not None else 0.0)
        ax.plot(ns, ys, "o-", color=col, ms=4, label=marg)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(["2k", "8k", "32k"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("$n$ per domain")
    ax.set_ylabel(r"apparent $\rho^*$")
    ax.set_title("(c) stable only for compact support", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.5, loc="center right")
    fig.savefig(f"{OUT}/fig2.pdf"); plt.close(fig)
    print("fig2 done")


# ---------------------------------------------------------------- FIGURE 3
def fig3_floor_instrument():
    """The floor is a random frame; sigma_min(L) is the right axis."""
    from analysis.random_frame_null import analytic_moments, cdf, GAP_MAX, ks_test
    fig = plt.figure(figsize=(6.6, 2.3))
    gs = gridspec.GridSpec(1, 3, wspace=0.38)

    # (a) pooled empirical CDF vs the analytic random-frame law
    ax = fig.add_subplot(gs[0])
    s4 = pd.read_csv(f"{PROBE}/results/main_sweep.csv")
    s4 = s4[(s4.rho == 1.0) & (s4.delta == 0.0)]
    s4 = s4.assign(gap=s4.learned_cca - s4.learned_mcc)
    xs = np.linspace(0, GAP_MAX, 300)
    ax.plot(xs, cdf(xs), "-", color="#333", lw=1.6, label="random-frame law")
    for arm, col in (("conditional", C_SEE), ("ivae_2env", "#e08a1e"),
                     ("ivae_5env", C_BLIND)):
        g = np.sort(np.clip(s4[s4.arm == arm].gap.values, 0, GAP_MAX))
        ax.step(g, np.arange(1, len(g) + 1) / len(g), where="post",
                color=col, lw=1.1, label=arm)
    ax.set_xlabel(r"gap $=$ CCA $-$ MCC"); ax.set_ylabel("empirical CDF")
    ax.set_title("(a) no arm rejects the null", fontsize=8)
    ax.legend(frameon=False, loc="lower right", fontsize=6)

    # (b) gap -> frame angle
    ax = fig.add_subplot(gs[1])
    phi = np.linspace(0, 45, 200)
    ax.plot(phi, 1 - np.cos(np.radians(phi)), color="#333", lw=1.6)
    m, sd = analytic_moments()
    ax.axhline(m, color=C_BLIND, ls="--", lw=1.0)
    ax.fill_between([0, 45], m - sd, m + sd, color=C_BLIND, alpha=0.12)
    ax.text(1.5, m + 0.012, r"$E[\mathrm{gap}]=0.0997$", fontsize=6.3, color=C_BLIND)
    ax.set_xlabel(r"frame angle $\varphi$ (deg)"); ax.set_ylabel("gap")
    ax.set_title(r"(b) gap $=1-\cos\varphi$", fontsize=8)

    # (c) sigma_min sweep
    ax = fig.add_subplot(gs[2])
    sm = pd.read_csv(f"{PROBE}/results/sigma_min.csv")
    g = sm.groupby("s_level").agg(x=("sigma_min", "mean"), y=("gap", "mean"),
                                  e=("gap", "sem")).sort_values("x")
    ax.errorbar(g.x, g.y, yerr=g.e, fmt="o-", color=C_SEE, ms=4, capsize=2)
    ax.axhline(m, color=C_BLIND, ls="--", lw=1.0)
    ax.fill_between([g.x.min(), g.x.max()], m - sd / np.sqrt(15), m + sd / np.sqrt(15),
                    color=C_BLIND, alpha=0.12)
    ax.text(g.x.min(), m + 0.006, "random-frame mean", fontsize=6.2, color=C_BLIND)
    ax.set_xscale("log"); ax.set_xlabel(r"$\sigma_{\min}(L)$  (fixed $n_{\rm env}=9$)")
    ax.set_ylabel("floor (gap)")
    ax.set_title(r"(c) monotone, but never rejects", fontsize=8)
    fig.savefig(f"{OUT}/fig3.pdf"); plt.close(fig)
    print("fig3 done")


# ---------------------------------------------------------------- FIGURE 4
def fig4_application():
    """Overlap governs recovery iff the objective aligns."""
    # taller, with room reserved at the bottom for one shared legend
    fig = plt.figure(figsize=(6.6, 2.7))
    gs = gridspec.GridSpec(1, 3, wspace=0.38, bottom=0.30)
    al = pd.read_csv(f"{PROBE}/results/alignment.csv")
    rhos = sorted(al.rho.unique(), reverse=True)
    style = {"faithful": (C_SEE, "faithful encoder"),
             "mix_200": ("#e08a1e", r"moment-match $\lambda{=}200$"),
             "mix_1000": (C_BLIND, r"moment-match $\lambda{=}1000$"),
             "adv_100": ("#7a5199", "adversarial (DANN)")}

    ax = fig.add_subplot(gs[0])
    for arm, (col, lab) in style.items():
        s = al[al.arm == arm]
        mu = [s[s.rho == r].cca.mean() for r in rhos]
        se = [s[s.rho == r].cca.sem() for r in rhos]
        ax.errorbar(rhos, mu, yerr=se, fmt="o-", color=col, ms=3.5, lw=1.2,
                    capsize=1.5, label=lab)
    ax.set_xlabel(r"support overlap $\rho$"); ax.set_ylabel("CCA (subspace recovery)")
    ax.set_title("(a) recovery vs overlap", fontsize=8)
    handles, labels = ax.get_legend_handles_labels()

    ax = fig.add_subplot(gs[1])
    for arm, (col, lab) in style.items():
        s = al[al.arm == arm]
        p = [(s[s.rho == r].cca < 0.85).mean() for r in rhos]
        ax.plot(rhos, p, "o-", color=col, ms=3.5, lw=1.2)
    ax.set_xlabel(r"support overlap $\rho$")
    ax.set_ylabel(r"$P(\mathrm{CCA}<0.85)$")
    ax.set_title("(b) collapse is stochastic", fontsize=8)

    ax = fig.add_subplot(gs[2])
    for arm, (col, lab) in style.items():
        s = al[al.arm == arm]
        mu = [(s[s.rho == r].cca - s[s.rho == r].mcc).mean() for r in rhos]
        ax.plot(rhos, mu, "o-", color=col, ms=3.5, lw=1.2)
    from analysis.random_frame_null import analytic_moments
    m, _ = analytic_moments()
    ax.axhline(m, color="#333", ls="--", lw=1.0)
    ax.text(0.03, m + 0.004, "random-frame mean", fontsize=6.2, color="#333",
            ha="left", va="bottom")
    ax.set_xlabel(r"support overlap $\rho$"); ax.set_ylabel("CCA $-$ MCC")
    ax.set_title("(c) the gap sees none of it", fontsize=8)

    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.01), handlelength=1.6,
               columnspacing=1.4)
    fig.savefig(f"{OUT}/fig4.pdf"); plt.close(fig)
    print("fig4 done")


if __name__ == "__main__":
    fig1_taxonomy()
    fig3_floor_instrument()
    fig4_application()
    fig2_geometry_constant()      # slowest (regenerates kNN sweeps)
    print("all figures ->", OUT)
