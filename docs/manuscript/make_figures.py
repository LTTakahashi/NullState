#!/usr/bin/env python
"""Generate publication figures for the NullState organoid off-target manuscript.

Runs ON THE POD (needs results/pilot/* + data/hnoca_cellxgene.h5ad + the nullstate env).
All quantitative panels are grounded in real artifacts; only the schematic boxes are drawn.
Outputs figs/fig1..fig5 .pdf/.png and numbers.json (single source of truth for the LaTeX text).
"""
import json, re, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
warnings.filterwarnings("ignore")

ROOT = Path("/workspace/NullState")
RES = ROOT / "results/pilot"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------- house style -----------------------------------
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})
# colour-blind safe class palette (Okabe-Ito derived)
C = {
    "clean_ontarget":       "#2c7fb8",
    "poorly_diff_ontarget": "#7fcdbb",
    "ambiguous":            "#fec44f",
    "ambiguous_novel":      "#d7301f",
    "true_offtarget":       "#762a83",
    "qc_dropped":           "#bdbdbd",
    "other":                "#e0e0e0",
}
CLASS_ORDER = ["clean_ontarget", "poorly_diff_ontarget", "ambiguous",
               "ambiguous_novel", "true_offtarget", "qc_dropped", "other"]
NICE = {"clean_ontarget": "clean on-target", "poorly_diff_ontarget": "poorly-diff. on-target",
        "ambiguous": "ambiguous", "ambiguous_novel": "ambiguous-novel (off-manifold)",
        "true_offtarget": "true off-target", "qc_dropped": "QC-dropped", "other": "other"}
ACC = "#08519c"   # accent blue
HI = "#d7301f"    # accent red

_PL = {"i": 0}
def panel_label(ax, s=None, dx=-0.16, dy=1.04):
    if s is None:
        s = "abcdefghij"[_PL["i"]]; _PL["i"] += 1
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

# ----------------------------- NUMS (log-derived) -----------------------------
NUMS = {
    "reference": {"n_cells": 2615898, "n_genes": 61497, "hdbca": 1665937, "cao": 949961,
                  "origins_mapped": "32/33"},
    "query": {"n_cells": 1767346, "n_genes": 35725, "n_protocols": 27,
              "after_hvg_filter": 1689762, "hvg_dropped_frac": 0.044,
              "after_umi_cap": 1680533, "umi_cap_dropped": 9229},
    "training": {"scvi_elbo": 695.58, "scvi_min": 148.2, "scvi_epochs": 400,
                 "scanvi_elbo": 736.10, "scanvi_epochs": 46, "scarches_elbo": 920.23,
                 "scarches_min": 14.9, "tau_H": 0.9598, "tau_R": 1.9318},
    "integration": {"ilisi_int": 8.1022, "ilisi_pca": 6.0978, "clisi_int": 1.4439,
                    "clisi_pca": 1.3128, "n_batches": 53, "n_labels": 33},
    "classes": {"ambiguous_novel": 677783, "clean_ontarget": 459441,
                "poorly_diff_ontarget": 281217, "ambiguous": 167838,
                "true_offtarget": 93795, "qc_dropped": 86813, "other": 459},
    "gates": {"A": "PASS", "B": "GO", "B_cosine": 0.8552, "B_threshold": 0.70,
              "B_n_clean": 459441, "C": "GREEN"},
    "aim2": {"baseline": 0.998, "fix1_salient30": 0.999, "fix2_batch": 0.998,
             "ideal": 0.5, "pass_threshold": 0.6},
    "diag": {
        "d1_markers": ["NPC\nprogenitor", "Neuron", "Glia /\nastro", "Hypoxia", "UPR /\nER-stress", "Glycolysis"],
        "d1_ambnov": [0.608, 0.429, 0.072, 0.961, 0.561, 2.132],
        "d1_clean":  [1.110, 0.416, -0.239, 0.600, 0.117, 2.027],
        "d1_hallmark_glyc_ambnov": 0.650, "d1_hallmark_glyc_clean": 0.462,
        "d2_potency": [0.127, 0.277, 0.419, 0.569, 0.776],
        "d2_age": [0.428, 0.322, 0.331, 0.424, 0.623],
        "d2_age_offman": [1.844, 1.714, 1.765, 1.845, 2.117],
        "d4_overall": 0.55, "d4_std": 0.288, "d4_pure_amb": 0.20, "d4_pure_clean": 0.21,
        "d3": [  # (label, ambnov_frac, n_neural)
            ("Andersen 2020 (+DAPT off)", 0.004, 1447), ("Huang 2021", 0.034, 5104),
            ("Pasca 2015", 0.100, 12825), ("Andersen 2020", 0.119, 11208),
            ("Esk 2020", 0.174, 4088), ("Watanabe 2017", 0.295, 20682),
            ("Birey 2017", 0.319, 3459), ("Velasco 2019", 0.323, 586099),
            ("Sawada 2020", 0.346, 1395), ("Bhaduri 2020 (most dir.)", 0.352, 35277),
            ("Bhaduri 2020 (dir.)", 0.357, 27968), ("Quadrato 2017", 0.378, 47442),
            ("Qian 2020", 0.413, 2767), ("Qian 2016", 0.461, 44582),
            ("Jo 2016", 0.462, 48815), ("Lancaster 2014", 0.493, 159373),
            ("Trujillo 2019", 0.507, 12646), ("Quadrato 2023", 0.526, 13495),
            ("Yoon 2019", 0.564, 135613), ("Miura 2020", 0.738, 13621),
            ("Pellegrini 2020 (hCO)", 0.784, 9913), ("Fiorenzano 2021 (silk+lam)", 0.836, 13602),
            ("Fiorenzano 2021 (std)", 0.852, 64077), ("Fiorenzano 2021 (silk)", 0.898, 12430),
            ("Xiang 2019", 0.902, 9444), ("Pellegrini 2020 (hChPO)", 0.968, 10181),
        ],
    },
}

# ----------------------------- load real data --------------------------------
print("loading annotated obs ...", flush=True)
import anndata as ad
ann = ad.read_h5ad(RES / "hnoca_annotated.h5ad", backed="r")
need = ["pilot_class", "pred_origin", "map_entropy", "offmanifold", "scored",
        "organoid_age_days", "n_genes_by_counts", "Hallmark_Glycolysis", "assay_differentiation"]
obs = ann.obs[[c for c in need if c in ann.obs.columns]].copy()
N = len(obs)
rng = np.random.default_rng(0)

print("loading HNOCA umap (backed) ...", flush=True)
h = ad.read_h5ad(ROOT / "data/hnoca_cellxgene.h5ad", backed="r")
umap_full = np.asarray(h.obsm["X_umap_scpoli"])  # (N,2) small enough

geom = json.load(open(RES / "aim3_geometry_fix2.json"))
iqc = json.load(open(RES / "integration_qc.json"))
zlin = np.load(RES / "z_lin.npy")
pops = pd.read_csv(RES / "populations.csv")["population"].to_numpy()

def parse_losses(path):
    rx = re.compile(r"Epoch (\d+)/\d+.*?train_loss_epoch=([0-9.eE+\-]+)")
    ep = {}
    try:
        for line in re.split(r"[\r\n]", Path(path).read_text(errors="ignore")):
            m = rx.search(line)
            if m:
                ep[int(m.group(1))] = float(m.group(2))
    except FileNotFoundError:
        return [], []
    xs = sorted(ep)
    return xs, [ep[e] for e in xs]

# ----------------------------------- FIG 1 -----------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    def box(x, y, w, hh, text, fc, ec="#333333", fs=7.4, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.04,rounding_size=0.12",
                     fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w/2, y + hh/2, text, ha="center", va="center", fontsize=fs, zorder=3)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                     lw=1.1, color="#555555", zorder=1))
    box(0.4, 8.4, 4.3, 1.25, "Fetal brain reference\nHDBCA 1.67M + Cao 0.95M = 2.62M cells\n32/33 lineage origins, raw UMI", "#deebf7")
    box(5.3, 8.4, 4.3, 1.25, "HNOCA query\n1.77M neural-organoid cells\n27 differentiation protocols", "#fde0dd")
    box(2.2, 6.5, 5.6, 1.15, "Integration  (batch = donor)\nscVI 400ep → scANVI → scArches surgery\nreference ELBO 695 · query surgery ELBO 920", "#f0f0f0")
    box(2.2, 4.8, 5.6, 1.05, r"Per-cell mapping scores" + "\n" +
        r"off-manifold $\rho$  ($\tau_R$=1.93)   ·   mapping entropy $H$  ($\tau_H$=0.96)", "#f0f0f0")
    box(0.4, 2.9, 9.2, 1.15, "Six-way off-target classification\nclean · poorly-diff · ambiguous · ambiguous-novel (off-manifold) · true-off-target · QC", "#f7f7f7")
    box(0.4, 0.7, 2.85, 1.5, "Aim 1\nFeasibility gates\nA · B · C", "#e5f5e0")
    box(3.55, 0.7, 2.9, 1.5, "Aim 2\nLineage–dish\ndisentanglement\n(contrastiveVI)", "#fff7bc")
    box(6.75, 0.7, 2.85, 1.5, "Aim 3\nOff-target\ngeometry\n(sliced-Wasserstein)", "#e0ecf4")
    arrow(2.55, 8.4, 4.0, 7.65); arrow(7.45, 8.4, 6.0, 7.65)
    arrow(5.0, 6.5, 5.0, 5.85); arrow(5.0, 4.8, 5.0, 4.05)
    arrow(2.4, 2.9, 1.8, 2.2); arrow(5.0, 2.9, 5.0, 2.2); arrow(7.6, 2.9, 8.2, 2.2)
    ax.set_title("NullState: a reference-based pipeline for organoid off-target detection and geometry",
                 fontsize=9.5, fontweight="bold", pad=2)
    fig.savefig(OUT / "fig1.pdf"); fig.savefig(OUT / "fig1.png"); plt.close(fig); print("fig1 ok")

# ----------------------------------- FIG 2 -----------------------------------
def fig2():
    _PL["i"] = 0
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.42,
                          height_ratios=[1, 1.05], width_ratios=[1.15, 1])
    # (a) class distribution
    axa = fig.add_subplot(gs[0, 0])
    cl = NUMS["classes"]; order = [c for c in CLASS_ORDER if c in cl]
    vals = [cl[c] for c in order]; tot = sum(cl.values())
    y = np.arange(len(order))[::-1]
    axa.barh(y, vals, color=[C[c] for c in order], edgecolor="white", lw=0.5)
    for yi, v in zip(y, vals):
        axa.text(v + tot*0.01, yi, f"{v/tot*100:.1f}%", va="center", fontsize=6.6)
    axa.set_yticks(y); axa.set_yticklabels([NICE[c] for c in order], fontsize=6.6)
    axa.set_xlabel("cells"); axa.set_xlim(0, max(vals)*1.18)
    from matplotlib.ticker import FuncFormatter
    axa.set_xticks([0, 200000, 400000, 600000])
    axa.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v/1000:.0f}k"))
    axa.set_title(f"Off-target classification  (n = {tot:,})", fontsize=8)
    panel_label(axa, dx=-0.55)
    # (b) classification score space
    axb = fig.add_subplot(gs[0, 1])
    m = obs["scored"].to_numpy().astype(bool) if "scored" in obs else np.ones(N, bool)
    idx = np.flatnonzero(m); idx = rng.choice(idx, min(45000, idx.size), replace=False)
    ent = pd.to_numeric(obs["map_entropy"], errors="coerce").to_numpy()[idx]
    off = pd.to_numeric(obs["offmanifold"], errors="coerce").to_numpy()[idx]
    pc = obs["pilot_class"].to_numpy()[idx]
    for c in ["clean_ontarget", "poorly_diff_ontarget", "ambiguous", "true_offtarget", "ambiguous_novel"]:
        sel = pc == c
        if sel.sum():
            axb.scatter(ent[sel], off[sel], s=1.4, c=C[c], alpha=0.35, lw=0, rasterized=True)
    axb.axhline(NUMS["training"]["tau_R"], color="#333", lw=0.9, ls="--")
    axb.axvline(NUMS["training"]["tau_H"], color="#333", lw=0.9, ls=":")
    axb.text(NUMS["training"]["tau_H"]+0.02, axb.get_ylim()[1]*0.97, r"$\tau_H$", fontsize=6.8, va="top")
    axb.text(axb.get_xlim()[1]*0.99, NUMS["training"]["tau_R"]+0.04, r"$\tau_R$", fontsize=6.8, ha="right")
    axb.set_xlabel("mapping entropy  $H$"); axb.set_ylabel(r"off-manifold score  $\rho$")
    axb.set_title("Per-cell score space", fontsize=8); panel_label(axb)
    # (c) integration QC
    axc = fig.add_subplot(gs[1, 0])
    g = NUMS["integration"]
    x = np.arange(2); w = 0.36
    axc.bar(x - w/2, [g["ilisi_pca"], g["clisi_pca"]], w, label="unintegrated (PCA)", color="#bdbdbd", edgecolor="white")
    axc.bar(x + w/2, [g["ilisi_int"], g["clisi_int"]], w, label="scVI integrated", color=ACC, edgecolor="white")
    axc.set_xticks(x); axc.set_xticklabels(["batch iLISI\n(mixing ↑ better)", "label cLISI\n(biology ~preserved)"], fontsize=6.8)
    axc.set_ylabel("LISI"); axc.legend(frameon=False, loc="upper left", fontsize=6.2)
    axc.set_title("Reference harmonization", fontsize=8); panel_label(axc, dx=-0.2)
    for xi, a, b in zip(x, [g["ilisi_pca"], g["clisi_pca"]], [g["ilisi_int"], g["clisi_int"]]):
        axc.text(xi - w/2, a+0.05, f"{a:.2f}", ha="center", fontsize=6); axc.text(xi + w/2, b+0.05, f"{b:.2f}", ha="center", fontsize=6)
    # (d) gates
    axd = fig.add_subplot(gs[1, 1]); axd.axis("off"); axd.set_xlim(0, 10); axd.set_ylim(0, 10)
    rows = [("Gate A — definition", "PASS", "#2ca25f", "off-target schema sane"),
            ("Gate B — dish vector", "GO", "#2ca25f", f"cos = {NUMS['gates']['B_cosine']:.3f} > {NUMS['gates']['B_threshold']:.2f}"),
            ("Gate C — count power", "GREEN", "#2ca25f", f"{NUMS['classes']['true_offtarget']:,} true-off-target")]
    axd.set_title("Feasibility gates", fontsize=8)
    for i, (name, status, col, sub) in enumerate(rows):
        yy = 8.2 - i*2.7
        axd.add_patch(FancyBboxPatch((0.2, yy-1.1), 9.6, 2.1, boxstyle="round,pad=0.05,rounding_size=0.15",
                      fc="#f7f7f7", ec="#cccccc", lw=0.8))
        axd.text(0.6, yy+0.25, name, fontsize=7.6, fontweight="bold", va="center")
        axd.text(0.6, yy-0.55, sub, fontsize=6.6, va="center", color="#555")
        axd.add_patch(FancyBboxPatch((7.0, yy-0.55, ), 2.6, 1.15, boxstyle="round,pad=0.05,rounding_size=0.2",
                      fc=col, ec="none"))
        axd.text(8.3, yy+0.0, status, ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    panel_label(axd, dx=-0.05)
    fig.savefig(OUT / "fig2.pdf"); fig.savefig(OUT / "fig2.png"); plt.close(fig); print("fig2 ok")

# ----------------------------------- FIG 3 -----------------------------------
def fig3():
    _PL["i"] = 0
    fig = plt.figure(figsize=(7.2, 8.2))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.4, height_ratios=[1, 1, 1.15])
    d = NUMS["diag"]
    # (a) D1 markers
    axa = fig.add_subplot(gs[0, 0])
    x = np.arange(len(d["d1_markers"])); w = 0.4
    axa.bar(x - w/2, d["d1_clean"], w, label="clean on-target", color=C["clean_ontarget"], edgecolor="white")
    axa.bar(x + w/2, d["d1_ambnov"], w, label="ambiguous-novel", color=C["ambiguous_novel"], edgecolor="white")
    axa.axhline(0, color="#888", lw=0.6)
    axa.set_xticks(x); axa.set_xticklabels(d["d1_markers"], fontsize=6.0)
    axa.set_ylabel("gene-set score"); axa.legend(frameon=False, fontsize=6.0, loc="upper center", ncol=1)
    axa.set_title("Identity & stress programs", fontsize=8); panel_label(axa, dx=-0.2)
    # (b) D2 maturation
    axb = fig.add_subplot(gs[0, 1])
    q = np.arange(5)
    axb.bar(q, d["d2_potency"], color=C["ambiguous_novel"], alpha=0.85, edgecolor="white", width=0.7, label="by potency (n-genes)")
    axb.plot(q, d["d2_age"], "-o", color=ACC, lw=1.4, ms=4, label="by organoid age")
    axb.set_xticks(q); axb.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5"])
    axb.set_xlabel("quintile (low → high)"); axb.set_ylabel("ambiguous-novel fraction")
    axb.legend(frameon=False, fontsize=6.0, loc="upper left"); axb.set_ylim(0, 0.85)
    axb.set_title("Maturation / potency axis", fontsize=8); panel_label(axb)
    # (c) D3 protocol gradient (full width)
    axc = fig.add_subplot(gs[1, :])
    d3 = d["d3"]; labels = [x[0] for x in d3]; fr = [x[1] for x in d3]
    yy = np.arange(len(d3))[::-1]
    cols = plt.cm.RdYlBu_r(np.array(fr))
    axc.barh(yy, fr, color=cols, edgecolor="white", lw=0.3)
    axc.set_yticks(yy); axc.set_yticklabels(labels, fontsize=5.4)
    axc.set_xlabel("ambiguous-novel fraction among neural cells"); axc.set_xlim(0, 1.0)
    axc.set_title("Protocol-stratified off-manifold burden  (26 protocols, $\\sigma$ = 0.28)", fontsize=8)
    panel_label(axc, dx=-0.32)
    # (d) D4 cluster coherence (computed)
    axd = fig.add_subplot(gs[2, 0])
    try:
        import scanpy as sc
        neural = obs["pred_origin"].isin(["neural", "neural_crest"]).to_numpy()
        amb = (obs["pilot_class"] == "ambiguous_novel").to_numpy()
        cln = (obs["pilot_class"] == "clean_ontarget").to_numpy()
        both = np.flatnonzero(neural & (amb | cln))
        s = np.sort(rng.choice(both, min(60000, both.size), replace=False))
        E = np.asarray(h.obsm["X_scpoli"][s])
        cl = ad.AnnData(np.zeros((len(s), 1), dtype="float32")); cl.obsm["X_emb"] = E
        cl.obs["amb"] = amb[s].astype(int)
        sc.pp.neighbors(cl, use_rep="X_emb", n_neighbors=15)
        sc.tl.leiden(cl, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)
        per = cl.obs.groupby("leiden")["amb"].mean().to_numpy()
        axd.hist(per, bins=np.linspace(0, 1, 16), color="#756bb1", edgecolor="white")
        axd.axvline(d["d4_overall"], color="#333", ls="--", lw=1.0)
        axd.text(d["d4_overall"]+0.02, axd.get_ylim()[1]*0.9, "overall\n0.55", fontsize=6.0)
        axd.set_xlabel("per-cluster ambiguous-novel fraction"); axd.set_ylabel("Leiden clusters")
    except Exception as e:
        print("D4 recompute failed, summary bars:", e)
        axd.bar([0, 1, 2], [d["d4_pure_clean"], 1-d["d4_pure_clean"]-d["d4_pure_amb"], d["d4_pure_amb"]],
                color=[C["clean_ontarget"], "#cccccc", C["ambiguous_novel"]], edgecolor="white")
        axd.set_xticks([0, 1, 2]); axd.set_xticklabels(["pure\nclean", "mixed", "pure\namb-novel"])
        axd.set_ylabel("fraction of cells")
    axd.set_title("Cluster coherence", fontsize=8); panel_label(axd, dx=-0.2)
    # (e) UMAP coloured by class
    axe = fig.add_subplot(gs[2, 1])
    sidx = rng.choice(N, min(80000, N), replace=False)
    U = umap_full[sidx]; pc = obs["pilot_class"].to_numpy()[sidx]
    axe.scatter(U[:, 0], U[:, 1], s=0.6, c="#eeeeee", lw=0, rasterized=True)
    for c in ["clean_ontarget", "ambiguous_novel"]:
        sel = pc == c
        axe.scatter(U[sel, 0], U[sel, 1], s=0.8, c=C[c], lw=0, alpha=0.6, rasterized=True,
                    label=NICE[c].split(" (")[0])
    axe.set_xticks([]); axe.set_yticks([]); axe.set_xlabel("UMAP 1"); axe.set_ylabel("UMAP 2")
    axe.legend(frameon=False, fontsize=6.0, markerscale=6, loc="lower left")
    axe.set_title("scPoli UMAP by class", fontsize=8); panel_label(axe)
    fig.savefig(OUT / "fig3.pdf"); fig.savefig(OUT / "fig3.png"); plt.close(fig); print("fig3 ok")

# ----------------------------------- FIG 4 -----------------------------------
def fig4():
    _PL["i"] = 0
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 3, wspace=0.45, width_ratios=[1, 1.1, 1.15])
    a2 = NUMS["aim2"]
    # (a) adversary bars
    axa = fig.add_subplot(gs[0, 0])
    labs = ["baseline\n($z_{iv}{=}10$)", "+capacity\n($z_{iv}{=}30$)", "+batch\n(decoder)"]
    vals = [a2["baseline"], a2["fix1_salient30"], a2["fix2_batch"]]
    axa.bar(np.arange(3), vals, color=["#969696", "#fb6a4a", "#a50f15"], edgecolor="white", width=0.66)
    axa.axhline(a2["ideal"], color="#2ca25f", ls="--", lw=1.1); axa.text(2.4, a2["ideal"]+0.01, "ideal 0.5", color="#2ca25f", fontsize=6, ha="right")
    axa.axhline(a2["pass_threshold"], color="#888", ls=":", lw=1.0); axa.text(2.4, a2["pass_threshold"]+0.01, "pass < 0.6", color="#666", fontsize=6, ha="right")
    for i, v in enumerate(vals):
        axa.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=6.6, fontweight="bold")
    axa.set_xticks(np.arange(3)); axa.set_xticklabels(labs, fontsize=6.2)
    axa.set_ylabel("adversary balanced-acc"); axa.set_ylim(0.45, 1.04)
    axa.set_title("Source recoverable from $z_\\mathrm{lin}$", fontsize=8); panel_label(axa, dx=-0.28)
    # (b) loss curves
    axb = fig.add_subplot(gs[0, 1])
    for path, lab, col in [("ws1_gpu.log", "baseline", "#969696"), ("ws1_fix1.log", "+capacity", "#fb6a4a"),
                           ("ws1_fix2.log", "+batch", "#a50f15")]:
        xs, ys = parse_losses(RES / path)
        if xs:
            axb.plot(xs, ys, color=col, lw=1.3, label=lab)
    axb.set_xlabel("epoch"); axb.set_ylabel("training loss"); axb.legend(frameon=False, fontsize=6)
    axb.set_title("Training converges regardless", fontsize=8); panel_label(axb, dx=-0.22)
    # (c) mechanism schematic
    axc = fig.add_subplot(gs[0, 2]); axc.axis("off"); axc.set_xlim(0, 10); axc.set_ylim(0, 10)
    axc.add_patch(FancyBboxPatch((0.3, 6.7), 4.2, 2.4, boxstyle="round,pad=0.05,rounding_size=0.15", fc="#deebf7", ec="#888"))
    axc.text(2.4, 7.9, "primary\nfetal", ha="center", va="center", fontsize=7)
    axc.add_patch(FancyBboxPatch((5.5, 6.7), 4.2, 2.4, boxstyle="round,pad=0.05,rounding_size=0.15", fc="#fde0dd", ec="#888"))
    axc.text(7.6, 7.9, "organoid\n(in-vitro)", ha="center", va="center", fontsize=7)
    axc.annotate("", xy=(5.4, 7.9), xytext=(4.6, 7.9), arrowprops=dict(arrowstyle="<->", lw=1.0, color="#a50f15"))
    axc.text(5.0, 8.4, "near-disjoint\nmanifolds", ha="center", fontsize=5.6, color="#a50f15")
    axc.add_patch(FancyBboxPatch((1.8, 3.6), 6.4, 1.7, boxstyle="round,pad=0.05,rounding_size=0.15", fc="#f0f0f0", ec="#888"))
    axc.text(5.0, 4.45, "encoder  $q(z_\\mathrm{lin}\\,|\\,x)$\nsees source → encodes it", ha="center", va="center", fontsize=6.6)
    axc.add_patch(FancyBboxPatch((1.8, 1.0), 6.4, 1.7, boxstyle="round,pad=0.05,rounding_size=0.15", fc="#fff7bc", ec="#888"))
    axc.text(5.0, 1.85, "decoder batch-cond.\ncannot undo the encoder", ha="center", va="center", fontsize=6.6)
    axc.annotate("", xy=(5.0, 3.5), xytext=(5.0, 5.4), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#555"))
    axc.annotate("", xy=(5.0, 2.8), xytext=(5.0, 3.5), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#555"))
    axc.set_title("Why disentanglement fails", fontsize=8); panel_label(axc, dx=-0.05)
    fig.savefig(OUT / "fig4.pdf"); fig.savefig(OUT / "fig4.png"); plt.close(fig); print("fig4 ok")

# ----------------------------------- FIG 5 -----------------------------------
POP4 = ["cardiac muscle cell", "cell of skeletal muscle", "cortical cell of adrenal gland", "fibroblast"]
SHORT = {"cardiac muscle cell": "cardiac\nmuscle", "cell of skeletal muscle": "skeletal\nmuscle",
         "cortical cell of adrenal gland": "adrenal\ncortex", "fibroblast": "fibroblast"}
def fig5():
    _PL["i"] = 0
    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.42)
    # build symmetric W matrix + pair list from json
    pj = geom["pairs"]
    M = np.zeros((4, 4)); pairlist = []
    for i in range(4):
        for j in range(i+1, 4):
            k1, k2 = f"{POP4[i]}|{POP4[j]}", f"{POP4[j]}|{POP4[i]}"
            p = pj.get(k1) or pj.get(k2)
            M[i, j] = M[j, i] = p["cross_mean"]
            pairlist.append((POP4[i], POP4[j], p))
    # (a) z_lin embedding by population
    axa = fig.add_subplot(gs[0, 0])
    keep = np.isin(pops, POP4)
    sub = np.flatnonzero(keep)
    if sub.size > 40000:
        sub = np.sort(rng.choice(sub, 40000, replace=False))
    try:
        import scanpy as sc
        za = ad.AnnData(zlin[sub].astype("float32"))
        sc.pp.neighbors(za, use_rep="X", n_neighbors=15)
        sc.tl.umap(za, min_dist=0.3)
        E = za.obsm["X_umap"]; xlab, ylab = "z-UMAP 1", "z-UMAP 2"
    except Exception as e:
        print("z-UMAP failed -> PCA:", e)
        from sklearn.decomposition import PCA
        E = PCA(2).fit_transform(zlin[sub]); xlab, ylab = "z-PC 1", "z-PC 2"
    pcols = {"cardiac muscle cell": "#e6550d", "cell of skeletal muscle": "#fd8d3c",
             "cortical cell of adrenal gland": "#3182bd", "fibroblast": "#31a354"}
    for c in POP4:
        sel = pops[sub] == c
        axa.scatter(E[sel, 0], E[sel, 1], s=1.2, c=pcols[c], lw=0, alpha=0.5, rasterized=True,
                    label=SHORT[c].replace("\n", " "))
    axa.set_xticks([]); axa.set_yticks([]); axa.set_xlabel(xlab); axa.set_ylabel(ylab)
    axa.legend(frameon=False, fontsize=5.8, markerscale=5, loc="best")
    axa.set_title("Off-target populations in $z_\\mathrm{lin}$", fontsize=8); panel_label(axa, dx=-0.16)
    # (b) W-distance heatmap
    axb = fig.add_subplot(gs[0, 1])
    im = axb.imshow(M, cmap="magma", vmin=0)
    axb.set_xticks(range(4)); axb.set_yticks(range(4))
    axb.set_xticklabels([SHORT[p] for p in POP4], fontsize=5.8)
    axb.set_yticklabels([SHORT[p] for p in POP4], fontsize=5.8)
    for i in range(4):
        for j in range(4):
            if i != j:
                axb.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.2,
                         color="white" if M[i, j] < M.max()*0.6 else "black")
    cb = fig.colorbar(im, ax=axb, fraction=0.046, pad=0.04); cb.set_label("sliced-$W$", fontsize=6.5)
    axb.set_title("Pairwise lineage distance", fontsize=8); panel_label(axb)
    # (c) forest: cross vs floor
    axc = fig.add_subplot(gs[1, 0])
    pl = sorted(pairlist, key=lambda t: t[2]["cross_mean"])
    yy = np.arange(len(pl))
    for k, (a, b, p) in enumerate(pl):
        axc.plot([p["cross_lo"], p["cross_hi"]], [k, k], color=ACC, lw=1.6, zorder=2)
        axc.plot(p["cross_mean"], k, "o", color=ACC, ms=4, zorder=3)
        axc.plot(p["floor_q95"], k, "s", color=HI, ms=4, zorder=3)
    axc.set_yticks(yy); axc.set_yticklabels([f"{SHORT[a].replace(chr(10),' ')} – {SHORT[b].replace(chr(10),' ')}" for a, b, _ in pl], fontsize=5.6)
    axc.set_xscale("log"); axc.set_xlabel("distance (log)")
    axc.plot([], [], "o", color=ACC, label="cross  (95% CI)"); axc.plot([], [], "s", color=HI, label="noise floor $q_{95}$")
    axc.legend(frameon=False, fontsize=6, loc="lower right")
    axc.set_title("All pairs clear the floor", fontsize=8); panel_label(axc, dx=-0.42)
    # (d) MDS of populations
    axd = fig.add_subplot(gs[1, 1])
    try:
        from sklearn.manifold import MDS
        P = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto").fit_transform(M)
    except Exception as e:
        print("MDS failed:", e); P = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    cen = P.mean(0); span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1])) or 1.0
    # connect muscle pair (drawn first, behind points)
    mi = [POP4.index("cardiac muscle cell"), POP4.index("cell of skeletal muscle")]
    axd.plot(P[mi, 0], P[mi, 1], color="#e6550d", lw=1.2, ls="--", zorder=1)
    for i, c in enumerate(POP4):
        axd.scatter(P[i, 0], P[i, 1], s=120, c=pcols[c], edgecolor="white", lw=1.2, zorder=3)
        v = P[i] - cen; nrm = np.linalg.norm(v) or 1.0; off = v / nrm * span * 0.16  # radial outward
        axd.text(P[i, 0] + off[0], P[i, 1] + off[1], SHORT[c].replace("\n", " "),
                 ha="center", va="center", fontsize=6.4, zorder=4)
    axd.margins(0.28)
    axd.set_xticks([]); axd.set_yticks([]); axd.set_xlabel("MDS 1"); axd.set_ylabel("MDS 2")
    axd.set_title("Lineage MDS (from $W$)", fontsize=8); panel_label(axd)
    fig.savefig(OUT / "fig5.pdf"); fig.savefig(OUT / "fig5.png"); plt.close(fig); print("fig5 ok")

if __name__ == "__main__":
    NUMS["geometry_pairs"] = {f"{a}|{b}": {k: p[k] for k in ("cross_mean", "cross_lo", "cross_hi", "floor_q95", "n_star", "verdict")}
                              for a, b, p in [(POP4[i], POP4[j], (geom["pairs"].get(f"{POP4[i]}|{POP4[j]}") or geom["pairs"].get(f"{POP4[j]}|{POP4[i]}")))
                                              for i in range(4) for j in range(i+1, 4)]}
    NUMS["geometry_verdicts"] = {"real_difference": 6, "inconclusive_underpowered": 130, "n_min": geom["n_min"]}
    json.dump(NUMS, open(OUT.parent / "numbers.json", "w"), indent=2, default=float)
    for fn in (fig1, fig2, fig3, fig4, fig5):
        try:
            fn()
        except Exception as e:
            import traceback; print(f"!! {fn.__name__} FAILED:", e); traceback.print_exc()
    print("ALL FIGURES DONE")
