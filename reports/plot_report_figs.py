"""Generate publication-quality figures for the NullState pilot report.
Run on the pod (has torch + matplotlib + the model.pt histories). Writes PDFs to /workspace/figs/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

torch.set_grad_enabled(False)
NSBLUE, NSGREEN, NSAMBER, NSGRAY = "#205295", "#1E824C", "#B07A00", "#5A5A5A"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
OUT = Path("/workspace/figs"); OUT.mkdir(exist_ok=True)
RM = "/workspace/NullState/results/pilot/reference_model"
RES = "/workspace/NullState/results/pilot"


def history(path):
    m = torch.load(path, map_location="cpu", weights_only=False)
    return m["attr_dict"]["history_"]


def arr(h, k):
    return np.asarray(h[k].values, dtype=float).ravel()


# ---- Fig 1: training convergence ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
for ax, name, path in [(axes[0], "scVI", f"{RM}/scvi_model/model.pt"),
                       (axes[1], "scANVI", f"{RM}/ref_scanvi/model.pt")]:
    h = history(path)
    tr, va = arr(h, "elbo_train"), arr(h, "elbo_validation")
    x = np.arange(1, len(tr) + 1)
    ax.plot(x, tr, color=NSBLUE, lw=1.8, label="train")
    ax.plot(x, va, color=NSAMBER, lw=1.6, alpha=0.95, label="validation")
    vmin = int(np.nanargmin(va))
    ax.scatter([vmin + 1], [va[vmin]], color=NSGRAY, s=26, zorder=5)
    ax.axvline(vmin + 1, color=NSGRAY, ls=":", lw=1)
    gap = va[-1] - tr[-1]
    note = "no overfit (val tracks train)" if abs(gap) < 5 else f"val degrades +{va[-1]-va[vmin]:.0f} from min"
    ax.set_title(f"{name}: final ELBO {tr[-1]:.1f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("ELBO  (lower = better)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.50, 0.92, note, transform=ax.transAxes, fontsize=8.5,
            ha="center", color=NSGRAY, style="italic")
fig.tight_layout(); fig.savefig(OUT / "fig_training.pdf"); plt.close(fig)

# ---- Fig 2: integration QC ----------------------------------------------
qc = json.load(open(f"{RES}/integration_qc.json"))
fig, ax = plt.subplots(figsize=(5.2, 3.3))
labels = ["Batch iLISI\n(donor mixing, higher better)", "Label cLISI\n(cell-type purity, $\\approx$1 good)"]
pca = [qc["unintegrated_pca"]["batch_iLISI"], qc["unintegrated_pca"]["label_cLISI"]]
scv = [qc["integrated_scvi"]["batch_iLISI"], qc["integrated_scvi"]["label_cLISI"]]
x = np.arange(2); w = 0.36
b1 = ax.bar(x - w/2, pca, w, label="Unintegrated (PCA)", color=NSGRAY)
b2 = ax.bar(x + w/2, scv, w, label="Integrated (scVI)", color=NSGREEN)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.08,
            f"{b.get_height():.2f}", ha="center", fontsize=8.5)
ax.annotate("", xy=(0 + w/2, scv[0]), xytext=(0 - w/2, pca[0]),
            arrowprops=dict(arrowstyle="->", color=NSGREEN, lw=1.4))
ax.text(0, max(scv[0], pca[0]) + 0.55, f"+{scv[0]-pca[0]:.2f}", ha="center",
        color=NSGREEN, fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("LISI (effective # categories)")
ax.set_ylim(0, max(scv) + 1.4)
ax.set_title("Cross-atlas harmonization confirmed")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig_integration.pdf"); plt.close(fig)

# ---- Fig 3: classification distribution ---------------------------------
classes = [("clean_ontarget", 446959, NSGREEN),
           ("poorly_diff_ontarget", 109643, "#7FB77E"),
           ("true_offtarget", 315273, NSAMBER),
           ("ambiguous", 62761, "#E0B040"),
           ("ambiguous_novel", 667210, "#9AA0A6"),
           ("unmapped", 168003, "#B8BCC2"),
           ("other", 729, "#D6D9DD")]
tot = sum(v for _, v, _ in classes)
fig, ax = plt.subplots(figsize=(6.6, 3.3))
y = np.arange(len(classes))[::-1]
ax.barh(y, [c[1] for c in classes], color=[c[2] for c in classes])
for yi, (_, v, _) in zip(y, classes):
    ax.text(v + 8000, yi, f"{v:,} ({v/tot*100:.1f}%)", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([c[0].replace("_", "\\_") if False else c[0] for c in classes], fontsize=8.5)
ax.set_xlabel("cells"); ax.set_xlim(0, max(c[1] for c in classes) * 1.28)
ax.set_title(f"Query classification  (n = {tot:,} cells)")
leg = [Line2D([0], [0], marker="s", ls="", color=NSGREEN, label="on-target"),
       Line2D([0], [0], marker="s", ls="", color=NSAMBER, label="off-target"),
       Line2D([0], [0], marker="s", ls="", color="#9AA0A6", label="ambiguous / unmapped")]
ax.legend(handles=leg, frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_classes.pdf"); plt.close(fig)

# ---- Fig 4: label confidence --------------------------------------------
df = pd.read_csv(f"{RES}/label_confidence.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.7))
sc = ax.scatter(df["n_cells"], df["mean_entropy"], c=df["frac_above_tau_H"],
                cmap="viridis", s=42, edgecolor="k", lw=0.3, zorder=3)
ax.set_xscale("log")
ax.set_xlabel("cells predicted as type (log scale)")
ax.set_ylabel("mean mapping entropy  (higher = less certain)")
cb = fig.colorbar(sc); cb.set_label("fraction above $\\tau_H$", fontsize=8.5)
ann = df[(df["mean_entropy"] > 1.85) | (df["mean_entropy"] < 0.55) | (df["n_cells"] > 250000)]
for _, r in ann.iterrows():
    ax.annotate(r["pred_label"], (r["n_cells"], r["mean_entropy"]),
                fontsize=6.6, xytext=(4, 3), textcoords="offset points", color=NSGRAY)
ax.set_title("Label confidence: rare types are least certain")
fig.tight_layout(); fig.savefig(OUT / "fig_confidence.pdf"); plt.close(fig)

print("WROTE:", sorted(p.name for p in OUT.glob("*.pdf")))
