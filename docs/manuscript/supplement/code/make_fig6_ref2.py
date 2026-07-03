#!/usr/bin/env python
"""Figure 6: the second-reference test (reviewer point 3), clean-calibrated framing.

The organoid->independent-reference displacement is SHARED by clean and ambiguous-novel
cells (the clean positive control quantifies it); the clean-vs-ambnov contrast cancels that
shared source gap and isolates genuine ADDITIONAL divergence of the ambiguous-novel cells.
Reads results/pilot/ref2_scores.npz (ref_self, cln_off, amb_off) from scripts/ref2_test.py.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

RES = Path("/workspace/NullState/results/pilot")
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.6, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42})
C_REF, C_CLEAN, C_AMB = "#969696", "#2c7fb8", "#d7301f"

d = np.load(RES / "ref2_scores.npz")
ref_self, cln, amb = d["ref_self"], d["cln_off"], d["amb_off"]
U, p = mannwhitneyu(amb, cln, alternative="greater"); auc = U / (len(amb) * len(cln))
c50, c95 = np.percentile(cln, 50), np.percentile(cln, 95)
qs = [50, 75, 90, 95]; beyond = [100 * np.mean(amb > np.percentile(cln, q)) for q in qs]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.34})

# (a) off-manifold score distributions: all organoid displaced from ref2, ambnov more than clean
hi = np.quantile(np.concatenate([ref_self, amb]), 0.99)
bins = np.linspace(0, hi, 60)
for x, c, lab in [(ref_self, C_REF, "independent ref (self)"), (cln, C_CLEAN, "clean neural (organoid)"),
                  (amb, C_AMB, "ambiguous-novel neural")]:
    axa.hist(x, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=c, lw=1.1, edgecolor=c, label=lab)
ymax = max(np.histogram(ref_self, bins=bins, density=True)[0].max(),
           np.histogram(amb, bins=bins, density=True)[0].max()) * 1.18
axa.set_ylim(0, ymax)
axa.axvline(c50, color=C_CLEAN, ls="--", lw=0.9); axa.axvline(c95, color=C_CLEAN, ls=":", lw=0.9)
axa.text(c50, ymax * 0.40, "clean median", rotation=90, fontsize=5.6, color=C_CLEAN, va="center", ha="right")
axa.text(c95, ymax * 0.40, "clean p95", rotation=90, fontsize=5.6, color=C_CLEAN, va="center", ha="right")
axa.set_xlabel("off-manifold score vs independent cortex atlas  (kNN distance)")
axa.set_ylabel("density"); axa.legend(frameon=False, loc="upper right", fontsize=6.2)
axa.set_title("All organoid cells displaced; ambiguous-novel further out", fontsize=8)
axa.text(-0.12, 1.05, "a", transform=axa.transAxes, fontsize=11, fontweight="bold", va="top")

# (b) clean-calibrated excess divergence (source gap cancelled)
axb.bar(np.arange(len(qs)), beyond, color=C_AMB, edgecolor="white", width=0.66)
for i, v in enumerate(beyond):
    axb.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=6.6, fontweight="bold")
axb.set_xticks(np.arange(len(qs))); axb.set_xticklabels([f"clean\np{q}" for q in qs], fontsize=6.6)
axb.set_ylabel("% ambiguous-novel beyond clean"); axb.set_ylim(0, 80)
axb.set_title(f"Excess divergence  (AUC={auc:.2f}, $\\delta$={2*auc-1:.2f})", fontsize=8)
axb.text(-0.22, 1.05, "b", transform=axb.transAxes, fontsize=11, fontweight="bold", va="top")

fig.savefig(OUT / "fig6.pdf"); fig.savefig(OUT / "fig6.png")
# persist the stats for the manuscript text
json.dump({"auc": float(auc), "cliffs_delta": float(2 * auc - 1), "p_value": float(p),
           "median_refself": float(np.median(ref_self)), "median_clean": float(np.median(cln)),
           "median_ambnov": float(np.median(amb)),
           "ambnov_beyond_clean_p50": float(beyond[0] / 100), "ambnov_beyond_clean_p95": float(beyond[3] / 100),
           "n_clean": int(len(cln)), "n_ambnov": int(len(amb))},
          open(OUT.parent / "ref2_stats.json", "w"), indent=2)
print(f"fig6 ok: AUC={auc:.3f} delta={2*auc-1:.3f} beyond_p50={beyond[0]:.0f}% beyond_p95={beyond[3]:.0f}%")
