"""
Tier-1 core production run (README execution step 1).

Runs, in order:
  1. sweep_frontier  -> results_frontier.csv   (HEADLINE / Panel A, the Gate-G2 artifact)
  2. sweep_grid      -> results_grid.csv       (rho x delta dissociation, Panel B)
  3. sweep_n         -> results_n.csv          (structural vs practical, Panel C)
then renders the figures.

Each sweep checkpoints its CSV after every run, so partial results survive an interrupt
and can be inspected while later sweeps are still running.
"""
from __future__ import annotations
import sys, time, traceback
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sweep import sweep_frontier, sweep_grid, sweep_n
import plot as P


def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t_all = time.time()

    # ---------------- 1. HEADLINE: feasibility frontier ---------------------
    stamp("=== 1/3 sweep_frontier (headline, Panel A) ===")
    t0 = time.time()
    df_f = sweep_frontier(
        rhos=(0.05, 0.15, 0.25, 0.35, 0.6, 1.0),
        adv_lambdas=(0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
        seeds=(0, 1, 2),
        n=3000, delta=1.0,
        out_csv="results_frontier.csv",
    )
    stamp(f"frontier done: {len(df_f)} runs in {(time.time()-t0)/60:.1f} min")
    # Gate-G2 readout: best achievable balance per rho
    g = df_f.groupby("rho")["joint"].max().sort_index(ascending=False)
    stamp("GATE G2 -- max_lambda min(bio_recovery, batch_removed) by rho:")
    for r, v in g.items():
        stamp(f"    rho={r:<5} joint={v:.3f}")

    # ---------------- 2. rho x delta dissociation ---------------------------
    stamp("=== 2/3 sweep_grid (rho x delta dissociation, Panel B) ===")
    t0 = time.time()
    df_g = sweep_grid(out_csv="results_grid.csv")
    stamp(f"grid done: {len(df_g)} runs in {(time.time()-t0)/60:.1f} min")

    # ---------------- 3. structural vs practical ----------------------------
    stamp("=== 3/3 sweep_n (structural vs practical, Panel C) ===")
    t0 = time.time()
    df_n = sweep_n(out_csv="results_n.csv")
    stamp(f"n-sweep done: {len(df_n)} runs in {(time.time()-t0)/60:.1f} min")

    # ---------------- figures ----------------------------------------------
    # NOTE: plot.make_figure expects df_rho (the older recovery-vs-rho design). The
    # headline here is the FRONTIER (README design note 2), so we assemble the
    # composite from the individual panels instead.
    stamp("=== rendering figures ===")
    for name, fn, arg in [
        ("figure_frontier.png", P.panel_frontier, df_f),
        ("figure_joint_vs_rho.png", P.panel_joint_vs_rho, df_f),
    ]:
        try:
            fn(arg); plt.savefig(name, bbox_inches="tight", dpi=200); plt.close("all")
            stamp(f"wrote {name}")
        except Exception as e:
            stamp(f"FAILED {name}: {e}"); traceback.print_exc()

    try:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        P.panel_frontier(df_f, ax=axes[0, 0])
        P.panel_joint_vs_rho(df_f, ax=axes[0, 1])
        P.panel_b(df_g, ax=axes[1, 0])
        P.panel_c(df_n, ax=axes[1, 1])
        fig.suptitle("Support-overlap identifiability: feasibility frontier, "
                     "rho/delta dissociation, structural vs practical", fontsize=12)
        fig.tight_layout()
        fig.savefig("figure_core.png", bbox_inches="tight", dpi=200)
        plt.close("all")
        stamp("wrote figure_core.png")
    except Exception as e:
        stamp(f"composite failed (non-fatal, individual panels written): {e}")
        traceback.print_exc()

    stamp(f"ALL DONE in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
