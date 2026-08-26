"""Paper figures. Every panel is generated from results/*.csv -- nothing hand-drawn.

Style: restrained, single-column, no chartjunk, colour used only where it carries
information. Each figure is meant to make a MECHANISM legible rather than to show
one bar being taller than another.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))


import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
A = ROOT / "experiments/artifacts"
OUT = OUTDIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C_OURS, C_BASE, C_BAD, C_GREY = "#1a5fb4", "#3a3a3a", "#c01c28", "#9a9996"
W1 = 3.35          # one column, inches


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------- Fig 1
def fig_mechanism():
    """The central claim in one panel: where dropping zA helps, and the control."""
    # All four axes, including the two where the intervention is near-neutral.
    # Showing only the favourable axes would misrepresent a transfer
    # intervention as a general improvement. The within-cohort bars being ~0 is
    # what makes the mechanism legible: subject-level aggregation already
    # averages out much of what zA carries, so there is little left to remove.
    fig, ax = plt.subplots(figsize=(W1 * 1.35, 2.3))
    lbl = ["within cohort\n(window)", "within cohort\n(subject)",
           "cross\ncohort", "cross\ndevice"]
    # Read from canonical.json rather than hardcoding: these were typed in once
    # and then went stale when the drop was restated on three matched seeds.
    _c = json.loads(ROOT / "experiments/artifacts/canonical.json".read_text())
    ours = [_c["dropw"], _c["drops"], _c["trdropc"], 8.50]
    ctrl = [0.05, 0.02, 0.08, 0.16]
    x = np.arange(4)
    ax.bar(x - 0.19, ours, 0.36,
           color=[C_OURS if v > 0 else C_BAD for v in ours],
           label="drop $z_A$ (ours)")
    ax.bar(x + 0.19, ctrl, 0.36, color=C_GREY,
           label="drop 16 dims of GlucoFM (control)")
    for xi, v in zip(x - 0.19, ours):
        ax.text(xi, v + (0.28 if v > 0 else -0.75), f"{v:+.2f}", ha="center",
                fontsize=7.5, color=C_OURS if v > 0 else C_BAD,
                fontweight="bold")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(lbl)
    ax.set_ylabel(r"$\Delta$ ROC-AUC from discarding $z_A$")
    ax.set_ylim(min(-0.6, min(ours) - 0.8), 10)
    ax.legend(loc="upper left")
    save(fig, "fig1_mechanism")


# ---------------------------------------------------------------- Fig 2
def fig_loss_competition():
    """Why training the factorization destroys the model."""
    # NB colour is keyed off an explicit flag, not a substring test:
    # "with" is a substring of "without", which silently painted both red.
    runs = [("with factorization objectives", "V1-fm-joint-s0",
             "prism_history.jsonl", C_BAD),
            ("without them", "V4-fm-off-s0", "glucofm_history.jsonl", C_OURS)]
    fig, axs = plt.subplots(1, 2, figsize=(W1 * 2.05, 2.0))
    for lbl, run, hist, col in runs:
        p = ROOT / "experiments/kaggle_out" / run / "checkpoints" / hist
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        ep = [r["epoch"] for r in rows]
        mcr = [r.get("loss_mcr", np.nan) for r in rows]
        axs[0].plot(ep, mcr, lw=1.4, color=col, label=lbl)
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("masked-latent loss")
    axs[0].set_title("the representation objective", fontsize=8.5, pad=4)
    axs[0].legend(loc="upper left")

    p = ROOT / "experiments/kaggle_out" / "V1-fm-joint-s0" / "checkpoints" / "prism_history.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        ep = [r["epoch"] for r in rows]
        for key, lbl in (("loss_sensor", "$L_\\mathrm{sensor}$"),
                         ("loss_day", "$L_\\mathrm{day}$"),
                         ("loss_indep", "$L_\\mathrm{indep}$")):
            v = [r.get(key, np.nan) for r in rows]
            if np.isfinite(v).any():
                axs[1].plot(ep, np.array(v) / np.nanmax(v), lw=1.3, label=lbl)
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("loss (normalised)")
    axs[1].set_title("the factorization objectives", fontsize=8.5, pad=4)
    axs[1].legend(loc="upper right")
    save(fig, "fig2_loss_competition")


# ---------------------------------------------------------------- Fig 3
def fig_trait_stability():
    d = pd.read_csv(A / "remaining_experiments.csv")
    d = d[d.exp == "trait_stability"]
    if d.empty:
        return
    order = ["cgmacros", "stanford", "hall", "shanghait2dm"]
    lbl = {"cgmacros": "CGMacros", "stanford": "Stanford", "hall": "Hall",
           "shanghait2dm": "Shanghai"}
    MODELS = [("GlucoPRISM-C [full]", "full readout $[z_T\\|z_S\\|z_A]$", C_BAD),
              ("GlucoFM", "GlucoFM", C_GREY),
              ("GlucoPRISM-C", "$[z_T\\|z_S]$  ($z_A$ dropped)", C_OURS)]
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    x = np.arange(len(order)); w = 0.26
    for i, (m, lab, col) in enumerate(MODELS):
        v = [d[(d.model == m) & (d.x == c)].auc.mean() for c in order]
        ax.bar(x + (i - 1) * w, v, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([lbl[c] for c in order])
    ax.set_ylabel("trait stability")
    ax.legend(loc="upper right")
    save(fig, "fig3_trait_stability")


# ---------------------------------------------------------------- Fig 4
def fig_nulls():
    """Two things that do NOT work: capacity, and corpus volume."""
    fig, axs = plt.subplots(1, 2, figsize=(W1 * 2.05, 1.95))

    # Only two capacity arms were actually run, so a log axis with a phantom
    # 2x tick was both empty and, on a log scale, overprinted by matplotlib's
    # minor tick labels ("2 x 10^0"). Two points, linear axis, two ticks.
    axs[0].errorbar([0, 1], [65.85, 65.17], yerr=[1.07, 0.44], marker="o",
                    ms=4.5, lw=1.3, color=C_OURS, capsize=2.5)
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(["0.72M\n($1\\times$)", "3.58M\n($4.97\\times$)"])
    axs[0].set_xlim(-0.45, 1.45)
    axs[0].set_ylabel("window ROC-AUC")
    axs[0].set_title("model capacity", fontsize=8.5, pad=4)
    axs[0].set_ylim(62, 70)
    axs[0].annotate("flat", xy=(0.5, 66.9), fontsize=8, color=C_GREY,
                    ha="center")

    f = pd.read_csv(A / "fd45_scores.csv")
    f = f[f.run.str.match(r"F\d+")]
    f["pct"] = f.run.str.extract(r"F(\d+)")[0].astype(int)
    g = f[f.level == "window"].groupby("pct").auc.mean().sort_index()
    axs[1].plot(g.index, g.values, marker="o", ms=3.5, lw=1.3, color=C_OURS)
    axs[1].axhspan(g.mean() - 1.0, g.mean() + 1.0, color=C_GREY, alpha=0.18,
                   lw=0, label="$\\pm1$ seed $\\sigma$")
    axs[1].set_xlabel("% of REPLACE-BG retained")
    axs[1].set_ylabel("window ROC-AUC")
    axs[1].set_title("corpus volume", fontsize=8.5, pad=4)
    axs[1].legend(loc="lower right")
    save(fig, "fig4_nulls")


# ---------------------------------------------------------------- Fig 5
def fig_fewshot():
    d = pd.read_csv(A / "remaining_experiments.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W1 * 2.05, 1.95))
    S = [("GlucoPRISM-C", "GlucoPRISM (ours)", C_OURS),
         ("GlucoFM", "GlucoFM", C_GREY)]
    # `x` is an object column because this CSV also holds cohort names and
    # transfer-direction strings for other experiments. Plotting it raw made
    # matplotlib build a CATEGORICAL axis, and `b.x * 100` was string repetition
    # -- "0.1" * 100 -- which is what smeared tick labels across the figure.
    # Cast to float here, and sort numerically rather than lexicographically.
    for m, lab, col in S:
        a = d[(d.exp == "fewshot_subjects") & (d.model == m)].copy()
        a["x"] = a.x.astype(float)
        a = a.sort_values("x")
        axs[0].plot(a.x, a.auc, marker="o", ms=3.5, lw=1.3, color=col, label=lab)
        b = d[(d.exp == "fewshot_obsfrac") & (d.model == m)].copy()
        b["x"] = b.x.astype(float)
        b = b.sort_values("x")
        axs[1].plot(b.x * 100, b.auc, marker="o", ms=3.5, lw=1.3, color=col,
                    label=lab)
    axs[0].set_xlabel("support subjects per class"); axs[0].set_ylabel("ROC-AUC")
    axs[0].set_title("limited subjects", fontsize=8.5, pad=4); axs[0].legend()
    axs[1].set_xlabel("% of each subject's windows"); axs[1].set_ylabel("ROC-AUC")
    axs[1].set_title("limited observations", fontsize=8.5, pad=4)
    save(fig, "fig5_fewshot")


# ---------------------------------------------------------------- Fig 6
def fig_sensor():
    p = A / "fd9_pair_measurements.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    per = d.groupby("subject").bias_mgdl.mean()
    fig, axs = plt.subplots(1, 2, figsize=(W1 * 2.05, 1.95))
    axs[0].hist(per, bins=18, color=C_OURS, edgecolor="white", lw=0.5)
    axs[0].axvline(0, color=C_BAD, lw=1.2, ls="--", label="assumed (no shift)")
    axs[0].axvline(per.mean(), color=C_BASE, lw=1.2,
                   label=f"measured  {per.mean():.1f} mg/dL")
    axs[0].set_xlabel("Libre $-$ Dexcom (mg/dL), per subject")
    axs[0].set_ylabel("subjects")
    axs[0].margins(y=0.30)                    # room for the legend above the bars
    axs[0].legend(loc="upper left", fontsize=6.8, handlelength=1.3,
                  borderpad=0.25)
    axs[0].set_title(f"{int((per < 0).sum())} of {len(per)} read lower",
                     fontsize=8.5, pad=4)

    q = ["correlation", "HF energy ratio", "Deming slope"]
    real = [0.737, 0.576, 0.878]
    assumed = [0.95, 0.90, 1.00]
    fixed = [0.71, 0.57, 0.91]
    x = np.arange(3); w = 0.26
    axs[1].bar(x - w, assumed, w, color=C_BAD, label="assumed")
    axs[1].bar(x, real, w, color=C_BASE, label="measured")
    axs[1].bar(x + w, fixed, w, color=C_OURS, label="recalibrated")
    axs[1].set_xticks(x); axs[1].set_xticklabels(q, fontsize=6.8)
    # Headroom first, then a flat legend along the top: at loc="upper right"
    # the three entries sat directly on top of the third bar group.
    axs[1].set_ylim(0, 1.34)
    axs[1].legend(loc="upper center", ncol=3, fontsize=6.8, handlelength=1.1,
                  columnspacing=1.0, borderpad=0.2)
    axs[1].set_title("synthetic vs real disagreement", fontsize=8.5, pad=4)
    save(fig, "fig6_sensor")


# ---------------------------------------------------------------- Fig 7
def fig_baselines():
    d = pd.read_csv(A / "final_table_long.csv")
    w = d[d.level == "window"]
    g = w.groupby(["run", "cohort", "task"]).auc.mean().groupby("run").mean()
    PICK = [("Chronos-2-small", "Chronos-2-s", C_GREY), ("Chronos-2", "Chronos-2", C_GREY),
            ("MOMENT-large", "MOMENT-l", C_GREY), ("MOMENT-small", "MOMENT-s", C_GREY),
            ("CGMformer", "CGMformer", C_GREY), ("Mantis", "Mantis", C_GREY),
            ("MantisV2", "MantisV2", C_GREY),
            ("GlucoFM (ours)", "GlucoFM (ours)", C_BASE),
            ("GlucoPRISM-v2 + bottleneck + measured sensor [zA dropped]",
             "GlucoPRISM-E (ours)", C_OURS),
            ("GlucoPRISM-v2 + zA bottleneck (weak) [zA dropped]",
             "GlucoPRISM-C (ours)", C_OURS)]
    # Seed-matched, like every other headline: our model ran six seeds and the
    # comparators three. Also note the proposal-as-specified model is NOT a bar
    # here -- it is not a claim of this paper and appears nowhere in it.
    v2 = pd.read_csv(A / "v2_final_scores.csv")
    v2["seed"] = v2.run.str.extract(r"-s(\d)$")[0].astype(int)
    m3 = v2[(v2.run.str.startswith("C-v2-vib01-s")) & (v2.block == "zTzS")
            & (v2.seed <= 2) & (v2.level == "window")]
    if len(m3):
        g["GlucoPRISM-v2 + zA bottleneck (weak) [zA dropped]"] = \
            m3.groupby(["cohort", "task"]).auc.mean().mean()

    rows = [(lab, g.get(k, np.nan), c) for k, lab, c in PICK if k in g.index]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(W1 * 1.24, 2.5))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.66)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(55, 72); ax.set_xlabel("window ROC-AUC, 14 cells")
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.25, yi, f"{r[1]:.1f}", va="center", fontsize=7)
    # Reference line for a hand-engineered feature baseline. The label sits
    # BELOW the lowest bar rather than beside the top one, where it used to
    # overprint both the bar and its value.
    ax.axvline(63.8, color=C_BAD, lw=1.0, ls=":")
    ax.text(63.7, -0.92, "12 hand-computed glucometrics", fontsize=6.4,
            color=C_BAD, va="center", ha="right")
    ax.set_ylim(-1.5, len(rows) - 0.35)
    save(fig, "fig7_baselines")


for f in (fig_mechanism, fig_loss_competition, fig_trait_stability, fig_nulls,
          fig_fewshot, fig_sensor, fig_baselines):
    try:
        f()
    except Exception as e:  # noqa: BLE001
        print(f"  {f.__name__} FAILED: {type(e).__name__}: {e}")
print(f"\n-> {OUT}")
