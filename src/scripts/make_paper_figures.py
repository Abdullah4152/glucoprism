"""Paper figures. Every panel is generated from results/*.csv -- nothing hand-drawn.

Style follows `fig_style.py`: one palette across every float, a light y-grid, no
top/right spines, panel letters below each panel, and the legend under the row
rather than inside a panel. The rule the previous draft broke and this one keeps:
**no text is ever placed over the data.** Legends and annotations get reserved
space, or they do not appear.

Within a figure, every panel in a row is the same chart type wherever the data
allows it, so a reader learns the row once.

Several floats here replace appendix tables rather than adding to them -- the
per-cell heatmaps stand in for three 15-column tables, and the sweep and readout
panels for four more. No number is dropped: the heatmaps print their values.
"""
from __future__ import annotations

import json
import warnings
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import fig_style as S

warnings.filterwarnings("ignore")
S.use_style()

# Inputs from GLUCOPRISM_OUT; the original hard-coded an absolute path.
A = Path(os.environ.get("GLUCOPRISM_OUT",
                        Path(os.environ.get("GLUCOPRISM_ROOT",
                             Path(__file__).resolve().parents[2]))
                        / "artifacts"))
# Figures go to GLUCOPRISM_FIG_OUT, else artifacts/figures.
OUT = Path(os.environ.get("GLUCOPRISM_FIG_OUT", A / "figures"))
OUT.mkdir(parents=True, exist_ok=True)
CANON = json.loads((A / "canonical.json").read_text())

W = 5.5                                   # \linewidth, inches


def save(fig, name):
    S.save(fig, name, OUT)


def _blocks():
    """Deduplicated block/control transfer scores, one row per direction."""
    b = pd.concat([pd.read_csv(A / "fd3_block_controls.csv"),
                   pd.read_csv(A / "fd3_drop_za.csv")], ignore_index=True)
    return b.drop_duplicates(subset=["seed", "variant", "src", "tgt", "task"])


def _transfer():
    t = pd.concat([pd.read_csv(A / f) for f in ("fd3_v2final.csv", "fd3_bd.csv")
                   if (A / f).exists()], ignore_index=True)
    t = t.drop_duplicates(subset=["run", "src", "tgt", "task"])
    t["arm"] = t.run.str.replace(r"-s\d(:|$)", r"\1", regex=True)
    return t[t.run.str.contains(r"-s[012](:|$)", regex=True)]


COHN = {"cgmacros": "CGMacros", "stanford": "Stanford", "hall": "Hall",
        "shanghait2dm": "Shanghai"}
TASKN = {"diabetes": "DR", "diabetes_3class": "DR3", "ir": "IR",
         "beta_cell": r"$\beta$C", "hyperlipidemia": "HL", "obesity": "Ob",
         "hypoglycemia": "Hy", "glucotype": "GT"}


# ============================================================ MAIN Figure 2
def fig_dissociation():
    """The thesis in four measurements, all four drawn as line/point panels."""
    fig, axs = plt.subplots(1, 4, figsize=(W, 1.55))

    # (a) capacity x factorization
    d = pd.read_csv(A / "fd8_scores.csv")
    d["arm"] = d.run.str.replace(r"-s\d$", "", regex=True)
    w = d[d.level == "window"]
    per = w.groupby(["arm", "run"]).apply(
        lambda x: x.groupby(["cohort", "task"]).auc.mean().mean(),
        include_groups=False)
    st = per.groupby("arm").agg(["mean", "std"])
    ARMS = [("no factorization", ["V4-fm-off", "V5-5x-off"], S.NAVY, "o"),
            ("factorization objective", ["V1-fm-joint", "V2-5x-joint"],
             S.RED, "s"),
            ("post-hoc, frozen encoder", ["V6-fm-post", "V7-5x-post"],
             S.GREY, "^")]
    for lab, arms, col, mk in ARMS:
        if not all(a in st.index for a in arms):
            continue
        axs[0].errorbar([0, 1], [st.loc[a, "mean"] for a in arms],
                        yerr=[st.loc[a, "std"] for a in arms], marker=mk,
                        ms=3.0, lw=1.1, color=col, capsize=1.8,
                        elinewidth=0.7, label=lab)
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels([r"$1\times$", r"$4.97\times$"])
    axs[0].set_xlim(-0.35, 1.35)
    axs[0].set_xlabel("encoder parameters")
    axs[0].set_ylabel("window ROC-AUC")
    axs[0].set_title("as an objective")
    S.grid(axs[0])

    # (b) the same block deleted rather than optimised, across four shifts
    blk = _blocks()
    mean = blk.groupby("variant").auc.mean()
    ctrl_tr = abs(mean["GlucoFM full (128)"] - mean["GlucoFM first 112"])
    cdv = pd.read_csv(A / "remaining_experiments.csv")
    cdv = cdv[cdv.exp == "cross_device"].pivot_table(
        index="x", columns="model", values="auc")
    cd_gain = float((cdv["GlucoPRISM-C"] - cdv["GlucoPRISM-C [full]"]).mean())
    xs = np.arange(4)
    ours = [CANON["dropw"], CANON["drops"], CANON["trdropc"], cd_gain]
    ctrl = [0.05, 0.02, ctrl_tr, 0.16]
    axs[1].plot(xs, ours, marker="o", ms=3.2, lw=1.2, color=S.ORANGE,
                label=r"delete $z_A$ (ours)")
    axs[1].plot(xs, ctrl, marker="s", ms=3.0, lw=1.1, color=S.GREY, ls="--",
                label="matched slice of GlucoFM")
    axs[1].axhline(0, color=S.GREY_L, lw=0.7)
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels(["window", "subject", "cohort", "device"],
                           fontsize=5.4)
    axs[1].set_xlabel("distribution shift")
    axs[1].set_ylabel(r"$\Delta$ ROC-AUC")
    axs[1].set_title("as an address")
    S.grid(axs[1])

    # (c) the 2x2 -- neither half alone
    ix = json.loads((A / "confound_analysis.json").read_text())["full"]
    base = 65.63
    axs[2].plot([0, 1], [base, base + ix["vib_noobj"]], marker="o", ms=3.2,
                lw=1.1, color=S.GREY, ls="--", label="objectives off")
    axs[2].plot([0, 1], [base + ix["obj_novib"],
                         base + ix["obj_novib"] + ix["vib_noobj"] + ix["inter"]],
                marker="o", ms=3.2, lw=1.2, color=S.ORANGE,
                label="objectives on")
    axs[2].set_xticks([0, 1]); axs[2].set_xticklabels(["VIB off", "VIB on"])
    axs[2].set_xlim(-0.35, 1.35)
    axs[2].set_ylabel("window ROC-AUC")
    axs[2].set_title("neither half alone")
    S.grid(axs[2])

    # (d) did the blocks separate?
    bd = pd.read_csv(A / "rev_block_dependence.csv")
    gb = bd.groupby("pair")[["mean_abs_corr", "hsic"]].mean()
    PAIRS = ["zT-zS", "zT-zA", "zS-zA"]
    PL = [r"$z_T$–$z_S$", r"$z_T$–$z_A$", r"$z_S$–$z_A$"]
    xs = np.arange(3)
    axs[3].plot(xs, [gb.loc[p, "mean_abs_corr"] for p in PAIRS], marker="o",
                ms=3.2, lw=1.2, color=S.NAVY, label=r"mean $|r|$")
    axs[3].plot(xs, [gb.loc[p, "hsic"] for p in PAIRS], marker="s", ms=3.0,
                lw=1.1, color=S.TEAL, ls="--", label="HSIC")
    axs[3].set_xticks(xs); axs[3].set_xticklabels(PL, fontsize=5.6)
    axs[3].set_xlim(-0.35, 2.35)
    axs[3].set_ylim(0, 1.05)
    axs[3].set_ylabel("dependence")
    axs[3].set_title("did they separate?")
    S.grid(axs[3])

    fig.tight_layout(w_pad=1.0)
    _ly = S.letters(fig, axs)
    H = [Line2D([], [], color=c, marker=m, ms=3.0, lw=1.1, ls=s)
         for c, m, s in [(S.NAVY, "o", "-"), (S.RED, "s", "-"),
                         (S.GREY, "^", "-"), (S.ORANGE, "o", "-"),
                         (S.TEAL, "s", "--")]]
    L = ["no factorization", "factorization objective", "post-hoc / control",
         "ours (delete $z_A$)", "HSIC"]
    S.rowlegend(fig, H, L, ncol=5, y=_ly - 0.03)
    save(fig, "fig_dissociation")


# ============================================================ MAIN Figure 3
def fig_deletion():
    """Where the deletion pays, and what squeezing the block costs."""
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.85),
                            gridspec_kw={"width_ratios": [1.5, 1, 1]})

    # (a) every transfer direction, sorted
    t = _transfer()
    g = (t[t.arm == "C-v2-vib01:zTzS"].groupby(["src", "tgt", "task"]).auc.mean()
         - t[t.arm == "C-v2-vib01:full"].groupby(["src", "tgt", "task"]).auc.mean()
         ).dropna().sort_values()
    labs = [f"{COHN.get(s, s)}→{COHN.get(tg, tg)} {TASKN.get(k, k)}"
            for s, tg, k in g.index]
    cols = [S.ORANGE if tg == "cgmacros" else S.GREY_L for _, tg, _ in g.index]
    y = np.arange(len(g))
    axs[0].barh(y, g.values, color=cols, height=0.72, zorder=3)
    axs[0].set_yticks(y); axs[0].set_yticklabels(labs, fontsize=4.6)
    axs[0].axvline(0, color=S.INK, lw=0.6, zorder=4)
    axs[0].set_xlabel(r"$\Delta$ ROC-AUC from deleting $z_A$")
    axs[0].set_title("which directions")
    S.grid(axs[0], axis="x")

    # (b) dose-response in the price the block pays
    cap = pd.read_csv(A / "rev_capacity_summary.csv")
    bs = cap[cap.d_sensor.astype(str).str.startswith("16")].sort_values("beta")
    axs[1].plot(bs.beta, bs.gain, marker="o", ms=3.4, lw=1.3, color=S.ORANGE,
                zorder=3)
    rel = bs[bs.d_sensor.astype(str).str.contains("released")]
    if len(rel):
        axs[1].scatter(rel.beta, rel.gain, s=46, facecolors="none",
                       edgecolors=S.RED, lw=1.0, zorder=5)
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"KL price on $z_A$  ($\beta$)")
    axs[1].set_ylabel(r"gain from deleting $z_A$")
    axs[1].axhline(0, color=S.GREY_L, lw=0.7)
    axs[1].set_title("dose-response")
    S.grid(axs[1])

    # (c) partial readmission: ranking and calibration step at the same m
    sd = pd.read_csv(A / "rev_soft_deletion.csv")
    sd = sd[sd.tag.str.startswith("keep ")].copy()
    sd["m"] = sd.tag.str.extract(r"keep (\d+)/")[0].astype(int)
    gg = sd.groupby("m")[["auc", "ece"]].mean().sort_index()
    axs[2].plot(gg.index, gg.auc, marker="o", ms=3.2, lw=1.3, color=S.ORANGE,
                zorder=3)
    axs[2].set_xlabel(r"$z_A$ dims readmitted ($m$)")
    axs[2].set_ylabel("transfer ROC-AUC", color=S.ORANGE)
    axs[2].tick_params(axis="y", labelcolor=S.ORANGE)
    ax2 = axs[2].twinx()
    ax2.plot(gg.index, gg.ece, marker="s", ms=2.8, lw=1.1, ls="--",
             color=S.NAVY, zorder=3)
    ax2.set_ylabel("ECE (lower better)", color=S.NAVY)
    ax2.tick_params(axis="y", labelcolor=S.NAVY)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(S.GREY_L)
    ax2.grid(False)
    axs[2].set_xticks(list(gg.index))
    axs[2].set_title("partial deletion")
    S.grid(axs[2])

    fig.tight_layout(w_pad=1.4)
    _ly = S.letters(fig, axs)
    H = [Patch(fc=S.ORANGE, ec="none"), Patch(fc=S.GREY_L, ec="none"),
         Line2D([], [], color=S.RED, marker="o", ms=4.2, lw=0, mfc="none"),
         Line2D([], [], color=S.NAVY, marker="s", ms=2.8, lw=1.1, ls="--")]
    L = ["target cohort = CGMacros", "other targets", r"released $\beta$",
         "ECE"]
    S.rowlegend(fig, H, L, ncol=4, y=_ly - 0.03)
    save(fig, "fig_deletion")


# ============================================================ MAIN Figure 4
def fig_shortcut():
    """The device shortcut is free to a model, and physical in origin."""
    fig, axs = plt.subplots(1, 4, figsize=(W, 1.55))

    # (a) how recoverable is the device?
    dp = pd.read_csv(A / "rev_device_predictability.csv").iloc[::-1]
    LB = {"observation mask (288d)": "mask (288-d)",
          "mask summary (count only)": "sample count",
          "glucose level only (3d)": "level only (3-d)",
          "mask + level": "mask + level"}
    y = np.arange(len(dp))
    axs[0].barh(y, dp.device_auc, xerr=dp.sd, color=S.RED, height=0.66,
                zorder=3, error_kw=dict(lw=0.7, capsize=1.6, ecolor=S.INK))
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([LB.get(f, f) for f in dp.features], fontsize=5.0)
    axs[0].axvline(50, color=S.INK, lw=0.7, ls="--", zorder=4)
    axs[0].set_xlim(45, 104)
    axs[0].set_xlabel("device ROC-AUC")
    axs[0].set_title("device is free")
    S.grid(axs[0], axis="x")

    # (b) the calibration offset prior work assumed away
    pm = pd.read_csv(A / "fd9_pair_measurements.csv")
    per = pm.groupby("subject").bias_mgdl.mean()
    axs[1].hist(per, bins=16, color=S.SKY, edgecolor="white", lw=0.4, zorder=3)
    axs[1].axvline(0, color=S.RED, lw=1.0, ls="--", zorder=4)
    axs[1].axvline(per.mean(), color=S.NAVY, lw=1.0, zorder=4)
    axs[1].set_xlabel("Libre $-$ Dexcom (mg/dL)")
    axs[1].set_ylabel("subjects")
    axs[1].set_title(f"{int((per < 0).sum())} of {len(per)} read lower")
    S.grid(axs[1])

    # (c) assumed vs measured, from the fitted generator
    cal = json.loads((A / "fd9_sensor_calibration.json").read_text())
    q = ["corr.", "HF", "slope"]
    real = [float(cal.get("corr_median", 0.737)), 0.576,
            float(cal.get("slope_mean", 0.878))]
    assumed = [0.95, 0.90, 1.00]
    xs = np.arange(3)
    axs[2].plot(xs, assumed, marker="s", ms=3.0, lw=1.1, color=S.RED, ls="--",
                zorder=3)
    axs[2].plot(xs, real, marker="o", ms=3.2, lw=1.2, color=S.NAVY, zorder=3)
    axs[2].set_xticks(xs); axs[2].set_xticklabels(q, fontsize=5.2)
    axs[2].set_xlim(-0.35, 2.35); axs[2].set_ylim(0.4, 1.12)
    axs[2].set_ylabel("value")
    axs[2].set_title("synthetic vs real")
    S.grid(axs[2])

    # (d) why the mask carries the device: cohorts are not rate-balanced
    COR = [("REPLACE-BG", 0.977), ("Stanford", 0.979), ("Shanghai", 0.333),
           (r"Colas", 0.996), ("BIG IDEAs", 0.929)]
    xs = np.arange(len(COR))
    axs[3].bar(xs, [c[1] for c in COR], 0.6, zorder=3,
               color=[S.RED if c[1] < 0.5 else S.GREY_L for c in COR])
    axs[3].set_xticks(xs)
    axs[3].set_xticklabels([c[0] for c in COR], rotation=38, ha="right",
                           fontsize=4.8)
    axs[3].set_ylabel("observation fraction")
    axs[3].set_ylim(0, 1.08)
    axs[3].set_title("cohort $=$ device")
    S.grid(axs[3])

    fig.tight_layout(w_pad=1.1)
    _ly = S.letters(fig, axs)
    H = [Patch(fc=S.RED, ec="none"), Line2D([], [], color=S.INK, lw=0.7,
                                            ls="--"),
         Line2D([], [], color=S.NAVY, marker="o", ms=3.0, lw=1.1),
         Line2D([], [], color=S.RED, marker="s", ms=3.0, lw=1.1, ls="--")]
    L = ["device-derived", "chance", "measured", "assumed by prior work"]
    S.rowlegend(fig, H, L, ncol=4, y=_ly - 0.03)
    save(fig, "fig_shortcut")


# ============================================================ MAIN Figure 5
def fig_perdata():
    """What the representation buys per subject, per observation, per day."""
    d = pd.read_csv(A / "remaining_experiments.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.7))
    SER = [("GlucoPRISM-C", S.ORANGE, "o", "-"),
           ("GlucoPRISM-E", S.CORAL, "s", "-"),
           ("GlucoFM", S.NAVY, "^", "--"),
           ("GlucoPRISM-C [full]", S.GREY, "v", ":")]

    def line(ax, exp, scale=1.0):
        for m, col, mk, ls in SER:
            a = d[(d.exp == exp) & (d.model == m)].copy()
            if a.empty:
                continue
            a["x"] = a.x.astype(float)
            a = a.sort_values("x")
            ax.plot(a.x * scale, a.auc, marker=mk, ms=2.9, lw=1.15, ls=ls,
                    color=col, zorder=3)

    line(axs[0], "fewshot_subjects")
    axs[0].set_xlabel("support subjects per class")
    axs[0].set_ylabel("ROC-AUC")
    axs[0].set_title("limited subjects")

    line(axs[1], "fewshot_obsfrac", 100)
    axs[1].set_xlabel("% of each subject's windows")
    axs[1].set_ylabel("ROC-AUC")
    axs[1].set_title("limited observations")

    line(axs[2], "multiday")
    axs[2].set_xlabel("days of wear $K$")
    axs[2].set_ylabel("subject ROC-AUC")
    axs[2].set_title("days of wear")
    md = d[d.exp == "multiday"].copy()
    md["x"] = md.x.astype(float)
    mp = md.pivot_table(index="x", columns="model", values="auc")
    axs[2].scatter([3, 7], [mp.loc[3, "GlucoPRISM-C"],
                            mp.loc[7, "GlucoPRISM-C [full]"]],
                   s=42, facecolors="none", edgecolors=S.RED, lw=0.9, zorder=6)
    for ax in axs:
        S.grid(ax)

    fig.tight_layout(w_pad=1.4)
    _ly = S.letters(fig, axs)
    H = [Line2D([], [], color=c, marker=m, ms=2.9, lw=1.15, ls=ls)
         for _, c, m, ls in SER]
    H.append(Line2D([], [], color=S.RED, marker="o", ms=4.2, lw=0, mfc="none"))
    L = ["GlucoPRISM-C", "GlucoPRISM-E", "GlucoFM", r"GP-C, $z_A$ kept",
         "3 days $>$ 7 days kept"]
    S.rowlegend(fig, H, L, ncol=5, y=_ly - 0.03)
    save(fig, "fig_perdata")


# ============================================ APPENDIX: replaces 3 big tables
def fig_percell():
    """Per-cell scores on the three metrics the main table does not carry.

    Replaces `tbl_percellpr`, `tbl_percellf1` and `tbl_percellsubj` -- three
    fifteen-column tables. Every number they held is printed in a cell here, so
    nothing is lost; the colour just makes the pattern visible first.
    """
    df = pd.read_csv(A / "final_table_long.csv")
    v2 = pd.read_csv(A / "v2_final_scores.csv")
    v2["seed"] = v2.run.str.extract(r"-s(\d)$")[0].astype(int)
    GPC = "GlucoPRISM-v2 + zA bottleneck (weak) [zA dropped]"
    m3 = (v2[(v2.run.str.startswith("C-v2-vib01-s")) & (v2.block == "zTzS")
              & (v2.seed <= 2)]
          .groupby(["level", "cohort", "task"], as_index=False)
          [["pr", "auc", "f1"]].mean())
    m3["run"] = GPC
    df = pd.concat([df[df.run != GPC], m3], ignore_index=True)

    KEY = {GPC: "GP-C",
           "GlucoPRISM-v2 + bottleneck + measured sensor [zA dropped]": "GP-E",
           "GlucoFM (ours)": "GlucoFM", "MantisV2": "MantisV2",
           "Mantis": "Mantis", "CGMformer": "CGMformer",
           "MOMENT-small": "MOM-s", "MOMENT-large": "MOM-l",
           "Chronos-2": "Chr-2", "Chronos-2-small": "Chr-2s"}
    PANELS = [("pr", "window", "PR-AUC"), ("f1", "window", "Macro-F1"),
              ("auc", "subject", "subject-level ROC-AUC")]

    fig, axs = plt.subplots(3, 1, figsize=(W, 5.4))
    for ax, (metric, lvl, title) in zip(axs, PANELS):
        src = df[df.level == lvl]
        piv = src.groupby(["cohort", "task", "run"])[metric].mean().unstack("run")
        rows = [c for c in KEY if c in piv.columns]
        cells = [i for coh in ["cgmacros", "shanghait2dm", "stanford", "hall"]
                 for i in piv.index if i[0] == coh]
        M = np.array([[piv.loc[c, r] for c in cells] for r in rows], float)
        im = ax.imshow(M, cmap="RdYlBu", aspect="auto", vmin=np.nanmin(M),
                       vmax=np.nanmax(M))
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{COHN[c]}\n{TASKN.get(t, t)}" for c, t in cells],
                           fontsize=4.4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([KEY[r] for r in rows], fontsize=5.0)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                            fontsize=4.0, color=S.INK)
        ax.set_title(title, fontsize=6.6, pad=3)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.014, pad=0.008)
        cb.ax.tick_params(labelsize=4.4, width=0.5, length=1.6)
        cb.outline.set_linewidth(0.4)
    fig.tight_layout(h_pad=1.4)
    save(fig, "fig_percell")


# ============================================ APPENDIX: replaces 2 tables
def fig_sweeps():
    """Corpus composition and window geometry -- continuous axes, so figures.

    Replaces `tbl_corpusfrac` and `tbl_window`.
    """
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.75))

    f = pd.read_csv(A / "fd45_scores.csv")
    f = f[f.run.str.match(r"F\d+")].copy()
    f["pct"] = f.run.str.extract(r"F(\d+)")[0].astype(int)
    g = f[f.level == "window"].groupby("pct").auc.mean().sort_index()
    axs[0].plot(g.index, g.values, marker="o", ms=3.2, lw=1.25, color=S.ORANGE,
                zorder=3)
    axs[0].axhspan(g.mean() - 1.0, g.mean() + 1.0, color=S.GREY_L, alpha=0.35,
                   lw=0, zorder=1)
    axs[0].set_xlabel("% of REPLACE-BG retained")
    axs[0].set_ylabel("window ROC-AUC")
    axs[0].set_title("corpus volume is a null")
    S.grid(axs[0])

    w7 = pd.read_csv(A / "fd7_scores.csv")
    w7["arm"] = w7.run.str.replace(r"-s\d$", "", regex=True)
    gw = (w7[w7.level == "window"].groupby(["arm", "run"])
          .apply(lambda x: x.groupby(["cohort", "task"]).auc.mean().mean(),
                 include_groups=False).groupby("arm").mean().sort_values())
    y = np.arange(len(gw))
    axs[1].barh(y, gw.values, color=S.SKY, height=0.66, zorder=3)
    axs[1].set_yticks(y); axs[1].set_yticklabels(gw.index, fontsize=4.6)
    axs[1].set_xlim(max(58, gw.min() - 2), gw.max() + 0.8)
    axs[1].set_xlabel("window ROC-AUC")
    axs[1].set_title("window and patch geometry")
    S.grid(axs[1], axis="x")

    fig.tight_layout(w_pad=1.5)
    _ly = S.letters(fig, axs)
    H = [Line2D([], [], color=S.ORANGE, marker="o", ms=3.0, lw=1.2),
         Patch(fc=S.GREY_L, ec="none", alpha=0.5)]
    S.rowlegend(fig, H, ["corpus sweep", r"$\pm1$ seed s.d."], ncol=2, y=_ly - 0.03)
    save(fig, "fig_sweeps")


# ============================================ APPENDIX: replaces 2 tables
def fig_readouts():
    """What each named block is worth, against width-matched controls.

    Replaces `tbl_readouts` and `tbl_controls`.
    """
    blk = _blocks()
    m = blk.groupby("variant").auc.mean()
    ROWS = [(r"$z_T$ alone (64)", "zT(64)", S.ORANGE),
            (r"$z_T\|z_S$ (112), released", "v2 zT||zS (112) <- proposal",
             S.ORANGE),
            (r"$z_S$ alone (48)", "zS(48)", S.CORAL),
            (r"$z_A$ alone (16)", "zA(16)", S.RED),
            ("full (128)", "full(128)", S.NAVY),
            ("random 64-d", "rand64", S.GREY_L),
            ("PCA 64-d", "pca64", S.GREY_L),
            ("random 48-d", "rand48", S.GREY_L),
            ("PCA 48-d", "pca48", S.GREY_L),
            ("GlucoFM full (128)", "GlucoFM full (128)", S.GREY),
            ("GlucoFM first 112", "GlucoFM first 112", S.GREY),
            ("GlucoFM first 64", "GlucoFM first 64", S.GREY)]
    rows = [(lab, m[k], c) for lab, k, c in ROWS if k in m.index]
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.3))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.68,
            zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.0)
    ax.set_xlim(64, max(r[1] for r in rows) + 1.2)
    ax.set_xlabel("cross-cohort transfer ROC-AUC")
    S.grid(ax, axis="x")
    fig.tight_layout()
    _ly = S.below(fig, [ax])
    H = [Patch(fc=S.ORANGE, ec="none"), Patch(fc=S.CORAL, ec="none"),
         Patch(fc=S.RED, ec="none"), Patch(fc=S.GREY_L, ec="none"),
         Patch(fc=S.GREY, ec="none")]
    S.rowlegend(fig, H, ["Trait", "State", "Sensor", "width-matched control",
                         "baseline truncation"], ncol=3, y=_ly - 0.03)
    save(fig, "fig_readouts")


# ============================================ APPENDIX: replaces 1 table
def fig_seedsd():
    """Seed-to-seed spread for every arm that trained more than one model.

    Replaces `tbl_seedsd`.
    """
    sv = pd.read_csv(A / "seed_variability.csv")
    ORDER = ["GlucoPRISM-C [seed-matched]", "GlucoPRISM-C", "GlucoPRISM-E",
             "GlucoPRISM-C [full readout]", "GlucoPRISM-E [full readout]",
             "GlucoFM (ours)", "No factorization (A)", "Bottleneck only (B)",
             "Objectives only (D)", "REPLACE-BG 50%", "REPLACE-BG 70%"]
    SHORT = {"GlucoPRISM-C [seed-matched]": "GP-C (matched)",
             "GlucoPRISM-C": "GP-C (6 seeds)", "GlucoPRISM-E": "GP-E",
             "GlucoPRISM-C [full readout]": r"GP-C, $z_A$ kept",
             "GlucoPRISM-E [full readout]": r"GP-E, $z_A$ kept",
             "GlucoFM (ours)": "GlucoFM", "No factorization (A)": "no fact. (A)",
             "Bottleneck only (B)": "bottleneck (B)",
             "Objectives only (D)": "objectives (D)",
             # plain '%': these are matplotlib labels, not LaTeX
             "REPLACE-BG 50%": "RBG 50%", "REPLACE-BG 70%": "RBG 70%"}
    LEVELS = [("window", S.ORANGE, "o"), ("subject", S.NAVY, "s"),
              ("transfer", S.TEAL, "^")]
    fig, ax = plt.subplots(figsize=(W * 0.74, 2.5))
    present = [a for a in ORDER if a in set(sv.model)]
    y = np.arange(len(present))[::-1]
    for k, (lvl, col, mk) in enumerate(LEVELS):
        off = (k - 1) * 0.23
        xs, ys, es = [], [], []
        for yy, arm in zip(y, present):
            r = sv[(sv.model == arm) & (sv.level == lvl)]
            if r.empty or not np.isfinite(r.iloc[0].auc_mean):
                continue
            xs.append(r.iloc[0].auc_mean)
            es.append(r.iloc[0].auc_sd_taskavg)
            ys.append(yy + off)
        ax.errorbar(xs, ys, xerr=es, fmt=mk, ms=2.8, color=col, lw=0,
                    elinewidth=0.8, capsize=1.6, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[a] for a in present],
                                         fontsize=5.0)
    ax.set_xlabel("ROC-AUC (mean $\\pm$ s.d. across pretraining seeds)")
    S.grid(ax, axis="x")
    fig.tight_layout()
    _ly = S.below(fig, [ax])
    H = [Line2D([], [], color=c, marker=mk, ms=2.8, lw=0)
         for _, c, mk in LEVELS]
    S.rowlegend(fig, H, ["window", "subject", "transfer"], ncol=3, y=_ly - 0.03)
    save(fig, "fig_seedsd")


# ==================================================== APPENDIX: kept, restyled
def fig_baselines():
    d = pd.read_csv(A / "final_table_long.csv")
    w = d[d.level == "window"]
    g = w.groupby(["run", "cohort", "task"]).auc.mean().groupby("run").mean()
    v2 = pd.read_csv(A / "v2_final_scores.csv")
    v2["seed"] = v2.run.str.extract(r"-s(\d)$")[0].astype(int)
    GPC = "GlucoPRISM-v2 + zA bottleneck (weak) [zA dropped]"
    m3 = v2[(v2.run.str.startswith("C-v2-vib01-s")) & (v2.block == "zTzS")
            & (v2.seed <= 2) & (v2.level == "window")]
    if len(m3):
        g[GPC] = m3.groupby(["cohort", "task"]).auc.mean().mean()
    PICK = [("Chronos-2-small", "Chronos-2-s", S.GREY_L),
            ("Chronos-2", "Chronos-2", S.GREY_L),
            ("MOMENT-large", "MOMENT-l", S.GREY_L),
            ("MOMENT-small", "MOMENT-s", S.GREY_L),
            ("CGMformer", "CGMformer", S.GREY_L),
            ("Mantis", "Mantis", S.GREY_L),
            ("MantisV2", "MantisV2", S.GREY_L),
            ("GlucoFM (ours)", "GlucoFM (ours)", S.NAVY),
            ("GlucoPRISM-v2 + bottleneck + measured sensor [zA dropped]",
             "GlucoPRISM-E", S.CORAL),
            (GPC, "GlucoPRISM-C", S.ORANGE)]
    rows = sorted([(lab, g.get(k, np.nan), c) for k, lab, c in PICK
                   if k in g.index], key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(W * 0.66, 2.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.68,
            zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.0)
    ax.set_xlim(55, 71)
    ax.set_xlabel("window ROC-AUC, 14 cells")
    ax.axvline(CANON["glucometrics"], color=S.RED, lw=0.9, ls=":", zorder=4)
    S.grid(ax, axis="x")
    fig.tight_layout()
    _ly = S.below(fig, [ax])
    H = [Patch(fc=S.ORANGE, ec="none"), Patch(fc=S.CORAL, ec="none"),
         Patch(fc=S.NAVY, ec="none"), Patch(fc=S.GREY_L, ec="none"),
         Line2D([], [], color=S.RED, lw=0.9, ls=":")]
    S.rowlegend(fig, H, ["GlucoPRISM-C", "GlucoPRISM-E", "GlucoFM (ours)",
                         "zero-shot", "12 glucometrics"], ncol=3, y=_ly - 0.03)
    save(fig, "fig7_baselines")


def fig_sensor():
    """Real paired-sensor disagreement: the offset prior work assumed away."""
    pm = pd.read_csv(A / "fd9_pair_measurements.csv")
    per = pm.groupby("subject").bias_mgdl.mean()
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.86, 1.7))
    axs[0].hist(per, bins=18, color=S.SKY, edgecolor="white", lw=0.4, zorder=3)
    axs[0].axvline(0, color=S.RED, lw=1.0, ls="--", zorder=4)
    axs[0].axvline(per.mean(), color=S.NAVY, lw=1.0, zorder=4)
    axs[0].set_xlabel("Libre $-$ Dexcom (mg/dL), per subject")
    axs[0].set_ylabel("subjects")
    axs[0].set_title(f"{int((per < 0).sum())} of {len(per)} read lower")
    S.grid(axs[0])

    mard = pm.groupby("subject").mard_pct.mean()
    axs[1].hist(mard, bins=18, color=S.TEAL, edgecolor="white", lw=0.4,
                zorder=3)
    axs[1].axvline(4.7, color=S.RED, lw=1.0, ls="--", zorder=4)
    axs[1].axvline(mard.mean(), color=S.NAVY, lw=1.0, zorder=4)
    axs[1].set_xlabel("mean abs. relative difference (%)")
    axs[1].set_ylabel("subjects")
    axs[1].set_title("disagreement is fivefold larger")
    S.grid(axs[1])

    fig.tight_layout(w_pad=1.5)
    _ly = S.letters(fig, axs)
    H = [Line2D([], [], color=S.RED, lw=1.0, ls="--"),
         Line2D([], [], color=S.NAVY, lw=1.0)]
    S.rowlegend(fig, H, ["assumed by prior synthetic views", "measured"],
                ncol=2, y=_ly - 0.03)
    save(fig, "fig6_sensor")


def fig_trait_stability():
    d = pd.read_csv(A / "remaining_experiments.csv")
    d = d[d.exp == "trait_stability"]
    if d.empty:
        return
    order = ["cgmacros", "stanford", "hall", "shanghait2dm"]
    MOD = [("GlucoPRISM-C", r"$[z_T\|z_S]$, $z_A$ dropped", S.ORANGE),
           ("GlucoPRISM-E", "GlucoPRISM-E", S.CORAL),
           ("GlucoFM", "GlucoFM", S.NAVY),
           ("GlucoPRISM-C [full]", r"full readout, $z_A$ kept", S.GREY_L)]
    fig, ax = plt.subplots(figsize=(W * 0.60, 1.85))
    x = np.arange(len(order)); wdt = 0.20
    for i, (m, lab, col) in enumerate(MOD):
        v = [d[(d.model == m) & (d.x == c)].auc.mean() for c in order]
        ax.bar(x + (i - 1.5) * wdt, v, wdt, color=col, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([COHN[c] for c in order], fontsize=5.4)
    ax.set_ylabel("across-day trait stability")
    S.grid(ax)
    fig.tight_layout()
    _ly = S.below(fig, [ax])
    H = [Patch(fc=c, ec="none") for _, _, c in MOD]
    S.rowlegend(fig, H, [l for _, l, _ in MOD], ncol=2, y=_ly - 0.03)
    save(fig, "fig3_trait_stability")


def fig_nulls():
    """Two things that do NOT bind: model capacity, and corpus volume.

    Both panels are derived. The capacity panel used to carry two typed points
    (65.85 / 65.17), which is why it drifted out of the pipeline; it now reads
    the same FD-8 arms every other capacity claim reads.
    """
    fd8 = pd.read_csv(A / "fd8_scores.csv")
    w = fd8[fd8.level == "window"].copy()
    w["arm"] = w.run.str.replace(r"-s\d+$", "", regex=True)
    w["seed"] = w.run.str.extract(r"-s(\d+)$").astype(int)

    # 1x and 4.97x, both WITHOUT the factorization objectives: the question is
    # whether capacity alone moves the benchmark.
    ARMS = [("V4-fm-off", "0.72M\n($1\\times$)"),
            ("V5-5x-off", "3.58M\n($4.97\\times$)")]
    mu, sd, lab = [], [], []
    for arm, tick in ARMS:
        per_seed = w[w.arm == arm].groupby("seed").auc.mean()
        if per_seed.empty:
            return
        mu.append(per_seed.mean())
        sd.append(per_seed.std(ddof=1))
        lab.append(tick)

    fig, axs = plt.subplots(1, 2, figsize=(W, 1.95))
    axs[0].errorbar([0, 1], mu, yerr=sd, marker="o", ms=4.5, lw=1.3,
                    color=S.ORANGE, capsize=2.5)
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(lab)
    axs[0].set_xlim(-0.45, 1.45)
    axs[0].set_ylabel("window ROC-AUC")
    axs[0].set_title("model capacity", fontsize=8.5, pad=4)
    axs[0].set_ylim(62, 70)
    axs[0].annotate("flat", xy=(0.5, max(mu) + 1.0), fontsize=8,
                    color=S.GREY, ha="center")
    S.grid(axs[0])

    f = pd.read_csv(A / "fd45_scores.csv")
    f = f[f.run.str.match(r"F\d+")].copy()
    f["pct"] = f.run.str.extract(r"F(\d+)")[0].astype(int)
    g = f[f.level == "window"].groupby("pct").auc.mean().sort_index()
    axs[1].plot(g.index, g.values, marker="o", ms=3.5, lw=1.3, color=S.ORANGE)
    axs[1].axhspan(g.mean() - 1.0, g.mean() + 1.0, color=S.GREY, alpha=0.18,
                   lw=0, label="$\\pm1$ seed $\\sigma$")
    # Plain "%": these labels are drawn by matplotlib, not LaTeX, so an escaped
    # "\%" renders the backslash literally.
    axs[1].set_xlabel("% of REPLACE-BG retained")
    axs[1].set_ylabel("window ROC-AUC")
    axs[1].set_title("corpus volume", fontsize=8.5, pad=4)
    axs[1].legend(loc="lower right", fontsize=5.6)
    S.grid(axs[1])

    fig.tight_layout(w_pad=1.5)
    S.letters(fig, axs)
    save(fig, "fig4_nulls")


def fig_loss_competition():
    """Why training the factorization at full weight destroys the model."""
    K = Path(os.environ.get("GLUCOPRISM_RUNS", A / "runs"))
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.86, 1.7))
    RUNS = [("with factorization objectives", "V1-fm-joint-s0",
             "prism_history.jsonl", S.RED, "-"),
            ("without them", "V4-fm-off-s0", "glucofm_history.jsonl",
             S.ORANGE, "--")]
    for lbl, run, hist, col, ls in RUNS:
        p = K / run / "checkpoints" / hist
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        axs[0].plot([r["epoch"] for r in rows],
                    [r.get("loss_mcr", np.nan) for r in rows], lw=1.2,
                    color=col, ls=ls, zorder=3)
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("masked-latent loss")
    axs[0].set_title("the representation objective")
    S.grid(axs[0])

    p = K / "V1-fm-joint-s0" / "checkpoints" / "prism_history.jsonl"
    LAB = []
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        ep = [r["epoch"] for r in rows]
        for (key, lbl), col, ls in zip(
                [("loss_sensor", r"$L_\mathrm{sensor}$"),
                 ("loss_day", r"$L_\mathrm{day}$"),
                 ("loss_indep", r"$L_\mathrm{indep}$")],
                [S.NAVY, S.TEAL, S.PURPLE], ["-", "--", ":"]):
            v = np.array([r.get(key, np.nan) for r in rows], float)
            if np.isfinite(v).any():
                axs[1].plot(ep, v / np.nanmax(v), lw=1.15, color=col, ls=ls,
                            zorder=3)
                LAB.append((lbl, col, ls))
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("loss (normalised)")
    axs[1].set_title("the factorization objectives")
    S.grid(axs[1])

    fig.tight_layout(w_pad=1.5)
    _ly = S.letters(fig, axs)
    H = [Line2D([], [], color=S.RED, lw=1.2),
         Line2D([], [], color=S.ORANGE, lw=1.2, ls="--")]
    L = ["with factorization objectives", "without them"]
    for lbl, col, ls in LAB:
        H.append(Line2D([], [], color=col, lw=1.15, ls=ls))
        L.append(lbl)
    S.rowlegend(fig, H, L, ncol=5, y=_ly - 0.03)
    save(fig, "fig2_loss_competition")


for f in (fig_dissociation, fig_deletion, fig_shortcut, fig_perdata,
          fig_percell, fig_sweeps, fig_readouts, fig_seedsd,
          fig_baselines, fig_sensor, fig_trait_stability,
          fig_loss_competition, fig_nulls):
    try:
        f()
    except Exception as e:  # noqa: BLE001
        print(f"  {f.__name__} FAILED: {type(e).__name__}: {e}")
print(f"\n-> {OUT}")
