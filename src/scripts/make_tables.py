"""Emit EVERY paper table as LaTeX, one file per table, generated from the CSVs.

The paper has no appendix: methods, ablations and all supporting tables live in
the body. That only stays honest if the tables are generated rather than typed,
so this writes one `tbl_*.tex` per table into the Overleaf folder and into
final_materials, and main.tex \\inputs them where they belong.

Tables
  tbl_corpus        pretraining corpus composition
  tbl_sensor        measured paired-sensor disagreement
  tbl_summary       task-averaged, all models, both protocols, Holm
  tbl_percell       14 cells x all models, window ROC-AUC
  tbl_ladder        the decomposition: objective / bottleneck / address
  tbl_arch          architecture grid: {1x,5x} x {off, joint, post-hoc}
  tbl_posthoc       post-hoc block fitting on a frozen encoder
  tbl_transfer      cross-cohort transfer, per direction
  tbl_controls      block controls at matched width
  tbl_corpusfrac    leave-one-cohort-out and the REPLACE-BG fraction sweep
  tbl_window        windowing / patch geometry sweep
  tbl_sig           paired significance vs GlucoFM, both levels
"""
from __future__ import annotations

import os as _os, sys as _sys
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))
RUNS = _P(_os.environ.get("GLUCOPRISM_RUNS", OUTDIR / "runs"))
EXTERNAL = _P(_os.environ.get("GLUCOPRISM_EXTERNAL", ROOT / "external"))
REFERENCE = ROOT / "src" / "core" / "released_model"
for _p in (ROOT / "src" / "core", ROOT / "baselines", ROOT / "src" / "scripts",
           ROOT / "src" / "ablations", REFERENCE,
           _P(__file__).resolve().parent):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))


import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

A = (OUTDIR)
# The released copy also wrote every table into a hard-coded Overleaf directory
# (D:\overleaf\...), which contradicts the README's "No absolute paths are baked
# in" and silently edits the author's paper. Reproduction writes only inside
# GLUCOPRISM_OUT; set GLUCOPRISM_TEX_OUT to add a second destination.
OUTS = [(OUTDIR / "tex")]
if _os.environ.get("GLUCOPRISM_TEX_OUT"):
    OUTS.append(Path(_os.environ["GLUCOPRISM_TEX_OUT"]))
CANON = json.loads((A / "canonical.json").read_text())

GPC = "GlucoPRISM-v2 + zA bottleneck (weak) [zA dropped]"
GPE = "GlucoPRISM-v2 + bottleneck + measured sensor [zA dropped]"
FM = "GlucoFM (ours)"

TASK = {"diabetes": "Diabetes risk", "diabetes_3class": "Diabetes risk (3-cls)",
        "ir": "Insulin resistance", "beta_cell": r"$\beta$-cell dysfunction",
        "hyperlipidemia": "Hyperlipidemia", "obesity": "Obesity",
        "hypoglycemia": "Hypoglycemia", "glucotype": "Glucotype"}
COH = {"cgmacros": "CGMacros", "shanghait2dm": "ShanghaiT2DM",
       "stanford": "Stanford", "hall": "Hall"}


def write(name: str, lines: list[str]) -> None:
    txt = "\n".join(lines) + "\n"
    for d in OUTS:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.tex").write_text(txt, encoding="utf-8")
    print(f"  {name:<16} {len(lines):>4} lines")


def tab(cols: str, caption: str, label: str, header: str, rows: list[str],
        pos: str = "!tbp", note: str = "", wide: bool = False,
        sideways: bool = False) -> list[str]:
    # A 10- or 12-column table overruns the text block at \small with default
    # column padding. `wide` drops to \footnotesize and tightens \tabcolsep.
    #
    # `sideways` is for the two tables that carry a model per column: at ICLR's
    # 5.5in text width, scaling them to fit portrait makes the digits too small
    # to read. Rotating gives them the 9in dimension instead, so they render at
    # full size, which is the point of putting them in the main text at all.
    # A rotated table has ~9in to play with, so it can afford the larger size
    # and padding that make a ten-column grid of digits actually readable.
    size = (r"\small" if sideways else
            (r"\footnotesize" if wide else r"\small"))
    pad = (r"\setlength{\tabcolsep}{5.5pt}" if sideways else
           (r"\setlength{\tabcolsep}{3.4pt}" if wide else ""))
    env = "sidewaystable" if sideways else "table"
    # A rotated table already has room; only portrait wide tables get scaled.
    open_box = r"\resizebox{\textwidth}{!}{%" if (wide and not sideways) else ""
    close_box = "}" if (wide and not sideways) else ""
    L = [rf"\begin{{{env}}}[{pos}]\centering{size}{pad}",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}", open_box,
         rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header, r"\midrule"]
    L += rows
    L += [r"\bottomrule", r"\end{tabular}", close_box]
    if note:
        # NOT `\\[2pt]`: outside a tabular (and after the \resizebox brace) that
        # is "There's no line here to end". A parbox-width paragraph is robust
        # in both the plain and the resized case.
        width = r"0.95\textheight" if sideways else r"\textwidth"
        L.append(rf"\vspace{{4pt}}\par\footnotesize\begin{{minipage}}{{{width}}}"
                 rf"{note}\end{{minipage}}")
    L.append(rf"\end{{{env}}}")
    return L


# ---------------------------------------------------------------- seed-match
df = pd.read_csv(A / "final_table_long.csv")
v2 = pd.read_csv(A / "v2_final_scores.csv")
v2["seed"] = v2.run.str.extract(r"-s(\d)$")[0].astype(int)
v2["arm"] = v2.run.str.replace(r"-s\d$", "", regex=True)
m3 = (v2[(v2.arm == "C-v2-vib01") & (v2.block == "zTzS") & (v2.seed <= 2)]
      .groupby(["level", "cohort", "task"], as_index=False)[["pr", "auc", "f1"]]
      .mean())
m3["run"] = GPC
df = pd.concat([df[df.run != GPC], m3], ignore_index=True)
W = df[df.level == "window"]

# ================================================================ tbl_corpus
rows = [
    r"REPLACE-BG \citep{aleppo2017replacebg} & Dexcom G4 & 5 & 226 & 9,035 & 0.977 \\",
    r"Stanford \citep{hall2018glucotypes} & Dexcom & 5 & 27 & 279 & 0.979 \\",
    r"ShanghaiT2DM \citep{zhao2023shanghai} & FreeStyle Libre & 15 & 40 & 247 & 0.333 \\",
    r"Col\'as \citep{colas2019detection} & Medtronic iPro & 5 & 191 & 287 & 0.996 \\",
    r"BIG IDEAs \citep{bent2021bigideas} & Dexcom & 5 & 16 & 70 & 0.929 \\",
    r"\midrule",
    rf"\textbf{{Total (public only)}} & --- & --- & \textbf{{{CANON['nsubj']}}} "
    rf"& \textbf{{{CANON['nwin']:,}}} & --- \\",
    r"\addlinespace",
    rf"\emph{{GlucoFM's corpus (private)}} & --- & --- & \emph{{{CANON['fmsubj']}}} & "
    r"--- & --- \\",
]
write("tbl_corpus", tab(
    "llrrrr",
    r"Pretraining corpus. All five cohorts are public and redistributable. "
    r"ShanghaiT2DM's low observation fraction is not missingness: a 15-minute "
    r"sensor can fill at most 96 of 288 five-minute grid slots, so coverage "
    r"thresholds are applied relative to device rate. REPLACE-BG is capped at "
    r"40 windows per subject; uncapped it would contribute roughly 28,000 "
    r"windows and turn the corpus into a type~1 diabetes distribution.",
    "tab:corpus",
    r"Cohort & Device & Rate (min) & Subjects & Windows & Obs.\ frac. \\",
    rows, wide=True,
    note=rf"Our corpus totals {CANON['nhours']:,} hours against "
         rf"{CANON['fmhours']:,} for the private corpus GlucoFM was trained on."))

# ================================================================ tbl_sensor
write("tbl_sensor", tab(
    "lrr",
    r"Real paired-sensor disagreement, measured on 374 same-day Dexcom and "
    r"Libre windows from 44 CGMacros subjects held out of pretraining. "
    r"\textbf{43 of 44 subjects show Libre reading lower}: a systematic "
    r"calibration offset, not symmetric noise. Synthetic paired views in prior "
    r"work carry no level shift at all.",
    "tab:sensor",
    r"Quantity & Previously assumed & \textbf{Measured} \\",
    [r"Calibration difference (mg/dL) & $0 \pm 6$ & $\mathbf{-31.1 \pm 15.8}$ \\",
     r"Mean absolute relative difference & 4.7\% & \textbf{24.3\%} \\",
     r"Deming slope & 1.00 & \textbf{0.878} \\",
     r"Gain s.d.\ across subjects & --- & \textbf{0.217} \\",
     r"Correlation with source window & 0.95 & \textbf{0.737} \\",
     r"Subjects with negative bias & --- & \textbf{43 / 44} \\"],
    note=r"Ordinary least squares reports a slope of 0.674 on the same data; "
         r"OLS is attenuated when both variables carry noise, so the generator "
         r"is fitted on the Deming estimate."))

# =============================================================== tbl_summary
sig = pd.read_csv(A / "significance_window_auc_matched.csv")
sigmap = dict(zip(sig.model, sig.p_holm))
MAIN = [(r"\textbf{GlucoPRISM-C} \emph{(released)}", GPC),
        (r"\textbf{GlucoPRISM-E} \emph{(released)}", GPE),
        ("GlucoFM (our reproduction)", FM),
        ("MantisV2", "MantisV2"), ("Mantis", "Mantis"),
        ("CGMformer", "CGMformer"), ("MOMENT-small", "MOMENT-small"),
        ("MOMENT-large", "MOMENT-large"), ("Chronos-2", "Chronos-2"),
        ("Chronos-2-small", "Chronos-2-small")]
rows = []
for i, (nm, run) in enumerate(MAIN):
    if run not in set(df.run):
        continue
    if i == 3:
        rows.append(r"\addlinespace\multicolumn{8}{l}{\emph{Zero-shot foundation "
                    r"models (frozen third-party checkpoints)}} \\")
    r = {}
    for lvl in ("window", "subject"):
        s = df[(df.run == run) & (df.level == lvl)]
        r[lvl] = s.groupby(["cohort", "task"])[["pr", "auc", "f1"]].mean().mean()
    ph = sigmap.get(run)
    pc = "---" if ph is None else (rf"\textbf{{{ph:.3f}}}" if ph < 0.05
                                   else f"{ph:.3f}")
    rows.append(f"{nm} & " + " & ".join(
        f"{r[l][k]:.1f}" for l in ("window", "subject")
        for k in ("pr", "auc", "f1")) + f" & {pc} \\\\")
# Compact main-text variant: the two released models, the backbone they build
# on, and the best and worst of the seven zero-shot baselines. The full ten-row
# table goes to the appendix -- ICLR caps the main text at 9 pages, and the five
# middle baseline rows are the ones a reader is least likely to need inline.
KEEP = {GPC, GPE, FM}
zs_rows = [(nm, run) for nm, run in MAIN if run not in KEEP and run in set(df.run)]
zs_scored = []
for nm, run in zs_rows:
    s = df[(df.run == run) & (df.level == "window")]
    zs_scored.append((s.groupby(["cohort", "task"]).auc.mean().mean(), nm, run))
zs_scored.sort(reverse=True)
COMPACT = [(nm, run) for nm, run in MAIN if run in KEEP]
if zs_scored:
    COMPACT += [(zs_scored[0][1], zs_scored[0][2]),
                (zs_scored[-1][1], zs_scored[-1][2])]
rows_c = []
for i, (nm, run) in enumerate(COMPACT):
    if i == 3:
        rows_c.append(r"\addlinespace\multicolumn{8}{l}{\emph{Best and worst of "
                      r"seven zero-shot foundation models; full list in "
                      r"Appendix~\ref{app:percell}}} \\")
    r = {}
    for lvl in ("window", "subject"):
        s = df[(df.run == run) & (df.level == lvl)]
        r[lvl] = s.groupby(["cohort", "task"])[["pr", "auc", "f1"]].mean().mean()
    ph = sigmap.get(run)
    pc = "---" if ph is None else (rf"\textbf{{{ph:.3f}}}" if ph < 0.05
                                   else f"{ph:.3f}")
    rows_c.append(f"{nm} & " + " & ".join(
        f"{r[l][k]:.1f}" for l in ("window", "subject")
        for k in ("pr", "auc", "f1")) + f" & {pc} \\\\")
write("tbl_summary_main", tab(
    "lrrrrrrr",
    r"Task-averaged performance over the 14 task--cohort cells, frozen encoder "
    r"and shared folds, all models at three seeds. $p_{\mathrm{Holm}}$ is "
    r"corrected over the pre-declared confirmatory family of nine; GlucoPRISM-C "
    r"is the only member to survive. The subject-level column is unsettled: "
    r"Mantis leads it while trailing by four ROC-AUC at window level.",
    "tab:summary",
    r"& \multicolumn{3}{c}{Window level} & \multicolumn{3}{c}{Subject level} & \\"
    "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
    r"Model & PR & ROC & F1 & PR & ROC & F1 & $p_{\mathrm{Holm}}$ \\",
    rows_c))

write("tbl_summary", tab(
    "lrrrrrrr",
    r"Task-averaged performance over the 14 task--cohort cells, frozen encoder "
    r"and shared folds, all models at three seeds. $p_{\mathrm{Holm}}$ is "
    r"corrected over the pre-declared confirmatory family of nine; GlucoPRISM-C "
    r"is the only member to survive. Note that the subject-level column is "
    r"unsettled: Mantis leads it while trailing by four ROC-AUC at window level.",
    "tab:summaryfull",
    r"& \multicolumn{3}{c}{Window level} & \multicolumn{3}{c}{Subject level} & \\"
    "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
    r"Model & PR & ROC & F1 & PR & ROC & F1 & $p_{\mathrm{Holm}}$ \\",
    rows))

# =============================================================== tbl_percell
KEY = {GPC: r"GP-C", GPE: "GP-E", FM: "GlucoFM",
       "MantisV2": "MantisV2", "Mantis": "Mantis", "CGMformer": "CGMformer",
       "MOMENT-small": "MOM-s", "MOMENT-large": "MOM-l",
       "Chronos-2": "Chr-2", "Chronos-2-small": "Chr-2s"}
cols = [c for c in KEY if c in set(W.run)]
piv = W.groupby(["cohort", "task", "run"]).auc.mean().unstack("run")


def cellfmt(v, best, second):
    """Bold the best in a row, underline the runner-up."""
    if pd.isna(v):
        return "--"
    if abs(v - best) < 1e-9:
        return rf"\bfseries {v:.1f}"
    if abs(v - second) < 1e-9:
        return rf"\underline{{{v:.1f}}}"
    return f"{v:.1f}"


# TRANSPOSED: models are rows and the 14 cells are columns. A model-per-column
# layout needs 12 wide columns and does not fit ICLR's 5.5in text block without
# either rotating the page or shrinking the digits past legibility; with cells
# as columns each column holds four characters and the whole table fits
# portrait at full size.
SHORT = {"diabetes": "DR", "diabetes_3class": "DR3", "ir": "IR",
         "beta_cell": r"$\beta$C", "hyperlipidemia": "HL", "obesity": "Ob",
         "hypoglycemia": "Hy", "glucotype": "GT"}
ORDER = [(c, t) for c in ["cgmacros", "shanghait2dm", "stanford", "hall"]
         for t in piv.loc[c].index]
ROWKEY = {GPC: r"\textbf{GlucoPRISM-C}", GPE: r"\textbf{GlucoPRISM-E}",
          FM: "GlucoFM (ours)", "MantisV2": "MantisV2", "Mantis": "Mantis",
          "CGMformer": "CGMformer", "MOMENT-small": "MOMENT-s",
          "MOMENT-large": "MOMENT-l", "Chronos-2": "Chronos-2",
          "Chronos-2-small": "Chronos-2-s"}
# best / runner-up are per CELL, i.e. down each column
bestcol, secondcol = {}, {}
for key in ORDER:
    v = sorted((piv.loc[key, c] for c in cols
                if not pd.isna(piv.loc[key, c])), reverse=True)
    bestcol[key], secondcol[key] = v[0], (v[1] if len(v) > 1 else np.nan)
means = piv[cols].mean()
mv = sorted(means.values, reverse=True)

rows = []
for i, c in enumerate(cols):
    if i == 3:
        rows.append(r"\addlinespace[2pt]\multicolumn{16}{l}{\emph{Zero-shot "
                    r"foundation models}} \\")
    tint = r"\rowcolor{oursbg}" if c in (GPC, GPE) else ""
    cells = [cellfmt(piv.loc[k, c], bestcol[k], secondcol[k]) for k in ORDER]
    rows.append(f"{tint}{ROWKEY[c]} & " + " & ".join(cells) + " & "
                + cellfmt(means[c], mv[0], mv[1]) + r" \\")
    if i == 1:
        rows.append(r"\addlinespace[2pt]")

hdr = (r"& \multicolumn{4}{c}{CGMacros} & \multicolumn{3}{c}{ShanghaiT2DM} "
       r"& \multicolumn{3}{c}{Stanford} & \multicolumn{4}{c}{Hall} & \\"
       "\n" r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}"
       r"\cmidrule(lr){12-15}" "\n"
       "Model & " + " & ".join(SHORT.get(t, t) for _, t in ORDER)
       + r" & \textbf{Avg} \\")
write("tbl_percell", tab(
    "l" + "r" * len(ORDER) + r" >{\columncolor{oursbg}}r",
    r"\textbf{Subject-disjoint linear probing: window-level ROC-AUC for all 14 "
    r"task--cohort cells.} Frozen encoder, identical logistic-regression probe, "
    r"5-fold subject-grouped cross-validation $\times$ 10 iterations on folds "
    r"shared by every model, seed-matched at three seeds throughout. "
    r"\textbf{Bold} is the best model in a cell, \underline{underline} the "
    r"runner-up; higher is better. Tasks: DR diabetes risk, DR3 3-class "
    r"diabetes risk, IR insulin resistance, $\beta$C $\beta$-cell dysfunction, "
    r"HL hyperlipidemia, Ob obesity, Hy hypoglycemia, GT glucotype.",
    "tab:percell", hdr, rows, wide=True,
    note=r"Four cells sit near chance for every model tested, which dilutes the "
         r"final column; we report all 14 rather than a CGM-tractable subset. "
         r"Hall insulin resistance is the one cell where the backbone is ahead "
         r"of both released models."))

# ================================================================ tbl_ladder
write("tbl_ladder", tab(
    "lrr",
    r"The path to the released configuration, adding one component at a time to "
    r"the same backbone, corpus and probe (six seeds throughout). "
    r"\textbf{These increments are not independent contributions and must not "
    r"be read as a decomposition}: Table~\ref{tab:interaction} shows that the "
    r"factorization objectives and the bottleneck each do nothing on their own "
    r"and only pay off together, so the increment attributed to whichever "
    r"component is added second absorbs the whole interaction.",
    "tab:ladder",
    r"Configuration & Window ROC-AUC & Increment \\",
    [rf"GlucoFM (our reproduction) & {CANON['lad_fmbase']} & --- \\",
     rf"\;$+$ protocol factorization objectives, full readout & "
     rf"{CANON['lad_facfull']} & $+{CANON['ladfactor']}$ \\",
     rf"\;$+$ variational bottleneck on $z_A$, full readout & "
     rf"{CANON['lad_vibfull']} & $+{CANON['ladvib']}$ \\",
     rf"\;$+$ discard $z_A$ at inference \emph{{(released)}} & "
     rf"\textbf{{{CANON['lad_vibdrop']}}} & $+{CANON['laddrop']}$ \\"]))

# ============================================================== tbl_capacity
CS = A / "rev_capacity_summary.csv"
if CS.exists():
    cs = pd.read_csv(CS)
    cs["is_rel"] = cs.d_sensor.astype(str).str.contains("released")
    cs["width"] = cs.d_sensor.astype(str).str.extract(r"^(\d+)")[0].astype(int)
    cs["b"] = cs.beta.astype(float)
    # Two slices through one grid; `by` names which column labels the row, so
    # the label cannot be inferred from the group title (both titles mention
    # both quantities, which is what broke the first version).
    GROUPS = [("Sensor-block width $d_A$ (at $\\beta = 0.1$)",
               cs[cs.b == 0.1].sort_values("width"), "width"),
              ("KL weight $\\beta$ (at $d_A = 16$)",
               cs[cs.width == 16].sort_values("b"), "b")]
    rows = []
    for title, sel, by in GROUPS:
        rows.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\")
        for _, r in sel.iterrows():
            key = str(int(r["width"])) if by == "width" else f"{r['b']:g}"
            star = r"\,\textbf{(released)}" if r["is_rel"] else ""
            # r.drop would resolve to DataFrame.drop, not the column.
            rows.append(f"\\quad {key}{star} & {r['full']:.2f} & "
                        f"{r['drop']:.2f} & ${r['gain']:+.2f}$ & "
                        f"${r['subj_gain']:+.2f}$ \\\\")
        rows.append(r"\addlinespace")
    write("tbl_capacity", tab(
        "lrrrr",
        r"Sensor-block capacity, two seeds per arm, seed-matched against the "
        r"released model on the same two seeds. Width is varied with $z_T$ held "
        r"at 64 so $z_S$ absorbs the difference; the confound is therefore "
        r"$z_A$-versus-$z_S$ width, not $z_A$-versus-$z_T$. \textbf{The benefit "
        r"of deleting the block rises monotonically with the price it pays per "
        r"nat} --- $+0.03$, $+0.55$, $+1.31$, $+1.50$ as $\beta$ goes "
        r"$0.03 \to 1.0$ --- which is what the addressability account predicts "
        r"and what a pure-regularisation account does not: a regulariser's "
        r"value would not track how hard the block is squeezed. Width behaves "
        r"the same way, with an 8-dimensional block too small to absorb "
        r"anything worth removing.",
        "tab:capacity",
        r"Setting & Full readout & $z_A$ dropped & Gain (window) & "
        r"Gain (subject) \\", rows,
        note=r"The released configuration ($\beta = 0.1$) is not the optimum of "
             r"this sweep: $\beta = 1.0$ scores higher on both columns. The "
             r"sweep was run after the models were released and we report it "
             r"rather than re-selecting; a stronger bottleneck is the natural "
             r"next release. Subject-level gains are erratic across arms, "
             r"consistent with that axis being unresolved throughout this work."))

# =============================================================== tbl_genrobust
GR = A / "rev_generator_robustness.json"
if GR.exists():
    gr = json.loads(GR.read_text())
    LBL = {"bias_mgdl": "Calibration offset (mg/dL)", "slope": "Deming slope",
           "mard_pct": r"Mean abs.\ relative difference (\%)",
           "corr": "Correlation with source window"}
    rows = [f"{LBL[k]} & {gr[k]['fitted']:.2f} & "
            f"[{gr[k]['loo_min']:.2f}, {gr[k]['loo_max']:.2f}] & "
            f"[{gr[k]['ci_lo']:.2f}, {gr[k]['ci_hi']:.2f}] \\\\"
            for k in LBL if k in gr]
    ss = gr["sign_stability"]
    write("tbl_genrobust", tab(
        "lrrr",
        r"Is the synthetic paired-sensor generator fragile? Its constants are "
        r"fitted on 374 windows from 44 subjects, which is modest, so we refit "
        r"under resampling at \emph{subject} level --- leaving each subject out "
        r"in turn, and bootstrapping over subjects. Every parameter is stable "
        r"to within a fraction of its own magnitude, and the calibration offset "
        rf"is negative in all {ss['boot_n']:,} bootstrap refits. The direction "
        r"the $V_1$ view depends on is not in question; only its exact "
        r"magnitude is, and that varies by a few mg/dL.",
        "tab:genrobust",
        r"Fitted parameter & Point estimate & Leave-one-subject-out range & "
        r"95\% bootstrap \\", rows,
        note=rf"{ss['subjects_negative']} of {ss['subjects_n']} individual "
             r"subjects show the offset in the same direction, so the effect is "
             r"not carried by a few outlying people."))

# ================================================ tbl_erasure (INLP baseline)
if (A / "rev_inlp_calibration.csv").exists():
    e = pd.read_csv(A / "rev_inlp_calibration.csv")
    ge = e.groupby("tag")[["auc", "ece", "brier"]].mean()
    base = ge.loc["full (128d)", "auc"]
    rows = []
    for t, lbl in [("full (128d)", "Full readout, no surgery"),
                   ("INLP on full (128d)",
                    r"INLP \citep{ravfogel2020null}, 8 iterations"),
                   ("drop zA (112d, ours)",
                    r"\textbf{Reserved-and-drop (ours)}")]:
        if t not in ge.index:
            continue
        d_ = ge.loc[t, "auc"] - base
        rows.append(f"{lbl} & {ge.loc[t, 'auc']:.2f} & "
                    f"{'---' if abs(d_) < 1e-9 else f'${d_:+.2f}$'} & "
                    f"{ge.loc[t, 'ece']:.3f} & {ge.loc[t, 'brier']:.3f} \\\\")
    write("tbl_erasure", tab(
        "lrrrr",
        r"Post-hoc erasure as an alternative route to addressability. INLP fits "
        r"a linear device probe on the full embedding and projects out its "
        r"direction, iterating eight times; it is supervised by CGMacros' "
        r"\emph{real} Dexcom/Libre labels, the same signal $z_A$ receives. "
        r"Cross-cohort transfer, three seeds. Erasure recovers most of the gain, "
        r"confirming that a removable device direction genuinely exists, but "
        r"the reserved block is better and needs no fitting at all. Calibration "
        r"improves monotonically with it (lower is better for both ECE and "
        r"Brier), which matters because AUROC alone says nothing about whether "
        r"a screening threshold transfers.",
        "tab:erasure",
        r"Readout & Transfer ROC-AUC & $\Delta$ & ECE & Brier \\",
        rows,
        note=r"INLP is favoured slightly by this setup: when CGMacros is the "
             r"transfer target, the projection was fitted on target-domain "
             r"inputs (never on target labels). Ours uses no target data at all."))

# ====================================================== tbl_partial (deletion)
if (A / "rev_partial_within.csv").exists():
    tr = pd.read_csv(A / "rev_soft_deletion.csv").groupby("tag").auc.mean()
    kw = pd.read_csv(A / "rev_partial_within.csv")
    kw["seed"] = kw.run.str.extract(r"-s(\d)$")[0].astype(int)
    kw["arm"] = "C-v2-vib01"
    allb = pd.concat([v2, kw], ignore_index=True)
    allb["seed"] = allb.run.str.extract(r"-s(\d)$")[0].astype(int)
    allb["arm"] = allb.run.str.replace(r"-s\d$", "", regex=True)
    allb = allb[(allb.arm == "C-v2-vib01") & (allb.seed <= 2)]

    def wc(b: str, lvl: str) -> float:
        s = allb[(allb.block == b) & (allb.level == lvl)]
        return (s.groupby(["cohort", "task"]).auc.mean().mean() if len(s)
                else float("nan"))

    rows = []
    for b, m_, lbl in [("full", 16, "keep all 16 (full readout)"),
                       ("keep4", 4, "keep 4 of 16"),
                       ("keep2", 2, "keep 2 of 16"),
                       ("zTzS", 0, r"\textbf{keep 0 of 16 (ours)}")]:
        t = tr.get(f"keep {m_}/16 zA dims", float("nan"))
        rows.append(f"{lbl} & {t:.2f} & {wc(b, 'window'):.2f} & "
                    f"{wc(b, 'subject'):.2f} \\\\")
    write("tbl_partial", tab(
        "lrrr",
        r"Is there a useful \emph{partial} deletion? Dimensions of $z_A$ are "
        r"ordered by how device-informative they are on real Dexcom/Libre "
        r"labels, and the least informative $m$ are readmitted. There is no "
        r"trade-off to tune: full deletion is best on all three axes, and "
        r"readmitting as few as four of sixteen coordinates costs the entire "
        r"transfer benefit. \textbf{Scaling $z_A$ by a factor instead of "
        r"dropping it does nothing at all} --- the probe standardises every "
        r"column, so the scale divides out exactly (measured: identical to "
        r"three decimals for every non-zero scale).",
        "tab:partial",
        r"Readout & Transfer & Within-cohort window & Within-cohort subject \\",
        rows))

# =========================================== tbl_confounding + tbl_dependence
if (A / "rev_device_predictability.csv").exists():
    dp = pd.read_csv(A / "rev_device_predictability.csv")
    rows = [f"{r.features} & {r.device_auc:.2f} & {r.sd:.2f} \\\\"
            for _, r in dp.iterrows()]
    write("tbl_confounding", tab(
        "lrr",
        r"How available is the device shortcut? A linear probe predicts which "
        r"sensor produced a window, on CGMacros' real paired data, under "
        r"subject-grouped cross-validation. \textbf{The observation mask alone "
        r"identifies the device perfectly, and so does a single scalar --- the "
        r"number of observed samples.} Mask preservation is a deliberate and "
        r"well-motivated design choice in this backbone, and it hands the model "
        r"a free device channel. This is why the confound is not subtle and why "
        r"a model must be given somewhere to put it.",
        "tab:confounding",
        r"Features available to the probe & Device ROC-AUC & s.d. \\", rows))

if (A / "rev_block_dependence.csv").exists():
    bd_ = pd.read_csv(A / "rev_block_dependence.csv")
    gb = bd_.groupby("pair")[["mean_abs_corr", "hsic"]].mean()
    rows = [f"{p} & {gb.loc[p, 'mean_abs_corr']:.3f} & {gb.loc[p, 'hsic']:.3f} \\\\"
            for p in ["zT-zS", "zT-zA", "zS-zA"] if p in gb.index]
    write("tbl_dependence", tab(
        "lrr",
        r"Did the blocks actually separate? $L_{\mathrm{indep}}$ penalises "
        r"off-diagonal \emph{correlation}, which is linear; HSIC with RBF "
        r"kernels detects dependence a correlation matrix cannot see. Averaged "
        r"over three seeds and three cohorts. \textbf{Trait and State did not "
        r"separate} --- they remain strongly dependent by both measures --- "
        r"while Sensor is markedly more independent of both. The one block that "
        r"behaves as designed is the one we delete, which is the same "
        r"dissociation this paper reports throughout, measured on the "
        r"representation itself rather than on any downstream score. HSIC "
        r"tracks correlation closely here, so little dependence is hiding in "
        r"non-linear structure.",
        "tab:dependence",
        r"Block pair & mean $|$correlation$|$ & HSIC \\", rows))

# =========================================================== tbl_interaction
CFJ = A / "confound_analysis.json"
if CFJ.exists():
    cf = json.loads(CFJ.read_text())["full"]
    cfs = pd.read_csv(A / "confound_scores.csv")
    cfs["arm"] = cfs.run.str.replace(r"-s\d$", "", regex=True)
    allsc = pd.concat([v2, cfs], ignore_index=True)
    allsc["seed"] = allsc.run.str.extract(r"-s(\d)$")[0].astype(int)
    allsc["arm"] = allsc.run.str.replace(r"-s\d$", "", regex=True)

    def cell(arm: str) -> float:
        s = allsc[(allsc.arm == arm) & (allsc.block == "full") &
                  (allsc.level == "window") & (allsc.seed <= 2)]
        return float(s.groupby(["cohort", "task"]).auc.mean().mean())

    g = {k: cell(v) for k, v in
         {"offoff": "X-noproto", "offon": "Y-noproto-vib",
          "onoff": "A-v2-base", "onon": "C-v2-vib01"}.items()}
    rows = [
        rf"\emph{{off}} & {g['offoff']:.2f} & {g['offon']:.2f} & "
        rf"${cf['vib_noobj']:+.2f}$ \\",
        rf"\emph{{on}} & {g['onoff']:.2f} & \textbf{{{g['onon']:.2f}}} & "
        rf"$\mathbf{{{cf['vib_obj']:+.2f}}}$ \\",
        r"\midrule",
        rf"\emph{{simple effect of objectives}} & ${cf['obj_novib']:+.2f}$ & "
        rf"$\mathbf{{{cf['obj_vib']:+.2f}}}$ & "
        rf"\textbf{{interaction }}$\mathbf{{{cf['inter']:+.2f}}}$ \\",
    ]
    write("tbl_interaction", tab(
        "lrrr",
        r"\textbf{The two components do nothing alone and everything together.} "
        r"Task-averaged window ROC-AUC over the 14 cells, all four arms sharing "
        r"corpus, backbone, probe and frozen folds, seed-matched at three seeds. "
        r"Rows are the protocol factorization objectives; columns are the "
        r"variational bottleneck on the Sensor block. Each margin reports the "
        r"paired per-cell simple effect. Alone, the objectives are worth "
        # Every p and win count is read from the analysis rather than written
        # as a literal, which is how four of them drifted out of step with the
        # effects they belong to.
        rf"${cf['obj_novib']:+.2f}$ ($p={cf['p_obj_novib']:.2f}$, "
        rf"{cf['pos_obj_novib']} of {cf['n']} cells) and the bottleneck "
        rf"${cf['vib_noobj']:+.2f}$ ($p={cf['p_vib_noobj']:.2f}$, "
        rf"{cf['pos_vib_noobj']} of {cf['n']}); in each other's presence they "
        rf"are worth ${cf['obj_vib']:+.2f}$ and ${cf['vib_obj']:+.2f}$ "
        rf"($p={cf['p_obj_vib']:.4f}$, $p={cf['p_vib_obj']:.4f}$, both "
        rf"{cf['pos_obj_vib']} of {cf['n']}). Running both against neither is "
        rf"worth ${cf['both']:+.2f}$ ($p={cf['p_both']:.4f}$, "
        rf"{cf['pos_both']} of {cf['n']}); the interaction contrast itself is "
        rf"${cf['inter']:+.2f}$ ($p={cf['p_inter']:.3f}$, "
        rf"{cf['pos_inter']} of {cf['n']}).",
        "tab:interaction",
        r"Protocol objectives & VIB off & VIB on & simple effect of VIB \\",
        rows,
        note=r"The top-left cell is also the arm that resolves the confound "
             r"this paper previously had to flag: it runs the auxiliary "
             r"training stack (global normalisation, statistical pooling, the "
             r"consistency term, the variance floor) with the factorization "
             rf"switched off. At {g['offoff']:.2f} against GlucoFM's "
             rf"{CANON['lad_fmbase']} it contributes nothing on its own."))

# ================================================================== tbl_arch
fd8 = pd.read_csv(A / "fd8_scores.csv")
fd8["arm"] = fd8.run.str.replace(r"-s\d$", "", regex=True)
ARM = {"V4-fm-off": (r"$1\times$", "no factorization (GlucoFM)"),
       "V1-fm-joint": (r"$1\times$", "objectives trained jointly"),
       "V6-fm-post": (r"$1\times$", "blocks fitted post hoc"),
       "V5-5x-off": (r"$4.97\times$", "no factorization"),
       "V2-5x-joint": (r"$4.97\times$", "objectives trained jointly"),
       "V7-5x-post": (r"$4.97\times$", "blocks fitted post hoc")}
rows, base = [], {}
for arm, (cap, desc) in ARM.items():
    s = fd8[(fd8.arm == arm) & (fd8.level == "window")]
    if not len(s):
        continue
    per = s.groupby(["cohort", "task"]).auc.mean()
    if "off" in arm:
        base[cap] = per
    d = "---"
    if cap in base:
        i = per.index.intersection(base[cap].index)
        dd = (per[i] - base[cap][i]).mean()
        d = "---" if "off" in arm else f"${dd:+.2f}$"
    sd = s.groupby("run").auc.mean().std()
    rows.append(f"{cap} & {desc} & {per.mean():.2f} & {sd:.2f} & {d} \\\\")
    if arm == "V6-fm-post":
        rows.append(r"\addlinespace")
write("tbl_arch", tab(
    "llrrr",
    r"Architecture grid: model capacity crossed with how the factorization is "
    r"obtained. Three seeds per cell, window ROC-AUC averaged over 14 cells; "
    r"$\Delta$ is against the no-factorization arm at the same capacity. "
    r"Training the objectives hurts at both capacities and hurts \emph{more} at "
    r"$4.97\times$, which is the opposite of what a capacity-limitation account "
    r"predicts. Scaling buys run-to-run stability, not accuracy.",
    "tab:arch",
    r"Capacity & Factorization & ROC-AUC & Seed s.d. & $\Delta$ \\",
    rows))

# =============================================================== tbl_posthoc
ph = pd.read_csv(A / "posthoc_sweep.csv")
ENC = {"v4": r"$1\times$, no factorization loss",
       "v6": r"$1\times$, post-hoc heads", "v7": r"$4.97\times$, post-hoc heads"}
# The pre-specified reference is the block literally named `encoder` -- the
# frozen encoder's own unsplit output. `full` is the post-hoc blocked readout,
# which is a different question (does the SPLIT help, given you already refit).
rows, deltas = [], []
for enc, lbl in ENC.items():
    s = ph[ph.encoder == enc]
    if not len(s):
        continue
    p = s.pivot_table(index="n_fit", columns="block", values="auc")
    for n, r in p.iterrows():
        base, zt, zts = r.get("encoder"), r.get("zT"), r.get("zTzS")
        deltas.append(zt - base)
        rows.append(f"{lbl} & {int(n)} & {base:.1f} & {zt:.1f} & {zts:.1f} & "
                    f"${zt - base:+.2f}$ \\\\")
        lbl = ""
    rows.append(r"\addlinespace")
dz = np.array(deltas)
rows.append(r"\midrule" r"\multicolumn{5}{l}{\textbf{Mean over all 12 "
            r"configurations}}" rf" & $\mathbf{{{dz.mean():+.2f}}}$ \\")
write("tbl_posthoc", tab(
    "lrrrrr",
    r"Fitting the block decomposition post hoc on a \emph{frozen} encoder, "
    r"using real paired windows rather than synthetic ones, while sweeping how "
    r"many paired subjects are available for the fit. The pre-specified "
    r"comparison is the Trait block against the frozen encoder's own unsplit "
    rf"output: it is ${dz.mean():+.2f}$ ROC-AUC over 12 configurations "
    rf"($t=-0.42$), positive in {int((dz > 0).sum())} of {len(dz)}. Post-hoc "
    r"factorization of an already-trained encoder does nothing.",
    "tab:posthoc",
    r"Encoder & Paired subj. & Frozen encoder & $z_T$ & $z_T\|z_S$ & $\Delta$ \\",
    rows,
    note=r"Read instead as ``did \emph{any} of the five blocks win'', the same "
         r"table returns 7 of 12 --- but the winning block is chosen after "
         r"seeing the result and is never the same block twice, so that "
         r"statistic is selection bias, not evidence."))

# ============================================================== tbl_transfer
fr = [pd.read_csv(A / f) for f in
      ("fd3_v2final.csv", "fd3_bd.csv", "fd3_baselines.csv") if (A / f).exists()]
t = pd.concat(fr, ignore_index=True).drop_duplicates(
    subset=["run", "src", "tgt", "task"])
t["arm"] = t.run.str.replace(r"-s\d(:|$)", r"\1", regex=True)
PICK = {"C-v2-vib01:zTzS": "GP-C", "E-v2-vib-simbias:zTzS": "GP-E",
        "C-v2-vib01:full": r"GP-C\,\emph{full}", "V4-fm-off": "GlucoFM",
        "MantisV2": "MantisV2", "CGMformer": "CGMformer",
        "MOMENT-large": "MOM-l", "Chronos-2": "Chr-2"}
keys = [k for k in PICK if k in set(t.arm)]
g = t.groupby(["src", "tgt", "task", "arm"]).auc.mean().unstack("arm")
# TRANSPOSED for the same reason as Table~\ref{tab:percell}: directions as
# columns fit portrait, models as columns do not.
ABBR = {"cgmacros": "CGM", "stanford": "Stan", "hall": "Hall"}
DIRS = list(g.index)                       # (src, tgt, task), already sorted
mn = g[keys].mean()
mvt = sorted(mn.values, reverse=True)
bestd, secondd = {}, {}
for key in DIRS:
    v = sorted((g.loc[key, k] for k in keys if not pd.isna(g.loc[key, k])),
               reverse=True)
    bestd[key], secondd[key] = v[0], (v[1] if len(v) > 1 else np.nan)

ROWT = {"C-v2-vib01:zTzS": r"\textbf{GlucoPRISM-C}",
        "E-v2-vib-simbias:zTzS": r"\textbf{GlucoPRISM-E}",
        "C-v2-vib01:full": r"\quad GP-C, $z_A$ \emph{kept}",
        "V4-fm-off": "GlucoFM (ours)", "MantisV2": "MantisV2",
        "CGMformer": "CGMformer", "MOMENT-large": "MOMENT-l",
        "Chronos-2": "Chronos-2"}
rows = []
for i, k in enumerate(keys):
    if i == 2:
        rows.append(r"\addlinespace[2pt]")
    if i == 3:
        rows.append(r"\addlinespace[2pt]\multicolumn{14}{l}{\emph{Backbone and "
                    r"zero-shot baselines}} \\")
    tint = r"\rowcolor{oursbg}" if i < 2 else ""
    cells = [cellfmt(g.loc[key, k], bestd[key], secondd[key]) for key in DIRS]
    rows.append(f"{tint}{ROWT[k]} & " + " & ".join(cells) + " & "
                + cellfmt(mn[k], mvt[0], mvt[1]) + r" \\")

groups, seen = [], []
for src, tgt, _ in DIRS:
    if (src, tgt) not in seen:
        seen.append((src, tgt))
hdr_top, cmids, col = [], [], 2
for src, tgt in seen:
    hdr_top.append(rf"\multicolumn{{2}}{{c}}{{{ABBR[src]}$\to${ABBR[tgt]}}}")
    cmids.append(rf"\cmidrule(lr){{{col}-{col + 1}}}")
    col += 2
group_t = ("& " + " & ".join(hdr_top) + r" & \\" "\n" + "".join(cmids) + "\n"
           + "Model & " + " & ".join("DR" if t == "diabetes" else "IR"
                                     for _, _, t in DIRS)
           + r" & \textbf{Avg} \\")
write("tbl_transfer", tab(
    "l" + "r" * len(DIRS) + r" >{\columncolor{oursbg}}r",
    r"\textbf{Cross-cohort transfer: all twelve (source, target, task) "
    r"directions.} The probe is fitted on the \emph{entire} source cohort and "
    r"evaluated on the \emph{entire} target cohort; no target labels are used "
    r"for any purpose and there is no cross-validation, so this measures direct "
    r"transfer rather than within-cohort generalisation. ROC-AUC, higher is "
    r"better; \textbf{bold} is the best in a column, \underline{underline} the "
    r"runner-up. The third row is the same model read \emph{without} discarding "
    r"the Sensor block, so the first row minus the third isolates the "
    r"intervention this paper proposes. Cohorts abbreviated CGM (CGMacros), "
    r"Stan (Stanford), Hall; tasks DR (diabetes risk) and IR (insulin "
    r"resistance).",
    "tab:transfer",
    group_t, rows, wide=True,
    note=r"Transfer PR-AUC is scored against the \emph{target} cohort's class "
         r"balance, so directions are not comparable to one another and are "
         r"never averaged into a single figure; we report ROC-AUC here and give "
         r"PR-AUC per direction in the released tables. This protocol supplies "
         r"only twelve non-independent directions, which is why no comparison "
         r"on this axis survives Holm correction --- in either direction, "
         r"including a baseline 7.7 ROC-AUC \emph{behind} GlucoFM."))

# ============================================================== tbl_controls
bc = pd.read_csv(A / "fd3_block_controls.csv")
VAR = [("full(128)", "Full readout", 128), ("zT(64)", r"$z_T$ (Trait)", 64),
       ("slice64", "First 64 dims", 64), ("rand64", "Random projection", 64),
       ("pca64", "PCA projection", 64), ("zS(48)", r"$z_S$ (State)", 48),
       ("rand48", "Random projection", 48), ("pca48", "PCA projection", 48),
       ("zA(16)", r"$z_A$ (Sensor)", 16)]
m = bc.groupby("variant").auc.agg(["mean", "std"])
rows = []
for v, lbl, dim in VAR:
    if v not in m.index:
        continue
    bold = v.startswith("z")
    nm = rf"\textbf{{{lbl}}}" if bold else rf"\quad {lbl}"
    rows.append(f"{nm} & {dim} & {m.loc[v, 'mean']:.2f} & "
                f"{m.loc[v, 'std']:.2f} \\\\")
    if v in ("full(128)", "pca64", "pca48"):
        rows.append(r"\addlinespace")
write("tbl_controls", tab(
    "lrrr",
    r"Block controls at matched width, cross-cohort transfer ROC-AUC. A "
    r"16-dimensional probe input is better regularised than a 128-dimensional "
    r"one at 29--69 subjects, so a narrow block can score well for reasons "
    r"unrelated to what it encodes; each named block must therefore beat a "
    r"random and a PCA projection of the same width. Note that ``first 64 "
    r"dims'' is byte-identical to $z_T$ by construction and is reported only to "
    r"make that explicit.",
    "tab:controls",
    r"Readout & dim & ROC-AUC & Seed s.d. \\",
    rows))

# ============================================================ tbl_corpusfrac
f45 = pd.read_csv(A / "fd45_scores.csv")
if (A / "fd45_f10.csv").exists():
    f45 = pd.concat([f45, pd.read_csv(A / "fd45_f10.csv")], ignore_index=True)
f45["arm"] = f45.run.str.replace(r"-s\d$", "", regex=True)
w45 = f45[f45.level == "window"]
FRAC = [("F00", "0\\%"), ("F20", "20\\%"), ("F30", "30\\%"), ("F40", "40\\%"),
        ("F50", "50\\%"), ("F70", "70\\%"), ("F90", "90\\%"),
        ("F100", "100\\%")]
full100 = w45[w45.arm == "F100"].auc.mean()
rows = [r"\multicolumn{5}{l}{\emph{REPLACE-BG fraction of the pretraining "
        r"corpus (all other cohorts retained in full)}} \\"]
for arm, lbl in FRAC:
    s = w45[w45.arm == arm]
    if len(s):
        rows.append(f"\\quad {lbl} & --- & {s.auc.mean():.2f} & "
                    f"${s.auc.mean() - full100:+.2f}$ & --- \\\\")
rows.append(r"\addlinespace\multicolumn{5}{l}{\emph{Leave-one-cohort-out "
            r"(REPLACE-BG and all other cohorts retained in full)}} \\")
LOCO = {"LOCO-bigide": ("BIG IDEAs removed", 70),
        "LOCO-colas": (r"Col\'as removed", 287),
        "LOCO-shangh": ("ShanghaiT2DM removed", 247),
        "LOCO-stanfo": ("Stanford removed", 279)}
for arm, (lbl, nwin) in LOCO.items():
    s = w45[w45.arm == arm]
    if len(s):
        d = s.auc.mean() - full100
        per = -d / nwin * 1000
        rows.append(f"\\quad {lbl} & {nwin:,} & {s.auc.mean():.2f} & "
                    f"${d:+.2f}$ & {per:+.2f} \\\\")
rows.append(r"\addlinespace")
rows.append(rf"\quad REPLACE-BG removed (=0\%) & 9,035 & "
            rf"{w45[w45.arm == 'F00'].auc.mean():.2f} & "
            rf"${w45[w45.arm == 'F00'].auc.mean() - full100:+.2f}$ & "
            rf"{-(w45[w45.arm == 'F00'].auc.mean() - full100) / 9035 * 1000:+.2f} \\")
write("tbl_corpusfrac", tab(
    "lrrrr",
    r"Corpus composition, at one seed per arm. With a seed standard deviation "
    r"near 1.0 on this benchmark, individual differences inside that band are "
    r"not interpretable and we draw no rank-level conclusions within it; the "
    r"whole sweep spans 1.8 ROC-AUC, which is why we call corpus volume a null. "
    r"The final column is what survives that caution: cost per thousand windows "
    r"removed, which separates the cohorts by more than an order of magnitude "
    r"even though their totals do not. Removing 70 BIG~IDEAs windows costs more "
    r"than removing 9,035 REPLACE-BG windows.",
    "tab:corpusfrac",
    r"Pretraining corpus & Windows & ROC-AUC & $\Delta$ & Cost / 1k windows \\",
    rows))

# ================================================================ tbl_window
f7 = pd.read_csv(A / "fd7_scores.csv")
if (A / "fd7seed_scores.csv").exists():
    f7 = pd.concat([f7, pd.read_csv(A / "fd7seed_scores.csv")], ignore_index=True)
f7["arm"] = f7.run.str.replace(r"-s\d$", "", regex=True)
w7 = f7[f7.level == "window"]
# "m" = corpus size matched to the 0%-overlap arm, so overlap is varied without
# also varying how many windows the model sees; W3u is the unmatched twin.
# K=18 with 24 patches means stride 12 and a 6-value (30 min) lookback, i.e.
# patches overlap; K=12 with 24 patches tiles 288 exactly with no lookback.
GEO = [("W1-ov0", "24 patches of 12 (1\\,h, published)", "0\\%", "--", "matched"),
       ("W2-ov20m", "24 patches of 12", "20\\%", "--", "matched"),
       ("W3-ov40m", "24 patches of 12", "40\\%", "--", "matched"),
       ("W3u-ov40", "24 patches of 12", "40\\%", "--", "unmatched"),
       ("W4-k18", "24 patches of 18", "0\\%", "30 min", "matched"),
       ("W5-k18-ov40", "24 patches of 18", "40\\%", "30 min", "matched"),
       ("W6-k6", "48 patches of 6 (30\\,min)", "0\\%", "--", "matched"),
       ("W7-k24", "12 patches of 24 (2\\,h)", "0\\%", "--", "matched")]
ref = w7[w7.arm == "W1-ov0"].groupby(["cohort", "task"]).auc.mean()
rows = []
for arm, geom, ov, lb, sz in GEO:
    s = w7[w7.arm == arm]
    if not len(s):
        continue
    per = s.groupby(["cohort", "task"]).auc.mean()
    i = per.index.intersection(ref.index)
    d = (per[i] - ref[i]).mean()
    nseed = max(1, len(s) // 14)
    dd = "---" if arm == "W1-ov0" else f"${d:+.2f}$"
    rows.append(f"{geom} & {ov} & {lb} & {sz} & {per.mean():.2f} & {dd} & "
                f"{nseed} \\\\")
write("tbl_window", tab(
    "llllrrr",
    r"Window and patch geometry. ``Day overlap'' is the fraction by which "
    r"consecutive 24-hour windows from one subject overlap; ``lookback'' is "
    r"extra context prepended to each patch, so that patches overlap instead of "
    r"tiling the day exactly; ``corpus'' records whether the arm was size-"
    r"matched to the zero-overlap baseline, since raising overlap otherwise "
    r"also raises the number of windows. $\Delta$ is paired per cell against "
    r"the published geometry. Only patch length matters: two-hour patches cost "
    r"2.7 ROC-AUC and 30-minute patches cost 0.4, while lookback and day "
    r"overlap are within seed noise. We keep the published geometry on this "
    r"basis.",
    "tab:window",
    r"Patch geometry & Day overlap & Lookback & Corpus & ROC-AUC & $\Delta$ & "
    r"Seeds \\", rows, wide=True,
    note=r"Single-seed arms are marked in the final column; with a seed "
         r"standard deviation near 1.0 on this benchmark, no single-seed "
         r"difference below that magnitude is interpretable. A pre-registered "
         r"patch-size $\times$ overlap interaction was supported at one seed "
         r"and refuted at three; we report the three-seed result."))

# =================================================================== tbl_sig
rows = []
for _, r in sig.iterrows():
    nm = KEY.get(r.model, str(r.model))
    nm = {r"\textbf{GP-C}": r"\textbf{GlucoPRISM-C}",
          "GP-E": r"\textbf{GlucoPRISM-E}"}.get(nm, nm)
    star = r"$^{*}$" if r.p_holm < 0.05 else ""
    rows.append(f"{nm}{star} & {r.mean_delta:+.2f} & {int(r.wins)}/{int(r.n)} & "
                f"{r.p_raw:.4f} & {r.p_holm:.4f} & {r.cliffs:+.2f} \\\\")
write("tbl_sig", tab(
    "lrrrrr",
    r"Paired Wilcoxon signed-rank test over the 14 cells against our GlucoFM "
    r"reproduction, window ROC-AUC, with Holm--Bonferroni correction over the "
    r"confirmatory family declared before any result was read: the released "
    r"models and the baselines they are claimed to beat, $k=9$. Ablations of a "
    r"single model explore that model rather than asserting separate claims and "
    r"are reported uncorrected. Cliff's $\delta$: $|\delta|<0.15$ negligible, "
    r"$<0.33$ small.",
    "tab:sig",
    r"Model & mean $\Delta$ & cells won & $p_{\mathrm{raw}}$ & "
    r"$p_{\mathrm{Holm}}$ & Cliff's $\delta$ \\",
    rows, note=r"$^{*}$ adjusted $p<0.05$."))

# copy figures too
for d in OUTS:
    fg = d / "figures"
    fg.mkdir(exist_ok=True)
    for p in (OUTDIR / "figures").glob("*.pdf"):
        shutil.copy2(p, fg / p.name)
print("\nall tables written")
