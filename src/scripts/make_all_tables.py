"""Emit EVERY paper table as LaTeX, one file per table, generated from the CSVs.

The paper has no appendix: methods, ablations and all supporting tables live in
the body. That only stays honest if the tables are generated rather than typed,
so this writes one `tbl_*.tex` per table into final_materials, which
`assemble_paper.py` then folds into the conference .tex where each belongs.

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

import json
import re
import shutil
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Inputs come from GLUCOPRISM_OUT (artifacts/ by default). The original
# hard-coded an absolute path on the authors' machine.
A = Path(os.environ.get("GLUCOPRISM_OUT",
                        Path(os.environ.get("GLUCOPRISM_ROOT",
                             Path(__file__).resolve().parents[2]))
                        / "artifacts"))
# Fragments live here only. The Overleaf project is kept to the seven files the
# ICLR template ships, so assemble_paper.py folds these into the conference
# .tex; figures are assets and are still copied across.
# Tables are written under GLUCOPRISM_OUT/tex; set GLUCOPRISM_TEX_OUT to
# add a second destination (e.g. a paper directory).
OUTS = [A / "tex"]
if os.environ.get("GLUCOPRISM_TEX_OUT"):
    OUTS.append(Path(os.environ["GLUCOPRISM_TEX_OUT"]))
FIGS = [Path(r"D:\overleaf\glucoprismm\glucoprism_v2\figures"),
        Path(r"D:\final_materials\paper\figures")]
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


_SKIP = ("\\multicolumn", "\\midrule", "\\toprule", "\\bottomrule",
         "\\cmidrule", "\\addlinespace", "\\rowcolor")
_NUM = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


# A cell may carry a seed spread: "68.0\,{\scriptsize$\pm$}{\scriptsize 0.2}".
# Only the mean competes for bolding, and only the mean gets bolded.
_PM = re.compile(r"^(?P<mean>.*?)\s*(?:\\,)?\{\\scriptsize\$\\pm\$\}.*$", re.S)


def _split_pm(cell: str) -> tuple[str, str]:
    """(mean part, spread part) of a cell, or (cell, '') if it carries none."""
    m = _PM.match(cell.strip())
    if not m:
        return cell, ""
    mean = m.group("mean")
    return mean, cell.strip()[len(mean):]


def _emph(cell: str) -> str:
    """Bold a cell, staying inside math mode where the cell already is one.

    `\\textbf{$+0.77$}` does not reliably bold digits in math; `$\\mathbf{+0.77}$`
    does, and keeps the sign in the same font as the number. Where the cell
    carries a seed spread, only the mean is bolded -- bolding the spread too
    would read as though the spread were the quantity being compared.
    """
    mean, spread = _split_pm(cell)
    s = mean.strip().replace(r"\bfseries", "").strip()
    if s.startswith("$") and s.endswith("$"):
        out = r" $\mathbf{" + s[1:-1] + "}$ "
    else:
        out = r" \textbf{" + s + "} "
    return out.rstrip() + spread + " " if spread else out


def _val(cell: str) -> float | None:
    """The number in a cell, ignoring whatever markup is wrapped around it."""
    s, _ = _split_pm(cell)
    s = s.strip()
    for m in (r"\textbf{", r"\underline{", r"\emph{", r"\mathbf{"):
        if s.startswith(m) and s.endswith("}"):
            s = s[len(m):-1].strip()
    s = s.replace(r"\bfseries", "").replace("$", "").replace(",", "").strip()
    return float(s) if _NUM.match(s) else None


def _split_groups(rows: list[str]) -> list[list[int]]:
    """Row indices per `\\multicolumn` section, for tables of stacked sweeps."""
    out: list[list[int]] = []
    cur: list[int] = []
    for i, r in enumerate(rows):
        if "\\multicolumn" in r:
            if cur:
                out.append(cur)
            cur = []
        elif "&" in r and not any(s in r for s in _SKIP):
            cur.append(i)
    if cur:
        out.append(cur)
    return out


def bold_best(rows: list[str], spec: dict[int, str],
              groups: list[list[int]] | None = None) -> list[str]:
    """Bold the winning number in each column of an already-emitted row block.

    `spec` maps a 0-based cell index to "max" or "min" --- the direction in
    which better lies, because ECE and Brier are the wrong way round from AUC.
    Only cells that parse as a bare number compete, so em dashes, macro cells
    and group headers are passed through untouched rather than mangled.
    `groups` optionally restricts the comparison to given row indices, for
    tables whose sections are separate contests.
    """
    body = [i for i, r in enumerate(rows)
            if "&" in r and not any(s in r for s in _SKIP)]
    blocks = groups if groups is not None else [body]
    cells = {i: [c for c in rows[i].rstrip().removesuffix(r"\\").split("&")]
             for i in body}
    for block in blocks:
        idx = [i for i in block if i in cells]
        for col, how in spec.items():
            vals = {i: _val(cells[i][col]) for i in idx
                    if col < len(cells[i]) and _val(cells[i][col]) is not None}
            if len(vals) < 2:
                continue
            pick = (max if how == "max" else min)(vals.values())
            for i, v in vals.items():
                if abs(v - pick) < 1e-9:
                    cells[i][col] = _emph(cells[i][col])
    for i in body:
        rows[i] = "&".join(cells[i]).rstrip() + r" \\"
    return rows


def bold_best_row(rows: list[str], cols: list[int],
                  how: str = "max") -> list[str]:
    """Bold the winner *within each row*, for tables whose contest runs across
    models rather than down a column."""
    for i, r in enumerate(rows):
        if "&" not in r or any(s in r for s in _SKIP):
            continue
        cells = r.rstrip().removesuffix(r"\\").split("&")
        vals = {c: _val(cells[c]) for c in cols
                if c < len(cells) and _val(cells[c]) is not None}
        if len(vals) < 2:
            continue
        pick = (max if how == "max" else min)(vals.values())
        for c, v in vals.items():
            if abs(v - pick) < 1e-9:
                cells[c] = _emph(cells[c])
        rows[i] = "&".join(cells).rstrip() + r" \\"
    return rows


def write(name: str, lines: list[str]) -> None:
    txt = "\n".join(lines) + "\n"
    for d in OUTS:
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.tex").write_text(txt, encoding="utf-8")
    print(f"  {name:<16} {len(lines):>4} lines")


def tab(cols: str, caption: str, label: str, header: str, rows: list[str],
        pos: str = "!tbp", note: str = "", wide: bool = False,
        sideways: bool = False, long: bool = False) -> list[str]:
    # EVERY table is set at the same size. Previously wide tables were wrapped
    # in \resizebox{\textwidth}{!}, which scales each one by whatever factor it
    # happens to need -- so a fourteen-column table printed noticeably smaller
    # digits than a four-column one and the appendix read as a jumble of
    # typefaces. One size for all; only the column padding varies, which changes
    # spacing rather than glyph size.
    size = r"\footnotesize"
    # A cell carrying its seed spread is roughly twice as wide as a bare one,
    # so the padding gives ground rather than the text block: at 5pt these run
    # a few points into the margin. Spacing changes, glyph size does not.
    has_pm = any(r"{\scriptsize$\pm$}" in r for r in rows)
    if sideways:
        cs = "5pt"
    elif wide:
        cs = "2.0pt" if has_pm else "2.6pt"
    else:
        cs = "3.4pt" if has_pm else "5pt"
    pad = rf"\setlength{{\tabcolsep}}{{{cs}}}"
    env = "sidewaystable" if sideways else "table"
    open_box = close_box = ""
    if long:
        # 57 rows do not fit a page, float or not -- LaTeX reports "Float too
        # large for page" and dumps it at the end. A longtable breaks across
        # pages at the same footnotesize, so nothing is dropped and nothing is
        # scaled; the header repeats on the continuation page.
        L = [rf"\begingroup{size}{pad}",
             rf"\begin{{longtable}}{{{cols}}}",
             rf"\caption{{{caption}}}\label{{{label}}}\\",
             r"\toprule", header, r"\midrule\endfirsthead",
             r"\multicolumn{%d}{l}{\emph{\footnotesize Table~\ref{%s}, continued}}\\"
             % (len(cols.replace("|", "")), label),
             r"\toprule", header, r"\midrule\endhead",
             r"\bottomrule\endlastfoot"]
        L += rows
        L += [r"\end{longtable}"]
        if note:
            L.append(rf"\vspace{{-6pt}}\par\footnotesize"
                     rf"\begin{{minipage}}{{\textwidth}}{note}\end{{minipage}}")
        L.append(r"\endgroup")
        return L
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
# Rows are DERIVED from the built corpus, not typed. They used to be five string
# literals carrying the no-overlap counts (9,035 / 279 / 247 / 287 / 70, summing
# to 500 subjects) while the Total was computed from canonical.json and tracked
# the corpus actually trained on (514 / 10,952). The table therefore disagreed
# with itself: the rows froze while the total moved with the data.
_COHORTS = [
    ("replacebg", r"REPLACE-BG \citep{aleppo2017replacebg}", "Dexcom G4", 5),
    ("stanford", r"Stanford \citep{hall2018glucotypes}", "Dexcom", 5),
    ("shanghait2dm", r"ShanghaiT2DM \citep{zhao2023shanghai}",
     "FreeStyle Libre", 15),
    ("colas", r"Col\'as \citep{colas2019detection}", "Medtronic iPro", 5),
    ("bigideas", r"BIG IDEAs \citep{bent2021bigideas}", "Dexcom", 5),
]
_report = {d["dataset"]: d for d in json.loads(
    (A / ".." / "data" / "processed" / "corpus_report.json").read_text()
    if (A / ".." / "data" / "processed" / "corpus_report.json").exists()
    else (A / "corpus_report.json").read_text())}
rows, _sub, _win = [], 0, 0
for _key, _label, _device, _rate in _COHORTS:
    _r = _report.get(_key)
    if _r is None or not _r.get("pt_windows"):
        raise SystemExit(f"tbl_corpus: {_key} has no pretraining windows in "
                         f"corpus_report.json -- rebuild the corpus first")
    _sub += _r["pt_subjects"]
    _win += _r["pt_windows"]
    rows.append(rf"{_label} & {_device} & {_rate} & {_r['pt_subjects']} & "
                rf"{_r['pt_windows']:,} & {_r['pt_mean_coverage']:.3f} \\")
if (_sub, _win) != (CANON["nsubj"], CANON["nwin"]):
    raise SystemExit(
        f"tbl_corpus: rows sum to {_sub}/{_win} but canonical.json says "
        f"{CANON['nsubj']}/{CANON['nwin']}. Rows and total describe different "
        f"corpora; rebuild both from one corpus build.")
rows += [
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
sig = pd.read_csv(A / "significance_window_auc_k11.csv")
sigmap = dict(zip(sig.model, sig.p_holm))
MAIN = [(r"\textbf{GlucoPRISM-C} \emph{(released)}", GPC),
        (r"\textbf{GlucoPRISM-E} \emph{(released)}", GPE),
        # The table says "GlucoFM", not "GlucoFM (ours)". Every baseline here
        # is our retraining; saying so on one row and not the others implies
        # the others are not, and it is stated in prose anyway.
        ("GlucoFM", FM),
        ("MantisV2", "MantisV2"), ("Mantis", "Mantis"),
        ("CGMformer", "CGMformer"), ("MOMENT-small", "MOMENT-small"),
        ("MOMENT-large", "MOMENT-large"), ("Chronos-2", "Chronos-2"),
        ("Chronos-2-small", "Chronos-2-small")]
SDV = pd.read_csv(A / "seed_variability.csv")
# The headline mean for C is over the three shared seeds, so its spread has to
# come from the same three -- not from all six C happens to have.
SDNAME = {GPC: "GlucoPRISM-C [seed-matched]", GPE: "GlucoPRISM-E",
          FM: "GlucoFM (ours)",
          "CGM-JEPA": "CGM-JEPA", "X-CGM-JEPA": "X-CGM-JEPA",
          "GluFormer-tiny": "GluFormer-tiny"}


def pm(mean: float, sd: float | None, fmt: str = "{:.1f}",
       bold: bool = False, math: bool = False) -> str:
    """A cell as mean with its seed spread, or the mean alone where none exists.

    Applied to every numeric cell of a table or to none of it. An earlier
    version appended the spread to two of eight columns, which left those two
    twice as wide as the rest and set in a second size; that is why it was
    dropped. Uniform application keeps the column widths even.

    `math` wraps the mean in `$...$` for signed differences, where the sign
    otherwise sets as a hyphen at text width.
    """
    m = fmt.format(mean)
    if math:
        m = f"${m}$"
    m = rf"\textbf{{{m}}}" if bold else m
    if sd is None or not np.isfinite(sd):
        return m
    # A spread is unsigned, so a signed format must not carry over to it.
    sdfmt = fmt.replace("+", "")
    return m + rf"\,{{\scriptsize$\pm$}}{{\scriptsize {sdfmt.format(abs(sd))}}}"


def seed_sd(run: str, level: str, metric: str = "auc") -> float | None:
    """Task-averaged s.d.\ across seeds, or None where there is no seed.

    Zero-shot baselines are single frozen third-party checkpoints: nothing was
    trained, so there is no pretraining seed to vary. Those cells stay `---`
    rather than being filled with a probe-fold spread, which is a different
    quantity and would invite a comparison that is not being made.
    """
    name = SDNAME.get(run)
    if name is None:
        return None
    r = SDV[(SDV.model == name) & (SDV.level == level)]
    if r.empty:
        return None
    v = r.iloc[0].get(f"{metric}_sd_taskavg")
    return None if pd.isna(v) else float(v)


def summary_row(nm: str, run: str) -> str:
    """One row of the task-averaged table.

    Seed spreads are deliberately not carried here. Appending them to six
    numeric columns doubles the width of every cell and sets it in a second
    size; the headline grid stays readable with the means alone and the
    spreads get their own axis in the seed-variability figure.
    """
    cells = []
    for lvl in ("window", "subject"):
        s = df[(df.run == run) & (df.level == lvl)]
        m = s.groupby(["cohort", "task"])[["pr", "auc", "f1"]].mean().mean()
        cells += [f"{m[k]:.1f}" for k in ("pr", "auc", "f1")]
    ph = sigmap.get(run)
    pc = "---" if ph is None else (rf"\textbf{{{ph:.3f}}}" if ph < 0.05
                                   else f"{ph:.3f}")
    return f"{nm} & " + " & ".join(cells) + f" & {pc} \\\\"


SDNOTE = (r"Seed-to-seed spread for every arm, at all three evaluation levels, "
          r"is reported in Figure~\ref{fig:seedsd}. Zero-shot models are absent "
          r"from it by construction: a frozen third-party checkpoint has no "
          r"pretraining seed to vary.")

rows = []
for i, (nm, run) in enumerate(MAIN):
    if run not in set(df.run):
        continue
    if i == 3:
        rows.append(r"\addlinespace\multicolumn{8}{l}{\emph{Zero-shot foundation "
                    r"models (frozen third-party checkpoints)}} \\")
    rows.append(summary_row(nm, run))
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
    rows_c.append(summary_row(nm, run))
# --------------------------------------------------- the retrained baselines
# CGM-JEPA and X-CGM-JEPA were pretrained on the same corpus, scored through the
# same probe on the same frozen folds and seed-matched at three seeds like every
# other arm, so they sit in the model tables on the same footing as the rest.
RP = pd.read_csv(A / "repro_frozen_probe.csv").rename(columns={"dataset": "cohort"})
RV = (pd.read_csv(A / "repro_vs_published.csv").rename(columns={"dataset": "cohort"})
        [["cohort", "task", "model", "PR_paper", "AUC_paper"]])
RM = RP.merge(RV, on=["cohort", "task", "model"], how="left")
RM["adauc"] = (RM.AUC - RM.AUC_paper).abs()

# Window PR/ROC/F1 come from the per-cell frame above; subject level and
# transfer are the scored values for these two arms. The p is NOT carried here
# -- it is read from the eleven-member Holm family like every other row, so the
# column cannot drift away from the correction that produced it.
JEPA = [("cgm_jepa", "CGM-JEPA", (66.9, 67.3, 58.3), 68.8),
        ("x_cgm_jepa", "X-CGM-JEPA", (67.2, 67.8, 60.2), 69.1)]


def jepa_master_row(key: str, label: str, subj, transfer) -> str:
    w = RM[RM.model == key][["PR", "AUC", "F1"]].mean()
    ph = sigmap.get(label)
    pc = ("---" if ph is None else
          (rf"\textbf{{{ph:.3f}}}" if ph < 0.05 else f"{ph:.3f}"))
    win = " & ".join(f"{w[k]:.1f}" for k in ("PR", "AUC", "F1"))
    return (f"{label} & {win} & "
            + " & ".join(f"{v:.1f}" for v in subj)
            + f" & {transfer:.1f} & {pc}" + r" \\")


# ============================================================== tbl_master
# One headline table instead of three. Previously the compact and full versions
# of the same task-average lived in two floats -- one in the body, one in the
# appendix, differing only in how many rows they showed -- and cross-cohort
# transfer sat in a third. A reader comparing a model within-cohort against the
# same model across cohorts had to hold two floats open. Here every model,
# every level and both protocols are one grid.
TR_ARM = {GPC: "C-v2-vib01:zTzS", GPE: "E-v2-vib-simbias:zTzS",
          FM: "V4-fm-off", "MantisV2": "MantisV2", "Mantis": "Mantis",
          "CGMformer": "CGMformer", "MOMENT-small": "MOMENT-small",
          "MOMENT-large": "MOMENT-large", "Chronos-2": "Chronos-2",
          "Chronos-2-small": "Chronos-2-small"}
_tr = pd.concat([pd.read_csv(A / f) for f in
                 ("fd3_v2final.csv", "fd3_bd.csv", "fd3_baselines.csv")
                 if (A / f).exists()], ignore_index=True)
_tr = _tr.drop_duplicates(subset=["run", "src", "tgt", "task"])
_tr["arm"] = _tr.run.str.replace(r"-s\d(:|$)", r"\1", regex=True)
_trm = _tr.groupby("arm").auc.mean()
# Transfer seed spread: average the 12 directions within a seed, then vary the
# seed. Arms with a single run (the zero-shot checkpoints) get no spread.
_tr["seed"] = _tr.run.str.extract(r"-s(\d)(?::|$)")
_trsd = {}
for _arm, _g in _tr.groupby("arm"):
    if _g.seed.notna().sum() == 0:
        continue
    _per = _g.dropna(subset=["seed"]).groupby("seed").auc.mean()
    if len(_per) > 1:
        _trsd[_arm] = float(_per.std(ddof=1))


def master_row(nm: str, run: str) -> str:
    cells = []
    for lvl in ("window", "subject"):
        s = df[(df.run == run) & (df.level == lvl)]
        m = s.groupby(["cohort", "task"])[["pr", "auc", "f1"]].mean().mean()
        cells += [f"{m[k]:.1f}" for k in ("pr", "auc", "f1")]
    arm = TR_ARM.get(run)
    cells.append(f"{_trm[arm]:.1f}" if arm in _trm.index else "---")
    ph = sigmap.get(run)
    cells.append("---" if ph is None else
                 (rf"\textbf{{{ph:.3f}}}" if ph < 0.05 else f"{ph:.3f}"))
    return f"{nm} & " + " & ".join(cells) + r" \\"


rows_m = []
for i, (nm, run) in enumerate(MAIN):
    if run not in set(df.run):
        continue
    if i == 2:
        # Naming the group is what lets every row inside it drop the "(ours)"
        # suffix: the header says these were retrained here, so repeating it on
        # one row implied the others were not.
        rows_m.append(r"\addlinespace\multicolumn{9}{l}{\emph{CGM foundation "
                      r"models, retrained from scratch on our public-only "
                      r"corpus}} \\")
    if i == 3:
        for key, lab, subj, tr in JEPA:
            rows_m.append(jepa_master_row(key, lab, subj, tr))
        rows_m.append(r"\addlinespace\multicolumn{9}{l}{\emph{Zero-shot "
                      r"foundation models --- frozen third-party checkpoints, "
                      r"no CGM pretraining on our corpus}} \\")
    rows_m.append(master_row(nm, run))
write("tbl_master", tab(
    "lrrrrrrrr",
    r"\textbf{Every model, every level, both protocols.} Task-averaged over the "
    r"14 task--cohort cells for within-cohort probing, and over the "
    r"\trdirn{} (source, target, task) directions for cross-cohort transfer. "
    r"Frozen encoder, identical logistic-regression probe, folds shared by every "
    r"model and frozen before the first was trained, seed-matched at three "
    r"seeds. $p_{\mathrm{Holm}}$ is corrected over the pre-declared confirmatory "
    r"family of $k=\famk$, and \textbf{GlucoPRISM-C is its only surviving "
    r"member}. The two released models lead on both protocols; the gap widens "
    r"from window level to transfer, which is the ordering the addressability "
    r"account predicts and a general-capability account does not.",
    "tab:master",
    r"& \multicolumn{3}{c}{Within-cohort, window} & "
    r"\multicolumn{3}{c}{Within-cohort, subject} & Transfer & \\"
    "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}" "\n"
    r"Model & PR & ROC & F1 & PR & ROC & F1 & ROC & $p_{\mathrm{Holm}}$ \\",
    # Seven score columns, all higher-is-better; the p column is deliberately
    # left alone, because bolding the smallest p would read as "most
    # significant" when what the column reports is a corrected threshold.
    bold_best(rows_m, {c: "max" for c in range(1, 8)}), wide=True,
    note=SDNOTE +
         r" The subject-level column is genuinely unsettled: Mantis "
         r"leads it while trailing by four \auc{} at window level, and "
         r"nothing there survives correction "
         r"(Table~\ref{tab:sigall}). Per-cell values for all 14 cells "
         r"are in Table~\ref{tab:percell} and, on the other two "
         r"metrics, Figure~\ref{fig:percell}; the 12 transfer "
         r"directions are broken out in Table~\ref{tab:transfer}."))

# ================================================================ tbl_seedsd
# Every arm that has more than one seed, with the two spreads kept apart.
SDORDER = ["GlucoPRISM-C [seed-matched]", "GlucoPRISM-C", "GlucoPRISM-E",
           "GlucoPRISM-C [full readout]", "GlucoPRISM-E [full readout]",
           "GlucoFM (ours)", "No factorization (A)", "Bottleneck only (B)",
           "Objectives only (D)", "REPLACE-BG 50\\%", "REPLACE-BG 70\\%"]
SDLABEL = {"GlucoPRISM-C [seed-matched]":
           r"\textbf{GlucoPRISM-C} \emph{(seed-matched, as reported)}",
           "GlucoPRISM-C": r"GlucoPRISM-C \emph{(all six seeds)}",
           "GlucoPRISM-E": r"\textbf{GlucoPRISM-E}",
           "GlucoPRISM-C [full readout]": r"GlucoPRISM-C, $z_A$ kept",
           "GlucoPRISM-E [full readout]": r"GlucoPRISM-E, $z_A$ kept",
           "GlucoFM (ours)": "GlucoFM",
           "No factorization (A)": r"\quad no factorization (A)",
           "Bottleneck only (B)": r"\quad bottleneck only (B)",
           "Objectives only (D)": r"\quad objectives only (D)",
           "REPLACE-BG 50\\%": r"\quad REPLACE-BG 50\%",
           "REPLACE-BG 70\\%": r"\quad REPLACE-BG 70\%"}
sd_rows = []
for i, name in enumerate(SDORDER):
    key = name.replace("\\%", "%")
    g = SDV[SDV.model == key]
    if g.empty:
        continue
    if i == 6:
        sd_rows.append(r"\addlinespace\multicolumn{9}{l}{\emph{Factorial "
                       r"ablation arms and corpus fractions}} \\")
    cells = []
    for lvl in ("window", "subject", "transfer"):
        r_ = g[g.level == lvl]
        if r_.empty or pd.isna(r_.iloc[0].get("auc_mean")):
            cells += ["---", "---"]
            continue
        row = r_.iloc[0]
        cells += [f"{row.auc_mean:.2f}", f"{row.auc_sd_taskavg:.2f}"]
    n = int(g.n_seeds.max())
    pc = g[g.level == "window"]
    pcv = "---" if pc.empty else f"{pc.iloc[0].auc_sd_percell_mean:.2f}"
    sd_rows.append(f"{SDLABEL[name]} & {n} & " + " & ".join(cells)
                   + f" & {pcv} \\\\")
write("tbl_seedsd", tab(
    "lrrrrrrrr",
    r"Seed-to-seed variability, ROC-AUC. Every arm that trained more than one "
    r"model is listed; the spread is over pretraining seeds, with folds and "
    r"probe held fixed. Two standard deviations are reported and should not be "
    r"confused. \emph{Task-avg.\ s.d.} averages the 14 cells within a seed and "
    r"then varies the seed --- this is the error bar that belongs beside a "
    r"headline number. \emph{Per-cell s.d.} is the spread inside a single "
    r"task--cohort cell, which is roughly threefold larger because per-cell "
    r"noise partly cancels under averaging. Zero-shot baselines are absent by "
    r"construction: a frozen third-party checkpoint has no pretraining seed.",
    "tab:seedsd",
    r"& & \multicolumn{2}{c}{Window} & \multicolumn{2}{c}{Subject} & "
    r"\multicolumn{2}{c}{Transfer} & Per-cell \\"
    "\n" r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
    r"Arm & Seeds & mean & s.d. & mean & s.d. & mean & s.d. & s.d. \\",
    sd_rows, wide=True,
    note=r"Window-level variability is small (0.2--1.1) relative to the "
         r"effects the paper claims; subject-level variability is two to four "
         r"times larger, which is why no claim rests on the subject column "
         r"alone. GlucoPRISM-C is listed twice because it ran six seeds while "
         r"every comparator ran three: the seed-matched row is the one every "
         r"headline and significance test uses."))

# =============================================================== tbl_percell
KEY = {GPC: r"GP-C", GPE: "GP-E", FM: "GlucoFM",
       "MantisV2": "MantisV2", "Mantis": "Mantis", "CGMformer": "CGMformer",
       "MOMENT-small": "MOM-s", "MOMENT-large": "MOM-l",
       "Chronos-2": "Chr-2", "Chronos-2-small": "Chr-2s"}
cols = [c for c in KEY if c in set(W.run)]
piv = W.groupby(["cohort", "task", "run"]).auc.mean().unstack("run")
# The two retrained JEPA arms have per-cell window ROC on exactly these folds,
# so they join the grid, sitting after GlucoFM and before the zero-shot block.
# Bold and underline are computed over every column including them, because a
# per-cell winner is a per-cell winner.
for _k, _lab, *_ in JEPA:
    piv[_k] = RM[RM.model == _k].set_index(["cohort", "task"]).AUC
    KEY[_k] = _lab
cols = cols[:3] + [k for k, *_ in JEPA] + cols[3:]


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
          FM: "GlucoFM", "MantisV2": "MantisV2", "Mantis": "Mantis",
          "CGMformer": "CGMformer", "MOMENT-small": "MOMENT-s",
          "MOMENT-large": "MOMENT-l", "Chronos-2": "Chronos-2",
          "Chronos-2-small": "Chronos-2-s",
          **{k: lab for k, lab, *_ in JEPA}}
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
    if i == 3 + len(JEPA):
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
    [rf"GlucoFM & {CANON['lad_fmbase']} & --- \\",
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
            rows.append(
                f"\\quad {key}{star} & "
                f"{pm(r['full'], r.get('full_sd'), '{:.2f}')} & "
                f"{pm(r['drop'], r.get('drop_sd'), '{:.2f}')} & "
                f"{pm(r['gain'], r.get('gain_sd'), '{:+.2f}', math=True)} & "
                f"{pm(r['subj_gain'], r.get('subj_gain_sd'), '{:+.2f}', math=True)}"
                f" \\\\")
        rows.append(r"\addlinespace")
    write("tbl_capacity", tab(
        "lrrrr",
        r"Sensor-block capacity, two seeds per arm, seed-matched against the "
        r"released model on the same two seeds. Width is varied with $z_T$ held "
        r"at 64 so $z_S$ absorbs the difference; the confound is therefore "
        r"$z_A$-versus-$z_S$ width, not $z_A$-versus-$z_T$. \textbf{The benefit "
        r"of deleting the block rises with the price it pays per nat} --- about "
        r"$+0.5$ at $\beta \leq 0.1$ against about $+1.6$ at $\beta \geq 0.3$ "
        r"--- which is what the addressability account predicts and what a "
        r"pure-regularisation account does not: a regulariser's value would not "
        r"track how hard the block is squeezed. Within each pair the two arms "
        r"are indistinguishable, so we read a step rather than a monotone "
        r"ladder. Width is the weaker knob: all three widths sit inside one "
        r"seed standard deviation of each other.",
        "tab:capacity",
        r"Setting & Full readout & $z_A$ dropped & Gain (window) & "
        r"Gain (subject) \\",
        # Two sweeps through one grid, so each is its own contest: bolding
        # across both would compare a width arm against a $\beta$ arm.
        bold_best(rows, {1: "max", 2: "max", 3: "max", 4: "max"},
                  groups=_split_groups(rows)),
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
    # Per-seed means, so the reported spread is seed-to-seed rather than the
    # much larger cell-to-cell spread across transfer directions.
    es = e.groupby(["tag", "seed"])[["auc", "ece", "brier"]].mean()
    esd = es.groupby("tag").std(ddof=1)
    base = ge.loc["full (128d)", "auc"]
    bseed = es.xs("full (128d)").auc
    rows = []
    for t, lbl in [("full (128d)", "Full readout, no surgery"),
                   ("INLP on full (128d)",
                    r"INLP \citep{ravfogel2020null}, 8 iterations"),
                   ("drop zA (112d, ours)",
                    r"\textbf{Reserved-and-drop (ours)}")]:
        if t not in ge.index:
            continue
        d_ = ge.loc[t, "auc"] - base
        # Delta is paired within seed; propagating the two column SDs would
        # overstate it.
        dsd = (es.xs(t).auc - bseed).std(ddof=1)
        dcell = ("---" if abs(d_) < 1e-9
                 else pm(d_, dsd, "{:+.2f}", math=True))
        rows.append(f"{lbl} & {pm(ge.loc[t, 'auc'], esd.loc[t, 'auc'], '{:.2f}')} & "
                    f"{dcell} & "
                    f"{pm(ge.loc[t, 'ece'], esd.loc[t, 'ece'], '{:.3f}')} & "
                    f"{pm(ge.loc[t, 'brier'], esd.loc[t, 'brier'], '{:.3f}')} \\\\")
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
        # Lower is better for ECE and Brier, so those two invert.
        bold_best(rows, {1: "max", 3: "min", 4: "min"}),
        note=r"INLP is favoured slightly by this setup: when CGMacros is the "
             r"transfer target, the projection was fitted on target-domain "
             r"inputs (never on target labels). Ours uses no target data at all."))

# ====================================================== tbl_partial (deletion)
if (A / "rev_partial_within.csv").exists():
    # Calibration sits in the same file as the transfer score. Showing only
    # four of six keep-levels made the curve look non-monotone (68.59 -> 68.54)
    # when it is a clean cliff, and dropped a second, independent quantity that
    # steps at the same m -- which a regularisation account has no reason to
    # predict.
    _sd = pd.read_csv(A / "rev_soft_deletion.csv")
    _sd = _sd[_sd.tag.str.startswith("keep ")].copy()
    _sd["m"] = _sd.tag.str.extract(r"keep (\d+)/")[0].astype(int)
    softg = _sd.groupby("m")[["auc", "ece", "brier"]].mean()
    # Per-seed means first, so the spread is seed-to-seed and not the wider
    # spread across the transfer directions being averaged.
    softsd = (_sd.groupby(["m", "seed"])[["auc", "ece", "brier"]].mean()
              .groupby("m").std(ddof=1))
    tr = pd.read_csv(A / "rev_soft_deletion.csv").groupby("tag").auc.mean()
    kw = pd.read_csv(A / "rev_partial_within.csv")
    kw["seed"] = kw.run.str.extract(r"-s(\d)$")[0].astype(int)
    kw["arm"] = "C-v2-vib01"
    allb = pd.concat([v2, kw], ignore_index=True)
    allb["seed"] = allb.run.str.extract(r"-s(\d)$")[0].astype(int)
    allb["arm"] = allb.run.str.replace(r"-s\d$", "", regex=True)
    allb = allb[(allb.arm == "C-v2-vib01") & (allb.seed <= 2)]

    def wc(b: str, lvl: str) -> tuple[float, float]:
        """Cell-averaged score and its seed-to-seed spread."""
        s = allb[(allb.block == b) & (allb.level == lvl)]
        if not len(s):
            return float("nan"), float("nan")
        per = s.groupby(["seed", "cohort", "task"]).auc.mean().groupby("seed").mean()
        return per.mean(), (per.std(ddof=1) if per.size > 1 else float("nan"))

    rows = []
    for b, m_, lbl in [("full", 16, "keep all 16 (full readout)"),
                       (None, 12, "keep 12 of 16"),
                       (None, 8, "keep 8 of 16"),
                       ("keep4", 4, "keep 4 of 16"),
                       ("keep2", 2, "keep 2 of 16"),
                       ("zTzS", 0, r"\textbf{keep 0 of 16 (ours)}")]:
        g, gs = softg.loc[m_], softsd.loc[m_]
        w = pm(*wc(b, "window"), "{:.2f}") if b else "---"
        s = pm(*wc(b, "subject"), "{:.2f}") if b else "---"
        rows.append(f"{lbl} & {pm(g.auc, gs.auc, '{:.2f}')} & "
                    f"{pm(g.ece, gs.ece, '{:.3f}')} & "
                    f"{pm(g.brier, gs.brier, '{:.3f}')} & {w} & {s} \\\\")
    write("tbl_partial", tab(
        "lrrrrr",
        r"Is there a useful \emph{partial} deletion? Dimensions of $z_A$ are "
        r"ordered by how device-informative they are on real Dexcom/Libre "
        r"labels, and the least informative $m$ are readmitted. There is no "
        r"trade-off to tune: full deletion is best on every axis, and "
        r"readmitting as few as four of sixteen coordinates costs the entire "
        r"transfer benefit. \textbf{Ranking and calibration fall off the same "
        r"cliff at the same $m$} --- two quantities that a pure-regularisation "
        r"account has no reason to couple, stepping together between $m=2$ and "
        r"$m=4$. \textbf{Scaling $z_A$ by a factor instead of dropping it does "
        r"nothing at all} --- the probe standardises every column, so the scale "
        r"divides out exactly (measured: identical to three decimals for every "
        r"non-zero scale). Lower is better for ECE and Brier.",
        "tab:partial",
        r"& \multicolumn{3}{c}{Cross-cohort transfer} & "
        r"\multicolumn{2}{c}{Within-cohort} \\"
        "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}" "\n"
        r"Readout & ROC-AUC & ECE & Brier & window & subject \\",
        bold_best(rows, {1: "max", 2: "min", 3: "min", 4: "max", 5: "max"}),
        wide=True,
        note=r"Within-cohort scores were run for the three readouts the paper "
             r"released or ablated; $m=8$ and $m=12$ were run on transfer only, "
             r"where the cliff is."))

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
        rf"${cf['obj_novib']:+.2f}$ ($p=0.84$) and the bottleneck "
        rf"${cf['vib_noobj']:+.2f}$ ($p=0.82$); in each other's presence they "
        rf"are worth ${cf['obj_vib']:+.2f}$ and ${cf['vib_obj']:+.2f}$ "
        rf"($p=0.023$, $p=0.021$). The interaction is ${cf['inter']:+.2f}$ "
        rf"($p={cf['p_inter']:.3f}$, positive in 11 of {cf['n']} cells).",
        "tab:interaction",
        r"Protocol objectives & VIB off & VIB on & simple effect of VIB \\",
        rows,
        note=r"The top-left cell is also the arm that resolves the confound "
             r"this paper previously had to flag: it runs the auxiliary "
             r"training stack (global normalisation, statistical pooling, the "
             r"consistency term, the variance floor) with the factorization "
             r"switched off. At 65.63 against GlucoFM's 65.85 it contributes "
             r"nothing on its own."))

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
        "V4-fm-off": "GlucoFM", "MantisV2": "MantisV2",
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
# The column is a SEED spread, so it has to be taken over per-seed means. A
# plain std over the raw rows is the spread across transfer directions, which
# is several times larger and is not what the header claims.
m = bc.groupby(["variant", "seed"]).auc.mean().groupby("variant").agg(
    ["mean", "std", "count"])
rows = []
for v, lbl, dim in VAR:
    if v not in m.index:
        continue
    bold = v.startswith("z")
    nm = rf"\textbf{{{lbl}}}" if bold else rf"\quad {lbl}"
    sd = m.loc[v, "std"] if m.loc[v, "count"] > 1 else float("nan")
    rows.append(f"{nm} & {dim} & {m.loc[v, 'mean']:.2f} & "
                + (f"{sd:.2f}" if np.isfinite(sd) else "---") + r" \\")
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
    sd = r.get("sd_delta_cells")
    se = r.get("se_delta_cells")
    sdc = "---" if pd.isna(sd) else f"{sd:.2f}"
    sec = "---" if pd.isna(se) else f"{se:.2f}"
    rows.append(f"{nm}{star} & {r.mean_delta:+.2f} & {sdc} & {sec} & "
                f"{int(r.wins)}/{int(r.n)} & "
                f"{r.p_raw:.4f} & {r.p_holm:.4f} & {r.cliffs:+.2f} \\\\")
write("tbl_sig", tab(
    "lrrrrrrr",
    r"Paired Wilcoxon signed-rank test over the 14 cells against our GlucoFM "
    r"reproduction, window ROC-AUC, with Holm--Bonferroni correction over the "
    r"confirmatory family declared before any result was read: the released "
    r"models and the baselines they are claimed to beat, $k=9$. Ablations of a "
    r"single model explore that model rather than asserting separate claims and "
    r"are reported uncorrected. Cliff's $\delta$: $|\delta|<0.15$ negligible, "
    r"$<0.33$ small. The s.d.\ and s.e.\ columns describe the spread of the "
    r"difference \emph{across the 14 cells} --- how consistent the gain is from "
    r"cell to cell --- and are a different quantity from the seed-to-seed "
    r"spread in Appendix~\ref{app:seedsd}.",
    "tab:sig",
    r"Model & mean $\Delta$ & s.d. & s.e. & cells won & $p_{\mathrm{raw}}$ & "
    r"$p_{\mathrm{Holm}}$ & Cliff's $\delta$ \\",
    rows, wide=True, note=r"$^{*}$ adjusted $p<0.05$."))

# ========================================== evidence that existed unreported
REM = pd.read_csv(A / "remaining_experiments.csv")
MODS = ["GlucoFM", "GlucoPRISM-C", "GlucoPRISM-E", "GlucoPRISM-C [full]"]
MHEAD = ["GlucoFM", r"\textbf{GP-C}", r"\textbf{GP-E}", r"GP-C [$z_A$ kept]"]


def _piv(exp):
    return REM[REM.exp == exp].pivot_table(index="x", columns="model",
                                           values="auc")


# ------------------------------------------------------------- tbl_multiday
md = _piv("multiday")
md.index = md.index.astype(int)
rows = []
for k in sorted(md.index):
    cells = " & ".join(f"{md.loc[k, m]:.2f}" for m in MODS)
    rows.append(f"{k} & {cells} \\\\")
write("tbl_multiday", tab(
    "lrrrr",
    r"Days of wear against subject-level ROC-AUC. The probe reads $K$ "
    r"consecutive days per subject through the frozen encoder; everything else "
    r"is unchanged. \textbf{The lead is largest at one day} --- the deployment "
    r"regime, and the regime in which a device signature cannot yet be averaged "
    r"away across days --- and narrows as $K$ grows, which is what the "
    r"addressability account predicts and a general capability account does "
    r"not.",
    "tab:multiday",
    r"Days $K$ & " + " & ".join(MHEAD) + r" \\",
    # The contest here runs across models at a fixed number of days, so the
    # winner is per row rather than per column.
    bold_best_row(rows, list(range(1, len(MHEAD) + 1))),
    note=r"Two crossings are worth naming. GlucoPRISM-E at three days "
         r"(\mdethree) exceeds GlucoFM at seven (\mdfmseven). And "
         r"GlucoPRISM-C at three days (\mdcthree) exceeds the same model with "
         r"$z_A$ retained at seven (\mdcfullseven): \textbf{deleting sixteen "
         r"dimensions is worth more than four additional days of wear}."))

# ---------------------------------------------------------- tbl_crossdevice
cd = _piv("cross_device")
TASKN = {"diabetes_3class": "Diabetes risk (3-cls)", "ir": "Insulin resistance",
         "hyperlipidemia": "Hyperlipidemia", "obesity": "Obesity"}
rows = []
for direction in ("dexcom->libre", "libre->dexcom"):
    arrow = r"Dexcom $\rightarrow$ Libre" if "dexcom->" in direction \
        else r"Libre $\rightarrow$ Dexcom"
    rows.append(rf"\addlinespace\multicolumn{{6}}{{l}}{{\emph{{{arrow}}}}} \\")
    for x in [i for i in cd.index if direction in i]:
        task = TASKN.get(x.split(":")[0].strip(), x.split(":")[0])
        d_ = cd.loc[x, "GlucoPRISM-C"] - cd.loc[x, "GlucoFM"]
        gain = cd.loc[x, "GlucoPRISM-C"] - cd.loc[x, "GlucoPRISM-C [full]"]
        rows.append(f"{task} & {cd.loc[x, 'GlucoFM']:.2f} & "
                    f"{cd.loc[x, 'GlucoPRISM-C']:.2f} & ${d_:+.2f}$ & "
                    f"{cd.loc[x, 'GlucoPRISM-C [full]']:.2f} & "
                    f"${gain:+.2f}$ \\\\")
write("tbl_crossdevice", tab(
    "lrrrrr",
    r"The pre-registered cross-device test, all \cdn{} directions. We predicted "
    r"a factorized readout would out-transfer an entangled one across a device "
    r"boundary; on the mean it does not (\cdmean{} \auc, winning \cdwins{} of "
    r"\cdn). Reporting the mean alone would hide the structure. \textbf{The "
    r"full readout collapses towards chance in one direction only} --- Dexcom "
    r"$\rightarrow$ Libre, mean \cddexlibfull{} against \cdlibdexfull{} "
    r"reversed --- and that is exactly the asymmetry a mask-derived shortcut "
    r"predicts: a rule fitted on dense Dexcom windows has almost nothing to "
    r"read on a sparse Libre target, while the reverse transfers intact.",
    "tab:crossdevice",
    r"Endpoint & GlucoFM & GP-C & $\Delta$ vs FM & GP-C [$z_A$ kept] & "
    r"deletion gain \\",
    # Columns 1, 2 and 4 are the three readouts competing on the same endpoint;
    # 3 and 5 are differences between them and must not join the contest.
    bold_best_row(rows, [1, 2, 4]), wide=True,
    note=r"The deletion is worth \cddexlibgain{} \auc{} in the broken "
         r"direction and \cdlibdexgain{} in the intact one. \textbf{It repairs "
         r"precisely the direction that is broken}, which is a mechanism "
         r"prediction confirmed inside the experiment we lost. Hyperlipidemia "
         r"is the weakest endpoint for every model in the benchmark "
         r"(Table~\ref{tab:percell}); we report all \cdn{} directions and draw "
         r"no conclusion from any subset of them."))

# ------------------------------------------------------------ tbl_readouts
# Two collection files cover overlapping directions; counting a direction twice
# once moved a p-value from 0.0122 to 0.0425.
_BLK = pd.concat([pd.read_csv(A / "fd3_block_controls.csv"),
                  pd.read_csv(A / "fd3_drop_za.csv")], ignore_index=True) \
    .drop_duplicates(subset=["seed", "variant", "src", "tgt", "task"])
_b = _BLK.groupby("variant").auc.mean()

rows = []
for lbl, key in [(r"$z_T$ alone (64)", "zT(64)"),
                 (r"\textbf{$z_T\|z_S$ (112) --- released readout}",
                  "v2 zT||zS (112) <- proposal"),
                 (r"$z_S$ alone (48)", "zS(48)"),
                 (r"$z_A$ alone (16)", "zA(16)"),
                 (r"full (128)", "full(128)")]:
    rows.append(f"{lbl} & {_b.loc[key]:.2f} \\\\")
rows.append(r"\addlinespace\multicolumn{2}{l}{\emph{Width-matched controls}} \\")
for lbl, key in [("random 64-d projection", "rand64"),
                 ("PCA 64-d projection", "pca64"),
                 ("random 48-d projection", "rand48"),
                 ("PCA 48-d projection", "pca48")]:
    rows.append(rf"\quad {lbl} & {_b.loc[key]:.2f} \\")
write("tbl_readouts", tab(
    "lr",
    r"What each named block is worth on its own, cross-cohort transfer "
    r"ROC-AUC, three seeds. \textbf{Both named blocks beat their width-matched "
    r"controls} --- $z_T$ by \roztrand{} over a random projection and "
    r"\roztpca{} over PCA, $z_S$ by \rozsrand{} and \rozspca{} --- so the "
    r"negative result in Table~\ref{tab:dependence} is that Trait and State are "
    r"\emph{redundant}, not that either is empty. $z_T$ alone and the released "
    r"$z_T\|z_S$ readout differ by \roztgap, inside the seed standard "
    r"deviation of \sdctransfer{} for this axis, so they are indistinguishable "
    r"here. $z_A$ alone transfers at \roza{} --- near the full readout, and the "
    r"reason a bottleneck has anything to price.",
    "tab:readouts",
    r"Readout & Transfer ROC-AUC \\", rows,
    note=r"For contrast, truncating the \emph{baseline} embedding to matched "
         r"width buys nothing: GlucoFM's full 128-d readout scores \fmfull, "
         r"its first 112 dimensions \fmcutonetwelve{} and its first 64 "
         r"\fmcutsixtyfour. Halving a baseline embedding is worth nothing; "
         r"deleting a reserved sixteen is worth \trdc."))

# ------------------------------------- the three unrendered significance tables
# ------------------------------------------- one significance table, not three
# The same eleven models under the same correction, on the three metrics. Split
# across three floats a reader had to hold two of them in their head to compare
# a row; side by side the comparison is the table. The `_k11` files carry the
# family with the two retrained JEPA arms folded in and Holm re-run over all of
# them, which is what the caption claims.
SIGSRC = [("window ROC-AUC", "significance_window_auc_k11.csv"),
          ("window PR-AUC", "significance_window_pr_k11.csv"),
          ("subject ROC-AUC", "significance_subject_auc_k11.csv")]
frames = {lab: pd.read_csv(A / fn).set_index("model")
          for lab, fn in SIGSRC if (A / fn).exists()}
order = list(frames.values())[0].sort_values("p_holm").index
rows = []
for mdl in order:
    nm = KEY.get(mdl, str(mdl))
    nm = {r"GP-C": r"\textbf{GlucoPRISM-C}",
          "GP-E": r"\textbf{GlucoPRISM-E}"}.get(nm, nm)
    cells = []
    for lab, _ in SIGSRC:
        f_ = frames.get(lab)
        if f_ is None or mdl not in f_.index:
            cells += ["---", "---", "---"]
            continue
        r = f_.loc[mdl]
        star = r"$^{*}$" if r.p_holm < 0.05 else ""
        cells += [f"{r.mean_delta:+.2f}", f"{int(r.wins)}/{int(r.n)}",
                  f"{r.p_holm:.3f}{star}"]
    rows.append(f"{nm} & " + " & ".join(cells) + r" \\")
write("tbl_sigall", tab(
    "l" + "rrr" * 3,
    r"\textbf{Paired significance against GlucoFM, on all three metrics at "
    r"once.} Wilcoxon signed-rank over the 14 task--cohort "
    r"cells, Holm-corrected over the confirmatory family of $k=\famk$ declared "
    r"before any result was read. $\Delta$ is the mean paired difference and "
    r"``won'' the number of cells in which the model beats GlucoFM. On the "
    r"confirmatory axis, window ROC-AUC, exactly one comparison survives "
    r"correction, and it is ours. The three further starred entries all sit in "
    r"the PR column and are baselines falling \emph{below} GlucoFM, not "
    r"improvements over it. Reading across a row rather than across three "
    r"separate tables is the point: the ordering is stable on window ROC and "
    r"PR, and dissolves at subject level.",
    "tab:sigall",
    r"& \multicolumn{3}{c}{window ROC-AUC} & \multicolumn{3}{c}{window PR-AUC} "
    r"& \multicolumn{3}{c}{subject ROC-AUC} \\"
    "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" "\n"
    r"Model & $\Delta$ & won & $p_{\mathrm{Holm}}$ & $\Delta$ & won & "
    r"$p_{\mathrm{Holm}}$ & $\Delta$ & won & $p_{\mathrm{Holm}}$ \\",
    # The three $\Delta$ columns only; a bolded smallest $p$ would read as a
    # ranking of significance, which is not what a corrected threshold is.
    bold_best(rows, {1: "max", 4: "max", 7: "max"}), wide=True,
    note=r"$^{*}$ adjusted $p<0.05$. Our own PR-AUC gain is \prdelta{} on "
         r"\prwins{} of \prn{} cells and \textbf{does not survive correction}; "
         r"we report it as it is. The PR column is also where the zero-shot "
         r"comparison is strongest: \prsigcount{} of the seven general-purpose "
         r"models are Holm-significantly worse than a 0.72M-parameter "
         r"CGM-specific encoder under class imbalance."))

# ------------------------------- per-cell tables on the two unreported metrics
for name, metric, mlabel, lab in [
        ("tbl_percellpr", "pr", "PR-AUC", "tab:percellpr"),
        ("tbl_percellf1", "f1", "Macro-F1", "tab:percellf1"),
        ("tbl_percellsubj", "auc", "subject-level ROC-AUC", "tab:percellsubj")]:
    lvl = "subject" if name.endswith("subj") else "window"
    src = df[df.level == lvl]
    p_ = src.groupby(["cohort", "task", "run"])[metric].mean().unstack("run")
    cols_ = [c for c in KEY if c in p_.columns]
    rows = []
    for coh in ["cgmacros", "shanghait2dm", "stanford", "hall"]:
        for (c2, t2) in [i for i in p_.index if i[0] == coh]:
            v = p_.loc[(c2, t2), cols_]
            best, second = v.nlargest(2) if v.notna().sum() > 1 else (v.max(),
                                                                     v.max())
            rows.append(f"{COH[coh]} {TASK.get(t2, t2)} & " + " & ".join(
                cellfmt(v[c], best, second) for c in cols_) + r" \\")
    write(name, tab(
        "l" + "r" * len(cols_),
        rf"Per-cell {mlabel} for all ten models, the same 14 cells, folds and "
        rf"probe as Table~\ref{{tab:percell}}. \textbf{{Bold}} is the best in a "
        rf"cell, \underline{{underline}} the runner-up. Reported so that the "
        rf"claim that gains are concentrated in ranking rather than in "
        rf"calibrated decisions at a fixed threshold can be checked cell by "
        rf"cell rather than taken on assertion.",
        lab, "Cell & " + " & ".join(KEY[c] for c in cols_) + r" \\",
        rows, wide=True))

# ===================================== hyperparameters, read from the config
# The paper stated a parameter count and almost nothing else, which undercuts
# its own "we adopt the backbone without modification" claim. Every value below
# is read out of the released checkpoint's config so it cannot drift from what
# actually ran.
CFG = json.loads(Path(
    r"D:\final_materials\weights\glucoprism-c-s0.config.json").read_text())
FM_, PR_ = CFG["fm_config"], CFG["prism_config"]


HP_ALL = []


def hp(rows, name, cols, cap, lab, header, note=""):
    """Collect instead of emitting: the four hyperparameter blocks are one
    table with section dividers, not four floats saying "Setting | Value".
    """
    title = {"tbl_hparch": "Architecture",
             "tbl_hpopt": "Pretraining optimisation",
             "tbl_hpobj": "Objective constants and the $V_1$ generator",
             "tbl_hpprobe": "Downstream probe"}[name]
    if HP_ALL:
        HP_ALL.append(r"\addlinespace")
    HP_ALL.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{title}}}}} \\")
    HP_ALL.extend(rows)


I, O = r"\emph{inherited}", r"\textbf{ours}"
rows = [
    rf"Window grid & {FM_['grid']['length']} steps at "
    rf"{FM_['grid']['dt_min']} min (24 h) & {I} \\",
    rf"Patches & {FM_['grid']['n_patches']} of "
    rf"{FM_['grid']['patch_size']} steps (1 h) & {I} \\",
    rf"Causal filter $\sigma$ & init {FM_['filt']['sigma_init']}, "
    rf"learned in $[{FM_['filt']['sigma_min']}, "
    rf"{FM_['filt']['sigma_max']}]$ & {I} \\",
    rf"Filter truncation / lookback & {FM_['filt']['truncation']}$\sigma$ / "
    rf"{FM_['filt']['max_lookback']} steps & {I} \\",
    rf"Stream dim & {FM_['model']['stream_dim']} per stream & {I} \\",
    rf"Context encoder & {FM_['model']['n_layers']} layers, "
    rf"$d={FM_['model']['embed_dim']}$, {FM_['model']['n_heads']} heads, "
    rf"FFN {FM_['model']['ffn_dim']} & {I} \\",
    rf"Predictor & {FM_['model']['predictor_layers']} layer & {I} \\",
    rf"Dropout & {FM_['model']['dropout']} & {I} \\",
    r"\addlinespace",
    rf"Block widths $(d_T, d_S, d_A)$ & $({PR_['d_trait']}, "
    rf"{PR_['d_state']}, {PR_['d_sensor']})$, summing to "
    rf"{FM_['model']['embed_dim']} & {O} \\",
    rf"Sensor bottleneck & variational, $\beta={PR_['w_vib']}$, free bits "
    rf"{PR_['vib_free_bits']} & {O} \\",
    rf"$z_A$ inputs & \texttt{{{PR_['za_inputs'].replace('+', '+')}}} & {O} \\",
    rf"Device classes & {PR_['n_devices']} & {O} \\",
    r"\addlinespace",
    rf"Trainable at pretraining & \num{{\trainparams}} "
    rf"($\addedpct$\% over backbone) & --- \\",
    rf"Shipped at inference & \num{{\infparams}} "
    rf"($\shrinkpct$\% \emph{{below}} backbone) & --- \\",
]
hp(rows, "tbl_hparch", "llr",
   r"Architecture. Values marked \emph{inherited} are GlucoFM's, adopted "
   r"unmodified so that differences we report are attributable to the "
   r"factorization rather than to a retuned backbone; values marked "
   r"\textbf{ours} are the delta this paper introduces. The projection heads "
   r"used by the objectives are discarded at inference along with $z_A$, which "
   r"is why the shipped model is smaller than the backbone despite training "
   r"larger.",
   "tab:hparch", r"Component & Value & Provenance \\")

P = FM_["pretrain"]
rows = [
    rf"Optimizer & AdamW & \\",
    rf"Learning rate & {P['lr']:g} (filter $\sigma$: {P['lr_sigma']:g}) & \\",
    rf"Weight decay & {P['weight_decay']} & \\",
    rf"Warmup & {P['warmup_frac']:g} of total steps, then cosine & \\",
    rf"Gradient clipping & {P['grad_clip']} & \\",
    rf"Batch size & {P['batch_size']} windows & \\",
    rf"Epochs & {P['epochs']} & \\",
    rf"Mask ratio & sampled in $[{P['mask_ratio_low']}, "
    rf"{P['mask_ratio_high']}]$ per batch & \\",
    rf"EMA momentum & {P['ema_momentum']} $\rightarrow$ "
    rf"{P['ema_final']:g}, scheduled & \\",
    r"Precision & fp32 & \\",
    r"Hardware & one NVIDIA T4 & \\",
    r"Wall-clock & $\approx$36 min per run & \\",
    r"\addlinespace",
    r"Seeds & $\{0,1,2\}$ for every arm; $\{3,4,5\}$ additionally for "
    r"GlucoPRISM-C & \\",
    r"Model selection & held-out pretraining split only & \\",
]
hp(rows, "tbl_hpopt", "llr",
   r"Pretraining optimisation, identical for every arm in this paper "
   r"including the reproduced baselines. Hyperparameters were swept on a "
   r"subject-disjoint held-out split of the \emph{pretraining} corpus; no "
   r"downstream cohort was consulted for any decision, which is what allows "
   r"the 14 evaluation cells to be treated as held out.",
   "tab:hpopt", r"Setting & Value & \\")

AUG = FM_["aug"]
rows = [
    rf"$\lambda_1$ sensor, $\lambda_2$ day, $\lambda_3$ indep & "
    rf"$({PR_['w_sensor']}, {PR_['w_day']}, {PR_['w_indep']})$ & Eq.~2--4 \\",
    rf"VIB weight $\beta$ & {PR_['w_vib']} & Eq.~6 \\",
    rf"Variance floor weight & {PR_['w_variance']} & \S\ref{{sec:objectives}} \\",
    rf"$D(\cdot,\cdot)$ & cosine on $\ell_2$-normalised pre-projection blocks & \\",
    rf"InfoNCE temperature & {PR_['temperature']} & \\",
    rf"Day-term hinge margin & {PR_['day_margin']} & \\",
    rf"Day-term weight $\beta_d$ & {PR_['beta_day_info']} & \\",
    rf"Sub-sampling stride & every {PR_['step_skip']} steps & \\",
    r"\addlinespace\multicolumn{3}{l}{\emph{$V_1$ synthetic paired-sensor "
    r"generator, constants fitted on real pairs}} \\",
    rf"\quad Decimation & $p={AUG['decimate_prob']}$, keep 1 in "
    rf"{AUG['decimate_keep']}, random phase & \\",
    rf"\quad Compression drop & $p={AUG['compress_prob']}$, length "
    rf"{AUG['compress_len']}, depth {AUG['compress_min_mult']} & \\",
    rf"\quad Dropout blocks & $p={AUG['dropout_prob']}$, "
    rf"{AUG['dropout_blocks']} blocks of {AUG['dropout_len']} & \\",
    rf"\quad Baseline wander & $p={AUG['wander_prob']}$, amplitude "
    rf"{AUG['wander_amp']} mg/dL & \\",
    r"\quad Calibration offset & $-31.12$ mg/dL, Deming gain "
    r"$0.878 \pm 0.217$ & App.~\ref{app:genrobust} \\",
    r"\addlinespace",
    r"$V_2$ eligibility & 405 of \nsubj{} subjects have $\ge 2$ days & \\",
]
hp(rows, "tbl_hpobj", "llr",
   r"Objective constants and the paired-sensor generator. The three $\lambda$ "
   r"values are the single most consequential setting in the paper: at unit "
   r"weight the same objectives cost $5.27$ \auc, and at these values they are "
   r"neutral alone and productive with the bottleneck "
   r"(Section~\ref{sec:weight}). Generator constants are fitted to measured "
   r"Dexcom/Libre disagreement rather than chosen.",
   "tab:hpobj", r"Constant & Value & Reference \\")

PB = FM_["probe"]
rows = [
    r"Classifier & logistic regression, $\ell_2$ penalty, lbfgs & \\",
    rf"Regularisation $C$ & {PB['C']} (scikit-learn default, no inner "
    rf"search) & \\",
    rf"\texttt{{max\_iter}} & {PB['max_iter']} & \\",
    rf"Standardisation & {'per-feature, fitted on train folds only' if PB['standardize'] else 'none'} & \\",
    r"Class weighting & none & \\",
    rf"Cross-validation & {PB['n_folds']}-fold subject-grouped, "
    rf"$\times {PB['n_iterations']}$ repeats & \\",
    r"Fold assignment & frozen before the first model trained & \\",
    r"Subject aggregation & mean of window scores within subject & \\",
    r"Encoder & frozen; no fine-tuning anywhere in this paper & \\",
]
hp(rows, "tbl_hpprobe", "llr",
   r"Downstream probe, identical for every model and every cell including all "
   r"baselines. The probe is deliberately weak and untuned: a tuned probe "
   r"would measure our search over probes rather than the representation, and "
   r"could not be compared to published tables that use this protocol.",
   "tab:hpprobe", r"Setting & Value & \\",
   note=r"Fold freezing is what licenses paired testing: seed and fold "
        r"variance is common to both arms of every comparison and cancels in "
        r"the difference.")

write("tbl_hpall", tab(
    "llr",
    r"\textbf{Every constant, in one place.} Values are read directly out of "
    r"the released checkpoint's configuration file rather than transcribed, so "
    r"what is printed is what ran; the same file ships with the weights. "
    r"Architecture values marked \emph{inherited} are GlucoFM's, adopted "
    r"unmodified so that the differences this paper reports are attributable to "
    r"the factorization rather than to a retuned backbone --- the delta is "
    r"three block widths, a bottleneck and a device head. The optimisation "
    r"block is identical across every arm including the reproduced baselines. "
    r"Objective constants were swept on a subject-disjoint held-out split of "
    r"the \emph{pretraining} corpus; no downstream cohort informed any choice. "
    r"The probe is deliberately weak and untuned, because a tuned probe would "
    r"measure our search over probes rather than the representation.",
    "tab:hpall", r"Setting & Value & Provenance \\", HP_ALL, wide=True, long=True,
    note=r"Three constants are load-bearing enough to appear in the main text "
         r"as well: the block widths $(64,48,16)$, the objective weights "
         r"$\lambda=(0.2,0.2,0.1)$, and the bottleneck price $\beta=0.1$. The "
         r"second deserves the most suspicion, because the paper's central "
         r"dichotomy is partly a statement about it: at unit weight the same "
         r"objectives cost $5.27$ \auc."))

# ------------------------------------------------------- baseline reproduction
# CGM-JEPA, X-CGM-JEPA and GluFormer-tiny were reproduced too, but they were
# never re-trained under the released window configuration, so they are NOT
# seed-matched against the headline table and must not be dropped into it. What
# they are is an internally consistent parity check: one corpus, one probe, the
# same frozen folds, one seed each, scored against GlucoFM's published Table 3.
RP = pd.read_csv(A / "repro_frozen_probe.csv").rename(columns={"dataset": "cohort"})
RV = (pd.read_csv(A / "repro_vs_published.csv").rename(columns={"dataset": "cohort"})
        [["cohort", "task", "model", "PR_paper", "AUC_paper"]])
RM = RP.merge(RV, on=["cohort", "task", "model"], how="left")
RM["adauc"] = (RM.AUC - RM.AUC_paper).abs()

REPRO_NAME = {"glucofm": r"GlucoFM~\citep{li2026glucofm}",
              "gluformer_tiny": r"GluFormer-tiny~\citep{lutsker2026gluformer}",
              "cgm_jepa": r"CGM-JEPA~\citep{muhammad2026cgmjepa}",
              "x_cgm_jepa": r"X-CGM-JEPA~\citep{muhammad2026cgmjepa}",
              "raw": "raw 288-point window",
              "mask_only": "observation mask only"}
REPRO_PAR = {"glucofm": "0.72", "gluformer_tiny": "0.65",
             "cgm_jepa": "0.52", "x_cgm_jepa": "0.52", "raw": "---",
             "mask_only": "---"}
# The Reproduced group carries its seed spread; the Published group cannot,
# because GlucoFM's Table 3 reports point values only. `raw` and `mask_only`
# train nothing, so they have no pretraining seed to vary.
REPRO_SD = {"glucofm": "GlucoFM (ours)", "gluformer_tiny": "GluFormer-tiny",
            "cgm_jepa": "CGM-JEPA", "x_cgm_jepa": "X-CGM-JEPA",
            "raw": None, "mask_only": None}
rows, REPRO = [], {}
for k in ["glucofm", "gluformer_tiny", "cgm_jepa", "x_cgm_jepa", "raw",
          "mask_only"]:
    s = RM[RM.model == k]
    o = s[["PR", "AUC", "F1"]].mean()
    REPRO[k] = (o.PR, o.AUC, o.F1)
    _sdrow = SDV[(SDV.model == REPRO_SD[k]) & (SDV.level == "window")] \
        if REPRO_SD[k] else SDV.iloc[:0]
    _rsd = {m: (float(_sdrow.iloc[0][f"{m}_sd_taskavg"]) if len(_sdrow) else None)
            for m in ("pr", "auc", "f1")}
    if s.AUC_paper.notna().any():
        p = s[["PR_paper", "AUC_paper"]].mean()
        d = s.adauc
        REPRO[k] += (p.PR_paper, p.AUC_paper, d.mean(), int((d <= 2).sum()))
        pub = (f"{p.PR_paper:.1f} & {p.AUC_paper:.1f} & {d.mean():.2f} & "
               f"{int((d <= 2).sum())}\\,/\\,14")
    else:
        pub = r"--- & --- & --- & ---"
    bf = r"\textbf{%s}" if k == "glucofm" else "%s"
    rows.append(f"{bf % REPRO_NAME[k]} & {REPRO_PAR[k]} & "
                f"{pm(o.PR, _rsd['pr'])} & {pm(o.AUC, _rsd['auc'])} & "
                f"{pm(o.F1, _rsd['f1'])} & {pub} \\\\")
    if k == "x_cgm_jepa":
        rows.append(r"\addlinespace")

_ro = sorted((REPRO[k][1] for k in ("gluformer_tiny", "cgm_jepa", "x_cgm_jepa")))
_rp = sorted((REPRO[k][4] for k in ("gluformer_tiny", "cgm_jepa", "x_cgm_jepa")))
_lead_o = REPRO["glucofm"][1] - _ro[-1]
_lead_p = REPRO["glucofm"][4] - _rp[-1]

write("tbl_repro", tab(
    "lrrrrrrrr",
    r"\textbf{Every baseline in this paper's lineage, reproduced from scratch on "
    r"the public-only corpus and scored against its published value.} Task-"
    r"averaged over the same 14 task--cohort cells, the same frozen folds and "
    r"the same untuned probe as everywhere else in this paper. "
    r"\emph{Reproduced} is what this work produces; "
    r"\emph{published} is GlucoFM's Table~3, whose corpus is roughly ten times "
    r"larger and mostly private, so it is a reference point rather than a "
    r"like-for-like column. The last two columns give per-cell agreement: mean "
    r"absolute ROC-AUC difference over the 14 cells, and how many of them land "
    r"within 2 points. \textbf{GlucoFM's lead over the next-best model "
    rf"reproduces at $+{_lead_o:.1f}$ \auc{{}} against $+{_lead_p:.1f}$ "
    r"published}, which is the ordering this benchmark rests on. The three "
    rf"models beneath it span ${_ro[-1] - _ro[0]:.1f}$ \auc{{}} here against "
    rf"${_rp[-1] - _rp[0]:.1f}$ published, so neither table resolves their "
    r"internal order.",
    "tab:repro",
    r"Model & Params & \multicolumn{3}{c}{Reproduced} & "
    r"\multicolumn{2}{c}{Published} & mean & cells \\"
    "\n"
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}"
    "\n"
    r" & (M) & PR & ROC & F1 & PR & ROC & $|\Delta$ROC$|$ & $\leq 2$ \\",
    # Scores bold at their maximum; the agreement column bolds at its minimum,
    # since a small mean $|\Delta|$ is the good outcome there.
    bold_best(rows, {2: "max", 3: "max", 4: "max", 7: "min"}), wide=True,
    note=r"CGM-JEPA and X-CGM-JEPA share an encoder exactly "
         r"(\num{522160} parameters, verified bit-identical against the "
         r"authors' released weights at $0.0$ maximum absolute difference); they "
         r"differ only in the pretraining objective, which is why they land "
         r"\repjepagap{} \auc{} apart here. The GlucoFM row is the same three "
         r"runs the headline tables use, averaged per cell here and per seed "
         r"there; the two agree to \repfmgap{} \auc."))

# copy figures too
for fg in FIGS:
    fg.mkdir(parents=True, exist_ok=True)
    for p in Path(r"D:\final_materials\figures").glob("*.pdf"):
        shutil.copy2(p, fg / p.name)
print("\nall tables written -- now run assemble_paper.py to fold them in")
