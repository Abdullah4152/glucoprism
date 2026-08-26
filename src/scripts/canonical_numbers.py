"""ONE source of truth for every number in the paper.

Two verification agents found 25 places where our documents stated different
values for the same quantity -- including whether our corpus is larger or
smaller than GlucoFM's (asserted both ways inside one file). The cause is that
numbers were typed into prose in half a dozen places.

This emits `canonical.json` + `canonical.tex` (LaTeX \newcommand macros). The
paper cites macros, never literals, so a contradiction becomes impossible rather
than merely discouraged.

Seed matching: GlucoPRISM-C has 6 seeds and its comparators have 3, and C's
seeds 3-5 were better than 0-2 in all four conditions. Every headline comparison
here is therefore computed on the SHARED seed set, and the 6-seed figure is
reported separately and labelled.
"""
from __future__ import annotations

import os as _os, sys as _sys
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))
for _p in (ROOT / "src" / "core", ROOT / "baselines"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))


import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

A = ROOT / "experiments/artifacts"
OUT_J = A / "canonical.json"
# Written to BOTH trees. Keeping only the final_materials copy meant the Overleaf
# project silently kept a stale canonical.tex and every new macro compiled as an
# undefined control sequence.
OUT_TS = [OUTDIR / "tex/canonical.tex",
          Path(r"D:\overleaf\glucoprismm\glucoprism_v2\canonical.tex")]
OUT_T = OUT_TS[0]

OURS = "GlucoPRISM-v2 + zA bottleneck (weak) [zA dropped]"
OURS_FULL = "GlucoPRISM-v2 + zA bottleneck (weak) [full]"
FM = "GlucoFM (ours)"
C = {}


def cells(df, run, level, metric="auc"):
    s = df[(df.run == run) & (df.level == level)]
    return s.groupby(["cohort", "task"])[metric].mean()


def paired(df, a, b, level, metric="auc"):
    x, y = cells(df, a, level, metric), cells(df, b, level, metric)
    i = x.index.intersection(y.index)
    d = (x[i] - y[i]).to_numpy(float)
    try:
        p = wilcoxon(x[i], y[i])[1]
    except ValueError:
        p = 1.0
    return dict(n=len(d), mean=float(d.mean()), wins=int((d > 0).sum()), p_raw=float(p))


# ------------------------------------------------------------- benchmark
d = pd.read_csv(A / "final_table_long.csv")
for tag, run in (("ours", OURS), ("oursfull", OURS_FULL), ("fm", FM)):
    for lvl in ("window", "subject"):
        for m in ("pr", "auc", "f1"):
            v = cells(d, run, lvl, m)
            if len(v):
                C[f"{tag}{lvl}{m}"] = round(float(v.mean()), 2)

for lvl in ("window", "subject"):
    r = paired(d, OURS, FM, lvl)
    C[f"delta{lvl}"] = round(r["mean"], 2)
    C[f"wins{lvl}"] = r["wins"]
    C[f"ncells{lvl}"] = r["n"]
    C[f"praw{lvl}"] = round(r["p_raw"], 4)

# ------------------------------------------------------------- the ladder
LAD = [("fmbase", FM), ("facfull", "GlucoPRISM-v2 [full]"),
       ("facdrop", "GlucoPRISM-v2 [zA dropped]"),
       ("vibfull", OURS_FULL), ("vibdrop", OURS)]
prev = None
for tag, run in LAD:
    v = cells(d, run, "window")
    if not len(v):
        continue
    C[f"lad_{tag}"] = round(float(v.mean()), 2)
    prev = float(v.mean())
C["ladfactor"] = round(C["lad_facfull"] - C["lad_fmbase"], 2)
C["ladvib"] = round(C["lad_vibfull"] - C["lad_facfull"], 2)
C["laddrop"] = round(C["lad_vibdrop"] - C["lad_vibfull"], 2)

# ------------------------------------------------- seed asymmetry (critical)
v2 = pd.read_csv(A / "v2_final_scores.csv")
v2["arm"] = v2.run.str.replace(r"-s\d$", "", regex=True)
v2["seed"] = v2.run.str.extract(r"-s(\d)$")[0].astype(int)
g = v2[(v2.arm == "C-v2-vib01") & (v2.level == "window") &
       (v2.block == "zTzS")].groupby("seed").auc.mean()
C["cseeds6"] = round(float(g.mean()), 2)
C["cseeds3"] = round(float(g.loc[[0, 1, 2]].mean()), 2)
C["cseeds6sd"] = round(float(g.std()), 2)
C["cseeds3sd"] = round(float(g.loc[[0, 1, 2]].std()), 2)
C["seedgap"] = round(float(g.loc[[3, 4, 5]].mean() - g.loc[[0, 1, 2]].mean()), 2)

# zA-drop, per level, on the released model. Reported on the SHARED THREE SEEDS,
# like every other headline in the paper -- quoting this one comparison at six
# seeds while everything else is at three is the seed-count mixing the paper
# criticises elsewhere. The six-seed figures are kept under a `six` suffix so
# the difference stays inspectable, but the paper cites the three-seed ones.
for lvl, tag in (("window", "w"), ("subject", "s")):
    p = v2[(v2.arm == "C-v2-vib01") & (v2.level == lvl)] \
        .groupby(["block", "seed"]).auc.mean().unstack(0)
    if {"full", "zTzS"} <= set(p.columns):
        dd = (p["zTzS"] - p["full"]).dropna()
        C[f"drop{tag}six"] = round(float(dd.mean()), 2)
        d3 = dd.loc[[i for i in dd.index if i <= 2]]
        C[f"drop{tag}"] = round(float(d3.mean()), 2)
        C[f"drop{tag}pos"] = f"{int((d3 > 0).sum())}/{len(d3)}"

# -------------------------------------------- the seed-MATCHED headline
# `deltawindow` above averages our model over 6 seeds and its comparators over 3,
# and our seeds 3-5 were the better ones. The headline the paper leads with is
# therefore this one: our model restricted to seeds {0,1,2}, paired per cell
# against GlucoFM's own 3-seed average, on the same frozen folds.
for lvl, tag in (("window", "window"), ("subject", "subject")):
    ours3 = v2[(v2.arm == "C-v2-vib01") & (v2.block == "zTzS") &
               (v2.level == lvl) & (v2.seed <= 2)] \
        .groupby(["cohort", "task"]).auc.mean()
    fm = cells(d, FM, lvl)
    i = ours3.index.intersection(fm.index)
    dd = (ours3[i] - fm[i]).dropna()
    C[f"m{tag}auc"] = round(float(ours3[i].mean()), 2)
    C[f"mdelta{tag}"] = round(float(dd.mean()), 2)
    C[f"mwins{tag}"] = int((dd > 0).sum())
    C[f"mncells{tag}"] = len(dd)
    C[f"mpraw{tag}"] = round(float(wilcoxon(ours3[i], fm[i])[1]), 4)

# Holm over the confirmatory family (k=9: the released models and the baselines
# they are claimed to beat), with our model's SEED-MATCHED row substituted in.
# The whole corrected table is written back out, so the paper's significance
# table and its headline can not disagree.
ours3w = v2[(v2.arm == "C-v2-vib01") & (v2.block == "zTzS") &
            (v2.level == "window") & (v2.seed <= 2)] \
    .groupby(["cohort", "task"]).auc.mean()
fmw = cells(d, FM, "window")
iw = ours3w.index.intersection(fmw.index)
dw = (ours3w[iw] - fmw[iw]).to_numpy(float)
# Cliff's delta the same way the existing table computes it: the two-sample form
# over all n*m cell pairs, NOT the sign of the paired differences.
x, y = ours3w[iw].to_numpy(float), fmw[iw].to_numpy(float)
cliff = float(((x[:, None] > y[None, :]).sum() -
               (x[:, None] < y[None, :]).sum()) / (len(x) * len(y)))

sig = pd.read_csv(A / "significance_window_auc.csv")
row = sig.model.str.contains("weak.*dropped", regex=True)
sig.loc[row, ["mean_delta", "median_delta", "wins", "n", "p_raw", "cliffs"]] = [
    float(dw.mean()), float(np.median(dw)), int((dw > 0).sum()), len(dw),
    C["mprawwindow"], cliff]

sig = sig.sort_values("p_raw").reset_index(drop=True)
k = len(sig)
sig["p_holm"] = np.minimum(
    np.maximum.accumulate([(k - j) * p for j, p in enumerate(sig.p_raw)]), 1.0)
sig["sig"] = np.where(sig.p_holm < 0.05, "*", "")
sig.to_csv(A / "significance_window_auc_matched.csv", index=False)

C["famk"] = k
C["mpholmwindow"] = round(float(sig.p_holm.iloc[0]), 4)
C["famsig"] = int((sig.p_holm < 0.05).sum())
C["mcliffwindow"] = round(cliff, 2)

# ------------------------------------------------------------- transfer
fr = []
for f in ("fd3_v2final.csv", "fd3_bd.csv", "fd3_baselines.csv", "fd3_rbgfrac.csv"):
    if (A / f).exists():
        fr.append(pd.read_csv(A / f))
t = pd.concat(fr, ignore_index=True)
# The transfer significance table was computed against a reference that
# triple-counted GlucoFM seed 0, because overlapping fd3 CSVs were concatenated
# without dedup. Deduplicate on the full key before anything is averaged.
t = t.drop_duplicates(subset=["run", "src", "tgt", "task"])
t["arm"] = t.run.str.replace(r"-s\d(:|$)", r"\1", regex=True)
TR = {"trfm": "V4-fm-off", "trc": "C-v2-vib01:zTzS", "tre": "E-v2-vib-simbias:zTzS",
      "trcfull": "C-v2-vib01:full"}
for tag, arm in TR.items():
    s = t[t.arm == arm]
    if len(s):
        C[tag] = round(float(s.auc.mean()), 2)
C["trdropc"] = round(C["trc"] - C["trcfull"], 2)

base = t[t.arm == "V4-fm-off"].groupby(["src", "tgt", "task"]).auc.mean()
for tag, arm in (("trdc", "C-v2-vib01:zTzS"), ("trde", "E-v2-vib-simbias:zTzS")):
    m = t[t.arm == arm].groupby(["src", "tgt", "task"]).auc.mean()
    i = m.index.intersection(base.index)
    dd = (m[i] - base[i]).dropna()
    C[tag] = round(float(dd.mean()), 2)
    C[f"{tag}n"] = len(dd)
    C[f"{tag}p"] = round(float(wilcoxon(m[i], base[i])[1]), 4)

# ------------------------------------- the 2x2: objectives x VIB bottleneck
# The --no-protocol arms close the confound the paper previously had to flag:
# no arm ran the v2 auxiliary stack WITHOUT the block structure. With them the
# decomposition stops being a ladder and becomes an interaction.
CF = A / "confound_analysis.json"
if CF.exists():
    cf = json.loads(CF.read_text())["full"]
    # Signed and fixed to 2dp so prose reads "+0.30" exactly as the table does;
    # round() would render 0.30 as "0.3" and drop the sign.
    C["ixn"] = f"{cf['inter']:+.2f}"
    C["ixnp"] = round(cf["p_inter"], 4)
    C["ixnn"] = cf["n"]
    C["objalone"] = f"{cf['obj_novib']:+.2f}"
    C["objgivenvib"] = f"{cf['obj_vib']:+.2f}"
    C["vibalone"] = f"{cf['vib_noobj']:+.2f}"
    C["vibgivenobj"] = f"{cf['vib_obj']:+.2f}"

    cfs = pd.read_csv(A / "confound_scores.csv")
    cfs["arm"] = cfs.run.str.replace(r"-s\d$", "", regex=True)
    for tag, arm in (("cellsonly", "X-noproto"), ("cellsvib", "Y-noproto-vib")):
        s = cfs[(cfs.arm == arm) & (cfs.level == "window") &
                (cfs.block == "full")]
        if len(s):
            C[tag] = round(float(s.groupby(["cohort", "task"]).auc.mean().mean()), 2)

# ------------------------------------------------- reviewer-requested analyses
if (A / "rev_inlp_calibration.csv").exists():
    e = pd.read_csv(A / "rev_inlp_calibration.csv")
    ge = e.groupby("tag")[["auc", "ece", "brier"]].mean()
    C["inlpauc"] = round(float(ge.loc["INLP on full (128d)", "auc"]), 2)
    C["inlpgain"] = f"{ge.loc['INLP on full (128d)', 'auc'] - ge.loc['full (128d)', 'auc']:+.2f}"
    C["eceful"] = round(float(ge.loc["full (128d)", "ece"]), 3)
    C["ecedrop"] = round(float(ge.loc["drop zA (112d, ours)", "ece"]), 3)
    C["brierfull"] = round(float(ge.loc["full (128d)", "brier"]), 3)
    C["brierdrop"] = round(float(ge.loc["drop zA (112d, ours)", "brier"]), 3)

if (A / "rev_soft_deletion.csv").exists():
    s = pd.read_csv(A / "rev_soft_deletion.csv").groupby("tag").auc.mean()
    for m_ in (0, 2, 4, 16):
        k = f"keep {m_}/16 zA dims"
        if k in s.index:
            C[f"keep{m_}"] = round(float(s[k]), 2)

if (A / "rev_device_predictability.csv").exists():
    dp = pd.read_csv(A / "rev_device_predictability.csv").set_index("features")
    C["devmask"] = round(float(dp.loc["observation mask (288d)", "device_auc"]), 1)
    C["devcount"] = round(float(dp.loc["mask summary (count only)", "device_auc"]), 1)
    C["devlevel"] = round(float(dp.loc["glucose level only (3d)", "device_auc"]), 1)

if (A / "rev_block_dependence.csv").exists():
    bd_ = pd.read_csv(A / "rev_block_dependence.csv").groupby("pair").mean(
        numeric_only=True)
    C["corrts"] = round(float(bd_.loc["zT-zS", "mean_abs_corr"]), 2)
    C["hsicts"] = round(float(bd_.loc["zT-zS", "hsic"]), 2)
    C["hsicta"] = round(float(bd_.loc["zT-zA", "hsic"]), 2)
    C["hsicsa"] = round(float(bd_.loc["zS-zA", "hsic"]), 2)

# The zA-drop at subject level changes SIGN between three and six seeds. Both
# are reported; the paper does not get to pick.
p3 = v2[(v2.arm == "C-v2-vib01") & (v2.level == "subject") & (v2.seed <= 2)] \
    .groupby(["block", "seed"]).auc.mean().unstack(0)
if {"full", "zTzS"} <= set(p3.columns):
    C["dropsthree"] = f"{(p3['zTzS'] - p3['full']).dropna().mean():+.2f}"

# ------------------------------------------------------------- corpus
C["nsubj"] = 514
C["nwin"] = 10952
C["nhours"] = 157709
C["fmsubj"] = 477
C["fmhours"] = 109066
C["rbgshare"] = 82.5

# ------------------------------------------------------------- misc
C["infparams"] = 506550
C["trainparams"] = 806168
C["fmparams"] = 720278
C["fmparamspub"] = 720241
C["glucometrics"] = 63.8

OUT_J.write_text(json.dumps(C, indent=2))

DIGITS = dict(zip("0123456789", "zero one two three four five six seven eight "
                                "nine".split()))


def mac(key: str) -> str:
    """A LaTeX control word may contain letters only.

    `\\lad_fmbase` and `\\cseeds3` are both parse errors -- the underscore ends
    the control word and starts math-mode subscript, and the digit just ends it.
    So `lad_fmbase` -> `ladfmbase`, `cseeds3sd` -> `cseedsthreesd`.
    """
    return "".join(DIGITS.get(ch, ch) for ch in key if ch != "_")


lines = ["% Generated by canonical_numbers.py -- DO NOT EDIT.",
         "% Every number in the paper is a macro from this file.", ""]
seen: dict[str, str] = {}
for k, v in sorted(C.items()):
    m = mac(k)
    if m in seen:
        raise SystemExit(f"macro collision: {k!r} and {seen[m]!r} both -> \\{m}")
    seen[m] = k
    val = f"{v}" if not isinstance(v, float) else f"{v:g}"
    lines.append(rf"\newcommand{{\{m}}}{{{val}}}")
for p in OUT_TS:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")

print(f"{len(C)} canonical values -> {OUT_J.name}, {OUT_T.name}\n")
for k in ["ladfactor", "ladvib", "laddrop", "deltawindow", "prawwindow",
          "cseeds3", "cseeds6", "seedgap", "dropw", "drops",
          "trfm", "trc", "tre", "trdropc", "trdc", "trdcp"]:
    if k in C:
        print(f"  {k:<14}{C[k]}")
