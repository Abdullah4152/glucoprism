"""Paired Wilcoxon + Holm-Bonferroni on CROSS-DATASET TRANSFER.

Same discipline as the within-cohort test: the paired unit is one
(source, target, task) direction, the reference is our GlucoFM reproduction, and
the correction runs over the confirmatory family -- the released models and the
baselines they are claimed to beat.

Reporting an uncorrected p on this axis while insisting on correction on the
other would be having it both ways.
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


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

A = (OUTDIR)

PICK = {"V4-fm-off": "GlucoFM (ours)", "V4-fm-off-s0": "GlucoFM (ours)",
        "C-v2-vib01:zTzS": "GlucoPRISM-C",
        "E-v2-vib-simbias:zTzS": "GlucoPRISM-E",
        "A-v2-base:zTzS": "v2 base [zA dropped]",
        "A-v2-base:full": "v2 base [full]",
        "MantisV2": "MantisV2", "Mantis": "Mantis", "CGMformer": "CGMformer",
        "MOMENT-small": "MOMENT-small", "MOMENT-large": "MOMENT-large",
        "Chronos-2": "Chronos-2", "Chronos-2-small": "Chronos-2-small"}
RELEASED = {"GlucoPRISM-C", "GlucoPRISM-E"}
BASELINES = {"MantisV2", "Mantis", "CGMformer", "MOMENT-small", "MOMENT-large",
             "Chronos-2", "Chronos-2-small"}


def cliffs(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum(xi > yj for xi in x for yj in y)
    lt = sum(xi < yj for xi in x for yj in y)
    return (gt - lt) / (len(x) * len(y))


ap = argparse.ArgumentParser()
ap.add_argument("--family", default="confirmatory", choices=["confirmatory", "all"])
ap.add_argument("--metric", default="auc", choices=["auc", "pr"])
a = ap.parse_args()

fr = [pd.read_csv(A / f) for f in
      ("fd3_v2final.csv", "fd3_bd.csv", "fd3_baselines.csv") if (A / f).exists()]
cd = pd.concat(fr, ignore_index=True)
cd["arm"] = cd.run.str.replace(r"-s\d(:|$)", r"\1", regex=True)
cd["nm"] = cd.arm.map(PICK)
cd = cd[cd.nm.notna()]

g = cd.groupby(["nm", "src", "tgt", "task"])[a.metric].mean()
ref = g.xs("GlucoFM (ours)", level="nm")

models = [m for m in g.index.get_level_values(0).unique() if m != "GlucoFM (ours)"]
if a.family == "confirmatory":
    models = [m for m in models if m in RELEASED | BASELINES]

rows = []
for m in models:
    v = g.xs(m, level="nm")
    idx = v.index.intersection(ref.index)
    x, y = v[idx].to_numpy(float), ref[idx].to_numpy(float)
    d = x - y
    try:
        _, p = wilcoxon(x, y)
    except ValueError:
        p = 1.0
    rows.append(dict(model=m, n=len(idx), mean_delta=d.mean(),
                     wins=int((d > 0).sum()), p_raw=p, cliffs=cliffs(x, y)))

r = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
k = len(r)
r["p_holm"] = [min(1.0, (k - i) * p) for i, p in enumerate(r.p_raw)]
r["p_holm"] = np.maximum.accumulate(r.p_holm)

print(f"\nCross-dataset transfer, paired Wilcoxon over "
      f"{int(r.n.max())} (source, target, task) directions vs GlucoFM (ours)")
print(f"Family: {a.family} (k={k}). Holm-Bonferroni; * = adjusted p < .05\n")
print(f"{'model':<24}{'mean d':>9}{'wins':>7}{'p raw':>9}{'p holm':>9}{'cliff':>8}")
print("-" * 68)
for _, x in r.iterrows():
    star = " *" if x.p_holm < 0.05 else ""
    print(f"{x.model:<24}{x.mean_delta:>+9.2f}{int(x.wins):>4}/{int(x.n):<2}"
          f"{x.p_raw:>9.4f}{x.p_holm:>9.4f}{x.cliffs:>+8.2f}{star}")

r.to_csv(A / f"significance_transfer_{a.metric}.csv", index=False)
print(f"\nwrote significance_transfer_{a.metric}.csv")
