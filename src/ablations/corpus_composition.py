"""Does reducing REPLACE-BG help the RELEASED architecture? A full accounting.

The pre-registered rule (PREREG_corpus_fraction.md) was decided on window level
and returned NOT BETTER. That rule governs whether we rebuild the release. It
does not exhaust what the six runs can tell us, and stopping there would be
closing the question early.

Here we look at every axis, pool the two fractions as two instantiations of one
hypothesis (giving 6 paired observations rather than 3), and check whether any
subject-level signal is spread across cells or concentrated in a few.
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


from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

A = (OUTDIR)
d = pd.read_csv(A / "v2_final_scores.csv")
d = d[d.block.isin(["zTzS", "full"])].copy()
d["arm"] = d.run.str.replace(r"-s\d$", "", regex=True)
d["seed"] = d.run.str.extract(r"-s(\d)$")[0].astype(int)
REF, SH = "C-v2-vib01", [0, 1, 2]

print("=" * 74)
print("1. EVERY AXIS, both readouts")
print("=" * 74)
print(f"{'axis':<26}{'100 %':>9}{'50 %':>9}{'70 %':>9}{'d50':>8}{'d70':>8}")
print("-" * 74)
rows = []
for lvl in ("window", "subject"):
    for blk in ("zTzS", "full"):
        g = d[(d.level == lvl) & (d.block == blk)].groupby(["arm", "seed"]).auc.mean()
        if REF not in g.index.get_level_values(0):
            continue
        r = g.xs(REF, level="arm")[SH]
        out = [r.mean()]
        ds = []
        for arm in ("C-rbg50", "C-rbg70"):
            if arm in g.index.get_level_values(0):
                v = g.xs(arm, level="arm")[SH]
                out.append(v.mean())
                ds.append(v.mean() - r.mean())
            else:
                out.append(np.nan); ds.append(np.nan)
        tag = f"{lvl}, {'zA dropped' if blk=='zTzS' else 'full readout'}"
        print(f"{tag:<26}{out[0]:>9.2f}{out[1]:>9.2f}{out[2]:>9.2f}"
              f"{ds[0]:>+8.2f}{ds[1]:>+8.2f}")

print("\n" + "=" * 74)
print("2. POOLED TEST: 'less REPLACE-BG' as ONE hypothesis, 6 paired obs")
print("=" * 74)
for blk in ("zTzS", "full"):
    for lvl in ("window", "subject"):
        g = d[(d.level == lvl) & (d.block == blk)].groupby(["arm", "seed"]).auc.mean()
        r = g.xs(REF, level="arm")
        diffs = []
        for arm in ("C-rbg50", "C-rbg70"):
            if arm in g.index.get_level_values(0):
                v = g.xs(arm, level="arm")
                diffs += [v[s] - r[s] for s in SH]
        diffs = np.array(diffs)
        t = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))
        try:
            _, p = wilcoxon(diffs)
        except ValueError:
            p = 1.0
        tag = f"{lvl}, {'zA dropped' if blk == 'zTzS' else 'FULL readout'}"
        print(f"  {tag:<26} n={len(diffs)}  mean {diffs.mean():+.2f}"
              f"  sd {diffs.std(ddof=1):.2f}  t={t:+.2f}  p={p:.4f}"
              f"  pos {int((diffs > 0).sum())}/{len(diffs)}"
              f"   {'REAL' if p < .05 else 'inside noise'}")

print("\n" + "=" * 74)
print("3. Is any subject-level gain SPREAD or CONCENTRATED?")
print("=" * 74)
s = d[(d.level == "subject") & (d.block == "zTzS")]
pv = s.groupby(["arm", "cohort", "task"]).auc.mean()
ref = pv.xs(REF, level="arm")
for arm in ("C-rbg50", "C-rbg70"):
    if arm not in pv.index.get_level_values(0):
        continue
    v = pv.xs(arm, level="arm")
    dd = (v - ref).dropna().sort_values()
    print(f"\n  {arm}: {int((dd>0).sum())}/{len(dd)} cells improved, "
          f"mean {dd.mean():+.2f}")
    print(f"    best  : " + ", ".join(f"{c}/{t} {x:+.1f}" for (c, t), x in dd.tail(3).items()))
    print(f"    worst : " + ", ".join(f"{c}/{t} {x:+.1f}" for (c, t), x in dd.head(3).items()))
    top2 = dd.tail(2).sum()
    print(f"    top-2 cells account for {100*top2/max(dd.sum(),1e-9):.0f} % of the total gain")

print("\n" + "=" * 74)
print("4. EFFICIENCY: performance per window")
print("=" * 74)
SIZE = {"C-v2-vib01": (10952, 514), "C-rbg50": (6432, 401), "C-rbg70": (8232, 446)}
g = d[(d.level == "window") & (d.block == "zTzS")].groupby(["arm", "seed"]).auc.mean()
print(f"{'corpus':<12}{'windows':>9}{'subjects':>10}{'window AUC':>12}{'vs 100 %':>10}")
print("-" * 55)
r = g.xs(REF, level="arm")[SH].mean()
for arm, (w, sub) in SIZE.items():
    if arm not in g.index.get_level_values(0):
        continue
    v = g.xs(arm, level="arm")[SH].mean()
    print(f"{arm:<12}{w:>9,}{sub:>10}{v:>12.2f}"
          f"{'' if arm == REF else f'{v-r:>+10.2f}'}")
print(f"\n  50 % uses {100*(1-6432/10952):.0f} % fewer windows and "
      f"{514-401} fewer subjects for indistinguishable performance.")
