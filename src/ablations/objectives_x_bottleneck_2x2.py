"""The confound arm completes a 2x2: protocol objectives x VIB bottleneck.

Until these runs landed, no arm ran the v2 auxiliary stack (global_norm,
stat_pool, L_CMP, variance floor) WITHOUT the block structure and objectives, so
the paper could only bound the factorization's contribution from above. The
--no-protocol arms close that gap, and they turn the additive "ladder" in the
paper into an interaction, which is a different claim.

    objectives OFF, VIB OFF   X-noproto          (components only)
    objectives OFF, VIB ON    Y-noproto-vib
    objectives ON,  VIB OFF   A-v2-base
    objectives ON,  VIB ON    C-v2-vib01         (released)

All arms share corpus, backbone, probe and frozen folds, and all are restricted
to seeds {0,1,2} so the comparison is seed-matched.
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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

A = (OUTDIR)

v2 = pd.read_csv(A / "v2_final_scores.csv")
cf = pd.read_csv(A / "confound_scores.csv")
d = pd.concat([v2, cf], ignore_index=True)
d["seed"] = d.run.str.extract(r"-s(\d)$")[0].astype(int)
d["arm"] = d.run.str.replace(r"-s\d$", "", regex=True)
d = d[d.seed <= 2]                                  # seed-matched throughout

CELLS = ["cohort", "task"]
GRID = {("off", "off"): "X-noproto",
        ("off", "on"): "Y-noproto-vib",
        ("on", "off"): "A-v2-base",
        ("on", "on"): "C-v2-vib01"}


def cells(arm: str, block: str, level: str = "window") -> pd.Series:
    s = d[(d.arm == arm) & (d.block == block) & (d.level == level)]
    return s.groupby(CELLS).auc.mean()


def show(title: str, block: str, level: str = "window") -> dict:
    print(f"\n{'=' * 66}\n{title}  [block={block}, {level} level, seeds 0-2]\n"
          f"{'=' * 66}")
    m = {}
    print(f"{'objectives':<12}{'VIB':<8}{'arm':<18}{'AUC':>8}")
    for (obj, vib), arm in GRID.items():
        c = cells(arm, block, level)
        if not len(c):
            print(f"{obj:<12}{vib:<8}{arm:<18}{'MISSING':>8}")
            continue
        m[(obj, vib)] = c
        print(f"{obj:<12}{vib:<8}{arm:<18}{c.mean():>8.2f}")
    if len(m) < 4:
        return m

    idx = m[("off", "off")].index
    for k in m:
        idx = idx.intersection(m[k].index)
    a = {k: m[k][idx].to_numpy(float) for k in m}
    n = len(idx)

    simple_obj_novib = a[("on", "off")] - a[("off", "off")]
    simple_obj_vib = a[("on", "on")] - a[("off", "on")]
    simple_vib_noobj = a[("off", "on")] - a[("off", "off")]
    simple_vib_obj = a[("on", "on")] - a[("on", "off")]
    inter = simple_vib_obj - simple_vib_noobj

    print(f"\n  simple effects over n={n} cells (paired):")
    for nm, v in [("objectives | no VIB ", simple_obj_novib),
                  ("objectives | VIB    ", simple_obj_vib),
                  ("VIB | no objectives ", simple_vib_noobj),
                  ("VIB | objectives    ", simple_vib_obj)]:
        t, p = ttest_rel(v, np.zeros_like(v))
        try:
            pw = wilcoxon(v)[1]
        except ValueError:
            pw = 1.0
        print(f"    {nm} {v.mean():+6.2f}   t={t:+6.2f}  p={p:.4f}  "
              f"wilcoxon={pw:.4f}  won {int((v > 0).sum())}/{n}")

    t, p = ttest_rel(inter, np.zeros_like(inter))
    print(f"\n    INTERACTION        {inter.mean():+6.2f}   t={t:+6.2f}  "
          f"p={p:.4f}  positive in {int((inter > 0).sum())}/{n}")
    main_obj = 0.5 * (simple_obj_novib + simple_obj_vib)
    main_vib = 0.5 * (simple_vib_noobj + simple_vib_obj)
    print(f"    main effect objectives {main_obj.mean():+6.2f}")
    print(f"    main effect VIB        {main_vib.mean():+6.2f}")
    return {"n": n, "inter": float(inter.mean()), "p_inter": float(p),
            "obj_novib": float(simple_obj_novib.mean()),
            "obj_vib": float(simple_obj_vib.mean()),
            "vib_noobj": float(simple_vib_noobj.mean()),
            "vib_obj": float(simple_vib_obj.mean()),
            "main_obj": float(main_obj.mean()), "main_vib": float(main_vib.mean())}


out = {}
for block in ("full", "zTzS"):
    r = show(f"protocol objectives x VIB bottleneck", block)
    if "n" in r:
        out[block] = r

# Does dropping zA still help once the objectives are off? The mechanism says
# zA only contains a shortcut worth deleting if something trained it to.
print(f"\n{'=' * 66}\nzA-drop benefit by arm (zTzS minus full, window)\n{'=' * 66}")
for (obj, vib), arm in GRID.items():
    f, z = cells(arm, "full"), cells(arm, "zTzS")
    i = f.index.intersection(z.index)
    if not len(i):
        continue
    dd = (z[i] - f[i]).to_numpy(float)
    print(f"  objectives={obj:<4} VIB={vib:<4} {arm:<18}{dd.mean():+6.2f}  "
          f"positive in {int((dd > 0).sum())}/{len(dd)}")

(A / "confound_analysis.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {A / 'confound_analysis.json'}")
