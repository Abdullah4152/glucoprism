"""Sensor-block capacity sweep: does the addressability claim depend on width
and KL price, or is it just regularization?

Every arm here ran seeds 0 and 1, so the released model is compared on the SAME
two seeds rather than on its three -- mixing seed counts is what this paper
criticises elsewhere.
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


from pathlib import Path

import pandas as pd

A = ROOT / "experiments/artifacts"
cap = pd.read_csv(A / "rev_capacity_within.csv")
v2 = pd.read_csv(A / "v2_final_scores.csv")
d = pd.concat([cap, v2], ignore_index=True)
d["seed"] = d.run.str.extract(r"-s(\d)$")[0].astype(int)
d["arm"] = d.run.str.replace(r"-s\d$", "", regex=True)
d = d[d.seed <= 1]                       # the sweep only ran seeds 0 and 1

ARMS = [("K-dA8", "8", "0.1"), ("C-v2-vib01", "16 (released)", "0.1"),
        ("K-dA32", "32", "0.1"), ("K-beta003", "16", "0.03"),
        ("K-beta03", "16", "0.3"), ("K-beta10", "16", "1.0")]


def cells(arm, block, lvl):
    s = d[(d.arm == arm) & (d.block == block) & (d.level == lvl)]
    return s.groupby(["cohort", "task"]).auc.mean()


print(f"{'arm':<14}{'zA':<14}{'beta':<7}{'full':>8}{'drop zA':>10}"
      f"{'gain':>8}{'subj drop':>11}")
print("-" * 74)
rows = []
for arm, w, b in ARMS:
    f_ = cells(arm, "full", "window")
    z_ = cells(arm, "zTzS", "window")
    fs = cells(arm, "full", "subject")
    zs = cells(arm, "zTzS", "subject")
    if not len(f_) or not len(z_):
        print(f"{arm:<14}{w:<14}{b:<7}{'MISSING':>8}")
        continue
    i = f_.index.intersection(z_.index)
    j = fs.index.intersection(zs.index)
    gain = (z_[i] - f_[i]).mean()
    sgain = (zs[j] - fs[j]).mean() if len(j) else float("nan")
    print(f"{arm:<14}{w:<14}{b:<7}{f_.mean():>8.2f}{z_.mean():>10.2f}"
          f"{gain:>+8.2f}{sgain:>+11.2f}")
    rows.append(dict(arm=arm, d_sensor=w, beta=b, full=f_.mean(),
                     drop=z_.mean(), gain=gain, subj_gain=sgain))

pd.DataFrame(rows).to_csv(A / "rev_capacity_summary.csv", index=False)
print(f"\nwrote {A / 'rev_capacity_summary.csv'}")
