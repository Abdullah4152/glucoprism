"""FD-8 analysis: paired across seeds, against the pre-registered criteria."""
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

import numpy as np
import pandas as pd

ART = ROOT / "experiments/artifacts"
df = pd.read_csv(ART / "fd8_scores.csv")
df["arm"] = df.run.str.replace(r"-s\d$", "", regex=True)
df["seed"] = df.run.str.extract(r"-s(\d)$").astype(int)

LABEL = {
    "V4-fm-off": "1x  no factorization  (CONTROL)",
    "V1-fm-joint": "1x  factorized, joint",
    "V6-fm-post": "1x  sensor-aug, heads post-hoc",
    "V5-5x-off": "5x  no factorization",
    "V2-5x-joint": "5x  factorized, joint",
    "V7-5x-post": "5x  sensor-aug, heads post-hoc",
}
ORDER = list(LABEL)
CRIT_T = 4.303          # n=3, two-sided 95%


def paired(piv, arm, base):
    d = (piv.loc[arm] - piv.loc[base]).to_numpy(dtype=float)
    m, s = d.mean(), d.std(ddof=1)
    t = m / (s / np.sqrt(len(d))) if s > 1e-9 else np.inf
    return d, m, s, t


for lvl in ("window", "subject"):
    p = df[df.level == lvl].groupby(["arm", "seed"])["auc"].mean().unstack()
    print(f"\n{'='*82}\n{lvl.upper()} level — AUC by seed\n{'='*82}")
    print(f"{'arm':<34}{'s0':>7}{'s1':>7}{'s2':>7}{'mean':>8}{'sd':>7}")
    print("-" * 70)
    for a in ORDER:
        if a not in p.index:
            continue
        r = p.loc[a]
        print(f"{LABEL[a]:<34}{r[0]:>7.1f}{r[1]:>7.1f}{r[2]:>7.1f}"
              f"{r.mean():>8.2f}{r.std():>7.2f}")

    print(f"\n--- paired vs the 1x control (V4) ---")
    print(f"{'arm':<34}{'d s0':>7}{'d s1':>7}{'d s2':>7}{'mean':>8}{'t':>7}   verdict")
    print("-" * 82)
    for a in ORDER[1:]:
        if a not in p.index:
            continue
        d, m, s, t = paired(p, a, "V4-fm-off")
        v = ("BETTER" if t > CRIT_T else "WORSE" if t < -CRIT_T else "inside noise")
        print(f"{LABEL[a]:<34}{d[0]:>+7.1f}{d[1]:>+7.1f}{d[2]:>+7.1f}"
              f"{m:>+8.2f}{t:>7.1f}   {v}")

    # Two questions the grid was built to answer, isolated.
    print(f"\n--- does capacity help?  5x vs 1x at matched factorization ---")
    for a5, a1, name in (("V5-5x-off", "V4-fm-off", "no factorization"),
                         ("V2-5x-joint", "V1-fm-joint", "joint factorization"),
                         ("V7-5x-post", "V6-fm-post", "sensor-aug")):
        if a5 in p.index and a1 in p.index:
            d, m, s, t = paired(p, a5, a1)
            v = ("BETTER" if t > CRIT_T else "WORSE" if t < -CRIT_T else "inside noise")
            print(f"  {name:<24}{m:>+7.2f} AUC  t={t:>6.1f}   {v}")

    print(f"\n--- does factorization help?  vs no-factorization at matched size ---")
    for af, ao, name in (("V1-fm-joint", "V4-fm-off", "1x joint"),
                         ("V2-5x-joint", "V5-5x-off", "5x joint"),
                         ("V6-fm-post", "V4-fm-off", "1x sensor-aug substrate"),
                         ("V7-5x-post", "V5-5x-off", "5x sensor-aug substrate")):
        if af in p.index and ao in p.index:
            d, m, s, t = paired(p, af, ao)
            v = ("BETTER" if t > CRIT_T else "WORSE" if t < -CRIT_T else "inside noise")
            print(f"  {name:<24}{m:>+7.2f} AUC  t={t:>6.1f}   {v}")

print("\n\nFD-7 reference on the same corpus (_ov40, 3 seeds): "
      "65.85 window / 68.95 subject")
