"""FD-7 analysis: which windowing wins, and does the W6 prediction hold?"""
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
df = pd.read_csv(ART / "fd7_scores.csv")

# GlucoFM's own seed spread on this benchmark, 3 seeds: +-0.57 PR / +-1.00 AUC
# (window) and +-1.61 / +-2.58 (subject). Nothing smaller than that is readable
# from a single seed.
SIGMA = {"window": 1.00, "subject": 2.58}
REF = "W1-ov0"

for lvl in ("window", "subject"):
    d = df[df.level == lvl]
    piv = d.groupby("run")[["pr", "auc", "f1"]].mean()
    base = piv.loc[REF, "auc"]
    print(f"\n=== {lvl} level (1 seed each; seed sigma ~{SIGMA[lvl]:.2f} AUC) ===")
    print(f"{'run':<14}{'PR':>7}{'AUC':>7}{'F1':>7}{'dAUC':>8}   verdict")
    print("-" * 62)
    for r in ["W1-ov0", "W2-ov20m", "W3-ov40m", "W3u-ov40",
              "W4-k18", "W5-k18-ov40", "W6-k6", "W7-k24"]:
        if r not in piv.index:
            continue
        row = piv.loc[r]
        d_auc = row["auc"] - base
        v = ("reference" if r == REF else
             "inside noise" if abs(d_auc) < SIGMA[lvl] else
             ("BETTER" if d_auc > 0 else "WORSE"))
        print(f"{r:<14}{row['pr']:>7.1f}{row['auc']:>7.1f}{row['f1']:>7.1f}"
              f"{d_auc:>+8.1f}   {v}")

# Pre-registered prediction (discussion.md 6.3): halving the patch to 30 min
# leaves a 15-minute Libre with only 2 readings per patch, so the per-patch
# standard deviation becomes near-meaningless. W6 should therefore hurt
# ShanghaiT2DM -- the 15-min cohort -- MORE than the 5-min cohorts.
print("\n\n=== pre-registered W6 prediction: K=6 hurts the 15-min cohort most ===")
w = df[df.level == "window"].pivot_table(index=["cohort", "task"],
                                         columns="run", values="auc")
delta = (w["W6-k6"] - w[REF]).rename("dAUC")
by_coh = delta.groupby("cohort").mean().sort_values()
print(f"\n{'cohort':<16}{'sampling':<12}{'mean dAUC vs W1':>17}")
print("-" * 45)
rate = {"shanghait2dm": "15 min", "cgmacros": "5/15 min",
        "stanford": "5 min", "hall": "5 min"}
for c, v in by_coh.items():
    print(f"{c:<16}{rate.get(c, '?'):<12}{v:>17.2f}")
sh = by_coh.get("shanghait2dm", np.nan)
others = by_coh.drop("shanghait2dm", errors="ignore").mean()
print(f"\nShanghaiT2DM {sh:+.2f} vs other cohorts {others:+.2f}  ->  "
      f"prediction {'HELD' if sh < others - 0.5 else 'NOT SUPPORTED'}")

print("\n\n=== W7 (2-hour patches): where does the damage land? ===")
d7 = (w["W7-k24"] - w[REF]).sort_values()
print(f"{'cohort':<16}{'task':<20}{'dAUC':>8}")
print("-" * 44)
for (c, t), v in d7.items():
    print(f"{c:<16}{t:<20}{v:>+8.1f}")

print("\n\n=== W4 (30-min patch lookback): where does the damage land? ===")
d4 = (w["W4-k18"] - w[REF]).sort_values()
for (c, t), v in d4.head(5).items():
    print(f"  {c:<16}{t:<20}{v:>+8.1f}")
print(f"  ... mean {d4.mean():+.2f} over {len(d4)} cells, "
      f"{int((d4 < 0).sum())} of {len(d4)} cells worse")
