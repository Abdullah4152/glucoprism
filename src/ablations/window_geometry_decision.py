"""FD-7 final call: paired across seeds, plus the W6 cohort-split claim."""
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
a = pd.read_csv(ART / "fd7_scores.csv")
b = pd.read_csv(ART / "fd7seed_scores.csv")
df = pd.concat([a, b], ignore_index=True)

# run id -> (arm, seed)
def split(r):
    if r.endswith("-s1"):
        return r[:-3], 1
    if r.endswith("-s2"):
        return r[:-3], 2
    return r, 0


df[["arm", "seed"]] = pd.DataFrame([split(r) for r in df.run], index=df.index)
ARMS = ["W1-ov0", "W3u-ov40", "W6-k6"]

print("Seed-to-seed spread is large but COMMON to every arm (same seed, same")
print("init), so a paired difference is far more sensitive than comparing means.\n")

for lvl in ("window", "subject"):
    d = df[(df.level == lvl) & (df.arm.isin(ARMS))]
    piv = d.groupby(["arm", "seed"])["auc"].mean().unstack()
    print(f"=== {lvl} level, AUC by seed ===")
    print(f"{'arm':<12}{'s0':>7}{'s1':>7}{'s2':>7}{'mean':>8}{'sd':>7}")
    print("-" * 48)
    for arm in ARMS:
        r = piv.loc[arm]
        print(f"{arm:<12}{r[0]:>7.1f}{r[1]:>7.1f}{r[2]:>7.1f}"
              f"{r.mean():>8.2f}{r.std():>7.2f}")

    base = piv.loc["W1-ov0"]
    print(f"\n{'vs W1-ov0':<12}{'d(s0)':>8}{'d(s1)':>8}{'d(s2)':>8}"
          f"{'mean d':>9}{'sd d':>8}{'t':>7}   verdict")
    print("-" * 76)
    for arm in ARMS[1:]:
        dd = (piv.loc[arm] - base).to_numpy(dtype=float)
        m, s = dd.mean(), dd.std(ddof=1)
        t = m / (s / np.sqrt(len(dd))) if s > 1e-9 else np.inf
        # n=3, two-sided 95% critical t = 4.303
        v = "REAL (p<.05, paired)" if abs(t) > 4.303 else "inside noise"
        print(f"{arm:<12}{dd[0]:>+8.1f}{dd[1]:>+8.1f}{dd[2]:>+8.1f}"
              f"{m:>+9.2f}{s:>8.2f}{t:>7.1f}   {v}")
    print()

# The W6 claim: patch size interacts with sampling rate. One seed made it; three
# decide whether it is a finding or a coincidence.
print("\n=== W6 (K=6) per-cohort, all 3 seeds ===")
w = df[df.level == "window"]
piv = w.pivot_table(index=["cohort", "task"], columns=["arm", "seed"], values="auc")
rate = {"shanghait2dm": "15 min", "cgmacros": "5/15 min",
        "stanford": "5 min", "hall": "5 min"}
print(f"{'cohort':<16}{'rate':<11}{'d s0':>8}{'d s1':>8}{'d s2':>8}{'mean':>9}")
print("-" * 60)
rows = {}
for coh in ["shanghait2dm", "hall", "cgmacros", "stanford"]:
    sel = piv.loc[coh]
    dd = [float((sel[("W6-k6", s)] - sel[("W1-ov0", s)]).mean()) for s in (0, 1, 2)]
    rows[coh] = dd
    print(f"{coh:<16}{rate[coh]:<11}{dd[0]:>+8.2f}{dd[1]:>+8.2f}{dd[2]:>+8.2f}"
          f"{np.mean(dd):>+9.2f}")

sh = np.mean(rows["shanghait2dm"])
dense = np.mean(rows["stanford"])
print(f"\n15-min cohort {sh:+.2f}  vs  densest 5-min cohort {dense:+.2f}"
      f"   gap {dense - sh:.2f} AUC")
signs = sum(1 for s in (0, 1, 2)
            if rows["shanghait2dm"][s] < rows["stanford"][s])
print(f"ShanghaiT2DM below Stanford in {signs}/3 seeds  ->  "
      f"{'HOLDS' if signs == 3 else 'does not replicate'}")
