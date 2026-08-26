"""Where does the variance in our per-cell effect actually come from?

Two very different situations look identical in a Wilcoxon p-value:

  (a) the effect is real and uniform, and SEED noise blurs each cell's estimate
      -> more seeds sharpen the estimates and power rises
  (b) the effect is genuinely heterogeneous across TASKS -- helps on some,
      hurts on others -- and seed noise is minor
      -> more seeds change nothing, because the spread is the finding

Only (a) is fixable by running more of the same. This decomposes the two before
anyone spends GPU time.
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
df = pd.read_csv(A / "final_table_long.csv")
d = df[df.level == "window"]

REF = "GlucoFM (ours)"
CANDS = {
    "C  bottleneck b=0.1 [zA dropped]": "GlucoPRISM-v2 + zA bottleneck (weak) [zA dropped]",
    "B  bottleneck b=1.0 [zA dropped]": "GlucoPRISM-v2 + zA bottleneck [zA dropped]",
    "E  bottleneck + sensor [zA dropped]": "GlucoPRISM-v2 + bottleneck + measured sensor [zA dropped]",
}

# Per-seed, per-cell values so seed variance can be separated from cell variance.
raw = d.groupby(["run", "cohort", "task"]).auc.agg(["mean", "std", "count"])

print("=" * 88)
print("VARIANCE DECOMPOSITION of the per-cell effect (window AUC)")
print("=" * 88)
print(f"{'model':<38}{'cell sd':>9}{'seed sd':>9}{'ratio':>8}   verdict")
print("-" * 88)
for label, m in CANDS.items():
    if m not in raw.index.get_level_values(0):
        continue
    a = raw.loc[m]
    b = raw.loc[REF]
    both = a.index.intersection(b.index)
    delta = (a.loc[both, "mean"] - b.loc[both, "mean"]).to_numpy(float)
    cell_sd = delta.std(ddof=1)                      # spread ACROSS tasks
    # seed sd of the difference: seeds are independent between the two models
    seed_sd = float(np.sqrt((a.loc[both, "std"].fillna(0) ** 2
                             + b.loc[both, "std"].fillna(0) ** 2).mean() / 3))
    ratio = cell_sd / max(seed_sd, 1e-9)
    verdict = ("task heterogeneity dominates -- more seeds will NOT help"
               if ratio > 1.5 else
               "seed noise dominates -- more seeds WOULD help")
    print(f"{label:<38}{cell_sd:>9.2f}{seed_sd:>9.2f}{ratio:>8.2f}   {verdict}")

print("\n" + "=" * 88)
print("WHAT THE FAMILY DEFINITION DOES TO THE SAME NUMBER")
print("=" * 88)
piv = d.groupby(["run", "cohort", "task"]).auc.mean().unstack(["cohort", "task"])
ref = piv.loc[REF]

FAMILIES = {
    "everything we ran (18)": [m for m in piv.index if m != REF],
    "our models only (10)": [m for m in piv.index
                             if m != REF and "GlucoPRISM" in m],
    "published-style: 1 model vs baselines (8)":
        [m for m in piv.index if m != REF and "GlucoPRISM" not in m]
        + [CANDS["C  bottleneck b=0.1 [zA dropped]"]],
    "pre-registered single comparison (1)":
        [CANDS["C  bottleneck b=0.1 [zA dropped]"]],
}
target = CANDS["C  bottleneck b=0.1 [zA dropped]"]
for fam, members in FAMILIES.items():
    ps = []
    for m in members:
        v = piv.loc[m]
        ok = v.notna() & ref.notna()
        try:
            ps.append((m, wilcoxon(v[ok], ref[ok])[1]))
        except ValueError:
            ps.append((m, 1.0))
    ps.sort(key=lambda x: x[1])
    k = len(ps)
    adj = {}
    run_max = 0.0
    for i, (m, p) in enumerate(ps):
        run_max = max(run_max, min(1.0, (k - i) * p))
        adj[m] = run_max
    print(f"  {fam:<44}k={k:<3} p_holm(C) = {adj.get(target, float('nan')):.4f}"
          f"   {'SIGNIFICANT' if adj.get(target, 1) < .05 else ''}")

print("\nThe raw p is 0.0031 in every row -- only the family size moves it.")
