"""Collect the post-hoc factorization sweep from the run logs."""
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


import re
from pathlib import Path

import pandas as pd

ART = (OUTDIR)
ENC = {"v4": "1x control (no sensor aug)",
       "v6": "1x sensor-aug substrate",
       "v7": "5x sensor-aug substrate"}

rows = []
for f in sorted(ART.glob("posthoc_*.log")):
    m = re.match(r"posthoc_(v\d)-n(\d+)\.log", f.name)
    if not m:
        continue
    enc, n = m.group(1), int(m.group(2))
    txt = f.read_text(errors="replace")
    tail = txt.split("task-averaged over 14 cells")[-1]
    for line in tail.splitlines():
        p = line.split()
        if len(p) == 3 and p[0] in ("zT", "zS", "zTzS", "encoder", "full", "zA"):
            rows.append(dict(encoder=enc, n_fit=n, block=p[0],
                             pr=float(p[1]), auc=float(p[2])))

df = pd.DataFrame(rows)
df.to_csv(ART / "posthoc_sweep.csv", index=False)

ORDER = ["encoder", "zT", "zS", "zTzS", "full", "zA"]
for enc in ("v4", "v6", "v7"):
    d = df[df.encoder == enc]
    if d.empty:
        continue
    piv = d.pivot(index="block", columns="n_fit", values="auc").reindex(ORDER)
    print(f"\n=== {ENC[enc]} — ROC-AUC by head-fitting subject count ===")
    print(f"{'block':<10}" + "".join(f"{c:>9}" for c in piv.columns) +
          f"{'vs encoder @45':>17}")
    print("-" * 62)
    base45 = piv.loc["encoder", 45]
    for b in ORDER:
        if b not in piv.index:
            continue
        r = piv.loc[b]
        mark = "" if b == "encoder" else f"{r[45] - base45:>+17.1f}"
        print(f"{b:<10}" + "".join(f"{v:>9.1f}" for v in r) + mark)

print("\n\n=== does more paired data help the factorization? ===")
print(f"{'encoder':<28}{'zT @15':>9}{'@20':>7}{'@30':>7}{'@45':>7}{'slope':>9}")
print("-" * 68)
for enc in ("v4", "v6", "v7"):
    d = df[(df.encoder == enc) & (df.block == "zT")].sort_values("n_fit")
    if d.empty:
        continue
    v = d.auc.to_numpy()
    slope = (v[-1] - v[0]) / (d.n_fit.iloc[-1] - d.n_fit.iloc[0]) * 30
    print(f"{ENC[enc]:<28}" + "".join(f"{x:>9.1f}" if i == 0 else f"{x:>7.1f}"
                                      for i, x in enumerate(v)) +
          f"{slope:>+9.2f}")
print("\nslope = AUC change per +30 head-fitting subjects")

print("\n\n=== best block vs the unfactorized encoder, every configuration ===")
print(f"{'encoder':<28}{'n':>4}{'best block':>12}{'best AUC':>10}"
      f"{'encoder AUC':>13}{'delta':>8}")
print("-" * 76)
wins = 0
for enc in ("v4", "v6", "v7"):
    for n in (15, 20, 30, 45):
        d = df[(df.encoder == enc) & (df.n_fit == n)]
        if d.empty:
            continue
        base = float(d[d.block == "encoder"].auc.iloc[0])
        blocks = d[d.block != "encoder"]
        best = blocks.loc[blocks.auc.idxmax()]
        delta = best.auc - base
        wins += delta > 0
        print(f"{ENC[enc]:<28}{n:>4}{best.block:>12}{best.auc:>10.1f}"
              f"{base:>13.1f}{delta:>+8.1f}")
print(f"\na block beat the unfactorized encoder in {wins}/12 configurations")

# "Best block" is chosen AFTER seeing the results, from 5 candidates. Under the
# null that blocks are just the encoder plus noise, the max of 5 noisy estimates
# beats a fixed baseline about half the time by construction -- which is exactly
# the error that produced the retracted F3. The honest test uses the block the
# proposal names in advance: zT, the Trait block, which every downstream claim
# rests on.
import numpy as np  # noqa: E402

print("\n\n=== pre-specified test: zT vs the unfactorized encoder ===")
print("(no post-hoc block selection; zT is what the proposal claims carries trait)")
d = []
for enc in ("v4", "v6", "v7"):
    for n in (15, 20, 30, 45):
        s = df[(df.encoder == enc) & (df.n_fit == n)]
        if s.empty:
            continue
        d.append(float(s[s.block == "zT"].auc.iloc[0]) -
                 float(s[s.block == "encoder"].auc.iloc[0]))
d = np.array(d)
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print(f"\n  n = {len(d)} configurations")
print(f"  mean delta = {d.mean():+.2f} AUC   sd = {d.std(ddof=1):.2f}   t = {t:.2f}")
print(f"  positive in {int((d > 0).sum())}/{len(d)}")
print(f"  verdict: {'REAL' if abs(t) > 2.2 else 'NULL -- zT is not better than not factorising'}")

exp_max = df[df.block != "encoder"].groupby(["encoder", "n_fit"]).auc.max()
base = df[df.block == "encoder"].set_index(["encoder", "n_fit"]).auc
print(f"\n  for contrast, mean 'best of 5 blocks' delta = "
      f"{float((exp_max - base).mean()):+.2f} AUC -- inflated purely by selection")
