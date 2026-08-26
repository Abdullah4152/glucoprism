"""Does dropping zA at inference -- what the proposal prescribes -- help transfer?

And is it specific to the factorization, or would truncating ANY embedding do the
same? The control is our own GlucoFM: drop its last 16 dimensions, which carry no
designated meaning. If that also gains, the effect is truncation, not zA.
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


import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = ROOT
for p in ("src", "scripts", "experiments/scripts"):
    sys.path.insert(0, str(ROOT / p))

import run_eval as RE                                      # noqa: E402
from cgmkit.data.datasets import WindowShard           # noqa: E402
from fd3_cross_dataset import labels, COHORTS, TASKS       # noqa: E402
from fd3_block_controls import transfer, subj              # noqa: E402

EMB = ROOT / "experiments" / "artifacts" / "v2emb"
OUT = ROOT / "experiments" / "kaggle_out"
shards = {c: WindowShard(ROOT / "data" / "processed" / f"{c}_ds.npz")
          for c in COHORTS}

recs = []
for seed in (0, 1, 2):
    full = {c: np.load(EMB / f"v2r-s{seed}__{c}__full.npy") for c in COHORTS}
    zT = {c: np.load(EMB / f"v2r-s{seed}__{c}__zT.npy") for c in COHORTS}
    zS = {c: np.load(EMB / f"v2r-s{seed}__{c}__zS.npy") for c in COHORTS}
    zTzS = {c: np.concatenate([zT[c], zS[c]], 1) for c in COHORTS}

    # Control: our GlucoFM, same 128-d width, truncated to the same 112.
    ck = OUT / f"V4-fm-off-s{seed}" / "checkpoints" / "glucofm.pt"
    fm = {c: RE.EMBEDDERS["glucofm"][1](ck, s) for c, s in shards.items()}
    fm112 = {c: v[:, :112] for c, v in fm.items()}
    fm64 = {c: v[:, :64] for c, v in fm.items()}

    variants = {
        "v2 full (128)": full,
        "v2 zT||zS (112) <- proposal": zTzS,
        "v2 zT (64)": zT,
        "GlucoFM full (128)": fm,
        "GlucoFM first 112": fm112,
        "GlucoFM first 64": fm64,
    }
    for name, E in variants.items():
        for src, tgt in itertools.permutations(COHORTS, 2):
            for task in TASKS:
                a = transfer(lambda c, k, E=E: E[c][k], src, tgt, task)
                if a is not None:
                    recs.append(dict(seed=seed, variant=name, src=src,
                                     tgt=tgt, task=task, auc=a))

df = pd.DataFrame(recs)
df.to_csv(ROOT / "experiments" / "artifacts" / "fd3_drop_za.csv", index=False)

g = df.groupby(["variant", "seed", "src", "tgt", "task"]).auc.mean()
print(f"{'variant':<30}{'s0':>7}{'s1':>7}{'s2':>7}{'mean':>8}")
print("-" * 60)
for v in ["v2 full (128)", "v2 zT||zS (112) <- proposal", "v2 zT (64)",
          "GlucoFM full (128)", "GlucoFM first 112", "GlucoFM first 64"]:
    per = [g.xs((v, s), level=("variant", "seed")).mean() for s in (0, 1, 2)]
    print(f"{v:<30}{per[0]:>7.1f}{per[1]:>7.1f}{per[2]:>7.1f}{np.mean(per):>8.2f}")


def paired(a, b, label):
    x = g.xs(a, level="variant")
    y = g.xs(b, level="variant")
    d = (x - y).dropna()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    verdict = "REAL" if abs(t) > 2.03 else "inside noise"
    print(f"  {label:<44}{d.mean():>+7.2f} AUC  t={t:>6.2f}   {verdict}")


print("\n--- does dropping zA help? ---")
paired("v2 zT||zS (112) <- proposal", "v2 full (128)", "v2: drop zA (128 -> 112)")
print("\n--- CONTROL: does truncating GlucoFM by the same amount help? ---")
paired("GlucoFM first 112", "GlucoFM full (128)", "GlucoFM: drop last 16 (128 -> 112)")
paired("GlucoFM first 64", "GlucoFM full (128)", "GlucoFM: keep first 64")
print("\n--- the headline comparison ---")
paired("v2 zT||zS (112) <- proposal", "GlucoFM full (128)",
       "proposal readout vs our GlucoFM")
