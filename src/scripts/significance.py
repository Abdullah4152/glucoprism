"""Paired Wilcoxon + Holm-Bonferroni across the 14 cells, with effect sizes.

GlucoFM reports no significance testing. We criticised that, so we owe it.

Pairing is what makes this readable: every model sees the SAME 14 cells on the
SAME frozen folds, so the per-cell difference removes the cell-to-cell variation
that dominates the raw spread.

Holm-Bonferroni over the family of comparisons against the reference, because
testing ~15 models against one baseline at alpha=.05 would otherwise produce a
false positive by construction.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

A = ROOT / "experiments/artifacts"


def cliffs_delta(x, y):
    """Non-parametric effect size: P(x>y) - P(x<y), in [-1, 1].

    Reported instead of Cohen's d because 14 paired cells are not normal and a
    standardised mean difference would overstate precision.
    """
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > yj) for xi in x for yj in y)
    lt = sum((xi < yj) for xi in x for yj in y)
    return (gt - lt) / (len(x) * len(y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="GlucoFM (ours)")
    ap.add_argument("--metric", default="auc", choices=["auc", "pr", "f1"])
    ap.add_argument("--level", default="window", choices=["window", "subject"])
    ap.add_argument("--family", default="confirmatory",
                    choices=["confirmatory", "all"],
                    help="'confirmatory' corrects over the models we CLAIM about "
                         "-- the released models and the baselines they are "
                         "claimed to beat. 'all' corrects over every "
                         "configuration ever run, ablations included.")
    ap.add_argument("--released", nargs="*", default=[
        "GlucoPRISM-v2 + zA bottleneck (weak) [zA dropped]",
        "GlucoPRISM-v2 + bottleneck + measured sensor [zA dropped]"])
    a = ap.parse_args()

    df = pd.read_csv(A / "final_table_long.csv")
    d = df[df.level == a.level]
    # One value per (model, cell): average over seeds first, so the paired unit
    # is the cell, not the seed-cell.
    piv = d.groupby(["run", "cohort", "task"])[a.metric].mean().unstack(["cohort", "task"])
    if a.ref not in piv.index:
        raise SystemExit(f"reference {a.ref!r} not present")
    ref = piv.loc[a.ref]

    # A multiple-comparison family should be the set of claims being made, not
    # every configuration that was ever run. Ablations of our own model (beta
    # strength, readout choice, the sensor-offset arm) explore ONE model; they
    # do not each assert a separate claim about beating the baselines, and
    # letting them inflate the correction penalises us for being thorough.
    #
    # The confirmatory family is therefore: the released models, plus the
    # baselines they are claimed to beat. Ablations are reported separately,
    # uncorrected, and labelled exploratory.
    if a.family == "confirmatory":
        keep = set(a.released) | {m for m in piv.index
                                  if "GlucoPRISM" not in m and m != a.ref}
        piv = piv.loc[[m for m in piv.index if m in keep or m == a.ref]]

    rows = []
    for m in piv.index:
        if m == a.ref:
            continue
        v = piv.loc[m]
        both = v.notna() & ref.notna()
        x, y = v[both].to_numpy(float), ref[both].to_numpy(float)
        diff = x - y
        try:
            stat, p = wilcoxon(x, y)
        except ValueError:
            p = 1.0
        rows.append(dict(model=m, n=int(both.sum()), mean_delta=diff.mean(),
                         median_delta=float(np.median(diff)),
                         wins=int((diff > 0).sum()), p_raw=p,
                         cliffs=cliffs_delta(x, y)))

    r = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    # Holm-Bonferroni: sort p ascending, threshold alpha/(m-i), and once a test
    # fails every later one fails too.
    m = len(r)
    r["p_holm"] = [min(1.0, (m - i) * p) for i, p in enumerate(r.p_raw)]
    r["p_holm"] = np.maximum.accumulate(r.p_holm)
    r["sig"] = np.where(r.p_holm < 0.05, "*", "")

    print(f"\nPaired Wilcoxon over {int(r.n.max())} cells vs {a.ref!r} "
          f"({a.level} level, {a.metric.upper()})")
    print(f"Family: {a.family} (k = {m}). Holm-Bonferroni; * = adjusted p < .05\n")
    print(f"{'model':<48}{'mean d':>8}{'wins':>6}{'p raw':>9}{'p holm':>9}"
          f"{'cliff':>7}  ")
    print("-" * 90)
    for _, x in r.iterrows():
        print(f"{x.model:<48}{x.mean_delta:>+8.2f}{x.wins:>4}/{x.n:<2}"
              f"{x.p_raw:>9.4f}{x.p_holm:>9.4f}{x.cliffs:>+7.2f}  {x.sig}")

    r.to_csv(A / f"significance_{a.level}_{a.metric}.csv", index=False)
    print(f"\nwrote significance_{a.level}_{a.metric}.csv")
    print("\nCliff's delta: |d|<0.15 negligible, <0.33 small, <0.47 medium, "
          "else large.")


if __name__ == "__main__":
    main()
