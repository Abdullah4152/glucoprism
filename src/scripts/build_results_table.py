"""The complete benchmark table: every model, one probe, one set of frozen folds.

GlucoFM's published Table 3 numbers sit alongside ours. Theirs are on a private
corpus ~10x larger and on 37 Stanford / 65 ShanghaiT2DM subjects against our
29 / 69, so they are a reference point, not a like-for-like column.
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

import numpy as np
import pandas as pd

A = ROOT / "experiments/artifacts"

# GlucoFM Table 3, task-averaged over the 14 cells (PR / ROC / Macro-F1).
PUBLISHED = {
    "GlucoFM (their paper)": (58.8, 66.7, 59.9),
    "CGM-JEPA (their paper)": (54.7, 62.6, 57.0),
    "X-CGM-JEPA (their paper)": (55.4, 63.3, 57.4),
    "GluFormer-tiny (their paper)": (54.0, 61.9, 56.3),
}

frames = []


def add(path, tag=None, filt=None):
    p = A / path
    if not p.exists():
        return
    d = pd.read_csv(p)
    if filt is not None:
        d = d[d.run.map(filt)]
    if tag:
        d = d.assign(run=d.run.map(tag))
    frames.append(d[["run", "level", "cohort", "task", "pr", "auc", "f1"]])


add("baseline_scores.csv")
add("fd7_scores.csv", filt=lambda r: r.startswith("W3u-ov40"),
    tag=lambda r: "GlucoFM (ours)")
add("fd7seed_scores.csv", filt=lambda r: r.startswith("W3u-ov40"),
    tag=lambda r: "GlucoFM (ours)")
add("fd8_scores.csv", filt=lambda r: r.startswith("V1-fm-joint"),
    tag=lambda r: "GlucoPRISM proposal")

v = A / "v2_final_scores.csv"
if v.exists():
    d = pd.read_csv(v)
    d["arm"] = d.run.str.replace(r"-s\d$", "", regex=True)
    NAME = {"A-v2-base": "GlucoPRISM-v2", "B-v2-vib1": "GlucoPRISM-v2 + zA bottleneck",
            "C-v2-vib01": "GlucoPRISM-v2 + zA bottleneck (weak)",
            "D-v2-simbias": "GlucoPRISM-v2 + measured sensor",
            "E-v2-vib-simbias": "GlucoPRISM-v2 + bottleneck + measured sensor"}
    d["run"] = d.arm.map(NAME) + np.where(d.block == "zTzS", " [zA dropped]", " [full]")
    frames.append(d[["run", "level", "cohort", "task", "pr", "auc", "f1"]])

df = pd.concat(frames, ignore_index=True)

ORDER = [
    "Chronos-2-small", "Chronos-2", "MOMENT-small", "MOMENT-large",
    "Mantis", "MantisV2", "CGMformer",
    "GlucoPRISM proposal",
    "GlucoFM (ours)",
    "GlucoPRISM-v2 [full]", "GlucoPRISM-v2 [zA dropped]",
    "GlucoPRISM-v2 + measured sensor [full]",
    "GlucoPRISM-v2 + measured sensor [zA dropped]",
    "GlucoPRISM-v2 + zA bottleneck (weak) [full]",
    "GlucoPRISM-v2 + zA bottleneck (weak) [zA dropped]",
    "GlucoPRISM-v2 + zA bottleneck [full]",
    "GlucoPRISM-v2 + zA bottleneck [zA dropped]",
    "GlucoPRISM-v2 + bottleneck + measured sensor [full]",
    "GlucoPRISM-v2 + bottleneck + measured sensor [zA dropped]",
]
GROUP = {**{k: "general-purpose time series (zero-shot)" for k in ORDER[:6]},
         "CGMformer": "CGM-specific, external pretraining (zero-shot)",
         "GlucoPRISM proposal": "ours, pretrained on our public corpus",
         "GlucoFM (ours)": "ours, pretrained on our public corpus"}

for lvl in ("window", "subject"):
    d = df[df.level == lvl]
    print(f"\n{'='*94}\n{lvl.upper()} LEVEL â€” 14 task-dataset cells, "
          f"task-averaged\n{'='*94}")
    print(f"{'model':<48}{'n':>3}{'PR-AUC':>10}{'ROC-AUC':>10}{'Macro-F1':>10}")
    print("-" * 94)
    last = None
    for m in ORDER:
        s = d[d.run == m]
        if s.empty:
            continue
        grp = GROUP.get(m, "ours, GlucoPRISM-v2 family")
        if grp != last:
            print(f"  -- {grp} --")
            last = grp
        n = s.groupby(["cohort", "task"]).size().max() // 1
        seeds = max(1, len(s) // 14)
        g = s.groupby(["cohort", "task"])[["pr", "auc", "f1"]].mean().mean()
        print(f"{m:<48}{seeds:>3}{g.pr:>10.1f}{g.auc:>10.1f}{g.f1:>10.1f}")
    if lvl == "window":
        print("  -- published, different corpus and cohort sizes --")
        for k, (pr, auc, f1) in PUBLISHED.items():
            print(f"{k:<48}{'-':>3}{pr:>10.1f}{auc:>10.1f}{f1:>10.1f}")

df.to_csv(A / "final_table_long.csv", index=False)
print(f"\nwrote {A / 'final_table_long.csv'}")

