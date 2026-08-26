"""Compute the per-dataset statistics that the docs in docs/datasets/ quote.

Everything written to artifacts/dataset_profiles.json is measured from the files
actually on disk -- nothing is copied from a paper. The dataset docs then cite
both: the measured value and the value the source publication reports, so any
divergence is visible rather than buried.

    python scripts/profile_datasets.py --datasets stanford hall colas
    python scripts/profile_datasets.py --all
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


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(ROOT / "src" / "core"))

from cgmkit.data import harmonize, labels as L  # noqa: E402
from cgmkit.data.windows import split_segments  # noqa: E402


def profile(name: str) -> dict:
    df = harmonize.READERS[name]()
    p: dict = {"dataset": name}

    p["n_readings"] = int(len(df))
    p["n_subjects"] = int(df["subject"].nunique())
    p["devices"] = sorted(df["device"].unique().tolist())
    p["sampling_min"] = sorted(df["sampling_min"].unique().tolist())
    p["date_range"] = [str(df["timestamp"].min()), str(df["timestamp"].max())]

    per_dev = {}
    for dev, g in df.groupby("device"):
        rate = int(g["sampling_min"].iloc[0])
        per_dev[dev] = {
            "subjects": int(g["subject"].nunique()),
            "readings": int(len(g)),
            "sampling_min": rate,
            "monitoring_hours": round(len(g) * rate / 60.0, 1),
        }
    p["per_device"] = per_dev
    p["monitoring_hours"] = round(sum(v["monitoring_hours"] for v in per_dev.values()), 1)

    gl = df["glucose_mgdl"]
    p["glucose_mgdl"] = {
        "min": float(gl.min()), "p1": float(gl.quantile(0.01)),
        "p25": float(gl.quantile(0.25)), "median": float(gl.median()),
        "mean": round(float(gl.mean()), 2), "p75": float(gl.quantile(0.75)),
        "p99": float(gl.quantile(0.99)), "max": float(gl.max()),
        "std": round(float(gl.std()), 2),
        "pct_below_70": round(float((gl < 70).mean() * 100), 3),
        "pct_above_180": round(float((gl > 180).mean() * 100), 3),
        "pct_in_range_70_180": round(float(((gl >= 70) & (gl <= 180)).mean() * 100), 2),
    }

    # per-subject wear duration and segment structure
    dur, nseg, gaps = [], [], []
    for _, g in df.groupby("subject"):
        span = (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 86400.0
        dur.append(span)
        seg = split_segments(g["timestamp"])
        nseg.append(int(len(np.unique(seg))))
        dt = g["timestamp"].diff().dt.total_seconds().div(60).dropna()
        gaps.append(float(dt.max()) if len(dt) else 0.0)
    p["days_per_subject"] = {"min": round(min(dur), 2), "median": round(float(np.median(dur)), 2),
                             "mean": round(float(np.mean(dur)), 2), "max": round(max(dur), 2)}
    p["segments_per_subject"] = {"median": float(np.median(nseg)), "max": int(max(nseg))}
    p["largest_gap_minutes"] = {"median": round(float(np.median(gaps)), 1),
                                "max": round(float(max(gaps)), 1)}

    # observed inter-reading interval, to confirm the nominal sampling rate
    ivals = []
    for _, g in df.groupby(["subject", "device"]):
        d = g["timestamp"].diff().dt.total_seconds().div(60).dropna()
        ivals.append(d[d <= 60])
    if ivals:
        iv = pd.concat(ivals)
        p["observed_interval_min"] = {"median": round(float(iv.median()), 2),
                                      "p90": round(float(iv.quantile(0.9)), 2)}

    fn = L.LABEL_READERS.get(name)
    if fn is not None:
        try:
            tbl = fn()
            tasks = L.TASK_MATRIX.get(name, [])
            p["labels"] = {
                "n_rows": int(len(tbl)),
                "columns": [c for c in tbl.columns if c != "subject"],
                "tasks": {t: {"n_labelled": int(tbl[t].notna().sum()),
                              "counts": {str(k): int(v)
                                         for k, v in tbl[t].value_counts(dropna=True).items()}}
                          for t in tasks if t in tbl.columns},
            }
            cols = [t for t in tasks if t in tbl.columns]
            if cols:
                p["labels"]["complete_profile_subjects"] = int(tbl[cols].notna().all(axis=1).sum())
        except Exception as e:  # noqa: BLE001
            p["labels"] = {"error": f"{type(e).__name__}: {e}"}
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    names = a.datasets or (list(harmonize.READERS) if a.all else [])
    if not names:
        ap.error("pass --all or --datasets ...")

    dest = ROOT / "artifacts" / "dataset_profiles.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(dest.read_text()) if dest.exists() else {}

    for n in names:
        print(f"=== profiling {n} ...", flush=True)
        try:
            out[n] = profile(n)
            d = out[n]
            print(f"    {d['n_subjects']} subjects, {d['n_readings']:,} readings, "
                  f"{d['monitoring_hours']:,.0f} h, devices={d['devices']}")
        except Exception as e:  # noqa: BLE001
            print(f"    [FAILED] {type(e).__name__}: {e}")
            out[n] = {"dataset": n, "error": f"{type(e).__name__}: {e}"}
        dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
