"""Per-arm corpus totals, so a sweep's arms are described by measurement."""
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


import sys
from pathlib import Path

import numpy as np

P = (ROOT / "data/processed")
tags = sys.argv[1:] or ["", "_ov0", "_ov20", "_ov40"]

print(f"{'arm':<9}{'windows':>9}{'subjects':>10}{'RBG win':>9}{'RBG %':>8}   per-cohort")
print("-" * 104)
for tag in tags:
    tot, rbg, subs, parts = 0, 0, set(), []
    for f in sorted(P.glob(f"*_pt{tag}.npz")):
        if tag == "" and any(c in f.stem for c in ("_ov", "_bal", "_ctl", "_v2r")):
            continue
        with np.load(f, allow_pickle=True) as z:
            n = len(z["glucose"])
            tot += n
            subs |= {str(s) for s in z["subject"]}
            parts.append(f"{f.stem.split('_pt')[0][:6]}={n:,}")
            if f.stem.startswith("replacebg"):
                rbg = n
    if not tot:
        print(f"{tag or '(base)':<9}   (no shards)")
        continue
    print(f"{tag or '(base)':<9}{tot:>9,}{len(subs):>10}{rbg:>9,}"
          f"{100 * rbg / tot:>7.1f}%   {'  '.join(parts)}")
