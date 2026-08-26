"""Embed only the confound arms (X-noproto, Y-noproto-vib).

Reuses v2_embed_runs' loader verbatim rather than duplicating it, so the
confound arms are embedded by exactly the same code path as every other arm --
if the two diverged, the comparison this stage exists to make would be invalid.
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


import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import embed_cohorts as V  # noqa: E402  (runs its own module-level embedding)

runs = sorted(p.name for p in V.RUNS.iterdir()
              if p.is_dir() and "noproto" in p.name
              and (p / "checkpoints" / "glucoprism.pt").exists())
print(f"\n=== re-embedding {len(runs)} confound arms ===")
for run in runs:
    ck = V.RUNS / run / "checkpoints" / "glucoprism.pt"
    enc, pool, miss, unexp, pc = V.load(ck)
    for coh in V.COHORTS:
        for block, arr in V.embed(enc, pool, coh).items():
            np.save(V.OUT / f"{run}__{coh}__{block}.npy", arr)
    print(f"  {run:<20} missing={miss} unexpected={unexp} "
          f"vib={getattr(pc, 'use_vib', False)}")
print(f"\nwrote to {V.OUT}")
