"""Embed a subset of kaggle_out runs, selected by substring.

    python embed_subset.py K-        # the capacity sweep
    python embed_subset.py noproto   # the confound arms

Reuses v2_embed_runs' loader and embedder verbatim so every arm is embedded by
the identical code path -- if they diverged, the comparisons would be invalid.
Block widths differ across the capacity sweep (zA is 8/16/32), so the per-block
.npy files have different widths by design; the scorer handles that because it
reads whatever width it finds.
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

import embed_cohorts as V  # noqa: E402

pat = sys.argv[1] if len(sys.argv) > 1 else "K-"
runs = sorted(p.name for p in V.RUNS.iterdir()
              if p.is_dir() and pat in p.name
              and (p / "checkpoints" / "glucoprism.pt").exists())
print(f"\n=== embedding {len(runs)} runs matching {pat!r} ===")
for run in runs:
    ck = V.RUNS / run / "checkpoints" / "glucoprism.pt"
    try:
        enc, pool, miss, unexp, pc = V.load(ck)
        for coh in V.COHORTS:
            for block, arr in V.embed(enc, pool, coh).items():
                np.save(V.OUT / f"{run}__{coh}__{block}.npy", arr)
        print(f"  {run:<16} zT={pc.d_trait} zS={pc.d_state} zA={pc.d_sensor} "
              f"w_vib={getattr(pc, 'w_vib', None)} "
              f"missing={miss} unexpected={unexp}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {run:<16} FAILED {type(exc).__name__}: {exc}")
print(f"\nwrote to {V.OUT}")
