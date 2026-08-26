"""Embed only the confound arms (X-noproto, Y-noproto-vib).

Reuses v2_embed_runs' loader verbatim rather than duplicating it, so the
confound arms are embedded by exactly the same code path as every other arm --
if the two diverged, the comparison this stage exists to make would be invalid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import v2_embed_runs as V  # noqa: E402  (runs its own module-level embedding)

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
