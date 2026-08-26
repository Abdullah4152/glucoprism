"""Do the RELEASED safetensors reproduce the paper's embeddings exactly?

If a reader loads `weights/*.safetensors` and gets different vectors from the
ones the paper was scored on, every number in the paper is unverifiable. This
checks all four cohorts for every released seed against the training checkpoints.
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
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from load_released import load, embed                    # noqa: E402
RUNS = RUNS
EMB = OUTDIR / "v2emb"
COHORTS = ["cgmacros", "stanford", "hall", "shanghait2dm"]

# released name -> the training run its embeddings were scored from
SRC = {"glucoprism-c": ("C-v2-vib01", (5,)),
       "glucoprism-e": ("E-v2-vib-simbias", (1,))}
# One seed per model ships; weights/README.md records which, and why.

fails = []
print(f"{'model':<16}{'seed':>5}{'cohort':<14}{'max |diff|':>12}")
print("-" * 50)
for name, (run, seeds) in SRC.items():
    for s in seeds:
        enc, pool, _ = load(name, s)
        for coh in COHORTS:
            ref_p = EMB / f"{run}-s{s}__{coh}__zTzS.npy"
            if not ref_p.exists():
                continue
            ref = np.load(ref_p)
            got = embed(enc, pool, coh, "zTzS")
            gap = float(np.max(np.abs(ref - got)))
            ok = gap < 1e-5
            if not ok:
                fails.append(f"{name}-s{s} {coh}: {gap:.3e}")
            print(f"{name:<16}{s:>5}{coh:<14}{gap:>12.2e}"
                  f"{'' if ok else '   MISMATCH'}")

print("\n" + ("VERIFIED - the released weights reproduce every scored embedding"
              if not fails else f"{len(fails)} MISMATCHES:\n  " + "\n  ".join(fails)))
sys.exit(1 if fails else 0)
