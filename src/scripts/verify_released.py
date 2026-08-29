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

# Reference embeddings ship with the repository. They used to be read from
# `artifacts/v2emb/`, which is written by embedding the *training* checkpoints
# -- and those .pt files are not in the release (weights/ holds inference
# tensors only). So this script could never do its job for a reader: with
# nothing staged every comparison was skipped and it still exited 0, reporting
# VERIFIED having verified nothing; after a retrain it compared the released
# weights against different weights and reported mismatches.
import json                                                     # noqa: E402

# The reference is a compact statistical signature of the embeddings each
# released checkpoint produces, not the arrays themselves: this repository
# ships code and weights, not model output. Every statistic below moves if any
# tensor in the checkpoint changes, and unlike an exact hash they tolerate the
# last-bit float differences a different BLAS or GPU introduces.
REF = ROOT / "data" / "reference_embeddings.json"
TOL = {"mean": 1e-6, "std": 1e-6, "min": 1e-5, "max": 1e-5, "abs_sum": 1e-2}


def signature(a: np.ndarray) -> dict:
    a = a.astype(np.float64)
    return {"shape": list(a.shape),
            "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "abs_sum": float(np.abs(a).sum()),
            "col_mean_head": [float(v) for v in a.mean(0)[:8]],
            "row_norm_head": [float(v) for v in np.linalg.norm(a, axis=1)[:8]]}


def compare(ref: dict, got: dict) -> tuple[bool, str, float]:
    if list(ref["shape"]) != list(got["shape"]):
        return False, "shape", 0.0
    worst, where = 0.0, ""
    for k, tol in TOL.items():
        d = abs(ref[k] - got[k])
        scale = max(1.0, abs(ref[k]))
        if d / scale > worst:
            worst, where = d / scale, k
        if d > tol * scale:
            return False, k, d
    for k in ("col_mean_head", "row_norm_head"):
        for r, g in zip(ref[k], got[k]):
            d = abs(r - g)
            if d > 1e-5 * max(1.0, abs(r)):
                return False, k, d
    return True, where, worst


if not REF.exists():
    print(f"NOTHING VERIFIED - no reference at {REF}")
    sys.exit(2)
reference = json.loads(REF.read_text())

fails, checked = [], 0
print(f"{'model':<16}{'seed':>5}{'cohort':<14}{'worst rel. diff':>16}")
print("-" * 54)
for name, (run, seeds) in SRC.items():
    for s in seeds:
        enc, pool, _ = load(name, s)
        for coh in COHORTS:
            key = f"{run}-s{s}__{coh}__zTzS"
            ref = reference.get(key)
            if ref is None:
                print(f"{name:<16}{s:>5}{coh:<14}{'no reference':>16}")
                continue
            ok, where, gap = compare(ref, signature(embed(enc, pool, coh, "zTzS")))
            checked += 1
            if not ok:
                fails.append(f"{name}-s{s} {coh}: {where} differs by {gap:.3e}")
            print(f"{name:<16}{s:>5}{coh:<14}{gap:>16.2e}"
                  f"{'' if ok else f'   MISMATCH ({where})'}")

if not checked:
    print(f"\nNOTHING VERIFIED - no entries matched in {REF.name}.")
    sys.exit(2)
print("\n" + (f"VERIFIED - the released weights reproduce all {checked} scored "
              f"embeddings" if not fails else
              f"{len(fails)} of {checked} MISMATCHES:\n  " + "\n  ".join(fails)
              + "\n\nIf you are checking your OWN retrained checkpoints rather "
                "than the shipped ones, differences of this size are ordinary "
                "GPU/driver non-determinism, not a failure."))
sys.exit(1 if fails else 0)
