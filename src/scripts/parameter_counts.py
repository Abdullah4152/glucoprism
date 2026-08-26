"""Trainable vs total parameters for every released model.

The distinction matters: a JEPA-style model carries an EMA *target* branch that
is a frozen copy of the online encoder and receives no gradient. Summing every
tensor in a checkpoint therefore roughly doubles the count and does not match
what the papers quote. GlucoFM is 0.72 M trainable / 1.17 M total.
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

import torch

RUNS = ROOT / "experiments" / "kaggle_out"

MODELS = {
    "GlucoFM (ours)": ("W3u-ov40", "glucofm.pt"),
    "GlucoPRISM (proposal)": ("V1-fm-joint-s0", "prism.pt"),
    "GlucoPRISM-C": ("C-v2-vib01-s0", "glucoprism.pt"),
    "GlucoPRISM-E": ("E-v2-vib-simbias-s0", "glucoprism.pt"),
}

# Keys whose tensors are a frozen EMA copy of the online branch, or optimiser /
# bookkeeping state, and are therefore NOT trainable parameters.
FROZEN_HINTS = ("target", "ema", "momentum")


def walk(o, pre=""):
    for k, v in o.items():
        key = f"{pre}{k}"
        if torch.is_tensor(v):
            yield key, v
        elif isinstance(v, dict):
            yield from walk(v, f"{key}.")


print(f"{'model':<24}{'trainable':>12}{'EMA target':>12}{'total':>12}"
      f"{'file MB':>9}")
print("-" * 71)
for name, (run, ck) in MODELS.items():
    p = RUNS / run / "checkpoints" / ck
    if not p.exists():
        print(f"{name:<24}  checkpoint not found ({run})")
        continue
    blob = torch.load(p, map_location="cpu", weights_only=False)

    seen, train, frozen = set(), 0, 0
    for k, v in walk(blob):
        ident = (v.untyped_storage().data_ptr(), tuple(v.shape),
                 tuple(v.stride()), v.storage_offset())
        if ident in seen:
            continue                       # the `full.` alias of the same tensor
        seen.add(ident)
        if any(h in k.lower() for h in FROZEN_HINTS):
            frozen += v.numel()
        else:
            train += v.numel()
    mb = p.stat().st_size / 1e6
    print(f"{name:<24}{train:>12,}{frozen:>12,}{train + frozen:>12,}{mb:>9.1f}")

print("\nPublished reference: GlucoFM 720,241 trainable / 1,173,698 total.")

