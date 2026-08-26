"""Where a released checkpoint lives.

The development tree kept every seed in one flat directory
(`glucoprism-c-s0.safetensors`). The release ships one checkpoint per model in
its own folder, chosen by the rule in `weights/README.md`. Scripts ask here
rather than building paths themselves, so the two layouts cannot drift.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("GLUCOPRISM_ROOT",
                           Path(__file__).resolve().parents[3]))
WEIGHTS = ROOT / "weights"

# accepted aliases -> folder under weights/
ALIASES = {
    "glucoprism-c": "glucoprism_c", "glucoprism_c": "glucoprism_c",
    "glucoprism-e": "glucoprism_e", "glucoprism_e": "glucoprism_e",
    "glucofm": "glucofm", "glucofm-ours": "glucofm", "glucofm.pt": "glucofm",
    "cgm-jepa": "cgm_jepa", "cgm_jepa": "cgm_jepa", "cgm_jepa.pt": "cgm_jepa",
    "x-cgm-jepa": "x_cgm_jepa", "x_cgm_jepa": "x_cgm_jepa",
    "x_cgm_jepa.pt": "x_cgm_jepa",
    "gluformer": "gluformer", "gluformer_tiny": "gluformer",
    "gluformer_tiny.pt": "gluformer",
}


def folder(model: str) -> Path:
    key = ALIASES.get(model, ALIASES.get(model.lower(), model))
    d = WEIGHTS / key
    if not d.is_dir():
        raise FileNotFoundError(
            f"no released weights for {model!r} at {d}. "
            f"Available: {sorted(p.name for p in WEIGHTS.iterdir() if p.is_dir())}")
    return d


def checkpoint(model: str, seed: int | None = None) -> Path:
    """The released .safetensors for a model.

    `seed` is accepted and ignored: exactly one seed per model is released, so
    a caller that asks for a specific one gets the released one rather than a
    confusing failure. `weights/README.md` records which seed that is and why.
    """
    d = folder(model)
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {d}")
    return files[0]


def config(model: str, seed: int | None = None) -> Path | None:
    p = folder(model) / "config.json"
    return p if p.exists() else None
