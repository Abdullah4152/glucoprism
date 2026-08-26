"""Stage 1 for the v2 finalisation runs: embed the downstream cohorts.

Separate process, only the v2 tree on sys.path -- their package is also called
`glucoprism`. Writes each block separately so the zA-drop readout can be scored
against the full one.
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


import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
REF = str(REFERENCE)
sys.path.insert(0, REF)
from glucofm.config import Config                       # noqa: E402
from glucofm.model import GlucoFMEncoder                # noqa: E402
from glucoprism.model import BlockedPool, PrismConfig   # noqa: E402

PROC = (ROOT / "data/processed")
RUNS = (RUNS)
OUT = (OUTDIR / "v2emb")
COHORTS = ["cgmacros", "stanford", "hall", "shanghait2dm"]
OUT.mkdir(parents=True, exist_ok=True)


def load(ck: Path):
    b = torch.load(ck, map_location="cpu", weights_only=False)
    fm = Config()
    for sec in ("model", "grid", "filt"):
        for k, v in b["fm_config"].get(sec, {}).items():
            if hasattr(getattr(fm, sec), k):
                setattr(getattr(fm, sec), k, v)
    pc = PrismConfig()
    for k, v in b["prism_config"].items():
        if hasattr(pc, k):
            setattr(pc, k, v)
    enc = GlucoFMEncoder(fm)
    m1, u1 = enc.load_state_dict(b["online"], strict=False)
    pool = BlockedPool(fm.model.embed_dim, pc)
    m2, u2 = pool.load_state_dict(b["pool"], strict=False)
    enc.eval()
    pool.eval()
    return enc, pool, len(m1) + len(m2), len(u1) + len(u2), pc


@torch.no_grad()
def embed(enc, pool, coh, batch=256):
    d = np.load(PROC / f"{coh}_ds.npz", allow_pickle=True)
    g = np.nan_to_num(d["glucose"].astype(np.float32), nan=0.0)
    m = d["mask"].astype(np.float32)
    s = d["start_idx"].astype(np.int64)
    parts: dict[str, list] = {"full": [], "zT": [], "zS": [], "zA": [], "zTzS": []}
    for i in range(0, len(g), batch):
        sl = slice(i, i + batch)
        z, *_ = enc(torch.tensor(g[sl]), torch.tensor(m[sl]),
                    torch.tensor(s[sl]), patch_mask=None)
        r = pool(z)
        zT, zS, zA = ((r["zT"], r["zS"], r["zA"]) if isinstance(r, dict)
                      else (r[0], r[1], r[2]))
        parts["zT"].append(zT.numpy())
        parts["zS"].append(zS.numpy())
        parts["zA"].append(zA.numpy())
        parts["zTzS"].append(torch.cat([zT, zS], -1).numpy())
        parts["full"].append(torch.cat([zT, zS, zA], -1).numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


runs = sorted(p.name for p in RUNS.iterdir()
              if p.is_dir() and (p / "checkpoints" / "glucoprism.pt").exists())
print(f"{len(runs)} v2 checkpoints\n")
for run in runs:
    ck = RUNS / run / "checkpoints" / "glucoprism.pt"
    try:
        enc, pool, miss, unexp, pc = load(ck)
        for coh in COHORTS:
            for block, arr in embed(enc, pool, coh).items():
                np.save(OUT / f"{run}__{coh}__{block}.npy", arr)
        print(f"  {run:<18} missing={miss} unexpected={unexp} "
              f"vib={getattr(pc, 'use_vib', False)}")
    except Exception as e:  # noqa: BLE001
        print(f"  {run:<18} FAILED {type(e).__name__}: {e}")
print(f"\nwrote to {OUT}")
