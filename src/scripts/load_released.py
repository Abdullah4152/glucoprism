"""Load a RELEASED model from its safetensors file and embed a cohort.

Everything in the paper can be reproduced from `weights/*.safetensors` rather
than from the training checkpoints. This is the path a reader has: the .pt files
are internal, the safetensors are what ships.

    python load_released.py --model glucoprism-c --seed 0 --cohort stanford

Verified in `verify_released_reproduces.py` to give embeddings identical to the
training checkpoints, so the paper's numbers and the released artifacts cannot
drift apart.
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
import warnings
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

warnings.filterwarnings("ignore")
REF = REFERENCE
sys.path.insert(0, str(REF))
from glucofm.config import Config                       # noqa: E402
from glucofm.model import GlucoFMEncoder                # noqa: E402
from glucoprism.model import BlockedPool, PrismConfig   # noqa: E402

from cgmkit import release_weights as _rw
WEIGHTS = _rw.WEIGHTS
PROC = ROOT / "data" / "processed"


def load(model: str, seed: int):
    """Rebuild encoder + pool from the released safetensors and its config."""
    st = load_file(str(_rw.checkpoint(model, seed)))
    cfg = json.loads((_rw.config(model, seed)).read_text())

    fm = Config()
    for sec in ("model", "grid", "filt"):
        for k, v in cfg["fm_config"].get(sec, {}).items():
            if hasattr(getattr(fm, sec), k):
                setattr(getattr(fm, sec), k, v)
    pc = PrismConfig()
    for k, v in cfg["prism_config"].items():
        if hasattr(pc, k):
            setattr(pc, k, v)

    enc = GlucoFMEncoder(fm)
    enc.load_state_dict({k[len("online."):]: v for k, v in st.items()
                         if k.startswith("online.")}, strict=False)
    pool = BlockedPool(fm.model.embed_dim, pc)
    pool.load_state_dict({k[len("pool."):]: v for k, v in st.items()
                          if k.startswith("pool.")}, strict=False)
    enc.eval()
    pool.eval()
    return enc, pool, pc


@torch.no_grad()
def embed(enc, pool, cohort: str, readout: str = "zTzS", batch: int = 256,
          seed: int = 0) -> np.ndarray:
    """`readout` is 'zTzS' (the released readout, zA discarded) or 'full'.

    The bottleneck makes zA a stochastic channel, so the RNG is seeded per call
    for reproducibility even though the released readout excludes zA.
    """
    torch.manual_seed(seed)
    d = np.load(PROC / f"{cohort}_ds.npz", allow_pickle=True)
    g = np.nan_to_num(d["glucose"].astype(np.float32), nan=0.0)
    m = d["mask"].astype(np.float32)
    s = d["start_idx"].astype(np.int64)
    out = []
    for i in range(0, len(g), batch):
        sl = slice(i, i + batch)
        z, *_ = enc(torch.tensor(g[sl]), torch.tensor(m[sl]),
                    torch.tensor(s[sl]), patch_mask=None)
        r = pool(z)
        zT, zS, zA = ((r["zT"], r["zS"], r["zA"]) if isinstance(r, dict)
                      else (r[0], r[1], r[2]))
        parts = [zT, zS] if readout == "zTzS" else [zT, zS, zA]
        out.append(torch.cat(parts, -1).numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glucoprism-c",
                    choices=["glucoprism-c", "glucoprism-e"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cohort", default="stanford")
    ap.add_argument("--readout", default="zTzS", choices=["zTzS", "full"])
    a = ap.parse_args()
    enc, pool, pc = load(a.model, a.seed)
    e = embed(enc, pool, a.cohort, a.readout)
    print(f"{a.model}-s{a.seed}  {a.cohort}  readout={a.readout}  -> {e.shape}")
    print(f"blocks: zT={pc.d_trait} zS={pc.d_state} zA={pc.d_sensor}")
