"""Stage 1: embed the downstream cohorts with the v2 stack -- our best-performing
GlucoPRISM (the only one that ever beat GlucoFM: 68.3 window / 70.9 subject).

Runs in its own process with ONLY the v2 tree on sys.path, because that package
is also called `glucoprism` and cannot be imported alongside ours. Writes plain
.npy so stage 2 can score it with our probe.

Also writes the individual blocks, so cross-dataset transfer can be measured for
zT / zS / zA separately -- the readout the proposal actually claims.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))


import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

REF = str(ROOT / "external/glucoprism_v2_reference")
sys.path.insert(0, REF)
from glucofm.config import Config                       # noqa: E402
from glucofm.model import GlucoFMEncoder                # noqa: E402
from glucoprism.model import BlockedPool, PrismConfig   # noqa: E402

PROC = str(ROOT / "data/processed")
OUT = str(ROOT / "experiments/artifacts/v2emb")
WEIGHTS = r"D:\glucoprism_v2\weights"
COHORTS = ["cgmacros", "stanford", "hall", "shanghait2dm"]
os.makedirs(OUT, exist_ok=True)


def load(ck):
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
    return enc, pool, len(m1) + len(m2), len(u1) + len(u2)


@torch.no_grad()
def embed(enc, pool, coh, batch=256):
    d = np.load(f"{PROC}/{coh}_ds.npz", allow_pickle=True)
    g = np.nan_to_num(d["glucose"].astype(np.float32), nan=0.0)
    m = d["mask"].astype(np.float32)
    s = d["start_idx"].astype(np.int64)
    parts = {"full": [], "zT": [], "zS": [], "zA": []}
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
        parts["full"].append(torch.cat([zT, zS, zA], -1).numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


for seed in (0, 1, 2):
    ck = os.path.join(WEIGHTS, f"v2r-s{seed}.pt")
    if not os.path.exists(ck):
        print(f"  missing {ck}")
        continue
    enc, pool, miss, unexp = load(ck)
    for coh in COHORTS:
        for block, arr in embed(enc, pool, coh).items():
            np.save(f"{OUT}/v2r-s{seed}__{coh}__{block}.npy", arr)
    print(f"  v2r-s{seed}: missing={miss} unexpected={unexp}  "
          f"dims full={arr.shape[1] if block == 'full' else '?'}")
print(f"\nwrote to {OUT}")
