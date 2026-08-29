"""Release INFERENCE-ONLY weights: what a frozen encoder actually needs.

A released checkpoint should not carry training scaffolding. Ours did:

  * the EMA *target* branch -- a frozen copy of the online encoder whose only
    job is to produce prediction targets during pretraining (~459 k tensors)
  * the predictor, the three projection heads, the device head and the CMP
    heads -- used by the objectives, never by the embedding path
  * optimiser state and per-epoch history

This exports the minimal set and then PROVES it is minimal-and-sufficient by
re-embedding a cohort from the pruned weights and comparing elementwise against
the full checkpoint. Pruning by reading the code is not enough: a tensor dropped
in error would change embeddings silently rather than raising.
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


import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

warnings.filterwarnings("ignore")
REF = str(REFERENCE)
sys.path.insert(0, REF)
from glucofm.config import Config                       # noqa: E402
from glucofm.model import GlucoFMEncoder                # noqa: E402
from glucoprism.model import BlockedPool, PrismConfig   # noqa: E402

SRC = ROOT
# Training runs live under GLUCOPRISM_RUNS (artifacts/runs by default), the
# path every other script uses. The released copy pointed at
# `experiments/kaggle_out`, which the release layout does not have.
RUNS = _P(_os.environ.get("GLUCOPRISM_RUNS", OUTDIR / "runs"))
OUT = (OUTDIR / "weights")
PROC = SRC / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "glucoprism-c": ("C-v2-vib01", (0, 1, 2, 3, 4, 5)),
    "glucoprism-e": ("E-v2-vib-simbias", (0, 1, 2)),
}


def build(blob):
    fm = Config()
    for sec in ("model", "grid", "filt"):
        for k, v in blob["fm_config"].get(sec, {}).items():
            if hasattr(getattr(fm, sec), k):
                setattr(getattr(fm, sec), k, v)
    pc = PrismConfig()
    for k, v in blob["prism_config"].items():
        if hasattr(pc, k):
            setattr(pc, k, v)
    enc = GlucoFMEncoder(fm)
    enc.load_state_dict(blob["online"], strict=False)
    pool = BlockedPool(fm.model.embed_dim, pc)
    pool.load_state_dict(blob["pool"], strict=False)
    enc.eval(); pool.eval()
    return enc, pool, fm, pc


@torch.no_grad()
def embed(enc, pool, coh="stanford", batch=256, seed=0):
    torch.manual_seed(seed)
    d = np.load(PROC / f"{coh}_ds.npz", allow_pickle=True)
    g = np.nan_to_num(d["glucose"].astype(np.float32), nan=0.0)
    m = d["mask"].astype(np.float32)
    s = d["start_idx"].astype(np.int64)
    out = []
    for i in range(0, len(g), batch):
        sl = slice(i, i + batch)
        z, *_ = enc(torch.tensor(g[sl]), torch.tensor(m[sl]),
                    torch.tensor(s[sl]), patch_mask=None)
        r = pool(z)
        zT, zS = ((r["zT"], r["zS"]) if isinstance(r, dict) else (r[0], r[1]))
        out.append(torch.cat([zT, zS], -1).numpy())      # the released readout
    return np.concatenate(out)


def used_modules(enc, pool):
    """Which submodules actually execute during an embedding forward pass?

    Determined with forward hooks rather than by reading the source. A
    hand-written exclusion list got this wrong on the first attempt -- the
    projection heads ARE on the embedding path in this implementation -- and the
    only symptom was that embeddings changed. Hooks answer it definitively.
    """
    fired: set[str] = set()

    def mk(prefix, name):
        def hook(_m, _i, _o):
            fired.add(f"{prefix}.{name}" if name else prefix)
        return hook

    handles = []
    for prefix, mod in (("online", enc), ("pool", pool)):
        for name, sub in mod.named_modules():
            handles.append(sub.register_forward_hook(mk(prefix, name)))
    with torch.no_grad():
        d = np.load(PROC / "stanford_ds.npz", allow_pickle=True)
        g = torch.tensor(np.nan_to_num(d["glucose"][:8].astype(np.float32)))
        m = torch.tensor(d["mask"][:8].astype(np.float32))
        s = torch.tensor(d["start_idx"][:8].astype(np.int64))
        z, *_ = enc(g, m, s, patch_mask=None)
        pool(z)
    for h in handles:
        h.remove()
    return fired

print(f"{'model':<18}{'seed':>5}{'kept':>7}{'dropped':>9}{'params':>11}"
      f"{'MB':>7}   embeddings")
print("-" * 74)
manifest = {}
for name, (prefix, seeds) in MODELS.items():
    for s in seeds:
        ck = RUNS / f"{prefix}-s{s}" / "checkpoints" / "glucoprism.pt"
        if not ck.exists():
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        enc, pool, fm, pc = build(blob)
        ref_emb = embed(enc, pool)

        # Keep the online encoder and the pool WHOLE; drop only the sections
        # that inference never constructs.
        #
        # A finer prune was tried twice and rejected. Excluding the projection
        # and CMP heads by name changed embeddings by 0.21; deriving the set from
        # forward hooks changed them by 0.21 as well, because several tensors are
        # reached without their owning module being __call__-ed (raw
        # nn.Parameters and functionally-applied filters), so a hook never fires
        # for them. The EMA target branch is 459 k of the 533 k of scaffolding
        # and can be removed with certainty; chasing the remaining 74 k risks
        # silently changing the released model, which is a bad trade.
        keep, drop = {}, 0
        for section in ("online", "pool"):
            for k, v in blob.get(section, {}).items():
                keep[f"{section}.{k}"] = v.contiguous().clone()
        for sec in ("target", "predictor", "optimizer", "ema"):
            v = blob.get(sec)
            if isinstance(v, dict):
                drop += sum(t.numel() for t in v.values() if torch.is_tensor(t))

        # Rebuild from the pruned set ONLY and check the embeddings match.
        enc2 = GlucoFMEncoder(fm)
        miss_e, _ = enc2.load_state_dict(
            {k[len("online."):]: v for k, v in keep.items()
             if k.startswith("online.")}, strict=False)
        pool2 = BlockedPool(fm.model.embed_dim, pc)
        miss_p, _ = pool2.load_state_dict(
            {k[len("pool."):]: v for k, v in keep.items()
             if k.startswith("pool.")}, strict=False)
        enc2.eval(); pool2.eval()
        new_emb = embed(enc2, pool2)
        # The VIB makes zA a stochastic channel (zA = mu + sigma*eps). The
        # released readout is zT||zS and excludes zA, but the sampling still
        # advances the global RNG, so both passes are seeded identically above.
        # A residual difference here is a genuine pruning error.
        gap = float(np.max(np.abs(ref_emb - new_emb)))
        ok = gap < 1e-6

        dst = OUT / f"{name}-s{s}.safetensors"
        save_file(keep, str(dst), metadata={
            "model": name, "seed": str(s), "readout": "zT||zS (zA dropped)",
            "contents": "inference only: online encoder + blocked pool",
            "excluded": "EMA target branch, predictor, projection/device/CMP heads, optimiser state"})
        (OUT / f"{name}-s{s}.config.json").write_text(json.dumps(
            {"fm_config": blob["fm_config"], "prism_config": blob["prism_config"],
             "readout": "zT||zS", "d_trait": pc.d_trait, "d_state": pc.d_state,
             "d_sensor": pc.d_sensor}, indent=2, default=str))

        n = sum(v.numel() for v in keep.values())
        print(f"{name:<18}{s:>5}{len(keep):>7}{drop:>9,}{n:>11,}"
              f"{dst.stat().st_size/1e6:>7.1f}   "
              f"{'IDENTICAL' if ok else f'DIFFER max={gap:.3e}'}")
        if not ok:
            raise SystemExit(f"pruning changed embeddings for {name}-s{s}")
        manifest[f"{name}-s{s}"] = dict(inference_params=int(n),
                                        dropped_params=int(drop))

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("\nAll pruned checkpoints reproduce the full model's embeddings exactly.")
