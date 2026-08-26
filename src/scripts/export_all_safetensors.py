"""Export every released model to safetensors: ours, the proposal, the baselines.

`.pt` is a pickle -- loading one executes arbitrary code, which is not an
acceptable requirement to put on anyone downloading a public checkpoint.
safetensors is also zero-copy and framework-agnostic.

Two wrinkles this handles:

  * our checkpoints bundle tensors WITH config dicts and training history;
    safetensors holds a flat tensor map only, so config goes to a sidecar .json
  * they store several tensors twice (top level and under a `full.` prefix)
    sharing storage. torch.save writes such aliases happily; safetensors refuses
    them, because on reload they would silently become two independent copies.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))


import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

SRC = ROOT
OUT = OUTDIR / "weights"
RUNS = SRC / "experiments" / "kaggle_out"
OUT.mkdir(parents=True, exist_ok=True)

# ours + the proposal. Baselines are handled separately: they are third-party
# checkpoints used zero-shot, so we redistribute the HF snapshot as-is rather
# than repackaging weights we did not train.
# Two released models plus our GlucoFM reproduction, which is the baseline every
# number is measured against and therefore has to ship.
#
# The lambda=1.0 configuration is NOT here. It is an ablation of GlucoPRISM-C
# (same blocked pooling, same 64|48|16, same objectives -- only lambda differs),
# not a separate system, and releasing it as one would misrepresent what it is.
# Its evidence lives in the ablation table and in the loss-competition figure.
OURS = {
    "glucoprism-c": ("C-v2-vib01", "glucoprism.pt", (0, 1, 2, 3, 4, 5)),
    "glucoprism-e": ("E-v2-vib-simbias", "glucoprism.pt", (0, 1, 2)),
    "glucofm-ours": ("W3u-ov40", "glucofm.pt", (0, 1, 2)),
}


def flatten(obj, prefix=""):
    tensors, meta = {}, {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if torch.is_tensor(v):
            tensors[key] = v
        elif isinstance(v, dict):
            t2, m2 = flatten(v, f"{key}.")
            tensors.update(t2)
            if m2:
                meta[k] = m2
        else:
            meta[k] = v
    return tensors, meta


def dedup(tensors):
    """One name per storage; keep the shorter (canonical) key."""
    seen, out = {}, {}
    for k in sorted(tensors, key=len):
        v = tensors[k]
        ident = (v.untyped_storage().data_ptr(), tuple(v.shape),
                 tuple(v.stride()), v.storage_offset())
        if ident in seen:
            continue
        seen[ident] = k
        out[k] = v.clone()
    return out


print("=== our models ===")
manifest = {}
for name, (run_prefix, ckname, seeds) in OURS.items():
    got = 0
    for s in seeds:
        # FD-7 runs have no -s0 suffix for seed 0
        cands = [RUNS / f"{run_prefix}-s{s}", RUNS / run_prefix] if s == 0 else \
                [RUNS / f"{run_prefix}-s{s}"]
        ck = next((c / "checkpoints" / ckname for c in cands
                   if (c / "checkpoints" / ckname).exists()), None)
        if ck is None:
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        tensors, meta = flatten(blob)
        n_before = len(tensors)
        tensors = dedup(tensors)
        tensors = {k: v for k, v in tensors.items() if v.dtype != torch.bool}
        dst = OUT / f"{name}-s{s}.safetensors"
        save_file(tensors, str(dst),
                  metadata={"format": "pt", "model": name, "seed": str(s),
                            "source_run": ck.parent.parent.name})
        (OUT / f"{name}-s{s}.config.json").write_text(
            json.dumps(meta, indent=2, default=str))
        got += 1
        n = sum(v.numel() for v in tensors.values())
    if got:
        manifest[name] = dict(seeds=got, params=int(n),
                              aliases_dropped=n_before - len(tensors))
        print(f"  {name:<22}{got} seeds, {n:,} params, "
              f"{n_before - len(tensors)} aliases dropped")
    else:
        print(f"  {name:<22}NO CHECKPOINTS FOUND")

print("\n=== zero-shot baselines ===")
# These already ship as safetensors on the Hub for the most part; the two that
# do not (CGMformer's .bin, and any .bin-only MOMENT mirror) are converted so
# the whole release is uniform and pickle-free.
import os
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
BASE_OUT = OUTDIR / "weights/baselines"
BASE_OUT.mkdir(parents=True, exist_ok=True)

CGMF = SRC / "external" / "cgmformer" / "ckpt" / "cgm_ckp" / "checkpoint-30000"
if (CGMF / "pytorch_model.bin").exists():
    sd = torch.load(CGMF / "pytorch_model.bin", map_location="cpu", weights_only=False)
    sd = dedup({k: v for k, v in sd.items() if torch.is_tensor(v)})
    save_file(sd, str(BASE_OUT / "cgmformer.safetensors"),
              metadata={"source": "YurunLu/CGMformer checkpoint-30000",
                        "note": "sinusoidal position embeddings are COMPUTED, "
                                "not stored -- see eval/run_baselines.py"})
    shutil.copy(CGMF / "config.json", BASE_OUT / "cgmformer.config.json")
    print(f"  cgmformer            {len(sd)} tensors -> safetensors")

from huggingface_hub import snapshot_download            # noqa: E402
HF = {"moment-small": "AutonLab/MOMENT-1-small",
      "moment-large": "AutonLab/MOMENT-1-large",
      "chronos-2": "amazon/chronos-2",
      "chronos-2-small": "autogluon/chronos-2-small"}
LOCAL = {"mantis": SRC / "external" / "Mantis",
         "mantis-v2": SRC / "external" / "MantisV2"}

for name, repo in list(HF.items()) + [(k, v) for k, v in LOCAL.items()]:
    try:
        d = Path(snapshot_download(repo)) if isinstance(repo, str) else repo
        st = next(d.glob("*.safetensors"), None)
        if st:
            shutil.copy(st, BASE_OUT / f"{name}.safetensors")
            cfg = d / "config.json"
            if cfg.exists():
                shutil.copy(cfg, BASE_OUT / f"{name}.config.json")
            print(f"  {name:<20} already safetensors -> copied")
        else:
            bin_ = next(d.glob("*.bin"), None)
            if bin_ is None:
                print(f"  {name:<20} no weights found")
                continue
            sd = dedup({k: v for k, v in
                        torch.load(bin_, map_location="cpu",
                                   weights_only=False).items()
                        if torch.is_tensor(v)})
            save_file(sd, str(BASE_OUT / f"{name}.safetensors"),
                      metadata={"source": str(repo)})
            print(f"  {name:<20} converted from .bin")
    except Exception as e:  # noqa: BLE001
        print(f"  {name:<20} FAILED {type(e).__name__}: {str(e)[:90]}")

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
tot = sum(p.stat().st_size for p in OUTDIR / "weights".rglob("*")
          if p.is_file()) / 1e6
print(f"\nweights total: {tot:.1f} MB")
