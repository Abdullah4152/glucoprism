"""Regression guard for FD-7's patch geometry.

The default configuration MUST stay bit-identical to the published GlucoFM --
strided patchification is new code on the hot path of every model we have, and a
silent change there would invalidate every number in the repo.
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
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(ROOT / "src"))
from cgmkit.models.glucofm import (          # noqa: E402
    GlucoFMConfig, GlucoFM, GlucoFMBackbone, patchify, glucofm_param_report)

def as_tokens(out):
    """The backbone returns a dict; pull the (B, P, D) token tensor out of it."""
    if isinstance(out, dict):
        for k in ("z", "tokens", "fused", "context", "hidden"):
            if k in out and torch.is_tensor(out[k]) and out[k].dim() == 3:
                return out[k]
        for v in out.values():
            if torch.is_tensor(v) and v.dim() == 3:
                return v
        raise KeyError(f"no (B,P,D) tensor in {list(out)}")
    return out[0] if isinstance(out, tuple) else out


torch.manual_seed(0)
B = 4
base = GlucoFMConfig()
x = torch.randn(B, 288) * 40 + 140
m = (torch.rand(B, 288) > 0.15).float()
si = torch.randint(0, 288, (B,))

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


print("1. patchify equals the original reshape at the default stride")
p = patchify(x, base)
check("shape (B,24,12)", tuple(p.shape) == (B, 24, 12), str(tuple(p.shape)))
check("bit-identical to x.view(B,P,K)", torch.equal(p, x.view(B, 24, 12)))

print("\n2. published parameter count unchanged")
rep = glucofm_param_report()
n = rep.get("trainable", rep.get("total"))
check("720,241 trainable", rep.get("trainable") == 720_241, f"got {rep.get('trainable'):,}")

print("\n3. default-config forward is deterministic and finite")
torch.manual_seed(1)
bb = GlucoFMBackbone(base).eval()
with torch.no_grad():
    z1t = as_tokens(bb(x, m, si))
    z2t = as_tokens(bb(x, m, si))
check("finite", bool(torch.isfinite(z1t).all()))
check("deterministic", torch.equal(z1t, z2t))

print("\n4. the FD-7 geometries build and run")
geoms = [("W1  K=12 P=24 stride=12", dict(K=12, P=24, patch_stride=None)),
         ("W4  K=18 P=24 stride=12", dict(K=18, P=24, patch_stride=12)),
         ("W6  K=6  P=48 stride=6 ", dict(K=6, P=48, patch_stride=None)),
         ("W7  K=24 P=12 stride=24", dict(K=24, P=12, patch_stride=None))]
for name, kw in geoms:
    cfg = replace(base, **kw)
    try:
        pt = patchify(x, cfg)
        mm = GlucoFMBackbone(cfg).eval()
        with torch.no_grad():
            out = as_tokens(mm(x, m, si))
        ok = (tuple(pt.shape) == (B, cfg.P, cfg.K)
              and torch.isfinite(out).all()
              and out.shape[1] == cfg.P)
        npar = sum(q.numel() for q in GlucoFM(cfg).parameters() if q.requires_grad)
        check(name, bool(ok), f"patches {tuple(pt.shape)}  tokens {tuple(out.shape)}  {npar:,} params")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"{type(e).__name__}: {e}")

print("\n5. patch 0's lookback is padded AND marked unobserved")
cfg18 = replace(base, K=18, P=24, patch_stride=12)
pm = patchify(m, cfg18)
check("patch 0 leading 6 mask entries are 0", bool((pm[:, 0, :6] == 0).all()))
check("patch 1 leading 6 come from patch 0's tail",
      torch.equal(patchify(x, cfg18)[:, 1, :6], x[:, 6:12]))
check("patch p covers [12p-6, 12p+12)",
      torch.equal(patchify(x, cfg18)[:, 5, :], x[:, 54:72]))

print("\n6. masking hides exactly the sampled fraction under overlap")
for cfg in (base, cfg18):
    pmask = torch.zeros(B, cfg.P, dtype=torch.bool)
    pmask[:, ::2] = True                       # hide half the patches
    hid = (~pmask).float().repeat_interleave(cfg.stride, dim=1)
    check(f"stride={cfg.stride}: zeroed length == L",
          hid.shape[1] == cfg.L, f"got {hid.shape[1]}")
    check(f"stride={cfg.stride}: exactly half hidden",
          abs(float(hid.mean()) - 0.5) < 1e-6, f"visible {float(hid.mean()):.4f}")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
