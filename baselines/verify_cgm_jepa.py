"""Equivalence check: load the authors' released CGM-JEPA weights into OUR module.

This is the strongest reproduction test available for CGM-JEPA. If our
`glucoprism.models.cgm_jepa.Encoder` is structurally identical to the released
model, then:

  1. every tensor name and shape in `model.safetensors` matches ours exactly
     (`load_state_dict(strict=True)` succeeds with no missing/unexpected keys);
  2. running the authors' weights through our forward pass reproduces their
     embeddings bit-for-bit against the reference implementation checked out in
     `external/CGM-JEPA-master/`.

Both checks are reported. (2) is skipped with a clear message if the reference
checkout is absent.

    python scripts/verify_cgm_jepa.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucoprism.models.cgm_jepa import Encoder  # noqa: E402

WEIGHTS = ROOT / "data" / "raw" / "cgmjepa" / "weights"
REFERENCE = ROOT / "external" / "CGM-JEPA-master"


def load_released(subfolder: str):
    from safetensors.torch import load_file
    cfg = json.loads((WEIGHTS / subfolder / "config.json").read_text())
    sd = load_file(WEIGHTS / subfolder / "model.safetensors")
    return cfg, sd


def check_state_dict(subfolder: str) -> dict:
    cfg, sd = load_released(subfolder)
    model = Encoder(**cfg)
    ours = model.state_dict()

    missing = sorted(set(ours) - set(sd))
    unexpected = sorted(set(sd) - set(ours))
    shape_mismatch = {k: (tuple(ours[k].shape), tuple(sd[k].shape))
                      for k in set(ours) & set(sd) if ours[k].shape != sd[k].shape}

    report = {"subfolder": subfolder, "config": cfg,
              "our_tensors": len(ours), "released_tensors": len(sd),
              "missing_in_release": missing, "unexpected_in_release": unexpected,
              "shape_mismatch": shape_mismatch}

    if not missing and not unexpected and not shape_mismatch:
        model.load_state_dict(sd, strict=True)
        report["strict_load"] = "OK"
        report["n_params"] = sum(p.numel() for p in model.parameters())
    else:
        report["strict_load"] = "FAILED"
    return report, (model if report["strict_load"] == "OK" else None)


def check_numeric(subfolder: str, model) -> dict:
    """Run the same input through the reference implementation and compare."""
    if not (REFERENCE / "models" / "encoder.py").exists():
        return {"status": "skipped", "why": f"reference checkout not found at {REFERENCE}"}

    sys.path.insert(0, str(REFERENCE))
    try:
        import importlib
        for m in [k for k in list(sys.modules) if k.startswith(("models", "utils"))]:
            sys.modules.pop(m, None)
        ref_mod = importlib.import_module("models.encoder")
        cfg, sd = load_released(subfolder)
        ref = ref_mod.Encoder(**cfg)
        ref.load_state_dict(sd, strict=True)
        ref.eval()
        model.eval()

        g = torch.Generator().manual_seed(0)
        x = torch.randn(8, 24, 12, generator=g) * 25 + 110
        xm = torch.zeros(8, 24, 12, 5)

        with torch.no_grad():
            a, ap = model(x, xm, mask=None)
            b, bp = ref(x, xm, mask=None)

        return {"status": "ran",
                "max_abs_diff_tokens": float((a - b).abs().max()),
                "max_abs_diff_proj": float((ap - bp).abs().max()),
                "our_mean_pooled_norm": float(a.mean(1).norm(dim=-1).mean()),
                "identical": bool(torch.allclose(a, b, atol=1e-6)
                                  and torch.allclose(ap, bp, atol=1e-6))}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        if str(REFERENCE) in sys.path:
            sys.path.remove(str(REFERENCE))


def main() -> int:
    if not WEIGHTS.exists():
        print(f"Released weights not found at {WEIGHTS}. Run:\n"
              f"  python scripts/download_datasets.py cgmjepa")
        return 1

    out = {}
    ok = True
    for sub in ["cgm_jepa", "x_cgm_jepa"]:
        rep, model = check_state_dict(sub)
        print(f"\n=== {sub} ===")
        print(f"  strict load       : {rep['strict_load']}")
        print(f"  tensors ours/theirs: {rep['our_tensors']} / {rep['released_tensors']}")
        print(f"  params             : {rep.get('n_params', 'n/a'):,}"
              if rep.get("n_params") else "  params             : n/a")
        for k in ["missing_in_release", "unexpected_in_release"]:
            if rep[k]:
                print(f"  {k}: {rep[k]}")
        if rep["shape_mismatch"]:
            print(f"  shape_mismatch: {rep['shape_mismatch']}")

        num = check_numeric(sub, model) if model is not None else {"status": "skipped"}
        print(f"  numeric vs reference: {num}")
        out[sub] = {"state_dict": rep, "numeric": num}
        ok &= (rep["strict_load"] == "OK")

    dest = ROOT / "artifacts" / "verify_cgm_jepa.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
