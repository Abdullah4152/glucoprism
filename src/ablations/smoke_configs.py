"""Smoke-test every FD-7 / FD-6 configuration for 2 epochs before any Kaggle push.

A config that dies on a GPU costs a wave of wall clock and a slot; a config that
dies here costs 30 seconds. Runs on a tiny subset so it is fast.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))


import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = ROOT
PY = sys.executable

CONFIGS = [
    # FD-7 windowing (all on bb-fm, 1 seed, geometry is the variable)
    ("W1  K12 P24 s12 ov0", ["--shard-suffix", "_ov0"]),
    ("W2  K12 P24 s12 ov20m", ["--shard-suffix", "_ov20m"]),
    ("W3  K12 P24 s12 ov40m", ["--shard-suffix", "_ov40m"]),
    ("W3u K12 P24 s12 ov40", ["--shard-suffix", "_ov40"]),
    ("W4  K18 P24 s12 ov0", ["--shard-suffix", "_ov0", "--patch-k", "18",
                             "--n-patches", "24", "--patch-stride", "12"]),
    ("W5  K18 P24 s12 ov40", ["--shard-suffix", "_ov40", "--patch-k", "18",
                              "--n-patches", "24", "--patch-stride", "12"]),
    ("W6  K6  P48 s6  ov0", ["--shard-suffix", "_ov0", "--patch-k", "6",
                             "--n-patches", "48"]),
    ("W7  K24 P12 s24 ov0", ["--shard-suffix", "_ov0", "--patch-k", "24",
                             "--n-patches", "12"]),
    # FD-6 scaling
    ("FD6 2x width", ["--shard-suffix", "_ov0", "--width-scale", "1.5",
                      "--n-heads", "6"]),
    ("FD6 5x width", ["--shard-suffix", "_ov0", "--width-scale", "2.375",
                      "--n-heads", "8"]),
]

fails = []
for name, extra in CONFIGS:
    with tempfile.TemporaryDirectory() as td:
        cmd = [PY, str(ROOT / "scripts" / "run_pretrain.py"),
               "--model", "glucofm", "--datasets", "bigideas", "stanford",
               "--epochs", "2", "--batch-size", "16", "--seed", "0",
               "--out", td, "--log-every", "1", *extra]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        out = (p.stdout or "") + (p.stderr or "")
        ok = p.returncode == 0
        par = next((ln.strip() for ln in out.splitlines() if "params:" in ln), "")
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<24} {par[:96]}")
        if not ok:
            fails.append(name)
            tail = [ln for ln in out.strip().splitlines() if ln.strip()][-6:]
            for ln in tail:
                print(f"        | {ln[:150]}")

print("\n" + ("ALL CONFIGS OK" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
