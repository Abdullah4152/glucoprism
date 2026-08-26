"""FD-9 validation: does the synthetic second sensor now look like a real one?

Runs the SAME measurement pipeline over (a) real CGMacros Dexcom/Libre pairs and
(b) synthetic pairs made by applying the generator to the Dexcom half. If the
generator is faithful, the two distributions overlap. Legacy and calibrated
generators are both scored so the size of the correction is visible.
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


import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = ROOT
sys.path.insert(0, str(ROOT / "src" / "core"))

from cgmkit.data.datasets import WindowShard          # noqa: E402
from cgmkit.data.views import real_pair_index         # noqa: E402
from cgmkit.data.augment import synthetic_libre_view  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
from fd9_sensor_analysis import to_absolute, hf_ratio_common_grid  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "artifacts"


def measure(gr, mr, ga, ma) -> dict | None:
    """Bias / MARD / Deming slope / correlation / HF ratio for one pair."""
    both = (mr > 0) & (ma > 0)
    if both.sum() < 24:
        return None
    x, y = gr[both], ga[both]
    sxx, syy = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    disc = (syy - sxx) ** 2 + 4 * sxy ** 2
    slope = ((syy - sxx) + np.sqrt(disc)) / (2 * sxy) if abs(sxy) > 1e-9 else np.nan
    return dict(bias_mgdl=float(np.mean(y - x)),
                mard_pct=float(np.mean(np.abs(y - x) / np.maximum(x, 1e-6)) * 100),
                slope=float(slope),
                corr=float(np.corrcoef(x, y)[0, 1]),
                hf_ratio=hf_ratio_common_grid(gr, mr, ga, ma))


def main() -> None:
    sh = WindowShard(PROC / "cgmacros_ds.npz")
    d = sh.data
    dev = np.asarray([str(x) for x in d["device"]])
    subj = np.asarray([str(s) for s in d["subject"]])
    pairs = real_pair_index(sh)

    real, synth_new, synth_old = [], [], []
    rng = np.random.default_rng(0)

    for i, j in pairs:
        ri, ai = (i, j) if dev[i] == "dexcom" else (j, i)
        if dev[ri] != "dexcom":
            continue
        gd = d["glucose"][ri].astype(np.float32)
        md = d["mask"][ri].astype(np.float32)

        gr, mr = to_absolute(gd.astype(float), md.astype(float), d["start_idx"][ri])
        ga, ma = to_absolute(d["glucose"][ai].astype(float),
                             d["mask"][ai].astype(float), d["start_idx"][ai])
        r = measure(gr, mr, ga, ma)
        if r:
            r["subject"] = subj[ri]
            real.append(r)

        # Synthetic partners are built from the SAME Dexcom window and compared
        # on that window's own axis, so no alignment step is needed.
        for tag, bucket in (("new", synth_new), ("old", synth_old)):
            gs, ms = synthetic_libre_view(gd, md, rng, legacy=(tag == "old"))
            s = measure(gd.astype(float), md.astype(float),
                        gs.astype(float), ms.astype(float))
            if s:
                s["subject"] = subj[ri]
                bucket.append(s)

    dfs = {"REAL pairs": pd.DataFrame(real),
           "synthetic (FD-9 calibrated)": pd.DataFrame(synth_new),
           "synthetic (legacy constants)": pd.DataFrame(synth_old)}

    cols = ["bias_mgdl", "mard_pct", "slope", "corr", "hf_ratio"]
    print(f"{'':<30}" + "".join(f"{c:>16}" for c in cols))
    print("-" * 110)
    stats = {}
    for name, df in dfs.items():
        per_s = df.groupby("subject")[cols].mean()
        stats[name] = {c: (float(per_s[c].mean()), float(per_s[c].std())) for c in cols}
        print(f"{name:<30}" + "".join(
            f"{per_s[c].mean():>10.2f}+-{per_s[c].std():<4.1f}" for c in cols))

    print("\ngap to real (subject-mean difference, smaller is more faithful):")
    print(f"{'':<30}" + "".join(f"{c:>16}" for c in cols))
    for name in ("synthetic (FD-9 calibrated)", "synthetic (legacy constants)"):
        line = "".join(f"{stats[name][c][0] - stats['REAL pairs'][c][0]:>16.2f}" for c in cols)
        print(f"{name:<30}{line}")

    (OUT / "fd9_generator_validation.json").write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {OUT / 'fd9_generator_validation.json'}")


if __name__ == "__main__":
    main()
