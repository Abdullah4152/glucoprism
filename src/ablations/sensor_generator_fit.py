"""FD-9 step 3: fit the generator's two free parameters to the measured targets.

Bias, gain and MARD are set directly from the paired measurements. Two
quantities are not directly invertible -- how much the partner is smoothed, and
how far it decorrelates -- so they are fitted by grid search against the real
targets. Writes the fitted values back into the calibration JSON.
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


import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = ROOT
sys.path.insert(0, str(ROOT / "src" / "core"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from cgmkit.data.datasets import WindowShard          # noqa: E402
from cgmkit.data.views import real_pair_index         # noqa: E402
from cgmkit.data.augment import synthetic_libre_view, LIBRE_CALIB  # noqa: E402
from fd9_validate_generator import measure                # noqa: E402

OUT = ROOT / "experiments" / "artifacts"
CAL = OUT / "fd9_sensor_calibration.json"

# Targets: subject-weighted means over the real pairs.
TARGET = dict(corr=0.737, hf_ratio=0.576, slope=0.878,
              mard_pct=24.25, bias_mgdl=-31.12)
# corr, hf_ratio and slope are what we are fitting; mard and bias are guards so
# the fit cannot buy one target by making the partner absurd on another.
WEIGHT = dict(corr=1.0, hf_ratio=1.0, slope=1.0, mard_pct=0.25, bias_mgdl=0.05)


def score(sigma: float, lf: float, srcs, gain: float | None = None,
          seed: int = 0) -> tuple[float, dict]:
    rng = np.random.default_rng(seed)
    cal = dict(LIBRE_CALIB, smooth_sigma=sigma, lf_noise_frac=lf)
    if gain is not None:
        cal["gain_mean"] = gain
    rows = []
    for gd, md, subj in srcs:
        gs, ms = synthetic_libre_view(gd, md, rng, calib=cal)
        r = measure(gd.astype(float), md.astype(float),
                    gs.astype(float), ms.astype(float))
        if r:
            r["subject"] = subj
            rows.append(r)
    df = pd.DataFrame(rows)
    per_s = df.groupby("subject")[list(TARGET)].mean()
    got = {k: float(per_s[k].mean()) for k in TARGET}
    # Normalised squared error so quantities on different scales compare.
    err = sum(WEIGHT[k] * ((got[k] - TARGET[k]) / max(abs(TARGET[k]), 1e-6)) ** 2
              for k in TARGET)
    return err, got


def main() -> None:
    sh = WindowShard(ROOT / "data" / "processed" / "cgmacros_ds.npz")
    d = sh.data
    dev = np.asarray([str(x) for x in d["device"]])
    subj = np.asarray([str(s) for s in d["subject"]])

    srcs = []
    for i, j in real_pair_index(sh):
        ri = i if dev[i] == "dexcom" else j
        if dev[ri] != "dexcom":
            continue
        srcs.append((d["glucose"][ri].astype(np.float32),
                     d["mask"][ri].astype(np.float32), subj[ri]))
    print(f"fitting on {len(srcs)} Dexcom source windows\n")

    # Gain and smoothing are coupled -- gain scales amplitude, which moves the
    # HF ratio as well as the slope -- so they must be fitted jointly rather
    # than one at a time.
    sigmas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    lfs = [0.4, 0.6, 0.8, 1.0, 1.2]
    gains = [0.60, 0.70, 0.80, 0.878, 0.95]

    best = (np.inf, None, None, None)
    print(f"{'sigma':>7}{'lf':>6}{'gain':>7}{'err':>9}{'corr':>8}{'hf':>8}"
          f"{'slope':>8}{'mard':>8}{'bias':>9}")
    print("-" * 70)
    for sg, lf, gn in itertools.product(sigmas, lfs, gains):
        err, got = score(sg, lf, srcs, gain=gn)
        if err < best[0]:
            best = (err, sg, lf, gn)
            print(f"{sg:>7.1f}{lf:>6.2f}{gn:>7.3f}{err:>9.4f}{got['corr']:>8.3f}"
                  f"{got['hf_ratio']:>8.3f}{got['slope']:>8.3f}"
                  f"{got['mard_pct']:>8.2f}{got['bias_mgdl']:>9.2f}   <- best so far")

    err, sg, lf, gn = best
    print(f"\nBEST  smooth_sigma={sg}  lf_noise_frac={lf}  gain_mean={gn}  (err {err:.4f})")

    _, got = score(sg, lf, srcs, gain=gn)
    print(f"\n{'quantity':<14}{'real':>10}{'fitted':>10}{'gap':>10}")
    print("-" * 44)
    for k in TARGET:
        print(f"{k:<14}{TARGET[k]:>10.3f}{got[k]:>10.3f}{got[k]-TARGET[k]:>10.3f}")

    cal = json.loads(CAL.read_text())
    cal["fitted_smooth_sigma"] = float(sg)
    cal["fitted_lf_noise_frac"] = float(lf)
    cal["fitted_gain_mean"] = float(gn)
    cal["fit_targets"] = TARGET
    cal["fit_achieved"] = got
    CAL.write_text(json.dumps(cal, indent=2))
    print(f"\nwrote fitted values into {CAL.name}")
    print(f"\n>>> set LIBRE_CALIB smooth_sigma={sg}, lf_noise_frac={lf}, "
          f"gain_mean={gn} in augment.py")


if __name__ == "__main__":
    main()
