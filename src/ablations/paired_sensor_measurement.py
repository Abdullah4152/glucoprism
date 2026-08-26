"""FD-9 - what actually differs between two CGM sensors on the same person, same day.

Measures the six quantities the synthetic second-sensor generator currently
hard-codes, using CGMacros' real same-day Dexcom/Libre window pairs. Writes a
calibration table that `augment.synthetic_libre_view` reads instead of constants.

No GPU. Output:
  experiments/artifacts/fd9_sensor_calibration.json   <- the table the generator uses
  experiments/artifacts/fd9_pair_measurements.csv     <- per-pair raw measurements
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
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(ROOT / "src" / "core"))

from cgmkit.data.datasets import WindowShard          # noqa: E402
from cgmkit.data.views import real_pair_index         # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = OUTDIR
OUT.mkdir(parents=True, exist_ok=True)


PAD = 288          # absolute axis is 3 x 288 so a +-288 offset never wraps


def to_absolute(g: np.ndarray, m: np.ndarray, start_idx: int):
    """Place a window on a shared absolute-time axis.

    Array index 0 is the window's OWN first reading, not midnight -- a Dexcom
    window can start at 11:39 and its same-day Libre partner at 10:30. Comparing
    the raw arrays element-wise compares 11:39 against 10:30. `start_idx` is the
    circadian start (floor of minutes-since-midnight / 5), so shifting each array
    by it puts both on one axis.
    """
    G = np.zeros(3 * PAD, np.float64)
    M = np.zeros(3 * PAD, np.float64)
    s = PAD + int(start_idx)
    G[s:s + len(g)] = g
    M[s:s + len(m)] = m
    return G, M


def hf_ratio_common_grid(gr, mr, ga, ma) -> float:
    """High-frequency energy of the sparse device relative to the dense one,
    measured on positions BOTH observe.

    Measuring each device on its own observed positions is not comparable: a
    15-minute sensor's consecutive readings are 15 minutes apart, so its first
    differences span 3x the interval and look larger for reasons that have
    nothing to do with smoothing. Restricting both to the shared positions makes
    the spacing identical, so a remaining difference is real filtering.
    """
    both = np.flatnonzero((mr > 0) & (ma > 0))
    if both.size < 12:
        return np.nan
    step = np.diff(both)
    keep = np.flatnonzero(step == step.min())      # evenly spaced runs only
    if keep.size < 8:
        return np.nan
    i0, i1 = both[keep], both[keep + 1]
    dr = np.var(gr[i1] - gr[i0])
    da = np.var(ga[i1] - ga[i0])
    return float(da / max(dr, 1e-9))


def lag_by_xcorr(a, ma, b, mb, max_lag: int = 36) -> float:
    """Residual offset after absolute alignment, in 5-minute steps.

    Searched over +-3 hours. If the estimate pegs at the boundary the pair is
    reported as unmeasurable rather than silently returning the bound.
    """
    best_lag, best_r = np.nan, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        bs, ms = np.roll(b, lag), np.roll(mb, lag)
        both = (ma > 0) & (ms > 0)
        if both.sum() < 24:
            continue
        x, y = a[both], bs[both]
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best_r:
            best_r, best_lag = r, lag
    return np.nan if abs(best_lag) >= max_lag else best_lag


def dropout_runs(m: np.ndarray, rate_min: float) -> list[int]:
    """Lengths of missing stretches, in grid steps, beyond what the device rate
    already explains.

    A 15-minute sensor leaves 2 of every 3 slots empty by construction. Only
    gaps LONGER than that nominal spacing are real dropouts.
    """
    nominal = max(1, int(round(rate_min / 5.0)))
    idx = np.flatnonzero(m > 0)
    if idx.size < 2:
        return []
    gaps = np.diff(idx) - nominal
    return [int(g) for g in gaps if g > 0]


def main() -> None:
    shard = WindowShard(PROC / "cgmacros_ds.npz")
    d = shard.data
    pairs = real_pair_index(shard)
    dev = np.asarray([str(x) for x in d["device"]])
    subj = np.asarray([str(s) for s in d["subject"]])
    print(f"CGMacros: {len(shard)} windows, {len(np.unique(subj))} subjects, "
          f"devices={sorted(set(dev))}")
    print(f"real same-day paired windows: {len(pairs)}")

    rate = {}
    for dv in set(dev):
        occ = d["mask"][dev == dv].mean()
        rate[dv] = 5.0 / max(occ, 1e-6)          # implied minutes between reads
    print("implied sampling interval:",
          {k: f"{v:.1f} min" for k, v in rate.items()})

    # Orient every pair the same way: reference = the denser device.
    ref_dev = min(rate, key=rate.get)
    alt_dev = max(rate, key=rate.get)
    print(f"reference (dense) = {ref_dev}   alt (sparse) = {alt_dev}")

    rows = []
    for i, j in pairs:
        if dev[i] == ref_dev:
            ri, ai = i, j
        elif dev[j] == ref_dev:
            ri, ai = j, i
        else:
            continue
        gr, mr = to_absolute(d["glucose"][ri].astype(float),
                             d["mask"][ri].astype(float), d["start_idx"][ri])
        ga, ma = to_absolute(d["glucose"][ai].astype(float),
                             d["mask"][ai].astype(float), d["start_idx"][ai])

        both = (mr > 0) & (ma > 0)
        if both.sum() < 24:
            continue
        x, y = gr[both], ga[both]

        # Additive vs proportional error. OLS regresses y on x assuming x is
        # noise-free; here BOTH are noisy sensors, and that biases the OLS slope
        # toward zero (regression dilution). Deming regression -- the standard
        # for method-comparison studies -- treats the two symmetrically. Both are
        # recorded so the size of the artifact is visible.
        A = np.vstack([x, np.ones_like(x)]).T
        slope_ols, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        sxx, syy = np.var(x, ddof=1), np.var(y, ddof=1)
        sxy = np.cov(x, y, ddof=1)[0, 1]
        lam = 1.0                                     # equal error variance
        disc = (syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2
        slope = ((syy - lam * sxx) + np.sqrt(disc)) / (2 * sxy) if abs(sxy) > 1e-9 else np.nan

        rows.append(dict(
            subject=subj[ri],
            n_overlap=int(both.sum()),
            start_offset=int(d["start_idx"][ai]) - int(d["start_idx"][ri]),
            bias_mgdl=float(np.mean(y - x)),
            mard_pct=float(np.mean(np.abs(y - x) / np.maximum(x, 1e-6)) * 100),
            slope=float(slope),
            slope_ols=float(slope_ols),
            intercept=float(intercept),
            corr=float(np.corrcoef(x, y)[0, 1]),
            lag_steps=lag_by_xcorr(gr, mr, ga, ma),
            hf_ratio=hf_ratio_common_grid(gr, mr, ga, ma),
            occ_ref=float(d["mask"][ri].mean()),
            occ_alt=float(d["mask"][ai].mean()),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fd9_pair_measurements.csv", index=False)
    print(f"\nusable pairs after overlap filter: {len(df)} "
          f"({df.subject.nunique()} subjects)\n")

    hf_ratio = df["hf_ratio"].replace([np.inf, -np.inf], np.nan)

    def q(s, name):
        s = pd.Series(s).dropna()
        return dict(name=name, n=int(s.size), mean=float(s.mean()),
                    sd=float(s.std()), p10=float(s.quantile(.10)),
                    p50=float(s.quantile(.50)), p90=float(s.quantile(.90)))

    summary = [
        q(df["bias_mgdl"], "calibration bias (mg/dL)"),
        q(df["mard_pct"], "mean abs rel difference (%)"),
        q(df["slope"], "slope, Deming (alt vs ref)"),
        q(df["slope_ols"], "slope, OLS (dilution-biased)"),
        q(df["corr"], "correlation"),
        q(df["lag_steps"], "residual lag (5-min steps)"),
        q(hf_ratio, "HF energy ratio (alt/ref)"),
        q(df["start_offset"], "window start offset (steps)"),
        q(df["occ_ref"], "occupancy ref"),
        q(df["occ_alt"], "occupancy alt"),
    ]
    print(f"{'quantity':<36}{'n':>5}{'mean':>10}{'sd':>9}{'p10':>9}{'p50':>9}{'p90':>9}")
    print("-" * 87)
    for s in summary:
        print(f"{s['name']:<36}{s['n']:>5}{s['mean']:>10.3f}{s['sd']:>9.3f}"
              f"{s['p10']:>9.3f}{s['p50']:>9.3f}{s['p90']:>9.3f}")

    # Dropout run-lengths, pooled per device.
    runs = {}
    for dv in (ref_dev, alt_dev):
        sel = np.flatnonzero(dev == dv)
        pooled: list[int] = []
        for k in sel:
            pooled += dropout_runs(d["mask"][k].astype(float), rate[dv])
        runs[dv] = pooled
        if pooled:
            a = np.array(pooled)
            print(f"\ndropout gaps beyond nominal spacing, {dv}: "
                  f"n={a.size}, median={np.median(a):.0f} steps, "
                  f"p90={np.quantile(a, .9):.0f}, max={a.max()}")
        else:
            print(f"\ndropout gaps beyond nominal spacing, {dv}: none")

    # Per-subject first, so prolific subjects do not dominate. 44 subjects
    # contribute 374 windows; an unweighted window mean would let the subject
    # with the most paired days set the calibration constants.
    per_s = df.groupby("subject")[["bias_mgdl", "mard_pct", "slope",
                                   "corr", "hf_ratio"]].mean()
    print(f"\nper-subject ({len(per_s)} subjects, window-weighted vs subject-weighted):")
    for c in per_s.columns:
        print(f"  {c:<14} window-mean {df[c].mean():>8.3f}   "
              f"subject-mean {per_s[c].mean():>8.3f}   "
              f"subject-sd {per_s[c].std():>7.3f}")
    n_lower = int((per_s["bias_mgdl"] < 0).sum())
    print(f"  subjects where Libre reads lower: {n_lower}/{len(per_s)}")

    # smooth_sigma implied by the measured HF ratio.  For a Gaussian kernel of
    # width s applied to a random walk, the variance of the first difference is
    # attenuated by roughly 1/(1+2s^2); invert that for the median ratio.
    r = float(pd.Series(hf_ratio).dropna().median())
    implied_sigma = float(np.sqrt(max((1.0 / max(r, 1e-6) - 1.0) / 2.0, 0.0)))

    calib = dict(
        source="CGMacros real same-day paired windows",
        n_pairs=int(len(df)), n_subjects=int(df["subject"].nunique()),
        reference_device=str(ref_dev), alt_device=str(alt_dev),
        sampling_min={str(k): round(float(v), 2) for k, v in rate.items()},
        # Subject-weighted -- these are the numbers the generator samples from.
        bias_mgdl_mean=float(per_s["bias_mgdl"].mean()),
        bias_mgdl_sd=float(per_s["bias_mgdl"].std()),
        slope_mean=float(per_s["slope"].mean()),
        slope_sd=float(per_s["slope"].std()),
        slope_ols_mean=float(df["slope_ols"].mean()),
        mard_pct_median=float(per_s["mard_pct"].median()),
        corr_median=float(per_s["corr"].median()),
        lag_steps_median=float(df["lag_steps"].dropna().median()),
        hf_ratio_median=r,
        implied_smooth_sigma=implied_sigma,
        dropout_runs={str(k): sorted(v)[:2000] for k, v in runs.items()},
        summary=summary,
        notes=[
            "Windows are cut starting at a reading, so index 0 is observed by "
            "construction and sensor warm-up is NOT observable in windowed data. "
            "warmup_steps is left at its current value rather than fabricated.",
        ],
    )
    (OUT / "fd9_sensor_calibration.json").write_text(json.dumps(calib, indent=2))

    print("\n" + "=" * 87)
    print("CURRENT HARD-CODED CONSTANTS vs MEASURED")
    print(f"  smooth_sigma   1.50  ->  {implied_sigma:.2f}")
    print(f"  calib_sd       6.00  ->  {per_s['bias_mgdl'].std():.2f}   "
          f"(plus a mean bias of {per_s['bias_mgdl'].mean():+.2f} mg/dL the generator omits)")
    print(f"  slope          1.00 (implicit)  ->  {per_s['slope'].mean():.3f} (Deming)")
    print(f"  warmup_steps  12     ->  not observable in windowed data (see notes)")
    print(f"\nwrote {OUT / 'fd9_sensor_calibration.json'}")


if __name__ == "__main__":
    main()
