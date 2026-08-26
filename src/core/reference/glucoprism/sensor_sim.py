"""
Calibrated sensor simulator for the GlucoPRISM V1 (paired-sensor) view.

Maps a Dexcom-like 5-minute window to a Libre-like one, so that synthetic V1
pairs (x, x') reproduce the device gap actually measured in CGMacros rather
than a hand-tuned guess.

Measured target (scripts/analyze_v1_pairs.py, 45 subjects, 15-min native grid):
    Libre - Dexcom bias   -32.4 mg/dL   (sd 16.2 across subjects)
    MARD                   25.5%
    lag                    -5 min (36/45 subjects in -5..-10 min)
    HF energy ratio L/D    0.65        (Libre is smoother)

Note that MARD ~25% is almost entirely *explained by* the bias: |-32| / ~120
mg/dL ~ 27%. So the transform is dominated by a level offset, not by added
noise - which is exactly what GlucoFM's structural sparsification lacks.

The model is deliberately simple and physically interpretable:

    x'(t) = smooth_sigma( gain * x(t - lag) + bias ) + noise
    then decimate 5 -> 15 min (mask only; values are never interpolated)

Every parameter is sampled per window so the synthetic view reproduces the
subject-to-subject spread, not just the mean.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class SensorSimParams:
    """Calibrated Dexcom -> Libre transform parameters."""
    bias_mean: float = -32.4      # mg/dL
    bias_sd: float = 16.2
    gain_mean: float = 1.0
    gain_sd: float = 0.0
    lag_steps_mean: float = -1.0  # in 5-min grid steps (-1 = -5 min)
    lag_steps_sd: float = 1.0
    smooth_sigma: float = 1.5     # grid steps, Gaussian low-pass
    noise_sd: float = 0.0         # residual mg/dL after bias/gain/lag/smooth
    decimate_factor: int = 3      # 5 min -> 15 min
    decimate_prob: float = 1.0
    hf_ratio_target: float = 0.65

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "SensorSimParams":
        return SensorSimParams(**json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
def _gauss_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    if sigma <= 1e-6:
        return np.array([1.0])
    radius = radius or max(1, int(np.ceil(3 * sigma)))
    r = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(r ** 2) / (2 * sigma * sigma))
    return k / k.sum()


def masked_smooth(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Symmetric Gaussian low-pass that ignores unobserved positions."""
    if sigma <= 1e-6:
        return x.copy()
    k = _gauss_kernel(sigma)
    num = np.convolve(x * m, k, mode="same")
    den = np.convolve(m, k, mode="same")
    out = np.where(den > 1e-8, num / np.maximum(den, 1e-8), x)
    return out


def hf_energy(x: np.ndarray, m: np.ndarray) -> float:
    """Std of first differences over positions where both ends are observed."""
    ok = (m[1:] > 0) & (m[:-1] > 0)
    if ok.sum() < 8:
        return float("nan")
    return float(np.std(np.diff(x)[ok]))


# --------------------------------------------------------------------------- #
class SensorSim:
    """Apply a calibrated Dexcom -> Libre transform to a 288-point window."""

    def __init__(self, params: SensorSimParams | None = None):
        self.p = params or SensorSimParams()

    def __call__(self, x: np.ndarray, m: np.ndarray,
                 rng: np.random.Generator | None = None,
                 decimate: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng()
        p = self.p
        x = np.asarray(x, dtype=np.float64).copy()
        m = np.asarray(m, dtype=np.float64).copy()

        # 1. temporal lag (integer grid steps; mask travels with the signal)
        lag = int(round(rng.normal(p.lag_steps_mean, p.lag_steps_sd)))
        if lag:
            x = np.roll(x, lag)
            m = np.roll(m, lag)

        # 2. gain + level offset - the dominant component of the real gap.
        #
        # Parameterised about the window mean:
        #     x' = xbar + bias + gain * (x - xbar)
        # so `bias` is exactly the mean level shift and `gain` is pure amplitude
        # scaling. Fitting a raw `gain*x + intercept` instead makes the two
        # strongly anti-correlated across subjects, and sampling them
        # independently then double-counts the offset.
        gain = rng.normal(p.gain_mean, p.gain_sd) if p.gain_sd > 0 else p.gain_mean
        bias = rng.normal(p.bias_mean, p.bias_sd)
        obs = m > 0
        xbar = float(x[obs].mean()) if obs.any() else float(x.mean())
        x = xbar + bias + gain * (x - xbar)

        # 3. low-pass to match the measured HF energy ratio
        if p.smooth_sigma > 0:
            x = masked_smooth(x, m, p.smooth_sigma)

        # 4. residual noise
        if p.noise_sd > 0:
            x = x + rng.normal(0.0, p.noise_sd, size=x.shape)

        # 5. structural: decimate 5 -> 15 min by dropping observations.
        #    Values are never interpolated - only the mask changes.
        do_dec = p.decimate_prob > 0 if decimate is None else decimate
        if do_dec and (decimate is not None or rng.random() < p.decimate_prob):
            off = int(rng.integers(0, p.decimate_factor))
            keep = np.zeros_like(m)
            keep[off::p.decimate_factor] = 1.0
            m = m * keep

        x = np.clip(x, 20.0, 600.0) * m
        return x.astype(np.float32), m.astype(np.float32)


# --------------------------------------------------------------------------- #
def fit_params(pairs: list[dict], verbose: bool = True) -> SensorSimParams:
    """Estimate transform parameters from real paired windows.

    `pairs` is a list of {'d': dexcom[288], 'l': libre[288], 'md':, 'ml':}
    on the common 5-minute grid.
    """
    biases, gains, lags, hf_d, hf_l, resid = [], [], [], [], [], []

    for pr in pairs:
        d, l = np.asarray(pr["d"], float), np.asarray(pr["l"], float)
        md, ml = np.asarray(pr["md"], float), np.asarray(pr["ml"], float)
        both = (md > 0) & (ml > 0)
        if both.sum() < 40:
            continue

        # lag: maximise correlation over +/- 6 steps (+/- 30 min)
        best_r, best_k = -2.0, 0
        a = d.copy(); b = l.copy()
        for k in range(-6, 7):
            bb = np.roll(b, -k)
            mm = both & (np.roll(both, -k))
            if mm.sum() < 40:
                continue
            r = np.corrcoef(a[mm], bb[mm])[0, 1]
            if np.isfinite(r) and r > best_r:
                best_r, best_k = float(r), k
        lags.append(best_k)

        bb = np.roll(b, -best_k)
        mm = both & (np.roll(both, -best_k))
        dv, lv = a[mm], bb[mm]
        if len(dv) < 40:
            continue

        # Centred parameterisation, matching SensorSim.__call__:
        #     l ~ mean(d) + bias + gain * (d - mean(d))
        # bias is the mean level shift; gain is the slope on centred values.
        dbar = float(dv.mean())
        bias = float(lv.mean() - dbar)
        dc = dv - dbar
        g = float(np.dot(dc, lv - lv.mean()) / max(np.dot(dc, dc), 1e-9))
        gains.append(g)
        biases.append(bias)
        resid.append(float(np.std(lv - (dbar + bias + g * dc))))

        hf_d.append(hf_energy(d, md))
        hf_l.append(hf_energy(l, ml))

    hf_d = np.asarray(hf_d, float); hf_l = np.asarray(hf_l, float)
    ok = np.isfinite(hf_d) & np.isfinite(hf_l) & (hf_d > 1e-6)
    ratio = float(np.median(hf_l[ok] / hf_d[ok])) if ok.any() else 0.65

    # choose smoothing sigma so that a Gaussian low-pass reproduces `ratio`
    sigma = _solve_sigma_for_ratio(ratio)

    p = SensorSimParams(
        bias_mean=float(np.mean(biases)),
        bias_sd=float(np.std(biases)),
        gain_mean=float(np.median(gains)),
        gain_sd=float(np.std(gains)),
        lag_steps_mean=float(np.mean(lags)),
        lag_steps_sd=float(np.std(lags)),
        smooth_sigma=float(sigma),
        noise_sd=float(np.median(resid)) if resid else 0.0,
        hf_ratio_target=ratio,
    )
    if verbose:
        print(f"    fitted on {len(biases)} paired windows")
        print(f"      bias   = {p.bias_mean:+.2f} +/- {p.bias_sd:.2f} mg/dL")
        print(f"      gain   = {p.gain_mean:.3f} +/- {p.gain_sd:.3f}")
        print(f"      lag    = {p.lag_steps_mean:+.2f} +/- {p.lag_steps_sd:.2f} steps "
              f"({p.lag_steps_mean*5:+.1f} min)")
        print(f"      HF ratio target = {ratio:.3f}  -> smooth_sigma = {sigma:.2f}")
        print(f"      residual noise  = {p.noise_sd:.2f} mg/dL")
    return p


def _solve_sigma_for_ratio(target: float, n: int = 288, trials: int = 60) -> float:
    """Find the Gaussian sigma whose low-pass gives the target HF energy ratio."""
    if not np.isfinite(target) or target >= 0.999:
        return 0.0
    rng = np.random.default_rng(0)
    # a realistic-ish CGM-like signal: smooth trend + fluctuation
    base = np.cumsum(rng.normal(0, 3, size=n))
    base = base - base.mean() + 120
    m = np.ones(n)
    h0 = hf_energy(base, m)
    lo, hi = 0.0, 12.0
    for _ in range(trials):
        mid = 0.5 * (lo + hi)
        h = hf_energy(masked_smooth(base, m, mid), m)
        if h / h0 > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
