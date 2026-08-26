"""GlucoFM's CGM-aware augmentation pipeline (paper Sec. 3.1, App. C.7).

Two families, applied on the fly during pretraining:

  Value perturbations -- change observed glucose, preserve the observation mask
    * baseline wander        p=0.25, amplitude ~ U[5, 15] mg/dL,
                                     frequency ~ U[0.5, 2.0] cycles/window
    * compression-like drop  p=0.10, a contiguous 6-12 step segment attenuated by
                                     a V-shaped curve, min multiplier ~ U[0.4, 0.7]

  Structural sparsification -- change the observation mask itself
    * decimation             p=0.40, only on dense windows (>200 observed positions):
                                     keep every third observation from a random
                                     5-minute offset -> 15-minute-like sampling
    * disconnection blocks   p=0.05, remove 1-3 contiguous blocks of 2-12 grid steps

  "Candidate augmentations are evaluated in random order; after one is applied,
   subsequent probabilities are multiplied by 0.25 to avoid excessive corruption."

The same value-perturbation and sparsification machinery is what GlucoPRISM
reuses to synthesise its V1 second-sensor view (proposal Sec. 4.3), so the
building blocks are exposed individually as well as through `augment()`.
"""

from __future__ import annotations

import numpy as np

DECAY = 0.25            # probability multiplier once an augmentation has fired
DENSE_THRESHOLD = 200   # "dense windows with more than 200 observed positions"


# ------------------------------------------------------- value perturbations

def baseline_wander(g: np.ndarray, m: np.ndarray, rng: np.random.Generator,
                    amp_range=(5.0, 15.0), freq_range=(0.5, 2.0)):
    """Low-frequency sinusoidal drift, standing in for sensor baseline drift."""
    L = len(g)
    amp = rng.uniform(*amp_range)
    freq = rng.uniform(*freq_range)
    phase = rng.uniform(0, 2 * np.pi)
    wave = amp * np.sin(2 * np.pi * freq * np.arange(L) / L + phase)
    out = g.copy()
    obs = m > 0
    out[obs] = out[obs] + wave[obs]
    return out, m


def compression_drop(g: np.ndarray, m: np.ndarray, rng: np.random.Generator,
                     length_range=(6, 12), min_mult_range=(0.4, 0.7)):
    """V-shaped attenuation of a short run -- the classic 'lying on the sensor' artifact."""
    L = len(g)
    n = int(rng.integers(length_range[0], length_range[1] + 1))
    if n >= L:
        return g, m
    start = int(rng.integers(0, L - n))
    lo = rng.uniform(*min_mult_range)

    # V curve: 1 -> lo at the midpoint -> 1
    t = np.linspace(-1.0, 1.0, n)
    mult = lo + (1.0 - lo) * np.abs(t)

    out = g.copy()
    seg = slice(start, start + n)
    obs = m[seg] > 0
    out[seg] = np.where(obs, out[seg] * mult, out[seg])
    return out, m


# --------------------------------------------------- structural sparsification

def decimate(g: np.ndarray, m: np.ndarray, rng: np.random.Generator, factor: int = 3):
    """5-min -> 15-min-like sampling: keep every `factor`-th slot from a random offset."""
    if m.sum() <= DENSE_THRESHOLD:
        return g, m
    offset = int(rng.integers(0, factor))
    keep = np.zeros_like(m)
    keep[offset::factor] = 1.0
    return g, (m * keep).astype(np.float32)


def disconnection_blocks(g: np.ndarray, m: np.ndarray, rng: np.random.Generator,
                         n_range=(1, 3), len_range=(2, 12)):
    """Physical non-wear / sensor-disconnection gaps."""
    L = len(m)
    out = m.copy()
    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        n = int(rng.integers(len_range[0], len_range[1] + 1))
        if n >= L:
            continue
        start = int(rng.integers(0, L - n))
        out[start:start + n] = 0.0
    return g, out.astype(np.float32)


# ---------------------------------------------------------------- the pipeline

_AUGMENTATIONS = [
    ("baseline_wander", baseline_wander, 0.25),
    ("compression_drop", compression_drop, 0.10),
    ("decimate", decimate, 0.40),
    ("disconnection_blocks", disconnection_blocks, 0.05),
]


def augment(g: np.ndarray, m: np.ndarray, rng: np.random.Generator,
            enable: set[str] | None = None, return_applied: bool = False):
    """Apply the four candidates in random order with the decayed-probability rule.

    `enable` restricts the candidate set, which is how the paper's augmentation
    ablation (Fig. 10: value-only vs structural-only vs full) is reproduced.
    """
    g, m = np.asarray(g, np.float32).copy(), np.asarray(m, np.float32).copy()
    order = rng.permutation(len(_AUGMENTATIONS))
    decay = 1.0
    applied: list[str] = []

    for i in order:
        name, fn, p = _AUGMENTATIONS[i]
        if enable is not None and name not in enable:
            continue
        if rng.random() < p * decay:
            g, m = fn(g, m, rng)
            applied.append(name)
            decay *= DECAY

    if return_applied:
        return g, m, applied
    return g, m


VALUE_AUGS = {"baseline_wander", "compression_drop"}
STRUCTURAL_AUGS = {"decimate", "disconnection_blocks"}


# ------------------------------------------- GlucoPRISM synthetic sensor view

# FD-9: measured on CGMacros' 374 usable real same-day Dexcom/Libre pairs
# (44 subjects), after aligning both windows to absolute time via `start_idx`.
# See experiments/artifacts/fd9_sensor_calibration.json.
#
# The values these replaced were guesses, and two of them were badly wrong:
# `calib_sd=6.0` with an implicit zero mean, where the real device difference is
# a systematic -31 mg/dL that 43 of 44 subjects show; and no gain term at all,
# where the real Deming slope is 0.878. Level is what every downstream label is
# defined on, so a synthetic "second sensor" that differed from its partner only
# by symmetric noise was not exercising the sensor invariance it was built for.
LIBRE_CALIB = dict(
    bias_mean=-31.12, bias_sd=15.77,      # mg/dL, subject-weighted; measured
    gain_sd=0.217,                        # Deming slope spread; measured
    # gain_mean, smooth_sigma and lf_noise_frac are FITTED jointly by
    # fd9_fit_generator.py rather than read off the measurements, because the
    # three are coupled: gain scales amplitude, which moves the HF ratio as well
    # as the slope, and the low-pass noise inflates the Deming slope. Fitting one
    # at a time lands none of them.
    #
    # Achieved vs real: corr 0.707/0.737, HF 0.584/0.576, slope 0.912/0.878,
    # MARD 25.4/24.3, bias -31.2/-31.1.
    gain_mean=0.70,
    smooth_sigma=1.5,
    lf_noise_frac=1.0,
    lf_noise_sigma=12.0,                  # 1 h; slow divergence, not white noise
    dropout_extra_median=3,               # grid steps beyond nominal 15-min spacing
)


def synthetic_libre_view(g: np.ndarray, m: np.ndarray, rng: np.random.Generator,
                         smooth_sigma: float | None = None, calib_sd: float | None = None,
                         warmup_steps: int = 0, calib: dict | None = None,
                         legacy: bool = False):
    """Proposal Sec. 4.3 (V1): render a Dexcom-like 5-min window as a Libre-like view.

    Extends GlucoFM's structural sparsification with the device effects the
    proposal names: 5->15-min decimation at a random phase, Libre-like smoothing,
    and a calibration difference. Compression drops come from `compression_drop`.

    The calibration difference is now *measured* (FD-9) rather than assumed: a
    per-window gain and offset drawn from the distributions observed across real
    paired sensors, applied about the window's own mean so that both the measured
    bias and the measured slope are reproduced.

    `legacy=True` restores the pre-FD-9 constants for before/after comparison.

    Warm-up truncation defaults to OFF. Windows are cut starting at a reading, so
    index 0 is observed by construction and warm-up is not observable in the
    paired data -- zeroing a fixed prefix was an unmeasured assumption that
    removed real signal.
    """
    g = np.asarray(g, np.float32).copy()
    m = np.asarray(m, np.float32).copy()

    if legacy:
        c = dict(bias_mean=0.0, bias_sd=6.0, gain_mean=1.0, gain_sd=0.0,
                 smooth_sigma=1.5)
        warmup_steps = warmup_steps or 12
    else:
        c = dict(calib or LIBRE_CALIB)
    if smooth_sigma is not None:
        c["smooth_sigma"] = smooth_sigma
    if calib_sd is not None:
        c["bias_sd"] = calib_sd

    # Libre-like smoothing over observed positions only.
    if c["smooth_sigma"] > 0:
        s = float(c["smooth_sigma"])
        r = int(np.ceil(3 * s))
        lags = np.arange(-r, r + 1)
        k = np.exp(-lags ** 2 / (2 * s ** 2))
        k /= k.sum()
        gm = np.nan_to_num(g) * m
        num = np.convolve(gm, k, mode="same")
        den = np.convolve(m, k, mode="same")
        g = np.where((m > 0) & (den > 1e-6), num / np.maximum(den, 1e-6), g).astype(np.float32)

    # Gain + offset about the window's own observed mean, so the transform
    # reproduces the measured mean bias AND the measured slope simultaneously.
    gain = float(rng.normal(c["gain_mean"], c["gain_sd"]))
    bias = float(rng.normal(c["bias_mean"], c["bias_sd"]))
    obs = m > 0
    centre = float(np.nan_to_num(g)[obs].mean()) if obs.any() else 0.0
    g = np.where(obs, gain * (g - centre) + centre + bias, g).astype(np.float32)

    # Slow independent divergence. Two sensors on the same person correlate at
    # 0.74, not 0.97: they sit at different body sites and drift apart over
    # hours. A purely deterministic transform of the source cannot reproduce
    # that, and a partner that tracks its source at 0.97 makes the sensor
    # invariance loss trivially satisfiable. Low-pass noise decorrelates on the
    # right timescale without adding high-frequency raggedness real Libre
    # does not have.
    lf = float(c.get("lf_noise_frac", 0.0))
    if lf > 0 and obs.any():
        s = float(c.get("lf_noise_sigma", 12.0))
        r = int(np.ceil(3 * s))
        lags = np.arange(-r, r + 1)
        k = np.exp(-lags ** 2 / (2 * s ** 2))
        k /= np.sqrt((k ** 2).sum())          # unit-energy: preserves noise sd
        z = np.convolve(rng.standard_normal(len(g)), k, mode="same")
        scale = lf * float(np.nan_to_num(g)[obs].std())
        g = (g + (z * scale).astype(np.float32)).astype(np.float32)

    g, m = compression_drop(g, m, rng)
    g, m = decimate(g, m, rng)                           # 5 -> 15 min, random phase
    if warmup_steps > 0:
        m[:warmup_steps] = 0.0
    return g, m.astype(np.float32)
