"""Segmentation and chronological grid alignment shared by all three models.

Implements GlucoFM App. A.1 (segmentation) and App. C.1 (grid alignment):

  * split a subject's raw trace into continuous recording segments -- a timestamp
    gap of at most 1 hour stays inside a segment, anything longer starts a new one;
  * align each 24 h segment to L=288 five-minute grid positions, keeping the
    observation mask M rather than interpolating;
  * the first timestamp fixes the circadian start index s (Eq. 6), so absolute
    time-of-day is preserved.

CGM-JEPA and GluFormer both want a dense 288-vector, so `densify()` provides the
interpolated view those baselines were originally trained on. Which view a model
gets is a per-model decision, not a per-dataset one -- that separation is the
point of GlucoFM's dense-interpolation ablation (paper Fig. 11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GRID_MINUTES = 5
L = 288                      # 24 h / 5 min
SEGMENT_GAP_MINUTES = 60     # App. A.1: gaps > 1 h split the segment


@dataclass
class Window:
    dataset: str
    subject: str
    device: str
    segment: int
    start_time: pd.Timestamp
    start_idx: int           # circadian start index s, Eq. 6
    glucose: np.ndarray      # (L,) float32, NaN where unobserved
    mask: np.ndarray         # (L,) float32, 1 where physically observed

    @property
    def coverage(self) -> float:
        return float(self.mask.mean())


def split_segments(ts: pd.Series, gap_minutes: int = SEGMENT_GAP_MINUTES) -> np.ndarray:
    """Label each reading with a segment id; a gap > `gap_minutes` starts a new one."""
    dt = ts.diff().dt.total_seconds().div(60).fillna(0.0)
    return (dt > gap_minutes).cumsum().to_numpy()


def align_to_grid(times: pd.Series, values: np.ndarray, *, binning: str = "floor"
                  ) -> tuple[np.ndarray, np.ndarray, int, pd.Timestamp]:
    """App. C.1. Returns (glucose[L] with NaN gaps, mask[L], start_idx, start_time).

    `binning` selects the rule B(.) of Eq. 7: "floor" for most cohorts, "nearest"
    where a dataset's timestamp convention makes that a better fit (e.g. Libre
    15-minute records whose clock drifts a couple of minutes).
    Readings outside [0, L-1] are dropped; collisions on one grid slot are averaged.
    """
    t0 = times.iloc[0]
    start_idx = int((60 * t0.hour + t0.minute) // GRID_MINUTES)

    elapsed = (times - t0).dt.total_seconds().to_numpy() / (60.0 * GRID_MINUTES)
    idx = np.floor(elapsed) if binning == "floor" else np.rint(elapsed)
    idx = idx.astype(np.int64)

    keep = (idx >= 0) & (idx <= L - 1) & np.isfinite(values)
    idx, vals = idx[keep], np.asarray(values, dtype=np.float64)[keep]

    total = np.zeros(L, dtype=np.float64)
    count = np.zeros(L, dtype=np.float64)
    np.add.at(total, idx, vals)
    np.add.at(count, idx, 1.0)

    mask = (count > 0).astype(np.float32)
    glucose = np.where(count > 0, total / np.maximum(count, 1.0), np.nan).astype(np.float32)
    return glucose, mask, start_idx, t0


def achievable_coverage(sampling_min: float) -> float:
    """Max fraction of the 288 grid slots a device at this nominal rate can fill.

    A 15-minute Libre reaches 96/288 = 0.333 even with a perfect recording, so a
    coverage threshold has to be expressed relative to this, not absolutely --
    otherwise every Libre window is discarded and the device-shift cohort that
    the whole GlucoPRISM A2 argument rests on vanishes from the corpus.
    """
    return min(1.0, GRID_MINUTES / max(float(sampling_min), GRID_MINUTES))


def iter_windows(df: pd.DataFrame, *, dataset: str, subject: str, device: str,
                 time_col: str = "timestamp", value_col: str = "glucose_mgdl",
                 stride_hours: float = 24.0, min_coverage: float = 0.5,
                 binning: str = "floor", sampling_min: float = 5.0):
    """Yield non-overlapping (or strided) 24 h windows from one subject-device trace.

    `min_coverage` is a fraction of `achievable_coverage(sampling_min)`, so 0.5
    means "at least half the readings this device could have produced".
    """
    min_coverage = min_coverage * achievable_coverage(sampling_min)
    d = df[[time_col, value_col]].dropna().sort_values(time_col).reset_index(drop=True)
    if d.empty:
        return
    seg_ids = split_segments(d[time_col])
    stride = pd.Timedelta(hours=stride_hours)
    day = pd.Timedelta(hours=24)

    for seg in np.unique(seg_ids):
        s = d[seg_ids == seg]
        if len(s) < 2:
            continue
        t_start, t_end = s[time_col].iloc[0], s[time_col].iloc[-1]
        cursor = t_start
        while cursor + day <= t_end + pd.Timedelta(minutes=GRID_MINUTES):
            chunk = s[(s[time_col] >= cursor) & (s[time_col] < cursor + day)]
            if len(chunk) >= 2:
                g, m, si, t0 = align_to_grid(chunk[time_col], chunk[value_col].to_numpy(),
                                             binning=binning)
                if m.mean() >= min_coverage:
                    yield Window(dataset, subject, device, int(seg), t0, si, g, m)
            cursor = cursor + stride


def sample_overlapping_windows(df: pd.DataFrame, *, dataset: str, subject: str, device: str,
                               rng: np.random.Generator, coverage_lo: float = 0.2,
                               coverage_hi: float = 0.8, min_coverage: float = 0.5,
                               time_col: str = "timestamp", value_col: str = "glucose_mgdl",
                               binning: str = "floor", sampling_min: float = 5.0) -> list[Window]:
    """App. A.2 pretraining sampler: per segment, draw overlapping 24 h windows at a
    random coverage ratio in [20%, 80%] so the model sees diverse daily start times.

    Coverage ratio r means the sampled windows jointly cover r x (segment length),
    counting overlaps once -- i.e. n_windows = ceil(r * segment_hours / 24) draws at
    uniformly random offsets, which is how "a coverage ratio between 20% and 80%"
    is realised here (the paper states the range but not the sampler).
    """
    min_coverage = min_coverage * achievable_coverage(sampling_min)
    d = df[[time_col, value_col]].dropna().sort_values(time_col).reset_index(drop=True)
    if d.empty:
        return []
    seg_ids = split_segments(d[time_col])
    day = pd.Timedelta(hours=24)
    out: list[Window] = []

    for seg in np.unique(seg_ids):
        s = d[seg_ids == seg]
        span = (s[time_col].iloc[-1] - s[time_col].iloc[0])
        if span < day:
            continue
        hours = span.total_seconds() / 3600.0
        ratio = rng.uniform(coverage_lo, coverage_hi)
        n = max(1, int(np.ceil(ratio * hours / 24.0)))
        latest = s[time_col].iloc[-1] - day
        span_s = max((latest - s[time_col].iloc[0]).total_seconds(), 0.0)
        for _ in range(n):
            cursor = s[time_col].iloc[0] + pd.Timedelta(seconds=rng.uniform(0, span_s))
            chunk = s[(s[time_col] >= cursor) & (s[time_col] < cursor + day)]
            if len(chunk) < 2:
                continue
            g, m, si, t0 = align_to_grid(chunk[time_col], chunk[value_col].to_numpy(),
                                         binning=binning)
            if m.mean() >= min_coverage:
                out.append(Window(dataset, subject, device, int(seg), t0, si, g, m))
    return out


def densify(glucose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear interpolation + edge fill, for baselines that require a dense input.

    GlucoFM App. B.3 uses exactly this for the CGM-JEPA and GluFormer baselines
    ("aligned to a 5-minute grid, linearly interpolated to 288-point daily
    sequences"); GlucoFM itself never sees it.
    """
    g = np.asarray(glucose, dtype=np.float64).copy()
    obs = mask > 0
    if not obs.any():
        return np.zeros_like(g, dtype=np.float32)
    idx = np.arange(len(g))
    g[~obs] = np.interp(idx[~obs], idx[obs], g[obs])
    return g.astype(np.float32)


def windows_to_arrays(windows: list[Window]) -> dict[str, np.ndarray]:
    """Pack a window list into the column arrays used by the .npz shard format."""
    return {
        "glucose": np.stack([w.glucose for w in windows]).astype(np.float32),
        "mask": np.stack([w.mask for w in windows]).astype(np.float32),
        "start_idx": np.array([w.start_idx for w in windows], dtype=np.int64),
        "subject": np.array([w.subject for w in windows], dtype=object),
        "dataset": np.array([w.dataset for w in windows], dtype=object),
        "device": np.array([w.device for w in windows], dtype=object),
        "segment": np.array([w.segment for w in windows], dtype=np.int64),
        "start_time": np.array([str(w.start_time) for w in windows], dtype=object),
    }
