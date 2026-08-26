"""Glucodensity cross-view for X-CGM-JEPA (CGM-JEPA paper M.5.2).

One day-window -> a 32x32x3 image-like distributional summary -> 16 ViT-style
tokens of 8x8x3 = 192 dims.

Procedure, verbatim from the paper and matched to the reference
`utils/glucodensity_utils.py`:

  1. map indices to hours, t = 5i/60;
  2. fit `scipy.interpolate.UnivariateSpline` with smoothing factor 1.0;
  3. take level G, speed dG/dt, acceleration d2G/dt2 from the fitted curve;
  4. 2-D Gaussian KDE (Scott bandwidth) for (G, G'), (G, G''), (G', G'');
  5. evaluate each on a 32x32 grid spanning the empirical 1st-99th percentile
     range of that channel pair -- the trim keeps spline-derivative spikes from
     collapsing the grid;
  6. normalise each grid by its own maximum so values lie in [0, 1];
  7. stack as channels, tile into non-overlapping 8x8 spatial patches.

The KDE dominates wall time and the view is a deterministic function of the
window, so `precompute()` caches it once per corpus (paper M.5.2,
"Pre-computation and caching").
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import interpolate
from scipy.stats import gaussian_kde

GRID_SIZE = 32
SPATIAL_PATCH = 8
SMOOTHING_FACTOR = 1.0


def _kde_grid(a: np.ndarray, b: np.ndarray, gridsize: int = GRID_SIZE) -> np.ndarray:
    """2-D Gaussian KDE of (a, b) on a `gridsize` square, peak-normalised to [0, 1].

    Degenerate windows are real in a large corpus: a near-flat trace (a clamped
    sensor, a long constant stretch in a T1D record) makes the (level, speed,
    acceleration) cloud collapse onto a point or a line, and `gaussian_kde` then
    raises LinAlgError on a singular covariance. The reference corpus never
    contained such a window, so the reference implementation does not guard for
    it; REPLACE-BG does. Fall back to an all-zero grid, which is exactly the
    value a masked spatial patch already takes in the cross-view objective.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_lo, a_hi = np.percentile(a, [1, 99])
    b_lo, b_hi = np.percentile(b, [1, 99])

    # A collapsed axis leaves nothing for the KDE (and no grid) to resolve.
    if not np.isfinite([a_lo, a_hi, b_lo, b_hi]).all() or a_hi <= a_lo or b_hi <= b_lo:
        return np.zeros((gridsize, gridsize), dtype=np.float64)

    try:
        kde = gaussian_kde(np.vstack([a, b]), bw_method="scott")
    except np.linalg.LinAlgError:
        return np.zeros((gridsize, gridsize), dtype=np.float64)

    Ag, Bg = np.meshgrid(np.linspace(a_lo, a_hi, gridsize),
                         np.linspace(b_lo, b_hi, gridsize))
    try:
        Z = kde(np.vstack([Ag.ravel(), Bg.ravel()])).reshape(Ag.shape)
    except np.linalg.LinAlgError:
        return np.zeros((gridsize, gridsize), dtype=np.float64)
    if not np.isfinite(Z).all():
        return np.zeros((gridsize, gridsize), dtype=np.float64)
    return Z / (Z.max() + 1e-12)


def glucodensity_image(cgm: np.ndarray, gridsize: int = GRID_SIZE,
                       smoothing_factor: float = SMOOTHING_FACTOR) -> np.ndarray:
    """(L,) day-window -> (gridsize, gridsize, 3) normalised KDE stack."""
    x = np.asarray(cgm, dtype=np.float64).ravel()
    t = np.arange(len(x)) * 5.0 / 60.0                     # hours

    spline = interpolate.UnivariateSpline(t, x, s=smoothing_factor)
    G = spline(t)
    Gd = spline.derivative(1)(t)
    Gdd = spline.derivative(2)(t)

    return np.stack([
        _kde_grid(G, Gd, gridsize),      # level x speed
        _kde_grid(G, Gdd, gridsize),     # level x acceleration
        _kde_grid(Gd, Gdd, gridsize),    # speed x acceleration
    ], axis=-1)


def to_spatial_patches(image: np.ndarray, patch_size: int = SPATIAL_PATCH) -> np.ndarray:
    """(H, W, C) -> (num_patches, patch_size, patch_size, C), row-major ViT order."""
    H, W, C = image.shape
    nh, nw = H // patch_size, W // patch_size
    img = image[:nh * patch_size, :nw * patch_size, :]
    p = img.reshape(nh, patch_size, nw, patch_size, C).transpose(0, 2, 1, 3, 4)
    return p.reshape(nh * nw, patch_size, patch_size, C)


def glucodensity_patches(cgm: np.ndarray, gridsize: int = GRID_SIZE,
                         patch_size: int = SPATIAL_PATCH,
                         smoothing_factor: float = SMOOTHING_FACTOR) -> np.ndarray:
    """End to end: (L,) -> (16, 8, 8, 3) for gridsize=32, patch_size=8."""
    return to_spatial_patches(
        glucodensity_image(cgm, gridsize, smoothing_factor), patch_size)


def apply_spatial_mask(num_patches: int, mask_ratio: float, rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray]:
    """MAE-style random patch masking over the Glucodensity tokens."""
    n_mask = int(num_patches * mask_ratio)
    mask_idx = rng.choice(num_patches, size=n_mask, replace=False)
    non_mask_idx = np.setdiff1d(np.arange(num_patches), mask_idx)
    return np.sort(mask_idx), non_mask_idx


def precompute(windows: np.ndarray, keys: list, out_path: str | Path,
               gridsize: int = GRID_SIZE, patch_size: int = SPATIAL_PATCH,
               n_workers: int = 8, verbose: bool = True) -> Path:
    """Cache the Glucodensity view for a whole corpus.

    `windows`: (N, L) dense glucose array. `keys`: N hashable ids in the same order.
    Writes a pickle of {key: (num_patches, p, p, 3) float32} plus the config, so a
    later run can assert it is reading a cache built with matching settings.
    """
    from concurrent.futures import ProcessPoolExecutor

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fn = _Worker(gridsize, patch_size)
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            patches = list(ex.map(fn, windows, chunksize=16))
    else:
        patches = [fn(w) for w in windows]

    cache = {"config": {"gridsize": gridsize, "spatial_patch_size": patch_size,
                        "smoothing_factor": SMOOTHING_FACTOR, "L": int(windows.shape[1])},
             "gluco_patches": {k: p for k, p in zip(keys, patches)}}
    with open(out_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"cached {len(patches)} Glucodensity views -> {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


class _Worker:
    """Picklable top-level callable so ProcessPoolExecutor works on Windows spawn."""

    def __init__(self, gridsize: int, patch_size: int):
        self.gridsize, self.patch_size = gridsize, patch_size

    def __call__(self, w):
        return glucodensity_patches(w, self.gridsize, self.patch_size).astype(np.float32)


def load_cache(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
