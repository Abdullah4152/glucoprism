"""Torch datasets over the harmonised 24-hour window shards.

A shard is a single .npz written by `scripts/build_corpus.py` holding the column
arrays from `windows.windows_to_arrays`. Each model gets the *view* its paper
specifies, from the same underlying shard:

    GlucoFM        (glucose with NaN gaps, mask, start_idx)  -- mask preserved
    CGM-JEPA       dense 288 -> 24 patches of 12              -- linearly interpolated
    X-CGM-JEPA     the above plus the cached Glucodensity tokens
    GluFormer      dense 288 -> 288 discrete glucose tokens   -- clipped to [40, 500]

GlucoFM App. B.3 is explicit that the CGM-JEPA and GluFormer baselines are fed
the interpolated dense sequence, which is why `densify()` lives on the baseline
side of the fence and never touches GlucoFM's own input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment
from .glucodensity import load_cache
from .gluformer_tokens import to_tokens  # noqa: F401  (re-export convenience)
from .windows import densify


class WindowShard:
    """Lazy reader over one or more .npz window shards."""

    def __init__(self, paths: str | Path | list[str | Path]):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        arrays: dict[str, list[np.ndarray]] = {}
        for p in paths:
            with np.load(Path(p), allow_pickle=True) as z:
                for k in z.files:
                    arrays.setdefault(k, []).append(z[k])
        self.data = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}

    def __len__(self) -> int:
        return len(self.data["glucose"])

    @property
    def subjects(self) -> np.ndarray:
        return self.data["subject"]

    def subset(self, idx: np.ndarray) -> "WindowShard":
        s = WindowShard.__new__(WindowShard)
        s.data = {k: v[idx] for k, v in self.data.items()}
        return s


class GlucoFMDataset(Dataset):
    """Mask-preserving view. Missing positions are zero-filled *for tensor
    construction only* -- the mask is what the model actually reads (App. C.1)."""

    def __init__(self, shard: WindowShard, augment_prob: bool = False, seed: int = 0,
                 sensor_aug: float = 0.0):
        """`sensor_aug` is the probability of ALSO rendering the window through
        the FD-9-calibrated second-sensor transform.

        FD-8's post-hoc factorization arm (V6/V7) fits sensor heads onto a frozen
        encoder. If that encoder never saw sensor variation during pretraining it
        has no reason to have encoded any, and the arm would fail for a reason
        that has nothing to do with factorization. Generating a partner and
        discarding it would achieve nothing -- with the sensor loss off, nothing
        reads the partner -- so the transform is applied to the ANCHOR instead.
        """
        self.shard = shard
        self.do_augment = augment_prob
        self.seed = seed
        self.sensor_aug = float(sensor_aug)

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, i):
        g = self.shard.data["glucose"][i].astype(np.float32)
        m = self.shard.data["mask"][i].astype(np.float32)
        s = int(self.shard.data["start_idx"][i])

        if self.do_augment:
            rng = np.random.default_rng((self.seed, i, torch.initial_seed() % (1 << 31)))
            g, m = augment(np.nan_to_num(g), m, rng)
            if self.sensor_aug and rng.random() < self.sensor_aug:
                from .augment import synthetic_libre_view
                g, m = synthetic_libre_view(np.nan_to_num(g), m, rng)

        g = np.nan_to_num(g, nan=0.0) * m
        return (torch.from_numpy(g), torch.from_numpy(m), torch.tensor(s, dtype=torch.long))


class CGMJEPADataset(Dataset):
    """Dense 288 -> P=24 patches of 12, with per-sample random patch masking."""

    def __init__(self, shard: WindowShard, patch_size: int = 12, mask_ratio: float = 0.25,
                 gluco_cache: str | Path | None = None, normalize: bool = False,
                 seed: int = 43):
        self.shard = shard
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.normalize = normalize
        self.seed = seed
        self.mean = self.std = None

        self.gluco = None
        if gluco_cache is not None:
            self.gluco = load_cache(gluco_cache)["gluco_patches"]

    def fit_stats(self):
        """CGM-JEPA's `compute_stats(normalize_x=...)`; off by default, as in their config."""
        dense = np.stack([densify(self.shard.data["glucose"][i], self.shard.data["mask"][i])
                          for i in range(len(self.shard))])
        self.mean, self.std = float(dense.mean()), float(dense.std())
        return self.mean, self.std

    def __len__(self):
        return len(self.shard)

    def _dense(self, i):
        x = densify(self.shard.data["glucose"][i], self.shard.data["mask"][i])
        if self.normalize and self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        return x.astype(np.float32)

    def __getitem__(self, i):
        x = self._dense(i)
        P = len(x) // self.patch_size
        patches = x.reshape(P, self.patch_size)

        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % (1 << 31)))
        n_mask = int(P * self.mask_ratio)
        mask_idx = np.sort(rng.choice(P, size=n_mask, replace=False))
        non_mask_idx = np.setdiff1d(np.arange(P), mask_idx)

        item = [torch.from_numpy(patches),
                torch.zeros(P, self.patch_size, 5),          # time features disabled by default
                torch.from_numpy(mask_idx).long(),
                torch.from_numpy(non_mask_idx).long()]

        if self.gluco is not None:
            key = (str(self.shard.data["subject"][i]), int(self.shard.data["segment"][i]),
                   str(self.shard.data["start_time"][i]))
            item.append(torch.from_numpy(np.asarray(self.gluco[key], dtype=np.float32)))
        return tuple(item)


class GluFormerDataset(Dataset):
    """Dense 288 -> 288 discrete glucose tokens over the [40, 500] mg/dL range."""

    def __init__(self, shard: WindowShard, n_bins: int = 460):
        self.shard = shard
        self.n_bins = n_bins

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, i):
        x = densify(self.shard.data["glucose"][i], self.shard.data["mask"][i])
        return torch.from_numpy(to_tokens(x, self.n_bins)).long()


def make_gluco_cache_keys(shard: WindowShard) -> list[tuple]:
    return [(str(shard.data["subject"][i]), int(shard.data["segment"][i]),
             str(shard.data["start_time"][i])) for i in range(len(shard))]
