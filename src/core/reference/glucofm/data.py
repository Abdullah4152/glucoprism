"""
Datasets and CGM-aware augmentation for GlucoFM (section 3.1, Appendix C.7).

Windows come from scripts/preprocess_cgm.py as .npz files holding
    x    [N, 288] float32, NaN at unobserved positions
    m    [N, 288] uint8    observation mask
    s    [N]      int16    circadian start index
    subj [N]      str      subject id
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import AugConfig, Config, DEFAULT


# --------------------------------------------------------------------------- #
def load_windows(paths: list[Path]) -> dict:
    """Concatenate several window files, prefixing subjects with the cohort name."""
    xs, ms, ss, subs, coh = [], [], [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        name = p.stem.replace("_pretrain", "").replace("_downstream", "")
        xs.append(d["x"]); ms.append(d["m"]); ss.append(d["s"])
        subs.append(np.array([f"{name}/{v}" for v in d["subj"]]))
        coh.append(np.array([name] * len(d["x"])))
    if not xs:
        raise FileNotFoundError("no window files found")
    return {
        "x": np.concatenate(xs).astype(np.float32),
        "m": np.concatenate(ms).astype(np.float32),
        "s": np.concatenate(ss).astype(np.int64),
        "subj": np.concatenate(subs),
        "cohort": np.concatenate(coh),
    }


# --------------------------------------------------------------------------- #
class CGMAugment:
    """Appendix C.7.

    Value perturbations keep the observation mask and change observed values:
      * baseline wander  - p=0.25, sinusoid, amplitude [5,15] mg/dL,
                           frequency [0.5,2.0] cycles per window
      * compression drop - p=0.10, V-shaped attenuation over 6-12 steps,
                           minimum multiplier [0.4,0.7]

    Structural sparsification changes the observation pattern itself:
      * decimation to 15-minute-like sampling
      * contiguous disconnection blocks
    Magnitudes for these two are NOT stated in the paper (gap G4).

    "randomized ordering and decayed co-occurrence probabilities prevent
     over-corrupting individual sequences" - operations are shuffled and each
    applied one after another at a probability decayed by `cooccurrence_decay`
    for every operation already applied to this window.
    """

    def __init__(self, cfg: AugConfig, length: int = 288, rng: np.random.Generator | None = None):
        self.c = cfg
        self.L = length
        self.rng = rng or np.random.default_rng()

    # -- value perturbations ------------------------------------------------ #
    def _wander(self, x, m):
        amp = self.rng.uniform(*self.c.wander_amp)
        freq = self.rng.uniform(*self.c.wander_freq)
        phase = self.rng.uniform(0, 2 * np.pi)
        t = np.arange(self.L) / self.L
        return x + amp * np.sin(2 * np.pi * freq * t + phase) * m, m

    def _compress(self, x, m):
        ln = int(self.rng.integers(self.c.compress_len[0], self.c.compress_len[1] + 1))
        if ln >= self.L:
            return x, m
        st = int(self.rng.integers(0, self.L - ln))
        lo = self.rng.uniform(*self.c.compress_min_mult)
        # V-shaped: 1 -> lo -> 1 across the segment
        half = ln / 2.0
        idx = np.arange(ln)
        v = lo + (1.0 - lo) * np.abs(idx - half) / half
        mult = np.ones(self.L, dtype=np.float32)
        mult[st:st + ln] = v
        return x * mult, m

    # -- structural sparsification ------------------------------------------ #
    def _decimate(self, x, m):
        keep = self.c.decimate_keep
        off = int(self.rng.integers(0, keep))
        sel = np.zeros(self.L, dtype=np.float32)
        sel[off::keep] = 1.0
        return x, m * sel

    def _disconnect(self, x, m):
        n = int(self.rng.integers(self.c.dropout_blocks[0], self.c.dropout_blocks[1] + 1))
        m = m.copy()
        for _ in range(n):
            ln = int(self.rng.integers(self.c.dropout_len[0], self.c.dropout_len[1] + 1))
            if ln >= self.L:
                continue
            st = int(self.rng.integers(0, self.L - ln))
            m[st:st + ln] = 0.0
        return x, m

    # ----------------------------------------------------------------------- #
    def __call__(self, x: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.c.enabled:
            return x, m
        ops = [
            (self._wander, self.c.wander_prob),
            (self._compress, self.c.compress_prob),
            (self._decimate, self.c.decimate_prob),
            (self._disconnect, self.c.dropout_prob),
        ]
        self.rng.shuffle(ops)                      # randomised ordering
        x, m = x.copy(), m.copy()
        applied = 0
        for fn, p in ops:
            p_eff = p * (self.c.cooccurrence_decay ** applied)   # decayed co-occurrence
            if self.rng.random() < p_eff:
                x, m = fn(x, m)
                applied += 1
        if m.sum() < 2:                            # never return an empty window
            return x, np.ones_like(m) * (m.sum() > 0) if m.sum() else m
        return x, m


# --------------------------------------------------------------------------- #
class PretrainDataset(Dataset):
    """Unlabelled 24-hour windows with on-the-fly augmentation and patch masking."""

    def __init__(self, data: dict, cfg: Config = DEFAULT, train: bool = True,
                 seed: int = 0):
        self.d = data
        self.cfg = cfg
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.aug = CGMAugment(cfg.aug, cfg.grid.length, self.rng)

    def __len__(self):
        return len(self.d["x"])

    def __getitem__(self, i):
        g, p = self.cfg.grid, self.cfg.pretrain
        x = np.nan_to_num(self.d["x"][i], nan=0.0).astype(np.float32)
        m = self.d["m"][i].astype(np.float32)
        if self.train:
            x, m = self.aug(x, m)
        # NumPy promotes to float64 inside the augmentations (e.g. the sinusoid
        # in baseline wander); torch layers are float32.
        x = (x * m).astype(np.float32)                # unobserved positions carry no value
        m = m.astype(np.float32)

        # patch mask: ratio ~ U[0.5, 0.6] (App. C.5)
        P = g.n_patches
        ratio = self.rng.uniform(p.mask_ratio_low, p.mask_ratio_high)
        k = max(1, int(round(ratio * P)))
        pm = np.zeros(P, dtype=np.float32)
        pm[self.rng.choice(P, size=k, replace=False)] = 1.0

        return {
            "x": torch.from_numpy(x),
            "m": torch.from_numpy(m),
            "s": torch.tensor(int(self.d["s"][i]), dtype=torch.long),
            "patch_mask": torch.from_numpy(pm),
        }


class EmbedDataset(Dataset):
    """Windows without augmentation or masking, for frozen feature extraction."""

    def __init__(self, data: dict):
        self.d = data

    def __len__(self):
        return len(self.d["x"])

    def __getitem__(self, i):
        x = np.nan_to_num(self.d["x"][i], nan=0.0).astype(np.float32)
        m = self.d["m"][i].astype(np.float32)
        return {
            "x": torch.from_numpy(x * m),
            "m": torch.from_numpy(m),
            "s": torch.tensor(int(self.d["s"][i]), dtype=torch.long),
            "idx": torch.tensor(i, dtype=torch.long),
        }
