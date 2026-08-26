# GluFormer's tokeniser lives with the GluFormer baseline, under
# baselines/common/models/, because it is that model's vocabulary
# rather than part of this paper's method.
"""Thin wrapper so the dataset layer does not import the model package."""

from __future__ import annotations

import numpy as np

from common.models.gluformer import GLUCOSE_MAX, GLUCOSE_MIN, N_BINS, tokenize_glucose


def to_tokens(g: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    return tokenize_glucose(g, n_bins=n_bins, lo=GLUCOSE_MIN, hi=GLUCOSE_MAX)
