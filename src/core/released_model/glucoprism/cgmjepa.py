"""
CGM-JEPA / X-CGM-JEPA adapter, so the official released encoders can be scored
by the same probe as GlucoFM and GlucoPRISM.

Weights: `checkpoints/cgm_jepa_official/{cgm_jepa,x_cgm_jepa}/model.safetensors`
(HuggingFace `CRUISEResearchGroup/CGM-JEPA`). Nothing is retrained here - the
encoder is frozen exactly as released.

Three details of CGM-JEPA's own pipeline are reproduced, because getting any of
them wrong would silently handicap the baseline:

1. **Global, not per-window, normalisation.** `data_transformer.PatchDataTransformer`
   applies `(x - global_mean) / global_std` with ONE mean/std pair for the whole
   dataset, and leaves unobserved positions at the sentinel `-1`. This is the
   opposite of GlucoFM's `masked_instance_norm`, and it is why CGM-JEPA is not
   level-blind: a global affine map preserves between-subject level differences.
   (See `details_glucoprism.md` A2 - that difference is the whole story of this
   project, so the baseline must be given its real advantage.)

2. **Patch grid.** `dim_in = 12`, `kernel_size = 3`: the series is cut into
   patches of 12 samples, and a stride-3 Conv1d runs inside each patch. At
   5-minute resolution over 288 points that is 24 hourly patches - the same grid
   GlucoFM uses, so the two models see identical temporal structure.

3. **Time features.** Pretraining and evaluation both pass `timestamp_patches`
   (`pretrain_cgm_jepa.py:187`, `trainer.py:167`), built by
   `time_features(timeenc=1, freq='t')` = [MinuteOfHour, HourOfDay, DayOfWeek,
   DayOfMonth, DayOfYear], each scaled to [-0.5, 0.5]. Our windows are
   circadian-aligned by `start_idx`, so minute-of-hour and hour-of-day are exact;
   the three calendar features are unavailable and are set to 0.0, their
   mid-range value. Dropping x_mark entirely would have been a distribution
   shift, since the encoder was pretrained with it.

Downstream readout follows `PatchDataTransformer.encode`: mean over patch
tokens of the encoder output (NOT the projection head).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CGM_JEPA_SRC = ROOT / "CGM-JEPA"
OFFICIAL = ROOT / "checkpoints" / "cgm_jepa_official"

# Global normalisation statistics. CGM-JEPA recomputes these from the training
# split of whichever dataset it is evaluating; we fix them once on OUR
# pretraining corpus instead, so the value is identical for every cohort and
# every CV fold and cannot leak downstream labels.
NORM_CACHE = ROOT / "data" / "processed" / "cgmjepa_norm.json"


def _import_encoder():
    """Import the released Encoder class from the vendored CGM-JEPA source."""
    for p in (CGM_JEPA_SRC, CGM_JEPA_SRC / "models"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from models.encoder import Encoder  # noqa: E402
    return Encoder


def global_stats(corpus_path: Path | None = None) -> tuple[float, float]:
    """Mean / sd of OBSERVED glucose over the pretraining corpus, cached."""
    if NORM_CACHE.exists():
        d = json.loads(NORM_CACHE.read_text(encoding="utf-8"))
        return float(d["mean"]), float(d["std"])
    corpus_path = corpus_path or (ROOT / "data" / "processed" / "corpus" /
                                  "corpus_balanced.npz")
    d = np.load(corpus_path, allow_pickle=True)
    x = np.nan_to_num(d["x"].astype(np.float64), nan=0.0)
    m = d["m"].astype(bool)
    v = x[m]
    mu, sd = float(v.mean()), float(v.std())
    NORM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    NORM_CACHE.write_text(json.dumps(
        {"mean": mu, "std": sd, "n_observed": int(v.size),
         "source": str(corpus_path.name)}, indent=2), encoding="utf-8")
    return mu, sd


class CGMJepaEncoder(torch.nn.Module):
    """Frozen wrapper: (x, m, start_idx) on our 288-grid -> patch tokens.

    `normalize` and `time_features` must match how the encoder was PRETRAINED,
    not what the eval script happens to default to. `config_pretrain.py` ships
    `normalize_x: False` ("off = use raw values") and `use_time_feature: False`
    ("since the evaluation dataset doesn't have timestamp, we don't use this to
    prevent embedding mismatch"), and `base_loader.__getitem__` returns
    `np.zeros_like(x)` for x_mark - so the RELEASED weights saw raw mg/dL and no
    time features. Both settings are measured rather than assumed; see
    `scripts/cgmjepa_input_convention.py`.
    """

    def __init__(self, encoder, mean: float, std: float,
                 patch_size: int = 12, length: int = 288,
                 normalize: bool = False, time_features: bool = False):
        super().__init__()
        self.enc = encoder
        self.mean, self.std = mean, std
        self.patch_size, self.length = patch_size, length
        self.n_patches = length // patch_size
        self.normalize, self.time_features = normalize, time_features

    def _x_mark(self, start_idx: torch.Tensor) -> torch.Tensor:
        """[B] -> [B, N, patch_size, 5] time features, Informer freq='t'."""
        B = start_idx.shape[0]
        dev = start_idx.device
        j = torch.arange(self.length, device=dev).unsqueeze(0)
        abs_i = (start_idx.view(-1, 1) + j) % self.length          # [B, L]
        minute_of_day = abs_i * 5
        minute = (minute_of_day % 60).float() / 59.0 - 0.5
        hour = (minute_of_day // 60).float() / 23.0 - 0.5
        zeros = torch.zeros_like(minute)
        # [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
        f = torch.stack([minute, hour, zeros, zeros, zeros], dim=-1)  # [B, L, 5]
        return f.reshape(B, self.n_patches, self.patch_size, 5)

    def forward(self, x, m, start_idx):
        """x, m: [B, 288]. Returns patch tokens [B, 24, 96]."""
        xs = (x - self.mean) / self.std if self.normalize else x
        xs = torch.where(m > 0, xs, torch.full_like(xs, -1.0))   # CGM-JEPA sentinel
        patches = xs.reshape(x.shape[0], self.n_patches, self.patch_size)
        xm = (self._x_mark(start_idx) if self.time_features
              else torch.zeros(x.shape[0], self.n_patches, self.patch_size, 5,
                               device=x.device, dtype=patches.dtype))
        tok, _proj = self.enc(patches, xm)
        return tok


def load(variant: str = "cgm_jepa", device: str = "cpu",
         root: Path | None = None, normalize: bool | None = None,
         time_features: bool | None = None) -> CGMJepaEncoder:
    """variant in {'cgm_jepa', 'x_cgm_jepa'}, or a directory holding a run we
    pretrained ourselves (`cgmjepa.pt` + `config.json` + `input_convention`)."""
    root = Path(root or OFFICIAL) / variant
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    conv = {k: cfg.pop(k) for k in ("normalize", "time_features") if k in cfg}
    Encoder = _import_encoder()
    enc = Encoder(**cfg)

    sd_path = root / "model.safetensors"
    if sd_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(sd_path))
    else:
        f = root / "cgmjepa.pt"
        state = torch.load(f if f.exists() else root / "pytorch_model.bin",
                           map_location="cpu")
        state = state.get("encoder", state)
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"CGM-JEPA weights incomplete, missing: {missing[:6]}")

    mu, sd = global_stats()
    norm = conv.get("normalize", False) if normalize is None else normalize
    tf = conv.get("time_features", False) if time_features is None else time_features
    model = CGMJepaEncoder(enc, mu, sd, normalize=norm,
                           time_features=tf).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in enc.parameters())
    print(f"  CGM-JEPA[{variant}]: {n:,} params, embed_dim={cfg['embed_dim']}, "
          f"normalize={norm}, time_features={tf}, unexpected keys={len(unexpected)}")
    return model
