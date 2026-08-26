"""
GlucoFM hyperparameters, transcribed from arXiv:2605.30865.

Every value carries the paper section it comes from. Values marked ASSUMED are
not stated in the paper and are reproduction choices (gap G4 in reproduction.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class GridConfig:
    """Section 3.1 / Appendix C.1 - chronological grid alignment."""
    length: int = 288          # L, 24 h at 5-minute resolution
    dt_min: int = 5            # delta t
    n_patches: int = 24        # P, one patch per hour
    patch_size: int = 12       # K = L / P


@dataclass
class FilterConfig:
    """Appendix C.3 - causal, mask-aware Gaussian state/event decoupling."""
    sigma_init: float = 6.0    # initial bandwidth, grid steps
    sigma_min: float = 2.0     # ~10 min
    sigma_max: float = 12.0    # ~60 min
    truncation: int = 3        # R = ceil(3 * sigma_max) = 36
    max_lookback: int = 9      # Eq. 10 - backward search for rate-of-change

    @property
    def radius(self) -> int:
        return int(self.truncation * self.sigma_max)


@dataclass
class ModelConfig:
    """Appendix C.4 - architecture and dimensionality."""
    # state stream feature widths
    state_wave_dim: int = 64
    state_diff_dim: int = 16
    state_stat_dim: int = 48
    # event stream feature widths
    event_wave_dim: int = 48
    event_roc_dim: int = 48
    event_stat_dim: int = 32
    # stream tokens and fused token
    stream_dim: int = 64
    embed_dim: int = 128       # D
    # transformer
    n_heads: int = 4
    ffn_dim: int = 256
    n_layers: int = 3          # context and EMA target encoders
    predictor_layers: int = 1  # lightweight masked-context predictor
    # "two lightweight transition heads" (App. C.4) - width not stated.
    # ASSUMED equal to the Transformer FFN width; see gap G4.
    transition_hidden: int = 256
    dropout: float = 0.1       # ASSUMED
    # ---- GlucoPRISM extensions (all OFF by default: with these at their
    # defaults the module is byte-for-byte the GlucoFM of the paper) ----
    #
    # masked_instance_norm z-scores every window and throws mu and sigma away,
    # so the entire stack sees SHAPE ONLY. Nearly every downstream label here is
    # a function of absolute level or spread (A1c ~ (mean + 46.7)/28.7;
    # "hypoglycemia" is literally a reading below 70 mg/dL), so that discarded
    # pair is not nuisance - it is the signal. `scale_inject` feeds
    # (log mu, log sd, CV, density) back in as a gated per-window token offset.
    scale_inject: bool = False
    scale_hidden: int = 64
    # `scale_inject` restores the window's mean and sd as SCALARS, but the
    # waveform the transformer reads is still unit-variance: a day with sd 10
    # and a day with sd 60 look identical in shape. CGM-JEPA normalises with one
    # GLOBAL (mean, sd) instead and beats GlucoFM at window level, so absolute
    # excursion magnitude carries signal on its own. `global_norm` adds a second,
    # globally-normalised view of the waveform as a gated per-patch offset,
    # making the encoder a superset of both designs.
    global_norm: bool = False
    global_mean: float = 141.07   # observed mean over corpus_balanced, mg/dL
    global_std: float = 57.97
    # target reported in the paper, asserted by build_model()
    expected_trainable: int = 720_000   # 0.72 M
    expected_total: int = 1_180_000     # 1.18 M
    param_tolerance: float = 0.08       # +/- 8% when checking the budget


@dataclass
class PretrainConfig:
    """Section 4 / Appendix C.5 - objectives and optimisation."""
    # masking
    mask_ratio_low: float = 0.5
    mask_ratio_high: float = 0.6
    # EMA target encoder
    ema_momentum: float = 0.997
    ema_final: float = 1.0     # ASSUMED cosine schedule 0.997 -> 1.0
    ema_schedule: bool = True  # ASSUMED
    # losses, both weights 1.0 (App. C.5)
    w_mcr: float = 1.0
    w_td: float = 1.0
    smooth_l1_beta: float = 1.0   # ASSUMED (torch default)
    # optimisation (section 4, "120 epochs ... batch size 128, lr 1e-4,
    # weight decay 1e-2, and a separate learning rate of 1e-3 for the
    # learnable Gaussian bandwidth")
    epochs: int = 120
    batch_size: int = 128
    lr: float = 1e-4
    lr_sigma: float = 1e-3
    weight_decay: float = 1e-2
    warmup_frac: float = 0.05   # ASSUMED
    grad_clip: float = 1.0      # ASSUMED
    seed: int = 0


@dataclass
class AugConfig:
    """Appendix C.7 - CGM-aware augmentation."""
    # value perturbations (probabilities and ranges ARE stated)
    wander_prob: float = 0.25
    wander_amp: tuple[float, float] = (5.0, 15.0)        # mg/dL
    wander_freq: tuple[float, float] = (0.5, 2.0)        # cycles per window
    compress_prob: float = 0.10
    compress_len: tuple[int, int] = (6, 12)              # grid steps
    compress_min_mult: tuple[float, float] = (0.4, 0.7)
    # structural sparsification - described in section 3.1 but magnitudes
    # are NOT stated in the paper. ASSUMED below.
    decimate_prob: float = 0.20        # 5-min -> 15-min-like sampling
    decimate_keep: int = 3             # keep 1 of every 3 positions
    dropout_prob: float = 0.25         # contiguous disconnection blocks
    dropout_blocks: tuple[int, int] = (1, 3)
    dropout_len: tuple[int, int] = (6, 24)               # grid steps
    # "randomized ordering and decayed co-occurrence probabilities
    #  prevent over-corrupting individual sequences" (section 3.1)
    cooccurrence_decay: float = 0.5
    enabled: bool = True


@dataclass
class ProbeConfig:
    """Section 4.2 / Appendix D - frozen linear probe."""
    n_folds: int = 5           # subject-grouped
    n_iterations: int = 10
    max_iter: int = 2000       # logistic regression solver budget
    C: float = 1.0
    standardize: bool = True
    seed: int = 0


@dataclass
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    filt: FilterConfig = field(default_factory=FilterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()
