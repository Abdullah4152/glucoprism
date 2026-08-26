"""GlucoFM: a dual-stream foundation model for CGM (Li et al., 2026, arXiv:2605.30865).

Reproduced from the paper text (Sec. 3) plus Appendix C, which specifies the
architecture down to the per-stream feature widths. No official code release
exists at the time of writing, so every equation number cited below points at
the paper; anything the paper leaves underspecified is flagged with
`# UNDERSPECIFIED:` and the choice we made.

Pipeline (Fig. 2, top to bottom):

    aligned 24 h grid (L=288) + observation mask M
      -> mask-aware instance normalisation
      -> learnable causal mask-aware Gaussian filter (sigma in [2, 12])   Eq. 11-13
      -> state = filtered trend, event = (X - state) * M                  Eq. 1, 14
      -> per-stream patch embedders (P=24 patches of K=12)                App. C.4
      -> fusion -> unified physiological patch token (D=128)
      -> + circular time-of-day features, gated with positional embedding
      -> 3-layer Transformer context encoder / EMA target encoder
      -> L_MCR (masked contextual latent prediction)                      Eq. 2
      -> L_TD  (next-patch state/event dynamics)                          Eq. 3-4

Budget check: 0.72M trainable / 1.18M total during pretraining (paper Sec. 3.3).
`glucofm_param_report()` reproduces both numbers.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# ------------------------------------------------------------------- config

def patchify(t: torch.Tensor, cfg: "GlucoFMConfig") -> torch.Tensor:
    """(B, L) -> (B, P, K), honouring `cfg.patch_stride`.

    With the default stride == K this is exactly the papers' reshape, and the
    fast path below keeps it bit-identical. With stride < K each patch also
    carries the preceding `K - stride` grid steps.

    Patch 0's lookback would need positions before the window starts. Those are
    zero-padded, and because the observation mask is patchified through this same
    function they arrive marked unobserved -- the model is told the history is
    missing rather than being handed a fabricated value, which is the discipline
    the 288-grid already uses for sensor dropouts.
    """
    B, L = t.shape
    P, K, stride, lead = cfg.P, cfg.K, cfg.stride, cfg.lookback
    if lead == 0:
        if P * K != L:
            raise ValueError(f"P*K ({P}*{K}) != L ({L})")
        return t.view(B, P, K)
    if P * stride != L:
        raise ValueError(f"P*stride ({P}*{stride}) != L ({L})")
    pad = t.new_zeros(B, lead)
    return torch.cat([pad, t], dim=1).unfold(dimension=1, size=K, step=stride)


@dataclass
class GlucoFMConfig:
    L: int = 288                 # grid positions in a 24 h window (5 min)
    P: int = 24                  # temporal patches
    K: int = 12                  # grid steps per patch (1 hour)

    d_model: int = 128           # unified physiological patch token
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.0

    d_stream: int = 64           # state / event token width before fusion

    # FD-7 (W4/W5): patches may overlap. `patch_stride` is how far apart
    # consecutive patches START; `K` is how wide each one is. stride == K is the
    # papers' non-overlapping tokenisation and stays the default. stride < K
    # gives every patch `K - stride` grid steps of lookback, so a meal beginning
    # at 12:50 is visible to the 13:00 patch instead of being split across a
    # boundary. `P * patch_stride == L` is required either way, so the patches
    # still span exactly one day.
    patch_stride: int | None = None      # None -> K (non-overlapping)

    @property
    def stride(self) -> int:
        return self.K if self.patch_stride is None else self.patch_stride

    @property
    def lookback(self) -> int:
        return self.K - self.stride

    # state stream feature widths (App. C.4)
    state_wave_dim: int = 64
    state_trend_dim: int = 16
    state_stats_dim: int = 48
    # event stream feature widths (App. C.4)
    event_wave_dim: int = 48
    event_roc_dim: int = 48
    event_stats_dim: int = 32

    # causal Gaussian filter (App. C.3)
    sigma_min: float = 2.0
    sigma_max: float = 12.0
    sigma_init: float = 6.0
    trunc_factor: float = 3.0    # R = ceil(trunc_factor * sigma_max) = 36

    # rate-of-change search window (App. C.2)
    roc_max_lag: int = 9

    # App. C.2 (Eq. 9, 10) computes patch statistics and rate-of-change from the
    # *aligned* sequence X-hat, while App. C.3 (Eq. 11, 14) filters the
    # *normalised* sequence X-tilde. Keeping the statistics un-normalised is what
    # preserves absolute glycemic level and between-window variability -- exactly
    # the signal a task like Hall's glucotype depends on. Raw mg/dL would swamp a
    # freshly-initialised Linear, so the statistics are put on an O(1) scale with
    # these fixed physiological constants (NOT per-window, which would destroy the
    # very information Eq. 9 exists to keep).
    glucose_center: float = 100.0   # mg/dL
    glucose_scale: float = 50.0     # mg/dL

    # UNDERSPECIFIED: Sec. 3.3 only calls g_S / g_E "two lightweight transition
    # heads". Everything else in the architecture is pinned by App. C.4, so we
    # solve for this one free width against the reported parameter budget:
    # 208 gives 720,241 trainable / 1,173,698 total vs the paper's 0.72M / 1.18M.
    d_transition: int = 208

    # pretraining
    mask_ratio_low: float = 0.5
    mask_ratio_high: float = 0.6
    ema_momentum: float = 0.997
    lambda_mcr: float = 1.0
    lambda_td: float = 1.0

    # --- Clinical Metric Prediction (L_CMP). OFF by default: with the flags at
    # their defaults this module is byte-for-byte the GlucoFM of the paper, and
    # `glucofm_param_report()` still returns 720,241.
    #
    # Both JEPA objectives live in latent space on a per-window normalised
    # signal, so nothing in training ever asks the representation to retain a
    # quantity measured in mg/dL -- yet every downstream label here is defined on
    # one (HbA1c ~ mean glucose; hypoglycemia is a reading < 70; glucotype is a
    # variability class). L_CMP closes that gap with NO label: the targets are
    # standard CGM glucometrics computed from the raw window, predicted from the
    # MASKED context view so it stays a predictive task rather than reconstruction.
    use_cmp: bool = False
    w_cmp: float = 1.0
    cmp_circadian: bool = True      # add nocturnal / daytime means

    # --- Global-scale waveform view. OFF by default.
    #
    # `_normalise` z-scores every window, so the WAVEFORM the Transformer reads is
    # unit-variance: a day spanning 80-110 mg/dL and a day spanning 60-300 have
    # the same shape input. Eq. 9 does restore absolute level, but only as two
    # scalars per patch (patch mean and sd). This adds the full waveform on ONE
    # fixed corpus-wide scale as a gated per-patch offset, so between-window
    # differences in excursion magnitude reach the encoder as a signal it can
    # attend over rather than as a summary statistic.
    #
    # Constants measured over the 2,745,507 observed readings of the pretraining
    # corpus (`data/processed/*_pt.npz`).
    global_norm: bool = False
    global_mean: float = 156.90     # mg/dL
    global_std: float = 63.87
    # deeper / wider encoder, for the capacity sweep
    n_layers_override: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------- causal Gaussian filter

class CausalMaskAwareGaussianFilter(nn.Module):
    """Eq. 11-13. One-sided (r >= 0) Gaussian smoother with a learnable bandwidth.

    Mask-aware: the kernel is renormalised over *observed* positions only, so a
    dropout gap does not pull the trend toward zero. Causal: only current and
    past grid positions contribute, so no future glucose leaks into the state.
    """

    def __init__(self, cfg: GlucoFMConfig):
        super().__init__()
        self.sigma_min, self.sigma_max = cfg.sigma_min, cfg.sigma_max
        self.R = int(math.ceil(cfg.trunc_factor * cfg.sigma_max))  # 36

        # rho is the unconstrained parameter; sigma = min + (max-min)*sigmoid(rho).
        # Solve sigmoid(rho_0) = (sigma_init - min) / (max - min) so sigma starts at 6.0.
        frac = (cfg.sigma_init - cfg.sigma_min) / (cfg.sigma_max - cfg.sigma_min)
        frac = min(max(frac, 1e-4), 1 - 1e-4)
        self.rho = nn.Parameter(torch.tensor(math.log(frac / (1 - frac)), dtype=torch.float32))
        self.register_buffer("lags", torch.arange(self.R + 1, dtype=torch.float32))

    @property
    def sigma(self) -> torch.Tensor:
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(self.rho)

    def kernel(self) -> torch.Tensor:
        """K_sigma(r) for r = 0..R, normalised to sum 1 (Eq. 12)."""
        k = torch.exp(-self.lags.pow(2) / (2 * self.sigma.pow(2)))
        return k / (k.sum() + EPS)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """x, m: (B, L) -> state estimate (B, L). Positions with no observed
        support in the causal window fall back to x itself."""
        k = self.kernel()
        # F.conv1d is cross-correlation, so flip so that tap i maps to lag R-i.
        w = torch.flip(k, dims=[0]).view(1, 1, -1)

        xm = (x * m).unsqueeze(1)
        mm = m.unsqueeze(1)
        num = F.conv1d(F.pad(xm, (self.R, 0)), w).squeeze(1)
        den = F.conv1d(F.pad(mm, (self.R, 0)), w).squeeze(1)

        state = num / (den + EPS)
        # Guard positions whose entire causal window is unobserved.
        return torch.where(den > EPS, state, x)


# ---------------------------------------------------------- stream encoders

class StateStreamEmbedder(nn.Module):
    """App. C.4 state stream: waveform (64) + intra-patch trend diff (16)
    + projected patch mean/std (48) -> 64-d state token."""

    def __init__(self, cfg: GlucoFMConfig):
        super().__init__()
        self.cfg = cfg
        self.wave = nn.Linear(cfg.K, cfg.state_wave_dim)
        self.trend = nn.Linear(cfg.K - 1, cfg.state_trend_dim)
        self.stats = nn.Linear(2, cfg.state_stats_dim)
        self.proj = nn.Linear(cfg.state_wave_dim + cfg.state_trend_dim + cfg.state_stats_dim,
                              cfg.d_stream)
        self.act = nn.GELU()

    def forward(self, state: torch.Tensor, mask: torch.Tensor, raw: torch.Tensor):
        """state (normalised, Eq. 11), mask, raw (aligned X-hat): each (B, L).

        Waveform features come from the *normalised* state stream; the patch
        statistics of Eq. 9 come from the *aligned* sequence. Returns
        (B, P, d_stream) and the patch density (B, P).
        """
        B, L = state.shape
        K = self.cfg.K
        s = patchify(state, self.cfg)
        m = patchify(mask, self.cfg)
        r = patchify(raw, self.cfg)

        n_obs = m.sum(-1)                                   # (B, P)
        density = n_obs / K                                 # d_i, Eq. 8

        mu = (m * r).sum(-1) / (n_obs + EPS)                                   # Eq. 9
        var = (m * (r - mu.unsqueeze(-1)).pow(2)).sum(-1) / (n_obs + EPS)
        # sqrt is not differentiable at 0 and empty/constant patches hit exactly 0,
        # so clamp inside the sqrt rather than after it.
        sd = (var + EPS).sqrt()
        mu = (mu - self.cfg.glucose_center) / self.cfg.glucose_scale
        sd = sd / self.cfg.glucose_scale

        feats = torch.cat([
            self.act(self.wave(s * m)),                     # zero out unobserved steps
            self.act(self.trend(torch.diff(s, dim=-1))),
            self.act(self.stats(torch.stack([mu, sd], dim=-1))),
        ], dim=-1)
        tok = self.proj(feats)

        valid = (n_obs > 0).float().unsqueeze(-1)           # empty patches are zeroed
        return tok * valid, density


class EventStreamEmbedder(nn.Module):
    """App. C.4 event stream: residual waveform (48) + rate-of-change (48)
    + projected RoC mean/std (32) -> 64-d event token."""

    def __init__(self, cfg: GlucoFMConfig):
        super().__init__()
        self.cfg = cfg
        self.wave = nn.Linear(cfg.K, cfg.event_wave_dim)
        self.roc = nn.Linear(cfg.K, cfg.event_roc_dim)
        self.stats = nn.Linear(2, cfg.event_stats_dim)
        self.proj = nn.Linear(cfg.event_wave_dim + cfg.event_roc_dim + cfg.event_stats_dim,
                              cfg.d_stream)
        self.act = nn.GELU()

    def forward(self, event: torch.Tensor, mask: torch.Tensor,
                roc: torch.Tensor, roc_valid: torch.Tensor):
        B, L = event.shape
        e = patchify(event, self.cfg)
        m = patchify(mask, self.cfg)
        r = patchify(roc, self.cfg)
        rv = patchify(roc_valid, self.cfg)

        n_obs = m.sum(-1)
        n_roc = rv.sum(-1)
        # Eq. 10's rate-of-change is defined on the aligned sequence, so it arrives
        # in mg/dL per grid step; rescale rather than re-normalise per window.
        r = r / self.cfg.glucose_scale
        mu = (rv * r).sum(-1) / (n_roc + EPS)
        var = (rv * (r - mu.unsqueeze(-1)).pow(2)).sum(-1) / (n_roc + EPS)
        sd = (var + EPS).sqrt()

        feats = torch.cat([
            self.act(self.wave(e * m)),
            self.act(self.roc(r * rv)),
            self.act(self.stats(torch.stack([mu, sd], dim=-1))),
        ], dim=-1)
        tok = self.proj(feats)
        return tok * (n_obs > 0).float().unsqueeze(-1)


class UnifiedEmbedder(nn.Module):
    """Fuse state+event into the D=128 physiological patch token and add
    circadian + positional information through a learnable gate (Sec. 3.2)."""

    def __init__(self, cfg: GlucoFMConfig):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(2 * cfg.d_stream, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.time_proj = nn.Linear(2, cfg.d_model)          # tau = [sin, cos]
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.P, cfg.d_model))
        # UNDERSPECIFIED: the paper says "combined ... through a learnable gate"
        # without giving the form. We use a per-dimension sigmoid gate that mixes
        # the circadian projection against the patch positional embedding, which
        # is the minimal parameterisation consistent with that sentence.
        self.gate = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, state_tok, event_tok, tau):
        h = self.fuse(torch.cat([state_tok, event_tok], dim=-1))
        g = torch.sigmoid(self.gate)
        return h + g * self.time_proj(tau) + (1.0 - g) * self.pos_emb


# ------------------------------------------------------------- transformers

class TransformerStack(nn.Module):
    """Pre-norm encoder, d_model=128, 4 heads, FF 256 (Sec. 3.3)."""

    def __init__(self, cfg: GlucoFMConfig, n_layers: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x):
        return self.norm(self.enc(x))


class TransitionHead(nn.Module):
    """g_S / g_E in Eq. 3: residual next-patch predictor over [own, other, tau]."""

    def __init__(self, cfg: GlucoFMConfig):
        super().__init__()
        d_in = 2 * cfg.d_stream + cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(d_in, cfg.d_transition),
            nn.GELU(),
            nn.Linear(cfg.d_transition, cfg.d_stream),
        )

    def forward(self, own, other, tau_emb):
        return own + self.net(torch.cat([own, other, tau_emb], dim=-1))


# ---------------------------------------------------------------- backbone

class GlucoFMBackbone(nn.Module):
    """Everything that is mirrored into the EMA target branch."""

    def __init__(self, cfg: GlucoFMConfig):
        super().__init__()
        self.cfg = cfg
        self.filter = CausalMaskAwareGaussianFilter(cfg)
        self.state_embed = StateStreamEmbedder(cfg)
        self.event_embed = EventStreamEmbedder(cfg)
        self.unified = UnifiedEmbedder(cfg)
        self.encoder = TransformerStack(cfg, cfg.n_layers)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Gated so the encoder can ignore it entirely; the gate starts at
        # sigmoid(0) = 0.5 rather than 0, because a zero-initialised gate gives
        # the branch no gradient to escape from.
        if cfg.global_norm:
            self.global_wave = nn.Linear(cfg.K, cfg.d_model)
            self.global_gate = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        else:
            self.global_wave = None

    # -- signal decomposition ------------------------------------------------

    def _normalise(self, x, m):
        """Mask-aware instance normalisation (Fig. 2)."""
        n = m.sum(-1, keepdim=True)
        mu = (x * m).sum(-1, keepdim=True) / (n + EPS)
        var = (m * (x - mu).pow(2)).sum(-1, keepdim=True) / (n + EPS)
        # clamp the scale, not the variance: a fully flat / fully missing window
        # would otherwise divide by zero and produce NaNs through the filter.
        sd = (var + EPS).sqrt().clamp_min(1e-3)
        return (x - mu) / sd

    def _rate_of_change(self, x, m):
        """Eq. 10: backward search up to `roc_max_lag` for the closest observed pair."""
        B, L = x.shape
        roc = torch.zeros_like(x)
        valid = torch.zeros_like(x)
        for b in range(1, self.cfg.roc_max_lag + 1):
            prev_x = F.pad(x, (b, 0))[:, :L]
            prev_m = F.pad(m, (b, 0))[:, :L]
            usable = (m > 0) & (prev_m > 0) & (valid == 0)
            roc = torch.where(usable, (x - prev_x) / b, roc)
            valid = torch.where(usable, torch.ones_like(valid), valid)
        return roc, valid

    def decompose(self, x, m):
        """Eq. 1 / 11 / 14.

        The state/event split runs on the normalised sequence X-tilde, while the
        rate-of-change of Eq. 10 is defined on the aligned sequence X-hat. Returns
        (state, event, roc, roc_valid) with the first two normalised and the third
        in mg/dL per grid step.
        """
        xn = self._normalise(x, m)
        state = self.filter(xn, m)
        event = (xn - state) * m
        roc, roc_valid = self._rate_of_change(x, m)
        return state, event, roc, roc_valid

    # -- tokenisation --------------------------------------------------------

    def circadian(self, start_idx: torch.Tensor) -> torch.Tensor:
        """tau_i = [sin(2*pi*a/L), cos(2*pi*a/L)] at each patch's absolute grid index."""
        P, L = self.cfg.P, self.cfg.L
        # A patch is located by where its OWN hour begins, not by where its
        # lookback begins -- so an overlapping tokenisation keeps the same
        # circadian phase per patch as the non-overlapping one, and W1 vs W4 is
        # a clean comparison. stride == K makes this identical to `arange(P)*K`.
        offs = torch.arange(P, device=start_idx.device) * self.cfg.stride
        a = (start_idx.unsqueeze(1) + offs.unsqueeze(0)) % L
        ang = 2 * math.pi * a.float() / L
        return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)

    def tokenise(self, x, m, start_idx):
        state, event, roc, roc_valid = self.decompose(x, m)
        s_tok, density = self.state_embed(state, m, x)
        e_tok = self.event_embed(event, m, roc, roc_valid)
        tau = self.circadian(start_idx)
        fused = self.unified(s_tok, e_tok, tau)
        return fused, s_tok, e_tok, tau, density

    # -- forward -------------------------------------------------------------

    def forward(self, x, m, start_idx, patch_mask=None):
        """patch_mask: (B, P) bool, True where the patch is hidden from the context branch."""
        if patch_mask is not None:
            # Hidden patches are removed from the *signal* before stats/filtering,
            # so masked content cannot leak through the mask-aware statistics.
            # Each patch OWNS `stride` grid steps; with overlap it also reads the
            # previous patch's tail. Zeroing `stride` steps per masked patch
            # covers the sequence exactly once (P * stride == L). Using K here
            # would double-zero the overlap regions and hide more than the
            # sampled mask ratio. stride == K makes this the original line.
            m = m * (~patch_mask).float().repeat_interleave(self.cfg.stride, dim=1)

        fused, s_tok, e_tok, tau, density = self.tokenise(x, m, start_idx)

        if self.global_wave is not None:
            # `m` is already the branch's visible mask, so hidden patches contribute
            # nothing here and cannot leak through the global view.
            cfg = self.cfg
            xg = ((x - cfg.global_mean) / cfg.global_std) * m
            gtok = self.global_wave(patchify(xg, cfg))
            fused = fused + torch.sigmoid(self.global_gate) * gtok

        if patch_mask is not None:
            fused = torch.where(patch_mask.unsqueeze(-1),
                                self.mask_token.expand_as(fused), fused)

        z = self.encoder(fused)
        return {"z": z, "state_tok": s_tok, "event_tok": e_tok,
                "tau": tau, "density": density}


# ------------------------------------------------- clinical metric targets

LEVEL_METRICS = ("mean", "tbr70", "tir", "tar180", "tar250")
DISP_METRICS = ("sd", "cv", "mad", "hourly_sd", "range")
CIRCADIAN_METRICS = ("nocturnal", "daytime")


@torch.no_grad()
def window_glucometrics(x: torch.Tensor, m: torch.Tensor,
                        start_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Mask-aware CGM summary statistics used as L_CMP targets. (B, L) -> (B, K).

    These are targets only -- no gradient flows here. Everything is returned on
    roughly unit scale so one SmoothL1 weight is sensible across all of them.

    The circadian pair exists because HOMA-IR and the ADA diabetes thresholds are
    defined on FASTING glucose, not a 24-hour mean, and our windows carry an
    absolute time-of-day index so nocturnal glucose is directly addressable.
    """
    m = m.float()
    cnt = m.sum(1).clamp(min=1.0)
    mean = (x * m).sum(1) / cnt
    var = (((x - mean.unsqueeze(1)) ** 2) * m).sum(1) / cnt
    sd = (var + EPS).sqrt()

    def frac(cond):
        return (cond.float() * m).sum(1) / cnt

    level = torch.stack([
        mean / 100.0,
        frac(x < 70),                                  # time below range
        frac((x >= 70) & (x <= 180)),                  # time in range
        frac(x > 180),                                 # time above range L1
        frac(x > 250),                                 # time above range L2
    ], dim=-1)

    mad = ((x - mean.unsqueeze(1)).abs() * m).sum(1) / cnt
    B, L = x.shape
    P, K = 24, L // 24
    xm = (x * m).view(B, P, K).sum(-1) / m.view(B, P, K).sum(-1).clamp(min=1.0)
    hourly_sd = xm.std(dim=1)
    lo = torch.where(m > 0, x, torch.full_like(x, 1e4)).min(1).values
    hi = torch.where(m > 0, x, torch.full_like(x, -1e4)).max(1).values
    disp = torch.stack([
        sd / 50.0, sd / mean.clamp(min=1.0), mad / 50.0,
        hourly_sd / 50.0, (hi - lo).clamp(min=0.0) / 100.0,
    ], dim=-1)

    out = [level, disp]
    if start_idx is not None:
        j = torch.arange(L, device=x.device).unsqueeze(0)
        a = (start_idx.view(-1, 1).to(x.device) + j) % L
        night = (a < L // 4).float() * m                # 00:00-06:00
        day = (a >= L // 4).float() * m
        noct = (x * night).sum(1) / night.sum(1).clamp(min=1.0)
        dayt = (x * day).sum(1) / day.sum(1).clamp(min=1.0)
        # a window with no observed night falls back to the daily mean, not 0
        noct = torch.where(night.sum(1) > 0, noct, mean)
        dayt = torch.where(day.sum(1) > 0, dayt, mean)
        out.append(torch.stack([noct / 100.0, dayt / 100.0], -1))
    return torch.cat(out, dim=-1)


def n_glucometrics(circadian: bool = True) -> int:
    return len(LEVEL_METRICS) + len(DISP_METRICS) + (2 if circadian else 0)


class GlucoFM(nn.Module):
    """Online branch + EMA target branch + the two pretraining heads."""

    def __init__(self, cfg: GlucoFMConfig | None = None):
        super().__init__()
        self.cfg = cfg or GlucoFMConfig()
        self.online = GlucoFMBackbone(self.cfg)

        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

        self.predictor = TransformerStack(self.cfg, 1)
        self.g_state = TransitionHead(self.cfg)
        self.g_event = TransitionHead(self.cfg)
        # Transition heads compare against the target branch's stream tokens, which
        # are taken *before* global self-attention (Sec. 3.3), so no extra projection.
        self.tau_proj = nn.Linear(2, self.cfg.d_model)

        self.cmp_head = None
        if self.cfg.use_cmp:
            k = n_glucometrics(self.cfg.cmp_circadian)
            self.cmp_head = nn.Sequential(
                nn.Linear(self.cfg.d_model, self.cfg.d_model), nn.GELU(),
                nn.Linear(self.cfg.d_model, k))

    # -- EMA -----------------------------------------------------------------

    @torch.no_grad()
    def ema_update(self, m: float) -> None:
        for q, k in zip(self.online.parameters(), self.target.parameters()):
            k.data.mul_(m).add_((1.0 - m) * q.detach().data)
        for q, k in zip(self.online.buffers(), self.target.buffers()):
            k.data.copy_(q.data)

    # -- losses --------------------------------------------------------------

    def forward(self, x, m, start_idx, patch_mask):
        cfg = self.cfg
        out_ctx = self.online(x, m, start_idx, patch_mask=patch_mask)
        with torch.no_grad():
            out_tgt = self.target(x, m, start_idx, patch_mask=None)

        # --- L_MCR (Eq. 2): SmoothL1 at masked patches, weighted by patch density
        z_pred = self.predictor(out_ctx["z"])
        z_tgt = out_tgt["z"]
        per_patch = F.smooth_l1_loss(z_pred, z_tgt, reduction="none").mean(-1)  # (B, P)
        w = out_tgt["density"] * patch_mask.float()
        loss_mcr = (w * per_patch).sum() / (w.sum() + EPS)

        # --- L_TD (Eq. 3-4): next-patch state / event prediction
        tau_emb = self.tau_proj(out_ctx["tau"])
        S, E = out_ctx["state_tok"], out_ctx["event_tok"]
        S_hat = self.g_state(S[:, :-1], E[:, :-1], tau_emb[:, :-1])
        E_hat = self.g_event(E[:, :-1], S[:, :-1], tau_emb[:, :-1])
        S_tgt, E_tgt = out_tgt["state_tok"][:, 1:], out_tgt["event_tok"][:, 1:]

        d = out_tgt["density"]
        q = (1.0 - patch_mask.float())[:, :-1] * d[:, :-1] * d[:, 1:]            # Eq. 16
        ls = (q * F.smooth_l1_loss(S_hat, S_tgt, reduction="none").mean(-1)).sum() / (q.sum() + EPS)
        le = (q * F.smooth_l1_loss(E_hat, E_tgt, reduction="none").mean(-1)).sum() / (q.sum() + EPS)
        loss_td = 0.5 * (ls + le)

        total = cfg.lambda_mcr * loss_mcr + cfg.lambda_td * loss_td

        # --- L_CMP: predict the day's glucometrics from the MASKED context view.
        # Predicting from the context branch (not the full window) keeps this a
        # predictive task in the JEPA sense -- the model must infer the day's
        # level and spread from a partially hidden day.
        loss_cmp = torch.zeros((), device=x.device)
        if self.cmp_head is not None:
            tgt = window_glucometrics(x, m, start_idx if cfg.cmp_circadian else None)
            pred = self.cmp_head(out_ctx["z"].mean(dim=1))
            loss_cmp = F.smooth_l1_loss(pred, tgt)
            total = total + cfg.w_cmp * loss_cmp

        return {"loss": total, "loss_mcr": loss_mcr, "loss_td": loss_td,
                "loss_cmp": loss_cmp.detach(),
                # exposed so a composing model can pool the context tokens without
                # paying for a second encoder pass; ignored by the baseline loop
                "z_ctx": out_ctx["z"],
                "sigma": self.online.filter.sigma.detach()}

    # -- downstream ----------------------------------------------------------

    @torch.no_grad()
    def embed(self, x, m, start_idx) -> torch.Tensor:
        """App. C.6: frozen online branch, global average pooling over the 24 patches."""
        return self.online(x, m, start_idx, patch_mask=None)["z"].mean(dim=1)


def sample_patch_mask(B: int, cfg: GlucoFMConfig, device, generator=None) -> torch.Tensor:
    """Random patch mask with ratio drawn uniformly from [0.5, 0.6] (App. C.5)."""
    ratio = (torch.rand(B, 1, device=device, generator=generator)
             * (cfg.mask_ratio_high - cfg.mask_ratio_low) + cfg.mask_ratio_low)
    n_mask = (ratio * cfg.P).round().long().clamp(1, cfg.P - 1)     # (B, 1)
    order = torch.rand(B, cfg.P, device=device, generator=generator).argsort(dim=1)
    rank = order.argsort(dim=1)
    return rank < n_mask


def glucofm_param_report(cfg: GlucoFMConfig | None = None) -> dict:
    """Reproduce the paper's 0.72M trainable / 1.18M total figures."""
    model = GlucoFM(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    online = sum(p.numel() for p in model.online.parameters())
    return {"trainable": trainable, "total": total, "online_branch": online,
            "target_branch": total - trainable,
            "trainable_M": round(trainable / 1e6, 3), "total_M": round(total / 1e6, 3)}
