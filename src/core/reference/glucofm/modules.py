"""
GlucoFM building blocks: mask-aware normalisation, the causal Gaussian filter,
patch statistics and the two stream embedders.

All operations are mask-aware: unobserved grid positions never contribute to a
statistic, a filter output or a loss term (Appendix C.2).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# --------------------------------------------------------------------------- #
def masked_instance_norm(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-window normalisation over observed positions only.

    x, m: [B, L].  Returns (x_norm, mu, sigma); unobserved positions are zeroed.

    Forced to float32: this runs on RAW glucose (20-600 mg/dL), so the sum of
    squared deviations reaches ~2e7 over 288 positions, which overflows float16's
    65504 ceiling under autocast. That produced inf gradients, a GradScaler that
    skipped every step, and a perfectly flat training loss.
    """
    dtype = x.dtype
    x32, m32 = x.float(), m.float()
    cnt = m32.sum(dim=1, keepdim=True).clamp(min=1.0)
    mu = (x32 * m32).sum(dim=1, keepdim=True) / cnt
    var = (((x32 - mu) ** 2) * m32).sum(dim=1, keepdim=True) / cnt
    sd = torch.sqrt(var + EPS)
    xn = ((x32 - mu) / sd) * m32
    return xn.to(dtype), mu.squeeze(1).to(dtype), sd.squeeze(1).to(dtype)


class CausalGaussianFilter(nn.Module):
    """Appendix C.3, Eqs. 11-13.

    state_j = sum_{r=0..R} K_s(r) M_{j-r} X_{j-r} / sum_{r=0..R} K_s(r) M_{j-r}

    One-sided (causal) kernel, so no future value informs the current state.
    Bandwidth is learnable and squashed into [sigma_min, sigma_max].
    """

    def __init__(self, sigma_init: float = 6.0, sigma_min: float = 2.0,
                 sigma_max: float = 12.0, radius: int = 36):
        super().__init__()
        self.sigma_min, self.sigma_max, self.radius = sigma_min, sigma_max, radius
        # invert sigma = min + (max-min)*sigmoid(rho) at sigma_init
        frac = (sigma_init - sigma_min) / (sigma_max - sigma_min)
        frac = min(max(frac, 1e-4), 1 - 1e-4)
        self.rho = nn.Parameter(torch.tensor(math.log(frac / (1 - frac)), dtype=torch.float32))
        self.register_buffer("lags", torch.arange(radius + 1, dtype=torch.float32))

    @property
    def sigma(self) -> torch.Tensor:
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(self.rho)

    def kernel(self) -> torch.Tensor:
        s = self.sigma
        k = torch.exp(-(self.lags ** 2) / (2 * s * s + EPS))
        return k / (k.sum() + EPS)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """x, m: [B, L] -> state [B, L]."""
        B, L = x.shape
        k = self.kernel()                    # [R+1], index r = lag
        # conv1d computes sum_t w[t] * in[j + t]; with left padding R and
        # w[t] = K(R - t) this evaluates sum_r K(r) * in[j - r].
        w = k.flip(0).view(1, 1, -1)
        xm = (x * m).unsqueeze(1)
        mm = m.unsqueeze(1)
        pad = (self.radius, 0)
        num = F.conv1d(F.pad(xm, pad), w).squeeze(1)
        den = F.conv1d(F.pad(mm, pad), w).squeeze(1)
        return num / (den + EPS)


# --------------------------------------------------------------------------- #
def patchify(t: torch.Tensor, n_patches: int, patch_size: int) -> torch.Tensor:
    """[B, L] -> [B, P, K]."""
    return t.reshape(t.shape[0], n_patches, patch_size)


def patch_stats(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Appendix C.2, Eqs. 8-9.  x, m: [B, P, K] -> (mu, sd, density) each [B, P]."""
    cnt = m.sum(-1)
    mu = (x * m).sum(-1) / (cnt + EPS)
    var = (((x - mu.unsqueeze(-1)) ** 2) * m).sum(-1) / (cnt + EPS)
    sd = torch.sqrt(var + EPS)
    dens = cnt / m.shape[-1]
    valid = (cnt > 0).float()
    return mu * valid, sd * valid, dens


def rate_of_change(x: torch.Tensor, m: torch.Tensor, max_lookback: int = 9
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Appendix C.2, Eq. 10.

    r_j = (X_j - X_{j-b}) / b using the CLOSEST previous observed position,
    searching back at most `max_lookback` steps. Positions with no valid
    partner get 0 and are marked invalid.

    x, m: [B, L] -> (roc [B, L], roc_mask [B, L]).
    """
    B, L = x.shape
    roc = torch.zeros_like(x)
    rmask = torch.zeros_like(m)
    found = torch.zeros_like(m)
    for b in range(1, max_lookback + 1):
        prev_x = F.pad(x, (b, 0))[:, :L]
        prev_m = F.pad(m, (b, 0))[:, :L]
        # valid where current observed, previous observed, and nothing closer yet
        cand = m * prev_m * (1.0 - found)
        roc = roc + cand * (x - prev_x) / b
        rmask = rmask + cand
        found = torch.clamp(found + cand, max=1.0)
    return roc * rmask, rmask


# --------------------------------------------------------------------------- #
class StreamEmbedder(nn.Module):
    """Shared shape for the state and event streams (Appendix C.4).

    Each stream turns a patch into three features - a waveform projection, a
    secondary projection (intra-patch differences for state, rate-of-change for
    event) and an MLP over patch-level scalar statistics - then projects the
    concatenation to a `out_dim` stream token.
    """

    def __init__(self, patch_size: int, wave_dim: int, aux_dim: int,
                 stat_dim: int, out_dim: int, aux_len: int, dropout: float = 0.1):
        super().__init__()
        self.wave = nn.Linear(patch_size, wave_dim)
        self.aux = nn.Linear(aux_len, aux_dim)
        self.stat = nn.Sequential(
            nn.Linear(3, stat_dim),   # (mu, sd, density)
            nn.GELU(),
            nn.Linear(stat_dim, stat_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(wave_dim + aux_dim + stat_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, wave: torch.Tensor, aux: torch.Tensor,
                stats: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.wave(wave), self.aux(aux), self.stat(stats)], dim=-1)
        return self.norm(self.proj(h))


class CircadianEncoding(nn.Module):
    """Section 3.2 / Appendix C.4.

    Circular time-of-day features tau(i) = [sin(2*pi*i/L), cos(2*pi*i/L)] from the
    ABSOLUTE grid index are projected to D and combined with a learnable patch
    positional embedding through a learnable gate.
    """

    def __init__(self, n_patches: int, embed_dim: int, length: int, patch_size: int):
        super().__init__()
        self.length, self.patch_size, self.n_patches = length, patch_size, n_patches
        self.time_proj = nn.Linear(2, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.gate = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, tokens: torch.Tensor, start_idx: torch.Tensor) -> torch.Tensor:
        """tokens [B, P, D]; start_idx [B] circadian start s."""
        B = tokens.shape[0]
        dev = tokens.device
        # absolute grid index at the start of each patch: a = (s + p*K) mod L
        p = torch.arange(self.n_patches, device=dev).view(1, -1) * self.patch_size
        a = (start_idx.view(-1, 1).to(dev) + p) % self.length          # [B, P]
        ang = 2 * math.pi * a.float() / self.length
        tau = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)    # [B, P, 2]
        g = torch.sigmoid(self.gate)
        return tokens + g * self.time_proj(tau) + (1 - g) * self.pos.expand(B, -1, -1)


class ScaleEncoding(nn.Module):
    """Re-inject the window scale that masked_instance_norm removes.

    GlucoFM normalises each window to zero mean / unit variance before anything
    else touches it, which makes the representation invariant to exactly the
    quantities the clinical labels are defined on. This module embeds

        [ log(mu / 100), log(sigma + 1), sigma / mu, wear fraction ]

    and adds it to every patch token through a learnable gate, in the same
    additive-gated style as CircadianEncoding. Because it is a per-window (not
    per-patch) offset it survives mean-pooling into the frozen representation.
    """

    def __init__(self, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
        self.gate = nn.Parameter(torch.zeros(1, 1, embed_dim))

    @staticmethod
    def features(mu: torch.Tensor, sd: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """mu, sd: [B]; m: [B, L] -> [B, 4]."""
        mu = mu.clamp(min=1.0)
        sd = sd.clamp(min=EPS)
        return torch.stack([
            torch.log(mu / 100.0), torch.log1p(sd), sd / mu, m.mean(dim=1),
        ], dim=-1)

    def forward(self, tokens: torch.Tensor, mu, sd, m) -> torch.Tensor:
        e = self.net(self.features(mu, sd, m)).unsqueeze(1)      # [B, 1, D]
        return tokens + torch.sigmoid(self.gate) * e


def transformer_encoder(embed_dim: int, n_heads: int, ffn_dim: int,
                        n_layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers,
                                 norm=nn.LayerNorm(embed_dim))
