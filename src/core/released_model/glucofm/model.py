"""
GlucoFM - dual-stream state/event CGM foundation model (arXiv:2605.30865, section 3).

    GlucoFMEncoder  online branch: decomposition -> dual-stream embed -> fusion
                    -> circadian encoding -> 3-layer Transformer
    GlucoFM         pretraining wrapper: EMA target branch, masked-context
                    predictor, two transition heads, L_MCR + L_TD

Downstream use (Appendix C.6): freeze the online branch, discard the EMA branch
and the prediction heads, mean-pool the P=24 patch tokens.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, DEFAULT
from .modules import (
    EPS,
    CausalGaussianFilter,
    CircadianEncoding,
    ScaleEncoding,
    StreamEmbedder,
    masked_instance_norm,
    patch_stats,
    patchify,
    rate_of_change,
    transformer_encoder,
)


class GlucoFMEncoder(nn.Module):
    """The online (and, copied, the EMA target) branch."""

    def __init__(self, cfg: Config = DEFAULT):
        super().__init__()
        self.cfg = cfg
        g, m, f = cfg.grid, cfg.model, cfg.filt

        self.filter = CausalGaussianFilter(
            f.sigma_init, f.sigma_min, f.sigma_max, f.radius)

        self.state_embed = StreamEmbedder(
            patch_size=g.patch_size, wave_dim=m.state_wave_dim,
            aux_dim=m.state_diff_dim, stat_dim=m.state_stat_dim,
            out_dim=m.stream_dim, aux_len=g.patch_size - 1, dropout=m.dropout)
        self.event_embed = StreamEmbedder(
            patch_size=g.patch_size, wave_dim=m.event_wave_dim,
            aux_dim=m.event_roc_dim, stat_dim=m.event_stat_dim,
            out_dim=m.stream_dim, aux_len=g.patch_size, dropout=m.dropout)

        self.fuse = nn.Sequential(
            nn.Linear(2 * m.stream_dim, m.embed_dim),
            nn.GELU(),
            nn.Linear(m.embed_dim, m.embed_dim),
        )
        self.fuse_norm = nn.LayerNorm(m.embed_dim)
        self.circadian = CircadianEncoding(
            g.n_patches, m.embed_dim, g.length, g.patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, m.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.scale_enc = (ScaleEncoding(m.embed_dim, m.scale_hidden)
                          if getattr(m, "scale_inject", False) else None)
        if getattr(m, "global_norm", False):
            self.global_wave = nn.Linear(g.patch_size, m.embed_dim)
            self.global_gate = nn.Parameter(torch.zeros(1, 1, m.embed_dim))
        else:
            self.global_wave = None

        self.encoder = transformer_encoder(
            m.embed_dim, m.n_heads, m.ffn_dim, m.n_layers, m.dropout)

    # ------------------------------------------------------------------ #
    def streams(self, x: torch.Tensor, m: torch.Tensor):
        """Decompose and embed. Returns (state_tok, event_tok, density, scale).

        x: [B, L] glucose with 0 at unobserved positions
        m: [B, L] observation mask actually visible to this branch

        `scale` is the (mu, sigma) pair that the normalisation removes. GlucoFM
        discards it; keeping it is what `scale_inject` needs.
        """
        g, fc = self.cfg.grid, self.cfg.filt
        xn, mu, sd = masked_instance_norm(x, m)
        x_state = self.filter(xn, m)                    # Eq. 11
        x_event = (xn - x_state) * m                    # Eq. 14

        ps, pm = patchify(x_state, g.n_patches, g.patch_size), patchify(m, g.n_patches, g.patch_size)
        pe = patchify(x_event, g.n_patches, g.patch_size)

        s_mu, s_sd, dens = patch_stats(ps, pm)          # Eqs. 8-9
        s_diff = ps[..., 1:] - ps[..., :-1]             # intra-patch trend difference
        state_tok = self.state_embed(
            ps * pm, s_diff, torch.stack([s_mu, s_sd, dens], dim=-1))

        roc, rmask = rate_of_change(xn, m, fc.max_lookback)   # Eq. 10
        proc, prm = patchify(roc, g.n_patches, g.patch_size), patchify(rmask, g.n_patches, g.patch_size)
        e_mu, e_sd, _ = patch_stats(proc, prm)
        event_tok = self.event_embed(
            pe * pm, proc, torch.stack([e_mu, e_sd, dens], dim=-1))

        return state_tok, event_tok, dens, (mu, sd)

    def fuse_tokens(self, state_tok, event_tok):
        return self.fuse_norm(self.fuse(torch.cat([state_tok, event_tok], dim=-1)))

    def forward(self, x, m, start_idx, patch_mask=None):
        """patch_mask [B, P] with 1 = masked (replaced by the learnable token)."""
        state_tok, event_tok, dens, (mu, sd) = self.streams(x, m)
        tok = self.fuse_tokens(state_tok, event_tok)
        if patch_mask is not None:
            keep = (1.0 - patch_mask).unsqueeze(-1)
            tok = tok * keep + self.mask_token * (1.0 - keep)
        tok = self.circadian(tok, start_idx)
        if self.scale_enc is not None:
            # after masking, so mu/sd are the statistics of what this branch can
            # actually see - the online branch must infer the full-day scale
            # from a partial day, which is the same predictive setup as L_MCR
            tok = self.scale_enc(tok, mu, sd, m)
        if self.global_wave is not None:
            # CGM-JEPA's view: one global affine map, so absolute excursion
            # magnitude survives. Gated, so the model can ignore it.
            mc, gc = self.cfg.model, self.cfg.grid
            xg = ((x - mc.global_mean) / mc.global_std) * m
            gtok = self.global_wave(patchify(xg, gc.n_patches, gc.patch_size))
            tok = tok + torch.sigmoid(self.global_gate) * gtok
        z = self.encoder(tok)
        return z, state_tok, event_tok, dens

    @torch.no_grad()
    def embed(self, x, m, start_idx) -> torch.Tensor:
        """Appendix C.6 - frozen window representation, mean-pooled over patches."""
        z, _, _, _ = self.forward(x, m, start_idx, patch_mask=None)
        return z.mean(dim=1)


# --------------------------------------------------------------------------- #
class TransitionHead(nn.Module):
    """Lightweight next-patch predictor for the temporal dynamics objective."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, t):
        return self.net(t)


class GlucoFM(nn.Module):
    """Pretraining wrapper: online + EMA target branch and both objectives."""

    def __init__(self, cfg: Config = DEFAULT):
        super().__init__()
        self.cfg = cfg
        m, p = cfg.model, cfg.pretrain

        self.online = GlucoFMEncoder(cfg)
        self.target = copy.deepcopy(self.online)
        for q in self.target.parameters():
            q.requires_grad_(False)

        self.predictor = transformer_encoder(
            m.embed_dim, m.n_heads, m.ffn_dim, m.predictor_layers, m.dropout)
        self.pred_proj = nn.Linear(m.embed_dim, m.embed_dim)

        self.state_transition = TransitionHead(m.stream_dim, m.transition_hidden)
        self.event_transition = TransitionHead(m.stream_dim, m.transition_hidden)

        self.ema_m = p.ema_momentum

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_target(self, momentum: float | None = None):
        mm = self.ema_m if momentum is None else momentum
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.mul_(mm).add_(op.detach(), alpha=1.0 - mm)
        for tb, ob in zip(self.target.buffers(), self.online.buffers()):
            tb.copy_(ob)

    # ------------------------------------------------------------------ #
    def forward(self, x, m, start_idx, patch_mask):
        """Returns a dict with the two losses and their sum.

        x, m       : [B, L]  glucose (0 where unobserved) and physical mask
        start_idx  : [B]     circadian start index s
        patch_mask : [B, P]  1 = masked in the online branch
        """
        cfg = self.cfg
        g, pc = cfg.grid, cfg.pretrain
        B, P = patch_mask.shape

        # Masked patches are hidden from the online branch's own statistics and
        # filtering, not merely token-replaced (Appendix C.5).
        vis = (1.0 - patch_mask).repeat_interleave(g.patch_size, dim=1)
        m_online = m * vis

        # d_i is the PHYSICAL observation density (Eq. 15), computed from the full
        # mask - not from the mask the online branch can see. Using the online
        # density would make w_i identically zero on masked patches and silently
        # zero out L_MCR entirely.
        dens = patchify(m, g.n_patches, g.patch_size).mean(dim=-1)

        z_ctx, s_on, e_on, _ = self.online(x, m_online, start_idx, patch_mask)
        z_pred = self.pred_proj(self.predictor(z_ctx))

        with torch.no_grad():
            z_tgt, s_tgt, e_tgt, _ = self.target(x, m, start_idx, patch_mask=None)

        # ---- L_MCR: masked contextual latent prediction (Eq. 15 weighting) ----
        w = dens * patch_mask                                    # w_i = d_i on masked patches
        per_patch = F.smooth_l1_loss(
            z_pred, z_tgt.detach(), beta=pc.smooth_l1_beta, reduction="none").mean(-1)
        loss_mcr = (w * per_patch).sum() / (w.sum() + EPS)

        # ---- L_TD: next-patch state/event prediction (Eq. 16 weighting) ----
        q = (1.0 - patch_mask[:, :-1]) * dens[:, :-1] * dens[:, 1:]
        s_hat = self.state_transition(s_on[:, :-1])
        e_hat = self.event_transition(e_on[:, :-1])
        td_s = F.smooth_l1_loss(s_hat, s_tgt[:, 1:].detach(),
                                beta=pc.smooth_l1_beta, reduction="none").mean(-1)
        td_e = F.smooth_l1_loss(e_hat, e_tgt[:, 1:].detach(),
                                beta=pc.smooth_l1_beta, reduction="none").mean(-1)
        loss_td = (q * (td_s + td_e)).sum() / (2 * q.sum() + EPS)

        total = pc.w_mcr * loss_mcr + pc.w_td * loss_td
        return {"loss": total, "loss_mcr": loss_mcr.detach(),
                "loss_td": loss_td.detach(),
                "sigma": self.online.filter.sigma.detach()}


# --------------------------------------------------------------------------- #
def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def build_model(cfg: Config = DEFAULT, verbose: bool = True) -> GlucoFM:
    """Build GlucoFM and check the parameter budget stated in Appendix C.4."""
    model = GlucoFM(cfg)
    trainable, total = count_params(model)
    if verbose:
        exp_tr, exp_to = cfg.model.expected_trainable, cfg.model.expected_total
        print(f"  GlucoFM parameters: trainable={trainable:,} (paper {exp_tr:,})")
        print(f"                      total    ={total:,} (paper {exp_to:,})")
        tol = cfg.model.param_tolerance
        for name, got, exp in (("trainable", trainable, exp_tr), ("total", total, exp_to)):
            dev = (got - exp) / exp          # signed: negative = under budget
            flag = "OK" if abs(dev) <= tol else "OUT OF BUDGET"
            print(f"                      {name:<9} deviation {dev:+6.1%}  {flag}")
        # The EMA delta is the sharpest check: it isolates the encoder alone.
        delta = total - trainable
        exp_delta = exp_to - exp_tr
        print(f"                      EMA branch {delta:,} (paper implies {exp_delta:,}, "
              f"{(delta - exp_delta) / exp_delta:+.1%})")
    return model
