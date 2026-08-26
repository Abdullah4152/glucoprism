"""CGM-JEPA and X-CGM-JEPA (Muhammad et al., 2026, arXiv:2605.00933).

Reproduced from the paper's Methods (M.3, M.5) and cross-checked against the
official release at github.com/cruiseresearchgroup/CGM-JEPA. Parameter names and
module nesting are kept identical to the release so that the published
`model.safetensors` weights load into this file with `strict=True`
(see `scripts/verify_cgm_jepa.py`).

Architecture defaults (paper Table ED.1):
    patch embedder   patch size 12 / conv kernel 3 / bias True
    context encoder  embed 96 / 6 heads / 3 layers / dropout 0.0
    target encoder   EMA momentum 0.997
    predictor        embed 48 / 2 heads / 1 layer
    glucodensity     32x32 KDE grid, spatial patch 8 -> 16 tokens
    window           288 points -> P = 24 hourly patches
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cgmkit.models.blocks import MLP, Block, PositionalEmbedding, apply_mask


# ----------------------------------------------------------------- embedding

class ValueEmbedding(nn.Module):
    """Per-patch Conv1d embedder: each hourly patch is encoded independently.

    The reference implementation runs a stride==kernel_size Conv1d *inside* a
    patch (so exactly one conv position per patch of length `patch_size` when
    kernel_size == patch_size, and `(patch_size - k)//k + 1` positions otherwise),
    flattens, then projects to `dim`. Kernel size 3 over a 12-step patch gives
    4 conv positions -> Linear(4*dim, dim).
    """

    def __init__(self, dim: int, in_channels: int, patch_size: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Conv1d(1, dim, kernel_size=patch_size, stride=patch_size,
                              bias=bias, padding=0)
        conv_output_length = (in_channels - patch_size) // patch_size + 1
        self.fc = nn.Linear(dim * conv_output_length, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_patches, patch_length)
        B, N, L = x.shape
        x = x.reshape(-1, 1, L)          # (B*N, 1, L)
        x = self.proj(x)                 # (B*N, dim, L')
        x = x.reshape(x.size(0), -1)     # (B*N, dim*L')
        x = self.fc(x)                   # (B*N, dim)
        return x.view(B, N, -1)


class TimeFeatureEmbedding(nn.Module):
    """Informer-style calendar features, averaged within a patch. Off by default."""

    def __init__(self, d_model: int, d_inp: int):
        super().__init__()
        self.proj = nn.Linear(d_inp, d_model)

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        B, N, patch_size, d_inp = x_mark.shape
        x = self.proj(x_mark.reshape(-1, d_inp).float())
        return x.view(B, N, patch_size, -1).mean(dim=2)


class DataEmbedding(nn.Module):
    def __init__(self, dim: int, in_channels: int, patch_size: int,
                 time_inp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = ValueEmbedding(dim, in_channels, patch_size)
        self.positional_embedding = PositionalEmbedding(dim)
        self.timefeature_embedding = TimeFeatureEmbedding(dim, time_inp_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        val = self.value_embedding(x)
        out = val + self.positional_embedding(val.size(1))
        if x_mark is not None and not torch.allclose(x_mark, torch.zeros_like(x_mark)):
            out = out + self.timefeature_embedding(x_mark)
        return self.dropout(out)


# ------------------------------------------------------------------ encoders

class Encoder(nn.Module):
    """Context / EMA-target encoder over hourly CGM patch tokens."""

    def __init__(self, dim_in: int, kernel_size: int, embed_dim: int, embed_bias: bool,
                 nhead: int, num_layers: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale=None, drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 norm_layer=nn.LayerNorm, jepa: bool = False,
                 embed_activation=nn.GELU(), time_inp_dim: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.jepa = jepa
        self.activation = embed_activation or nn.GELU()

        self.data_embedding = DataEmbedding(dim=embed_dim, in_channels=dim_in,
                                            patch_size=kernel_size,
                                            time_inp_dim=time_inp_dim, dropout=drop_rate)
        self.predictor_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=nhead, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                  act_layer=nn.GELU, norm_layer=norm_layer)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        # Kept for checkpoint compatibility; unused downstream (embeddings are the
        # mean-pooled `encoder_norm` outputs, per paper M.5.3 step 3).
        self.proj = MLP(in_features=embed_dim, hidden_features=1024,
                        out_features=48, act_layer=nn.GELU)

    def forward(self, x, x_mark=None, mask=None):
        if x_mark is None:
            x_mark = torch.zeros_like(x)
        x_mark = x_mark.to(x.device)

        x = self.data_embedding(x, x_mark)
        if mask is not None and self.jepa:
            x = apply_mask(x, mask)
        for blk in self.predictor_blocks:
            x = blk(x, mask=None)
        x = self.encoder_norm(x)
        return x, self.proj(x)

    @torch.no_grad()
    def embed(self, x, x_mark=None):
        """Frozen-encoder feature used by the linear probe: mean over patch tokens."""
        tokens, _ = self.forward(x, x_mark, mask=None)
        return tokens.mean(dim=1)


class SpatialPatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        # x: (B, num_patches, p, p, C)
        B, N, ph, pw, C = x.shape
        assert ph == pw == self.patch_size, f"expected {self.patch_size}^2 patches, got {ph}x{pw}"
        return self.proj(x.reshape(B, N, ph * pw * C))


class GlucodensitySpatialEncoder(nn.Module):
    """ViT-style encoder over the 3-channel 32x32 Glucodensity image (16 tokens)."""

    def __init__(self, num_gluco_patches: int, gluco_patch_size: int, gluco_in_channels: int,
                 embed_dim: int, embed_bias: bool, nhead: int, num_layers: int,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, qk_scale=None,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 norm_layer=nn.LayerNorm, jepa: bool = False, embed_activation=nn.GELU()):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_gluco_patches = num_gluco_patches
        self.jepa = jepa

        self.gluco_embedding = SpatialPatchEmbedding(gluco_patch_size, gluco_in_channels, embed_dim)
        self.gluco_pos_embed = PositionalEmbedding(embed_dim)
        self.encoder_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=nhead, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                  act_layer=nn.GELU, norm_layer=norm_layer)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.proj = MLP(in_features=embed_dim, hidden_features=1024,
                        out_features=48, act_layer=nn.GELU)

    def forward(self, gluco_patches, gluco_mask=None):
        x = self.gluco_embedding(gluco_patches)
        x = x + self.gluco_pos_embed(x.size(1))
        if self.jepa and gluco_mask is not None:
            x = apply_mask(x, [gluco_mask])
        for blk in self.encoder_blocks:
            x = blk(x, mask=None)
        x = self.encoder_norm(x)
        return x, self.proj(x)


# ---------------------------------------------------------------- predictors

class Predictor(nn.Module):
    """Maps context tokens -> latent predictions at the masked patch positions."""

    def __init__(self, encoder_embed_dim: int = 96, predictor_embed_dim: int = 48,
                 nhead: int = 2, num_layers: int = 1, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, qk_scale=None, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, norm_layer=nn.LayerNorm,
                 embed_activation=nn.GELU(), time_inp_dim: int = 5):
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim
        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim, bias=True)
        self.pos_embed = PositionalEmbedding(predictor_embed_dim)
        self.time_embed = TimeFeatureEmbedding(predictor_embed_dim, time_inp_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=nhead, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, act_layer=nn.GELU, norm_layer=norm_layer)
            for _ in range(num_layers)
        ])
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim, bias=True)

    def forward(self, encoded_vals, x_mark, masks=None, non_masks=None):
        assert masks is not None and encoded_vals is not None, "No input found"
        B, L_full, _patch, _t = x_mark.size()
        _, L_ctx, _ = encoded_vals.size()

        x = self.predictor_embed(encoded_vals)
        pos_full = self.pos_embed(L_full).repeat(B, 1, 1)
        if x_mark is not None and not torch.allclose(x_mark, torch.zeros_like(x_mark)):
            pos_full = pos_full + self.time_embed(x_mark)

        x = x + apply_mask(pos_full, non_masks)
        tgt_pos = apply_mask(pos_full, masks)
        pred_tokens = self.mask_token.repeat(B, tgt_pos.size(1), 1) + tgt_pos

        x = torch.cat([x, pred_tokens], dim=1)
        for blk in self.predictor_blocks:
            x = blk(x, mask=None)
        x = self.predictor_norm(x)[:, L_ctx:]
        return self.predictor_proj(x)


class CrossModalPredictor(nn.Module):
    """X-CGM-JEPA cross-view head: CGM context tokens -> masked Glucodensity latents."""

    def __init__(self, num_gluco_patches: int, encoder_embed_dim: int = 96,
                 predictor_embed_dim: int = 48, nhead: int = 2, num_layers: int = 1,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, qk_scale=None,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 norm_layer=nn.LayerNorm, embed_activation=nn.GELU()):
        super().__init__()
        self.num_gluco_patches = num_gluco_patches
        self.predictor_embed_dim = predictor_embed_dim
        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim, bias=True)
        self.pos_embed = PositionalEmbedding(predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=nhead, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, act_layer=nn.GELU, norm_layer=norm_layer)
            for _ in range(num_layers)
        ])
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim, bias=True)

    def forward(self, cgm_context, gluco_masks=None, gluco_non_masks=None):
        assert gluco_masks is not None and cgm_context is not None, "No input found"
        B, L_cgm, _ = cgm_context.size()
        L_gluco = self.num_gluco_patches

        x = self.predictor_embed(cgm_context)
        pred_tokens = self.mask_token.repeat(B, L_gluco, 1) + self.pos_embed(L_gluco).repeat(B, 1, 1)

        x = torch.cat([x, pred_tokens], dim=1)
        for blk in self.predictor_blocks:
            x = blk(x, mask=None)
        x = self.predictor_norm(x)[:, L_cgm:]

        Dp = x.size(-1)
        x = torch.gather(x, dim=1, index=gluco_masks.unsqueeze(-1).repeat(1, 1, Dp))
        return self.predictor_proj(x)


# --------------------------------------------------------------------- loss

def jepa_l1_loss(pred, target):
    """L1 (MAE) between predicted and stop-gradient target latents -- paper Eq. L_CGM."""
    if isinstance(pred, list) and isinstance(target, list):
        return sum(torch.mean(torch.abs(p - t)) for p, t in zip(pred, target)) / len(pred)
    return torch.mean(torch.abs(pred - target))


def build_cgm_jepa(cfg: dict):
    """Instantiate the (context encoder, EMA target encoder, predictor) triple."""
    import copy

    ctx = Encoder(
        dim_in=cfg["patch_size"], kernel_size=cfg["encoder_kernel_size"],
        embed_dim=cfg["encoder_embed_dim"], embed_bias=cfg["encoder_embed_bias"],
        nhead=cfg["encoder_nhead"], num_layers=cfg["encoder_num_layers"],
        jepa=True, time_inp_dim=cfg["time_inp_dim"], drop_rate=cfg["encoder_dropout"],
    )
    predictor = Predictor(
        encoder_embed_dim=cfg["encoder_embed_dim"], predictor_embed_dim=cfg["predictor_embed"],
        nhead=cfg["predictor_nhead"], num_layers=cfg["predictor_num_layers"],
        time_inp_dim=cfg["time_inp_dim"],
    )
    target = copy.deepcopy(ctx)
    for p in target.parameters():
        p.requires_grad = False
    return ctx, target, predictor


def build_x_cgm_jepa(cfg: dict):
    """CGM-JEPA plus the Glucodensity encoder and cross-view predictor."""
    ctx, target, predictor = build_cgm_jepa(cfg)
    n_side = cfg["gluco_gridsize"] // cfg["gluco_spatial_patch_size"]
    num_gluco_patches = n_side * n_side

    gluco_encoder = GlucodensitySpatialEncoder(
        num_gluco_patches=num_gluco_patches,
        gluco_patch_size=cfg["gluco_spatial_patch_size"],
        gluco_in_channels=3,
        embed_dim=cfg["encoder_embed_dim"], embed_bias=cfg["encoder_embed_bias"],
        nhead=cfg["encoder_nhead"], num_layers=cfg["encoder_num_layers"],
        jepa=True, drop_rate=cfg["encoder_dropout"],
    )
    cross_predictor = CrossModalPredictor(
        num_gluco_patches=num_gluco_patches,
        encoder_embed_dim=cfg["encoder_embed_dim"],
        predictor_embed_dim=cfg["predictor_embed"],
        nhead=cfg["predictor_nhead"], num_layers=cfg["predictor_num_layers"],
    )
    return ctx, target, predictor, gluco_encoder, cross_predictor
