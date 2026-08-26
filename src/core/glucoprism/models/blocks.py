"""Transformer primitives shared by the reproduced CGM encoders.

These match the reference CGM-JEPA implementation
(`cruiseresearchgroup/CGM-JEPA`, `utils/modules.py` + `utils/embed.py`)
term for term: pre-norm blocks, GELU MLP with `mlp_ratio` expansion,
non-flash scaled-dot-product attention, and sinusoidal position tables.
Keeping them byte-compatible is what lets us load the released
`model.safetensors` checkpoints into our own code for equivalence checks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x)), attn


class Block(nn.Module):
    """Pre-norm Transformer block (norm -> attn -> residual -> norm -> mlp -> residual)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        return x + self.mlp(self.norm2(x))


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal table, registered as a buffer (never trained)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pos_emb = torch.zeros(max_len, d_model).float()
        pos_emb.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pos_emb.unsqueeze(0))

    def forward(self, x_len: int):
        return self.pos_emb[:, :x_len]


def apply_mask(x: torch.Tensor, masks) -> torch.Tensor:
    """Gather the patch positions listed in `masks` (list of (B, K) index tensors)."""
    out = []
    for m in masks:
        keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        out.append(torch.gather(x, dim=1, index=keep))
    return torch.cat(out, dim=0)


def init_weights(m: nn.Module) -> None:
    """Truncated-normal linear init, as used by the reference JEPA codebases."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
