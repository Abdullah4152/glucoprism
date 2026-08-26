"""GluFormer (Lutsker et al., 2026, Nature) -- decoder-only autoregressive CGM model.

Two variants, both reproduced here:

  base  -- the original paper's configuration: d=1024, 16 heads, 16 layers,
           FF=2048, vocabulary 461 (460 glucose bins over 40-500 mg/dL + <MASK>).
           GlucoFM retrains this on its own corpus with Adam lr 5e-5, 76 epochs.

  tiny  -- the size-matched baseline used by *both* CGM-JEPA and GlucoFM so that
           an objective-vs-objective comparison is not confounded by capacity.
           GlucoFM App. B.3: "a smaller 4-layer Transformer with hidden dimension
           128, 4 attention heads, and feed-forward dimension 256 ... trained with
           Adam at learning rate 1e-4, batch size 128, for 100 epochs. Downstream
           embeddings are extracted from frozen hidden states over valid
           non-padding tokens using max pooling by default."

`tiny` is the variant this project reproduces for the GlucoPRISM baseline table.

Sources reconciled:
  * Nature Methods section -- tokenization, vocabulary, causal masking,
    cross-entropy next-token objective, max-pooling readout.
  * GlucoFM App. B.3 -- retraining recipe, tiny dimensions, [40, 500] clipping.
  * github.com/cruiseresearchgroup/CGM-JEPA `models/gluformer/gluformer.py` --
    reference PyTorch layout for the size-matched baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GLUCOSE_MIN = 40.0     # FreeStyle Libre Pro / Dexcom lower reporting limit, mg/dL
GLUCOSE_MAX = 500.0    # upper reporting limit, mg/dL
N_BINS = 460           # "460 evenly spaced intervals" (Nature Methods)


@dataclass
class GluFormerConfig:
    vocab_size: int = N_BINS          # real glucose tokens; PAD/<MASK> is index vocab_size
    embed_dim: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    max_seq_length: int = 25000
    dropout: float = 0.1
    pooling: str = "max"              # max pooling won the paper's readout comparison

    @classmethod
    def tiny(cls) -> "GluFormerConfig":
        """GlucoFM App. B.3 size-matched baseline (~0.65M params)."""
        return cls(embed_dim=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1)

    @classmethod
    def base(cls) -> "GluFormerConfig":
        """Original Nature configuration (~135M params as reported by GlucoFM)."""
        return cls(embed_dim=1024, nhead=16, num_layers=16, dim_feedforward=2048, dropout=0.1)


# ------------------------------------------------------------- tokenization

def tokenize_glucose(g: np.ndarray, n_bins: int = N_BINS,
                     lo: float = GLUCOSE_MIN, hi: float = GLUCOSE_MAX) -> np.ndarray:
    """Clip to the device's physiological range, then bin into `n_bins` even
    intervals -- the "shifted glucose tokens" of GlucoFM App. B.3.

    Token id 0 <-> 40 mg/dL, token id 459 <-> 500 mg/dL; `n_bins` itself is
    reserved for PAD/<MASK>.
    """
    g = np.clip(np.asarray(g, dtype=np.float64), lo, hi)
    width = (hi - lo) / n_bins
    idx = np.floor((g - lo) / width).astype(np.int64)
    return np.clip(idx, 0, n_bins - 1)


def detokenize_glucose(tok: np.ndarray, n_bins: int = N_BINS,
                       lo: float = GLUCOSE_MIN, hi: float = GLUCOSE_MAX) -> np.ndarray:
    """Bin centre in mg/dL, for generation / sanity checks."""
    width = (hi - lo) / n_bins
    return lo + (np.asarray(tok, dtype=np.float64) + 0.5) * width


# -------------------------------------------------------------------- model

class GluFormer(nn.Module):
    def __init__(self, cfg: GluFormerConfig | None = None):
        super().__init__()
        self.cfg = cfg or GluFormerConfig.tiny()
        c = self.cfg
        self.pad_token = c.vocab_size

        # +1 row for the PAD / <MASK> token (Nature: "vocabulary size of the model was 461").
        self.token_embedding = nn.Embedding(c.vocab_size + 1, c.embed_dim)
        self.register_buffer("pos_embedding",
                             self._sinusoidal(c.max_seq_length, c.embed_dim),
                             persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=c.embed_dim, nhead=c.nhead, dim_feedforward=c.dim_feedforward,
            dropout=c.dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=c.num_layers)
        self.output_head = nn.Linear(c.embed_dim, c.vocab_size)

    @staticmethod
    def _sinusoidal(max_len: int, dim: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe

    def _hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        T = tokens.size(1)
        x = self.token_embedding(tokens) + self.pos_embedding[:T].unsqueeze(0).to(tokens.device)
        causal = torch.triu(torch.ones(T, T, device=tokens.device, dtype=torch.bool), diagonal=1)
        return self.transformer(x, mask=causal)

    def forward(self, tokens: torch.Tensor, return_hidden: bool = False):
        """tokens: (B, T) int64. Returns next-token logits (B, T, vocab_size)."""
        h = self._hidden(tokens)
        return h if return_hidden else self.output_head(h)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Causal next-token cross-entropy; PAD positions excluded (Nature Methods)."""
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        logits = self.forward(inputs)
        return F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size),
                               targets.reshape(-1), ignore_index=self.pad_token)

    @torch.no_grad()
    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Frozen readout: pool hidden states over valid (non-PAD) positions.

        The Nature paper compared mean / max / min pooling and adopted max; GlucoFM
        App. B.3 keeps max as the default for the retrained variants.
        """
        h = self._hidden(tokens)                                  # (B, T, D)
        valid = (tokens != self.pad_token).unsqueeze(-1)          # (B, T, 1)
        # An all-PAD row would make max/min pooling return +-inf; fall back to
        # position 0 so the probe never sees a non-finite feature.
        valid = valid | (~valid).all(dim=1, keepdim=True) & (
            torch.arange(h.size(1), device=h.device).view(1, -1, 1) == 0)
        if self.cfg.pooling == "max":
            return h.masked_fill(~valid, torch.finfo(h.dtype).min).max(dim=1).values
        if self.cfg.pooling == "mean":
            return (h * valid).sum(1) / valid.sum(1).clamp_min(1)
        if self.cfg.pooling == "min":
            return h.masked_fill(~valid, torch.finfo(h.dtype).max).min(dim=1).values
        raise ValueError(f"unknown pooling {self.cfg.pooling!r}")


def gluformer_param_report(cfg: GluFormerConfig | None = None) -> dict:
    m = GluFormer(cfg)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {"trainable": n, "trainable_M": round(n / 1e6, 3)}
