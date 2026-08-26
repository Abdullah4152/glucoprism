"""Clinical Query Pooling (CQP) -- prediction-based factorization for CGM.

MOTIVATION, from this repo's own measurements rather than from the literature.

GlucoPRISM factorizes a CGM representation by *invariance*: make `zT` invariant to
day and sensor, keep `zS` day-discriminative, decorrelate the blocks. Implemented
faithfully, all three objectives converge to their targets and the factorization
still does not appear -- no block beats a dimension-matched random projection
(`results_final.md` §3), and the trait block ends up ANTI-correlated with subject
identity (§5).

The reason is identifiability. "Make `zT` invariant to day" has an enormous
solution set and essentially none of its members are "`zT` = the subject's
metabolic trait". Nothing in the objective selects the intended one.

CQP replaces invariance with **prediction against targets that have exactly one
right answer**. "What was this day's time-in-range" is a deterministic function of
the window -- computable, label-free, and unambiguous. A representation slot
trained to predict it cannot drift onto some other equally-valid solution, because
there isn't one. Identifiability comes from the target, not from a penalty.

ARCHITECTURE

    24 fused patch tokens (24 x 128)          <- GlucoFM backbone, unchanged
      -> M learned clinical queries cross-attend over them
      -> per-query projection to d_q
      -> concat = M * d_q = 128-d representation      (same width as GlucoFM)
      -> query k predicts glucometric k                (label-free supervision)

Two design constraints, both forced by measurements in this repo:

* **The representation stays 128-d.** Widening the readout was measured directly
  and it HURTS at window level: [mean|sd] 256-d costs -2.8 PR, [mean|sd|max] 384-d
  costs -2.9, because at 255-929 windows with a fixed-C probe a wider input
  overfits. So dispersion has to be captured *within* 128 dims, not appended to
  them. Attention pooling does that; concatenation does not.

* **Mean pooling is the thing being replaced.** GlucoFM App. C.6 mean-pools the 24
  hourly tokens, which annihilates their within-day dispersion -- precisely the
  quantity half the downstream labels are defined on (glucotype IS a variability
  class). A query that attends can read dispersion; a mean cannot.

The first `len(GLUCOMETRIC_NAMES)` queries are supervised, one per metric. The
remainder are free capacity with no target, so the model is not forced to spend
its whole representation on hand-chosen statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgmkit.models.glucofm import (EPS, GlucoFM, GlucoFMConfig, n_glucometrics,
                      window_glucometrics)

GLUCOMETRIC_NAMES = ("mean", "tbr70", "tir", "tar180", "tar250",
                     "sd", "cv", "mad", "hourly_sd", "range",
                     "nocturnal", "daytime")


@dataclass
class CQPConfig:
    fm: GlucoFMConfig = field(default_factory=GlucoFMConfig)

    n_queries: int = 16          # 12 supervised + 4 free
    d_query: int = 8             # n_queries * d_query must equal fm.d_model
    n_heads: int = 4
    cmp_circadian: bool = True

    w_cmp: float = 1.0           # weight on the per-query metric prediction
    # Predict the metrics from the MASKED context view, so this stays a predictive
    # task in the JEPA sense rather than a reconstruction of the visible signal.
    cmp_from_context: bool = True

    def __post_init__(self):
        if self.n_queries * self.d_query != self.fm.d_model:
            raise ValueError(
                f"n_queries*d_query = {self.n_queries * self.d_query} must equal "
                f"d_model = {self.fm.d_model}; the representation width is held "
                f"fixed because widening it was measured to hurt")
        if self.n_supervised > self.n_queries:
            raise ValueError("more metrics than queries")

    @property
    def n_supervised(self) -> int:
        return n_glucometrics(self.cmp_circadian)

    def to_dict(self) -> dict:
        return asdict(self)


class ClinicalQueryPool(nn.Module):
    """M learned queries cross-attending over the patch sequence.

    Replaces `z = mean_p(tokens)` with `z = concat_m proj(attn(q_m, tokens))`.
    Each query is free to attend to whatever part of the day answers its question:
    a fasting query can look at the nocturnal patches, a variability query can
    spread its attention and read the spread.
    """

    def __init__(self, cfg: CQPConfig):
        super().__init__()
        d = cfg.fm.d_model
        self.cfg = cfg
        self.queries = nn.Parameter(torch.randn(1, cfg.n_queries, d) * 0.02)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.proj = nn.Linear(d, cfg.d_query)

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        """(B, P, D) -> (B, M, d_q) per-query codes, and the flat (B, D) readout."""
        B = tokens.shape[0]
        q = self.norm_q(self.queries.expand(B, -1, -1))
        kv = self.norm_kv(tokens)
        out, w = self.attn(q, kv, kv, need_weights=return_attn,
                           average_attn_weights=True)
        codes = self.proj(out)                                  # (B, M, d_q)
        flat = codes.reshape(B, -1)                             # (B, M*d_q) = (B, D)
        return (codes, flat, w) if return_attn else (codes, flat)


class GlucoCQP(nn.Module):
    """GlucoFM backbone + clinical query pooling + per-query metric prediction."""

    def __init__(self, cfg: CQPConfig | None = None):
        super().__init__()
        self.cfg = cfg or CQPConfig()
        # Composed, never forked: GlucoFM's Eq. 9/10 handling (patch statistics on
        # the ALIGNED mg/dL sequence) is the most load-bearing detail in this repo.
        self.fm = GlucoFM(self.cfg.fm)
        self.pool = ClinicalQueryPool(self.cfg)
        # One scalar read-out per supervised query: query k answers metric k, and
        # nothing else. Sharing a head across queries would let the model satisfy
        # L_CMP without the queries specialising, which is the whole point.
        k, dq = self.cfg.n_supervised, self.cfg.d_query
        self.metric_w = nn.Parameter(torch.zeros(k, dq))
        self.metric_b = nn.Parameter(torch.zeros(k))
        nn.init.trunc_normal_(self.metric_w, std=0.02)

    def ema_update(self, m: float) -> None:
        self.fm.ema_update(m)

    def _predict_metrics(self, codes: torch.Tensor) -> torch.Tensor:
        k = self.cfg.n_supervised
        return (codes[:, :k] * self.metric_w).sum(-1) + self.metric_b   # (B, k)

    @torch.no_grad()
    def embed(self, x, m, start_idx) -> torch.Tensor:
        """Frozen readout: the concatenated query codes, 128-d, unmasked view."""
        tokens = self.fm.online(x, m, start_idx, patch_mask=None)["z"]
        _codes, flat = self.pool(tokens)
        return flat

    @torch.no_grad()
    def attention(self, x, m, start_idx):
        """(B, M, P) query->patch attention, for the interpretability figure."""
        tokens = self.fm.online(x, m, start_idx, patch_mask=None)["z"]
        _c, _f, w = self.pool(tokens, return_attn=True)
        return w

    def forward(self, x, m, start_idx, patch_mask) -> dict:
        cfg = self.cfg
        fm_out = self.fm(x, m, start_idx, patch_mask)
        loss_mcr, loss_td = fm_out["loss_mcr"], fm_out["loss_td"]

        if cfg.cmp_from_context:
            tokens = fm_out["z_ctx"]              # masked view, no extra forward
        else:
            tokens = self.fm.online(x, m, start_idx, patch_mask=None)["z"]

        codes, _flat = self.pool(tokens)
        tgt = window_glucometrics(x, m, start_idx if cfg.cmp_circadian else None)
        loss_cmp = F.smooth_l1_loss(self._predict_metrics(codes), tgt)

        total = (cfg.fm.lambda_mcr * loss_mcr + cfg.fm.lambda_td * loss_td
                 + cfg.w_cmp * loss_cmp)

        # collapse monitor: a query that has stopped discriminating shows up here
        with torch.no_grad():
            f = F.normalize(_flat.float(), dim=-1)
            c = f @ f.t()
            off = c[~torch.eye(len(c), dtype=torch.bool, device=c.device)]
        return {"loss": total, "loss_mcr": loss_mcr, "loss_td": loss_td,
                "loss_cmp": loss_cmp.detach(),
                "cos_z": off.mean().detach(),
                "std_z": _flat.float().std(0).mean().detach(),
                "sigma": fm_out["sigma"]}


def cqp_param_report(cfg: CQPConfig | None = None) -> dict:
    model = GlucoCQP(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    fm = sum(p.numel() for p in model.fm.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total,
            "glucofm_part": fm, "cqp_added": trainable - fm,
            "trainable_M": round(trainable / 1e6, 3)}
