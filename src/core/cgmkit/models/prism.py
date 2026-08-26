"""GlucoPRISM: protocol-supervised factorization of Trait, State and Sensor.

Implements `papers/Glucoprism.pdf` Sec. 4 on top of the reproduced GlucoFM
backbone. Module flow follows the proposal's Sec. 10 module list exactly:

    1. Chronological grid alignment + observation mask M      288 x 1
    2. Learnable causal mask-aware Gaussian filter (sigma)    state/event split
    3. State stream / Event stream encoders                   64-d each
    4. Fused physiological patch tokens + circadian encoding  24 x 128
    5. 3-layer Transformer context encoder / EMA target       24 x 128
    ---- steps 1-5 are GlucoFM, imported unchanged ----
    6. Blocked pooling -> [zT (64) | zS (48) | zA (16)]       NOVEL
    7. Heads: L_sensor (V1), L_day (V2), L_indep, + GlucoFM's L_MCR + L_TD
    8. Set aggregator phi over {zS^(k)}, mean over {zT^(k)}   NOVEL (post-hoc,
       see `aggregate.py`; not trained here by design decision D11)

Objectives (proposal Eqs. 2-5):

    L_sensor = E_V1[ D(zT,zT') + D(zS,zS') ] + CE( hA(zA), device_id )      (2)
    L_day    = E_V2[ D(zT,zT'') ] - beta * I( zS ; day_id | subject )       (3)
    L_indep  = || offdiag( Corr([zT; zS; zA]) ) ||_F^2                      (4)
    L        = L_MCR + L_TD + l1*L_sensor + l2*L_day + l3*L_indep           (5)

The backbone is *composed*, never forked: GlucoFM's Eq. 9/10 handling (patch
statistics over the aligned mg/dL sequence X-hat, not the normalised X-tilde) is
the most load-bearing detail in this repo and a fork would silently drift from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .glucofm import EPS, GlucoFM, GlucoFMConfig

# Device vocabulary for the L_CE term of Eq. 2. The synthetic V1 partner is its
# own class: the head must be able to tell the simulated sensor from a real one,
# which is exactly what makes zA sensor-discriminative.
DEVICES = ("dexcom", "libre", "ipro", "synthetic")
DEVICE_ID = {d: i for i, d in enumerate(DEVICES)}


# ------------------------------------------------------------------- config

@dataclass
class PrismConfig:
    fm: GlucoFMConfig = field(default_factory=GlucoFMConfig)

    # --- blocked pooling (Sec. 4.1). Must sum to fm.d_model; swept in E9. ---
    d_trait: int = 64
    d_state: int = 48
    d_sensor: int = 16

    # --- projection heads: "lightweight", one per block (Sec. 4.1) ---
    # 128 puts the three heads + device classifier at ~67k parameters, which
    # with the post-hoc set aggregator (~50k) lands near the proposal's stated
    # ~0.9M against GlucoFM's 0.72M. `prism_param_report()` prints the split so
    # the gap is visible rather than assumed.
    d_head: int = 128

    # D9: the proposal's Sec. 4.1 says heads exist "for the objectives", but
    # Eqs. 2-3 write D(zT, zT') on the bare block. Settled by sweep, not by
    # convention. `L_indep` always reads raw blocks; `L_CE` always uses hA.
    align_on: str = "head"          # {"head", "block"}

    # --- objective weights (Eq. 5); swept on the held-out pretraining split ---
    lambda_sensor: float = 1.0
    lambda_day: float = 1.0
    lambda_indep: float = 1.0
    beta_day: float = 1.0           # weight on -I(zS; day | subject)
    infonce_temp: float = 0.1

    # D10: auxiliary CE is the default, exactly as Eq. 2 writes it.
    # Gradient reversal is built but off; it is an E9 ablation.
    adversarial_sensor: bool = False
    adversarial_lambda: float = 1.0

    n_devices: int = len(DEVICES)

    def __post_init__(self):
        total = self.d_trait + self.d_state + self.d_sensor
        if total != self.fm.d_model:
            raise ValueError(
                f"blocked pooling must partition the token: "
                f"dT+dS+dA = {total} != d_model = {self.fm.d_model}")
        if self.align_on not in ("head", "block"):
            raise ValueError(f"align_on must be 'head' or 'block', got {self.align_on!r}")

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------- components

class ProjectionHead(nn.Module):
    """Lightweight per-block head used by the objectives only (Sec. 4.1).

    Downstream probing always reads the *pre-projection* block. Reading the
    projection instead is the standard way to make a healthy block look
    collapsed, because the head is free to discard whatever the objective
    penalises.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))

    def forward(self, x):
        return self.net(x)


class GradientReversal(torch.autograd.Function):
    """Scales the gradient by -lambda on the way back. E9 ablation only."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd: float = 1.0):
    return GradientReversal.apply(x, lambd)


def blocked_pool(z_tokens: torch.Tensor, cfg: PrismConfig):
    """Step 6. (B, P, D) patch tokens -> (zT, zS, zA).

    Mean-pool over the P patches exactly as GlucoFM's downstream readout does
    (App. C.6), then partition the pooled vector. The encoder is shared, so the
    factorization is a property of the representation rather than of three
    separate networks (Sec. 4.1).
    """
    z = z_tokens.mean(dim=1)                                    # (B, D)
    a, b = cfg.d_trait, cfg.d_trait + cfg.d_state
    return z[:, :a], z[:, a:b], z[:, b:]


def align_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """`D` in Eqs. 2-3: normalized SmoothL1 / cosine alignment.

    Both halves are scale-free, so neither term can be trivially minimised by
    shrinking the representation: the SmoothL1 acts on L2-normalised vectors and
    the cosine term is normalised by construction.
    """
    un, vn = F.normalize(u, dim=-1), F.normalize(v, dim=-1)
    smooth = F.smooth_l1_loss(un, vn, reduction="none").mean(-1)
    cosine = 1.0 - (un * vn).sum(-1)
    return (0.5 * (smooth + cosine)).mean()


def infonce_day(z_anchor: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor,
                temp: float = 0.1) -> torch.Tensor:
    """Lower bound on I(zS ; day_id | subject) -- Eq. 3's second term.

    The protocol supplies both ends for free:
      positive  = the V1 partner  (same subject, SAME day, different sensor)
      negatives = the V2 partner  (same subject, DIFFERENT day) + the rest of the
                  batch, which are other subjects' days.

    Conditioning on subject is what makes the V2 partner the right negative: it
    holds the trait fixed and varies only the day, so the only way to separate
    them is to encode day-level state. Returned as a *bound to maximise*; Eq. 3
    subtracts beta times this, so minimising L_day pushes it up.
    """
    a = F.normalize(z_anchor, dim=-1)
    p = F.normalize(z_pos, dim=-1)
    n = F.normalize(z_neg, dim=-1)

    pos = (a * p).sum(-1, keepdim=True) / temp                  # (B, 1)
    neg = a @ n.t() / temp                                      # (B, B)
    logits = torch.cat([pos, neg], dim=1)
    # log p(positive) under the softmax = the InfoNCE bound, up to log(N).
    return -F.cross_entropy(logits, torch.zeros(len(a), dtype=torch.long, device=a.device))


def independence_penalty(zT: torch.Tensor, zS: torch.Tensor, zA: torch.Tensor) -> torch.Tensor:
    """Eq. 4: squared Frobenius norm of the off-diagonal correlation.

    Two implementation choices Eq. 4 leaves open, both made explicitly:

    1. Only *between-block* entries are penalised. Within-block correlation is
       not a violation of the factorization -- zeroing it would impose a
       whitening constraint the proposal never asks for -- so those are masked.

    2. The sum is divided by the number of between-block entries, making this a
       *mean* squared correlation in [0, 1] rather than a raw Frobenius sum. The
       raw sum is ~9,700 entries wide at d=128 and lands two to three orders of
       magnitude above L_MCR, so at the proposal's lambda_3 = 1.0 it would
       swamp every other term. Normalising also keeps the penalty comparable
       across the (dT, dS, dA) sweep in E9, where the entry count changes.
    """
    z = torch.cat([zT, zS, zA], dim=-1)                         # (B, D)
    z = z - z.mean(0, keepdim=True)
    z = z / (z.std(0, keepdim=True) + EPS)
    corr = (z.t() @ z) / max(len(z) - 1, 1)                     # (D, D)

    dims = [zT.shape[-1], zS.shape[-1], zA.shape[-1]]
    block_mask = torch.zeros_like(corr)
    off = 0
    for d in dims:
        block_mask[off:off + d, off:off + d] = 1.0              # within-block
        off += d
    between_mask = 1.0 - block_mask
    n_between = between_mask.sum().clamp_min(1.0)
    return (corr * between_mask).pow(2).sum() / n_between


@torch.no_grad()
def collapse_stats(blocks: dict[str, torch.Tensor]) -> dict:
    """Per-block collapse monitor, logged every epoch.

    Measured at initialisation on real Stanford windows, every block starts at a
    mean pairwise cosine of ~0.999 with a per-dimension std of ~0.02 -- i.e. the
    encoder maps every window to essentially one direction. That matters here
    rather than being a curiosity: the alignment terms of Eqs. 2-3 are *already
    minimised* by exactly that degenerate solution, so they supply almost no
    gradient at the start and actively reward staying there. Prop. 1 argues the
    day- and sensor-discriminative terms prevent this, but those act on the
    projection heads while the alignment acts on the blocks.

    So we watch two numbers per block and report them either way:
      `cos_*`  mean off-diagonal cosine  -> 1.0 means directional collapse
      `std_*`  mean per-dimension std    -> 0.0 means magnitude collapse
    A magnitude-only check is not enough: a block can hold a healthy per-dim std
    while every sample points the same way.
    """
    out = {}
    for name, z in blocks.items():
        if len(z) < 2:
            continue
        zn = F.normalize(z.float(), dim=-1)
        c = zn @ zn.t()
        off = c[~torch.eye(len(c), dtype=torch.bool, device=c.device)]
        out[f"cos_{name}"] = off.mean().detach()
        out[f"std_{name}"] = z.float().std(0).mean().detach()
    return out


# ------------------------------------------------------------------- model

class GlucoPRISM(nn.Module):
    """GlucoFM backbone + blocked pooling + the three protocol objectives."""

    def __init__(self, cfg: PrismConfig | None = None):
        super().__init__()
        self.cfg = cfg or PrismConfig()
        # Composed, not forked. `self.fm` owns the online backbone, the EMA
        # target, the masked-context predictor and both transition heads, and
        # supplies L_MCR + L_TD unchanged.
        self.fm = GlucoFM(self.cfg.fm)

        self.head_T = ProjectionHead(self.cfg.d_trait, self.cfg.d_head)
        self.head_S = ProjectionHead(self.cfg.d_state, self.cfg.d_head)
        self.head_A = ProjectionHead(self.cfg.d_sensor, self.cfg.d_head)
        self.device_clf = nn.Linear(self.cfg.d_head, self.cfg.n_devices)

    # -- convenience --------------------------------------------------------

    def ema_update(self, m: float) -> None:
        self.fm.ema_update(m)

    def blocks(self, x, m, start_idx):
        """Unmasked online pass -> (zT, zS, zA). Used by every protocol term and
        by downstream extraction."""
        z_tokens = self.fm.online(x, m, start_idx, patch_mask=None)["z"]
        return blocked_pool(z_tokens, self.cfg)

    def _align_pair(self, block_u, block_v, head):
        """Honour D9: align through the projection head, or on the raw block."""
        if self.cfg.align_on == "head":
            return align_distance(head(block_u), head(block_v))
        return align_distance(block_u, block_v)

    @torch.no_grad()
    def embed(self, x, m, start_idx, block: str = "full") -> torch.Tensor:
        """Frozen readout. `block` in {full, zT, zS, zA, zTzS}.

        Always pre-projection. `zA` is dropped at inference for the subject-level
        representation (Sec. 4.4) -- `zTzS` is that concatenation.
        """
        zT, zS, zA = self.blocks(x, m, start_idx)
        return {"full": torch.cat([zT, zS, zA], dim=-1),
                "zT": zT, "zS": zS, "zA": zA,
                "zTzS": torch.cat([zT, zS], dim=-1)}[block]

    # -- objectives ---------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """`batch` keys:
            x, m, s           anchor window, mask, circadian start   (B, 288)/(B,)
            patch_mask        (B, P) bool, for L_MCR                 (B, P)
            device_id         (B,) long   -- anchor's real device
            x1, m1, s1        V1 partner: same subject, same day, other sensor
            device_id1        (B,) long   -- the partner's device (synthetic)
            x2, m2, s2        V2 partner: same subject, different day
            has_v2            (B,) bool   -- False for singleton subjects
        """
        cfg = self.cfg

        # ---- GlucoFM's own objectives, untouched -------------------------
        fm_out = self.fm(batch["x"], batch["m"], batch["s"], batch["patch_mask"])
        loss_mcr, loss_td = fm_out["loss_mcr"], fm_out["loss_td"]

        # ---- blocks for anchor and both protocol partners ----------------
        zT, zS, zA = self.blocks(batch["x"], batch["m"], batch["s"])
        zT1, zS1, zA1 = self.blocks(batch["x1"], batch["m1"], batch["s1"])

        # ---- L_sensor (Eq. 2) --------------------------------------------
        # V1 holds (t, s) fixed and varies only the sensor, so zT AND zS must
        # both transport across the pair, while zA stays device-discriminative.
        l_sensor = (self._align_pair(zT, zT1, self.head_T)
                    + self._align_pair(zS, zS1, self.head_S))

        hA = self.head_A(zA)
        hA1 = self.head_A(zA1)
        if cfg.adversarial_sensor:                      # E9 ablation, off by default
            hA = grad_reverse(hA, cfg.adversarial_lambda)
            hA1 = grad_reverse(hA1, cfg.adversarial_lambda)
        dev_logits = torch.cat([self.device_clf(hA), self.device_clf(hA1)], dim=0)
        dev_target = torch.cat([batch["device_id"], batch["device_id1"]], dim=0)
        l_device = F.cross_entropy(dev_logits, dev_target)
        l_sensor = l_sensor + l_device

        # ---- L_day (Eq. 3) -----------------------------------------------
        # V2 holds t fixed and varies the day. Only subjects with >= 2 days
        # contribute; singletons carry has_v2 == False and are masked out rather
        # than padded with a copy of themselves, which would be a free win.
        has2 = batch["has_v2"].bool()
        if has2.any():
            zT2, zS2, _ = self.blocks(batch["x2"], batch["m2"], batch["s2"])
            if cfg.align_on == "head":
                a2 = align_distance(self.head_T(zT[has2]), self.head_T(zT2[has2]))
            else:
                a2 = align_distance(zT[has2], zT2[has2])
            mi = infonce_day(zS[has2], zS1[has2], zS2[has2], temp=cfg.infonce_temp)
            l_day = a2 - cfg.beta_day * mi
        else:
            l_day = torch.zeros((), device=zT.device)
            mi = torch.zeros((), device=zT.device)

        # ---- L_indep (Eq. 4) ----------------------------------------------
        l_indep = independence_penalty(zT, zS, zA)

        total = (loss_mcr + loss_td
                 + cfg.lambda_sensor * l_sensor
                 + cfg.lambda_day * l_day
                 + cfg.lambda_indep * l_indep)

        out = {"loss": total,
               "loss_mcr": loss_mcr.detach(), "loss_td": loss_td.detach(),
               "loss_sensor": l_sensor.detach(), "loss_device": l_device.detach(),
               "loss_day": l_day.detach(), "mi_day": mi.detach(),
               "loss_indep": l_indep.detach(),
               "sigma": fm_out["sigma"]}
        out.update(collapse_stats({"zT": zT, "zS": zS, "zA": zA}))
        return out


def prism_param_report(cfg: PrismConfig | None = None) -> dict:
    """Proposal Sec. 4 states ~0.9M trainable, vs GlucoFM's 0.72M."""
    model = GlucoPRISM(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    fm_trainable = sum(p.numel() for p in model.fm.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total,
            "glucofm_part": fm_trainable, "prism_added": trainable - fm_trainable,
            "trainable_M": round(trainable / 1e6, 3), "total_M": round(total / 1e6, 3)}
