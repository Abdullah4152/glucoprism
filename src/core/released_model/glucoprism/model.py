"""
GlucoPRISM - protocol-supervised factorization of Trait / State / Sensor.

Built ON the GlucoFM backbone (arXiv:2605.30865), not as a replacement:
the mask-aware grid, causal Gaussian state/event split, dual-stream embedders,
circadian encoding, 3-layer context encoder and the L_MCR + L_TD objectives are
inherited unchanged from `glucofm/`. GlucoPRISM adds

  1. blocked pooling of the 24 patch tokens into z = [zT | zS | zA]
     (dT=64, dS=48, dA=16)                                     - proposal 4.1
  2. three protocol objectives L_sensor, L_day, L_indep         - proposal 4.2
  3. a permutation-invariant set aggregator over days           - proposal 4.4

Variants
--------
GlucoPRISM        single-scale, 24 hourly patches (the proposal as written)
GlucoPRISMHJEPA   adds a second temporal scale: 6 four-hour super-patches with
                  their own masked-latent objective (hierarchical-JEPA variant,
                  after Zhang et al., arXiv:2604.03208 - the step_skip idea
                  transplanted from action-conditioned planning to CGM time)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucofm.config import Config as GlucoFMConfig  # noqa: E402
from glucofm.model import GlucoFMEncoder, TransitionHead  # noqa: E402
from glucofm.modules import EPS, patchify, transformer_encoder  # noqa: E402


# --------------------------------------------------------------------------- #
@dataclass
class PrismConfig:
    # block dimensions (proposal 4.1); sum must equal GlucoFM's embed_dim
    d_trait: int = 64
    d_state: int = 48
    d_sensor: int = 16
    # objective weights (proposal 4.2, lambda_1..3).
    # MEASURED: at lambda=1.0 the protocol terms start ~5x larger than
    # L_MCR/L_TD and swamp them - the no-protocol control then beats the full
    # model. Scaled down so the representation objectives stay dominant.
    w_sensor: float = 0.2
    w_day: float = 0.2
    w_indep: float = 0.1
    # keeps zS day-discriminative (hinge on cross-day similarity)
    beta_day_info: float = 0.5
    day_margin: float = 0.3
    temperature: float = 0.1
    # MEASURED: without this, the V1/V2 alignment terms collapse zT and zS to a
    # constant - within- and between-subject cosine both hit 0.9999. Proposition
    # 1 assumes the day- and sensor-discriminative terms prevent the degenerate
    # solution, but those act on LEARNED PROJECTION HEADS, which can stay
    # separated while the block itself collapses. An explicit VICReg-style
    # variance floor on the blocks themselves is what actually holds them apart.
    w_variance: float = 1.0
    var_target: float | None = None   # None -> 1/sqrt(block_dim) on the unit sphere
    n_devices: int = 8
    # hierarchical variant
    hierarchical: bool = False
    step_skip: int = 4          # 24 hourly patches -> 6 four-hour super-patches
    w_l2: float = 1.0
    # Clinical Metric Prediction: predict the day's glucometrics from a masked
    # view. Level metrics -> zT, dispersion metrics -> zS. Requires
    # ModelConfig.scale_inject, without which the encoder is scale-blind and the
    # level targets are unpredictable in principle.
    use_cmp: bool = False
    w_cmp: float = 1.0
    # Cross-day metric prediction: from day 1's zT, predict day 2's LEVEL
    # metrics (same subject, different day). Within-day CMP only asks zT to
    # summarise the day in front of it; this asks it for the part of the level
    # that GENERALISES across days, which is the trait/state split stated as a
    # prediction problem rather than as an invariance constraint.
    use_xday_cmp: bool = False
    w_xday: float = 1.0
    # add nocturnal / daytime level metrics to L_CMP (fasting glucose is what
    # HOMA-IR and the ADA diabetes thresholds are actually defined on)
    circadian_metrics: bool = False
    # pool [mean ; sd] of the patch tokens instead of the mean alone
    stat_pool: bool = False
    # Adversarial suppression of clinical content in the SENSOR block.
    #
    # MEASURED (research/scripts/block_audit.py): zA scores 71.28 E1b ROC-AUC
    # against 72.06 for a dimension-matched random projection of `full` and
    # 70.92 for a random contiguous slice - i.e. zA is statistically
    # indistinguishable from "any 16 dimensions", and the proposal's prediction
    # that it is NEAR CHANCE on clinical labels fails.
    #
    # The cause is that nothing in the objective ever asks zA to be clinically
    # uninformative. L_indep only DECORRELATES, and decorrelation is not
    # information removal - two blocks can be uncorrelated and both predict the
    # label. (Measured, L_indep does not even decorrelate at w=0.1: zS~zA sits
    # at 0.575 against a 0.390 null.)
    #
    # L_ADV is the missing term, and it is exactly symmetric with L_CMP: where
    # L_CMP asks zT and zS to PREDICT the day's glucometrics, L_ADV asks zA to
    # make them UNPREDICTABLE, through a gradient-reversal head. Label-free, so
    # it costs nothing in supervision.
    use_adv: bool = False
    w_adv: float = 1.0
    adv_ramp: float = 0.2      # fraction of training over which lambda ramps 0->1
    # ---- STRUCTURAL routing (research round r10) ------------------------------
    #
    # The proposal splits ONE pooled vector by coordinate:
    #     z = LayerNorm(stat_proj([mean_p ; sd_p]));  zT,zS,zA = split(z)
    # so nothing makes coordinates 0:64 categorically different from 112:128
    # except three weak losses - and measurement says they do not manage it
    # (no block beats a random projection of its own width).
    #
    # MEASURED (research/scripts/why_blocks_fail.py), the free decomposition is
    # already better separated than the learned one:
    #     mean~sd 0.217   sd~max 0.213   mean~max 0.382
    #     zT~zS   0.348   zT~zA  0.316   zS~zA   0.575
    #
    # `structural` therefore builds the split ON those statistics: each block
    # reads a DIFFERENT function of the token sequence, so they differ by
    # construction rather than by penalty.
    #     zT <- mean_p(tokens)                     person / day-invariant level
    #     zS <- [sd_p ; range_p]                   within-day excursion amplitude
    #     zA <- [mean_p|delta tokens| ; density]   high-frequency roughness and
    #                                              the missingness pattern
    # This also restricts what zA CAN contain, which matters because the device
    # label is readable from wear fraction alone (ROC-AUC 100.0), leaving 15 of
    # zA's 16 dimensions unconstrained under the coordinate-slice design.
    structural: bool = False
    # What the SENSOR block is allowed to read. MEASURED (research/scripts/
    # za_leak.py), against a true null of 49.44 E1b ROC-AUC:
    #
    #     rough  (128-d)   73.13   +23.70   <- clinically informative
    #     rough_mean (1-d) 64.70   +15.27   <- STILL informative, so the problem
    #                                          is not capacity, it is content
    #     mask_feats (5-d) 47.71    -1.72   <- clinically NULL
    #
    # Treating high-frequency roughness as a sensor artifact was wrong for CGM:
    # normalised roughness measures the SMOOTHNESS OF GLUCOSE DYNAMICS, and
    # diabetic and healthy dynamics genuinely differ - it is physiology, not
    # noise. The missingness pattern is the only sensor signal here that is
    # clinically null, and it is also what identifies the device (wear fraction
    # alone gives ROC-AUC 100.0).
    #
    #   "mask"        zA <- mask features only. Sensor-sufficient, clinically
    #                 null; roughness moves to zS where it belongs.
    #   "rough+mask"  the r10 routing, kept so the ablation is reproducible.
    za_inputs: str = "rough+mask"
    # Where token ROUGHNESS is routed when za_inputs="mask".
    #
    # MEASURED (r11): with roughness in zS, the state block OVERTAKES the trait
    # block (72.45 vs 69.58 E1b ROC-AUC) - which inverts the proposal, since 16
    # of the 18 cells are trait tasks. Roughness alone scores 73.13, so whichever
    # block holds it wins. The open question is whether roughness is actually a
    # TRAIT (stable within a person across days) or a STATE (varies day to day);
    # if it is trait-like it belongs in zT and the inversion is a routing error
    # rather than a finding. `rough_to` runs that experiment.
    rough_to: str = "state"      # "state" | "trait"
    # Per-block LayerNorm. A single LayerNorm over all 128 dims normalises across
    # the block boundaries, mechanically coupling every coordinate to the mean
    # and variance of all the others - the last operation before the split
    # actively entangles it.
    block_norm: bool = False
    # L_CMP's dispersion targets, residualised against the level targets, so zS
    # is asked for the LEVEL-FREE part of dispersion (Q1: the two target sets are
    # 0.52 correlated, and `cv = sd/mean` is level-contaminated by construction).
    decorr_targets: bool = False
    # ---- Variational information bottleneck on the SENSOR block (r11) --------
    #
    # r9 tried to make zA clinically uninformative with a gradient-reversal
    # adversary and it BACKFIRED: at w_adv=3, zA rose to 72.80 against a 71.38
    # dimension-matched control - the only block in the study to beat its own
    # control, in the wrong direction. That is the documented failure mode of
    # adversarial attribute removal (Elazar & Goldberg, 2018): the encoder wins
    # by making the mapping hard for THAT head to invert, not by discarding the
    # information, and a fresh probe recovers it.
    #
    # The fix has to bound INFORMATION rather than defeat a head. Make zA a
    # stochastic channel
    #       zA = mu(h) + sigma(h) * eps,   eps ~ N(0, I)
    # and pay KL(q(zA|x) || N(0,I)) for every nat it carries. I(x; zA) is upper
    # bounded by that KL, so at small capacity zA CANNOT carry clinical signal -
    # near-chance is forced, not encouraged.
    #
    # This is the proposal's own claim stated correctly: zA should be a MINIMAL
    # SUFFICIENT STATISTIC FOR THE SENSOR. Device identity is worth about one bit
    # here (wear fraction alone gives ROC-AUC 100.0), so a ~1-nat channel keeps
    # the sensor task and has nothing left over.
    use_vib: bool = False
    w_vib: float = 1.0           # beta; the capacity price per nat
    vib_free_bits: float = 0.0   # nats allowed before the KL starts charging
    # Train the DeepSets day-aggregator (proposal 4.4 / Proposition 2). Until
    # now `SetAggregator` was constructed and checkpointed but never referenced
    # in forward(), so it received no gradient and Proposition 2 had never
    # actually been tested. The objective is the one the proposition motivates:
    # from the SET of a subject's daily state blocks, predict the BETWEEN-DAY
    # dispersion - exactly the statistic mean-pooling annihilates.
    use_agg: bool = False
    w_agg: float = 1.0
    # Day-level JEPA: from K-1 of a subject's days, predict the held-out day's
    # trait block, as scored by the EMA branch. The intra-day hierarchy
    # (hour -> 4-hour) never paid across five configurations; the hierarchy that
    # actually exists in CGM is day -> subject, and it is also the level the
    # headline protocol (E1b) aggregates at. The trained day encoder is then
    # usable AS the aggregator at evaluation time (E1d), instead of the fixed
    # [mean|sd|p10|p90].
    use_dayjepa: bool = False
    w_dayjepa: float = 1.0
    n_days: int = 4
    day_layers: int = 2
    # ablation switches
    use_sensor: bool = True
    use_day: bool = True
    use_indep: bool = True

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
class BlockedPool(nn.Module):
    """Pool 24 patch tokens to z, then split into [zT | zS | zA].

    The encoder is shared; the split is a property of the representation, not
    three separate networks (proposal 4.1). Each block gets a light projection
    head so the objectives act on block-specific spaces.
    """

    def __init__(self, embed_dim: int, cfg: PrismConfig):
        super().__init__()
        d = cfg.d_trait + cfg.d_state + cfg.d_sensor
        # Under coordinate slicing the blocks ARE a partition of z, so the widths
        # must sum to D. Structural routing gives each block its own projection
        # from its own statistic, which frees the widths - and that matters,
        # because the best readout in the whole study (`rich`, 384-d) is exactly
        # [mean;sd;max] of the tokens. Structural routing can therefore be given
        # rich's width and absorb it INTO the factorisation.
        if not cfg.structural:
            assert d == embed_dim, f"block dims {d} != embed_dim {embed_dim}"
        self.cfg = cfg
        self.norm = nn.LayerNorm(embed_dim)
        # Appendix C.6 mean-pools the 24 hourly tokens, which annihilates their
        # WITHIN-DAY dispersion - precisely the quantity Proposition 2 says the
        # state block should own. `stat_pool` keeps it by projecting
        # [mean ; sd] back down to D, so block widths are unchanged.
        self.stat_proj = (nn.Linear(2 * embed_dim, embed_dim)
                          if cfg.stat_pool else None)

        # ---- structural routing: one projection per block, different inputs --
        self.n_mask_feat = 5
        if cfg.structural:
            mask_only = cfg.za_inputs == "mask"
            r_trait = mask_only and cfg.rough_to == "trait"
            self.proj_t = nn.Linear((2 if r_trait else 1) * embed_dim,
                                    cfg.d_trait)
            # roughness is a DYNAMICS statistic, so under "mask" it joins the
            # state block instead of the sensor block - unless `rough_to` sends
            # it to the trait block instead
            self.proj_s = nn.Linear(
                (2 if (r_trait or not mask_only) else 3) * embed_dim,
                cfg.d_state)
            self.proj_a = nn.Linear(
                self.n_mask_feat if mask_only else embed_dim + self.n_mask_feat,
                cfg.d_sensor)
        if cfg.structural or cfg.block_norm:
            self.norm_t = nn.LayerNorm(cfg.d_trait)
            self.norm_s = nn.LayerNorm(cfg.d_state)
            self.norm_a = nn.LayerNorm(cfg.d_sensor)

        # Stochastic sensor channel: one head for the mean, one for the
        # log-variance.
        #
        # MEASURED: putting these AFTER `norm_a` makes the bottleneck inert. A
        # LayerNorm pins ||mu||^2 ~ d_sensor by construction, so the KL has a
        # hard floor (~24 nats at 48 dims) that no beta can push through - a
        # sweep over beta in {0.01 ... 10} left the KL at ~41 nats and the
        # clinical R^2 at 0.78, completely flat. The mean head must therefore be
        # an UNCONSTRAINED linear map, and `norm_a` is bypassed when VIB is on.
        if cfg.use_vib:
            self.vib_mu = nn.Linear(cfg.d_sensor, cfg.d_sensor)
            self.vib_lv = nn.Linear(cfg.d_sensor, cfg.d_sensor)
            nn.init.zeros_(self.vib_lv.weight)
            nn.init.constant_(self.vib_lv.bias, -2.0)   # start near-deterministic
        self.head_t = nn.Sequential(nn.Linear(cfg.d_trait, cfg.d_trait), nn.GELU(),
                                    nn.Linear(cfg.d_trait, cfg.d_trait))
        self.head_s = nn.Sequential(nn.Linear(cfg.d_state, cfg.d_state), nn.GELU(),
                                    nn.Linear(cfg.d_state, cfg.d_state))
        self.head_a = nn.Sequential(nn.Linear(cfg.d_sensor, cfg.d_sensor), nn.GELU(),
                                    nn.Linear(cfg.d_sensor, cfg.d_sensor))

    @staticmethod
    def _mask_feats(dens: torch.Tensor) -> torch.Tensor:
        """Per-patch density [B, P] -> 5 summary numbers describing missingness."""
        P = dens.shape[1]
        return torch.stack([
            dens.mean(1), dens.std(1), dens.amin(1),
            dens[:, :P // 2].mean(1), dens[:, P // 2:].mean(1),
        ], dim=-1)

    def forward(self, tokens: torch.Tensor, dens: torch.Tensor | None = None) -> dict:
        """tokens [B, P, D] -> dict of z, zT, zS, zA.

        `dens` is the per-patch observation density; structural routing feeds it
        to the sensor block, which is the only block that should see it.
        """
        c = self.cfg
        if c.structural:
            mu = tokens.mean(dim=1)
            sd = tokens.std(dim=1)
            rng = tokens.amax(dim=1) - tokens.amin(dim=1)
            # Temporal roughness: sensor noise is high-frequency, physiological
            # excursion is low-frequency but large, so FREQUENCY CONTENT belongs
            # to the sensor block and AMPLITUDE belongs to the state block.
            #
            # It must therefore be SCALE-FREE. A raw mean|delta| scales with
            # amplitude, so scaling the tokens about their mean moved raw_A by
            # 0.1295 against raw_S's 0.0135 - the sensor block was reading the
            # state block's statistic. Dividing by sd removes exactly that:
            # under tokens -> mu + a*(tokens - mu) both numerator and denominator
            # scale by a, so the ratio is invariant.
            rough = ((tokens[:, 1:] - tokens[:, :-1]).abs().mean(dim=1)
                     / (sd + EPS))
            mf = (self._mask_feats(dens) if dens is not None
                  else torch.zeros(tokens.shape[0], self.n_mask_feat,
                                   device=tokens.device, dtype=tokens.dtype))
            if c.za_inputs == "mask":
                # roughness is physiology, not sensor noise -> it goes to a
                # CLINICAL block; `rough_to` decides which one
                if c.rough_to == "trait":
                    zt = self.norm_t(self.proj_t(torch.cat([mu, rough], -1)))
                    zs = self.norm_s(self.proj_s(torch.cat([sd, rng], -1)))
                else:
                    zt = self.norm_t(self.proj_t(mu))
                    zs = self.norm_s(self.proj_s(torch.cat([sd, rng, rough], -1)))
                a_in = self.proj_a(mf)
            else:
                zt = self.norm_t(self.proj_t(mu))
                zs = self.norm_s(self.proj_s(torch.cat([sd, rng], dim=-1)))
                a_in = self.proj_a(torch.cat([rough, mf], dim=-1))
            # bypass norm_a under VIB - see the note in __init__: a LayerNorm
            # here pins ||mu|| and makes the capacity penalty unpayable
            za = a_in if c.use_vib else self.norm_a(a_in)
            z = torch.cat([zt, zs, za], dim=-1)
        else:
            if self.stat_proj is not None:
                z = self.norm(self.stat_proj(
                    torch.cat([tokens.mean(dim=1), tokens.std(dim=1)], dim=-1)))
            else:
                z = self.norm(tokens.mean(dim=1))
            zt, zs, za = torch.split(z, [c.d_trait, c.d_state, c.d_sensor], dim=-1)
            if c.block_norm:
                zt, zs, za = self.norm_t(zt), self.norm_s(zs), self.norm_a(za)
                z = torch.cat([zt, zs, za], dim=-1)
        out = {}
        if c.use_vib:
            # zA becomes a stochastic channel. I(x; zA) <= KL, so charging for
            # the KL is a hard cap on what zA can carry - unlike an adversary,
            # which only makes the information awkward to read.
            mu_a = self.vib_mu(za)
            lv = self.vib_lv(za).clamp(-8.0, 4.0)
            if self.training:
                za = mu_a + torch.randn_like(mu_a) * torch.exp(0.5 * lv)
            else:
                za = mu_a          # deterministic at evaluation
            kl = 0.5 * (mu_a.pow(2) + lv.exp() - 1.0 - lv).sum(-1)
            if c.vib_free_bits > 0:
                kl = torch.clamp(kl - c.vib_free_bits, min=0.0)
            out["kl_a"] = kl.mean()
            out["nats_a"] = (0.5 * (mu_a.pow(2) + lv.exp() - 1.0 - lv)
                             ).sum(-1).mean().detach()
            z = torch.cat([zt, zs, za], dim=-1) if c.structural else z

        out.update({"z": z, "zT": self.head_t(zt), "zS": self.head_s(zs),
                    "zA": self.head_a(za), "raw_T": zt, "raw_S": zs, "raw_A": za})
        return out


class SetAggregator(nn.Module):
    """Permutation-invariant DeepSets encoder over {zS^(k)} across days.

    Proposition 2: the mean is the minimum-variance estimator for a day-INVARIANT
    block, and a lossy one otherwise. So zT is mean-pooled and zS goes through an
    aggregator whose input features explicitly include dispersion (mean, sd,
    range), which is exactly the statistic mean-pooling annihilates.
    """

    def __init__(self, d_state: int, hidden: int = 128, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or d_state
        self.phi = nn.Sequential(nn.Linear(3 * d_state, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, zs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """zs [B, K, dS] -> [B, out_dim]. mask [B, K] marks valid days."""
        if mask is None:
            mask = torch.ones(zs.shape[:2], device=zs.device, dtype=zs.dtype)
        w = mask.unsqueeze(-1)
        n = w.sum(dim=1).clamp(min=1.0)
        mean = (zs * w).sum(dim=1) / n
        var = ((zs - mean.unsqueeze(1)) ** 2 * w).sum(dim=1) / n
        sd = torch.sqrt(var + EPS)
        big = zs.masked_fill(w == 0, float("-inf")).max(dim=1).values
        small = zs.masked_fill(w == 0, float("inf")).min(dim=1).values
        rng = torch.nan_to_num(big - small, neginf=0.0, posinf=0.0)
        return self.phi(torch.cat([mean, sd, rng], dim=-1))


# --------------------------------------------------------------------------- #
def align_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Normalized alignment D(.,.): cosine distance on L2-normalized blocks."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (1.0 - (a * b).sum(-1)).mean()


def info_nce(anchor: torch.Tensor, positive: torch.Tensor,
             temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE lower bound on I(anchor; positive), in-batch negatives."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = a @ p.t() / temperature
    tgt = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def variance_floor(z: torch.Tensor, target: float | None = None) -> torch.Tensor:
    """Anti-collapse hinge, applied on the UNIT SPHERE.

    A plain VICReg variance floor on raw activations is not enough here: it
    keeps per-dimension magnitude varying while every vector still points the
    same way (cosine similarity 1.000 with healthy per-dim std). Since the
    alignment objectives are cosine-based, the collapse they drive is
    DIRECTIONAL, so the floor has to be measured in the same geometry.

    For d-dimensional unit vectors spread over the sphere the per-dimension std
    is ~1/sqrt(d), which is the default target.
    """
    zn = F.normalize(z, dim=-1)
    d = zn.shape[-1]
    tgt = (1.0 / math.sqrt(d)) if target is None else target
    std = torch.sqrt(zn.var(dim=0) + 1e-8)
    return (F.relu(tgt - std).mean() / max(tgt, 1e-8))


# --------------------------------------------------------------------------- #
# Clinical Metric Prediction (L_CMP) - GlucoPRISM's addition to the objective.
#
# The JEPA objectives are defined entirely in latent space on z-scored windows,
# so nothing in GlucoFM's training signal ever asks the representation to retain
# a quantity measured in mg/dL. Every downstream label here is defined on such a
# quantity. L_CMP closes that gap WITHOUT using any label: the targets are
# standard CGM glucometrics computed from the raw full-day window, and the model
# must predict them from a MASKED view, which keeps it a predictive task in the
# JEPA sense rather than a reconstruction.
#
# The split across blocks is the point: LEVEL metrics are a person-property and
# are routed to zT, DISPERSION metrics are a day-property and are routed to zS.
# That gives the proposal's Trait/State factorisation actual content, instead of
# relying on the alignment objectives alone to discover it.
LEVEL_METRICS = ("mean", "tbr70", "tir", "tar180", "tar250")
DISP_METRICS = ("sd", "cv", "mad", "hourly_sd", "range")
# Optional circadian extension. HOMA-IR and the ADA diabetes thresholds are
# defined on FASTING glucose, not on a 24-hour mean, and the dawn phenomenon is
# a distinct physiological signal. Windows are circadian-aligned by `start_idx`,
# so absolute time of day is recoverable and these can be targeted directly.
CIRCADIAN_METRICS = ("nocturnal", "daytime")


def n_level_metrics(circadian: bool = False) -> int:
    return len(LEVEL_METRICS) + (len(CIRCADIAN_METRICS) if circadian else 0)


def window_glucometrics(x: torch.Tensor, m: torch.Tensor,
                        start_idx: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask-aware CGM summary statistics. x, m: [B, L] -> ([B, 5 or 7], [B, 5]).

    Returned in roughly unit scale so a single smooth-L1 weight is sensible for
    all of them. No gradient flows here - these are targets. Passing
    `start_idx` appends the two circadian level metrics.
    """
    m = m.float()
    cnt = m.sum(1).clamp(min=1.0)
    mean = (x * m).sum(1) / cnt
    var = (((x - mean.unsqueeze(1)) ** 2) * m).sum(1) / cnt
    sd = torch.sqrt(var + EPS)

    def frac(cond):
        return ((cond.float() * m).sum(1) / cnt)

    level = torch.stack([
        mean / 100.0,
        frac(x < 70),                               # time below range
        frac((x >= 70) & (x <= 180)),               # time in range
        frac(x > 180),                              # time above range, level 1
        frac(x > 250),                              # time above range, level 2
    ], dim=-1)

    if start_idx is not None:
        B, L = x.shape
        # absolute grid index of every position, given the window's circadian
        # start; L = 288 five-minute steps, so 00:00-06:00 is [0, 72)
        j = torch.arange(L, device=x.device).unsqueeze(0)
        abs_i = (start_idx.view(-1, 1).to(x.device) + j) % L
        night = ((abs_i < L // 4).float() * m)
        day = ((abs_i >= L // 4).float() * m)
        noct = (x * night).sum(1) / night.sum(1).clamp(min=1.0)
        dayt = (x * day).sum(1) / day.sum(1).clamp(min=1.0)
        # windows with no observed night fall back to the daily mean rather than 0
        noct = torch.where(night.sum(1) > 0, noct, mean)
        dayt = torch.where(day.sum(1) > 0, dayt, mean)
        level = torch.cat([level, torch.stack([noct / 100.0, dayt / 100.0], -1)], -1)

    # successive absolute difference over positions observed on BOTH sides
    d = (x[:, 1:] - x[:, :-1]).abs()
    dm = m[:, 1:] * m[:, :-1]
    mad = (d * dm).sum(1) / dm.sum(1).clamp(min=1.0)

    # dispersion of hourly means: the within-day swing the state block should own
    B, L = x.shape
    K = L // 24
    xh = x[:, :24 * K].reshape(B, 24, K)
    mh = m[:, :24 * K].reshape(B, 24, K)
    ch = mh.sum(-1)
    hm = (xh * mh).sum(-1) / ch.clamp(min=1.0)
    hv = (ch > 0).float()
    hmean = (hm * hv).sum(1) / hv.sum(1).clamp(min=1.0)
    hsd = torch.sqrt((((hm - hmean.unsqueeze(1)) ** 2) * hv).sum(1)
                     / hv.sum(1).clamp(min=1.0) + EPS)

    big = x.masked_fill(m == 0, float("-inf")).amax(1)
    small = x.masked_fill(m == 0, float("inf")).amin(1)
    rng = torch.nan_to_num(big - small, neginf=0.0, posinf=0.0)

    disp = torch.stack([
        sd / 50.0, sd / mean.clamp(min=1.0), mad / 10.0, hsd / 50.0, rng / 100.0,
    ], dim=-1)
    return level, disp


class _GradReverse(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward (DANN)."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


def offdiag_corr_penalty(*blocks: torch.Tensor) -> torch.Tensor:
    """|| offdiag( Corr([zT; zS; zA]) ) ||_F^2, cross-block entries only."""
    z = torch.cat(blocks, dim=-1)
    z = z - z.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, keepdim=True) + EPS)
    n = max(z.shape[0] - 1, 1)
    c = (z.t() @ z) / n
    # mask: keep only entries BETWEEN different blocks
    sizes = [b.shape[-1] for b in blocks]
    mask = torch.ones_like(c)
    off = 0
    for s in sizes:
        mask[off:off + s, off:off + s] = 0.0
        off += s
    return ((c * mask) ** 2).sum() / max(mask.sum(), 1.0)


# --------------------------------------------------------------------------- #
class GlucoPRISM(nn.Module):
    """GlucoFM backbone + blocked pooling + protocol objectives."""

    def __init__(self, fm_cfg: GlucoFMConfig, cfg: PrismConfig):
        super().__init__()
        self.fm_cfg, self.cfg = fm_cfg, cfg
        m = fm_cfg.model

        self.online = GlucoFMEncoder(fm_cfg)
        self.target = copy.deepcopy(self.online)
        for q in self.target.parameters():
            q.requires_grad_(False)

        self.predictor = transformer_encoder(
            m.embed_dim, m.n_heads, m.ffn_dim, m.predictor_layers, m.dropout)
        self.pred_proj = nn.Linear(m.embed_dim, m.embed_dim)
        self.state_transition = TransitionHead(m.stream_dim, m.transition_hidden)
        self.event_transition = TransitionHead(m.stream_dim, m.transition_hidden)

        self.pool = BlockedPool(m.embed_dim, cfg)
        self.pool_t = copy.deepcopy(self.pool)          # target-side pooling
        for q in self.pool_t.parameters():
            q.requires_grad_(False)

        self.device_head = nn.Linear(cfg.d_sensor, cfg.n_devices)
        self.day_head = nn.Sequential(nn.Linear(cfg.d_state, cfg.d_state), nn.GELU(),
                                      nn.Linear(cfg.d_state, cfg.d_state))
        self.aggregator = SetAggregator(cfg.d_state)
        self.agg_head = (nn.Linear(cfg.d_state, len(DISP_METRICS))
                         if cfg.use_agg else None)

        if cfg.use_dayjepa:
            dt = cfg.d_trait + cfg.d_state
            self.day_in = nn.Linear(dt, m.embed_dim)
            self.day_encoder = transformer_encoder(
                m.embed_dim, m.n_heads, m.ffn_dim, cfg.day_layers, m.dropout)
            self.day_out = nn.Linear(m.embed_dim, cfg.d_trait)
            self.day_mask_token = nn.Parameter(torch.zeros(1, 1, m.embed_dim))
            nn.init.trunc_normal_(self.day_mask_token, std=0.02)
            self.day_pos = nn.Parameter(torch.zeros(1, cfg.n_days, m.embed_dim))
            nn.init.trunc_normal_(self.day_pos, std=0.02)

        if cfg.use_cmp or cfg.use_xday_cmp:
            self.cmp_level = nn.Sequential(
                nn.Linear(cfg.d_trait, cfg.d_trait), nn.GELU(),
                nn.Linear(cfg.d_trait, n_level_metrics(cfg.circadian_metrics)))
            self.cmp_disp = nn.Sequential(
                nn.Linear(cfg.d_state, cfg.d_state), nn.GELU(),
                nn.Linear(cfg.d_state, len(DISP_METRICS)))

        # adversary: tries to read the SAME glucometrics out of zA. Its gradient
        # is reversed into the encoder, so zA is pushed to carry none of them.
        # Deliberately given MORE capacity than the CMP heads - a weak adversary
        # would be defeated by a zA that merely hides the information nonlinearly
        # rather than discarding it.
        if cfg.use_adv:
            n_out = n_level_metrics(cfg.circadian_metrics) + len(DISP_METRICS)
            self.adv_head = nn.Sequential(
                nn.Linear(cfg.d_sensor, 64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(), nn.Linear(64, n_out))

        # ---- hierarchical (H-JEPA) second scale ----
        if cfg.hierarchical:
            self.l2_proj = nn.Linear(m.embed_dim, m.embed_dim)
            self.l2_encoder = transformer_encoder(
                m.embed_dim, m.n_heads, m.ffn_dim, 2, m.dropout)
            self.l2_predictor = transformer_encoder(
                m.embed_dim, m.n_heads, m.ffn_dim, 1, m.dropout)
            self.l2_pred_proj = nn.Linear(m.embed_dim, m.embed_dim)
            self.l2_pos = nn.Parameter(
                torch.zeros(1, fm_cfg.grid.n_patches // cfg.step_skip, m.embed_dim))
            nn.init.trunc_normal_(self.l2_pos, std=0.02)

        self.ema_m = fm_cfg.pretrain.ema_momentum
        # fraction of training elapsed; only the adversarial ramp reads it
        self._progress = 0.0
        self._vib_views: list = []

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_target(self, momentum: float | None = None):
        mm = self.ema_m if momentum is None else momentum
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.mul_(mm).add_(op.detach(), alpha=1.0 - mm)
        for tb, ob in zip(self.target.buffers(), self.online.buffers()):
            tb.copy_(ob)
        for tp, op in zip(self.pool_t.parameters(), self.pool.parameters()):
            tp.mul_(mm).add_(op.detach(), alpha=1.0 - mm)

    # ------------------------------------------------------------------ #
    def encode(self, x, m, s) -> dict:
        """Frozen forward used downstream.

        Returns the PRE-projection blocks. The objectives act on the projection
        heads (SimCLR/VICReg/BYOL convention), and the head is discarded at
        evaluation time. Evaluating the projection instead is what made zT and
        zS read as perfectly collapsed downstream (separation +0.000) even once
        the variance floor was satisfied on the block itself.
        """
        tok, _, _, dens = self.online(x, m, s, patch_mask=None)
        o = self.pool(tok, dens)
        out = {"z": o["z"], "zT": o["raw_T"], "zS": o["raw_S"], "zA": o["raw_A"],
               "proj_T": o["zT"], "proj_S": o["zS"], "proj_A": o["zA"],
               "tokens": tok}
        # The CMP heads' own outputs are the model's predicted glucometrics -
        # ten numbers in clinical units. A LINEAR probe cannot in general recover
        # them from z, because they are a nonlinear readout of it, so exposing
        # them as part of the representation is not the same as leaving them
        # implicit. `zC` is z with those predictions appended.
        if getattr(self, "cmp_level", None) is not None:
            c = torch.cat([self.cmp_level(o["raw_T"]), self.cmp_disp(o["raw_S"])], -1)
            out["cmp"] = c
            out["zC"] = torch.cat([o["z"], c], dim=-1)
        # The EMA target branch is a weight average of the online encoder over
        # training, which in BYOL-family models is often the better frozen
        # feature extractor. Appendix C.6 discards it; it costs one extra
        # forward pass to check whether that is leaving anything on the table.
        tok_e, _, _, dens_e = self.target(x, m, s, patch_mask=None)
        out["ema"] = self.pool_t(tok_e, dens_e)["z"]
        return out

    def _l2_tokens(self, tok: torch.Tensor) -> torch.Tensor:
        """Pool L1 hourly patches into L2 four-hour super-patches."""
        B, P, D = tok.shape
        k = self.cfg.step_skip
        return self.l2_proj(tok.reshape(B, P // k, k, D).mean(dim=2))

    # ------------------------------------------------------------------ #
    def forward(self, batch: dict) -> dict:
        """One training step.

        batch keys:
          x, m, s, patch_mask            the anchor window
          x1, m1, s1                     V1 partner (synthetic or real sensor view)
          x2, m2, s2                     V2 partner (same subject, different day)
          device                         device-family id for the anchor
          has_v2                         [B] 1 where a V2 partner exists
        """
        fm, cfg = self.fm_cfg, self.cfg
        g, pc = fm.grid, fm.pretrain
        x, m, s, pm = batch["x"], batch["m"], batch["s"], batch["patch_mask"]

        vis = (1.0 - pm).repeat_interleave(g.patch_size, dim=1)
        dens = patchify(m, g.n_patches, g.patch_size).mean(dim=-1)

        tok_ctx, s_on, e_on, dens_ctx = self.online(x, m * vis, s, pm)
        z_pred = self.pred_proj(self.predictor(tok_ctx))
        with torch.no_grad():
            tok_tgt, s_tgt, e_tgt, _ = self.target(x, m, s, patch_mask=None)

        # ---- inherited GlucoFM objectives -------------------------------
        w = dens * pm
        per_patch = F.smooth_l1_loss(z_pred, tok_tgt.detach(),
                                     beta=pc.smooth_l1_beta, reduction="none").mean(-1)
        loss_mcr = (w * per_patch).sum() / (w.sum() + EPS)

        q = (1.0 - pm[:, :-1]) * dens[:, :-1] * dens[:, 1:]
        td_s = F.smooth_l1_loss(self.state_transition(s_on[:, :-1]), s_tgt[:, 1:].detach(),
                                beta=pc.smooth_l1_beta, reduction="none").mean(-1)
        td_e = F.smooth_l1_loss(self.event_transition(e_on[:, :-1]), e_tgt[:, 1:].detach(),
                                beta=pc.smooth_l1_beta, reduction="none").mean(-1)
        loss_td = (q * (td_s + td_e)).sum() / (2 * q.sum() + EPS)

        out = {"loss_mcr": loss_mcr, "loss_td": loss_td}
        total = loss_mcr + loss_td

        # ---- hierarchical second scale ----------------------------------
        if cfg.hierarchical:
            l2_ctx = self.l2_encoder(self._l2_tokens(tok_ctx) + self.l2_pos)
            l2_pred = self.l2_pred_proj(self.l2_predictor(l2_ctx))
            with torch.no_grad():
                l2_tgt = self.l2_encoder(self._l2_tokens(tok_tgt) + self.l2_pos)
            k = cfg.step_skip
            pm2 = pm.reshape(pm.shape[0], -1, k).amax(dim=2)      # masked if any child was
            d2 = dens.reshape(dens.shape[0], -1, k).mean(dim=2)
            w2 = d2 * pm2
            pp2 = F.smooth_l1_loss(l2_pred, l2_tgt.detach(), beta=pc.smooth_l1_beta,
                                   reduction="none").mean(-1)
            loss_l2 = (w2 * pp2).sum() / (w2.sum() + EPS)
            out["loss_l2"] = loss_l2
            total = total + cfg.w_l2 * loss_l2

        # ---- blocked representation -------------------------------------
        za = self.pool(tok_ctx, dens_ctx)
        # every pooled view produced this step; L_VIB is charged on all of them
        self._vib_views = [za]

        # ---- L_CMP: clinical metrics of the FULL day, from the MASKED view --
        if cfg.use_cmp or cfg.use_adv:
            with torch.no_grad():
                lvl, dsp = window_glucometrics(x, m, s if cfg.circadian_metrics else None)
                if cfg.decorr_targets:
                    # MEASURED: the level and dispersion target sets are 0.52
                    # correlated (tir~sd alone is 0.734), because a high-mean day
                    # IS a high-variance day. Routing correlated targets to
                    # different blocks cannot produce uncorrelated blocks. Regress
                    # the dispersion targets on the level targets within the batch
                    # and keep the RESIDUAL, so zS is asked only for the part of
                    # dispersion that level does not already explain.
                    A = torch.cat([lvl, torch.ones_like(lvl[:, :1])], dim=-1)
                    try:
                        beta = torch.linalg.lstsq(A, dsp).solution
                        r = dsp - A @ beta
                        # keep the residual on the original scale so one
                        # smooth-L1 weight still suits every target
                        dsp = r / (r.std(dim=0, keepdim=True) + EPS) * \
                            dsp.std(dim=0, keepdim=True)
                    except Exception:      # noqa: BLE001 - degenerate batch
                        pass

        if cfg.use_cmp:
            l_lvl = F.smooth_l1_loss(self.cmp_level(za["raw_T"]), lvl,
                                     beta=pc.smooth_l1_beta)
            l_dsp = F.smooth_l1_loss(self.cmp_disp(za["raw_S"]), dsp,
                                     beta=pc.smooth_l1_beta)
            loss_cmp = l_lvl + l_dsp
            out.update(loss_cmp=loss_cmp, loss_cmp_level=l_lvl.detach(),
                       loss_cmp_disp=l_dsp.detach())
            total = total + cfg.w_cmp * loss_cmp

        # ---- L_ADV: the SENSOR block must NOT carry the clinical metrics ----
        # Ramped, because a full-strength reversal from step 0 fights the encoder
        # before it has anything worth suppressing.
        if cfg.use_adv:
            lam = min(1.0, self._progress / max(cfg.adv_ramp, 1e-6))
            tgt = torch.cat([lvl, dsp], dim=-1)
            pred = self.adv_head(grad_reverse(za["raw_A"], lam))
            l_adv = F.smooth_l1_loss(pred, tgt, beta=pc.smooth_l1_beta)
            out.update(loss_adv=l_adv, adv_lambda=torch.tensor(lam))
            total = total + cfg.w_adv * l_adv

        # L_sensor (Eq. 2): V1 pairs share (t, s) and differ only in a
        if cfg.use_sensor:
            tok1, _, _, dens1 = self.online(batch["x1"], batch["m1"], batch["s1"], None)
            zb = self.pool(tok1, dens1)
            self._vib_views.append(zb)   # UNMASKED view - the eval regime
            l_align = align_loss(za["zT"], zb["zT"]) + align_loss(za["zS"], zb["zS"])
            # device-identity head: needs LABELS, not pairs - so it is trained on
            # the real corpus device families as well as the synthetic partner
            logits = torch.cat([self.device_head(za["zA"]), self.device_head(zb["zA"])])
            tgt = torch.cat([batch["device"], batch["device_v1"]])
            l_dev = F.cross_entropy(logits, tgt)
            loss_sensor = l_align + l_dev
            out.update(loss_sensor=loss_sensor, loss_sensor_align=l_align.detach(),
                       loss_sensor_dev=l_dev.detach())
            total = total + cfg.w_sensor * loss_sensor

        # L_day (Eq. 3): V2 pairs share t, differ in s
        if cfg.use_day and batch.get("has_v2") is not None and batch["has_v2"].any():
            sel = batch["has_v2"] > 0
            tok2, _, _, dens2 = self.online(batch["x2"][sel], batch["m2"][sel],
                                                 batch["s2"][sel], None)
            zc = self.pool(tok2, dens2)
            self._vib_views.append(zc)
            l_trait = align_loss(za["zT"][sel], zc["zT"])
            # The -beta I(zS; day) term must keep zS DAY-DISCRIMINATIVE. An
            # InfoNCE between the two V2 windows does the opposite: it treats
            # different days of one subject as a positive pair and so makes zS
            # day-INVARIANT, collapsing it onto zT. Instead push zS apart across
            # days with a hinge, which is what "discriminative of day identity
            # within subject" actually asks for, and which also prevents the
            # degenerate zS = const solution.
            zs_a = F.normalize(self.day_head(za["zS"][sel]), dim=-1)
            zs_b = F.normalize(self.day_head(zc["zS"]), dim=-1)
            sim = (zs_a * zs_b).sum(-1)
            l_sep = F.relu(sim - cfg.day_margin).mean()
            loss_day = l_trait + cfg.beta_day_info * l_sep
            out.update(loss_day=loss_day, loss_day_trait=l_trait.detach(),
                       loss_day_sep=l_sep.detach())
            total = total + cfg.w_day * loss_day

            # ---- L_agg: the day-set must carry BETWEEN-DAY dispersion -------
            if cfg.use_agg:
                zs_set = torch.stack([za["raw_S"][sel], zc["raw_S"]], dim=1)
                with torch.no_grad():
                    _, da = window_glucometrics(x[sel], m[sel])
                    _, db = window_glucometrics(batch["x2"][sel], batch["m2"][sel])
                l_agg = F.smooth_l1_loss(self.agg_head(self.aggregator(zs_set)),
                                         (da - db).abs(), beta=pc.smooth_l1_beta)
                out["loss_agg"] = l_agg
                total = total + cfg.w_agg * l_agg

            # ---- L_xday: day 1's zT must predict day 2's level metrics ------
            if cfg.use_xday_cmp:
                cm = cfg.circadian_metrics
                with torch.no_grad():
                    lvl_a, _ = window_glucometrics(x[sel], m[sel],
                                                   s[sel] if cm else None)
                    lvl_b, _ = window_glucometrics(
                        batch["x2"][sel], batch["m2"][sel],
                        batch["s2"][sel] if cm else None)
                # symmetric: each day predicts the other's level
                l_x = 0.5 * (F.smooth_l1_loss(self.cmp_level(za["raw_T"][sel]), lvl_b,
                                              beta=pc.smooth_l1_beta)
                             + F.smooth_l1_loss(self.cmp_level(zc["raw_T"]), lvl_a,
                                                beta=pc.smooth_l1_beta))
                out["loss_xday"] = l_x
                total = total + cfg.w_xday * l_x

        # anti-collapse variance floor on the blocks themselves. Applied whenever
        # any alignment objective is active, since those are what pull toward the
        # constant solution.
        # ---- L_VIB: pay for every nat the sensor channel carries ------------
        #
        # The KL must be charged on EVERY view the encoder produces, not only on
        # the masked anchor. MEASURED: charging the anchor alone drove its KL to
        # 0.0 during training while evaluation - which reads the UNMASKED view -
        # still showed 8.58 nats and a clinical R^2 of 0.80, with device AUC at
        # 95.8. A genuinely 0-nat channel cannot support device identity at 95.8,
        # so the encoder had simply learned to satisfy the bottleneck on the
        # masked view and stay informative on the full one. `_vib_views`
        # collects the pooled dicts produced in this step (anchor, V1 partner,
        # V2 partner) so the price is paid on all of them.
        if cfg.use_vib:
            kls = [d["kl_a"] for d in self._vib_views if "kl_a" in d]
            if kls:
                kl = torch.stack(kls).mean()
                out["loss_vib"] = kl
                out["nats_a"] = torch.stack(
                    [d["nats_a"] for d in self._vib_views if "nats_a" in d]).mean()
                total = total + cfg.w_vib * kl

        if cfg.w_variance > 0 and (cfg.use_sensor or cfg.use_day):
            # Applied to BOTH the projections (where the alignment loss acts and
            # so where collapse is driven) and the blocks themselves (what
            # downstream actually reads).
            # With VIB active the sensor block is DELIBERATELY being squeezed
            # toward the prior, so a variance floor on it directly opposes the
            # KL. Exclude zA in that case and hold only zT and zS apart.
            parts = [variance_floor(za["zT"], cfg.var_target),
                     variance_floor(za["zS"], cfg.var_target),
                     variance_floor(za["raw_T"], cfg.var_target),
                     variance_floor(za["raw_S"], cfg.var_target)]
            if not cfg.use_vib:
                parts += [variance_floor(za["zA"], cfg.var_target),
                          variance_floor(za["raw_A"], cfg.var_target)]
            l_var = sum(parts) / len(parts)
            out["loss_var"] = l_var
            total = total + cfg.w_variance * l_var

        # L_indep (Eq. 4)
        if cfg.use_indep:
            loss_indep = offdiag_corr_penalty(za["raw_T"], za["raw_S"], za["raw_A"])
            out["loss_indep"] = loss_indep
            total = total + cfg.w_indep * loss_indep

        # ---- L_dayjepa: predict a held-out DAY's trait block from the others --
        if cfg.use_dayjepa and "xk" in batch:
            xk, mk_, sk = batch["xk"], batch["mk"], batch["sk"]
            kv = batch["kvalid"]
            B, K, L = xk.shape
            flat = lambda t: t.reshape(B * K, *t.shape[2:])          # noqa: E731
            tk, _, _, dens_k = self.online(flat(xk), flat(mk_), flat(sk), None)
            zk = self.pool(tk, dens_k)
            feat = torch.cat([zk["raw_T"], zk["raw_S"]], -1).reshape(B, K, -1)
            with torch.no_grad():
                tk_t, _, _, dens_kt = self.target(flat(xk), flat(mk_), flat(sk), None)
                zt = self.pool_t(tk_t, dens_kt)["raw_T"].reshape(B, K, -1)

            # hold out one VALID day per sample (day 0 is the anchor and is
            # always valid, so it is the one predicted)
            h = self.day_in(feat) + self.day_pos[:, :K]
            keep = torch.ones(B, K, 1, device=h.device)
            keep[:, 0] = 0.0
            h = h * keep + self.day_mask_token * (1.0 - keep)
            # padded slots must not inform the prediction either
            h = h * kv.unsqueeze(-1) + self.day_mask_token * (1.0 - kv).unsqueeze(-1)
            pred = self.day_out(self.day_encoder(h)[:, 0])

            # only subjects with >=2 real days carry signal
            w = (kv[:, 1:].sum(1) > 0).float()
            per = F.smooth_l1_loss(pred, zt[:, 0].detach(),
                                   beta=pc.smooth_l1_beta, reduction="none").mean(-1)
            loss_dj = (w * per).sum() / (w.sum() + EPS)
            out["loss_dayjepa"] = loss_dj
            total = total + cfg.w_dayjepa * loss_dj

        out["loss"] = total
        out["sigma"] = self.online.filter.sigma.detach()
        return out

    @torch.no_grad()
    def encode_subject(self, zT: torch.Tensor, zS: torch.Tensor) -> torch.Tensor:
        """Aggregate one subject's day blocks with the TRAINED day encoder.

        zT [N, dT], zS [N, dS] for N windows of one subject -> [dT + D].
        Used by E1d as a learned replacement for the fixed [mean|sd|p10|p90].
        """
        if getattr(self, "day_encoder", None) is None:
            raise RuntimeError("checkpoint has no day encoder")
        K = self.cfg.n_days
        feat = torch.cat([zT, zS], -1).unsqueeze(0)                  # [1, N, dT+dS]
        h = self.day_in(feat)
        n = h.shape[1]
        pos = self.day_pos[:, :K]
        # tile the learned day positions when a subject has more days than K
        reps = int(math.ceil(n / K))
        h = h + pos.repeat(1, reps, 1)[:, :n]
        ctx = self.day_encoder(torch.cat(
            [self.day_mask_token, h], dim=1))                        # query first
        return torch.cat([self.day_out(ctx[:, 0]), ctx[:, 1:].mean(1)], -1)[0]


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def build(fm_cfg: GlucoFMConfig | None = None, cfg: PrismConfig | None = None,
          verbose: bool = True) -> GlucoPRISM:
    fm_cfg = fm_cfg or GlucoFMConfig()
    cfg = cfg or PrismConfig()
    model = GlucoPRISM(fm_cfg, cfg)
    if verbose:
        tr, to = count_params(model)
        tag = "GlucoPRISM-HJEPA" if cfg.hierarchical else "GlucoPRISM"
        print(f"  {tag}: trainable={tr:,}  total={to:,}  "
              f"(proposal target ~0.9M trainable)")
    return model
