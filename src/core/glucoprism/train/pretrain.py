"""Self-supervised pretraining loops for the three reproduced models.

Each loop follows the optimiser recipe its own paper states:

  GlucoFM         120 epochs, batch 128, AdamW lr 1e-4, weight decay 1e-2, and a
                  *separate* lr of 1e-3 for the learnable Gaussian bandwidth
                  (paper Sec. 4.1). Losses L_MCR + L_TD, both weighted 1.0.
                  CGM-aware augmentation on.

  CGM-JEPA /      101 epochs, batch 128, lr 1e-4, mask ratio 0.25, EMA momentum
  X-CGM-JEPA      0.997 on a linear schedule to 1.0, warmup 15% then linear decay,
                  grad-norm clip 1.0, seed 43. L1 latent loss; for X, plus the
                  Glucodensity cross-view loss at lambda = 1.0.
                  (Paper M.5.2; optimiser details cross-checked against the
                  official `pretrain/pretrain_x_cgm_jepa.py`, which uses AdamW and
                  a warmup+linear-decay LambdaLR rather than the StepLR named in
                  the text -- we follow the code and flag it in the repro report.)

  GluFormer tiny  100 epochs, batch 128, Adam lr 1e-4, causal next-token
                  cross-entropy with PAD ignored (GlucoFM App. B.3).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.cgm_jepa import build_cgm_jepa, build_x_cgm_jepa, jepa_l1_loss
from ..models.blocks import apply_mask, init_weights
from ..models.glucofm import GlucoFM, GlucoFMConfig, sample_patch_mask
from ..models.gluformer import GluFormer, GluFormerConfig


def get_device() -> torch.device:
    """CUDA if it actually works, else CPU.

    `torch.cuda.is_available()` returns True on Kaggle's Tesla P100 (sm_60) even
    though the installed wheel is built for sm_70+, and every kernel launch then
    fails at runtime. Probe with a real op instead of trusting the flag. These
    models are 0.5-0.7 M parameters, so the CPU fallback is a slowdown, not a
    blocker.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(8, device="cuda").add_(1).sum().item()
        return torch.device("cuda")
    except Exception as e:  # noqa: BLE001
        print(f"[device] CUDA present but unusable ({type(e).__name__}: {e}); using CPU")
        return torch.device("cpu")


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class History:
    def __init__(self, out_dir: Path, name: str):
        self.path = Path(out_dir) / f"{name}_history.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, **kw):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\n")
        print("  " + "  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in kw.items()))


# ------------------------------------------------------------------- GlucoFM

def pretrain_glucofm(dataset, out_dir: str | Path, *, cfg: GlucoFMConfig | None = None,
                     epochs: int = 120, batch_size: int = 128, lr: float = 1e-4,
                     weight_decay: float = 1e-2, sigma_lr: float = 1e-3,
                     seed: int = 0, num_workers: int = 0, device=None,
                     log_every: int = 1) -> Path:
    cfg = cfg or GlucoFMConfig()
    device = device or get_device()
    seed_everything(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hist = History(out_dir, "glucofm")

    model = GlucoFM(cfg).to(device)

    # Sec. 4.1: "a separate learning rate of 1e-3 for the learnable Gaussian bandwidth".
    sigma_params = [p for n, p in model.named_parameters() if n.endswith("filter.rho")]
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.endswith("filter.rho")]
    opt = torch.optim.AdamW(
        [{"params": other, "lr": lr, "weight_decay": weight_decay},
         {"params": sigma_params, "lr": sigma_lr, "weight_decay": 0.0}])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # EMA momentum ramps 0.997 -> 1.0 over training, as in the JEPA reference schedule.
    total_steps = max(epochs * len(loader), 1)
    step = 0
    for ep in range(epochs):
        model.train()
        agg = {"loss": 0.0, "loss_mcr": 0.0, "loss_td": 0.0}
        t0 = time.time()
        for g, m, s in loader:
            g, m, s = g.to(device), m.to(device), s.to(device)
            pm = sample_patch_mask(g.size(0), cfg, device)

            out = model(g, m, s, pm)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            mom = cfg.ema_momentum + (1.0 - cfg.ema_momentum) * step / total_steps
            model.ema_update(mom)
            step += 1
            for k in agg:
                agg[k] += float(out[k])

        n = max(len(loader), 1)
        if ep % log_every == 0 or ep == epochs - 1:
            hist.log(epoch=ep, **{k: v / n for k, v in agg.items()},
                     sigma=float(model.online.filter.sigma), secs=round(time.time() - t0, 1))

    ckpt = out_dir / "glucofm.pt"
    torch.save({"model": model.state_dict(), "online": model.online.state_dict(),
                "cfg": asdict(cfg), "epochs": epochs, "seed": seed}, ckpt)
    print(f"saved {ckpt}")
    return ckpt


# ------------------------------------------------------------------- GlucoCQP

def pretrain_cqp(dataset, out_dir: str | Path, *, cfg=None,
                 epochs: int = 120, batch_size: int = 128, lr: float = 1e-4,
                 weight_decay: float = 1e-2, sigma_lr: float = 1e-3,
                 seed: int = 0, num_workers: int = 0, device=None,
                 log_every: int = 1) -> Path:
    """Clinical Query Pooling. Optimiser is GlucoFM's Sec. 4.1, unchanged, so the
    only difference against the baseline is the pooling and the L_CMP term."""
    from ..models.cqp import CQPConfig, GlucoCQP

    cfg = cfg or CQPConfig()
    device = device or get_device()
    seed_everything(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hist = History(out_dir, "cqp")

    model = GlucoCQP(cfg).to(device)
    sigma_params = [p for n, p in model.named_parameters() if n.endswith("filter.rho")]
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.endswith("filter.rho")]
    opt = torch.optim.AdamW(
        [{"params": other, "lr": lr, "weight_decay": weight_decay},
         {"params": sigma_params, "lr": sigma_lr, "weight_decay": 0.0}])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    total_steps = max(epochs * len(loader), 1)
    step = 0
    keys = ("loss", "loss_mcr", "loss_td", "loss_cmp", "cos_z", "std_z")

    for ep in range(epochs):
        model.train()
        agg = {k: 0.0 for k in keys}
        t0 = time.time()
        for g, m, s in loader:
            g, m, s = g.to(device), m.to(device), s.to(device)
            pm = sample_patch_mask(g.size(0), cfg.fm, device)
            out = model(g, m, s, pm)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            mom = cfg.fm.ema_momentum + (1.0 - cfg.fm.ema_momentum) * step / total_steps
            model.ema_update(mom)
            step += 1
            for k in keys:
                v = out[k]
                agg[k] += float(v.detach() if torch.is_tensor(v) else v)

        n = max(len(loader), 1)
        if ep % log_every == 0 or ep == epochs - 1:
            hist.log(epoch=ep, **{k: agg[k] / n for k in keys},
                     sigma=float(model.fm.online.filter.sigma),
                     secs=round(time.time() - t0, 1))

    ckpt = out_dir / "cqp.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                "epochs": epochs, "seed": seed}, ckpt)
    print(f"saved {ckpt}")
    return ckpt


# ----------------------------------------------------------------- GlucoPRISM

def pretrain_prism(dataset, out_dir: str | Path, *, cfg=None,
                   epochs: int = 120, batch_size: int = 128, lr: float = 1e-4,
                   weight_decay: float = 1e-2, sigma_lr: float = 1e-3,
                   seed: int = 0, num_workers: int = 0, device=None,
                   log_every: int = 1) -> Path:
    """GlucoPRISM pretraining (proposal Sec. 4).

    Optimiser settings are inherited from GlucoFM Sec. 4.1 unchanged -- AdamW,
    lr 1e-4, weight decay 1e-2, and a separate lr of 1e-3 with no decay for the
    learnable Gaussian bandwidth -- because GlucoPRISM is that backbone plus
    protocol objectives, and changing two things at once would make the
    comparison against the GlucoFM baseline uninterpretable.

    Per-epoch logging carries the collapse monitor (`cos_*`, `std_*`). The
    alignment terms of Eqs. 2-3 are minimised by a fully collapsed
    representation, so those two numbers are what tell us whether the
    day- and sensor-discriminative terms are actually holding the blocks apart.
    """
    from ..data.views import collate
    from ..models.prism import GlucoPRISM, PrismConfig

    cfg = cfg or PrismConfig()
    device = device or get_device()
    seed_everything(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hist = History(out_dir, "prism")

    model = GlucoPRISM(cfg).to(device)

    sigma_params = [p for n, p in model.named_parameters() if n.endswith("filter.rho")]
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.endswith("filter.rho")]
    opt = torch.optim.AdamW(
        [{"params": other, "lr": lr, "weight_decay": weight_decay},
         {"params": sigma_params, "lr": sigma_lr, "weight_decay": 0.0}])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"),
                        collate_fn=lambda items: collate(items, cfg.fm))

    total_steps = max(epochs * len(loader), 1)
    step = 0
    keys = ("loss", "loss_mcr", "loss_td", "loss_sensor", "loss_device",
            "loss_day", "mi_day", "loss_indep",
            "cos_zT", "cos_zS", "cos_zA", "std_zT", "std_zS", "std_zA")

    for ep in range(epochs):
        model.train()
        agg = {k: 0.0 for k in keys}
        t0 = time.time()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            mom = cfg.fm.ema_momentum + (1.0 - cfg.fm.ema_momentum) * step / total_steps
            model.ema_update(mom)
            step += 1
            for k in keys:
                v = out[k]
                agg[k] += float(v.detach() if torch.is_tensor(v) else v)

        n = max(len(loader), 1)
        if ep % log_every == 0 or ep == epochs - 1:
            hist.log(epoch=ep, **{k: agg[k] / n for k in keys},
                     sigma=float(model.fm.online.filter.sigma),
                     secs=round(time.time() - t0, 1))

    ckpt = out_dir / "prism.pt"
    torch.save({"model": model.state_dict(), "online": model.fm.online.state_dict(),
                "cfg": cfg.to_dict(), "epochs": epochs, "seed": seed}, ckpt)
    print(f"saved {ckpt}")
    return ckpt


# ------------------------------------------------------- CGM-JEPA / X-CGM-JEPA

CGM_JEPA_CFG = dict(
    patch_size=12, encoder_kernel_size=3, encoder_embed_dim=96, encoder_embed_bias=True,
    encoder_nhead=6, encoder_num_layers=3, encoder_dropout=0.0,
    predictor_embed=48, predictor_nhead=2, predictor_num_layers=1, time_inp_dim=5,
    gluco_gridsize=32, gluco_spatial_patch_size=8,
)


def _jepa_lr_lambda(warmup_steps: int, total_steps: int):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - prog)
    return fn


def pretrain_cgm_jepa(dataset, out_dir: str | Path, *, cross_view: bool = False,
                      model_cfg: dict | None = None, epochs: int = 101,
                      batch_size: int = 128, lr: float = 1e-4, mask_ratio: float = 0.25,
                      gluco_loss_weight: float = 1.0, ema_momentum: float = 0.997,
                      warmup_ratio: float = 0.15, ipe_scale: float = 1.25,
                      clip_grad: float = 1.0, seed: int = 43, num_workers: int = 0,
                      device=None, log_every: int = 1) -> Path:
    mcfg = {**CGM_JEPA_CFG, **(model_cfg or {})}
    device = device or get_device()
    seed_everything(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    name = "x_cgm_jepa" if cross_view else "cgm_jepa"
    hist = History(out_dir, name)

    if cross_view:
        ctx, tgt, pred, genc, xpred = build_x_cgm_jepa(mcfg)
        modules = [ctx, genc, pred, xpred]
    else:
        ctx, tgt, pred = build_cgm_jepa(mcfg)
        genc = xpred = None
        modules = [ctx, pred]

    for mod in modules + [tgt]:
        for sub in mod.modules():
            init_weights(sub)
    tgt.load_state_dict(ctx.state_dict())
    for p in tgt.parameters():
        p.requires_grad = False

    for mod in modules + [tgt]:
        mod.to(device)

    opt = torch.optim.AdamW([{"params": m.parameters()} for m in modules], lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    total_steps = max(epochs * len(loader), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _jepa_lr_lambda(int(warmup_ratio * total_steps), total_steps))

    n_ema = int(epochs * ipe_scale) + 1
    ema_sched = iter([ema_momentum + i * (1 - ema_momentum) / (epochs * ipe_scale)
                      for i in range(n_ema)])

    n_gluco = (mcfg["gluco_gridsize"] // mcfg["gluco_spatial_patch_size"]) ** 2
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        m_ema = next(ema_sched)
        for mod in modules:
            mod.train()
        agg = {"loss": 0.0, "loss_cgm": 0.0, "loss_gluco": 0.0}
        t0 = time.time()

        for batch in loader:
            if cross_view:
                patches, tmark, masks, non_masks, gluco = batch
                gluco = gluco.to(device)
            else:
                patches, tmark, masks, non_masks = batch
            patches, tmark = patches.to(device), tmark.to(device)
            masks, non_masks = masks.to(device), non_masks.to(device)
            B = patches.size(0)

            with torch.no_grad():
                t_cgm, _ = tgt(patches, tmark, mask=None)
                t_cgm = torch.nn.functional.layer_norm(t_cgm, (t_cgm.size(-1),))
                t_cgm = apply_mask(t_cgm, [masks])

            ctx_tok, _ = ctx(patches, tmark, mask=non_masks)
            p_cgm = pred(ctx_tok, tmark, masks=masks, non_masks=non_masks)
            loss_cgm = jepa_l1_loss(p_cgm, t_cgm)
            loss = loss_cgm
            loss_gluco = torch.zeros((), device=device)

            if cross_view:
                n_mask_g = int(n_gluco * mask_ratio)
                gm = np.sort(rng.choice(n_gluco, size=n_mask_g, replace=False))
                gnm = np.setdiff1d(np.arange(n_gluco), gm)
                gm_t = torch.as_tensor(gm, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
                gnm_t = torch.as_tensor(gnm, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)

                # Gradients DO flow into the Glucodensity encoder (paper M.3.4.2):
                # it is trained jointly through the auxiliary objective.
                t_gl, _ = genc(gluco, gluco_mask=None)
                t_gl = torch.nn.functional.layer_norm(t_gl, (t_gl.size(-1),))
                t_gl = apply_mask(t_gl, [gm_t])
                p_gl = xpred(cgm_context=ctx_tok, gluco_masks=gm_t, gluco_non_masks=gnm_t)
                loss_gluco = jepa_l1_loss(p_gl, t_gl)
                loss = loss_cgm + gluco_loss_weight * loss_gluco

            opt.zero_grad(set_to_none=True)
            loss.backward()
            params = [p for m in modules for p in m.parameters()]
            nn.utils.clip_grad_norm_(params, max_norm=clip_grad)
            opt.step()
            sched.step()

            with torch.no_grad():
                for q, k in zip(ctx.parameters(), tgt.parameters()):
                    k.data.mul_(m_ema).add_((1.0 - m_ema) * q.detach().data)

            agg["loss"] += float(loss)
            agg["loss_cgm"] += float(loss_cgm)
            agg["loss_gluco"] += float(loss_gluco)

        n = max(len(loader), 1)
        if ep % log_every == 0 or ep == epochs - 1:
            hist.log(epoch=ep, **{k: v / n for k, v in agg.items()},
                     lr=opt.param_groups[0]["lr"], ema=m_ema, secs=round(time.time() - t0, 1))

    ckpt = out_dir / f"{name}.pt"
    payload = {"encoder": ctx.state_dict(), "predictor": pred.state_dict(),
               "cfg": mcfg, "epochs": epochs, "seed": seed, "mask_ratio": mask_ratio}
    if cross_view:
        payload["gluco_encoder"] = genc.state_dict()
        payload["cross_predictor"] = xpred.state_dict()
        payload["gluco_loss_weight"] = gluco_loss_weight
    torch.save(payload, ckpt)
    print(f"saved {ckpt}")
    return ckpt


# ----------------------------------------------------------------- GluFormer

def pretrain_gluformer(dataset, out_dir: str | Path, *, variant: str = "tiny",
                       epochs: int = 100, batch_size: int = 128, lr: float | None = None,
                       seed: int = 43, num_workers: int = 0, device=None,
                       log_every: int = 1) -> Path:
    cfg = GluFormerConfig.tiny() if variant == "tiny" else GluFormerConfig.base()
    lr = lr if lr is not None else (1e-4 if variant == "tiny" else 5e-5)
    device = device or get_device()
    seed_everything(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hist = History(out_dir, f"gluformer_{variant}")

    model = GluFormer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    for ep in range(epochs):
        model.train()
        tot, t0 = 0.0, time.time()
        for tokens in loader:
            tokens = tokens.to(device)
            loss = model.loss(tokens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        n = max(len(loader), 1)
        if ep % log_every == 0 or ep == epochs - 1:
            hist.log(epoch=ep, loss=tot / n, ppl=float(np.exp(min(tot / n, 20))),
                     secs=round(time.time() - t0, 1))

    ckpt = out_dir / f"gluformer_{variant}.pt"
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                "variant": variant, "epochs": epochs, "seed": seed}, ckpt)
    print(f"saved {ckpt}")
    return ckpt
