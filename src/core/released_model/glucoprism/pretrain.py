"""GlucoPRISM pretraining loop (inherits GlucoFM's optimisation settings)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from glucofm.config import Config as GlucoFMConfig
from glucofm.pretrain import ema_momentum, lr_scale

from .data import PrismPretrainDataset, load_corpus, make_weighted_sampler
from .model import GlucoPRISM, build, count_params
from .sensor_sim import SensorSimParams


def build_optimizer(model: GlucoPRISM, fm: GlucoFMConfig):
    p = fm.pretrain
    sigma, other = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (sigma if name.endswith(".rho") else other).append(prm)
    return torch.optim.AdamW([
        {"params": other, "lr": p.lr, "weight_decay": p.weight_decay},
        {"params": sigma, "lr": p.lr_sigma, "weight_decay": 0.0},
    ])


def pretrain(corpus_path: Path, out_dir: Path, fm_cfg: GlucoFMConfig,
             prism_cfg, sim_params: SensorSimParams,
             epochs: int | None = None, device: str | None = None,
             sampler_alpha: float | None = None, log_every: int = 5,
             amp: bool = False, augment: bool = False) -> dict:
    # AMP is OFF by default. The model is ~0.7M parameters and a full 120-epoch
    # run takes minutes on a T4, so fp16 buys nothing, while the mask-aware
    # preprocessing is numerically delicate (see masked_instance_norm).
    p = fm_cfg.pretrain
    epochs = epochs or p.epochs
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(p.seed)
    np.random.seed(p.seed)

    corpus = load_corpus(Path(corpus_path))
    print(f"  corpus: {len(corpus['x']):,} windows | "
          f"{len(set(corpus['subj'].tolist())):,} subjects")
    for c in sorted(set(corpus["cohort"].tolist())):
        sel = corpus["cohort"] == c
        print(f"     {c:<14}{int(sel.sum()):>7} windows  "
              f"{len(set(corpus['subj'][sel].tolist())):>4} subjects")

    ds = PrismPretrainDataset(corpus, sim_params, fm_cfg.grid.n_patches,
                              p.mask_ratio_low, p.mask_ratio_high, p.seed, True,
                              augment=fm_cfg.aug if augment else None,
                              n_days=(prism_cfg.n_days
                                      if prism_cfg.use_dayjepa else 1))
    sampler = make_weighted_sampler(corpus, sampler_alpha) if sampler_alpha else None
    dl = DataLoader(ds, batch_size=p.batch_size, shuffle=(sampler is None),
                    sampler=sampler, drop_last=False, num_workers=0,
                    pin_memory=(device == "cuda"))

    model = build(fm_cfg, prism_cfg).to(device)
    opt = build_optimizer(model, fm_cfg)
    total_steps = epochs * max(len(dl), 1)
    warmup = int(p.warmup_frac * total_steps)
    base = [g["lr"] for g in opt.param_groups]
    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"  device={device} amp={use_amp} epochs={epochs} "
          f"steps/epoch={len(dl)} total={total_steps}")

    keys = ("loss", "loss_mcr", "loss_td", "loss_l2", "loss_sensor",
            "loss_day", "loss_indep", "loss_var", "loss_cmp", "loss_xday",
            "loss_agg", "loss_dayjepa", "loss_adv", "loss_vib", "nats_a")
    hist, step, t0 = [], 0, time.time()
    for ep in range(epochs):
        model.train()
        agg = {k: 0.0 for k in keys}
        agg["n"] = 0
        for batch in dl:
            sc = lr_scale(step, total_steps, warmup)
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * sc
            # the adversarial reversal strength ramps with training progress
            model._progress = step / max(total_steps, 1)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch)
                loss = out["loss"]

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [q for q in model.parameters() if q.requires_grad], p.grad_clip)
            scaler.step(opt)
            scaler.update()
            model.update_target(ema_momentum(step, total_steps, fm_cfg))

            bs = batch["x"].shape[0]
            for k in keys:
                if k in out:
                    agg[k] += float(out[k].detach()) * bs
            agg["n"] += bs
            step += 1

        rec = {"epoch": ep + 1,
               **{k: agg[k] / max(agg["n"], 1) for k in keys},
               "sigma": float(model.online.filter.sigma.detach().cpu()),
               "lr": opt.param_groups[0]["lr"],
               "elapsed_s": round(time.time() - t0, 1)}
        hist.append(rec)
        if (ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1:
            print(f"    ep {ep+1:>4}/{epochs} loss={rec['loss']:.4f} "
                  f"mcr={rec['loss_mcr']:.4f} td={rec['loss_td']:.4f} "
                  f"l2={rec['loss_l2']:.4f} sens={rec['loss_sensor']:.4f} "
                  f"day={rec['loss_day']:.4f} ind={rec['loss_indep']:.4f} "
                  f"var={rec['loss_var']:.4f} cmp={rec['loss_cmp']:.4f} "
                  f"xday={rec['loss_xday']:.4f} agg={rec['loss_agg']:.4f} "
                  f"dayj={rec['loss_dayjepa']:.4f} adv={rec['loss_adv']:.4f} " f"nats={rec['nats_a']:.3f} "
                  f"sig={rec['sigma']:.2f} ({rec['elapsed_s']:.0f}s)")

    # Sanity guard: a flat loss means the optimiser never stepped (this exact
    # failure shipped once, from an fp16 overflow in the mask-aware norm).
    if len(hist) >= 5:
        first, last = hist[0]["loss"], hist[-1]["loss"]
        rel = abs(last - first) / max(abs(first), 1e-9)
        sig0, sig1 = hist[0]["sigma"], hist[-1]["sigma"]
        if rel < 1e-3:
            print(f"  ** WARNING: loss is flat ({first:.6f} -> {last:.6f}, "
                  f"rel change {rel:.2e}). The model almost certainly did not "
                  f"train. Check for inf/NaN gradients.")
        if abs(sig1 - sig0) < 1e-6:
            print(f"  ** WARNING: learnable filter bandwidth never moved "
                  f"(sigma = {sig0:.6f}); expect it to adapt away from its init.")
        print(f"  training check: loss {first:.4f} -> {last:.4f} "
              f"({100*(1-last/max(first,1e-9)):.1f}% reduction), "
              f"sigma {sig0:.3f} -> {sig1:.3f}")

    tr, to = count_params(model)
    ckpt = {
        "online": model.online.state_dict(),
        "pool": model.pool.state_dict(),
        "aggregator": model.aggregator.state_dict(),
        "device_head": model.device_head.state_dict(),
        "full": model.state_dict(),
        "fm_config": fm_cfg.to_dict(),
        "prism_config": prism_cfg.to_dict(),
        "sim_params": sim_params.__dict__,
        "history": hist,
        "params": {"trainable": tr, "total": to},
        "corpus": {"windows": int(len(corpus["x"])),
                   "subjects": int(len(set(corpus["subj"].tolist()))),
                   "path": str(corpus_path)},
    }
    torch.save(ckpt, out_dir / "glucoprism.pt")
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"  saved -> {out_dir/'glucoprism.pt'}")
    return {"history": hist, "checkpoint": str(out_dir / "glucoprism.pt")}
