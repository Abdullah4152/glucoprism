"""
GlucoFM pretraining loop (section 4, Appendix C.5).

120 epochs, batch 128, AdamW lr 1e-4, weight decay 1e-2, and a separate
learning rate of 1e-3 for the learnable Gaussian bandwidth. EMA target momentum
starts at 0.997.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, DEFAULT
from .data import PretrainDataset, load_windows
from .model import GlucoFM, build_model


def build_optimizer(model: GlucoFM, cfg: Config):
    """Separate parameter group for the learnable Gaussian bandwidth (lr 1e-3)."""
    p = cfg.pretrain
    sigma_params, other = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (sigma_params if name.endswith(".rho") else other).append(prm)
    groups = [
        {"params": other, "lr": p.lr, "weight_decay": p.weight_decay},
        {"params": sigma_params, "lr": p.lr_sigma, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups)


def lr_scale(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def ema_momentum(step: int, total: int, cfg: Config) -> float:
    p = cfg.pretrain
    if not p.ema_schedule:
        return p.ema_momentum
    prog = min(step / max(total, 1), 1.0)
    return p.ema_final - (p.ema_final - p.ema_momentum) * (
        0.5 * (1.0 + math.cos(math.pi * prog)))


def pretrain(window_files: list[Path], out_dir: Path, cfg: Config = DEFAULT,
             device: str | None = None, epochs: int | None = None,
             log_every: int = 10) -> dict:
    """Run pretraining and write checkpoint + history into out_dir."""
    p = cfg.pretrain
    epochs = epochs or p.epochs
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(p.seed)
    np.random.seed(p.seed)

    data = load_windows(window_files)
    print(f"  corpus: {len(data['x']):,} windows | "
          f"{len(set(data['subj'])):,} subjects | "
          f"{len(set(data['cohort']))} cohorts")
    for c in sorted(set(data["cohort"])):
        sel = data["cohort"] == c
        print(f"     {c:<18} {int(sel.sum()):>6} windows  "
              f"{len(set(data['subj'][sel])):>4} subjects  "
              f"observed {data['m'][sel].mean():5.1%}")

    ds = PretrainDataset(data, cfg, train=True, seed=p.seed)
    dl = DataLoader(ds, batch_size=p.batch_size, shuffle=True, drop_last=False,
                    num_workers=0, pin_memory=(device == "cuda"))

    model = build_model(cfg).to(device)
    opt = build_optimizer(model, cfg)
    total_steps = epochs * max(len(dl), 1)
    warmup = int(p.warmup_frac * total_steps)
    base_lrs = [g["lr"] for g in opt.param_groups]

    print(f"  device={device}  epochs={epochs}  steps/epoch={len(dl)}  "
          f"total_steps={total_steps}")

    hist, step, t0 = [], 0, time.time()
    for ep in range(epochs):
        model.train()
        agg = {"loss": 0.0, "loss_mcr": 0.0, "loss_td": 0.0, "n": 0}
        for batch in dl:
            scale = lr_scale(step, total_steps, warmup)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * scale

            x = batch["x"].to(device, non_blocking=True)
            m = batch["m"].to(device, non_blocking=True)
            s = batch["s"].to(device, non_blocking=True)
            pm = batch["patch_mask"].to(device, non_blocking=True)

            out = model(x, m, s, pm)
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [q for q in model.parameters() if q.requires_grad], p.grad_clip)
            opt.step()
            model.update_target(ema_momentum(step, total_steps, cfg))

            bs = x.shape[0]
            agg["loss"] += loss.item() * bs
            agg["loss_mcr"] += out["loss_mcr"].item() * bs
            agg["loss_td"] += out["loss_td"].item() * bs
            agg["n"] += bs
            step += 1

        rec = {
            "epoch": ep + 1,
            "loss": agg["loss"] / max(agg["n"], 1),
            "loss_mcr": agg["loss_mcr"] / max(agg["n"], 1),
            "loss_td": agg["loss_td"] / max(agg["n"], 1),
            "sigma": float(model.online.filter.sigma.detach().cpu()),
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": round(time.time() - t0, 1),
        }
        hist.append(rec)
        if (ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1:
            print(f"    ep {ep+1:>4}/{epochs}  loss={rec['loss']:.4f}  "
                  f"mcr={rec['loss_mcr']:.4f}  td={rec['loss_td']:.4f}  "
                  f"sigma={rec['sigma']:.3f}  ({rec['elapsed_s']:.0f}s)")

    ckpt = {
        "online": model.online.state_dict(),
        "config": cfg.to_dict(),
        "history": hist,
        "corpus": {
            "windows": int(len(data["x"])),
            "subjects": int(len(set(data["subj"]))),
            "cohorts": sorted(set(data["cohort"].tolist())),
        },
    }
    torch.save(ckpt, out_dir / "glucofm_encoder.pt")
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"  saved -> {out_dir/'glucofm_encoder.pt'}")
    return {"history": hist, "checkpoint": str(out_dir / "glucofm_encoder.pt")}
