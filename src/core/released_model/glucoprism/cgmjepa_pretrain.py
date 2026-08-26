"""
Pretrain CGM-JEPA / X-CGM-JEPA on OUR corpus.

The released encoders were pretrained on CGM-JEPA's own corpus (22 Stanford +
206 Colas subjects), so comparing them against models trained on ours is the
same uncontrolled comparison that made the first GlucoFM baseline meaningless.
This trains their architecture and their objective on `corpus_balanced`, exactly
as `b3-glucofm-fix` does for GlucoFM.

Faithful to `CGM-JEPA/pretrain/pretrain_{cgm,x_cgm}_jepa.py`:

  * patches of 12 samples over a 288-point day (24 hourly patches)
  * mask_ratio 0.25 -> 6 masked / 18 context patches, resampled per window
  * target = LayerNorm(EMA-encoder(all patches)) gathered at the masked indices
  * context = encoder(patches, mask=non_masked) - JEPA-style, masked patches are
    dropped from the sequence, not token-replaced
  * predictor prepends learnable mask queries + pos/time encodings
  * loss = mean |pred - target| (L1)
  * AdamW lr 1e-4, linear warm-up over 15% of steps then linear decay to 0,
    grad-clip 1.0, EMA momentum 0.997 -> 1.0 linear over epochs
  * X-CGM-JEPA adds a cross-modal term: from the CGM context, predict masked
    patches of the glucodensity KDE image, with a trainable spatial encoder
    providing the targets

Three documented departures, each for a stated reason:

1. `apply_mask(x, masks)` in `pretrain_cgm_jepa.py` passes a (B, K) TENSOR where
   the helper expects a LIST of (B, K) tensors. Iterating the tensor yields (K,)
   rows, which `repeat(1, 1, D)` promotes to (1, K, D), so `torch.gather` reads
   **batch element 0 for the whole batch** - verified directly. Every target and
   every context in a batch would come from one sample. We use the corrected
   form `apply_mask(x, [masks])`, which is what the newer `pretrain_x_cgm_jepa.py`
   already does. Training the buggy form would produce an unfairly weak baseline.

2. `normalize_x` is True here, not False. `config_pretrain.py` sets it False
   ("use raw values"), but measured on all 18 downstream cells the released
   weights score better under normalised input (65.50 vs 64.30 ROC-AUC;
   `results/cgmjepa_input_convention.csv`), and their own evaluation script
   defaults `normalize_for_encoder=True`. We give the baseline its better
   setting.

3. Our windows have missing samples; CGM-JEPA's corpus does not. Missing
   positions carry the `-1` sentinel their own transformer uses, and the
   glucodensity spline is fitted on observed positions only.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = dict(
    patch_size=12, length=288, mask_ratio=0.25,
    embed_dim=96, nhead=6, num_layers=3, kernel_size=3, embed_bias=True,
    dropout=0.0, predictor_embed=48, predictor_nhead=2, predictor_num_layers=1,
    time_inp_dim=5, lr=1e-4, batch_size=128, epochs=101, ema_momentum=0.997,
    ipe_scale=1.25, warmup_ratio=0.15, clip=1.0, seed=43,
    normalize=True, time_features=True,
    gluco_patch_size=8, gluco_gridsize=32, gluco_in_channels=3,
    gluco_loss_weight=1.0,
)


def _src(path: Path | None = None):
    from . import cgmjepa
    base = Path(path) if path else cgmjepa.CGM_JEPA_SRC
    for p in (base, base / "models"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return base


# --------------------------------------------------------------------------- #
class JepaWindowDataset(Dataset):
    """Our 288-point windows in CGM-JEPA's patch/mask format."""

    def __init__(self, corpus: dict, cfg: dict, cross_modal: bool = False,
                 gluco: np.ndarray | None = None):
        self.c, self.cfg = corpus, cfg
        self.P = cfg["length"] // cfg["patch_size"]
        self.K = cfg["patch_size"]
        self.n_mask = int(self.P * cfg["mask_ratio"])
        self.rng = np.random.default_rng(cfg["seed"])
        self.cross_modal, self.gluco = cross_modal, gluco
        obs = corpus["m"].astype(bool)
        v = np.nan_to_num(corpus["x"].astype(np.float64))[obs]
        self.mean, self.std = float(v.mean()), float(v.std())

    def __len__(self):
        return len(self.c["x"])

    def _time_feat(self, s: int) -> np.ndarray:
        j = np.arange(self.cfg["length"])
        abs_i = (s + j) % self.cfg["length"]
        mod = abs_i * 5
        f = np.zeros((self.cfg["length"], 5), dtype=np.float32)
        if self.cfg["time_features"]:
            f[:, 0] = (mod % 60) / 59.0 - 0.5          # MinuteOfHour
            f[:, 1] = (mod // 60) / 23.0 - 0.5         # HourOfDay
        return f.reshape(self.P, self.K, 5)

    def __getitem__(self, i: int):
        x = np.nan_to_num(self.c["x"][i].astype(np.float32))
        m = self.c["m"][i].astype(bool)
        v = (x - self.mean) / self.std if self.cfg["normalize"] else x
        v = np.where(m, v, -1.0).astype(np.float32)     # CGM-JEPA sentinel

        idx = self.rng.permutation(self.P)
        mask_idx = np.sort(idx[:self.n_mask])
        non_mask_idx = np.sort(idx[self.n_mask:])

        out = [torch.from_numpy(v.reshape(self.P, self.K)),
               torch.from_numpy(self._time_feat(int(self.c["s"][i]))),
               torch.from_numpy(mask_idx.astype(np.int64)),
               torch.from_numpy(non_mask_idx.astype(np.int64))]
        if self.cross_modal:
            out.append(torch.from_numpy(self.gluco[i]))
        return tuple(out)


# --------------------------------------------------------------------------- #
def compute_glucodensity(corpus: dict, cfg: dict, verbose: bool = True) -> np.ndarray:
    """KDE glucodensity patches for every window, mask-aware.

    `compute_glucodensity_grids` fits a spline on a uniform 288-point time axis;
    our windows have gaps, so the spline is fitted on OBSERVED positions only and
    evaluated on the same grid. Windows with too few observations fall back to
    zeros (the value a masked spatial patch takes in their own pipeline).
    """
    _src()
    from utils.glucodensity_utils import (compute_2d_kde_grid,  # noqa: E402
                                          create_spatial_patches_from_grid)
    from scipy import interpolate

    n = len(corpus["x"])
    P = cfg["gluco_gridsize"] // cfg["gluco_patch_size"]
    out = np.zeros((n, P * P, cfg["gluco_patch_size"], cfg["gluco_patch_size"],
                    cfg["gluco_in_channels"]), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        x = np.nan_to_num(corpus["x"][i].astype(np.float64))
        m = corpus["m"][i].astype(bool)
        if m.sum() < 32:
            continue
        t_all = np.arange(len(x)) * 5.0 / 60.0
        try:
            sp = interpolate.UnivariateSpline(t_all[m], x[m], s=1.0)
            G, dG, ddG = sp(t_all), sp.derivative(1)(t_all), sp.derivative(2)(t_all)
            grids = np.stack([compute_2d_kde_grid(G, dG, cfg["gluco_gridsize"])[0],
                              compute_2d_kde_grid(G, ddG, cfg["gluco_gridsize"])[0],
                              compute_2d_kde_grid(dG, ddG, cfg["gluco_gridsize"])[0]],
                             axis=-1)
            out[i] = create_spatial_patches_from_grid(grids, cfg["gluco_patch_size"])
        except Exception:      # noqa: BLE001 - degenerate windows stay zero
            continue
        if verbose and (i + 1) % 1000 == 0:
            print(f"    glucodensity {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    if verbose:
        print(f"    glucodensity done: {out.shape} in {time.time()-t0:.0f}s")
    return out


# --------------------------------------------------------------------------- #
def pretrain(corpus_path: Path, out_dir: Path, cfg: dict | None = None,
             cross_modal: bool = False, device: str | None = None,
             log_every: int = 10) -> dict:
    from .cgmjepa import _import_encoder
    from .data import load_corpus

    cfg = {**DEFAULTS, **(cfg or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    corpus = load_corpus(Path(corpus_path))
    print(f"  corpus: {len(corpus['x']):,} windows | "
          f"{len(set(corpus['subj'].tolist())):,} subjects")

    gluco = compute_glucodensity(corpus, cfg) if cross_modal else None
    ds = JepaWindowDataset(corpus, cfg, cross_modal, gluco)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)

    _src()
    Encoder = _import_encoder()
    from models.predictor import Predictor  # noqa: E402
    from utils.mask_utils import apply_mask  # noqa: E402

    def init_weights(mod):
        """Verbatim from CGM-JEPA/utils/main_utils.py.

        Copied rather than imported: `main_utils` imports wandb at module level,
        which is not installed in the offline Kaggle kernel and is not needed.
        """
        if isinstance(mod, nn.Linear):
            nn.init.trunc_normal_(mod.weight, std=0.02)
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0)
        elif isinstance(mod, nn.LayerNorm):
            nn.init.constant_(mod.bias, 0)
            nn.init.constant_(mod.weight, 1.0)

    enc = Encoder(dim_in=cfg["patch_size"], kernel_size=cfg["kernel_size"],
                  embed_dim=cfg["embed_dim"], embed_bias=cfg["embed_bias"],
                  nhead=cfg["nhead"], num_layers=cfg["num_layers"], jepa=True,
                  time_inp_dim=cfg["time_inp_dim"], drop_rate=cfg["dropout"])
    pred = Predictor(encoder_embed_dim=cfg["embed_dim"],
                     predictor_embed_dim=cfg["predictor_embed"],
                     nhead=cfg["predictor_nhead"],
                     num_layers=cfg["predictor_num_layers"],
                     time_inp_dim=cfg["time_inp_dim"])
    for mod in enc.modules():
        init_weights(mod)
    for mod in pred.modules():
        init_weights(mod)

    parts = [enc, pred]
    g_enc = g_pred = None
    if cross_modal:
        from models.glucodensity_spatial_encoder import GlucodensitySpatialEncoder
        from models.cross_modal_predictor import CrossModalPredictor
        npatch = (cfg["gluco_gridsize"] // cfg["gluco_patch_size"]) ** 2
        g_enc = GlucodensitySpatialEncoder(
            num_gluco_patches=npatch, gluco_patch_size=cfg["gluco_patch_size"],
            gluco_in_channels=cfg["gluco_in_channels"], embed_dim=cfg["embed_dim"],
            embed_bias=cfg["embed_bias"], nhead=cfg["nhead"],
            num_layers=cfg["num_layers"], jepa=True, drop_rate=cfg["dropout"])
        g_pred = CrossModalPredictor(
            num_gluco_patches=npatch, encoder_embed_dim=cfg["embed_dim"],
            predictor_embed_dim=cfg["predictor_embed"],
            nhead=cfg["predictor_nhead"], num_layers=cfg["predictor_num_layers"])
        for mod in list(g_enc.modules()) + list(g_pred.modules()):
            init_weights(mod)
        parts += [g_enc, g_pred]

    parts = [p.to(device) for p in parts]
    enc, pred = parts[0], parts[1]
    if cross_modal:
        g_enc, g_pred = parts[2], parts[3]

    opt = torch.optim.AdamW([{"params": p.parameters()} for p in parts],
                            lr=cfg["lr"])
    enc_ema = copy.deepcopy(enc)
    for q in enc_ema.parameters():
        q.requires_grad = False

    steps_per_epoch = max(len(dl), 1)
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup = int(cfg["warmup_ratio"] * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, 1.0 - (step - warmup) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    n_ema = int(cfg["epochs"] * cfg["ipe_scale"])
    ema_vals = [cfg["ema_momentum"] + i * (1 - cfg["ema_momentum"]) / n_ema
                for i in range(n_ema + 1)]

    tag = "X-CGM-JEPA" if cross_modal else "CGM-JEPA"
    n_par = sum(p.numel() for p in enc.parameters())
    print(f"  {tag}: encoder {n_par:,} params | device={device} | "
          f"epochs={cfg['epochs']} steps/epoch={steps_per_epoch} "
          f"| normalize={cfg['normalize']} time_features={cfg['time_features']}")

    hist, t0 = [], time.time()
    for ep in range(cfg["epochs"]):
        mm = ema_vals[min(ep, len(ema_vals) - 1)]
        for p in parts:
            p.train()
        agg = {"loss": 0.0, "cgm": 0.0, "gluco": 0.0, "n": 0}
        for batch in dl:
            patches, tmark, masks, non_masks = [b.to(device) for b in batch[:4]]
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                tgt, _ = enc_ema(patches, tmark)
                tgt = F.layer_norm(tgt, (tgt.size(-1),))
                tgt = apply_mask(tgt, [masks])          # corrected form, see docstring

            tok, _ = enc(patches, tmark, mask=[non_masks])
            p_cgm = pred(tok, x_mark=tmark, masks=[masks], non_masks=[non_masks])
            loss_cgm = (p_cgm - tgt).abs().mean()
            loss = loss_cgm
            loss_g = torch.zeros((), device=device)

            if cross_modal:
                gp = batch[4].to(device)
                B, npat = gp.shape[0], gp.shape[1]
                k = max(1, int(npat * cfg["mask_ratio"]))
                perm = torch.randperm(npat, device=device)
                gm = perm[:k].unsqueeze(0).repeat(B, 1)
                gn = perm[k:].unsqueeze(0).repeat(B, 1)
                g_t, _ = g_enc(gp, gluco_mask=None)
                g_t = F.layer_norm(g_t, (g_t.size(-1),))
                g_t = apply_mask(g_t, [gm])
                # CrossModalPredictor does its own gather and expects a plain
                # (B, K) tensor, unlike apply_mask which expects a list of them
                p_g = g_pred(cgm_context=tok, gluco_masks=gm, gluco_non_masks=gn)
                loss_g = (p_g - g_t).abs().mean()
                loss = loss + cfg["gluco_loss_weight"] * loss_g

            loss.backward()
            nn.utils.clip_grad_norm_(
                [q for p in parts for q in p.parameters()], cfg["clip"])
            opt.step()
            sched.step()
            with torch.no_grad():
                for a, b in zip(enc.parameters(), enc_ema.parameters()):
                    b.data.mul_(mm).add_((1.0 - mm) * a.detach().data)

            bs = patches.shape[0]
            agg["loss"] += float(loss) * bs
            agg["cgm"] += float(loss_cgm) * bs
            agg["gluco"] += float(loss_g) * bs
            agg["n"] += bs

        rec = {"epoch": ep + 1, "loss": agg["loss"] / max(agg["n"], 1),
               "loss_cgm": agg["cgm"] / max(agg["n"], 1),
               "loss_gluco": agg["gluco"] / max(agg["n"], 1),
               "lr": opt.param_groups[0]["lr"], "ema_m": mm,
               "elapsed_s": round(time.time() - t0, 1)}
        hist.append(rec)
        if (ep + 1) % log_every == 0 or ep == 0 or ep == cfg["epochs"] - 1:
            print(f"    ep {ep+1:>4}/{cfg['epochs']} loss={rec['loss']:.4f} "
                  f"cgm={rec['loss_cgm']:.4f} gluco={rec['loss_gluco']:.4f} "
                  f"lr={rec['lr']:.2e} ({rec['elapsed_s']:.0f}s)")

    if len(hist) >= 5:
        a, b = hist[0]["loss"], hist[-1]["loss"]
        rel = abs(b - a) / max(abs(a), 1e-9)
        if rel < 1e-3:
            print(f"  ** WARNING: flat loss ({a:.6f} -> {b:.6f}); did not train")
        print(f"  training check: loss {a:.4f} -> {b:.4f} "
              f"({100*(1-b/max(a,1e-9)):.1f}% reduction)")

    torch.save(enc.state_dict(), out_dir / "cgmjepa.pt")
    (out_dir / "config.json").write_text(json.dumps({
        "dim_in": cfg["patch_size"], "kernel_size": cfg["kernel_size"],
        "embed_dim": cfg["embed_dim"], "embed_bias": cfg["embed_bias"],
        "nhead": cfg["nhead"], "num_layers": cfg["num_layers"],
        "mlp_ratio": 4.0, "qkv_bias": True, "qk_scale": None,
        "drop_rate": 0.0, "attn_drop_rate": 0.0, "jepa": False,
        "time_inp_dim": cfg["time_inp_dim"],
        "normalize": cfg["normalize"], "time_features": cfg["time_features"],
    }, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"  saved -> {out_dir/'cgmjepa.pt'}")
    return {"history": hist, "checkpoint": str(out_dir / "cgmjepa.pt")}
