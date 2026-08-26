"""End-to-end fine-tuning of a pretrained encoder on each downstream cell.

WHY. Every number in this literature -- GlucoFM Table 3, CGM-JEPA, and all ~50 of
our own configurations -- is a FROZEN linear probe: 128 fixed features, one
logistic regression, fixed C, 250-900 windows per cell. That protocol is
variance-bound, which is exactly why fifty changes to the representation all
washed out inside a seed-sigma of ~1.0 AUC. Fine-tuning changes what the
classifier is allowed to do rather than what the features are.

FAIRNESS. Applied identically to every model: same folds, same schedule, same
learning rates, same head. The comparison is only meaningful if the protocol is
identical, so nothing here is tuned per-model.

OVERFITTING. At 29-69 subjects this can memorise trivially. Three guards:
  * a LOW encoder learning rate (1e-5) against a normal head rate (1e-3), so the
    backbone drifts rather than being rewritten;
  * a short fixed schedule with no early stopping, because there is no clean
    validation split inside a fold that would not leak;
  * subject-grouped folds, so no subject appears in both halves.

    python scripts/run_finetune.py --models glucofm cgm_jepa --epochs 15
"""

from __future__ import annotations

import os as _os, sys as _sys
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))
for _p in (ROOT / "src" / "core", ROOT / "baselines"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))


import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "core"))
sys.path.insert(0, str(ROOT / "scripts"))

from cgmkit.data.datasets import WindowShard          # noqa: E402
from cgmkit.data.labels import TASK_MATRIX            # noqa: E402
from cgmkit.data.windows import densify               # noqa: E402
from cgmkit.eval.probe import _metrics                # noqa: E402
from cgmkit.train.pretrain import get_device          # noqa: E402

PROCESSED = ROOT / "data" / "processed"
DEVICE = get_device()


# ------------------------------------------------------- per-model adapters

class FineTuneModel(nn.Module):
    """Pretrained encoder + a linear head on its pooled representation."""

    def __init__(self, encoder_fn, d_out: int, n_classes: int):
        super().__init__()
        self.encoder_fn = encoder_fn          # (x, m, s) -> (B, d_out), differentiable
        self.head = nn.Linear(d_out, n_classes)

    def forward(self, x, m, s):
        return self.head(self.encoder_fn(x, m, s))


def build_glucofm(ck: Path):
    from cgmkit.models.glucofm import GlucoFM, GlucoFMConfig
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    model = GlucoFM(GlucoFMConfig(**blob["cfg"]))
    model.load_state_dict(blob["model"])
    online = model.online
    # embed() is decorated no_grad, so pool the tokens directly for a
    # differentiable path.
    fn = lambda x, m, s: online(x, m, s, patch_mask=None)["z"].mean(1)   # noqa: E731
    return online, fn, model.cfg.d_model


def build_cqp(ck: Path):
    from cgmkit.models.cqp import CQPConfig, GlucoCQP
    from cgmkit.models.glucofm import GlucoFMConfig
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    c = dict(blob["cfg"]); c["fm"] = GlucoFMConfig(**c["fm"])
    model = GlucoCQP(CQPConfig(**c))
    model.load_state_dict(blob["model"])

    def fn(x, m, s):
        tok = model.fm.online(x, m, s, patch_mask=None)["z"]
        _codes, flat = model.pool(tok)
        return flat
    return nn.ModuleList([model.fm.online, model.pool]), fn, model.cfg.fm.d_model


def build_prism(ck: Path):
    from cgmkit.models.prism import GlucoPRISM, PrismConfig
    from cgmkit.models.glucofm import GlucoFMConfig
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    c = dict(blob["cfg"]); c["fm"] = GlucoFMConfig(**c.pop("fm"))
    model = GlucoPRISM(PrismConfig(fm=c["fm"], **{k: v for k, v in c.items() if k != "fm"}))
    model.load_state_dict(blob["model"])
    online = model.fm.online
    fn = lambda x, m, s: online(x, m, s, patch_mask=None)["z"].mean(1)   # noqa: E731
    return online, fn, model.cfg.fm.d_model


def build_cgm_jepa(ck: Path):
    from cgmkit.models.cgm_jepa import Encoder
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    enc = Encoder(dim_in=cfg["patch_size"], kernel_size=cfg["encoder_kernel_size"],
                  embed_dim=cfg["encoder_embed_dim"], embed_bias=cfg["encoder_embed_bias"],
                  nhead=cfg["encoder_nhead"], num_layers=cfg["encoder_num_layers"],
                  jepa=True, time_inp_dim=cfg["time_inp_dim"],
                  drop_rate=cfg["encoder_dropout"])
    enc.load_state_dict(blob["encoder"])
    P, K, T = 288 // cfg["patch_size"], cfg["patch_size"], cfg["time_inp_dim"]

    def fn(x, m, s):
        xm = torch.zeros(x.shape[0], P, K, T, device=x.device)
        tok, _ = enc(x.view(-1, P, K), xm, mask=None)
        return tok.mean(1)
    return enc, fn, cfg["encoder_embed_dim"]


BUILDERS = {"glucofm": ("glucofm.pt", build_glucofm),
            "cqp": ("cqp.pt", build_cqp),
            "prism": ("prism.pt", build_prism),
            "cgm_jepa": ("cgm_jepa.pt", build_cgm_jepa),
            "x_cgm_jepa": ("x_cgm_jepa.pt", build_cgm_jepa)}

# CGM-JEPA reads a densified window with a -1 sentinel; the others read the
# masked raw signal. Keeping this per-model is not a protocol difference -- it is
# each encoder's own documented input convention.
DENSIFY = {"cgm_jepa", "x_cgm_jepa"}


def cell_arrays(shard: WindowShard, model: str):
    g = np.nan_to_num(shard.data["glucose"].astype(np.float32), nan=0.0)
    m = shard.data["mask"].astype(np.float32)
    if model in DENSIFY:
        g = np.stack([densify(shard.data["glucose"][i], shard.data["mask"][i])
                      for i in range(len(shard))]).astype(np.float32)
    return g, m, shard.data["start_idx"].astype(np.int64)


def finetune_cell(builder, ck, X, M, S, y, groups, folds, n_classes,
                  epochs, enc_lr, head_lr, batch, seed):
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    yy = np.array([remap[v] for v in y])
    prs, aucs, f1s = [], [], []

    for it_folds in folds:
        for te_s in it_folds:
            te = np.isin(groups, np.asarray(te_s, dtype=str)); tr = ~te
            if tr.sum() == 0 or te.sum() == 0 or len(np.unique(yy[tr])) < 2:
                continue
            torch.manual_seed(seed)
            backbone, fn, d = builder(ck)
            model = FineTuneModel(fn, d, n_classes).to(DEVICE)
            backbone.to(DEVICE)
            opt = torch.optim.AdamW([
                {"params": backbone.parameters(), "lr": enc_lr, "weight_decay": 1e-2},
                {"params": model.head.parameters(), "lr": head_lr, "weight_decay": 1e-2}])
            # class weights, because several cells are 8/56 positives
            cnt = np.bincount(yy[tr], minlength=n_classes).astype(np.float64)
            w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) / n_classes,
                             dtype=torch.float32, device=DEVICE)
            lossf = nn.CrossEntropyLoss(weight=w)

            idx = np.flatnonzero(tr)
            xb = torch.tensor(X[idx]).to(DEVICE)
            mb = torch.tensor(M[idx]).to(DEVICE)
            sb = torch.tensor(S[idx]).to(DEVICE)
            yb = torch.tensor(yy[idx]).long().to(DEVICE)

            model.train(); backbone.train()
            for _ep in range(epochs):
                perm = torch.randperm(len(idx), device=DEVICE)
                for i in range(0, len(idx), batch):
                    b = perm[i:i + batch]
                    opt.zero_grad(set_to_none=True)
                    loss = lossf(model(xb[b], mb[b], sb[b]), yb[b])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            model.eval(); backbone.eval()
            with torch.no_grad():
                jdx = np.flatnonzero(te)
                logits = []
                for i in range(0, len(jdx), 256):
                    k = jdx[i:i + 256]
                    logits.append(model(torch.tensor(X[k]).to(DEVICE),
                                        torch.tensor(M[k]).to(DEVICE),
                                        torch.tensor(S[k]).to(DEVICE)).cpu())
                proba = torch.softmax(torch.cat(logits), dim=-1).numpy()
            r, p, f = _metrics(y[te], proba, classes)
            aucs.append(r); prs.append(p); f1s.append(f)
    return (100 * np.nanmean(prs), 100 * np.nanmean(aucs), 100 * np.nanmean(f1s),
            len(prs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["glucofm", "cgm_jepa"])
    ap.add_argument("--checkpoints", default=str(ROOT / "artifacts" / "checkpoints"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--enc-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-iters", type=int, default=2,
                    help="fold repetitions; 10 matches the frozen protocol but is "
                         "10x the cost, so the default is 2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "finetune"))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck_dir = Path(a.checkpoints)
    splits = json.loads((PROCESSED / "splits_frozen.json").read_text())["cells"]
    shards = {c: WindowShard(PROCESSED / f"{c}_ds.npz") for c in TASK_MATRIX}
    print(f"device: {DEVICE}")

    rows = []
    for mdl in a.models:
        if mdl not in BUILDERS:
            print(f"  [skip] {mdl}"); continue
        fname, builder = BUILDERS[mdl]
        ck = ck_dir / fname
        if not ck.exists():
            print(f"  [skip] {mdl}: no {fname}"); continue
        arrays = {c: cell_arrays(s, mdl) for c, s in shards.items()}
        for key, cell in splits.items():
            coh, task = cell["dataset"], cell["task"]
            folds = cell["folds"][:a.n_iters]
            lab = pd.read_csv(PROCESSED / f"{coh}_labels.csv")
            lab["subject"] = lab["subject"].astype(str)
            lab = lab.set_index("subject")[task]
            subj = np.asarray([str(s) for s in shards[coh].subjects])
            y = lab.reindex(subj).to_numpy(dtype=float)
            keep = np.isfinite(y)
            G, M, S = arrays[coh]
            pr, auc, f1, nf = finetune_cell(
                builder, ck, G[keep], M[keep], S[keep], y[keep].astype(int),
                subj[keep], folds, len(np.unique(y[keep])), a.epochs,
                a.enc_lr, a.head_lr, a.batch, a.seed)
            rows.append({"model": mdl, "cohort": coh, "task": task,
                         "PR": round(pr, 1), "AUC": round(auc, 1),
                         "F1": round(f1, 1), "folds": nf})
            print(f"  {mdl:<12} {key:<30} PR {pr:5.1f}  AUC {auc:5.1f}  F1 {f1:5.1f}",
                  flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(out / "finetune_cells.csv", index=False)
    print("\n=== fine-tuned, task-averaged ===")
    print(d.groupby("model")[["PR", "AUC", "F1"]].mean().round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
