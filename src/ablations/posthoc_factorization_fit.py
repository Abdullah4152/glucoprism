"""Post-hoc protocol factorization: fit zT/zS/zA on a FROZEN encoder, from REAL pairs.

THE IDEA. The proposal learns the Trait/State/Sensor split during pretraining, from
a SYNTHETIC second-sensor view. Two measurements say that is the wrong place and
the wrong data:

  * The synthetic V1 partner fails the proposal's own falsification test (§4.3):
    `zT` cosine is 0.853 across synthetic pairs but 0.498 across REAL Dexcom/Libre
    pairs, against a 0.207 unrelated baseline. The partner differs from the anchor
    by -0.646 in observed fraction and only -1.20 mg/dL in level, so `L_sensor`
    teaches invariance to missingness rather than to a device.
  * Running the objectives during pretraining costs representation quality:
    `L_MCR` RISES 0.066 -> 0.147 over training, and the resulting model loses
    6.9 PR to plain GlucoFM.

So: freeze the encoder, and fit three lightweight projections on top using
CGMacros' 376 REAL same-day dual-sensor pairs. The encoder keeps everything it
learned; the factorization becomes a readout, not a training constraint.

LEAKAGE. CGMacros supplies the real pairs AND four of the fourteen evaluation
cells, so a single global fit would train on the test set for those four.

  * 10 non-CGMacros cells -> heads fitted once on all 45 CGMacros subjects.
    Different cohort, disjoint subjects, no leakage.
  * 4 CGMacros cells      -> heads refitted inside EVERY training fold, on that
    fold's training subjects only. Stricter than holding out a fixed 5 subjects
    and it reuses the frozen 5-fold x 10-iteration splits unchanged.

    python scripts/run_posthoc_heads.py --encoder glucofm
    python scripts/run_posthoc_heads.py --encoder v2stack --epochs 400
"""

from __future__ import annotations

import os as _os, sys as _sys
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))
RUNS = _P(_os.environ.get("GLUCOPRISM_RUNS", OUTDIR / "runs"))
EXTERNAL = _P(_os.environ.get("GLUCOPRISM_EXTERNAL", ROOT / "external"))
REFERENCE = ROOT / "src" / "core" / "released_model"
for _p in (ROOT / "src" / "core", ROOT / "baselines", ROOT / "src" / "scripts",
           ROOT / "src" / "ablations", REFERENCE,
           _P(__file__).resolve().parent):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(ROOT / "src" / "core"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from cgmkit.data.datasets import WindowShard            # noqa: E402
from cgmkit.data.labels import TASK_MATRIX              # noqa: E402
from cgmkit.data.views import real_pair_index           # noqa: E402
from cgmkit.eval.probe import glucofm_probe             # noqa: E402

PROCESSED = ROOT / "data" / "processed"
EPS = 1e-6


class Heads(nn.Module):
    """Three lightweight projections of a frozen d-dim representation."""

    def __init__(self, d: int, dT: int = 64, dS: int = 48, dA: int = 16,
                 n_devices: int = 4):
        super().__init__()
        self.hT = nn.Sequential(nn.Linear(d, dT), nn.GELU(), nn.Linear(dT, dT))
        self.hS = nn.Sequential(nn.Linear(d, dS), nn.GELU(), nn.Linear(dS, dS))
        self.hA = nn.Sequential(nn.Linear(d, dA), nn.GELU(), nn.Linear(dA, dA))
        self.device_clf = nn.Linear(dA, n_devices)

    def forward(self, z):
        return self.hT(z), self.hS(z), self.hA(z)


def align(u, v):
    un, vn = F.normalize(u, dim=-1), F.normalize(v, dim=-1)
    return (0.5 * (F.smooth_l1_loss(un, vn, reduction="none").mean(-1)
                   + (1.0 - (un * vn).sum(-1)))).mean()


def indep(zT, zS, zA):
    z = torch.cat([zT, zS, zA], -1)
    z = (z - z.mean(0)) / (z.std(0) + EPS)
    c = (z.t() @ z) / max(len(z) - 1, 1)
    d = [zT.shape[-1], zS.shape[-1], zA.shape[-1]]
    mask = torch.ones_like(c); off = 0
    for k in d:
        mask[off:off + k, off:off + k] = 0; off += k
    return (c * mask).pow(2).sum() / mask.sum().clamp_min(1)


def variance_floor(z):
    """Anti-collapse on the unit sphere. The alignment terms are cosine-based, so
    the collapse they drive is DIRECTIONAL and a magnitude floor would not see
    it: a block can hold healthy per-dimension std while every sample points the
    same way."""
    zn = F.normalize(z, dim=-1)
    tgt = 1.0 / np.sqrt(zn.shape[-1])
    return F.relu(tgt - torch.sqrt(zn.var(0) + 1e-8)).mean() / tgt


def fit_heads(Z, pair_a, pair_b, dev_a, dev_b, day, subj, *, d, epochs, lr,
              w_sensor, w_day, w_indep, w_var, seed=0, device="cpu"):
    """Fit the three heads on real V1 pairs (+ V2 pairs drawn within subject)."""
    torch.manual_seed(seed)
    heads = Heads(d).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=1e-2)
    Zt = torch.tensor(Z, dtype=torch.float32, device=device)
    a = torch.tensor(pair_a, dtype=torch.long, device=device)
    b = torch.tensor(pair_b, dtype=torch.long, device=device)
    da = torch.tensor(dev_a, dtype=torch.long, device=device)
    db = torch.tensor(dev_b, dtype=torch.long, device=device)

    # V2: for each anchor, another window of the same subject on a DIFFERENT day.
    rng = np.random.default_rng(seed)
    v2, has2 = [], []
    for i in pair_a:
        cand = np.flatnonzero((subj == subj[i]) & (day != day[i]))
        v2.append(int(rng.choice(cand)) if len(cand) else int(i))
        has2.append(len(cand) > 0)
    v2 = torch.tensor(np.array(v2), dtype=torch.long, device=device)
    has2 = torch.tensor(np.array(has2), dtype=torch.bool, device=device)

    for _ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        zT, zS, zA = heads(Zt[a])
        zT1, zS1, zA1 = heads(Zt[b])
        # L_sensor: V1 holds (trait, state) fixed and varies only the device.
        l_sensor = align(zT, zT1) + align(zS, zS1)
        logits = torch.cat([heads.device_clf(zA), heads.device_clf(zA1)])
        l_dev = F.cross_entropy(logits, torch.cat([da, db]))
        # L_day: V2 holds the trait fixed and varies the day.
        if has2.any():
            zT2, zS2, _ = heads(Zt[v2])
            l_day = align(zT[has2], zT2[has2])
            # keep zS day-discriminative: push apart across days, hinge form
            cs = F.cosine_similarity(F.normalize(zS[has2], dim=-1),
                                     F.normalize(zS2[has2], dim=-1), dim=-1)
            l_day = l_day + 0.5 * F.relu(cs - 0.3).mean()
        else:
            l_day = torch.zeros((), device=device)
        l_ind = indep(zT, zS, zA)
        l_var = variance_floor(zT) + variance_floor(zS) + variance_floor(zA)
        loss = (w_sensor * (l_sensor + l_dev) + w_day * l_day
                + w_indep * l_ind + w_var * l_var)
        loss.backward()
        nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        opt.step()
    heads.eval()
    return heads, {"sensor": float(l_sensor), "device": float(l_dev),
                   "day": float(l_day), "indep": float(l_ind), "var": float(l_var)}


def _released_or_hint(name):
    """This analysis refits heads, so it needs a TRAINING
    checkpoint. The release ships inference-only tensors, so point
    the user at pretraining rather than at a missing file."""
    p = ROOT / "weights" / name
    if p.exists():
        return p
    raise SystemExit(
        f"{name} not found. This ablation refits projection heads "
        f"on a frozen encoder, so it needs a training checkpoint, "
        f"which the release does not ship (weights/ holds "
        f"inference tensors only). Pretrain first:\n"
        f"  python baselines/common/pretrain.py --model glucofm --seed 0\n"
        f"then pass --checkpoint <path>.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="glucofm",
                    choices=["glucofm", "cgm_jepa", "gluformer_tiny"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-sensor", type=float, default=1.0)
    ap.add_argument("--w-day", type=float, default=1.0)
    ap.add_argument("--w-indep", type=float, default=0.1)
    ap.add_argument("--w-var", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-fit-subjects", type=int, default=None,
                    help="FD-8: fit the heads on only N CGMacros subjects "
                         "(15/20/30). The remainder stay available to the probe, "
                         "so head-fitting and evaluation subjects stay disjoint "
                         "for the 4 CGMacros cells.")
    ap.add_argument("--skip-cgmacros", action="store_true",
                    help="score only the 10 non-CGMacros cells. Those need no "
                         "per-fold refit (disjoint cohort, disjoint subjects), so "
                         "this answers 'does the factorization transfer' in "
                         "minutes instead of hours")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "posthoc"))
    a = ap.parse_args()

    import evaluate_models as RE
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.checkpoint) if a.checkpoint else _released_or_hint(RE.EMBEDDERS[a.encoder][0])
    shards = {c: WindowShard(PROCESSED / f"{c}_ds.npz") for c in TASK_MATRIX}
    EMB = {c: RE.EMBEDDERS[a.encoder][1](ck, s) for c, s in shards.items()}
    d = EMB["cgmacros"].shape[1]

    cg = shards["cgmacros"]
    pairs = real_pair_index(cg)
    pa = np.array([p[0] for p in pairs]); pb = np.array([p[1] for p in pairs])
    dev = np.asarray([str(x) for x in cg.data["device"]])
    day = np.asarray([str(t)[:10] for t in cg.data["start_time"]])
    subj = np.asarray([str(s) for s in cg.subjects])
    devmap = {v: i for i, v in enumerate(sorted(set(dev)))}
    dva = np.array([devmap[dev[i]] for i in pa]); dvb = np.array([devmap[dev[i]] for i in pb])
    print(f"  encoder {a.encoder} ({d}-d) | {len(pairs)} REAL CGMacros pairs, "
          f"{len(set(subj))} subjects")

    splits = json.loads((PROCESSED / "splits_frozen.json").read_text())["cells"]
    kw = dict(d=d, epochs=a.epochs, lr=a.lr, w_sensor=a.w_sensor, w_day=a.w_day,
              w_indep=a.w_indep, w_var=a.w_var, seed=a.seed)

    # FD-8: how much paired data does the factorization need? Restrict head
    # fitting to N CGMacros subjects; the rest stay available to the probe.
    # Selection is by seeded permutation so a given N is the same set across
    # encoders, making the encoder the only variable.
    fit_subj = sorted(set(subj))
    if a.n_fit_subjects:
        rng = np.random.default_rng(a.seed)
        fit_subj = sorted(rng.permutation(fit_subj)[:a.n_fit_subjects].tolist())
        keep_pair = np.array([subj[i] in set(fit_subj) for i in pa])
        pa, pb = pa[keep_pair], pb[keep_pair]
        dva, dvb = dva[keep_pair], dvb[keep_pair]
        print(f"  fitting heads on {len(fit_subj)}/{len(set(subj))} CGMacros "
              f"subjects, {len(pa)} real pairs")

    # Global heads: fitted on the selected CGMacros subjects. Valid for the 10
    # cells that do not come from CGMacros.
    gh, stats = fit_heads(EMB["cgmacros"], pa, pb, dva, dvb, day, subj, **kw)
    print(f"  global heads fitted: " + "  ".join(f"{k}={v:.4f}" for k, v in stats.items()))

    def blocks(heads, X):
        with torch.no_grad():
            zT, zS, zA = heads(torch.tensor(X, dtype=torch.float32))
        return {"zT": zT.numpy(), "zS": zS.numpy(), "zA": zA.numpy(),
                "zTzS": np.concatenate([zT.numpy(), zS.numpy()], 1),
                "full": np.concatenate([zT.numpy(), zS.numpy(), zA.numpy()], 1)}

    rows = []
    for key, cell in splits.items():
        coh, task, folds = cell["dataset"], cell["task"], cell["folds"]
        if a.skip_cgmacros and coh == "cgmacros":
            continue
        lab = pd.read_csv(PROCESSED / f"{coh}_labels.csv")
        lab["subject"] = lab["subject"].astype(str)
        y_all = lab.set_index("subject")[task].reindex(
            np.asarray([str(s) for s in shards[coh].subjects])).to_numpy(dtype=float)
        keep = np.isfinite(y_all)
        g = np.asarray([str(s) for s in shards[coh].subjects])[keep]
        y = y_all[keep].astype(int)

        if coh != "cgmacros":
            B = blocks(gh, EMB[coh][keep])
            B["encoder"] = EMB[coh][keep]
            for name, X in B.items():
                r = glucofm_probe(X, y, g, splits=folds)
                rows.append({"cell": key, "block": name,
                             "PR": round(100 * r.pr_auc, 1),
                             "AUC": round(100 * r.roc_auc, 1), "fit": "global"})
        else:
            # Refit per training fold so no CGMacros test subject informs the heads.
            per = {n: {"pr": [], "auc": []} for n in
                   ("zT", "zS", "zA", "zTzS", "full", "encoder")}
            for it in folds:
                for te_s in it:
                    tr_mask = ~np.isin(subj, np.asarray(te_s, dtype=str))
                    sel = tr_mask[pa] & tr_mask[pb]
                    if sel.sum() < 20:
                        continue
                    h, _ = fit_heads(EMB["cgmacros"], pa[sel], pb[sel],
                                     dva[sel], dvb[sel], day, subj, **kw)
                    B = blocks(h, EMB[coh][keep]); B["encoder"] = EMB[coh][keep]
                    te = np.isin(g, np.asarray(te_s, dtype=str))
                    for n, X in B.items():
                        r = glucofm_probe(X, y, g, splits=[[te_s]])
                        per[n]["pr"].append(100 * r.pr_auc)
                        per[n]["auc"].append(100 * r.roc_auc)
            for n, v in per.items():
                if v["pr"]:
                    rows.append({"cell": key, "block": n,
                                 "PR": round(float(np.nanmean(v["pr"])), 1),
                                 "AUC": round(float(np.nanmean(v["auc"])), 1),
                                 "fit": "per-fold"})
        print(f"  {key:<32} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"posthoc_{a.encoder}.csv", index=False)
    print("\n=== post-hoc factorization, task-averaged over 14 cells ===")
    print(df.groupby("block")[["PR", "AUC"]].mean().round(1)
          .sort_values("AUC", ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
