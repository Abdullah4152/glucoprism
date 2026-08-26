"""Subject-disjoint linear probing of the frozen encoders (GlucoFM Table 3 protocol).

Rebuilds the 14 task-dataset cells of GlucoFM Table 3 for whichever checkpoints
are present, using the identical logistic-regression probe and the identical
subject splits for every method -- so the comparison is paired, as the paper
requires.

    python scripts/run_eval.py --checkpoints artifacts/checkpoints
    python scripts/run_eval.py --models glucofm cgm_jepa --datasets stanford hall
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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "core"))

from cgmkit.data.datasets import WindowShard  # noqa: E402
from cgmkit.data.gluformer_tokens import to_tokens  # noqa: E402
from cgmkit.data.windows import densify  # noqa: E402
from cgmkit.data.labels import TASK_MATRIX  # noqa: E402
from cgmkit.eval.probe import glucofm_probe, holm_bonferroni, paired_wilcoxon  # noqa: E402
from cgmkit.models.cgm_jepa import Encoder  # noqa: E402
from cgmkit.models.glucofm import GlucoFM, GlucoFMConfig  # noqa: E402
from cgmkit.models.gluformer import GluFormer, GluFormerConfig  # noqa: E402

from cgmkit.train.pretrain import get_device  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
DEVICE = get_device()


# ---------------------------------------------------------------- embedders

@torch.no_grad()
def embed_glucofm(ckpt: Path, shard: WindowShard, batch: int = 256) -> np.ndarray:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = GlucoFM(GlucoFMConfig(**blob["cfg"])).to(DEVICE).eval()
    model.load_state_dict(blob["model"])
    out = []
    for i in range(0, len(shard), batch):
        sl = slice(i, i + batch)
        g = torch.from_numpy(np.nan_to_num(shard.data["glucose"][sl])).float().to(DEVICE)
        m = torch.from_numpy(shard.data["mask"][sl]).float().to(DEVICE)
        s = torch.from_numpy(shard.data["start_idx"][sl]).long().to(DEVICE)
        out.append(model.embed(g * m, m, s).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_cgm_jepa(ckpt: Path, shard: WindowShard, batch: int = 256) -> np.ndarray:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    enc = Encoder(dim_in=cfg["patch_size"], kernel_size=cfg["encoder_kernel_size"],
                  embed_dim=cfg["encoder_embed_dim"], embed_bias=cfg["encoder_embed_bias"],
                  nhead=cfg["encoder_nhead"], num_layers=cfg["encoder_num_layers"],
                  jepa=True, time_inp_dim=cfg["time_inp_dim"],
                  drop_rate=cfg["encoder_dropout"]).to(DEVICE).eval()
    enc.load_state_dict(blob["encoder"])
    P = 288 // cfg["patch_size"]
    out = []
    for i in range(0, len(shard), batch):
        sl = slice(i, i + batch)
        dense = np.stack([densify(shard.data["glucose"][j], shard.data["mask"][j])
                          for j in range(*sl.indices(len(shard)))])
        x = torch.from_numpy(dense.reshape(len(dense), P, cfg["patch_size"])).float().to(DEVICE)
        xm = torch.zeros(len(dense), P, cfg["patch_size"], cfg["time_inp_dim"], device=DEVICE)
        tokens, _ = enc(x, xm, mask=None)
        out.append(tokens.mean(dim=1).cpu().numpy())     # paper M.5.3: mean-pool patch tokens
    return np.concatenate(out)


@torch.no_grad()
def embed_gluformer(ckpt: Path, shard: WindowShard, batch: int = 64) -> np.ndarray:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = GluFormer(GluFormerConfig(**blob["cfg"])).to(DEVICE).eval()
    model.load_state_dict(blob["model"])
    out = []
    for i in range(0, len(shard), batch):
        sl = slice(i, i + batch)
        dense = np.stack([densify(shard.data["glucose"][j], shard.data["mask"][j])
                          for j in range(*sl.indices(len(shard)))])
        tok = torch.from_numpy(np.stack([to_tokens(d) for d in dense])).long().to(DEVICE)
        out.append(model.embed(tok).cpu().numpy())       # max pooling, App. B.3
    return np.concatenate(out)


def _make_prism_embedder(block: str):
    """One embedder per representation block, so E4's block-routing table is just
    a set of rows in the same 14-cell probe rather than a separate pipeline.

    Always reads the PRE-projection block (`GlucoPRISM.embed`); the projection
    heads exist for the objectives only.
    """
    @torch.no_grad()
    def _embed(ckpt: Path, shard: WindowShard, batch: int = 256) -> np.ndarray:
        from cgmkit.models.prism import GlucoPRISM, PrismConfig
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg_d = dict(blob["cfg"])
        cfg = PrismConfig(fm=GlucoFMConfig(**cfg_d.pop("fm")), **cfg_d)
        model = GlucoPRISM(cfg).to(DEVICE).eval()
        model.load_state_dict(blob["model"])
        out = []
        for i in range(0, len(shard), batch):
            sl = slice(i, i + batch)
            g = torch.from_numpy(np.nan_to_num(shard.data["glucose"][sl])).float().to(DEVICE)
            m = torch.from_numpy(shard.data["mask"][sl]).float().to(DEVICE)
            s = torch.from_numpy(shard.data["start_idx"][sl]).long().to(DEVICE)
            out.append(model.embed(g, m, s, block=block).cpu().numpy())
        return np.concatenate(out)
    return _embed


@torch.no_grad()
def embed_cqp(ckpt: Path, shard: WindowShard, batch: int = 256) -> np.ndarray:
    """Clinical Query Pooling readout: the concatenated query codes (128-d)."""
    from cgmkit.models.cqp import CQPConfig, GlucoCQP
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = dict(blob["cfg"]); c["fm"] = GlucoFMConfig(**c["fm"])
    model = GlucoCQP(CQPConfig(**c)).to(DEVICE).eval()
    model.load_state_dict(blob["model"])
    out = []
    for i in range(0, len(shard), batch):
        sl = slice(i, i + batch)
        g = torch.from_numpy(np.nan_to_num(shard.data["glucose"][sl])).float().to(DEVICE)
        m = torch.from_numpy(shard.data["mask"][sl]).float().to(DEVICE)
        s = torch.from_numpy(shard.data["start_idx"][sl]).long().to(DEVICE)
        out.append(model.embed(g, m, s).cpu().numpy())
    return np.concatenate(out)


EMBEDDERS = {
    "glucofm": ("glucofm.pt", embed_glucofm),
    "cqp": ("cqp.pt", embed_cqp),
    "cgm_jepa": ("cgm_jepa.pt", embed_cgm_jepa),
    "x_cgm_jepa": ("x_cgm_jepa.pt", embed_cgm_jepa),
    "gluformer_tiny": ("gluformer_tiny.pt", embed_gluformer),
    "gluformer_base": ("gluformer_base.pt", embed_gluformer),
    # GlucoPRISM: `prism` is the headline row, the rest are E4 block routing.
    "prism": ("prism.pt", _make_prism_embedder("full")),
    "prism_zT": ("prism.pt", _make_prism_embedder("zT")),
    "prism_zS": ("prism.pt", _make_prism_embedder("zS")),
    "prism_zA": ("prism.pt", _make_prism_embedder("zA")),
    "prism_zTzS": ("prism.pt", _make_prism_embedder("zTzS")),
}


# ------------------------------------------------------------- raw baselines

def embed_raw(shard: WindowShard) -> np.ndarray:
    """Interpolated raw window -- a floor, and the input PCA is fitted on."""
    return np.stack([densify(shard.data["glucose"][i], shard.data["mask"][i])
                     for i in range(len(shard))])


def embed_mask_only(shard: WindowShard) -> np.ndarray:
    """Proposal E7: features derived from the observation mask ALONE.

    If a probe on these clears 60 AUC on any task, mask preservation is partly a
    shortcut rather than a physiological signal, and the headline numbers need
    controlling for it. Reported either way.
    """
    m = shard.data["mask"].astype(np.float32)
    P, K = 24, 12
    patch_density = m.reshape(len(m), P, K).mean(-1)
    runs = []
    for row in m:
        gaps, cur = [], 0
        for v in row:
            if v == 0:
                cur += 1
            elif cur:
                gaps.append(cur); cur = 0
        if cur:
            gaps.append(cur)
        runs.append([len(gaps), max(gaps) if gaps else 0, float(np.mean(gaps)) if gaps else 0.0])
    return np.concatenate([patch_density, m.mean(1, keepdims=True),
                           np.asarray(runs, np.float32),
                           shard.data["start_idx"].reshape(-1, 1).astype(np.float32)], axis=1)


# --------------------------------------------------------------------- main

def evaluate(shard: WindowShard, labels: pd.DataFrame, tasks: list[str],
             feats: dict[str, np.ndarray], dataset: str, n_iters: int, seed: int,
             splits_map: dict | None = None) -> list[dict]:
    subj = np.asarray([str(s) for s in shard.subjects])
    lab = labels.set_index("subject")
    rows = []
    for task in tasks:
        if task not in lab.columns:
            continue
        y_full = lab[task].reindex(subj).to_numpy(dtype=float)
        keep = np.isfinite(y_full)
        if keep.sum() == 0 or len(np.unique(y_full[keep])) < 2:
            print(f"    [skip] {dataset}/{task}: no usable labels")
            continue
        y, groups = y_full[keep].astype(int), subj[keep]
        # Frozen folds make every model score the identical partition, which is
        # what the paired Wilcoxon test assumes. Absent, folds are drawn live.
        cell_splits = (splits_map or {}).get(f"{dataset}/{task}")
        for name, X in feats.items():
            r = glucofm_probe(X[keep], y, groups, task=f"{dataset}/{task}",
                              n_iters=n_iters, seed=seed, splits=cell_splits)
            # as_row() carries its own qualified "task"; the plain names come last
            # so downstream joins against the published tables key correctly.
            row = {**r.as_row(), "dataset": dataset, "task": task, "model": name}
            row["_fold_pr"] = [f["pr"] for f in r.per_fold]
            row["_fold_auc"] = [f["roc"] for f in r.per_fold]
            rows.append(row)
            print(f"    {dataset:14s} {task:16s} {name:16s} "
                  f"PR {row['PR']:5.1f}  AUC {row['AUC']:5.1f}  F1 {row['F1']:5.1f}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default=str(ROOT / "artifacts" / "checkpoints"))
    ap.add_argument("--models", nargs="*", default=list(EMBEDDERS))
    ap.add_argument("--datasets", nargs="*", default=list(TASK_MATRIX))
    ap.add_argument("--n-iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", action="store_true",
                    help="also probe raw-window and mask-only features (proposal E7)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "eval"))
    ap.add_argument("--splits", default=str(PROCESSED / "splits_frozen.json"),
                    help="frozen fold assignment from scripts/freeze_splits.py; "
                         "pass '' to draw folds live")
    a = ap.parse_args()

    ck_dir = Path(a.checkpoints)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    # Every model must score the identical subject partition or the paired
    # Wilcoxon test downstream is meaningless.
    splits_map = None
    if a.splits:
        sp = Path(a.splits)
        if sp.exists():
            blob = json.loads(sp.read_text(encoding="utf-8"))
            splits_map = {k: v["folds"] for k, v in blob["cells"].items()}
            print(f"[splits] frozen: {len(splits_map)} cells from {sp.name}")
        else:
            print(f"[splits] {sp.name} not found -- drawing folds live")

    for dataset in a.datasets:
        shard_p = PROCESSED / f"{dataset}_ds.npz"
        label_p = PROCESSED / f"{dataset}_labels.csv"
        if not shard_p.exists() or not label_p.exists():
            print(f"\n=== {dataset}: skipped (missing {shard_p.name} or {label_p.name})")
            continue
        print(f"\n=== {dataset}")
        shard = WindowShard(shard_p)
        labels = pd.read_csv(label_p)
        print(f"    {len(shard)} windows, {len(set(shard.subjects))} subjects")

        feats: dict[str, np.ndarray] = {}
        for m in a.models:
            fname, fn = EMBEDDERS[m]
            p = ck_dir / fname
            if not p.exists():
                continue
            feats[m] = fn(p, shard)
        if a.baselines:
            feats["raw"] = embed_raw(shard)
            feats["mask_only"] = embed_mask_only(shard)
        if not feats:
            print("    [skip] no checkpoints found")
            continue

        all_rows += evaluate(shard, labels, TASK_MATRIX[dataset], feats,
                             dataset, a.n_iters, a.seed, splits_map=splits_map)

    if not all_rows:
        print("\nNothing evaluated.")
        return 1

    df = pd.DataFrame(all_rows)
    table = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    table.to_csv(out_dir / "linear_probe.csv", index=False)

    piv = table.pivot_table(index=["dataset", "task"], columns="model",
                            values=["PR", "AUC", "F1"])
    print("\n=== task-averaged (GlucoFM Table 3 style) ===")
    print(table.groupby("model")[["PR", "AUC", "F1"]].mean().round(1).to_string())

    # Paired Wilcoxon vs GlucoFM over matched folds, Holm-Bonferroni across cells.
    if "glucofm" in set(table["model"]):
        sig = []
        for (d, t), grp in df.groupby(["dataset", "task"]):
            base = grp[grp["model"] == "glucofm"]
            if base.empty:
                continue
            for _, r in grp[grp["model"] != "glucofm"].iterrows():
                _, p, eff = paired_wilcoxon(base.iloc[0]["_fold_pr"], r["_fold_pr"])
                sig.append({"dataset": d, "task": t, "vs": r["model"], "p": p, "effect": eff})
        if sig:
            s = pd.DataFrame(sig)
            s["p_holm"], s["significant"] = holm_bonferroni(s["p"].tolist())
            s.to_csv(out_dir / "significance.csv", index=False)
            print(f"\nsignificance -> {out_dir / 'significance.csv'} "
                  f"({int(s['significant'].sum())}/{len(s)} cells significant at 0.05)")

    (out_dir / "linear_probe_pivot.txt").write_text(piv.round(1).to_string(), encoding="utf-8")
    (out_dir / "linear_probe.json").write_text(
        json.dumps(all_rows, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_dir / 'linear_probe.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
