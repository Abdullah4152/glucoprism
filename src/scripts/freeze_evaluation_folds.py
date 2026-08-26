"""Freeze the evaluation folds and carve the held-out pretraining split.

Two artefacts, both written once and then treated as read-only:

  1. `data/processed/splits_frozen.json` -- for each of the 14 task-dataset cells,
     the exact subject -> fold assignment (5-fold subject-grouped x 10 iterations).
     Every model from here on scores the *identical* partition, which is what the
     paired Wilcoxon test in `eval/probe.py` assumes. Without this, GlucoPRISM and
     its baselines would each get their own folds and the pairing would be a lie.

  2. `data/processed/pretrain_holdout.json` -- a subject-disjoint ~10% slice of the
     pretraining corpus, stratified by cohort. Loss weights, beta and block dims
     are swept on this and nothing else; the downstream cohorts never inform model
     selection (proposal Sec. 9).

Also records a SHA-256 per shard so a later corpus change is visible rather than
silent.

    python scripts/freeze_splits.py
    python scripts/freeze_splits.py --force      # overwrite existing artefacts
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
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(ROOT / "src" / "core"))

from cgmkit.eval.probe import make_splits  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
SPLITS_PATH = PROCESSED / "splits_frozen.json"
HOLDOUT_PATH = PROCESSED / "pretrain_holdout.json"

# 4 cohorts x 7 tasks = 14 cells (GlucoFM Table 3).
TASK_MATRIX = {
    "cgmacros":     ["diabetes_3class", "ir", "hyperlipidemia", "obesity"],
    "shanghait2dm": ["ir", "hyperlipidemia", "hypoglycemia"],
    "stanford":     ["diabetes", "beta_cell", "ir"],
    "hall":         ["diabetes", "ir", "hyperlipidemia", "glucotype"],
}
DS_SHARD = {d: f"{d}_ds.npz" for d in TASK_MATRIX}
LABELS = {d: f"{d}_labels.csv" for d in TASK_MATRIX}

PT_SHARDS = ["replacebg_pt", "stanford_pt", "shanghait2dm_pt", "colas_pt", "bigideas_pt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- eval folds

def freeze_eval_splits(n_splits: int, n_iters: int, seed: int) -> dict:
    cells = {}
    for dataset, tasks in TASK_MATRIX.items():
        shard = np.load(PROCESSED / DS_SHARD[dataset], allow_pickle=True)
        subj = np.asarray([str(s) for s in shard["subject"]])
        lab = pd.read_csv(PROCESSED / LABELS[dataset])
        lab["subject"] = lab["subject"].astype(str)
        lab = lab.set_index("subject")

        for task in tasks:
            if task not in lab.columns:
                print(f"  [skip] {dataset}/{task}: task column absent")
                continue
            # Reproduce the probe's own filtering: windows whose subject has no
            # label for THIS task are dropped before folds are drawn, so the
            # subject set is per-cell, not per-cohort.
            y = lab[task].reindex(subj).to_numpy(dtype=float)
            keep = np.isfinite(y)
            if keep.sum() == 0 or len(np.unique(y[keep])) < 2:
                print(f"  [skip] {dataset}/{task}: no usable labels")
                continue

            groups = subj[keep]
            folds = make_splits(groups, n_splits=n_splits, n_iters=n_iters, seed=seed)
            cells[f"{dataset}/{task}"] = {
                "dataset": dataset,
                "task": task,
                "n_subjects": int(len(np.unique(groups))),
                "n_windows": int(keep.sum()),
                "class_counts": {str(int(k)): int(v) for k, v in
                                 zip(*np.unique(y[keep].astype(int), return_counts=True))},
                "folds": folds,
            }
            print(f"  {dataset:14s} {task:16s} "
                  f"{cells[f'{dataset}/{task}']['n_subjects']:>3} subj  "
                  f"{int(keep.sum()):>4} win  "
                  f"{n_iters}x{n_splits} folds")
    return cells


# --------------------------------------------- held-out pretraining split

def carve_pretrain_holdout(frac: float, seed: int) -> dict:
    """Subject-disjoint holdout, stratified by cohort so every device family is
    represented. Subjects with a single window are preferentially left in TRAIN,
    because V2 (repeated-day) pairs can only be drawn from subjects with >= 2."""
    rng = np.random.default_rng(seed)
    per_cohort, holdout = {}, []
    for name in PT_SHARDS:
        d = np.load(PROCESSED / f"{name}.npz", allow_pickle=True)
        subj = np.asarray([str(s) for s in d["subject"]])
        uniq, counts = np.unique(subj, return_counts=True)
        n_hold = max(1, int(round(frac * len(uniq))))

        multi = uniq[counts >= 2]
        pick_from = multi if len(multi) >= n_hold else uniq
        chosen = rng.choice(pick_from, size=min(n_hold, len(pick_from)), replace=False)

        holdout.extend(chosen.tolist())
        per_cohort[name] = {"n_subjects": int(len(uniq)),
                            "n_holdout": int(len(chosen)),
                            "n_windows_holdout": int(np.isin(subj, chosen).sum())}
        print(f"  {name:20s} {len(uniq):>4} subj -> holdout {len(chosen):>3} "
              f"({int(np.isin(subj, chosen).sum())} windows)")
    return {"frac": frac, "seed": seed,
            "subjects": sorted(holdout), "n_subjects": len(holdout),
            "per_cohort": per_cohort}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for p in (SPLITS_PATH, HOLDOUT_PATH):
        if p.exists() and not a.force:
            print(f"{p.name} already exists. These artefacts are meant to be frozen; "
                  f"pass --force only if you intend to invalidate every result "
                  f"computed against them.")
            return 1

    print("== evaluation folds ==")
    cells = freeze_eval_splits(a.n_splits, a.n_iters, a.seed)

    print("\n== held-out pretraining split ==")
    holdout = carve_pretrain_holdout(a.holdout_frac, a.seed)

    print("\n== shard fingerprints ==")
    fingerprints = {}
    for p in sorted(PROCESSED.glob("*.npz")):
        fingerprints[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        print(f"  {p.name:28s} {fingerprints[p.name]['sha256'][:16]}...")

    SPLITS_PATH.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": "GlucoFM App. D.1: 5-fold subject-grouped CV x 10 iterations",
        "n_splits": a.n_splits, "n_iters": a.n_iters, "seed": a.seed,
        "n_cells": len(cells),
        "shard_fingerprints": fingerprints,
        "cells": cells,
    }, indent=2), encoding="utf-8")

    HOLDOUT_PATH.write_text(json.dumps(holdout, indent=2), encoding="utf-8")

    print(f"\nwrote {SPLITS_PATH}  ({len(cells)} cells)")
    print(f"wrote {HOLDOUT_PATH}  ({holdout['n_subjects']} held-out pretraining subjects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
