"""Score pulled checkpoints on the FROZEN folds, both protocols.

The kernel runs its own probe, but for a PAIRED comparison every arm must be
scored on identical folds -- `splits_frozen.json` -- and at both window and
subject level. That is what makes a 0.5-point difference between two arms
readable at all.
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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = ROOT
sys.path.insert(0, str(ROOT / "src" / "core"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval as RE                                          # noqa: E402
from cgmkit.data.datasets import WindowShard               # noqa: E402
from cgmkit.data.labels import TASK_MATRIX                 # noqa: E402
from cgmkit.eval.aggregate import agg_rich, _group_slices  # noqa: E402
from cgmkit.eval.probe import glucofm_probe, _metrics, _align_proba  # noqa: E402
from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402

P = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "kaggle_out"
CELLS = json.loads((P / "splits_frozen.json").read_text())["cells"]
shards = {c: WindowShard(P / f"{c}_ds.npz") for c in TASK_MATRIX}
SUBJ = {c: np.asarray([str(x) for x in s.subjects]) for c, s in shards.items()}
_lab: dict = {}


def lab(coh, task):
    if coh not in _lab:
        d = pd.read_csv(P / f"{coh}_labels.csv")
        d["subject"] = d["subject"].astype(str)
        _lab[coh] = d.set_index("subject")
    return _lab[coh][task]


def score(emb, level):
    rows = []
    for _k, cell in CELLS.items():
        coh, task, folds = cell["dataset"], cell["task"], cell["folds"]
        s = lab(coh, task)
        y_all = s.reindex(SUBJ[coh]).to_numpy(dtype=float)
        keep = np.isfinite(y_all)
        X, g, y = emb[coh][keep], SUBJ[coh][keep], y_all[keep].astype(int)
        if level == "window":
            r = glucofm_probe(X, y, g, splits=folds)
            rows.append((coh, task, 100 * r.pr_auc, 100 * r.roc_auc, 100 * r.macro_f1))
            continue
        uniq, idx = _group_slices(g)
        A = agg_rich(X, idx)
        y_by = {k2: int(v) for k2, v in s.dropna().items()}
        yv = np.array([y_by[u] for u in uniq])
        cls = np.unique(yv)
        cp, ca, cf = [], [], []
        for it in folds:
            for te_s in it:
                te = np.isin(uniq, np.asarray(te_s, dtype=str))
                tr = ~te
                if tr.sum() == 0 or te.sum() == 0 or len(np.unique(yv[tr])) < 2:
                    continue
                sc = StandardScaler().fit(A[tr])
                clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)
                clf.fit(sc.transform(A[tr]), yv[tr])
                p = _align_proba(clf, sc.transform(A[te]), cls)
                a_, p_, f_ = _metrics(yv[te], p, cls)
                ca.append(a_); cp.append(p_); cf.append(f_)
        rows.append((coh, task, 100 * np.nanmean(cp), 100 * np.nanmean(ca),
                     100 * np.nanmean(cf)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "artifacts" / "fd7_scores.csv"))
    a = ap.parse_args()

    runs = a.runs or sorted(d.name for d in OUT.iterdir() if d.is_dir())
    recs = []
    print(f"{'run':<17}{'E1 window PR/AUC/F1':>26}{'E1b subject PR/AUC/F1':>28}")
    print("-" * 72)
    for r in runs:
        # A run is a GlucoFM run or a PRISM run depending on which checkpoint it
        # wrote; PRISM needs its own embedder because the readout differs.
        ckdir = OUT / r / "checkpoints"
        ck, kind = None, None
        for name, k in (("glucofm.pt", "glucofm"), ("prism.pt", "prism")):
            if (ckdir / name).exists():
                ck, kind = ckdir / name, k
                break
        if ck is None:
            print(f"{r:<17}  no checkpoint")
            continue
        emb = {c: RE.EMBEDDERS[kind][1](ck, s) for c, s in shards.items()}
        w = score(emb, "window")
        b = score(emb, "subject")
        for lvl, rows in (("window", w), ("subject", b)):
            for coh, task, pr, auc, f1 in rows:
                recs.append(dict(run=r, level=lvl, cohort=coh, task=task,
                                 pr=pr, auc=auc, f1=f1, kind=kind))
        wm = np.mean([[x[2], x[3], x[4]] for x in w], axis=0)
        bm = np.mean([[x[2], x[3], x[4]] for x in b], axis=0)
        print(f"{r:<17}{wm[0]:>8.1f}{wm[1]:>9.1f}{wm[2]:>9.1f}"
              f"{bm[0]:>10.1f}{bm[1]:>9.1f}{bm[2]:>9.1f}")
    pd.DataFrame(recs).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
