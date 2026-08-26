"""Score the frozen baseline embeddings on the 14 cells, both protocols.

Same probe, same frozen folds as every other model in this project, so the
comparison is paired: GlucoFM App. D.1 logistic regression, 5-fold
subject-grouped x 10 iterations.
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

from cgmkit.data.datasets import WindowShard               # noqa: E402
from cgmkit.data.labels import TASK_MATRIX                 # noqa: E402
from cgmkit.eval.aggregate import agg_rich, _group_slices  # noqa: E402
from cgmkit.eval.probe import glucofm_probe, _metrics, _align_proba  # noqa: E402
from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402

P = ROOT / "data" / "processed"
EMB = ROOT / "experiments" / "artifacts" / "baseline_emb"
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
                clf = LogisticRegression(solver="lbfgs", max_iter=1000,
                                         random_state=0).fit(sc.transform(A[tr]), yv[tr])
                p = _align_proba(clf, sc.transform(A[te]), cls)
                a_, p_, f_ = _metrics(yv[te], p, cls)
                ca.append(a_); cp.append(p_); cf.append(f_)
        rows.append((coh, task, 100 * np.nanmean(cp), 100 * np.nanmean(ca),
                     100 * np.nanmean(cf)))
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default=str(ROOT / "experiments" / "artifacts" / "baseline_scores.csv"))
a = ap.parse_args()

out = Path(a.out)
done = set()
if out.exists():
    done = set(pd.read_csv(out).run.unique())

models = a.models or sorted({p.name.split("__")[0] for p in EMB.glob("*__cgmacros.npy")})
n = 0
print(f"{'model':<18}{'window PR/AUC/F1':>24}{'subject PR/AUC/F1':>26}", flush=True)
print("-" * 70, flush=True)
for name in models:
    if name in done:
        continue
    try:
        emb = {c: np.load(EMB / f"{name}__{c}.npy") for c in shards}
    except FileNotFoundError:
        print(f"{name:<18}  incomplete -- skipped", flush=True)
        continue
    if a.limit and n >= a.limit:
        print("\nlimit reached -- rerun to continue", flush=True)
        break
    n += 1
    w, b = score(emb, "window"), score(emb, "subject")
    recs = []
    for lvl, rows in (("window", w), ("subject", b)):
        for coh, task, pr, auc, f1 in rows:
            recs.append(dict(run=name, level=lvl, cohort=coh, task=task,
                             pr=pr, auc=auc, f1=f1))
    pd.DataFrame(recs).to_csv(out, mode="a", header=not out.exists(), index=False)
    wm = np.mean([[x[2], x[3], x[4]] for x in w], axis=0)
    bm = np.mean([[x[2], x[3], x[4]] for x in b], axis=0)
    print(f"{name:<18}{wm[0]:>7.1f}/{wm[1]:>6.1f}/{wm[2]:>6.1f}"
          f"{bm[0]:>10.1f}/{bm[1]:>6.1f}/{bm[2]:>6.1f}", flush=True)
print(f"\nwrote {out}", flush=True)
