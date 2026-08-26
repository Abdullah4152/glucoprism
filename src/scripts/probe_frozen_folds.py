"""Stage 2: score pre-embedded v2 runs on the frozen folds, both protocols."""
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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT / "src" / "core"))

from cgmkit.data.datasets import WindowShard               # noqa: E402
from cgmkit.data.labels import TASK_MATRIX                 # noqa: E402
from cgmkit.eval.aggregate import agg_rich, _group_slices  # noqa: E402
from cgmkit.eval.probe import glucofm_probe, _metrics, _align_proba  # noqa: E402
from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402

P = ROOT / "data" / "processed"
EMB = OUTDIR / "v2emb"
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
ap.add_argument("--blocks", nargs="*", default=["full", "zTzS", "zT"])
ap.add_argument("--runs", nargs="*", default=None,
                help="restrict to these runs; the CSV is appended to, so the "
                     "job can be worked through in chunks")
ap.add_argument("--limit", type=int, default=None,
                help="stop after this many (run, block) passes")
ap.add_argument("--out", default=str(OUTDIR / "v2_final_scores.csv"))
a = ap.parse_args()

runs = sorted({p.name.split("__")[0] for p in EMB.glob("*__cgmacros__full.npy")})
runs = [r for r in runs if not r.startswith("v2r-")]
if a.runs:
    runs = [r for r in runs if r in set(a.runs)]

# Resumable: 36 scoring passes take hours, and a kill that loses everything is
# expensive. Append after EVERY (run, block) and skip what is already on disk.
out = Path(a.out)
done: set[tuple[str, str]] = set()
if out.exists():
    prev = pd.read_csv(out)
    done = set(map(tuple, prev[["run", "block"]].drop_duplicates().to_numpy()))
    print(f"resuming: {len(done)} (run, block) pairs already scored")

print(f"{'run':<18}{'block':<7}{'window PR/AUC':>16}{'subject PR/AUC':>17}", flush=True)
print("-" * 60, flush=True)
n_done = 0
for run in runs:
    for block in a.blocks:
        if (run, block) in done:
            continue
        if a.limit and n_done >= a.limit:
            print(f"\nlimit {a.limit} reached -- rerun to continue", flush=True)
            raise SystemExit(0)
        n_done += 1
        try:
            emb = {c: np.load(EMB / f"{run}__{c}__{block}.npy") for c in shards}
        except FileNotFoundError:
            continue
        recs = []
        w, b = score(emb, "window"), score(emb, "subject")
        for lvl, rows in (("window", w), ("subject", b)):
            for coh, task, pr, auc, f1 in rows:
                recs.append(dict(run=run, block=block, level=lvl, cohort=coh,
                                 task=task, pr=pr, auc=auc, f1=f1))
        df = pd.DataFrame(recs)
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        wm = np.mean([[x[2], x[3]] for x in w], axis=0)
        bm = np.mean([[x[2], x[3]] for x in b], axis=0)
        print(f"{run:<18}{block:<7}{wm[0]:>7.1f} /{wm[1]:>6.1f}"
              f"{bm[0]:>10.1f} /{bm[1]:>6.1f}", flush=True)
print(f"\nwrote {out}", flush=True)
