"""FD-3: cross-dataset transfer, GlucoFM Appendix D.3 exactly.

  * no cross-validation -- this measures direct transfer, not within-cohort
    generalisation
  * freeze the encoder, embed ALL labelled 24 h windows in source and target
  * train the FD-1 classifier on the ENTIRE source cohort
  * evaluate on the ENTIRE target cohort; target labels are never used for
    training, validation, model selection or threshold tuning

Tasks: diabetes risk and insulin resistance -- the two shared by CGMacros,
Stanford and Hall. 6 directions x 2 tasks x 2 metrics = 24 numbers, matching
their Table 4.
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
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT / "src" / "core"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

import evaluate_models as RE                                          # noqa: E402
from cgmkit.data.datasets import WindowShard               # noqa: E402
from cgmkit.eval.probe import _metrics, _align_proba       # noqa: E402
from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402

P = ROOT / "data" / "processed"
OUT = RUNS
COHORTS = ["cgmacros", "stanford", "hall"]
TASKS = ["diabetes", "ir"]

# GlucoFM's published Table 4, for the side-by-side.
PUBLISHED = {
    ("stanford", "hall", "diabetes"): (61.6, 74.7), ("stanford", "hall", "ir"): (61.6, 72.1),
    ("hall", "stanford", "diabetes"): (73.4, 69.7), ("hall", "stanford", "ir"): (67.7, 69.2),
    ("cgmacros", "hall", "diabetes"): (63.3, 78.2), ("cgmacros", "hall", "ir"): (61.7, 72.5),
    ("hall", "cgmacros", "diabetes"): (88.8, 81.3), ("hall", "cgmacros", "ir"): (90.0, 78.8),
    ("cgmacros", "stanford", "diabetes"): (77.1, 73.6), ("cgmacros", "stanford", "ir"): (65.4, 64.8),
    ("stanford", "cgmacros", "diabetes"): (88.3, 79.9), ("stanford", "cgmacros", "ir"): (87.3, 73.3),
}


# CGMacros stores diabetes risk as a 3-class column; Stanford and Hall use a
# binary one. Same construct, different column name and cardinality.
COLUMN = {("cgmacros", "diabetes"): "diabetes_3class"}


def labels(coh: str, task: str) -> pd.Series:
    d = pd.read_csv(P / f"{coh}_labels.csv")
    d["subject"] = d["subject"].astype(str)
    return d.set_index("subject")[COLUMN.get((coh, task), task)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["V4-fm-off-s0"])
    ap.add_argument("--npy", nargs="*", default=[],
                    help="pre-embedded runs as TAG:BLOCK, read from "
                         "artifacts/v2emb/TAG__COHORT__BLOCK.npy. Used for the v2 "
                         "stack, whose package cannot be imported alongside ours.")
    ap.add_argument("--baselines", nargs="*", default=[],
                    help="zero-shot baselines, read from artifacts/baseline_emb/")
    ap.add_argument("--out", default=str(OUTDIR /
                                         "fd3_cross_dataset.csv"))
    a = ap.parse_args()

    shards = {c: WindowShard(P / f"{c}_ds.npz") for c in COHORTS}
    subj = {c: np.asarray([str(x) for x in s.subjects]) for c, s in shards.items()}
    EMB = OUTDIR / "v2emb"

    jobs: list[tuple[str, dict]] = []
    for run in a.runs:
        ckdir = OUT / run / "checkpoints"
        ck = next((ckdir / n for n in ("glucofm.pt", "prism.pt")
                   if (ckdir / n).exists()), None)
        if ck is None:
            print(f"{run}: no checkpoint")
            continue
        kind = "prism" if ck.name == "prism.pt" else "glucofm"
        jobs.append((run, {c: RE.EMBEDDERS[kind][1](ck, s)
                           for c, s in shards.items()}))
    BEMB = OUTDIR / "baseline_emb"
    for name in a.baselines:
        try:
            jobs.append((name, {c: np.load(BEMB / f"{name}__{c}.npy")
                                for c in COHORTS}))
        except FileNotFoundError as e:
            print(f"{name}: {e}")
    for spec in a.npy:
        tag, _, block = spec.partition(":")
        block = block or "full"
        try:
            jobs.append((f"{tag}:{block}",
                         {c: np.load(EMB / f"{tag}__{c}__{block}.npy")
                          for c in COHORTS}))
        except FileNotFoundError as e:
            print(f"{spec}: {e}")

    recs = []
    for run, emb in jobs:

        for src, tgt in itertools.permutations(COHORTS, 2):
            for task in TASKS:
                ys = labels(src, task).reindex(subj[src]).to_numpy(dtype=float)
                yt = labels(tgt, task).reindex(subj[tgt]).to_numpy(dtype=float)
                ks, kt = np.isfinite(ys), np.isfinite(yt)
                Xs, Xt = emb[src][ks], emb[tgt][kt]
                ys, yt = ys[ks].astype(int), yt[kt].astype(int)
                # CGMacros' diabetes label is 3-class; the others are binary.
                # For transfer only, collapse to normal vs {pre-diabetes, T2D}
                # -- a 3-class probe cannot score against a binary target.
                if task == "diabetes":
                    if src == "cgmacros":
                        ys = (ys > 0).astype(int)
                    if tgt == "cgmacros":
                        yt = (yt > 0).astype(int)
                if len(np.unique(ys)) < 2 or len(np.unique(yt)) < 2:
                    continue
                sc = StandardScaler().fit(Xs)
                clf = LogisticRegression(penalty="l2", solver="lbfgs",
                                         max_iter=1000, random_state=0)
                clf.fit(sc.transform(Xs), ys)
                cls = np.unique(ys)
                p = _align_proba(clf, sc.transform(Xt), cls)
                auc, pr, f1 = _metrics(yt, p, cls)
                recs.append(dict(run=run, src=src, tgt=tgt, task=task,
                                 pr=100 * pr, auc=100 * auc,
                                 n_src=len(ys), n_tgt=len(yt)))

    df = pd.DataFrame(recs)
    df.to_csv(a.out, index=False)

    for run in df.run.unique():
        d = df[df.run == run]
        print(f"\n=== {run} â€” cross-dataset transfer (PR-AUC / ROC-AUC) ===")
        print(f"{'direction':<24}{'task':<10}{'ours':>16}{'GlucoFM paper':>18}")
        print("-" * 70)
        for _, r in d.iterrows():
            pub = PUBLISHED.get((r.src, r.tgt, r.task))
            ps = f"{pub[0]:.1f} / {pub[1]:.1f}" if pub else "-"
            print(f"{r.src + ' -> ' + r.tgt:<24}{r.task:<10}"
                  f"{r.pr:>7.1f} /{r.auc:>6.1f}{ps:>18}")
        print(f"{'MEAN':<34}{d.pr.mean():>7.1f} /{d.auc.mean():>6.1f}")
        pubs = [PUBLISHED[k] for k in PUBLISHED]
        print(f"{'GlucoFM paper mean':<34}"
              f"{np.mean([p[0] for p in pubs]):>7.1f} /"
              f"{np.mean([p[1] for p in pubs]):>6.1f}")


if __name__ == "__main__":
    main()

