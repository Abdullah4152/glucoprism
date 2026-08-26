"""The GlucoFM tables and GlucoPRISM proposal experiments still missing.

All probe-only: they re-read frozen embeddings, so they cost no GPU time.

  E8 / GlucoFM Fig. 3   few-shot adaptation, two regimes
  E5 / GlucoFM multiday multi-day aggregation, dK = AUC(K) - AUC(1)
  E3                    trait stability: within-subject across-day consistency
  E2                    cross-device transfer on CGMacros' REAL Dexcom/Libre pairs
  E4b                   block controls within cohort for the released model
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


import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT / "src" / "core"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from cgmkit.data.datasets import WindowShard               # noqa: E402
from cgmkit.data.labels import TASK_MATRIX                 # noqa: E402
from cgmkit.data.views import real_pair_index              # noqa: E402
from cgmkit.eval.probe import _metrics, _align_proba       # noqa: E402
from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402
import evaluate_models as RE                                          # noqa: E402

P = ROOT / "data" / "processed"
EMB = OUTDIR / "v2emb"
OUT = OUTDIR
CELLS = json.loads((P / "splits_frozen.json").read_text())["cells"]
shards = {c: WindowShard(P / f"{c}_ds.npz") for c in TASK_MATRIX}
SUBJ = {c: np.asarray([str(x) for x in s.subjects]) for c, s in shards.items()}
_lab: dict = {}

# The released model, its ablation partner, and the baseline everything is
# measured against.
MODELS = {
    "GlucoPRISM-C": ("C-v2-vib01", "zTzS", (0, 1, 2)),
    "GlucoPRISM-E": ("E-v2-vib-simbias", "zTzS", (0, 1, 2)),
    "GlucoPRISM-C [full]": ("C-v2-vib01", "full", (0, 1, 2)),
}
FM_RUNS = ["W3u-ov40", "W3u-ov40-s1", "W3u-ov40-s2"]


def lab(coh, task):
    if coh not in _lab:
        d = pd.read_csv(P / f"{coh}_labels.csv")
        d["subject"] = d["subject"].astype(str)
        _lab[coh] = d.set_index("subject")
    return _lab[coh][task]


def load_emb(name):
    """Per-seed embeddings for a model, as a list of {cohort: array}."""
    if name == "GlucoFM":
        out = []
        for r in FM_RUNS:
            ck = RUNS / r / "checkpoints" / "glucofm.pt"
            if ck.exists():
                out.append({c: RE.EMBEDDERS["glucofm"][1](ck, s)
                            for c, s in shards.items()})
        return out
    prefix, block, seeds = MODELS[name]
    out = []
    for s in seeds:
        try:
            out.append({c: np.load(EMB / f"{prefix}-s{s}__{c}__{block}.npy")
                        for c in shards})
        except FileNotFoundError:
            pass
    return out


def fit_eval(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000,
                             random_state=0).fit(sc.transform(Xtr), ytr)
    cls = np.unique(ytr)
    auc, pr, f1 = _metrics(yte, _align_proba(clf, sc.transform(Xte), cls), cls)
    return 100 * auc


# =========================================================== E8 / few-shot
def few_shot(emb, k_support, rng, n_draw=5):
    """GlucoFM Fig. 3 left: restrict to K labelled SUPPORT SUBJECTS per class
    inside each training fold, keeping the test fold intact."""
    scores = []
    for cell in CELLS.values():
        coh, task, folds = cell["dataset"], cell["task"], cell["folds"]
        s = lab(coh, task)
        y = s.reindex(SUBJ[coh]).to_numpy(dtype=float)
        keep = np.isfinite(y)
        X, g, y = emb[coh][keep], SUBJ[coh][keep], y[keep].astype(int)
        per = []
        for it in folds[:3]:                       # 3 iterations is enough here
            for te_s in it:
                te = np.isin(g, np.asarray(te_s, dtype=str))
                tr = ~te
                if tr.sum() == 0 or te.sum() == 0:
                    continue
                tr_subj = np.unique(g[tr])
                lab_of = {u: y[g == u][0] for u in tr_subj}
                for _ in range(n_draw):
                    chosen = []
                    for cls in np.unique(y):
                        pool = [u for u in tr_subj if lab_of[u] == cls]
                        if not pool:
                            continue
                        chosen += list(rng.choice(pool, min(k_support, len(pool)),
                                                  replace=False))
                    m = np.isin(g, np.asarray(chosen, dtype=str))
                    if m.sum() == 0:
                        continue
                    a = fit_eval(X[m], y[m], X[te], y[te])
                    if a is not None:
                        per.append(a)
        if per:
            scores.append(np.mean(per))
    return float(np.mean(scores)) if scores else np.nan


def obs_fraction(emb, frac, rng, n_draw=3):
    """GlucoFM Fig. 3 right: keep every training subject but only a fraction of
    each subject's windows."""
    scores = []
    for cell in CELLS.values():
        coh, task, folds = cell["dataset"], cell["task"], cell["folds"]
        s = lab(coh, task)
        y = s.reindex(SUBJ[coh]).to_numpy(dtype=float)
        keep = np.isfinite(y)
        X, g, y = emb[coh][keep], SUBJ[coh][keep], y[keep].astype(int)
        per = []
        for it in folds[:3]:
            for te_s in it:
                te = np.isin(g, np.asarray(te_s, dtype=str))
                tr = np.flatnonzero(~te)
                if tr.size == 0 or te.sum() == 0:
                    continue
                for _ in range(n_draw):
                    sel = []
                    for u in np.unique(g[tr]):
                        idx = tr[g[tr] == u]
                        n = max(1, int(round(len(idx) * frac)))
                        sel += list(rng.choice(idx, n, replace=False))
                    sel = np.asarray(sel)
                    a = fit_eval(X[sel], y[sel], X[te], y[te])
                    if a is not None:
                        per.append(a)
        if per:
            scores.append(np.mean(per))
    return float(np.mean(scores)) if scores else np.nan


# =========================================================== E5 / multi-day
def multiday(emb, K, rng, n_draw=5):
    """Aggregate K days per subject by mean pooling, then probe at subject level."""
    scores = []
    for cell in CELLS.values():
        coh, task, folds = cell["dataset"], cell["task"], cell["folds"]
        s = lab(coh, task)
        y_all = s.reindex(SUBJ[coh]).to_numpy(dtype=float)
        keep = np.isfinite(y_all)
        X, g = emb[coh][keep], SUBJ[coh][keep]
        y_by = {u: int(v) for u, v in s.dropna().items()}
        uniq = np.unique(g)
        per = []
        for it in folds[:3]:
            for te_s in it:
                te_u = np.asarray(te_s, dtype=str)
                tr_u = np.array([u for u in uniq if u not in set(te_u)])
                if len(tr_u) == 0 or len(te_u) == 0:
                    continue
                for _ in range(n_draw):
                    def agg(us):
                        A, Y = [], []
                        for u in us:
                            idx = np.flatnonzero(g == u)
                            if len(idx) == 0 or u not in y_by:
                                continue
                            pick = rng.choice(idx, min(K, len(idx)), replace=False)
                            A.append(X[pick].mean(0))
                            Y.append(y_by[u])
                        return np.array(A), np.array(Y)
                    Xtr, ytr = agg(tr_u)
                    Xte, yte = agg(te_u)
                    if len(Xtr) == 0 or len(Xte) == 0:
                        continue
                    a = fit_eval(Xtr, ytr, Xte, yte)
                    if a is not None:
                        per.append(a)
        if per:
            scores.append(np.mean(per))
    return float(np.mean(scores)) if scores else np.nan


# =========================================================== E3 / trait stability
def trait_stability(emb):
    """Within-subject across-day cosine consistency, minus the between-subject
    baseline. A representation that is genuinely subject-stable should be much
    more self-similar across a person's own days than across people."""
    out = {}
    for coh in shards:
        X, g = emb[coh], SUBJ[coh]
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        within, between = [], []
        us = np.unique(g)
        for u in us:
            idx = np.flatnonzero(g == u)
            if len(idx) < 2:
                continue
            S = Xn[idx] @ Xn[idx].T
            within.append(S[np.triu_indices(len(idx), 1)].mean())
        for i in range(len(us)):
            for j in range(i + 1, min(i + 6, len(us))):
                a, b = np.flatnonzero(g == us[i]), np.flatnonzero(g == us[j])
                between.append((Xn[a] @ Xn[b].T).mean())
        out[coh] = (float(np.mean(within)), float(np.mean(between)),
                    float(np.mean(within) - np.mean(between)))
    return out


# =========================================================== E2 / cross-device
def cross_device(emb):
    """Train on one device's windows, test on the other's, within CGMacros.
    The proposal's sharpest falsifiable claim: a trait representation should
    survive a device change."""
    sh = shards["cgmacros"]
    dev = np.asarray([str(x) for x in sh.data["device"]])
    g = SUBJ["cgmacros"]
    X = emb["cgmacros"]
    res = {}
    for task in TASK_MATRIX["cgmacros"]:
        s = lab("cgmacros", task)
        y = s.reindex(g).to_numpy(dtype=float)
        ok = np.isfinite(y)
        for src, tgt in (("dexcom", "libre"), ("libre", "dexcom")):
            m1 = ok & (dev == src)
            m2 = ok & (dev == tgt)
            if m1.sum() < 10 or m2.sum() < 10:
                continue
            a = fit_eval(X[m1], y[m1].astype(int), X[m2], y[m2].astype(int))
            if a is not None:
                res[f"{task}: {src}->{tgt}"] = a
    return res


def main() -> None:
    rng = np.random.default_rng(0)
    names = ["GlucoFM", "GlucoPRISM-C", "GlucoPRISM-E", "GlucoPRISM-C [full]"]
    embs = {n: load_emb(n) for n in names}
    embs = {k: v for k, v in embs.items() if v}
    for k, v in embs.items():
        print(f"  {k}: {len(v)} seeds")
    recs = []

    print("\n=== E8 / GlucoFM Fig. 3a: few-shot, K support subjects per class ===")
    print(f"{'model':<22}" + "".join(f"K={k:<6}" for k in (1, 2, 3, 5)))
    print("-" * 60)
    for n, es in embs.items():
        row = []
        for k in (1, 2, 3, 5):
            row.append(np.mean([few_shot(e, k, np.random.default_rng(k)) for e in es[:2]]))
            recs.append(dict(exp="fewshot_subjects", model=n, x=k, auc=row[-1]))
        print(f"{n:<22}" + "".join(f"{v:<8.1f}" for v in row), flush=True)

    print("\n=== E8 / GlucoFM Fig. 3b: fraction of each subject's windows ===")
    print(f"{'model':<22}" + "".join(f"{int(f*100)}%{'':<4}" for f in (0.1, 0.25, 0.5, 1.0)))
    print("-" * 60)
    for n, es in embs.items():
        row = []
        for f in (0.1, 0.25, 0.5, 1.0):
            row.append(np.mean([obs_fraction(e, f, np.random.default_rng(int(f*100)))
                                for e in es[:2]]))
            recs.append(dict(exp="fewshot_obsfrac", model=n, x=f, auc=row[-1]))
        print(f"{n:<22}" + "".join(f"{v:<8.1f}" for v in row), flush=True)

    print("\n=== E5: multi-day aggregation, dK = AUC(K) - AUC(1) ===")
    print(f"{'model':<22}" + "".join(f"K={k:<6}" for k in (1, 2, 3, 5, 7)))
    print("-" * 68)
    for n, es in embs.items():
        row = []
        for k in (1, 2, 3, 5, 7):
            row.append(np.mean([multiday(e, k, np.random.default_rng(k)) for e in es[:2]]))
            recs.append(dict(exp="multiday", model=n, x=k, auc=row[-1]))
        print(f"{n:<22}" + "".join(f"{v:<8.1f}" for v in row)
              + f"   dK={row[-1]-row[0]:+.1f}", flush=True)

    print("\n=== E3: trait stability (within-subject minus between-subject cosine) ===")
    print(f"{'model':<22}" + "".join(f"{c[:9]:<11}" for c in shards))
    print("-" * 70)
    for n, es in embs.items():
        st = trait_stability(es[0])
        print(f"{n:<22}" + "".join(f"{st[c][2]:<11.3f}" for c in shards), flush=True)
        for c, (wi, be, dl) in st.items():
            recs.append(dict(exp="trait_stability", model=n, x=c, auc=dl))

    print("\n=== E2: cross-device transfer within CGMacros (real Dexcom/Libre) ===")
    for n, es in embs.items():
        r = cross_device(es[0])
        if r:
            print(f"  {n}: mean {np.mean(list(r.values())):.1f}   "
                  + "  ".join(f"{k.split(':')[0]} {v:.0f}" for k, v in list(r.items())[:4]),
                  flush=True)
            for k, v in r.items():
                recs.append(dict(exp="cross_device", model=n, x=k, auc=v))

    pd.DataFrame(recs).to_csv(OUT / "remaining_experiments.csv", index=False)
    print(f"\nwrote {OUT / 'remaining_experiments.csv'}")


if __name__ == "__main__":
    main()
