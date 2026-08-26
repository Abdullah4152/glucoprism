"""Is zT good at transfer because it is TRAIT, or because it is NARROW?

zT is 64-d and the full readout is 128-d. A narrower representation overfits the
source cohort less, so it can transfer better for reasons that have nothing to do
with what it encodes. Same control discipline as within-cohort (decision D5).

Controls, all 64-d like zT:
  rand64   random projection of the full 128-d embedding, fitted on nothing
  slice64  the first 64 dims of the full embedding
  pca64    PCA fitted on the SOURCE cohort only (never the target)
and 48-d versions to match zS.
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


import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = ROOT
sys.path.insert(0, str(ROOT / "src" / "core"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from cgmkit.data.datasets import WindowShard          # noqa: E402
from cgmkit.eval.probe import _metrics, _align_proba  # noqa: E402
from sklearn.decomposition import PCA                     # noqa: E402
from sklearn.linear_model import LogisticRegression       # noqa: E402
from sklearn.preprocessing import StandardScaler          # noqa: E402
from fd3_cross_dataset import labels, COHORTS, TASKS      # noqa: E402

P = ROOT / "data" / "processed"
EMB = ROOT / "experiments" / "artifacts" / "v2emb"

shards = {c: WindowShard(P / f"{c}_ds.npz") for c in COHORTS}
subj = {c: np.asarray([str(x) for x in s.subjects]) for c, s in shards.items()}


def transfer(getX, src, tgt, task):
    ys = labels(src, task).reindex(subj[src]).to_numpy(dtype=float)
    yt = labels(tgt, task).reindex(subj[tgt]).to_numpy(dtype=float)
    ks, kt = np.isfinite(ys), np.isfinite(yt)
    ys, yt = ys[ks].astype(int), yt[kt].astype(int)
    if task == "diabetes":
        if src == "cgmacros":
            ys = (ys > 0).astype(int)
        if tgt == "cgmacros":
            yt = (yt > 0).astype(int)
    if len(np.unique(ys)) < 2 or len(np.unique(yt)) < 2:
        return None
    Xs, Xt = getX(src, ks), getX(tgt, kt)
    sc = StandardScaler().fit(Xs)
    clf = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000,
                             random_state=0).fit(sc.transform(Xs), ys)
    cls = np.unique(ys)
    p = _align_proba(clf, sc.transform(Xt), cls)
    auc, pr, _ = _metrics(yt, p, cls)
    return 100 * auc


recs = []
for seed in (0, 1, 2):
    full = {c: np.load(EMB / f"v2r-s{seed}__{c}__full.npy") for c in COHORTS}
    blocks = {b: {c: np.load(EMB / f"v2r-s{seed}__{c}__{b}.npy") for c in COHORTS}
              for b in ("zT", "zS", "zA")}
    D = full[COHORTS[0]].shape[1]

    variants: dict[str, object] = {
        "full(128)": lambda c, k, f=full: f[c][k],
        "zT(64)": lambda c, k, b=blocks: b["zT"][c][k],
        "zS(48)": lambda c, k, b=blocks: b["zS"][c][k],
        "zA(16)": lambda c, k, b=blocks: b["zA"][c][k],
    }
    for width in (64, 48):
        rng = np.random.default_rng(1000 + seed * 10 + width)
        R = rng.normal(size=(D, width)) / np.sqrt(D)
        variants[f"rand{width}"] = (
            lambda c, k, f=full, R=R: f[c][k] @ R)
        variants[f"slice{width}"] = (
            lambda c, k, f=full, w=width: f[c][k][:, :w])

    for name, fn in variants.items():
        for src, tgt in itertools.permutations(COHORTS, 2):
            for task in TASKS:
                a = transfer(fn, src, tgt, task)
                if a is not None:
                    recs.append(dict(seed=seed, variant=name, src=src,
                                     tgt=tgt, task=task, auc=a))

    # PCA must be fitted on the SOURCE only -- fitting on both would leak the
    # target distribution into the representation.
    for width in (64, 48):
        for src, tgt in itertools.permutations(COHORTS, 2):
            pca = PCA(n_components=width, random_state=0).fit(full[src])
            fn = lambda c, k, p=pca, f=full: p.transform(f[c][k])
            for task in TASKS:
                a = transfer(fn, src, tgt, task)
                if a is not None:
                    recs.append(dict(seed=seed, variant=f"pca{width}", src=src,
                                     tgt=tgt, task=task, auc=a))

df = pd.DataFrame(recs)
df.to_csv(ROOT / "experiments" / "artifacts" / "fd3_block_controls.csv", index=False)

base = df[df.variant == "full(128)"].groupby(["seed", "src", "tgt", "task"]).auc.mean()
order = ["zT(64)", "rand64", "slice64", "pca64",
         "zS(48)", "rand48", "slice48", "pca48", "zA(16)"]
print(f"{'variant':<12}{'mean AUC':>10}{'vs full':>9}{'t':>7}  better in")
print("-" * 52)
print(f"{'full(128)':<12}{base.mean():>10.2f}{'--':>9}{'':>7}")
for v in order:
    m = df[df.variant == v].groupby(["seed", "src", "tgt", "task"]).auc.mean()
    d = (m - base).dropna()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"{v:<12}{m.mean():>10.2f}{d.mean():>+9.2f}{t:>7.2f}  {int((d>0).sum())}/{len(d)}")

print("\n--- the question: does zT beat its OWN width-matched controls? ---")
zt = df[df.variant == "zT(64)"].groupby(["seed", "src", "tgt", "task"]).auc.mean()
for ctrl in ("rand64", "slice64", "pca64"):
    m = df[df.variant == ctrl].groupby(["seed", "src", "tgt", "task"]).auc.mean()
    d = (zt - m).dropna()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    v = "zT WINS" if t > 2.03 else "zT loses" if t < -2.03 else "tie"
    print(f"  zT vs {ctrl:<9}{d.mean():>+7.2f} AUC  t={t:>6.2f}   {v}")
zs = df[df.variant == "zS(48)"].groupby(["seed", "src", "tgt", "task"]).auc.mean()
for ctrl in ("rand48", "slice48", "pca48"):
    m = df[df.variant == ctrl].groupby(["seed", "src", "tgt", "task"]).auc.mean()
    d = (zs - m).dropna()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    v = "zS WINS" if t > 2.03 else "zS loses" if t < -2.03 else "tie"
    print(f"  zS vs {ctrl:<9}{d.mean():>+7.2f} AUC  t={t:>6.2f}   {v}")
