"""Reviewer-requested analyses that need no pretraining.

Five questions, all answerable from the already-scored embeddings:

  1. INLP baseline.  Is "reserve a block and drop it" better than discovering a
     device direction post hoc and projecting it out (Ravfogel et al.)?  The
     device classifier is fitted on CGMacros' REAL Dexcom/Libre windows -- the
     same supervision zA gets -- and the projection is then applied to every
     cohort.  Note this favours INLP slightly: when CGMacros is the transfer
     target, the projection saw target-domain inputs (never target labels).

  2. Soft deletion.  z(alpha) = [zT | zS | alpha * zA].  alpha=1 is the full
     readout, alpha=0 is the released model.  Does an intermediate alpha trade
     cross-cohort robustness against within-cohort subject-level performance?

  3. Calibration.  AUROC is a ranking metric; a screening threshold needs
     calibration.  ECE and Brier under transfer, with and without zA.

  4. How confounded is the corpus?  Predict device from the observation mask
     alone, from glucose level alone, and from both.  If the mask alone
     identifies the device, mask preservation is itself a shortcut channel.

  5. Non-linear block dependence.  L_indep penalises off-diagonal *correlation*,
     which is linear.  HSIC measures dependence a correlation matrix cannot see.

Writes one CSV per question into artifacts/.
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


import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT / "src" / "core"))
from cgmkit.eval.probe import _metrics, _align_proba      # noqa: E402
from sklearn.linear_model import LogisticRegression           # noqa: E402
from sklearn.preprocessing import StandardScaler              # noqa: E402

P = ROOT / "data" / "processed"
EMB = OUTDIR / "v2emb"
A = OUTDIR
COHORTS = ["cgmacros", "stanford", "hall"]
TASKS = ["diabetes", "ir"]
COLUMN = {("cgmacros", "diabetes"): "diabetes_3class"}
SEEDS = [0, 1, 2]
ARM = "C-v2-vib01"

shard = {c: np.load(P / f"{c}_ds.npz", allow_pickle=True) for c in
         COHORTS + ["shanghait2dm"]}
subj = {c: np.asarray([str(x) for x in shard[c]["subject"]]) for c in shard}


def labels(coh: str, task: str) -> pd.Series:
    d = pd.read_csv(P / f"{coh}_labels.csv")
    d["subject"] = d["subject"].astype(str)
    return d.set_index("subject")[COLUMN.get((coh, task), task)]


def blk(run: str, coh: str, b: str) -> np.ndarray:
    return np.load(EMB / f"{run}__{coh}__{b}.npy")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error on the positive-class probability."""
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


def transfer(embed_of, tag: str, seed: int) -> list[dict]:
    """GlucoFM App. D.3 protocol, with calibration metrics added."""
    recs = []
    for src, tgt in itertools.permutations(COHORTS, 2):
        for task in TASKS:
            ys = labels(src, task).reindex(subj[src]).to_numpy(float)
            yt = labels(tgt, task).reindex(subj[tgt]).to_numpy(float)
            ks, kt = np.isfinite(ys), np.isfinite(yt)
            Xs, Xt = embed_of(src)[ks], embed_of(tgt)[kt]
            ys, yt = ys[ks].astype(int), yt[kt].astype(int)
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
            pr_ = _align_proba(clf, sc.transform(Xt), cls)
            auc, pr, f1 = _metrics(yt, pr_, cls)
            pos = pr_[:, 1] if pr_.ndim > 1 and pr_.shape[1] == 2 else pr_.ravel()
            recs.append(dict(tag=tag, seed=seed, src=src, tgt=tgt, task=task,
                             auc=100 * auc, pr=100 * pr,
                             ece=ece(yt, pos), brier=float(((pos - yt) ** 2).mean())))
    return recs


# =================================================== 1 & 3: INLP + calibration
def inlp_projection(X: np.ndarray, d: np.ndarray, k: int = 8) -> np.ndarray:
    """Iterative nullspace projection: repeatedly fit a linear device probe and
    project its weight direction out. Returns the cumulative projection."""
    dim = X.shape[1]
    Pmat = np.eye(dim)
    Xw = X.copy()
    for _ in range(k):
        clf = LogisticRegression(max_iter=1000, random_state=0)
        clf.fit(Xw, d)
        w = clf.coef_[0]
        n = np.linalg.norm(w)
        if n < 1e-9:
            break
        w = w / n
        Pi = np.eye(dim) - np.outer(w, w)
        Pmat = Pi @ Pmat
        Xw = X @ Pmat.T
    return Pmat


print("=" * 70)
print("1+3. INLP erasure baseline vs reserved-and-drop, with calibration")
print("=" * 70)
dev = (shard["cgmacros"]["device"] == "libre").astype(int)
rows = []
for seed in SEEDS:
    run = f"{ARM}-s{seed}"
    full = {c: blk(run, c, "full") for c in COHORTS}
    ztzs = {c: blk(run, c, "zTzS") for c in COHORTS}
    sc = StandardScaler().fit(full["cgmacros"])
    Pm = inlp_projection(sc.transform(full["cgmacros"]), dev, k=8)
    inlp = {c: sc.transform(full[c]) @ Pm.T for c in COHORTS}

    rows += transfer(lambda c: full[c], "full (128d)", seed)
    rows += transfer(lambda c: ztzs[c], "drop zA (112d, ours)", seed)
    rows += transfer(lambda c: inlp[c], "INLP on full (128d)", seed)

inlp_df = pd.DataFrame(rows)
inlp_df.to_csv(A / "rev_inlp_calibration.csv", index=False)
g = inlp_df.groupby("tag")[["auc", "ece", "brier"]].mean()
print(f"\n{'readout':<26}{'transfer AUC':>13}{'ECE':>9}{'Brier':>9}")
for t in ["full (128d)", "INLP on full (128d)", "drop zA (112d, ours)"]:
    if t in g.index:
        print(f"{t:<26}{g.loc[t, 'auc']:>13.2f}{g.loc[t, 'ece']:>9.3f}"
              f"{g.loc[t, 'brier']:>9.3f}")

# ============================================================ 2: soft deletion
print("\n" + "=" * 70)
print("2. Soft deletion: z(alpha) = [zT | zS | alpha * zA]")
print("=" * 70)
# Scaling zA by alpha is a NO-OP under this probe: StandardScaler re-standardises
# every column, so alpha divides out exactly. Measured below to make the point,
# and then the real continuum is partial deletion by DIMENSION COUNT, which
# standardisation cannot undo. Dimensions are ordered by how device-informative
# they are on CGMacros' real Dexcom/Libre labels, so "keep m" keeps the m least
# device-informative coordinates.
rows = []
for seed in SEEDS:
    run = f"{ARM}-s{seed}"
    parts = {c: (blk(run, c, "zTzS"), blk(run, c, "zA")) for c in COHORTS}
    for al in (0.0, 0.5, 1.0):
        emb = {c: np.concatenate([parts[c][0], al * parts[c][1]], 1)
               for c in COHORTS}
        rows += transfer(lambda c, e=emb: e[c], f"scale alpha={al}", seed)

    zA_cg = blk(run, "cgmacros", "zA")
    sc0 = StandardScaler().fit(zA_cg)
    probe = LogisticRegression(max_iter=1000, random_state=0)
    probe.fit(sc0.transform(zA_cg), dev)
    order = np.argsort(np.abs(probe.coef_[0]))          # least informative first
    for m_ in (0, 2, 4, 8, 12, 16):
        keep = order[:m_]
        emb = {c: np.concatenate([parts[c][0], parts[c][1][:, keep]], 1)
               for c in COHORTS}
        rows += transfer(lambda c, e=emb: e[c], f"keep {m_}/16 zA dims", seed)
soft = pd.DataFrame(rows)
soft.to_csv(A / "rev_soft_deletion.csv", index=False)
gs = soft.groupby("tag").auc.agg(["mean", "std"])
print(f"\n{'readout':<24}{'transfer AUC':>14}{'sd':>8}")
for t in [f"scale alpha={a_}" for a_ in (0.0, 0.5, 1.0)] + \
         [f"keep {m_}/16 zA dims" for m_ in (0, 2, 4, 8, 12, 16)]:
    if t in gs.index:
        print(f"{t:<24}{gs.loc[t, 'mean']:>14.2f}{gs.loc[t, 'std']:>8.2f}")

# Emit partial-deletion blocks so the WITHIN-cohort frozen-fold probe can score
# them through the identical pipeline as every other number in the paper. The
# dimension order is fitted on CGMacros' real device labels once per seed and
# reused for every cohort, so "keep 2" means the same 2 coordinates everywhere.
for seed in SEEDS:
    run = f"{ARM}-s{seed}"
    sc0 = StandardScaler().fit(blk(run, "cgmacros", "zA"))
    pb = LogisticRegression(max_iter=1000, random_state=0)
    pb.fit(sc0.transform(blk(run, "cgmacros", "zA")), dev)
    order = np.argsort(np.abs(pb.coef_[0]))
    for c in list(shard):
        z, a_ = blk(run, c, "zTzS"), blk(run, c, "zA")
        for m_ in (2, 4):
            np.save(EMB / f"{run}__{c}__keep{m_}.npy",
                    np.concatenate([z, a_[:, order[:m_]]], 1))
print("\nwrote partial-deletion blocks (keep2, keep4) for the within-cohort probe")

# ================================================ 4: how confounded is the mask
print("\n" + "=" * 70)
print("4. Device predictability: is the observation mask itself a shortcut?")
print("=" * 70)
m = shard["cgmacros"]["mask"].astype(np.float32)
g_ = np.nan_to_num(shard["cgmacros"]["glucose"].astype(np.float32), nan=0.0)
obs = m.sum(1, keepdims=True)
lvl = np.stack([(g_ * m).sum(1) / np.maximum(m.sum(1), 1),
                g_.std(1), np.percentile(g_, 90, axis=1)], 1)
FEATS = {"observation mask (288d)": m,
         "mask summary (count only)": obs,
         "glucose level only (3d)": lvl,
         "mask + level": np.concatenate([m, lvl], 1)}
rows = []
sg = subj["cgmacros"]
uniq = np.unique(sg)
rng = np.random.default_rng(0)
folds = {s: i % 5 for i, s in enumerate(rng.permutation(uniq))}
fid = np.array([folds[s] for s in sg])
for name, X in FEATS.items():
    aucs = []
    for f in range(5):
        tr, te = fid != f, fid == f
        if len(np.unique(dev[tr])) < 2 or len(np.unique(dev[te])) < 2:
            continue
        s2 = StandardScaler().fit(X[tr])
        c2 = LogisticRegression(max_iter=1000, random_state=0)
        c2.fit(s2.transform(X[tr]), dev[tr])
        p2 = c2.predict_proba(s2.transform(X[te]))
        auc, _, _ = _metrics(dev[te], p2, np.unique(dev[tr]))
        aucs.append(100 * auc)
    rows.append(dict(features=name, device_auc=float(np.mean(aucs)),
                     sd=float(np.std(aucs))))
    print(f"  {name:<28}device AUROC {np.mean(aucs):6.2f} +- {np.std(aucs):.2f}")
pd.DataFrame(rows).to_csv(A / "rev_device_predictability.csv", index=False)

# ========================================= 5: non-linear dependence between blocks
print("\n" + "=" * 70)
print("5. Block dependence: linear correlation vs HSIC")
print("=" * 70)


def hsic(X: np.ndarray, Y: np.ndarray, n: int = 400, seed: int = 0) -> float:
    """Normalised HSIC with RBF kernels at the median heuristic."""
    r = np.random.default_rng(seed)
    i = r.choice(len(X), min(n, len(X)), replace=False)
    X, Y = X[i], Y[i]

    def K(Z):
        d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
        s = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        return np.exp(-d2 / (2 * s))

    Kx, Ky = K(X), K(Y)
    m = len(X)
    H = np.eye(m) - np.ones((m, m)) / m
    Kc, Lc = H @ Kx @ H, H @ Ky @ H
    num = (Kc * Lc).sum()
    den = np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    return float(num / den) if den > 0 else 0.0


rows = []
for seed in SEEDS:
    run = f"{ARM}-s{seed}"
    for c in COHORTS:
        zT, zS, zA = (blk(run, c, b) for b in ("zT", "zS", "zA"))
        for n1, x, n2, y in (("zT", zT, "zS", zS), ("zT", zT, "zA", zA),
                             ("zS", zS, "zA", zA)):
            corr = np.abs(np.corrcoef(x.T, y.T)[:x.shape[1], x.shape[1]:]).mean()
            rows.append(dict(seed=seed, cohort=c, pair=f"{n1}-{n2}",
                             mean_abs_corr=corr, hsic=hsic(x, y, seed=seed)))
dep = pd.DataFrame(rows)
dep.to_csv(A / "rev_block_dependence.csv", index=False)
gd = dep.groupby("pair")[["mean_abs_corr", "hsic"]].mean()
print(f"\n{'block pair':<14}{'mean |corr|':>13}{'HSIC':>9}")
for p_ in gd.index:
    print(f"{p_:<14}{gd.loc[p_, 'mean_abs_corr']:>13.4f}{gd.loc[p_, 'hsic']:>9.4f}")

print("\nwrote rev_*.csv to artifacts/")
