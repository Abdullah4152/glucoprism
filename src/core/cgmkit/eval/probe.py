"""Frozen-representation linear probes, in both papers' protocols.

GlucoFM (App. D.1) -- the protocol GlucoPRISM's E1 must match exactly:
    * encoder frozen, embeddings extracted from non-overlapping 24 h windows;
    * one scikit-learn LogisticRegression with L2, solver lbfgs, max_iter 1000,
      fixed random_state, identical for every method;
    * 5-fold subject-grouped CV repeated for 10 iterations with different fold
      assignments; all windows of a subject land in the same fold;
    * each held-out *window* is predicted independently and metrics are computed
      over all held-out windows;
    * the same subject splits are reused across methods, so comparisons are paired.
    * report ROC-AUC, PR-AUC, Macro-F1.

CGM-JEPA (M.5.3 step 4) -- the protocol its Tables 1-6 use:
    * LogisticRegressionCV, class-balanced, inner 2-fold over
      C in {1e-3, 1e-2, 1e-1, 1, 10, 100}, scored by AUROC;
    * outer 2-fold stratified CV repeated over 20 seeds -> 40 paired test folds.

Both are implemented so a reproduction can be checked against either paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    task: str
    roc_auc: float
    pr_auc: float
    macro_f1: float
    roc_auc_std: float = 0.0
    pr_auc_std: float = 0.0
    macro_f1_std: float = 0.0
    n_folds: int = 0
    n_subjects: int = 0
    n_windows: int = 0
    per_fold: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {"task": self.task,
                "PR": round(100 * self.pr_auc, 1), "PR_std": round(100 * self.pr_auc_std, 1),
                "AUC": round(100 * self.roc_auc, 1), "AUC_std": round(100 * self.roc_auc_std, 1),
                "F1": round(100 * self.macro_f1, 1), "F1_std": round(100 * self.macro_f1_std, 1),
                "folds": self.n_folds, "subjects": self.n_subjects, "windows": self.n_windows}


def _metrics(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> tuple:
    """ROC-AUC / PR-AUC / Macro-F1, handling binary and multiclass uniformly."""
    n_cls = len(classes)
    y_pred = classes[proba.argmax(axis=1)]
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if n_cls == 2:
        pos = proba[:, 1]
        roc = roc_auc_score(y_true, pos) if len(np.unique(y_true)) > 1 else np.nan
        pr = average_precision_score((y_true == classes[1]).astype(int), pos)
    else:
        onehot = np.stack([(y_true == c).astype(int) for c in classes], axis=1)
        present = onehot.sum(0) > 0
        roc = (roc_auc_score(onehot[:, present], proba[:, present],
                             average="macro", multi_class="ovr")
               if present.sum() > 1 else np.nan)
        pr = average_precision_score(onehot[:, present], proba[:, present], average="macro")
    return roc, pr, macro_f1


def make_splits(groups, n_splits: int = 5, n_iters: int = 10, seed: int = 0) -> list:
    """Subject-grouped fold assignments -> `[iter][fold] = list of test subject ids`.

    Sole source of truth for fold generation: `glucofm_probe` calls this, and
    `scripts/freeze_splits.py` serialises its output. Keeping one implementation
    is what guarantees a frozen split file and a live probe agree.

    GroupKFold is deterministic, so a different fold assignment per iteration
    comes from permuting the subject -> id mapping (the paper's "10 iterations
    with different fold assignments").
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    dummy = np.zeros(len(groups))                  # GroupKFold reads only n_samples
    out = []
    for it in range(n_iters):
        rng = np.random.default_rng(seed + it)
        perm = rng.permutation(len(uniq))
        remap = {s: int(perm[i]) for i, s in enumerate(uniq)}
        g = np.array([remap[s] for s in groups])
        out.append([sorted(map(str, np.unique(groups[te])))
                    for _tr, te in GroupKFold(n_splits=n_splits).split(dummy, dummy, groups=g)])
    return out


def glucofm_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, task: str = "",
                  n_splits: int = 5, n_iters: int = 10, seed: int = 0,
                  standardize: bool = True, splits: list | None = None) -> ProbeResult:
    """GlucoFM App. D.1: 10 iterations x 5-fold subject-grouped CV.

    `splits`: optional frozen fold assignment in `make_splits` format. Passing it
    makes every model score the *identical* subject partition, which is what the
    paired Wilcoxon test downstream assumes. Omit it and folds are generated on
    the fly exactly as before.
    """
    X, y, groups = np.asarray(X, float), np.asarray(y), np.asarray(groups)
    keep = ~np.isnan(np.asarray(y, dtype=float)) if y.dtype.kind == "f" else np.ones(len(y), bool)
    X, y, groups = X[keep], y[keep], groups[keep]
    classes = np.unique(y)
    gstr = groups.astype(str)

    rocs, prs, f1s, per_fold = [], [], [], []
    uniq = np.unique(groups)

    if splits is None:
        splits = make_splits(groups, n_splits=n_splits, n_iters=n_iters, seed=seed)

    for it, folds in enumerate(splits):
        for te_subjects in folds:
            te = np.isin(gstr, np.asarray(te_subjects, dtype=str))
            tr = ~te
            if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            xtr, xte = X[tr], X[te]
            if standardize:
                sc = StandardScaler().fit(xtr)          # fit on training subjects only
                xtr, xte = sc.transform(xtr), sc.transform(xte)

            clf = LogisticRegression(solver="lbfgs", max_iter=1000,
                                     random_state=seed)
            clf.fit(xtr, y[tr])
            proba = _align_proba(clf, xte, classes)
            roc, pr, f1 = _metrics(y[te], proba, classes)
            rocs.append(roc); prs.append(pr); f1s.append(f1)
            per_fold.append({"iter": it, "roc": roc, "pr": pr, "f1": f1,
                             "n_test": int(te.sum())})

    return ProbeResult(task=task,
                       roc_auc=float(np.nanmean(rocs)), roc_auc_std=float(np.nanstd(rocs)),
                       pr_auc=float(np.nanmean(prs)), pr_auc_std=float(np.nanstd(prs)),
                       macro_f1=float(np.nanmean(f1s)), macro_f1_std=float(np.nanstd(f1s)),
                       n_folds=len(rocs), n_subjects=len(uniq), n_windows=len(X),
                       per_fold=per_fold)


def cgmjepa_probe(X: np.ndarray, y: np.ndarray, task: str = "",
                  n_iters: int = 20, n_splits: int = 2, seed: int = 0) -> ProbeResult:
    """CGM-JEPA M.5.3: LogisticRegressionCV, class-balanced, 20 x stratified 2-fold."""
    X, y = np.asarray(X, float), np.asarray(y)
    classes = np.unique(y)
    Cs = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    rocs, prs, f1s, per_fold = [], [], [], []

    for it in range(n_iters):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + it)
        for tr, te in skf.split(X, y):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            xtr, xte = sc.transform(X[tr]), sc.transform(X[te])
            clf = LogisticRegressionCV(Cs=Cs, cv=2, solver="lbfgs",
                                       scoring="roc_auc", max_iter=1000,
                                       class_weight="balanced", random_state=seed + it)
            clf.fit(xtr, y[tr])
            proba = _align_proba(clf, xte, classes)
            roc, pr, f1 = _metrics(y[te], proba, classes)
            rocs.append(roc); prs.append(pr); f1s.append(f1)
            per_fold.append({"iter": it, "roc": roc, "pr": pr, "f1": f1})

    return ProbeResult(task=task,
                       roc_auc=float(np.nanmean(rocs)), roc_auc_std=float(np.nanstd(rocs)),
                       pr_auc=float(np.nanmean(prs)), pr_auc_std=float(np.nanstd(prs)),
                       macro_f1=float(np.nanmean(f1s)), macro_f1_std=float(np.nanstd(f1s)),
                       n_folds=len(rocs), n_subjects=len(X), n_windows=len(X),
                       per_fold=per_fold)


def transfer_probe(X_src, y_src, X_tgt, y_tgt, task: str = "") -> ProbeResult:
    """GlucoFM App. D.3 cross-dataset transfer: fit on the whole source cohort,
    evaluate once on the target. No target labels are used for anything."""
    classes = np.unique(np.concatenate([np.unique(y_src), np.unique(y_tgt)]))
    sc = StandardScaler().fit(X_src)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)
    clf.fit(sc.transform(X_src), y_src)
    proba = _align_proba(clf, sc.transform(X_tgt), classes)
    roc, pr, f1 = _metrics(np.asarray(y_tgt), proba, classes)
    return ProbeResult(task=task, roc_auc=roc, pr_auc=pr, macro_f1=f1,
                       n_folds=1, n_windows=len(X_tgt))


def _align_proba(clf, X, classes: np.ndarray) -> np.ndarray:
    """Expand predict_proba to the full class list (a fold may miss a rare class)."""
    p = clf.predict_proba(X)
    if len(clf.classes_) == len(classes) and np.array_equal(clf.classes_, classes):
        return p
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(clf.classes_):
        out[:, np.where(classes == c)[0][0]] = p[:, j]
    return out


def paired_wilcoxon(a: list[float], b: list[float]):
    """Paired Wilcoxon over matched folds -- the significance test GlucoFM omits
    and the proposal (Sec. 6) adds. Returns (statistic, p, rank-biserial effect size)."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.allclose(a, b):
        return np.nan, 1.0, 0.0
    stat, p = wilcoxon(a, b)
    n = len(a)
    effect = 1.0 - 2.0 * stat / (n * (n + 1) / 2.0)
    return float(stat), float(p), float(effect)


def holm_bonferroni(pvals: list[float], alpha: float = 0.05):
    """Holm-Bonferroni across the 14 task-dataset cells (proposal Sec. 6)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj, adj < alpha
