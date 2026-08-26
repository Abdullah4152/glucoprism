"""
Shared evaluation machinery for the GlucoPRISM experiments.

Provides: checkpoint loading (GlucoPRISM or plain GlucoFM), per-block frozen
feature extraction, the subject-grouped logistic probe, and the paired
significance testing the proposal asks for (Wilcoxon + Holm-Bonferroni).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from glucofm.config import (AugConfig, Config as GlucoFMConfig, FilterConfig,
                            GridConfig, ModelConfig, PretrainConfig, ProbeConfig)
from glucofm.model import GlucoFMEncoder

from .data import EvalWindows
from .model import GlucoPRISM, PrismConfig

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "data" / "processed" / "windows"
PROC = ROOT / "data" / "processed"

# cohort -> (window stem, labels file, id column, {task: column})
TASKS = {
    "cgmacros_dexcom": ("cgmacros_dexcom_downstream", "labels_cgmacros.csv", "subject_id",
                        {"diabetes": "diabetes_a1c", "ir": "ir",
                         "hyperlipidemia": "hyperlipidemia", "obesity": "obesity"}),
    "cgmacros_libre": ("cgmacros_libre_downstream", "labels_cgmacros.csv", "subject_id",
                       {"diabetes": "diabetes_a1c", "ir": "ir",
                        "hyperlipidemia": "hyperlipidemia", "obesity": "obesity"}),
    "shanghai": ("shanghai_downstream", "labels_shanghai.csv", "session_id",
                 {"hypoglycemia": "hypoglycemia", "ir": "ir",
                  "hyperlipidemia": "hyperlipidemia"}),
    "stanford": ("stanford_downstream", "labels_stanford.csv", "subject_id",
                 {"diabetes": "diabetes", "beta": "beta", "ir": "ir"}),
    "hall": ("hall_downstream", "labels_hall.csv", "subject_id",
             {"diabetes": "diabetes", "glucotype": "glucotype", "ir": "ir",
              "hyperlipidemia": "hyperlipidemia"}),
}

# proposal section 3.1: trait tasks satisfy y _||_ (s,a) | t; state tasks depend on Var_s
TRAIT_TASKS = {"ir", "beta", "diabetes", "hyperlipidemia", "obesity"}
STATE_TASKS = {"glucotype", "hypoglycemia"}


# --------------------------------------------------------------------------- #
def _cfg_from_dict(c: dict) -> GlucoFMConfig:
    return GlucoFMConfig(
        grid=GridConfig(**c["grid"]), filt=FilterConfig(**c["filt"]),
        model=ModelConfig(**c["model"]), pretrain=PretrainConfig(**c["pretrain"]),
        aug=AugConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                         for k, v in c["aug"].items()}),
        probe=ProbeConfig(**c["probe"]))


def load_model(ckpt_path: Path, device: str = "cpu"):
    """Load GlucoPRISM (with blocks), a plain GlucoFM encoder, or CGM-JEPA.

    Returns (model, kind, fm_cfg) where kind in {'prism', 'glucofm', 'cgmjepa'}.
    A DIRECTORY holding `config.json` + `model.safetensors` is read as one of the
    released CGM-JEPA encoders; anything else is a torch checkpoint.
    """
    ckpt_path = Path(ckpt_path)
    if ckpt_path.is_dir() and (ckpt_path / "config.json").exists():
        from . import cgmjepa
        model = cgmjepa.load(ckpt_path.name, device, root=ckpt_path.parent)
        return model, "cgmjepa", None
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "prism_config" in ck:
        fm = _cfg_from_dict(ck["fm_config"])
        pc = PrismConfig(**ck["prism_config"])
        model = GlucoPRISM(fm, pc)
        model.load_state_dict(ck["full"], strict=False)
        model.eval().to(device)
        return model, "prism", fm
    fm = _cfg_from_dict(ck["config"]) if "config" in ck else GlucoFMConfig()
    enc = GlucoFMEncoder(fm)
    enc.load_state_dict(ck["online"])
    enc.eval().to(device)
    return enc, "glucofm", fm


@torch.no_grad()
def extract(model, kind: str, x, m, s, device: str = "cpu",
            batch_size: int = 256, add_dim_controls: bool = True) -> dict:
    """Frozen features. For GlucoPRISM returns each block; for GlucoFM only 'full'.

    Also emits DIMENSION-MATCHED random-projection controls (`rand16`, `rand64`).
    Blocks differ in width (zT=64, zS=48, zA=16) and a narrower probe input is
    better regularised at these sample sizes (27-65 subjects), so a small block
    can win on PR-AUC for reasons that have nothing to do with what it encodes.
    Comparing zA against `rand16` separates the two explanations.
    """
    dl = DataLoader(EvalWindows(x, m, s), batch_size=batch_size, shuffle=False)
    acc = {}
    for b in dl:
        xb = b["x"].to(device); mb = b["m"].to(device); sb = b["s"].to(device)
        if kind == "prism":
            o = model.encode(xb, mb, sb)
            tok = o["tokens"]
            part = {"full": o["z"], "zT": o["zT"], "zS": o["zS"], "zA": o["zA"],
                    "zTS": torch.cat([o["zT"], o["zS"]], -1)}
            for k in ("zC", "cmp", "ema"):
                if k in o:
                    part[k] = o[k]
        elif kind == "cgmjepa":
            # PatchDataTransformer.encode: mean over patch tokens of the encoder
            # output, not the projection head
            tok = model(xb, mb, sb)
            part = {"full": tok.mean(1)}
        else:
            tok, _, _, _ = model(xb, mb, sb, patch_mask=None)
            part = {"full": tok.mean(1)}
        # `rich`: mean + sd + max over the 24 hourly tokens instead of mean alone.
        # Appendix C.6 mean-pools, which annihilates the WITHIN-DAY dispersion of
        # the token sequence - the state information the proposal's zS is meant
        # to hold. Available for both kinds so it is never a GlucoPRISM-only edge.
        part["rich"] = torch.cat([tok.mean(1), tok.std(1), tok.amax(1)], dim=-1)
        if "cmp" in part:
            # `rich` is the best pooled representation and `cmp` is the best
            # low-dimensional one; they are not obviously redundant, since `cmp`
            # is a nonlinear readout a linear probe cannot form from `rich`.
            part["richC"] = torch.cat([part["rich"], part["cmp"]], dim=-1)
        for k, v in part.items():
            acc.setdefault(k, []).append(v.float().cpu().numpy())
    out = {k: np.concatenate(v) for k, v in acc.items()}

    if add_dim_controls and "full" in out:
        rng = np.random.default_rng(0)
        d = out["full"].shape[1]
        for name, k in (("rand16", 16), ("rand64", 64)):
            if k < d:
                P = rng.normal(0, 1 / np.sqrt(k), size=(d, k))
                out[name] = out["full"] @ P
    return out


# --------------------------------------------------------------------------- #
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def subject_aggregate(emb: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """One vector per subject: [mean | sd | p10 | p90] of that subject's windows.

    Every label here is a SUBJECT property, so scoring one window at a time
    measures the model through a single noisy day. Measured on held-out runs
    this aggregation is worth ~+4 ROC-AUC on its own, independently of the
    encoder - so it is applied identically to every arm, including the GlucoFM
    baseline, and reported next to the legacy window-level number rather than
    instead of it.
    """
    us = np.unique(groups)
    X, Y = [], []
    for s in us:
        i = groups == s
        g = emb[i]
        X.append(np.concatenate([g.mean(0), g.std(0),
                                 np.percentile(g, 10, axis=0),
                                 np.percentile(g, 90, axis=0)]))
        v = pd.Series(y[i]).dropna()
        Y.append(v.mode().iloc[0] if len(v) else np.nan)
    return np.stack(X), np.array(Y), us


def _pick_C(A, ytr, gtr, n_classes, seed=0):
    """Choose L2 strength on an INNER subject-grouped split of the training fold."""
    try:
        itr, ite = next(iter(StratifiedGroupKFold(
            n_splits=3, shuffle=True, random_state=seed).split(A, ytr, gtr)))
    except ValueError:
        return 1.0
    if len(set(ytr[itr])) < n_classes or len(set(ytr[ite])) < 2:
        return 1.0
    best, bc = -1.0, 1.0
    for c in C_GRID:
        # max_iter is deliberately small here: this is model SELECTION, and at
        # subject level the design matrix is wide (4 x D features, 27-65 rows) so
        # the data are linearly separable and lbfgs never converges at large C -
        # it just burns the full iteration budget on every one of the ~29k inner
        # fits. The chosen C is then refit properly below.
        k = LogisticRegression(max_iter=300, C=c, class_weight="balanced")
        k.fit(A[itr], ytr[itr])
        p = k.predict_proba(A[ite])
        try:
            s = (roc_auc_score(ytr[ite], p[:, 1]) if n_classes == 2 else
                 roc_auc_score(ytr[ite], p, multi_class="ovr", average="macro",
                               labels=list(range(n_classes))))
        except ValueError:
            continue
        if s > best:
            best, bc = s, c
    return bc


def probe(emb: np.ndarray, y: np.ndarray, groups: np.ndarray,
          n_folds: int = 5, n_iter: int = 10, seed: int = 0,
          return_folds: bool = False, level: str = "window",
          sweep_c: bool = False):
    """Subject-grouped CV logistic probe. Returns metric means (+ per-fold PR)."""
    ok = ~pd.isna(y)
    emb, y, groups = emb[ok], y[ok].astype(int), groups[ok]
    if level == "subject":
        emb, y, groups = subject_aggregate(emb, y, groups)
        ok = ~pd.isna(y)
        emb, y, groups = emb[ok], y[ok].astype(int), groups[ok]
    if len(set(y)) < 2 or len(set(groups)) < n_folds:
        return ({"error": f"insufficient (pos={int((y==1).sum())}, "
                          f"neg={int((y==0).sum())}, subj={len(set(groups))})"},
                []) if return_folds else {"error": "insufficient"}

    pr, auc, f1 = [], [], []
    for it in range(n_iter):
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + it)
        try:
            splits = list(skf.split(emb, y, groups))
        except ValueError:
            continue
        for tr, te in splits:
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            sc = StandardScaler().fit(emb[tr])
            A, B = sc.transform(emb[tr]), sc.transform(emb[te])
            C = _pick_C(A, y[tr], groups[tr], 2, seed) if sweep_c else 1.0
            clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
            clf.fit(A, y[tr])
            p = clf.predict_proba(B)[:, 1]
            pr.append(average_precision_score(y[te], p))
            auc.append(roc_auc_score(y[te], p))
            f1.append(f1_score(y[te], (p >= .5).astype(int), average="macro"))
    if not pr:
        return ({"error": "no valid folds"}, []) if return_folds else {"error": "no valid folds"}
    res = {"pr_auc": 100 * float(np.mean(pr)), "pr_auc_std": 100 * float(np.std(pr)),
           "roc_auc": 100 * float(np.mean(auc)), "roc_auc_std": 100 * float(np.std(auc)),
           "macro_f1": 100 * float(np.mean(f1)), "macro_f1_std": 100 * float(np.std(f1)),
           "n_windows": int(len(y)), "n_subjects": int(len(set(groups))),
           "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
           "n_folds": len(pr)}
    return (res, [100 * v for v in pr]) if return_folds else res


def probe_multiclass(emb, y, groups, n_folds=5, n_iter=10, seed=0,
                     return_folds=False, level: str = "window",
                     sweep_c: bool = False):
    ok = np.array([v is not None and str(v) not in ("nan", "None") for v in y])
    emb, y, groups = emb[ok], y[ok], groups[ok]
    if level == "subject":
        emb, y, groups = subject_aggregate(emb, y, groups)
        ok = np.array([v is not None and str(v) not in ("nan", "None") for v in y])
        emb, y, groups = emb[ok], y[ok], groups[ok]
    classes = sorted(set(y.tolist()))
    if len(classes) < 2 or len(set(groups)) < n_folds:
        return ({"error": "insufficient"}, []) if return_folds else {"error": "insufficient"}
    cmap = {c: i for i, c in enumerate(classes)}
    yi = np.array([cmap[v] for v in y])
    pr, auc, f1 = [], [], []
    for it in range(n_iter):
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + it)
        try:
            splits = list(skf.split(emb, yi, groups))
        except ValueError:
            continue
        for tr, te in splits:
            if len(set(yi[tr])) < len(classes) or len(set(yi[te])) < 2:
                continue
            sc = StandardScaler().fit(emb[tr])
            A = sc.transform(emb[tr])
            C = _pick_C(A, yi[tr], groups[tr], len(classes), seed) if sweep_c else 1.0
            clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
            clf.fit(A, yi[tr])
            p = clf.predict_proba(sc.transform(emb[te]))
            oh = np.zeros_like(p); oh[np.arange(len(yi[te])), yi[te]] = 1
            present = sorted(set(yi[te]))
            # One-vs-rest macro AUC over the classes PRESENT in this test fold.
            # sklearn's multi_class="ovr" with `labels` spanning all classes
            # raises as soon as one class is absent, and at subject level a
            # 3-class 45-subject cohort has 9-subject folds where that happens
            # every time - which silently emptied `auc` and reported ROC-AUC as
            # NaN for both CGMacros `diabetes` cells. PR-AUC was already
            # computed this way; now AUC matches it.
            try:
                cell_pr = float(np.mean([average_precision_score(oh[:, c], p[:, c])
                                         for c in present]))
                cell_auc = float(np.mean([roc_auc_score(oh[:, c], p[:, c])
                                          for c in present]))
            except ValueError:
                continue
            pr.append(cell_pr)
            auc.append(cell_auc)
            f1.append(f1_score(yi[te], p.argmax(1), average="macro"))
    if not pr:
        return ({"error": "no valid folds"}, []) if return_folds else {"error": "no valid folds"}
    res = {"pr_auc": 100 * float(np.mean(pr)), "pr_auc_std": 100 * float(np.std(pr)),
           "roc_auc": 100 * float(np.mean(auc)), "roc_auc_std": 100 * float(np.std(auc)),
           "macro_f1": 100 * float(np.mean(f1)), "macro_f1_std": 100 * float(np.std(f1)),
           "n_windows": int(len(yi)), "n_subjects": int(len(set(groups))),
           "classes": {c: int((y == c).sum()) for c in classes}, "n_folds": len(pr)}
    return (res, [100 * v for v in pr]) if return_folds else res


# --------------------------------------------------------------------------- #
def load_cohort(cohort: str):
    """Downstream windows joined to labels. Returns (x, m, s, subj, label_table)."""
    stem, lab, idcol, tasks = TASKS[cohort]
    wf = WIN / f"{stem}.npz"
    lf = PROC / lab
    if not wf.exists() or not lf.exists():
        return None
    d = np.load(wf, allow_pickle=True)
    subj = np.array([str(v) for v in d["subj"]])
    lt = pd.read_csv(lf)
    lt[idcol] = lt[idcol].astype(str)
    if "split" in lt.columns:
        lt = lt[lt["split"] == "downstream"]
    lut = lt.set_index(idcol)
    keep = np.array([s in lut.index for s in subj])
    if keep.sum() == 0:
        return None
    return {"x": d["x"][keep], "m": d["m"][keep], "s": d["s"][keep],
            "subj": subj[keep], "lut": lut, "tasks": tasks, "idcol": idcol}


def label_vector(co: dict, col: str) -> np.ndarray:
    return co["lut"].loc[co["subj"], col].to_numpy()


# --------------------------------------------------------------------------- #
def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down; returns reject flags in the input order."""
    n = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * n
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (n - rank):
            reject[i] = True
        else:
            break
    return reject


def paired_test(a: list[float], b: list[float]) -> dict:
    """Paired Wilcoxon over matched folds + effect size (Cohen's d_z)."""
    n = min(len(a), len(b))
    if n < 6:
        return {"p": float("nan"), "effect": float("nan"), "n": n}
    a, b = np.asarray(a[:n]), np.asarray(b[:n])
    d = a - b
    if np.allclose(d, 0):
        return {"p": 1.0, "effect": 0.0, "n": n}
    try:
        _, p = wilcoxon(a, b)
    except ValueError:
        p = float("nan")
    return {"p": float(p), "effect": float(d.mean() / (d.std(ddof=1) + 1e-9)),
            "mean_diff": float(d.mean()), "n": n}
