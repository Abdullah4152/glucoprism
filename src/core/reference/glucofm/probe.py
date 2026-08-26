"""
Frozen-representation linear probe (section 4.2, Appendix C.6 / D).

Protocol: freeze the encoder, extract embeddings from non-overlapping 24-hour
windows, train logistic regression, and evaluate with subject-grouped
cross-validation so all windows from a subject fall in one fold. Reported over
10 iterations of 5-fold CV. Metrics: PR-AUC, ROC-AUC, Macro-F1.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import Config, DEFAULT
from .data import EmbedDataset, load_windows
from .model import GlucoFMEncoder

warnings.simplefilter("ignore")


@torch.no_grad()
def extract_embeddings(encoder: GlucoFMEncoder, data: dict, device: str,
                       batch_size: int = 256) -> np.ndarray:
    encoder.eval().to(device)
    dl = DataLoader(EmbedDataset(data), batch_size=batch_size, shuffle=False)
    out = []
    for b in dl:
        z = encoder.embed(b["x"].to(device), b["m"].to(device), b["s"].to(device))
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def probe_task(emb: np.ndarray, y: np.ndarray, groups: np.ndarray,
               cfg: Config = DEFAULT) -> dict:
    """Subject-grouped CV logistic probe for one binary task."""
    pc = cfg.probe
    ok = ~np.isnan(y.astype(float))
    emb, y, groups = emb[ok], y[ok].astype(int), groups[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    n_subj = len(set(groups))
    if n_pos == 0 or n_neg == 0 or n_subj < pc.n_folds:
        return {"error": f"insufficient data (pos={n_pos}, neg={n_neg}, subj={n_subj})"}

    pr, auc, f1 = [], [], []
    for it in range(pc.n_iterations):
        skf = StratifiedGroupKFold(n_splits=pc.n_folds, shuffle=True,
                                   random_state=pc.seed + it)
        try:
            splits = list(skf.split(emb, y, groups))
        except ValueError as e:
            return {"error": f"CV split failed: {e}"}
        for tr, te in splits:
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            xtr, xte = emb[tr], emb[te]
            if pc.standardize:
                sc = StandardScaler().fit(xtr)
                xtr, xte = sc.transform(xtr), sc.transform(xte)
            clf = LogisticRegression(max_iter=pc.max_iter, C=pc.C,
                                     class_weight="balanced")
            clf.fit(xtr, y[tr])
            p = clf.predict_proba(xte)[:, 1]
            pr.append(average_precision_score(y[te], p))
            auc.append(roc_auc_score(y[te], p))
            f1.append(f1_score(y[te], (p >= 0.5).astype(int), average="macro"))

    if not pr:
        return {"error": "no valid folds"}
    return {
        "pr_auc": 100 * float(np.mean(pr)), "pr_auc_std": 100 * float(np.std(pr)),
        "roc_auc": 100 * float(np.mean(auc)), "roc_auc_std": 100 * float(np.std(auc)),
        "macro_f1": 100 * float(np.mean(f1)), "macro_f1_std": 100 * float(np.std(f1)),
        "n_windows": int(len(y)), "n_subjects": int(n_subj),
        "n_pos": n_pos, "n_neg": n_neg, "n_folds_run": len(pr),
    }


def probe_multiclass(emb: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     cfg: Config = DEFAULT) -> dict:
    """3-class variant (CGMacros diabetes risk): macro one-vs-rest metrics."""
    pc = cfg.probe
    ok = np.array([v is not None and str(v) != "nan" for v in y])
    emb, y, groups = emb[ok], y[ok], groups[ok]
    classes = sorted(set(y.tolist()))
    if len(classes) < 2 or len(set(groups)) < pc.n_folds:
        return {"error": f"insufficient data (classes={classes})"}
    cmap = {c: i for i, c in enumerate(classes)}
    yi = np.array([cmap[v] for v in y])

    pr, auc, f1 = [], [], []
    for it in range(pc.n_iterations):
        skf = StratifiedGroupKFold(n_splits=pc.n_folds, shuffle=True,
                                   random_state=pc.seed + it)
        try:
            splits = list(skf.split(emb, yi, groups))
        except ValueError as e:
            return {"error": f"CV split failed: {e}"}
        for tr, te in splits:
            if len(set(yi[tr])) < len(classes) or len(set(yi[te])) < 2:
                continue
            xtr, xte = emb[tr], emb[te]
            if pc.standardize:
                sc = StandardScaler().fit(xtr)
                xtr, xte = sc.transform(xtr), sc.transform(xte)
            clf = LogisticRegression(max_iter=pc.max_iter, C=pc.C,
                                     class_weight="balanced")
            clf.fit(xtr, yi[tr])
            p = clf.predict_proba(xte)
            present = sorted(set(yi[te]))
            oh = np.zeros_like(p)
            oh[np.arange(len(yi[te])), yi[te]] = 1
            try:
                pr.append(np.mean([average_precision_score(oh[:, c], p[:, c])
                                   for c in present]))
                auc.append(roc_auc_score(yi[te], p, multi_class="ovr",
                                         average="macro", labels=list(range(len(classes)))))
            except ValueError:
                continue
            f1.append(f1_score(yi[te], p.argmax(1), average="macro"))

    if not pr:
        return {"error": "no valid folds"}
    return {
        "pr_auc": 100 * float(np.mean(pr)), "pr_auc_std": 100 * float(np.std(pr)),
        "roc_auc": 100 * float(np.mean(auc)), "roc_auc_std": 100 * float(np.std(auc)),
        "macro_f1": 100 * float(np.mean(f1)), "macro_f1_std": 100 * float(np.std(f1)),
        "n_windows": int(len(yi)), "n_subjects": int(len(set(groups))),
        "classes": {c: int((y == c).sum()) for c in classes},
        "n_folds_run": len(pr),
    }


def load_encoder(ckpt_path: Path, device: str = "cpu") -> tuple[GlucoFMEncoder, Config]:
    from .config import (AugConfig, Config, FilterConfig, GridConfig,
                         ModelConfig, PretrainConfig, ProbeConfig)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck.get("config", {})
    cfg = Config(
        grid=GridConfig(**c["grid"]), filt=FilterConfig(**c["filt"]),
        model=ModelConfig(**c["model"]), pretrain=PretrainConfig(**c["pretrain"]),
        aug=AugConfig(**c["aug"]), probe=ProbeConfig(**c["probe"]),
    ) if c else DEFAULT
    enc = GlucoFMEncoder(cfg)
    enc.load_state_dict(ck["online"])
    enc.eval()
    return enc, cfg
