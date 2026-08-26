"""
GlucoPRISM experiments E1-E9 (proposal section 6).

E1 subject-disjoint linear probing        E6 leakage audit / protocol gain
E2 cross-device transfer                  E7 mask-shortcut audit
E3 trait stability                        E8 few-shot + pretraining scale
E4 block routing                          E9 ablations (driven by run configs)
E5 multi-day scaling
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .evaluate import (PROC, STATE_TASKS, TASKS, TRAIT_TASKS, WIN, extract,
                       holm_bonferroni, label_vector, load_cohort, paired_test,
                       probe, probe_multiclass)

MULTICLASS = {("cgmacros_dexcom", "diabetes"), ("cgmacros_libre", "diabetes")}


def _feats(model, kind, co, device):
    return extract(model, kind, co["x"], co["m"], co["s"], device)


def _run_probe(cohort, task, emb, y, subj, folds=False, **kw):
    if (cohort, task) in MULTICLASS:
        return probe_multiclass(emb, y.astype(object), subj, return_folds=folds, **kw)
    return probe(emb, pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(),
                 subj, return_folds=folds, **kw)


# --------------------------------------------------------------------------- #
def E1_linear_probe(model, kind, device="cpu", blocks=("full",)) -> dict:
    """Frozen encoder + logistic probe, 5-fold subject-grouped CV x 10 iters."""
    out, folds = {}, {}
    for cohort in TASKS:
        co = load_cohort(cohort)
        if co is None:
            continue
        F = _feats(model, kind, co, device)
        out[cohort] = {}
        for task, col in co["tasks"].items():
            if col not in co["lut"].columns:
                continue
            y = label_vector(co, col)
            out[cohort][task] = {}
            for blk in blocks:
                if blk not in F:
                    continue
                r, pf = _run_probe(cohort, task, F[blk], y, co["subj"], folds=True)
                out[cohort][task][blk] = r
                folds[(cohort, task, blk)] = pf
    return {"results": out, "folds": {f"{a}|{b}|{c}": v for (a, b, c), v in folds.items()}}


def E1b_subject_probe(model, kind, device="cpu", blocks=("full",)) -> dict:
    """The same 18 cells under the SUBJECT-LEVEL, regularisation-tuned protocol.

    Two changes from E1, both evaluation-side and both applied identically to
    every arm including the GlucoFM baseline, so they cannot flatter GlucoPRISM:

      1. one vector per subject ([mean|sd|p10|p90] over that subject's windows)
         instead of one row per window - the labels are subject properties
      2. the logistic L2 strength is chosen on an inner subject-grouped split of
         each training fold instead of being fixed at C=1

    Measured separately (results/headroom.csv) these are worth about +4 and +1.6
    ROC-AUC respectively, on the encoder alone. They are reported ALONGSIDE E1,
    never as a replacement, because a protocol change is not a model result.
    """
    out, folds = {}, {}
    for cohort in TASKS:
        co = load_cohort(cohort)
        if co is None:
            continue
        F = _feats(model, kind, co, device)
        out[cohort] = {}
        for task, col in co["tasks"].items():
            if col not in co["lut"].columns:
                continue
            y = label_vector(co, col)
            out[cohort][task] = {}
            for blk in blocks:
                if blk not in F:
                    continue
                r, pf = _run_probe(cohort, task, F[blk], y, co["subj"], folds=True,
                                   level="subject", sweep_c=True)
                out[cohort][task][blk] = r
                folds[(cohort, task, blk)] = pf
    return {"results": out, "folds": {f"{a}|{b}|{c}": v for (a, b, c), v in folds.items()}}


def E1c_learned_agg(model, kind, device="cpu") -> dict:
    """Proposition 2 as intended: the LEARNED DeepSets aggregator over days.

    E1b builds each subject's row with a fixed `[mean|sd|p10|p90]`. The proposal
    instead specifies a permutation-invariant learned aggregator over the set of
    a subject's daily state blocks, with the trait block mean-pooled (the
    minimum-variance estimator for a day-invariant quantity) and the state block
    aggregated through phi. This runs exactly that:

        subject row = [ mean_k zT^(k)  |  aggregator({zS^(k)}) ]

    against the same subject-level, C-swept probe, so the only difference from
    E1b is the aggregation rule.
    """
    if kind != "prism":
        return {"error": "needs the blocked representation"}
    agg = getattr(model, "aggregator", None)
    if agg is None:
        return {"error": "no aggregator on this checkpoint"}

    out = {}
    for cohort in TASKS:
        co = load_cohort(cohort)
        if co is None:
            continue
        F = _feats(model, kind, co, device)
        if "zT" not in F or "zS" not in F:
            continue
        subj = co["subj"]
        us = np.unique(subj)
        rows = []
        with torch.no_grad():
            for s in us:
                i = np.flatnonzero(subj == s)
                zt = torch.from_numpy(F["zT"][i]).float().to(device)
                zs = torch.from_numpy(F["zS"][i]).float().to(device).unsqueeze(0)
                rows.append(np.concatenate([
                    zt.mean(0).cpu().numpy(),
                    agg(zs).squeeze(0).cpu().numpy()]))
        X = np.stack(rows)

        out[cohort] = {}
        for task, col in co["tasks"].items():
            if col not in co["lut"].columns:
                continue
            y_win = label_vector(co, col)
            y = np.array([pd.Series(y_win[subj == s]).dropna().mode().iloc[0]
                          if len(pd.Series(y_win[subj == s]).dropna()) else np.nan
                          for s in us], dtype=object)
            # already one row per subject, so probe at "window" level on it
            out[cohort][task] = _run_probe(cohort, task, X, y, us, sweep_c=True)
    return out


def E1d_day_encoder(model, kind, device="cpu") -> dict:
    """Aggregate a subject's days with the TRAINED day encoder instead of
    fixed order statistics.

    E1b builds each subject's row with `[mean|sd|p10|p90]`, which is a
    hand-chosen rule. E1c used the DeepSets aggregator trained on a proxy
    objective (between-day dispersion) and was only weakly better. E1d uses the
    day encoder trained by `L_dayjepa` on the task it will actually be asked to
    do: summarise a set of days into a trait estimate.
    """
    if kind != "prism" or getattr(model, "day_encoder", None) is None:
        return {"error": "checkpoint has no trained day encoder"}

    out = {}
    for cohort in TASKS:
        co = load_cohort(cohort)
        if co is None:
            continue
        F = _feats(model, kind, co, device)
        if "zT" not in F or "zS" not in F:
            continue
        subj = co["subj"]
        us = np.unique(subj)
        rows = []
        with torch.no_grad():
            for s in us:
                i = np.flatnonzero(subj == s)
                rows.append(model.encode_subject(
                    torch.from_numpy(F["zT"][i]).float().to(device),
                    torch.from_numpy(F["zS"][i]).float().to(device)
                ).cpu().numpy())
        X = np.stack(rows)

        out[cohort] = {}
        for task, col in co["tasks"].items():
            if col not in co["lut"].columns:
                continue
            y_win = label_vector(co, col)
            y = np.array([pd.Series(y_win[subj == s]).dropna().mode().iloc[0]
                          if len(pd.Series(y_win[subj == s]).dropna()) else np.nan
                          for s in us], dtype=object)
            out[cohort][task] = _run_probe(cohort, task, X, y, us, sweep_c=True)
    return out


# --------------------------------------------------------------------------- #
def E2_cross_device(model, kind, device="cpu", blocks=("full",)) -> dict:
    """Train the probe on one device, test on another. No target labels in training."""
    res = {}

    # (a) clean setting: CGMacros Dexcom -> Libre (same subjects, same days)
    src, tgt = load_cohort("cgmacros_dexcom"), load_cohort("cgmacros_libre")
    if src and tgt:
        Fs, Ft = _feats(model, kind, src, device), _feats(model, kind, tgt, device)
        res["cgmacros_dexcom->libre"] = _transfer(Fs, src, Ft, tgt, blocks)

    # (b) hard setting: Dexcom cohorts -> ShanghaiT2DM (Libre, 15-min)
    hall, sh = load_cohort("hall"), load_cohort("shanghai")
    if hall and sh:
        Fh, Fsh = _feats(model, kind, hall, device), _feats(model, kind, sh, device)
        res["hall->shanghai"] = _transfer(Fh, hall, Fsh, sh, blocks,
                                          shared=("ir", "hyperlipidemia"))
    return res


def _transfer(Fs, src, Ft, tgt, blocks, shared=None):
    out = {}
    tasks = shared or [t for t in src["tasks"] if t in tgt["tasks"]]
    for task in tasks:
        cs, ct = src["tasks"].get(task), tgt["tasks"].get(task)
        if not cs or not ct or cs not in src["lut"].columns or ct not in tgt["lut"].columns:
            continue
        ys = pd.to_numeric(pd.Series(label_vector(src, cs)), errors="coerce").to_numpy()
        yt = pd.to_numeric(pd.Series(label_vector(tgt, ct)), errors="coerce").to_numpy()
        out[task] = {}
        for blk in blocks:
            if blk not in Fs or blk not in Ft:
                continue
            a, b = ~pd.isna(ys), ~pd.isna(yt)
            if len(set(ys[a].astype(int))) < 2 or len(set(yt[b].astype(int))) < 2:
                continue
            sc = StandardScaler().fit(Fs[blk][a])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(Fs[blk][a]), ys[a].astype(int))
            p = clf.predict_proba(sc.transform(Ft[blk][b]))[:, 1]
            out[task][blk] = {
                "pr_auc": 100 * float(average_precision_score(yt[b].astype(int), p)),
                "roc_auc": 100 * float(roc_auc_score(yt[b].astype(int), p)),
                "n_src": int(a.sum()), "n_tgt": int(b.sum()),
            }
    return out


# --------------------------------------------------------------------------- #
def E3_trait_stability(model, kind, device="cpu") -> dict:
    """Within-subject across-day consistency of each block (the paper's core claim).

    Reports mean within-subject cosine similarity, the between-subject baseline,
    and ICC(1) = (MSB - MSW) / (MSB + (k-1) MSW) on the embedding norm-free
    cosine structure. A day-invariant block should have high within-subject
    similarity relative to between-subject.
    """
    out = {}
    for cohort in TASKS:
        co = load_cohort(cohort)
        if co is None:
            continue
        F = _feats(model, kind, co, device)
        subj = co["subj"]
        out[cohort] = {}
        rng = np.random.default_rng(0)
        for blk, E in F.items():
            En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            # Both sides must be measured at WINDOW level. Comparing
            # window-level within-subject pairs against SUBJECT-MEAN
            # between-subject pairs is not like-for-like: averaging smooths
            # noise and inflates the between-subject similarity, which made
            # every block look like it had negative separation.
            within, between = [], []
            for s in np.unique(subj):
                idx = np.flatnonzero(subj == s)
                if len(idx) < 2:
                    continue
                G = En[idx]
                iu = np.triu_indices(len(idx), k=1)
                within.append(float((G @ G.T)[iu].mean()))
                # equally many random cross-subject window pairs
                other = np.flatnonzero(subj != s)
                if len(other) == 0:
                    continue
                a = rng.choice(idx, size=min(len(idx), 64), replace=True)
                b = rng.choice(other, size=len(a), replace=True)
                between.append(float((En[a] * En[b]).sum(1).mean()))
            if not within or not between:
                continue
            w, bt = float(np.mean(within)), float(np.mean(between))
            icc = (w - bt) / (1.0 - bt + 1e-9)
            out[cohort][blk] = {"within_subject_cos": w, "between_subject_cos": bt,
                                "separation": w - bt, "icc_like": float(icc),
                                "n_subjects": int(len(within))}
    return out


# --------------------------------------------------------------------------- #
def E4_block_routing(e1: dict) -> dict:
    """Which block wins which task (the proposal's Figure 2).

    Prediction: trait tasks -> zT; state tasks -> zS; zA near chance clinically.
    """
    rows = []
    for cohort, tasks in e1["results"].items():
        for task, blocks in tasks.items():
            best, best_v = None, -1
            rec = {"cohort": cohort, "task": task,
                   "kind": "trait" if task in TRAIT_TASKS else
                           ("state" if task in STATE_TASKS else "other")}
            for blk, r in blocks.items():
                if "error" in r:
                    continue
                rec[blk] = r["pr_auc"]
                if blk in ("zT", "zS", "zA") and r["pr_auc"] > best_v:
                    best, best_v = blk, r["pr_auc"]
            rec["winner"] = best
            rows.append(rec)
    df = pd.DataFrame(rows)
    summary = {}
    if not df.empty and "winner" in df:
        for k in ("trait", "state"):
            sub = df[df["kind"] == k]
            if len(sub):
                summary[k] = sub["winner"].value_counts().to_dict()
        if "zA" in df:
            summary["zA_mean_pr"] = float(df["zA"].mean())
    return {"table": rows, "summary": summary}


def E4_device_control(model, kind, device="cpu") -> dict:
    """Positive control: zA should predict DEVICE IDENTITY near-perfectly."""
    if kind != "prism":
        return {"error": "device control needs the blocked representation"}
    dex, lib = load_cohort("cgmacros_dexcom"), load_cohort("cgmacros_libre")
    if not dex or not lib:
        return {"error": "CGMacros windows unavailable"}
    Fd, Fl = _feats(model, kind, dex, device), _feats(model, kind, lib, device)
    out = {}
    for blk in ("full", "zT", "zS", "zA"):
        if blk not in Fd:
            continue
        X = np.concatenate([Fd[blk], Fl[blk]])
        y = np.concatenate([np.zeros(len(Fd[blk])), np.ones(len(Fl[blk]))]).astype(int)
        g = np.concatenate([dex["subj"], lib["subj"]])
        out[blk] = probe(X, y, g, n_iter=3)
    return out


# --------------------------------------------------------------------------- #
def E5_multiday(model, kind, device="cpu", kmax=7) -> dict:
    """Delta_K = PR-AUC(K) - PR-AUC(1) under three aggregation strategies."""
    out = {}
    for cohort in TASKS:
        co = load_cohort(cohort)
        if co is None:
            continue
        F = _feats(model, kind, co, device)
        subj = co["subj"]
        out[cohort] = {}
        for task, col in co["tasks"].items():
            if col not in co["lut"].columns or (cohort, task) in MULTICLASS:
                continue
            y_all = pd.to_numeric(pd.Series(label_vector(co, col)),
                                  errors="coerce").to_numpy()
            out[cohort][task] = {}
            for strat in ("mean", "mean_max", "set"):
                curve = {}
                for K in range(1, kmax + 1):
                    Xs, ys, gs = [], [], []
                    for s in np.unique(subj):
                        idx = np.flatnonzero(subj == s)
                        if len(idx) < 1:
                            continue
                        rng = np.random.default_rng(hash(s) % 2**31)
                        for _ in range(3):                     # 3 draws per subject
                            take = rng.choice(idx, size=min(K, len(idx)),
                                              replace=len(idx) < K)
                            E = F["zTS"] if ("zTS" in F and strat == "set") else F["full"]
                            g = E[take]
                            if strat == "mean":
                                v = g.mean(0)
                            elif strat == "mean_max":
                                v = np.concatenate([g.mean(0), g.max(0)])
                            else:   # set: mean for trait half, dispersion for state half
                                # np.ptp(arr, axis=) - ndarray.ptp was removed in NumPy 2.0
                                v = np.concatenate([g.mean(0), g.std(0), np.ptp(g, axis=0)])
                            Xs.append(v); ys.append(y_all[take[0]]); gs.append(s)
                    Xs = np.stack(Xs); ys = np.asarray(ys); gs = np.asarray(gs)
                    r = probe(Xs, ys, gs, n_iter=3)
                    curve[K] = r.get("pr_auc")
                base = curve.get(1)
                out[cohort][task][strat] = {
                    "curve": curve,
                    "delta": {k: (None if (v is None or base is None) else v - base)
                              for k, v in curve.items()},
                }
    return out


# --------------------------------------------------------------------------- #
def _stanford_ogtt_windows(length=288, dt=5):
    """Build OGTT-aligned Stanford windows (CGM-JEPA's evaluation setting).

    The OGTT trace is 39 points on a -10..180 min grid; it is placed at the start
    of an otherwise-unobserved 288-point window so the same encoder can read it.
    """
    f = ROOT_DATA / "stanford" / "filtered_ogtt_glucose_timeseries_ctru_athome_venous_cgm_03222026.csv"
    if not f.exists():
        return None
    og = pd.read_csv(f)
    out = {}
    for method in og["SampleLocation_ExtractionMethod"].unique():
        sub = og[og["SampleLocation_ExtractionMethod"] == method]
        xs, ms, ss, subj = [], [], [], []
        for sid, g in sub.groupby("SubjectID"):
            g = g.sort_values("Timepoint")
            t = g["Timepoint"].to_numpy().astype(float)
            v = pd.to_numeric(g["Glucose"], errors="coerce").to_numpy()
            ok = np.isfinite(v)
            if ok.sum() < 8:
                continue
            j = ((t[ok] - t[ok].min()) / dt).round().astype(int)
            keep = (j >= 0) & (j < length)
            x = np.zeros(length, dtype=np.float32); m = np.zeros(length, dtype=np.float32)
            x[j[keep]] = v[ok][keep]; m[j[keep]] = 1.0
            xs.append(x); ms.append(m); ss.append(96); subj.append(str(sid))  # 08:00 start
        if xs:
            out[method] = {"x": np.stack(xs), "m": np.stack(ms),
                           "s": np.array(ss), "subj": np.array(subj)}
    return out


ROOT_DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def E6_leakage_audit(model, kind, device="cpu", blocks=("full",)) -> dict:
    """protocol gain = AUROC(OGTT-aligned) - AUROC(free-living), per method.

    If the gain is large and roughly method-independent, published CGM
    representation numbers are ranking the evaluation protocol, not the model.
    """
    og = _stanford_ogtt_windows()
    co = load_cohort("stanford")
    if og is None or co is None:
        return {"error": "Stanford OGTT or downstream windows unavailable"}
    lut, tasks = co["lut"], co["tasks"]
    free = _feats(model, kind, co, device)

    res = {"free_living": {}, "ogtt": {}, "protocol_gain": {}}
    for task, col in tasks.items():
        if col not in lut.columns:
            continue
        y = pd.to_numeric(pd.Series(label_vector(co, col)), errors="coerce").to_numpy()
        for blk in blocks:
            if blk in free:
                res["free_living"].setdefault(task, {})[blk] = probe(
                    free[blk], y, co["subj"], n_iter=5)

    for method, d in og.items():
        keep = np.array([s in lut.index for s in d["subj"]])
        if keep.sum() < 10:
            continue
        F = extract(model, kind, d["x"][keep], d["m"][keep], d["s"][keep], device)
        subj = d["subj"][keep]
        for task, col in tasks.items():
            if col not in lut.columns:
                continue
            y = pd.to_numeric(pd.Series(lut.loc[subj, col].to_numpy()),
                              errors="coerce").to_numpy()
            for blk in blocks:
                if blk not in F:
                    continue
                r = probe(F[blk], y, subj, n_iter=5)
                res["ogtt"].setdefault(method, {}).setdefault(task, {})[blk] = r
                fr = res["free_living"].get(task, {}).get(blk, {})
                if "roc_auc" in r and "roc_auc" in fr:
                    res["protocol_gain"].setdefault(method, {}).setdefault(task, {})[blk] = \
                        r["roc_auc"] - fr["roc_auc"]
    return res


# --------------------------------------------------------------------------- #
def E7_mask_shortcut() -> dict:
    """Probe on MASK-DERIVED features only. Model-independent by construction.

    Non-wear is not missing-at-random; if mask-only AUC > 60 on any task, mask
    preservation is partly a shortcut and must be controlled.
    """
    out = {}
    for cohort in TASKS:
        co = load_cohort(cohort)
        if co is None:
            continue
        m = co["m"]
        P, K = 24, 12
        dens = m.reshape(len(m), P, K).mean(-1)
        gaps = (1.0 - m)
        feats = np.concatenate([
            m.mean(1, keepdims=True), dens,
            gaps.sum(1, keepdims=True),
            (np.diff(m, axis=1) != 0).sum(1, keepdims=True).astype(float),
            m[:, :144].mean(1, keepdims=True), m[:, 144:].mean(1, keepdims=True),
        ], axis=1)
        out[cohort] = {}
        for task, col in co["tasks"].items():
            if col not in co["lut"].columns or (cohort, task) in MULTICLASS:
                continue
            y = pd.to_numeric(pd.Series(label_vector(co, col)), errors="coerce").to_numpy()
            out[cohort][task] = probe(feats, y, co["subj"], n_iter=5)
    return out


# --------------------------------------------------------------------------- #
def E8_fewshot(model, kind, device="cpu", ks=(1, 2, 3, 4, 5), block="full") -> dict:
    """K labelled support subjects per class."""
    out = {}
    for cohort in TASKS:
        co = load_cohort(cohort)
        if co is None:
            continue
        F = _feats(model, kind, co, device)
        if block not in F:
            continue
        E, subj = F[block], co["subj"]
        out[cohort] = {}
        for task, col in co["tasks"].items():
            if col not in co["lut"].columns or (cohort, task) in MULTICLASS:
                continue
            y = pd.to_numeric(pd.Series(label_vector(co, col)), errors="coerce").to_numpy()
            ok = ~pd.isna(y)
            Ei, yi, si = E[ok], y[ok].astype(int), subj[ok]
            # subject-level label for support selection
            sl = {s: int(pd.Series(yi[si == s]).mode().iloc[0]) for s in np.unique(si)}
            out[cohort][task] = {}
            for K in ks:
                scores = []
                for rep in range(10):
                    rng = np.random.default_rng(rep)
                    sup = []
                    for cls in (0, 1):
                        pool = [s for s, v in sl.items() if v == cls]
                        if len(pool) < K + 1:
                            sup = []
                            break
                        sup += list(rng.choice(pool, K, replace=False))
                    if not sup:
                        continue
                    tr = np.isin(si, sup); te = ~tr
                    if len(set(yi[tr])) < 2 or len(set(yi[te])) < 2:
                        continue
                    sc = StandardScaler().fit(Ei[tr])
                    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                    clf.fit(sc.transform(Ei[tr]), yi[tr])
                    p = clf.predict_proba(sc.transform(Ei[te]))[:, 1]
                    scores.append(100 * average_precision_score(yi[te], p))
                out[cohort][task][K] = (
                    {"pr_auc": float(np.mean(scores)), "std": float(np.std(scores)),
                     "n_reps": len(scores)} if scores else {"error": "insufficient"})
    return out
