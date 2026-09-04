"""Evaluate the released models on an external cohort, zero-shot.

    python evaluate_external.py --shard mycohort.npz --labels mycohort.csv

This is the driver behind the external-validation table in the paper. It is
deliberately cohort-agnostic: it knows nothing about any particular dataset, and
takes only the two artefacts every cohort in this repository is reduced to
before probing --- a window shard and a subject-level label table. Run it on a
cohort of your own and you get the paper's protocol, unchanged.

WHY THERE IS NO LOADER FOR THE PAPER'S EXTERNAL COHORT. The Human Phenotype
Project is owned by Pheno.AI, governed by a data use agreement, and reachable
only inside their trusted research environment. Its loader and label
construction would encode that dataset's internal schema, which is not ours to
publish, so they stay inside the enclave. The evaluation itself --- frozen
encoder, per-subject aggregation, linear probe --- is what matters for
reproducibility and is what you are reading. Anyone with HPP access can rebuild
the two inputs below from the description in the paper's appendix and reproduce
the table with this script; anyone without can run the identical protocol on
their own data.

INPUTS

  --shard   an .npz in the format `build_corpus.py` writes:
              glucose    (n, 288) float, NaN where unobserved
              mask       (n, 288) float, 1 where observed
              start_idx  (n,)     int, circadian phase of the window start
              subject    (n,)     str, dataset-prefixed subject id
            Optional: any further (n,) column may be used with --split-col.

  --labels  a .csv with a `subject` column matching the shard, and one further
            column per endpoint holding 0/1 (or NaN where unmeasured). Every
            non-`subject` column is treated as an endpoint unless --tasks is
            given. Rows missing a label are dropped for that endpoint only.

OUTPUT goes to GLUCOPRISM_OUT (default ./artifacts), never into the repository:
this project ships code and weights, not results.
"""
from __future__ import annotations

import os as _os, sys as _sys
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))
REFERENCE = ROOT / "src" / "core" / "released_model"
for _p in (ROOT / "src" / "core", ROOT / "baselines", ROOT / "src" / "scripts",
           REFERENCE, _P(__file__).resolve().parent):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

warnings.filterwarnings("ignore")

from cgmkit import release_weights as _rw            # noqa: E402
from cgmkit.data.gluformer_tokens import to_tokens   # noqa: E402
from cgmkit.data.windows import densify              # noqa: E402
from cgmkit.eval.probe import glucofm_probe          # noqa: E402
from cgmkit.models.glucofm import GlucoFM, GlucoFMConfig       # noqa: E402
from common.models.cgm_jepa import Encoder                     # noqa: E402
from common.models.gluformer import GluFormer, GluFormerConfig  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRICS = ("roc_auc", "pr_auc", "macro_f1")


# --------------------------------------------------------------- checkpoints

def read_release(model: str) -> tuple[dict, dict]:
    """Flat tensor map plus its sidecar config, as `weights/` ships them."""
    ck = _rw.checkpoint(model)
    cfg_p = _rw.config(model)
    cfg = json.loads(cfg_p.read_text()) if cfg_p else {}
    return load_file(str(ck)), cfg


def section(tensors: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}


def auto_state_dict(tensors: dict, module) -> tuple[dict, list[str]]:
    """Map a released tensor map onto a module's state_dict, section by section.

    `export_safetensors.py` flattens each checkpoint and then deduplicates
    tensors that share storage, keeping the SHORTER name. A module reachable by
    two paths therefore ends up split across sections -- GlucoFM's encoder is
    both `model.<deep.path>` and `online.<path>`, and the short names win -- so
    no single prefix restores the whole network. Rather than hardcode that, find
    for each section the mount point inside the target's state_dict that its key
    suffixes actually match, and take the assignment with the most hits.
    Sections matching nothing are reported unused: that is where the EMA target
    branch and the training heads land, and they are not on the embedding path.
    """
    want = set(module.state_dict())
    targets = [""]
    for k in want:
        bits = k.split(".")
        for depth in (1, 2):
            if len(bits) > depth:
                targets.append(".".join(bits[:depth]) + ".")
    targets = sorted(set(targets))

    sd: dict = {}
    plan: list[str] = []
    for sec in sorted({k.split(".")[0] for k in tensors}):
        items = section(tensors, sec + ".")
        best, hits = "", 0
        for tp in targets:
            h = sum(1 for k in items if tp + k in want)
            if h > hits:
                best, hits = tp, h
        if not hits:
            plan.append(f"{sec}.->unused({len(items)})")
            continue
        for k, v in items.items():
            if best + k in want and best + k not in sd:
                sd[best + k] = v
        plan.append(f"{sec}.->{best or '<root>'}({hits})")
    return sd, plan


def load_into(name: str, tensors: dict, module):
    """Load, and say plainly whether the result is a fully restored network.

    A checkpoint that loads with missing keys still produces embeddings; they
    are just embeddings of a partly random network, and no downstream metric
    would reveal it. So the count is printed every time, not only on failure.
    """
    sd, plan = auto_state_dict(tensors, module)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    flag = "" if not missing else "   <-- SUSPECT, embeddings are not trustworthy"
    print(f"  {name:<18} {' '.join(plan)}  "
          f"missing={len(missing)} unexpected={len(unexpected)}{flag}")
    if missing:
        print(f"  {'':<18} first missing: {sorted(missing)[:6]}")
    return module


# ---------------------------------------------------------------- embedders

@torch.no_grad()
def emb_glucofm(model: str, g, m, s, batch=256):
    t, cfg = read_release(model)
    net = GlucoFM(GlucoFMConfig(**cfg.get("cfg", cfg))).to(DEVICE)
    load_into(model, t, net).eval()
    out = []
    for i in range(0, len(g), batch):
        sl = slice(i, i + batch)
        gg = torch.from_numpy(g[sl]).float().to(DEVICE)
        mm = torch.from_numpy(m[sl]).float().to(DEVICE)
        ss = torch.from_numpy(s[sl]).long().to(DEVICE)
        out.append(net.embed(gg * mm, mm, ss).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def emb_cgm_jepa(model: str, g, m, s, batch=256):
    t, cfg = read_release(model)
    c = cfg.get("cfg", cfg)
    enc = Encoder(dim_in=c["patch_size"], kernel_size=c["encoder_kernel_size"],
                  embed_dim=c["encoder_embed_dim"],
                  embed_bias=c["encoder_embed_bias"], nhead=c["encoder_nhead"],
                  num_layers=c["encoder_num_layers"], jepa=True,
                  time_inp_dim=c["time_inp_dim"],
                  drop_rate=c["encoder_dropout"]).to(DEVICE).eval()
    # Explicit, not auto-mapped: X-CGM-JEPA also ships a Glucodensity encoder
    # whose keys would match this module just as well, and the embedding path is
    # the `encoder` in both variants.
    miss, unexp = enc.load_state_dict(section(t, "encoder."), strict=False)
    print(f"  {model:<18} encoder.  missing={len(miss)} unexpected={len(unexp)}")
    P = 288 // c["patch_size"]
    out = []
    for i in range(0, len(g), batch):
        sl = slice(i, i + batch)
        dense = np.stack([densify(g[j], m[j]) for j in range(*sl.indices(len(g)))])
        x = torch.from_numpy(dense.reshape(len(dense), P, c["patch_size"])
                             ).float().to(DEVICE)
        xm = torch.zeros(len(dense), P, c["patch_size"], c["time_inp_dim"],
                         device=DEVICE)
        tokens, _ = enc(x, xm, mask=None)
        out.append(tokens.mean(dim=1).cpu().numpy())     # paper M.5.3
    return np.concatenate(out)


@torch.no_grad()
def emb_gluformer(model: str, g, m, s, batch=64):
    t, cfg = read_release(model)
    net = GluFormer(GluFormerConfig(**cfg.get("cfg", cfg))).to(DEVICE)
    load_into(model, t, net).eval()
    out = []
    for i in range(0, len(g), batch):
        sl = slice(i, i + batch)
        dense = np.stack([densify(g[j], m[j]) for j in range(*sl.indices(len(g)))])
        tok = torch.from_numpy(np.stack([to_tokens(d) for d in dense])
                               ).long().to(DEVICE)
        out.append(net.embed(tok).cpu().numpy())         # max pooling, App. B.3
    return np.concatenate(out)


@torch.no_grad()
def emb_prism(model: str, g, m, s, batch=256, seed=0):
    """Returns every block separately, so a caller can probe the released
    readout, the blocks alone, or the undeleted representation."""
    from glucofm.config import Config
    from glucofm.model import GlucoFMEncoder
    from glucoprism.model import BlockedPool, PrismConfig

    t, cfg = read_release(model)
    fm = Config()
    for sec in ("model", "grid", "filt"):
        for k, v in cfg.get("fm_config", {}).get(sec, {}).items():
            if hasattr(getattr(fm, sec), k):
                setattr(getattr(fm, sec), k, v)
    pc = PrismConfig()
    for k, v in cfg.get("prism_config", {}).items():
        if hasattr(pc, k):
            setattr(pc, k, v)

    enc = GlucoFMEncoder(fm).to(DEVICE)
    m1, u1 = enc.load_state_dict(section(t, "online."), strict=False)
    pool = BlockedPool(fm.model.embed_dim, pc).to(DEVICE)
    m2, u2 = pool.load_state_dict(section(t, "pool."), strict=False)
    enc.eval(); pool.eval()
    print(f"  {model:<18} online.+pool.  missing={len(m1) + len(m2)} "
          f"unexpected={len(u1) + len(u2)}   "
          f"blocks zT={pc.d_trait} zS={pc.d_state} zA={pc.d_sensor}")

    torch.manual_seed(seed)      # zA is a stochastic channel under the VIB
    acc: dict[str, list] = {"zT": [], "zS": [], "zA": []}
    for i in range(0, len(g), batch):
        sl = slice(i, i + batch)
        z, *_ = enc(torch.from_numpy(g[sl]).float().to(DEVICE),
                    torch.from_numpy(m[sl]).float().to(DEVICE),
                    torch.from_numpy(s[sl]).long().to(DEVICE), patch_mask=None)
        r = pool(z)
        zT, zS, zA = ((r["zT"], r["zS"], r["zA"]) if isinstance(r, dict)
                      else (r[0], r[1], r[2]))
        for k, v in (("zT", zT), ("zS", zS), ("zA", zA)):
            acc[k].append(v.cpu().numpy())
    b = {k: np.concatenate(v) for k, v in acc.items()}
    b["zTzS"] = np.concatenate([b["zT"], b["zS"]], -1)   # the released readout
    b["full"] = np.concatenate([b["zT"], b["zS"], b["zA"]], -1)
    return b


EMBEDDERS = {
    "glucofm": emb_glucofm,
    "cgm-jepa": emb_cgm_jepa,
    "x-cgm-jepa": emb_cgm_jepa,
    "gluformer": emb_gluformer,
    "glucoprism-c": emb_prism,
    "glucoprism-e": emb_prism,
}
DEFAULT_MODELS = list(EMBEDDERS)


# ---------------------------------------------------------------- machinery

def subject_mean(X: np.ndarray, subj: np.ndarray):
    """(n_windows, d) -> (n_subjects, d): a subject's days are averaged.

    Labels are subject properties, so this is the level the paper reports at.
    """
    uniq = np.unique(subj)
    idx: dict[str, list[int]] = {s: [] for s in uniq}
    for i, s in enumerate(subj):
        idx[s].append(i)
    return np.stack([X[idx[s]].mean(0) for s in uniq]), uniq


def representations(models, g, m, s, blocks: bool):
    """name -> window-level matrix, one entry per representation to probe."""
    out: dict[str, np.ndarray] = {}
    out["raw"] = np.stack([densify(g[i], m[i]) for i in range(len(g))])
    for model in models:
        fn = EMBEDDERS.get(model)
        if fn is None:
            print(f"  {model:<18} unknown, skipped")
            continue
        try:
            e = fn(model, g, m, s)
        except Exception as ex:                              # noqa: BLE001
            print(f"  {model:<18} FAILED {type(ex).__name__}: {ex}")
            continue
        if isinstance(e, dict):
            keys = ("zTzS", "zT", "zS", "zA", "full") if blocks else ("zTzS",)
            for k in keys:
                out[f"{model}__{k}"] = e[k]
        else:
            out[model] = e
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Zero-shot evaluation of the released models on an "
                    "external cohort.")
    ap.add_argument("--shard", required=True, help="window .npz")
    ap.add_argument("--labels", required=True, help="subject-level label .csv")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="label columns to probe (default: all but `subject`)")
    ap.add_argument("--blocks", action="store_true",
                    help="also probe zT / zS / zA / full, not just the "
                         "released zT||zS readout")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--split-col", default=None,
                    help="shard column (e.g. a device or site label) to fit on "
                         "one value and score on the other")
    ap.add_argument("--out", default=str(OUTDIR / "external_results.csv"))
    a = ap.parse_args()

    d = np.load(a.shard, allow_pickle=True)
    g = np.nan_to_num(d["glucose"].astype(np.float32), nan=0.0)
    m = d["mask"].astype(np.float32)
    s = d["start_idx"].astype(np.int64)
    subj_w = d["subject"].astype(str)

    lab = pd.read_csv(a.labels)
    if "subject" not in lab.columns:
        raise SystemExit("--labels must have a `subject` column")
    lab = lab.set_index("subject")
    tasks = a.tasks or [c for c in lab.columns
                        if pd.api.types.is_numeric_dtype(lab[c])]

    subjects = np.unique(subj_w)
    lab = lab.reindex(subjects)
    print(f"device={DEVICE}  windows={len(g):,}  subjects={len(subjects):,}  "
          f"observed fraction={m.mean():.3f}")
    for t in tasks:
        y = lab[t]
        print(f"  {t:<18} labelled={int(y.notna().sum()):>6,}  "
              f"positives={int((y == 1).sum()):>6,}")
    print()

    reps = representations(a.models, g, m, s, a.blocks)
    if not reps:
        raise SystemExit("no representation could be built")

    rows = []
    for name, Xw in reps.items():
        X, idx = subject_mean(Xw, subj_w)
        for t in tasks:
            y = lab[t].reindex(idx).to_numpy(dtype=float)
            keep = np.isfinite(y)
            if keep.sum() < 50 or len(np.unique(y[keep])) < 2:
                continue
            r = glucofm_probe(X[keep], y[keep].astype(int), idx[keep], task=t,
                              n_splits=a.splits, n_iters=a.iters)
            for metric in METRICS:
                rows.append(dict(model=name, task=t, metric=metric,
                                 value=round(100 * getattr(r, metric), 3),
                                 sd=round(100 * getattr(r, f"{metric}_std"), 3),
                                 n=int(keep.sum()),
                                 n_pos=int((y[keep] == 1).sum()),
                                 dim=int(X.shape[1])))

    if a.split_col:
        rows += transfer(reps, subj_w, d, lab, tasks, a.split_col)

    res = pd.DataFrame(rows)
    _P(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)

    print(f"\n{'=' * 72}\nROC-AUC, mean over {a.iters} x {a.splits} "
          f"subject-grouped folds\n{'=' * 72}")
    piv = (res[res.metric == "roc_auc"]
           .pivot_table(index="model", columns="task", values="value"))
    piv["mean"] = piv.mean(axis=1)
    print(piv.sort_values("mean").round(1).to_string())
    print(f"\nwrote {a.out}")
    return 0


def transfer(reps, subj_w, shard, lab, tasks, col):
    """Fit on one value of a shard column and score on another.

    Subjects appearing under both values are dropped: with a per-window column
    a subject can otherwise land on both sides of the split, which turns a
    generalisation test into a memorisation one.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if col not in shard:
        print(f"[skip] --split-col {col!r} not in the shard")
        return []
    vals = shard[col].astype(str)
    per: dict[str, set] = {}
    for sb, v in zip(subj_w, vals):
        per.setdefault(sb, set()).add(v)
    pure = {sb: next(iter(v)) for sb, v in per.items() if len(v) == 1}
    uniq = sorted({v for v in pure.values()})
    if len(uniq) != 2:
        print(f"[skip] --split-col {col!r} has {len(uniq)} single-valued groups")
        return []
    src, dst = uniq
    print(f"\ntransfer on {col!r}: fit {src} -> score {dst}   "
          f"({len(pure):,} of {len(per):,} subjects are single-valued)")

    rows = []
    for name, Xw in reps.items():
        X, idx = subject_mean(Xw, subj_w)
        grp = np.array([pure.get(sb, "") for sb in idx])
        for t in tasks:
            y = lab[t].reindex(idx).to_numpy(dtype=float)
            ok = np.isfinite(y) & (grp != "")
            tr, te = ok & (grp == src), ok & (grp == dst)
            if tr.sum() < 50 or te.sum() < 50 or len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), y[tr].astype(int))
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            for metric, v in (("roc_auc", roc_auc_score(y[te].astype(int), p)),
                              ("pr_auc",
                               average_precision_score(y[te].astype(int), p))):
                rows.append(dict(model=f"{name}__transfer", task=t,
                                 metric=metric, value=round(100 * v, 3),
                                 sd=np.nan, n=int(te.sum()),
                                 n_pos=int((y[te] == 1).sum()),
                                 dim=int(X.shape[1])))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
