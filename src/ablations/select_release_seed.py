"""Which seed of each released model ships, and does the rule survive scrutiny?

Only one checkpoint per model is released, so the choice needs a stated rule
rather than an eye. The rule, fixed before the transfer axis was looked at:

    ship the seed with the best mean over the 14 within-cohort cells at
    window level -- the paper's primary evaluation protocol.

This applies that rule and then prints the two axes the rule does NOT use, so a
reader can see whether the alternatives would have chosen differently. For one
of the two models they do, and saying so is the point: choosing the axis after
seeing the numbers is how release selection turns into cherry-picking.

    python src/ablations/select_release_seed.py
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


from pathlib import Path

import pandas as pd

A = ROOT / "experiments/artifacts"

MODELS = [("GlucoPRISM-C", "C-v2-vib01", "C-v2-vib01:zTzS"),
          ("GlucoPRISM-E", "E-v2-vib-simbias", "E-v2-vib-simbias:zTzS")]
TRANSFER_FILES = ("fd3_v2final.csv", "fd3_bd.csv", "fd3_cseeds345.csv")


def load_scores() -> pd.DataFrame:
    d = pd.read_csv(A / "v2_final_scores.csv")
    d["seed"] = d.run.str.extract(r"-s(\d)$")[0].astype(int)
    d["arm"] = d.run.str.replace(r"-s\d$", "", regex=True)
    return d


def load_transfer() -> pd.DataFrame:
    fr = [pd.read_csv(A / f) for f in TRANSFER_FILES if (A / f).exists()]
    if not fr:
        return pd.DataFrame(columns=["arm", "seed", "auc"])
    t = pd.concat(fr, ignore_index=True).drop_duplicates(
        subset=["run", "src", "tgt", "task"])
    t["seed"] = t.run.str.extract(r"-s(\d)")[0].astype(int)
    t["arm"] = t.run.str.replace(r"-s\d(:|$)", r"\1", regex=True)
    return t


def main() -> None:
    v2, tr = load_scores(), load_transfer()
    for name, arm, tarm in MODELS:
        print(f"\n=== {name} ===")
        w = v2[(v2.arm == arm) & (v2.block == "zTzS") & (v2.level == "window")]
        s = v2[(v2.arm == arm) & (v2.block == "zTzS") & (v2.level == "subject")]
        if not len(w):
            print(f"  no scores for {arm}; run probe_frozen_folds.py first")
            continue
        print(f"{'seed':>5}{'window':>10}{'subject':>10}{'transfer':>11}")
        win, sub, trm = {}, {}, {}
        for sd in sorted(w.seed.unique()):
            win[sd] = w[w.seed == sd].groupby(["cohort", "task"]).auc.mean().mean()
            ss = s[s.seed == sd]
            sub[sd] = (ss.groupby(["cohort", "task"]).auc.mean().mean()
                       if len(ss) else float("nan"))
            tt = tr[(tr.arm == tarm) & (tr.seed == sd)]
            trm[sd] = tt.auc.mean() if len(tt) else float("nan")
            print(f"{sd:>5}{win[sd]:>10.2f}{sub[sd]:>10.2f}{trm[sd]:>11.2f}")

        chosen = max(win, key=win.get)
        print(f"\n  RULE (best window mean) -> seed {chosen}")
        for axis, d in (("subject", sub), ("transfer", trm)):
            avail = {k: v for k, v in d.items() if v == v}
            if not avail:
                continue
            alt = max(avail, key=avail.get)
            verdict = "agrees" if alt == chosen else f"would pick seed {alt}"
            print(f"    best {axis:<9} {verdict}")


if __name__ == "__main__":
    main()
