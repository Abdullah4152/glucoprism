"""How badly could the synthetic paired-sensor generator be misspecified?

The generator's parameters are fitted on 374 paired windows from 44 CGMacros
subjects, which is a modest sample. If those parameters move a lot when the
fitting set changes, the V1 view is fragile and every conclusion resting on it
inherits that fragility.

Two resamplings, both at SUBJECT level because windows from one person are not
independent:
  * leave-one-subject-out  -- 44 refits, worst case per parameter
  * bootstrap over subjects -- 2000 refits, a 95% interval
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


import json
from pathlib import Path

import numpy as np
import pandas as pd

A = ROOT / "experiments/artifacts"
d = pd.read_csv(A / "fd9_pair_measurements.csv")
subs = d.subject.unique()
PAR = {"bias_mgdl": "calibration offset (mg/dL)", "slope": "Deming slope",
       "mard_pct": "mean abs. rel. difference (%)", "corr": "correlation"}


def fit(frame: pd.DataFrame) -> dict:
    """The generator's fitted constants are subject-level means."""
    per = frame.groupby("subject")[list(PAR)].mean()
    return {k: float(per[k].mean()) for k in PAR}


full = fit(d)
loo = [fit(d[d.subject != s]) for s in subs]
rng = np.random.default_rng(0)
boot = [fit(d[d.subject.isin(rng.choice(subs, len(subs), replace=True))])
        for _ in range(2000)]

print(f"generator refits: {len(subs)} leave-one-subject-out, {len(boot)} bootstrap\n")
print(f"{'parameter':<34}{'fitted':>10}{'LOO range':>20}{'95% bootstrap':>24}")
out = {}
for k, lbl in PAR.items():
    lo_ = [x[k] for x in loo]
    bo = np.array([x[k] for x in boot])
    ci = np.percentile(bo, [2.5, 97.5])
    print(f"{lbl:<34}{full[k]:>10.2f}"
          f"{f'[{min(lo_):.2f}, {max(lo_):.2f}]':>20}"
          f"{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>24}")
    out[k] = dict(fitted=full[k], loo_min=min(lo_), loo_max=max(lo_),
                  ci_lo=float(ci[0]), ci_hi=float(ci[1]))

# The sign of the calibration offset is what the V1 view depends on: if some
# refits flipped it, the synthetic partner would sometimes read high.
neg = sum(x["bias_mgdl"] < 0 for x in boot)
print(f"\nbootstrap refits with a NEGATIVE calibration offset: {neg}/{len(boot)}")
persub = d.groupby("subject").bias_mgdl.mean()
print(f"subjects with a negative offset: {int((persub < 0).sum())}/{len(persub)}")
out["sign_stability"] = dict(boot_negative=neg, boot_n=len(boot),
                             subjects_negative=int((persub < 0).sum()),
                             subjects_n=len(persub))
(A / "rev_generator_robustness.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {A / 'rev_generator_robustness.json'}")
