"""Does the day-overlap dial actually vary start-time diversity?

The 40-window/subject cap on REPLACE-BG binds in every arm, so all arms hold
~9,040 REPLACE-BG windows regardless of overlap. That makes the WINDOW COUNT
nearly constant -- but the experiment is about seeing the same days at different
times of day, not about count. This checks whether the circadian start indices
actually spread out as overlap increases. If they do not, the sweep is measuring
nothing and should not be run.
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


from pathlib import Path

import numpy as np

P = (ROOT / "data/processed")
ARMS = ["_ov0", "_ov20m", "_ov40m", "_ov20", "_ov40"]

print(f"{'arm':<8}{'cohort':<14}{'windows':>8}{'uniq starts':>13}"
      f"{'per-subj uniq':>15}{'start sd (h)':>14}")
print("-" * 74)
for tag in ARMS:
    for f in sorted(P.glob(f"*_pt{tag}.npz")):
        coh = f.stem.split("_pt")[0]
        if coh not in ("replacebg", "stanford"):
            continue
        with np.load(f, allow_pickle=True) as z:
            si = np.asarray(z["start_idx"], dtype=float)
            subj = np.asarray([str(s) for s in z["subject"]])
        per = [len(np.unique(si[subj == s])) for s in np.unique(subj)]
        # circular sd of time-of-day, in hours
        ang = 2 * np.pi * si / 288.0
        R = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
        circ_sd_h = float(np.sqrt(max(-2 * np.log(max(R, 1e-9)), 0.0)) * 24 / (2 * np.pi))
        print(f"{tag:<8}{coh:<14}{len(si):>8,}{len(np.unique(si)):>13}"
              f"{np.mean(per):>15.1f}{circ_sd_h:>14.2f}")
    print()
