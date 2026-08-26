"""Convert our pretraining shards into the sibling repo's corpus format.

Their trainer (`external/glucoprism_v2_reference/glucoprism/pretrain.py`) expects a
single .npz with keys x, m, s, subj, cohort, day, device. Ours are per-cohort
shards with glucose, mask, start_idx, subject, dataset, device, segment,
start_time.

Building an adapter rather than reimplementing their model is deliberate: running
THEIR code on OUR corpus removes any chance that a reimplementation quietly
diverges. Six waves of one-factor-at-a-time additions failed to reproduce their
gain, so the remaining question is whether it is the component combination or the
corpus -- and that is only answerable with their exact training code.

    python scripts/build_v2_corpus.py --shard-suffix ""      -> our 9,918-window corpus
    python scripts/build_v2_corpus.py --shard-suffix _v2r    -> the v2-ratio corpus
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


import argparse
from pathlib import Path

import numpy as np
PROCESSED = ROOT / "data" / "processed"
DEFAULT_PT = ["replacebg", "stanford", "shanghait2dm", "colas", "bigideas", "d1namo"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_PT)
    ap.add_argument("--shard-suffix", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    keys = ("x", "m", "s", "subj", "cohort", "day", "device")
    acc = {k: [] for k in keys}
    for name in a.datasets:
        p = PROCESSED / f"{name}_pt{a.shard_suffix}.npz"
        if not p.exists():
            print(f"  [skip] {p.name}")
            continue
        d = np.load(p, allow_pickle=True)
        n = d["glucose"].shape[0]
        acc["x"].append(np.nan_to_num(d["glucose"].astype(np.float32), nan=0.0))
        acc["m"].append(d["mask"].astype(np.float32))
        acc["s"].append(d["start_idx"].astype(np.int64))
        acc["subj"].append(np.asarray([str(v) for v in d["subject"]]))
        acc["cohort"].append(np.asarray([name] * n))
        # `day` gates their V2 sampler: two windows count as different days only
        # if this differs. Our windows can overlap within a calendar date, so use
        # the date itself rather than the window index -- otherwise overlapping
        # windows of one day would be sampled as a "repeated day" pair.
        acc["day"].append(np.asarray([str(t)[:10] for t in d["start_time"]]))
        acc["device"].append(np.asarray([str(v) for v in d["device"]]))
        print(f"  {name:<14}{n:>7} windows  {len(set(str(v) for v in d['subject'])):>4} subjects")

    if not acc["x"]:
        raise SystemExit("no shards found")
    out = {k: np.concatenate(acc[k]) for k in keys}
    dest = Path(a.out) if a.out else PROCESSED / f"corpus_v2fmt{a.shard_suffix}.npz"
    np.savez_compressed(dest, **out)
    print(f"\n  -> {dest.name}: {len(out['x']):,} windows, "
          f"{len(set(out['subj'].tolist())):,} subjects, "
          f"{len(set(out['day'].tolist())):,} distinct dates, "
          f"{dest.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
