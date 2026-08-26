"""Build the GlucoPRISM public-only corpus: raw downloads -> window shards + splits.

Two products, matching the proposal's Table 1 roles:

  PT (pretraining)  overlapping 24 h windows sampled per GlucoFM App. A.2
                    (coverage ratio ~ U[0.2, 0.8] per continuous segment)
  DS (downstream)   non-overlapping 24 h windows, per GlucoFM App. A.3

Leakage discipline (proposal Sec. 5):
  * strict subject separation between PT and DS;
  * CGMacros is DS-only -- its real Dexcom/Libre pairs are the held-out V1
    validation set and never enter pretraining;
  * a subject appearing in two cohorts is assigned to exactly one.

    python scripts/build_corpus.py --all
    python scripts/build_corpus.py --datasets stanford hall --role ds
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


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "core"))

from cgmkit.data import harmonize, labels as label_mod  # noqa: E402
from cgmkit.data.windows import (iter_windows, sample_overlapping_windows,  # noqa: E402
                                     windows_to_arrays)

OUT = ROOT / "data" / "processed"

# Role assignment from proposal Table 1. Stanford and ShanghaiT2DM are split by
# subject: those with complete clinical profiles go to DS, the rest to PT.
ROLES = {
    "stanford":     {"pt", "ds"},
    "hall":         {"ds"},
    "cgmacros":     {"ds"},          # V1 validation -- never pretrained on
    "shanghait2dm": {"pt", "ds"},
    "colas":        {"pt"},
    "bigideas":     {"pt"},
    "replacebg":    {"pt"},          # 226 T1D adults, ~1.23 M hours -- the scale lever
    "d1namo":       {"pt"},          # Medtronic iPro2; CGM only in the diabetes arm
}

# REPLACE-BG has a median of 254 days per subject and would otherwise contribute
# ~30x more pretraining windows than every other cohort combined, turning the
# corpus into "REPLACE-BG plus noise". Cap per subject so subject diversity, not
# wear duration, drives the mixture (GlucoFM Fig. 3 right panel makes the same
# point: subject diversity is more limiting than dense per-subject recording).
MAX_PT_WINDOWS_PER_SUBJECT = {"replacebg": 40}

# MEASURED: at the cap above, REPLACE-BG still supplies 9,035 of 9,918 pretraining
# windows -- 91 % of the corpus. REPLACE-BG is a type-1-diabetes cohort while every
# downstream cohort is type-2 / prediabetes / normoglycemic, so the comment above
# is not satisfied by 40: subject diversity does NOT drive the mixture, wear
# duration still does. `--pt-cap N` applies a uniform cap of N windows per subject
# to EVERY cohort, which is the knob that actually rebalances it. Paired with
# `--out-suffix` so a rebalanced corpus can live beside the original rather than
# overwrite it, and the two can be compared.

# The sibling repo's `corpus_balanced` window counts (5,727 windows / 513 subjects
# over six cohorts), reproduced exactly so a GlucoPRISM-v2 comparison is not
# confounded by the pretraining mixture. Our PT/DS SUBJECT assignment is kept --
# changing it would invalidate the frozen evaluation folds -- so only the window
# counts are matched, not their subject split.
V2RATIO_TARGET = {
    "replacebg": 2712,      # 47.35 %
    "shanghait2dm": 1165,   # 20.34 %
    "stanford": 1015,       # 17.72 %
    "colas": 530,           #  9.25 %
    "bigideas": 245,        #  4.28 %
    "d1namo": 60,           #  1.05 %
}

BINNING = {"shanghait2dm": "nearest"}   # Libre timestamps drift off the 5-min grid


def label_table(name: str) -> pd.DataFrame | None:
    fn = label_mod.LABEL_READERS.get(name)
    if fn is None:
        return None
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] labels for {name} unavailable: {type(e).__name__}: {e}")
        return None


def labelled_subjects(name: str) -> set[str]:
    """Subjects with a complete clinical profile for this cohort's task set."""
    tbl = label_table(name)
    tasks = label_mod.TASK_MATRIX.get(name, [])
    if tbl is None or not tasks:
        return set()
    cols = [c for c in tasks if c in tbl.columns]
    if not cols:
        return set()
    ok = tbl[cols].notna().all(axis=1)
    return set(tbl.loc[ok, "subject"].astype(str))


def build(name: str, roles: set[str], *, pt_cap: int | None = None,
          pt_target: int | None = None, pt_density: float = 1.0,
          day_overlap: float | None = None, pt_match_suffix: str | None = None,
          pt_subject_frac: float | None = None,
          out_suffix: str = "", seed: int = 0, min_coverage: float = 0.5,
          verbose: bool = True) -> dict:
    reader = harmonize.READERS[name]
    print(f"\n=== {name}: reading raw ...")
    df = reader()
    print(f"    {len(df):,} readings, {df['subject'].nunique()} subjects, "
          f"devices={sorted(df['device'].unique())}")

    ds_subjects = labelled_subjects(name)
    all_subjects = set(df["subject"].astype(str))
    # The PT/DS assignment is a property of the DATASET, not of which role this
    # invocation happens to be writing. Deriving it from `roles` meant that
    # `--role pt` on a dual-role cohort assigned EVERY subject -- including the
    # downstream-labelled ones -- to pretraining, leaking the evaluation
    # subjects into the corpus. Stanford came out with 56 pretraining subjects
    # instead of 27, ShanghaiT2DM with 109 instead of 40.
    full = ROLES[name]
    if full == {"pt", "ds"}:
        assign = {s: ("ds" if s in ds_subjects else "pt") for s in all_subjects}
    else:
        assign = {s: next(iter(full)) for s in all_subjects}

    n_ds = sum(v == "ds" for v in assign.values())
    print(f"    role split: {n_ds} DS / {len(assign) - n_ds} PT")

    rng = np.random.default_rng(seed)
    binning = BINNING.get(name, "floor")
    out: dict[str, list] = {"pt": [], "ds": []}

    # A uniform --pt-cap overrides the per-dataset table for every cohort.
    cap = pt_cap if pt_cap else MAX_PT_WINDOWS_PER_SUBJECT.get(name)
    for (subj, dev), g in df.groupby(["subject", "device"], sort=True):
        role = assign[str(subj)]
        rate = float(g["sampling_min"].iloc[0])
        if role == "pt":
            if day_overlap is not None:
                # FD-7 / Reading B of App. A.2: consecutive 24 h windows overlap
                # by a FIXED fraction r, so the stride is (1 - r) * 24 h and
                # r = 0 is a plain non-overlapping tiling. Fixed rather than
                # drawn per segment, so overlap is a controlled variable instead
                # of noise -- see discussion.md 6.1/6.2.
                w = list(iter_windows(g, dataset=name, subject=str(subj), device=str(dev),
                                      min_coverage=min_coverage, binning=binning,
                                      sampling_min=rate,
                                      stride_hours=24.0 * (1.0 - float(day_overlap))))
            else:
                # Reading A (legacy): windows jointly cover r x the segment.
                # `pt_density` scales App. A.2's stated [20 %, 80 %] range.
                w = sample_overlapping_windows(g, dataset=name, subject=str(subj), device=str(dev),
                                               rng=rng, min_coverage=min_coverage, binning=binning,
                                               sampling_min=rate,
                                               coverage_lo=0.2 * pt_density,
                                               coverage_hi=0.8 * pt_density)
            if cap and len(w) > cap:
                keep = rng.choice(len(w), size=cap, replace=False)
                w = [w[i] for i in sorted(keep)]
        else:
            w = list(iter_windows(g, dataset=name, subject=str(subj), device=str(dev),
                                  min_coverage=min_coverage, binning=binning,
                                  sampling_min=rate))
        out[role].extend(w)

    # Trim the pretraining side to an exact target so a corpus can be built to a
    # prescribed cohort MIXTURE rather than to whatever each cohort happens to
    # yield. Subsampling is uniform over windows, which preserves the per-subject
    # distribution; it is done after the per-subject cap so no single subject can
    # dominate the trimmed set.
    # FD-5: keep only a fraction of this cohort's pretraining SUBJECTS (all of
    # their windows). Subject-wise rather than window-wise because an earlier arm
    # that capped windows per subject while keeping every subject landed back on
    # the full corpus's score -- the window axis is already known to be flat, and
    # GlucoFM's own few-shot section reports subject diversity as the binding
    # constraint.
    if pt_subject_frac is not None and pt_subject_frac < 1.0:
        pts = sorted({w.subject for w in out["pt"]})
        keep_n = max(1, int(round(len(pts) * float(pt_subject_frac))))
        keep = set(np.random.default_rng(seed).permutation(pts)[:keep_n].tolist())
        before = len(out["pt"])
        out["pt"] = [w for w in out["pt"] if w.subject in keep]
        print(f"    subject fraction {pt_subject_frac:.0%}: "
              f"{keep_n}/{len(pts)} subjects, {len(out['pt']):,}/{before:,} windows")

    # Size-matching: trim this cohort's pretraining windows to the count another
    # arm produced. Without it, a day-overlap sweep confounds two things -- more
    # windows, and windows at more diverse start times -- and the second is the
    # one the experiment is about. Matching every arm to the r = 0 tiling makes
    # overlap the only moving part, exactly as the LOCO size control does for
    # cohort composition.
    if pt_match_suffix is not None:
        ref = OUT / f"{name}_pt{pt_match_suffix}.npz"
        if ref.exists():
            with np.load(ref, allow_pickle=True) as z:
                n_ref = len(z["glucose"])
            if len(out["pt"]) > n_ref:
                keep = rng.choice(len(out["pt"]), size=n_ref, replace=False)
                out["pt"] = [out["pt"][i] for i in sorted(keep)]
                print(f"    matched to {ref.name}: {n_ref:,} windows")
            elif len(out["pt"]) < n_ref:
                print(f"    [warn] {name}: {len(out['pt']):,} pt windows < "
                      f"match target {n_ref:,} -- cannot size-match up")
        else:
            print(f"    [warn] match reference {ref.name} not found")

    if pt_target and len(out["pt"]) > pt_target:
        keep = rng.choice(len(out["pt"]), size=pt_target, replace=False)
        out["pt"] = [out["pt"][i] for i in sorted(keep)]
    elif pt_target and len(out["pt"]) < pt_target:
        print(f"    [warn] {name}: only {len(out['pt'])} pt windows, target was "
              f"{pt_target} -- raise --pt-density")

    stats = {"dataset": name, "readings": int(len(df)),
             "subjects": int(df["subject"].nunique()),
             "devices": sorted(df["device"].unique()),
             "hours": round(float((df.groupby("device").size()
                                   * df.groupby("device")["sampling_min"].first()).sum() / 60.0), 1)}

    OUT.mkdir(parents=True, exist_ok=True)
    for role, wins in out.items():
        if not wins or role not in roles:
            # `roles` is what this invocation was asked to WRITE; `full` above is
            # what the dataset HAS. Skipping here is what keeps `--role pt` from
            # overwriting the frozen downstream shards every evaluation depends on.
            continue
        arrs = windows_to_arrays(wins)
        # Downstream shards are never resampled, so a suffix would only fragment
        # them; only the pretraining side varies with --pt-cap.
        sfx = out_suffix if role == "pt" else ""
        path = OUT / f"{name}_{role}{sfx}.npz"
        np.savez_compressed(path, **arrs)
        cov = float(arrs["mask"].mean())
        stats[f"{role}_windows"] = len(wins)
        stats[f"{role}_subjects"] = int(len(set(arrs["subject"])))
        stats[f"{role}_mean_coverage"] = round(cov, 4)
        print(f"    -> {path.name}: {len(wins):,} windows, "
              f"{stats[f'{role}_subjects']} subjects, mean mask coverage {cov:.3f} "
              f"({path.stat().st_size/1e6:.1f} MB)")

    tbl = label_table(name)
    if tbl is not None:
        lp = OUT / f"{name}_labels.csv"
        tbl.to_csv(lp, index=False)
        stats["label_file"] = lp.name
        tasks = [t for t in label_mod.TASK_MATRIX.get(name, []) if t in tbl.columns]
        stats["label_counts"] = {t: {str(k): int(v) for k, v in
                                     tbl[t].value_counts(dropna=True).items()} for t in tasks}
        print(f"    -> {lp.name}: {len(tbl)} subjects, tasks={tasks}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--role", choices=["pt", "ds", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--pt-cap", type=int, default=None,
                    help="uniform cap of N pretraining windows per subject applied "
                         "to EVERY cohort; overrides MAX_PT_WINDOWS_PER_SUBJECT")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the pretraining shards, e.g. _bal, so a "
                         "rebalanced corpus sits beside the original")
    ap.add_argument("--pt-target", type=int, default=None,
                    help="exact number of pretraining windows to keep for EACH "
                         "dataset named in --datasets (use --preset for a mixture)")
    ap.add_argument("--pt-density", type=float, default=1.0,
                    help="scale App. A.2's [20%%, 80%%] coverage range; >1 oversamples "
                         "so a small cohort can reach its target")
    ap.add_argument("--preset", choices=["v2ratio"], default=None,
                    help="build to a prescribed cohort mixture; v2ratio reproduces "
                         "the sibling repo's corpus_balanced window counts")
    ap.add_argument("--pt-subject-frac", type=float, default=None,
                    help="FD-5: keep only this fraction of the cohort's pretraining SUBJECTS")
    ap.add_argument("--pt-match-suffix", default=None,
                    help="trim each cohort's pretraining windows to the count in "
                         "{cohort}_pt{SUFFIX}.npz, so two arms differ only in HOW "
                         "the windows were drawn and not in how many")
    ap.add_argument("--day-overlap", type=float, default=None,
                    help="FD-7: fixed fraction by which consecutive 24 h pretraining "
                         "windows overlap (stride = (1-r)*24 h). 0.0 is a plain "
                         "non-overlapping tiling. Overrides --pt-density.")
    a = ap.parse_args()

    names = a.datasets or (list(ROLES) if a.all else [])
    if not names:
        ap.error("pass --all or --datasets ...")

    # Merge into any previous report so building cohorts in separate invocations
    # still leaves one complete corpus summary.
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "corpus_report.json"
    prev = json.loads(p.read_text()) if p.exists() else []
    merged = {d["dataset"]: d for d in prev if isinstance(d, dict) and "dataset" in d}

    for n in names:
        roles = ROLES[n] if a.role == "both" else ({a.role} & ROLES[n])
        if not roles:
            print(f"\n=== {n}: skipped (no {a.role} role)")
            continue
        try:
            tgt = (V2RATIO_TARGET.get(n) if a.preset == "v2ratio" else a.pt_target)
            merged[n] = build(n, roles, pt_cap=a.pt_cap, pt_target=tgt,
                              pt_density=a.pt_density, day_overlap=a.day_overlap,
                              pt_match_suffix=a.pt_match_suffix,
                              pt_subject_frac=a.pt_subject_frac,
                              out_suffix=a.out_suffix,
                              seed=a.seed, min_coverage=a.min_coverage)
        except Exception as e:  # noqa: BLE001
            print(f"    [FAILED] {n}: {type(e).__name__}: {e}")
            merged[n] = {"dataset": n, "error": f"{type(e).__name__}: {e}"}

    report = [merged[k] for k in sorted(merged)]
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== corpus summary ===")
    tot_pt = sum(r.get("pt_windows", 0) for r in report)
    tot_ds = sum(r.get("ds_windows", 0) for r in report)
    print(f"  pretraining windows : {tot_pt:,}  (~{tot_pt*24:,} h of 24 h windows)")
    print(f"  downstream windows  : {tot_ds:,}")
    print(f"  raw monitoring hours: {sum(r.get('hours', 0) for r in report):,.0f}")
    print(f"  report -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



