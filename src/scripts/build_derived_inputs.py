"""Derive the summary tables `make_all_tables.py` consumes.

Six of its inputs are aggregations over the per-run scores rather than direct
outputs of the probing scripts: the two parity tables, the sensor-block capacity
summary, the partial-deletion probe, the seed-variability table, and the three
Holm-corrected significance tables.

Everything is derived from measured scores, with one exception: the `_paper`
columns of `repro_vs_published.csv` are the values each baseline's own
publication reports, read from `data/published_percell.csv`.

    python src/scripts/build_derived_inputs.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(os.environ.get("GLUCOPRISM_ROOT", Path(__file__).resolve().parents[1]))
A = Path(os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))
PUBLISHED = ROOT / "data" / "published_percell.csv"

GPC = "GlucoPRISM-v2 + zA bottleneck (weak) [zA dropped]"
REPRO_MODELS = ["glucofm", "cgm_jepa", "x_cgm_jepa", "gluformer_tiny",
                "raw", "mask_only"]


def _eval_frames(model: str) -> pd.DataFrame | None:
    """Seed-averaged per-cell scores from evaluate_models.py."""
    owner = "glucofm" if model in ("raw", "mask_only") else model
    rows = []
    for s in (0, 1, 2):
        p = A / f"eval_{owner}-s{s}" / "linear_probe.csv"
        if p.exists():
            rows.append(pd.read_csv(p))
    if not rows:
        return None
    d = pd.concat(rows, ignore_index=True)
    d = d[d.model == model]
    if d.empty:
        return None
    keep = ["PR", "PR_std", "AUC", "AUC_std", "F1", "F1_std",
            "folds", "subjects", "windows"]
    g = d.groupby(["dataset", "task"], as_index=False)[keep].mean()
    g["model"] = model
    return g


def repro_frozen_probe() -> None:
    frames = [f for f in (_eval_frames(m) for m in REPRO_MODELS) if f is not None]
    d = pd.concat(frames, ignore_index=True)
    d = d[["task", "PR", "PR_std", "AUC", "AUC_std", "F1", "F1_std",
           "folds", "subjects", "windows", "dataset", "model"]].round(1)
    d.to_csv(A / "repro_frozen_probe.csv", index=False)
    print(f"  repro_frozen_probe.csv        {len(d)} rows, "
          f"{d.model.nunique()} models, {d.groupby(['dataset','task']).ngroups} cells")


def repro_vs_published() -> None:
    """Our per-cell scores against each model's own published value.

    The `_paper` columns are the values those publications report and are read
    from `data/published_percell.csv`; nothing here can recompute someone
    else's number. The `_ours` columns and the deltas come from our probes.
    """
    ref = pd.read_csv(PUBLISHED)
    ours = pd.read_csv(A / "repro_frozen_probe.csv")
    m = ref[["dataset", "task", "model", "PR_paper", "AUC_paper"]].merge(
        ours[["dataset", "task", "model", "PR", "AUC"]],
        on=["dataset", "task", "model"], how="left")
    m = m.rename(columns={"PR": "PR_ours", "AUC": "AUC_ours"})
    m["dPR"] = (m.PR_ours - m.PR_paper).round(1)
    m["dAUC"] = (m.AUC_ours - m.AUC_paper).round(1)
    m = m[["dataset", "task", "model", "PR_ours", "PR_paper", "dPR",
           "AUC_ours", "AUC_paper", "dAUC"]]
    missing = int(m.PR_ours.isna().sum())
    m.to_csv(A / "repro_vs_published.csv", index=False)
    print(f"  repro_vs_published.csv        {len(m)} rows"
          + (f"  ({missing} without a reproduced value)" if missing else ""))


def capacity_summary() -> None:
    v2 = pd.read_csv(A / "v2_final_scores.csv")
    v2["seed"] = v2.run.str.extract(r"-s(\d+)$").astype(int)
    v2["arm"] = v2.run.str.replace(r"-s\d+$", "", regex=True)
    # (our arm, published arm label, d_sensor, beta)
    ARMS = [("C-dA8", "K-dA8", "8", 0.10),
            ("C-v2-vib01", "C-v2-vib01", "16 (released)", 0.10),
            ("C-dA32", "K-dA32", "32", 0.10),
            ("C-beta0p03", "K-beta003", "16", 0.03),
            ("C-beta0p3", "K-beta03", "16", 0.30),
            ("B-v2-vib1", "K-beta10", "16", 1.00)]
    rows = []
    for arm, label, d_sensor, beta in ARMS:
        s = v2[(v2.arm == arm) & (v2.seed <= 1)]        # two seeds, as published
        if s.empty:
            print(f"  [warn] capacity arm {arm} absent")
            continue
        def mean(level, block):
            return s[(s.level == level) & (s.block == block)].auc.mean()
        full, drop = mean("window", "full"), mean("window", "zTzS")
        sf, sd = mean("subject", "full"), mean("subject", "zTzS")
        rows.append(dict(arm=label, d_sensor=d_sensor, beta=beta,
                         full=full, drop=drop, gain=drop - full,
                         subj_gain=sd - sf))
    pd.DataFrame(rows).to_csv(A / "rev_capacity_summary.csv", index=False)
    print(f"  rev_capacity_summary.csv      {len(rows)} arms")


def partial_within() -> None:
    src = A / "rev_partial_within_raw.csv"
    if not src.exists():
        print("  [warn] rev_partial_within_raw.csv absent")
        return
    d = pd.read_csv(src)
    d = d[["run", "block", "level", "cohort", "task", "pr", "auc", "f1"]]
    d.to_csv(A / "rev_partial_within.csv", index=False)
    print(f"  rev_partial_within.csv        {len(d)} rows, "
          f"blocks {sorted(d.block.unique())}")


def seed_variability() -> None:
    v2 = pd.read_csv(A / "v2_final_scores.csv")
    v2["seed"] = v2.run.str.extract(r"-s(\d+)$").astype(int)
    v2["arm"] = v2.run.str.replace(r"-s\d+$", "", regex=True)
    fd7 = pd.read_csv(A / "fd7_scores.csv")
    fd7 = fd7[fd7.run.str.startswith("W3u-ov40")].copy()
    fd7["seed"] = fd7.run.str.extract(r"-s(\d+)$").fillna("0").astype(int)

    SPEC = [
        ("GlucoPRISM-C", "C-v2-vib01", "zTzS", None),
        ("GlucoPRISM-C [seed-matched]", "C-v2-vib01", "zTzS", 2),
        ("GlucoPRISM-C [full readout]", "C-v2-vib01", "full", None),
        ("GlucoPRISM-E", "E-v2-vib-simbias", "zTzS", None),
        ("GlucoPRISM-E [full readout]", "E-v2-vib-simbias", "full", None),
        ("No factorization (A)", "A-v2-base", "zTzS", None),
        ("Bottleneck only (B)", "B-v2-vib1", "zTzS", None),
        ("Objectives only (D)", "D-v2-simbias", "zTzS", None),
        ("REPLACE-BG 50%", "C-rbg50", "zTzS", None),
        ("REPLACE-BG 70%", "C-rbg70", "zTzS", None),
    ]
    rows = []
    for label, arm, block, cap in SPEC:
        d = v2[(v2.arm == arm) & (v2.block == block)]
        if cap is not None:
            d = d[d.seed <= cap]
        if d.empty:
            print(f"  [warn] seed_variability: {label} ({arm}) absent")
            continue
        for level in ("window", "subject"):
            s = d[d.level == level]
            if s.empty:
                continue
            per_seed = s.groupby("seed")[["pr", "auc", "f1"]].mean()
            per_cell = s.groupby(["cohort", "task"])[["pr", "auc", "f1"]].std(ddof=1)
            rows.append(dict(
                model=label, level=level, n_seeds=s.seed.nunique(),
                seeds=",".join(str(x) for x in sorted(s.seed.unique())),
                pr_mean=per_seed.pr.mean(), pr_sd_taskavg=per_seed.pr.std(ddof=1),
                auc_mean=per_seed.auc.mean(), auc_sd_taskavg=per_seed.auc.std(ddof=1),
                f1_mean=per_seed.f1.mean(), f1_sd_taskavg=per_seed.f1.std(ddof=1),
                pr_sd_percell_mean=per_cell.pr.mean(), pr_sd_percell_max=per_cell.pr.max(),
                auc_sd_percell_mean=per_cell.auc.mean(), auc_sd_percell_max=per_cell.auc.max(),
                f1_sd_percell_mean=per_cell.f1.mean(), f1_sd_percell_max=per_cell.f1.max()))

    # GlucoFM, from the fd7 runs
    for level in ("window", "subject"):
        s = fd7[fd7.level == level]
        if s.empty:
            continue
        per_seed = s.groupby("seed")[["pr", "auc", "f1"]].mean()
        per_cell = s.groupby(["cohort", "task"])[["pr", "auc", "f1"]].std(ddof=1)
        rows.append(dict(
            model="GlucoFM (ours)", level=level, n_seeds=s.seed.nunique(),
            seeds=",".join(str(x) for x in sorted(s.seed.unique())),
            pr_mean=per_seed.pr.mean(), pr_sd_taskavg=per_seed.pr.std(ddof=1),
            auc_mean=per_seed.auc.mean(), auc_sd_taskavg=per_seed.auc.std(ddof=1),
            f1_mean=per_seed.f1.mean(), f1_sd_taskavg=per_seed.f1.std(ddof=1),
            pr_sd_percell_mean=per_cell.pr.mean(), pr_sd_percell_max=per_cell.pr.max(),
            auc_sd_percell_mean=per_cell.auc.mean(), auc_sd_percell_max=per_cell.auc.max(),
            f1_sd_percell_mean=per_cell.f1.mean(), f1_sd_percell_max=per_cell.f1.max()))

    d = pd.DataFrame(rows).round(3)
    d.to_csv(A / "seed_variability.csv", index=False)
    print(f"  seed_variability.csv          {len(d)} rows, "
          f"{d.model.nunique()} models")


def cliffs(x, y):
    gt = sum(a > b for a in x for b in y)
    lt = sum(a < b for a in x for b in y)
    return (gt - lt) / (len(x) * len(y))


def significance_k11(level: str = "window", metric: str = "auc") -> None:
    """Holm over the pre-declared family of 11, seed-matched at three seeds.

    Same procedure as `significance_within_cohort.py`, but on a seed-matched
    table: the paper's Table 11 compares every arm at three seeds, while the
    released script averages over whatever seeds exist (six, for GlucoPRISM-C).

    Run for all three axes the paper reports: window ROC, window PR and subject
    ROC. CGM-JEPA and X-CGM-JEPA have no subject-level scores (evaluate_models.py
    is window-only), which is why the paper prints "---" for them there.
    """
    df = pd.read_csv(A / "final_table_long.csv")
    v2 = pd.read_csv(A / "v2_final_scores.csv")
    v2["seed"] = v2.run.str.extract(r"-s(\d+)$").astype(int)
    v2["arm"] = v2.run.str.replace(r"-s\d+$", "", regex=True)
    m3 = (v2[(v2.arm == "C-v2-vib01") & (v2.block == "zTzS") & (v2.seed <= 2)]
          .groupby(["level", "cohort", "task"], as_index=False)[["pr", "auc", "f1"]]
          .mean())
    m3["run"] = GPC
    df = pd.concat([df[df.run != GPC], m3], ignore_index=True)

    W = df[df.level == level]
    piv = W.pivot_table(index="run", columns=["cohort", "task"], values=metric)
    ref = piv.loc["GlucoFM (ours)"]
    keep = [m for m in piv.index
            if m == "GlucoFM (ours)" or "GlucoPRISM-v2" not in m
            or m in (GPC, "GlucoPRISM-v2 + bottleneck + measured sensor [zA dropped]")]
    keep = [m for m in keep if m not in ("GlucoPRISM proposal", "GluFormer-tiny")]
    rows = []
    for m in keep:
        if m == "GlucoFM (ours)":
            continue
        v = piv.loc[m]
        both = v.notna() & ref.notna()
        x, y = v[both].to_numpy(float), ref[both].to_numpy(float)
        d = x - y
        try:
            _, p = wilcoxon(x, y)
        except ValueError:
            p = 1.0
        rows.append(dict(model=m, n=int(both.sum()), mean_delta=d.mean(),
                         median_delta=float(np.median(d)),
                         sd_delta_cells=d.std(ddof=1),
                         se_delta_cells=d.std(ddof=1) / np.sqrt(len(d)),
                         wins=int((d > 0).sum()), p_raw=p, cliffs=cliffs(x, y)))
    r = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    # The family is always corrected over the DECLARED 11, even where an arm has
    # no scores on this axis. Shrinking the divisor to the number of tests that
    # happened to run would make the correction weaker exactly where the data is
    # thinner.
    k = 11
    if len(r) > k:
        raise SystemExit(f"family is {len(r)}, larger than the declared {k}: "
                         f"{sorted(r.model)}")
    r["p_holm"] = np.maximum.accumulate(
        [min(1.0, (k - i) * p) for i, p in enumerate(r.p_raw)])
    r["sig"] = pd.Series(np.where(r.p_holm < 0.05, "*", None), dtype="object")
    name = f"significance_{level}_{metric}_k11.csv"
    r.to_csv(A / name, index=False)
    print(f"  {name:<32} {len(r)}/{k} tested; "
          f"survivors: {list(r[r.p_holm < 0.05].model) or 'none'}")


if __name__ == "__main__":
    print("building generator inputs the release pipeline does not emit:")
    repro_frozen_probe()
    repro_vs_published()
    capacity_summary()
    partial_within()
    seed_variability()
    significance_k11("window", "auc")
    significance_k11("window", "pr")
    significance_k11("subject", "auc")
