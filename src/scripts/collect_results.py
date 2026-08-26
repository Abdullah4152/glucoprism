"""Merge every model's linear-probe output into one GlucoFM-Table-3-style comparison.

Reads `artifacts/kaggle/<model>/eval/linear_probe.csv` (and any local
`artifacts/eval/linear_probe.csv`), pivots to the paper's layout, and compares
against the published numbers so divergences are visible per cell rather than only
in the average.

    python scripts/collect_results.py
    python scripts/collect_results.py --roots artifacts/kaggle artifacts/eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# GlucoFM Table 3, PR / ROC-AUC, for the cells we can build from public data.
PUBLISHED = {
    ("cgmacros", "diabetes_3class"): {"glucofm": (65.9, 78.7), "cgm_jepa": (63.0, 75.9),
                                      "x_cgm_jepa": (63.6, 76.6), "gluformer_tiny": (59.4, 73.9)},
    ("cgmacros", "ir"): {"glucofm": (91.9, 81.2), "cgm_jepa": (86.2, 73.8),
                         "x_cgm_jepa": (86.6, 73.6), "gluformer_tiny": (86.1, 72.5)},
    ("cgmacros", "hyperlipidemia"): {"glucofm": (36.1, 54.7), "cgm_jepa": (28.7, 47.6),
                                     "x_cgm_jepa": (29.8, 48.0), "gluformer_tiny": (28.2, 47.9)},
    ("cgmacros", "obesity"): {"glucofm": (64.9, 62.6), "cgm_jepa": (55.5, 53.8),
                              "x_cgm_jepa": (55.3, 53.2), "gluformer_tiny": (60.2, 57.9)},
    ("shanghait2dm", "ir"): {"glucofm": (67.0, 57.8), "cgm_jepa": (69.1, 60.8),
                             "x_cgm_jepa": (66.9, 58.4), "gluformer_tiny": (58.1, 47.6)},
    ("shanghait2dm", "hyperlipidemia"): {"glucofm": (33.5, 50.5), "cgm_jepa": (37.4, 53.6),
                                         "x_cgm_jepa": (35.5, 52.1), "gluformer_tiny": (33.9, 51.1)},
    ("shanghait2dm", "hypoglycemia"): {"glucofm": (21.1, 59.2), "cgm_jepa": (17.8, 56.8),
                                       "x_cgm_jepa": (16.9, 55.7), "gluformer_tiny": (17.9, 53.3)},
    ("stanford", "diabetes"): {"glucofm": (77.3, 72.8), "cgm_jepa": (66.4, 61.2),
                               "x_cgm_jepa": (67.4, 61.8), "gluformer_tiny": (74.3, 68.9)},
    ("stanford", "beta_cell"): {"glucofm": (69.0, 68.7), "cgm_jepa": (58.7, 55.4),
                                "x_cgm_jepa": (60.0, 56.4), "gluformer_tiny": (63.3, 61.9)},
    ("stanford", "ir"): {"glucofm": (67.6, 69.1), "cgm_jepa": (61.5, 61.6),
                         "x_cgm_jepa": (61.9, 62.0), "gluformer_tiny": (64.5, 65.4)},
    ("hall", "diabetes"): {"glucofm": (66.2, 75.9), "cgm_jepa": (59.9, 73.5),
                           "x_cgm_jepa": (59.3, 73.0), "gluformer_tiny": (48.9, 63.3)},
    ("hall", "ir"): {"glucofm": (60.2, 70.7), "cgm_jepa": (56.8, 68.1),
                     "x_cgm_jepa": (56.2, 67.7), "gluformer_tiny": (50.9, 63.6)},
    ("hall", "hyperlipidemia"): {"glucofm": (14.4, 41.6), "cgm_jepa": (17.3, 40.5),
                                 "x_cgm_jepa": (17.5, 40.0), "gluformer_tiny": (21.6, 58.0)},
    ("hall", "glucotype"): {"glucofm": (88.3, 90.7), "cgm_jepa": (87.6, 90.7),
                            "x_cgm_jepa": (87.7, 90.6), "gluformer_tiny": (75.4, 81.7)},
}

E7_THRESHOLD = 60.0   # proposal E7: mask-only AUC above this means a shortcut


def load(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        for p in sorted(Path(root).resolve().rglob("linear_probe.csv")):
            d = pd.read_csv(p)
            try:
                d["source"] = str(p.relative_to(ROOT))
            except ValueError:            # a root outside the project tree
                d["source"] = str(p)
            frames.append(d)
    if not frames:
        raise SystemExit(f"no linear_probe.csv under {roots}")
    d = pd.concat(frames, ignore_index=True)
    # Older runs wrote the task as "<dataset>/<task>"; normalise so the join
    # against PUBLISHED works for both.
    d["task"] = d["task"].astype(str).str.split("/").str[-1]
    # Later sources win for a duplicated (dataset, task, model).
    return d.drop_duplicates(subset=["dataset", "task", "model"], keep="last")


def main() -> int:
    ap = argparse.ArgumentParser()
    # Later roots win on a duplicated (dataset, task, model), so local reruns
    # override the Kaggle ones.
    ap.add_argument("--roots", nargs="*",
                    default=[str(ROOT / "artifacts" / "kaggle"),
                             str(ROOT / "artifacts" / "local"),
                             str(ROOT / "artifacts" / "eval")])
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "results"))
    a = ap.parse_args()

    df = load([Path(r) for r in a.roots if Path(r).exists()])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    models = [m for m in ["glucofm", "cgm_jepa", "x_cgm_jepa", "gluformer_tiny",
                          "gluformer_base", "raw", "mask_only"] if m in set(df["model"])]

    print("=== PR-AUC (ours) ===")
    print(df.pivot_table(index=["dataset", "task"], columns="model", values="PR")
            .reindex(columns=models).round(1).to_string())
    print("\n=== ROC-AUC (ours) ===")
    print(df.pivot_table(index=["dataset", "task"], columns="model", values="AUC")
            .reindex(columns=models).round(1).to_string())

    print("\n=== task-averaged ===")
    avg = df[df["model"].isin(models)].groupby("model")[["PR", "AUC", "F1"]].mean().round(1)
    avg["cells"] = df[df["model"].isin(models)].groupby("model").size()
    print(avg.to_string())
    print("  (GlucoFM paper average over its 14 cells: PR 58.8 / AUC 66.7 / F1 59.9)")

    # per-cell delta vs the published table
    rows = []
    for _, r in df.iterrows():
        pub = PUBLISHED.get((r["dataset"], r["task"]), {}).get(r["model"])
        if pub is None:
            continue
        rows.append({"dataset": r["dataset"], "task": r["task"], "model": r["model"],
                     "PR_ours": r["PR"], "PR_paper": pub[0], "dPR": round(r["PR"] - pub[0], 1),
                     "AUC_ours": r["AUC"], "AUC_paper": pub[1], "dAUC": round(r["AUC"] - pub[1], 1)})
    if rows:
        cmp = pd.DataFrame(rows).sort_values(["model", "dataset", "task"])
        cmp.to_csv(out / "vs_published.csv", index=False)
        print("\n=== vs GlucoFM Table 3 (delta = ours - paper) ===")
        print(cmp.to_string(index=False))
        print("\nper-model |dAUC|:")
        print(cmp.groupby("model")["dAUC"].agg(
            mean_abs=lambda s: round(s.abs().mean(), 2),
            max_abs=lambda s: round(s.abs().max(), 2),
            within_2=lambda s: int((s.abs() <= 2).sum()),
            n="size").to_string())

    # proposal E7 -- is the observation mask a shortcut?
    mo = df[df["model"] == "mask_only"]
    if not mo.empty:
        print("\n=== E7: probe on mask-derived features ALONE ===")
        print(mo[["dataset", "task", "PR", "AUC", "F1"]].round(1).to_string(index=False))
        hits = mo[mo["AUC"] > E7_THRESHOLD]
        if len(hits):
            print(f"\n  {len(hits)} cell(s) exceed AUC {E7_THRESHOLD:.0f} from missingness "
                  f"structure alone -- mask preservation is partly a shortcut on these, "
                  f"and any headline number here needs controlling for it:")
            for _, r in hits.iterrows():
                print(f"    {r['dataset']}/{r['task']}: AUC {r['AUC']:.1f}")
        else:
            print(f"\n  no cell exceeds AUC {E7_THRESHOLD:.0f} -- no evidence of a mask shortcut.")
        mo.to_csv(out / "e7_mask_only.csv", index=False)

    df.to_csv(out / "all_results.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "task_averaged": avg.to_dict(orient="index"),
        "n_cells": int(len(df)),
        "models": models,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
