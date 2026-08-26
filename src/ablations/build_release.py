"""Assemble the release folder for the final model, weights in safetensors.

Our checkpoints bundle tensors + config + history in one torch.save() pickle.
safetensors holds a flat tensor map only, so each checkpoint is split:

    <name>.safetensors   the weights          (no code execution on load)
    <name>.config.json   config + provenance  (human-readable)

Loading a .pt executes arbitrary code by construction; for a public release that
is not acceptable, which is why the release ships safetensors and keeps the .pt
only as an internal archive.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))


import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

SRC = ROOT
REL = Path(r"D:\glucoprism_final")
# v2 + variational bottleneck on zA at beta=0.1, read as zT||zS.
#
# Chosen over beta=1.0 despite a 0.27 AUC lower mean, because the two are
# statistically indistinguishable from each other (paired over 14 cells,
# p = 0.76) and beta=0.1 is better on every axis that matters for a release:
# it wins 13/14 cells against 10/14, it is the only model in the table that
# survives Holm-Bonferroni correction (p = 0.049 vs 0.128), it leads at subject
# level and on cross-dataset transfer, and it degrades far more gracefully when
# someone uses the full readout instead of dropping zA (67.83 vs 66.50).
BEST = "C-v2-vib01"

for sub in ("weights", "code", "code/reference", "eval", "data", "docs", "artifacts"):
    (REL / sub).mkdir(parents=True, exist_ok=True)


def flatten(obj, prefix=""):
    """Pull every tensor out of a nested checkpoint dict into a flat map."""
    tensors, meta = {}, {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if torch.is_tensor(v):
            tensors[key] = v.contiguous()
        elif isinstance(v, dict):
            t2, m2 = flatten(v, f"{key}.")
            tensors.update(t2)
            if m2:
                meta[k] = m2
        else:
            meta[k] = v
    return tensors, meta


print("=== weights -> safetensors ===")
for seed in (0, 1, 2):
    ck = SRC / "experiments" / "kaggle_out" / f"{BEST}-s{seed}" / "checkpoints" / "glucoprism.pt"
    if not ck.exists():
        print(f"  missing {ck}")
        continue
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    tensors, meta = flatten(blob)

    # The checkpoint stores every tensor TWICE -- once at the top level and
    # again under a `full.` prefix, sharing storage. torch.save happily writes
    # aliases; safetensors refuses them, because on reload they would become two
    # independent copies. Keep one name per storage (the shorter, canonical
    # one), which also halves the file.
    seen: dict[tuple, str] = {}
    dedup = {}
    for k in sorted(tensors, key=len):
        v = tensors[k]
        ident = (v.untyped_storage().data_ptr(), v.shape, v.stride(),
                 v.storage_offset())
        if ident in seen:
            continue
        seen[ident] = k
        dedup[k] = v.clone()          # clone breaks any residual aliasing
    n_alias = len(tensors) - len(dedup)
    tensors = {k: v for k, v in dedup.items() if v.dtype != torch.bool}
    out = REL / "weights" / f"glucoprism-final-s{seed}.safetensors"
    save_file(tensors, str(out), metadata={"format": "pt", "model": BEST,
                                           "seed": str(seed)})
    (REL / "weights" / f"glucoprism-final-s{seed}.config.json").write_text(
        json.dumps(meta, indent=2, default=str))
    n = sum(v.numel() for v in tensors.values())
    print(f"  seed {seed}: {len(tensors)} tensors ({n_alias} aliases dropped), "
          f"{n:,} params, {out.stat().st_size/1e6:.1f} MB")

print("\n=== code ===")
# The trainer is the sibling repo's own package; it ships whole because our
# port only configures it.
shutil.copytree(SRC / "external" / "glucoprism_v2_reference", REL / "code" / "reference",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "weights", "*.pt"))
for f in ["run_v2port.py", "build_v2_corpus.py", "build_corpus.py", "freeze_splits.py",
          "run_pretrain.py", "run_eval.py"]:
    if (SRC / "scripts" / f).exists():
        shutil.copy(SRC / "scripts" / f, REL / "code" / f)
shutil.copytree(SRC / "src" / "glucoprism", REL / "code" / "glucoprism",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

print("=== eval ===")
for f in ["v2_embed_runs.py", "v2_score_npy.py", "score_runs.py", "score_baselines.py",
          "run_baselines.py", "fd3_cross_dataset.py", "fd3_block_controls.py",
          "fd3_drop_za.py", "final_table.py", "fd9_sensor_analysis.py",
          "fd9_validate_generator.py", "fd9_fit_generator.py",
          "test_patch_geometry.py", "corpus_summary.py"]:
    p = SRC / "experiments" / "scripts" / f
    if p.exists():
        shutil.copy(p, REL / "eval" / f)

print("=== data ===")
for f in ["splits_frozen.json", "pretrain_holdout.json", "corpus_v2fmt_ov40.npz"]:
    p = SRC / "data" / "processed" / f
    if p.exists():
        shutil.copy(p, REL / "data" / f)
for p in (SRC / "data" / "processed").glob("*_ds.npz"):
    shutil.copy(p, REL / "data" / p.name)
for p in (SRC / "data" / "processed").glob("*_labels.csv"):
    shutil.copy(p, REL / "data" / p.name)

print("=== docs + artifacts ===")
for f in ["RESULTS.md", "FINDINGS.md"]:
    p = SRC / "experiments" / f
    if p.exists():
        shutil.copy(p, REL / "docs" / f)
for f in ["final_decisions.md", "discussion.md"]:
    if (SRC / f).exists():
        shutil.copy(SRC / f, REL / "docs" / f)
for f in ["final_table_long.csv", "baseline_scores.csv", "v2_final_scores.csv",
          "fd3_v2final.csv", "fd3_drop_za.csv", "fd45_scores.csv", "fd8_scores.csv",
          "fd7_scores.csv", "fd7seed_scores.csv",
          "fd9_sensor_calibration.json", "fd9_generator_validation.json",
          "fd9_pair_measurements.csv"]:
    p = SRC / "experiments" / "artifacts" / f
    if p.exists():
        shutil.copy(p, REL / "artifacts" / f)

# Credentials must never enter a release folder.
leaked = [p for p in REL.rglob("*") if p.is_file()
          and ("kaggle" in p.name.lower() and p.suffix == ".json"
               or "huggingface" in p.name.lower())]
for p in leaked:
    p.unlink()
    print(f"  removed credential file: {p}")

n = sum(1 for p in REL.rglob("*") if p.is_file())
mb = sum(p.stat().st_size for p in REL.rglob("*") if p.is_file()) / 1e6
print(f"\n{REL}: {n} files, {mb:.1f} MB")
