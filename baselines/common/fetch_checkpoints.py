"""Fetch and stage the third-party checkpoints we use zero-shot.

All are used FROZEN -- none is retrained on our corpus, exactly as GlucoFM
App. B.1/B.2 specifies. None of these weights are redistributed by this
repository; they belong to their authors and keep their own licences.

Four of the seven load by Hugging Face repo id and only need the cache warmed.
Three do not: `embed_zeroshot.py` loads Mantis, MantisV2 and CGMformer from a
local `external/` tree, so those have to be staged there as well. An earlier
version of this script warmed the cache and stopped, which left the embedder
handing a filesystem path to the Hub as if it were a repo id:

    Mantis  cgmacros  FAILED HFValidationError: Repo id must use alphanumeric
    chars ... : '/home/you/glucoprism-rel...'

    python fetch_checkpoints.py                 # everything
    python fetch_checkpoints.py --models Mantis MantisV2
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# HF_HOME is left to the environment; the released copy hard-coded a path
# that exists only on the authors machine.
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

ROOT = Path(os.environ.get("GLUCOPRISM_ROOT",
                           Path(__file__).resolve().parents[2]))
EXTERNAL = Path(os.environ.get("GLUCOPRISM_EXTERNAL", ROOT / "external"))

# Model name -> Hugging Face repo. Keys are exactly the keys of
# `embed_zeroshot.py::MODELS`; keeping the two in step is the whole point.
REPOS = {
    "MOMENT-small": "AutonLab/MOMENT-1-small",
    "MOMENT-large": "AutonLab/MOMENT-1-large",
    "Mantis": "paris-noah/Mantis-8M",
    "MantisV2": "paris-noah/MantisV2",
    "Chronos-2": "amazon/chronos-2",
    "Chronos-2-small": "autogluon/chronos-2-small",
}

# Models the embedder loads from a local directory rather than by repo id.
# name -> (destination under external/, source repo or None if manual)
STAGE = {
    "Mantis": (EXTERNAL / "Mantis", "paris-noah/Mantis-8M"),
    "MantisV2": (EXTERNAL / "MantisV2", "paris-noah/MantisV2"),
    "CGMformer": (EXTERNAL / "cgmformer" / "ckpt" / "cgm_ckp", None),
}

CGMFORMER_NOTE = """\
CGMformer is not on the Hugging Face Hub and cannot be fetched automatically.
Obtain the released checkpoint from the authors' repository and unpack it so
that this layout exists:

    external/cgmformer/ckpt/cgm_ckp/checkpoint-30000/    (HF BertModel directory)
    external/cgmformer/ckpt/cgm_ckp/token2id.pkl
    external/cgmformer/ckpt/cgm_ckp/id2token.pkl

`embed_zeroshot.py::embed_cgmformer` reads exactly those three paths. Note that
CGMformer replaces BERT's learnable position embedding with a sinusoidal one, so
the checkpoint contains no `position_embeddings.weight`; the embedder rebuilds
it. Loading it with stock HuggingFace BERT and no such fix-up leaves that table
randomly initialised.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=sorted(set(REPOS) | set(STAGE)),
                    help="names as used by embed_zeroshot.py, e.g. MOMENT-large")
    a = ap.parse_args()

    unknown = [m for m in a.models if m not in REPOS and m not in STAGE]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}\n"
                         f"known: {sorted(set(REPOS) | set(STAGE))}")

    snapshots: dict[str, str] = {}
    for name in a.models:
        repo = REPOS.get(name)
        if not repo:
            continue
        try:
            from huggingface_hub import snapshot_download
            p = snapshot_download(repo)
            snapshots[name] = p
            mb = sum(f.stat().st_size for f in Path(p).rglob("*")
                     if f.is_file()) / 1e6
            print(f"  {name:<18} {repo:<32} {mb:>8.1f} MB", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<18} {repo:<32} FAILED {type(e).__name__}: "
                  f"{str(e)[:80]}", flush=True)

    # Stage the three the embedder loads from disk.
    missing = []
    for name in a.models:
        if name not in STAGE:
            continue
        dest, repo = STAGE[name]
        if dest.exists() and any(dest.iterdir()):
            print(f"  {name:<18} already staged at {dest}")
            continue
        src = snapshots.get(name)
        if repo and src:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            print(f"  {name:<18} staged -> {dest}")
        else:
            missing.append(name)

    if "CGMformer" in missing:
        print("\n" + CGMFORMER_NOTE, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
