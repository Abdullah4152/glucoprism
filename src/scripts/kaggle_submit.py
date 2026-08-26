"""Run GlucoPRISM pretraining + evaluation on Kaggle GPUs.

The whole project payload (source + the built corpus) is under 1 MB, so the
workflow is simple:

  1. `package`  -- bundle src/ + scripts/ + data/processed/ into a Kaggle Dataset
  2. `push`     -- create/update one kernel per model, GPU enabled
  3. `status`   -- poll every pushed kernel
  4. `pull`     -- download finished kernel outputs into artifacts/kaggle/

Credentials come from the .kaggle/ directory in the project root. Several
accounts are present; `--account` selects one so long runs can be spread across
them (Kaggle caps GPU hours per account per week).

    python scripts/kaggle_run.py package
    python scripts/kaggle_run.py push --models glucofm cgm_jepa x_cgm_jepa gluformer_tiny
    python scripts/kaggle_run.py status
    python scripts/kaggle_run.py pull
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
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
KAGGLE_DIR = ROOT / ".kaggle"
BUILD = ROOT / "kaggle"
ARTIFACTS = ROOT / "artifacts" / "kaggle"

DATASET_SLUG = "glucoprism-corpus"

# Machine shape passed to `kaggle kernels push --accelerator`.
# "NvidiaTeslaP100" is accepted but useless here: Kaggle's torch build ships no
# sm_60 kernels, so those sessions silently run on CPU.
ACCELERATOR = "NvidiaTeslaT4"
MODELS = ["glucofm", "cgm_jepa", "x_cgm_jepa", "gluformer_tiny", "gluformer_base",
          "prism", "cqp", "v2port"]

# Paper-faithful epoch counts (see docs/reproduction/*.md). GlucoPRISM inherits
# GlucoFM's 120 because it is that backbone plus protocol objectives.
EPOCHS = {"glucofm": 120, "cgm_jepa": 101, "x_cgm_jepa": 101,
          "gluformer_tiny": 100, "gluformer_base": 76, "prism": 120,
          "cqp": 120, "v2port": 300}

# GlucoPRISM has its own entry point and its own flags, so the kernel dispatches
# on the model rather than assuming `run_pretrain.py --model <name>`.
TRAIN_SCRIPT = {m: "pretrain.py" for m in MODELS}
TRAIN_SCRIPT["prism"] = "pretrain_proposal_variant.py"


# --------------------------------------------------------------- credentials

def resolve_account(name: str | None) -> tuple[str, Path]:
    """Point KAGGLE_CONFIG_DIR at a directory holding exactly the chosen kaggle.json."""
    files = sorted(KAGGLE_DIR.glob("kaggle*.json"))
    if not files:
        raise SystemExit(f"no kaggle*.json under {KAGGLE_DIR}")

    chosen = None
    if name:
        for f in files:
            cfg = json.loads(f.read_text())
            if cfg.get("username") == name or f.stem == name:
                chosen = f
                break
        if chosen is None:
            have = [json.loads(f.read_text()).get("username") for f in files]
            raise SystemExit(f"account {name!r} not found. Available: {have}")
    else:
        chosen = KAGGLE_DIR / "kaggle.json"
        if not chosen.exists():
            chosen = files[0]

    active = ROOT / ".kaggle_active"
    active.mkdir(exist_ok=True)
    shutil.copy(chosen, active / "kaggle.json")
    os.environ["KAGGLE_CONFIG_DIR"] = str(active)
    username = json.loads(chosen.read_text())["username"]
    print(f"[kaggle] account: {username}  (from {chosen.name})")
    return username, active


def _kaggle_cmd() -> list[str]:
    """`python -m kaggle` only works where the kaggle package ships a __main__.

    On a machine with several interpreters, `sys.executable` is often not the one
    the CLI was installed into (here: 3.13 vs 3.12), and `-m kaggle` then fails
    with "'kaggle' is a package and cannot be directly executed". Prefer the
    console script on PATH and fall back to `-m` only if it is absent.
    """
    exe = shutil.which("kaggle")
    return [exe] if exe else [sys.executable, "-m", "kaggle"]


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [*_kaggle_cmd(), *args]
    print("  $ kaggle " + " ".join(args))
    p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    # The kaggle console script can hand back None for a stream it never wrote
    # to, so normalise before touching either one.
    p.stdout = p.stdout or ""
    p.stderr = p.stderr or ""
    if p.stdout.strip():
        print("    " + p.stdout.strip().replace("\n", "\n    "))
    if p.stderr.strip() and "Warning" not in p.stderr:
        print("    ! " + p.stderr.strip().replace("\n", "\n    "))
    if check and p.returncode != 0 and "already exists" not in (p.stdout + p.stderr):
        raise SystemExit(f"kaggle {' '.join(args)} failed ({p.returncode})")
    return p


# ------------------------------------------------------------------ payload

def package(username: str) -> Path:
    """Bundle source + corpus into kaggle/dataset/ with dataset-metadata.json."""
    out = BUILD / "dataset"
    if out.exists():
        shutil.rmtree(out)
    (out / "src").mkdir(parents=True)
    (out / "scripts").mkdir(parents=True)
    (out / "processed").mkdir(parents=True)

    shutil.copytree(ROOT / "src" / "core" / "glucoprism", out / "src" / "glucoprism",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Must stay in step with the kernel template's copy loop: a script listed
    # there but missing here fails only once the kernel is already on a GPU.
    for s in ["pretrain.py", "evaluate_models.py", "build_corpus.py",
              "pretrain_proposal_variant.py", "freeze_evaluation_folds.py", "pretrain_glucoprism.py",
              "pack_corpus_for_trainer.py"]:
        if (ROOT / "src" / "scripts" / s).exists():
            shutil.copy(ROOT / "src" / "scripts" / s, out / "scripts" / s)
    # The v2 port runs the sibling repo's own trainer, so its source tree has to
    # travel with the payload. Weights are excluded -- we retrain, not reload.
    ref = REFERENCE
    if ref.exists():
        shutil.copytree(ref, out / "reference",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "weights"))
    for p in (ROOT / "data" / "processed").glob("*"):
        if p.is_file():
            shutil.copy(p, out / "processed" / p.name)

    meta = {"title": "GlucoPRISM public CGM corpus + code",
            "id": f"{username}/{DATASET_SLUG}",
            "licenses": [{"name": "CC0-1.0"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    n = sum(1 for f in out.rglob("*") if f.is_file())
    print(f"[package] {n} files, {size/1e6:.2f} MB -> {out}")
    return out


KERNEL_TEMPLATE = '''\
"""GlucoPRISM :: {model} -- generated by scripts/kaggle_run.py. Do not edit here."""
import os, subprocess, sys, shutil, json, glob, zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

print("=== /kaggle/input tree ===")
for p in sorted(INPUT.rglob("*")):
    if p.is_file():
        print(f"{{p.stat().st_size:>12,}}  {{p}}")

# Kaggle usually auto-extracts a directory-structured dataset, but a version that
# is still processing at launch can surface the raw .zip members instead. Handle
# both, and fail loudly with the tree above if neither is present.
DATA = Path("/kaggle/input/{dataset_slug}")
for z in DATA.glob("*.zip"):
    print("extracting", z)
    with zipfile.ZipFile(z) as f:
        f.extractall(DATA.name if not os.access(DATA, os.W_OK) else DATA)

def find(sub):
    for cand in [DATA / sub, Path(DATA.name) / sub, WORK / sub]:
        if cand.exists():
            return cand
    hits = [p for p in INPUT.rglob(sub) if p.is_dir()]
    if hits:
        return hits[0]
    raise FileNotFoundError(f"could not locate {{sub}} under {{INPUT}} or {{WORK}}")

SRC, SCRIPTS, PROC = find("src"), find("scripts"), find("processed")
print("resolved:", SRC, SCRIPTS, PROC)

proc = WORK / "data" / "processed"
proc.mkdir(parents=True, exist_ok=True)
for p in PROC.glob("*"):
    if p.is_file():
        shutil.copy(p, proc / p.name)
print("corpus shards:", sorted(q.name for q in proc.glob("*")))

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

# run_pretrain / run_eval resolve paths relative to their own parents[1],
# so recreate the project layout (scripts/ + src/ + data/) under /kaggle/working.
(WORK / "scripts").mkdir(exist_ok=True)
# Copy EVERY script in the payload rather than an explicit list. Keeping a second
# hand-maintained list in step with package() failed twice: a script present in
# the dataset but missing from this loop only surfaces once the kernel is already
# on a GPU, as "can't open file ... run_X.py".
for _s in SCRIPTS.glob("*.py"):
    shutil.copy(_s, WORK / "scripts" / _s.name)
(WORK / "src").mkdir(exist_ok=True)
if not (WORK / "src" / "glucoprism").exists():
    shutil.copytree(SRC / "glucoprism", WORK / "src" / "glucoprism")

# GlucoPRISM takes its proposal knobs on the command line; the baselines take
# --model. Both write their checkpoint into WORK/checkpoints for the probe.
if "{model}" == "v2port":
    # Runs the sibling repo's own trainer on our corpus; its package is also
    # called `glucoprism`, so it ships separately as `reference/`.
    try:
        REF = find("reference")
        if not (WORK / "external").exists():
            (WORK / "external").mkdir(exist_ok=True)
        if not (WORK / "external" / "glucoprism_v2_reference").exists():
            shutil.copytree(REF, WORK / "external" / "glucoprism_v2_reference")
    except FileNotFoundError:
        print("reference/ not in payload -- repackage")
    shutil.copy(SCRIPTS / "pack_corpus_for_trainer.py", WORK / "scripts" / "pack_corpus_for_trainer.py")
    subprocess.run([sys.executable, str(WORK / "scripts" / "pack_corpus_for_trainer.py")],
                   cwd=str(WORK))
    cmd = [sys.executable, str(WORK / "scripts" / "pretrain_glucoprism.py"),
           "--epochs", "{epochs}", "--seed", "{seed}",
           "--out", str(WORK / "checkpoints")] + {extra_args}
elif "{model}" == "prism":
    cmd = [sys.executable, str(WORK / "scripts" / "pretrain_proposal_variant.py"),
           "--epochs", "{epochs}", "--batch-size", "{batch_size}",
           "--seed", "{seed}", "--out", str(WORK / "checkpoints"),
           "--log-every", "5"] + {extra_args}
else:
    cmd = [sys.executable, str(WORK / "scripts" / "pretrain.py"),
           "--model", "{model}", "--epochs", "{epochs}",
           "--batch-size", "{batch_size}", "--seed", "{seed}",
           "--out", str(WORK / "checkpoints"), "--log-every", "5",
           "--gluco-cache", str(WORK / "gluco_cache.pkl")] + {extra_args}
print("train cmd:", " ".join(str(c) for c in cmd))
rc = subprocess.run(cmd, cwd=str(WORK)).returncode
print("pretrain exit", rc)

# Frozen linear probe on the downstream cohorts, same protocol for every model.
# v2port is skipped: it is the sibling repo's model class, whose package is also
# called `glucoprism`, so it cannot be imported alongside ours in one process.
# Those checkpoints are scored locally by the two-stage embed/score path instead.
if "{model}" == "v2port":
    print("eval skipped for v2port -- scored locally")
else:
    rc2 = subprocess.run([sys.executable, str(WORK / "scripts" / "evaluate_models.py"),
                          "--checkpoints", str(WORK / "checkpoints"),
                          "--models", "{model}",
                          "--out", str(WORK / "eval"), "--baselines"],
                         cwd=str(WORK)).returncode
    print("eval exit", rc2)

# Kaggle persists all of /kaggle/working as the kernel output. Drop the code and
# corpus copies we staged so the artefact is just checkpoints + eval + cache.
for junk in ["src", "scripts", "data", "external", "reference"]:
    shutil.rmtree(WORK / junk, ignore_errors=True)
for p in WORK.rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)

print("=== kernel output ===")
for f in sorted(glob.glob(str(WORK / "**" / "*"), recursive=True)):
    if os.path.isfile(f) and os.path.getsize(f) > 0:
        print(f"{{os.path.getsize(f):>12,}}  {{f}}")
'''


def push(models: list[str], username: str, batch_size: int, gpu: bool,
         seed: int = 0, extra_args: list[str] | None = None,
         tag: str | None = None) -> None:
    """`tag` gives a run its own kernel slug, so a sweep arm or a second seed
    does not overwrite the previous run's output on the same account."""
    for model in models:
        name = f"{model}-{tag}" if tag else model
        kdir = BUILD / "kernels" / name
        kdir.mkdir(parents=True, exist_ok=True)
        slug = f"glucoprism-{name.replace('_', '-')}"

        (kdir / f"{slug}.py").write_text(
            KERNEL_TEMPLATE.format(model=model, epochs=EPOCHS[model],
                                   batch_size=batch_size, seed=seed,
                                   extra_args=repr(list(extra_args or [])),
                                   dataset_slug=DATASET_SLUG), encoding="utf-8")
        (kdir / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{username}/{slug}",
            "title": f"GlucoPRISM {name}",
            "code_file": f"{slug}.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": gpu,
            "enable_internet": False,
            "dataset_sources": [f"{username}/{DATASET_SLUG}"],
            "competition_sources": [],
            "kernel_sources": [],
        }, indent=2), encoding="utf-8")

        print(f"[push] {slug}")
        # The machine shape is a CLI flag, NOT a kernel-metadata field -- a
        # metadata "accelerator" key is silently ignored. Without it Kaggle
        # picks freely and often hands out a Tesla P100 (sm_60), which the
        # preinstalled torch has no kernels for: cuda.is_available() is True,
        # every launch raises cudaErrorNoKernelImageForDevice, and get_device()
        # falls back to CPU. That is merely slow for the patch-level models and
        # fatal for GluFormer, which attends over 288 tokens. Pin T4 (sm_75).
        args = ["kernels", "push", "-p", str(kdir)]
        if gpu:
            args += ["--accelerator", ACCELERATOR]
        kaggle(*args)


def status(models: list[str], username: str, tag: str | None = None) -> None:
    for model in models:
        name = f"{model}-{tag}" if tag else model
        slug = f"glucoprism-{name.replace('_', '-')}"
        kaggle("kernels", "status", f"{username}/{slug}", check=False)


def pull(models: list[str], username: str, tag: str | None = None) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for model in models:
        name = f"{model}-{tag}" if tag else model
        slug = f"glucoprism-{name.replace('_', '-')}"
        dest = ARTIFACTS / name
        # The CLI SKIPS files that already exist, so a stale directory would be
        # silently re-reported as a fresh result. Clear it first.
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", f"{username}/{slug}", "-p", str(dest), check=False)
        got = [p.name for p in dest.rglob("*") if p.is_file()]
        print(f"[pull] {model}: {len(got)} files -> {dest}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["package", "push", "status", "pull", "all"])
    ap.add_argument("--models", nargs="*", default=["glucofm", "cgm_jepa",
                                                    "x_cgm_jepa", "gluformer_tiny"])
    ap.add_argument("--account", default=None,
                    help="kaggle username or kaggle_kagN filename stem")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--new-version", action="store_true",
                    help="upload a new version of an existing dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="suffix for the kernel slug, so a second seed or a "
                         "sweep arm gets its own kernel instead of overwriting")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="args forwarded verbatim to run_prism.py, e.g. "
                         "--extra --align-on block --hold-out")
    a = ap.parse_args()

    username, _ = resolve_account(a.account)
    bad = [m for m in a.models if m not in MODELS]
    if bad:
        raise SystemExit(f"unknown models: {bad}")

    if a.action in ("package", "all"):
        d = package(username)
        if a.new_version:
            kaggle("datasets", "version", "-p", str(d), "-m",
                   f"corpus refresh {time.strftime('%Y-%m-%d %H:%M')}", "-r", "zip")
        else:
            p = kaggle("datasets", "create", "-p", str(d), "-r", "zip", check=False)
            # Kaggle reports this two different ways depending on whether the
            # clash is on the slug or on the title, so match both.
            msg = (p.stdout + p.stderr).lower()
            if "already exists" in msg or "already in use" in msg:
                print("  dataset exists -> uploading a new version instead")
                kaggle("datasets", "version", "-p", str(d), "-m",
                       f"corpus refresh {time.strftime('%Y-%m-%d %H:%M')}", "-r", "zip")

    if a.action in ("push", "all"):
        push(a.models, username, a.batch_size, gpu=not a.no_gpu,
             seed=a.seed, extra_args=a.extra, tag=a.tag)
    if a.action in ("status", "all"):
        status(a.models, username, tag=a.tag)
    if a.action == "pull":
        pull(a.models, username, tag=a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
