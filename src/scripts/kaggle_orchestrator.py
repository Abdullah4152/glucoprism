"""Run the FD-4..FD-8 programme across 8 Kaggle accounts x 2 slots.

  python orchestrate.py plan                 # show the queue, push nothing
  python orchestrate.py init                 # upload the payload to every account
  python orchestrate.py step                 # one pass: pull done, push pending
  python orchestrate.py loop --minutes 12    # step forever
  python orchestrate.py status

State lives in experiments/artifacts/orchestrator_state.json so the loop can be
killed and resumed without losing track of what is on which account.

Scheduling rules (final_decisions.md):
  * no two seeds of one config on the same account -- a suspension must not take
    out a whole cell
  * never two 5x runs in one account's two slots
  * one account held back as a retry lane
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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = ROOT
KDIR = Path(os.environ["USERPROFILE"]) / ".kaggle"
ART = ROOT / "experiments" / "artifacts"
def state_path() -> Path:
    """One state file per stage, so stages can be run and resumed independently."""
    return ART / f"orchestrator_state_{STAGE}.json"
OUTDIR = ROOT / "experiments" / "kaggle_out"
BUILD = ROOT / "kaggle"
DATASET_SLUG = "glucoprism-corpus"
ACCELERATOR = "NvidiaTeslaT4"
SLOTS_PER_ACCOUNT = 2

sys.path.insert(0, str(ROOT / "scripts"))


# ------------------------------------------------------------------ accounts

# Accounts with no GPU allowance left. Excluded permanently rather than
# discovered per-push: a quota rejection costs a wasted API round-trip and, when
# it happens mid-pass, shuffles runs onto accounts the scheduler had reserved for
# something else. Override with GP_EXCLUDE="user1,user2" or clear it there.
#
# NOTE one account can own several key files, so dedupe by username.
# key files for one account, which is also why the pool dedupes by username.
EXCLUDED_DEFAULT: set[str] = set()   # set GP_EXCLUDE=user1,user2


def excluded_users() -> set[str]:
    env = os.environ.get("GP_EXCLUDE")
    if env is not None:
        return {u.strip() for u in env.split(",") if u.strip()}
    return set(EXCLUDED_DEFAULT)


def accounts() -> list[dict]:
    """Distinct usable accounts. Two files can hold the same account (kaggle.json
    and kaggle_kag3.json are both <kaggle-user>); dedupe by username or the pool
    is overcounted and slots are oversubscribed."""
    skip = excluded_users()
    seen, out, dropped = {}, [], set()
    for f in sorted(KDIR.glob("kaggle*.json")):
        try:
            c = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        u = c.get("username")
        if not u or u in seen:
            continue
        if u in skip:
            dropped.add(u)
            continue
        seen[u] = f
        out.append({"user": u, "file": str(f), "key": c["key"]})
    if dropped:
        print(f"  (excluded, no GPU allowance: {sorted(dropped)})")
    return out


def kenv(acc: dict) -> dict:
    e = dict(os.environ)
    e["KAGGLE_USERNAME"] = acc["user"]
    e["KAGGLE_KEY"] = acc["key"]
    e.pop("KAGGLE_CONFIG_DIR", None)
    return e


KAGGLE = shutil.which("kaggle") or "kaggle"


def kg(acc: dict, *args: str, timeout: int = 900) -> tuple[int, str]:
    """Run the Kaggle CLI for one account.

    encoding/errors are pinned: the CLI emits bytes that Windows' default cp1252
    cannot decode, which raises inside subprocess's reader thread and loses the
    output entirely.
    """
    try:
        p = subprocess.run([KAGGLE, *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           env=kenv(acc), timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def kg_failed(rc: int, out: str) -> bool:
    """The Kaggle CLI exits 0 on several real failures -- a title clash prints
    'Dataset creation error: ...' and still returns 0. Trusting rc alone means a
    stale dataset is silently left in place and every kernel on that account
    trains on the wrong corpus. Check the text too."""
    low = out.lower()
    return rc != 0 or "error" in low or "already in use" in low or "already exists" in low


# ---------------------------------------------------------------- run queue

def stage_w(tag: str, suffix: str, *geom: str, seed: int = 0) -> dict:
    return {"id": tag, "stage": "FD7", "model": "glucofm", "seed": seed,
            "extra": ["--shard-suffix", suffix, *geom]}


STAGE = os.environ.get("GP_STAGE", "fd7")


def build_queue(stage: str | None = None) -> list[dict]:
    stage = stage or STAGE

    if stage == "fd7":
        # Windowing. 1 seed each; geometry and corpus are the variables.
        return [
            stage_w("W1-ov0", "_ov0"),
            stage_w("W2-ov20m", "_ov20m"),
            stage_w("W3-ov40m", "_ov40m"),
            stage_w("W3u-ov40", "_ov40"),
            stage_w("W4-k18", "_ov0", "--patch-k", "18", "--n-patches", "24",
                    "--patch-stride", "12"),
            stage_w("W5-k18-ov40", "_ov40", "--patch-k", "18", "--n-patches", "24",
                    "--patch-stride", "12"),
            stage_w("W6-k6", "_ov0", "--patch-k", "6", "--n-patches", "48"),
            stage_w("W7-k24", "_ov0", "--patch-k", "24", "--n-patches", "12"),
        ]

    if stage == "fd7seed":
        # W1 and W3u finished 0.3 AUC apart on one seed, which is inside the
        # 1.00 seed sigma -- unreadable. This choice fixes the corpus for 126
        # downstream runs, so it gets 2 more seeds rather than a coin flip.
        # W6 joins them: its overall delta is inside noise but it split the
        # cohorts hard by sampling rate (ShanghaiT2DM -3.8, Stanford +3.6), and
        # that pattern needs a second seed before it becomes a claim.
        q = []
        for s in (1, 2):
            q += [stage_w(f"W1-ov0-s{s}", "_ov0", seed=s),
                  stage_w(f"W3u-ov40-s{s}", "_ov40", seed=s),
                  stage_w(f"W6-k6-s{s}", "_ov0", "--patch-k", "6",
                          "--n-patches", "48", seed=s)]
        return q

    if stage == "fd8":
        # Architecture grid. Geometry is FD-7's winner: K=12, P=24,
        # non-overlapping. Corpus suffix is filled from GP_CORPUS so the whole
        # stage moves together once FD-7's seed runs settle it.
        c = os.environ.get("GP_CORPUS", "_ov0")
        W5X = ["--width-scale", "2.375"]
        SENSOR = ["--sensor-aug", "0.5"]
        base = ["--shard-suffix", c]

        def prism(tag, seed, *extra):
            return {"id": f"{tag}-s{seed}", "stage": "FD8", "model": "prism",
                    "script": "run_prism.py", "seed": seed,
                    "extra": [*base, "--head-layers", "2",
                              "--lambda-sensor", "0.2", "--lambda-day", "0.2",
                              "--lambda-indep", "0.1", *extra]}

        def fm(tag, seed, *extra):
            return {"id": f"{tag}-s{seed}", "stage": "FD8", "model": "glucofm",
                    "script": "run_pretrain.py", "seed": seed,
                    "extra": [*base, *extra]}

        q = []
        for s in (0, 1, 2):
            q += [
                prism("V1-fm-joint", s),                       # 1x, factorized
                prism("V2-5x-joint", s, *W5X),                 # 5x, factorized
                fm("V4-fm-off", s),                            # the one control
                fm("V5-5x-off", s, *W5X),                      # 5x control
                fm("V6-fm-post", s, *SENSOR),                  # post-hoc substrate
                fm("V7-5x-post", s, *W5X, *SENSOR),            # 5x post-hoc substrate
            ]
        return q

    if stage == "fd45":
        # Corpus. Geometry K=12/P=24 non-overlapping, day overlap 0.4 -- FD-7's
        # winners. The encoder is the plain GlucoFM control (V4), because F7/F8
        # showed the factorization only subtracts, and a flat instrument cannot
        # measure a gradient.
        ALL = ["replacebg", "stanford", "shanghait2dm", "colas", "bigideas"]

        def run(tag, suffix, datasets, seed=0):
            return {"id": f"{tag}-s{seed}", "stage": "FD45", "model": "glucofm",
                    "script": "run_pretrain.py", "seed": seed,
                    "extra": ["--shard-suffix", suffix, "--datasets", *datasets]}

        q = []
        # FD-5: how much REPLACE-BG? Other four cohorts held constant, so the
        # only thing moving is the type-1 cohort's share and the corpus size.
        for f in (0, 10, 20, 30, 40, 50, 70, 90):
            ds = ALL if f > 0 else [d for d in ALL if d != "replacebg"]
            q.append(run(f"F{f:02d}", f"_ov40f{f}", ds))
        q.append(run("F100", "_ov40", ALL))          # the full corpus

        # FD-4: leave one cohort out. REPLACE-BG's drop is F00 above, so only the
        # other four need their own arm.
        for d in ALL[1:]:
            q.append(run(f"LOCO-{d[:6]}", "_ov40", [x for x in ALL if x != d]))
        return q

    if stage == "v2":
        # Finalising v2. Four arms x 3 seeds, all on the FD-7 corpus so every
        # number in the paper sits on one substrate.
        def v2(tag, seed, *extra):
            return {"id": f"{tag}-s{seed}", "stage": "V2", "model": "v2port",
                    "script": "run_v2port.py", "seed": seed,
                    "extra": ["--corpus", "corpus_v2fmt_ov40.npz", *extra]}

        q = []
        for s in (0, 1, 2):
            q += [
                # A: v2 exactly as published, on our chosen corpus. The baseline
                #    every other arm is measured against.
                v2("A-v2-base", s),
                # B: zA-drop-aware. A variational bottleneck bounds I(x; zA), so
                #    zA cannot carry clinical signal and discarding it is free by
                #    construction rather than by luck (FD-3 / F9).
                v2("B-v2-vib1", s, "--use-vib", "--w-vib", "1.0"),
                # C: the same at a tenth the capacity price -- beta=1.0 adds ~7.8
                #    to a loss whose base is 1.1, which may swamp the objectives
                #    the way lambda=1.0 did for the protocol losses.
                v2("C-v2-vib01", s, "--use-vib", "--w-vib", "0.1"),
                # D: FD-9's measured sensor offset restored. v2 sets the V1
                #    partner's level shift to zero; real pairs differ by
                #    -31.1 +- 15.8 mg/dL in 43 of 44 subjects.
                v2("D-v2-simbias", s, "--sim-bias", "measured"),
            ]
        return q

    if stage == "v2corpus":
        # The released models were pretrained on the FULL corpus (82.5 %
        # REPLACE-BG). The FD-5 sweep suggested 50-70 % might be better, but it
        # was measured on plain GlucoFM -- an instrument that is flat across
        # corpora -- at one seed, on a curve spanning 1.8 AUC against a +-1.0
        # sigma. The peak was explicitly not claimed at the time. This tests the
        # question on the architecture we actually release.
        return [{"id": f"C-rbg{p}-s{s}", "stage": "V2C", "model": "v2port",
                 "script": "run_v2port.py", "seed": s,
                 "extra": ["--corpus", f"corpus_v2fmt_ov40f{p}.npz",
                           "--use-vib", "--w-vib", "0.1"]}
                for p in (50, 70) for s in (0, 1, 2)]

    if stage == "v2corpus2":
        # Three more seeds at 50 % only. The 70 % arm is not extended: it landed
        # between 100 % and 50 % on every axis, so it is an interpolation point,
        # not a release candidate.
        return [{"id": f"C-rbg50-s{s}", "stage": "V2C", "model": "v2port",
                 "script": "run_v2port.py", "seed": s,
                 "extra": ["--corpus", "corpus_v2fmt_ov40f50.npz",
                           "--use-vib", "--w-vib", "0.1"]}
                for s in (3, 4, 5)]

    if stage == "v2cseeds":
        # Three more seeds for C only. Justified by the variance decomposition,
        # not by a wish for a smaller p: C's per-cell spread is 2.26 against a
        # seed-noise floor of 1.89 (ratio 1.19), so roughly half its spread is
        # measurement error that more seeds actually remove. B and E have ratios
        # of 1.67 and 2.06 -- their spread is genuine task heterogeneity and more
        # seeds would not touch it, so they do not get extra runs.
        return [{"id": f"C-v2-vib01-s{s}", "stage": "V2", "model": "v2port",
                 "script": "run_v2port.py", "seed": s,
                 "extra": ["--corpus", "corpus_v2fmt_ov40.npz",
                           "--use-vib", "--w-vib", "0.1"]}
                for s in (3, 4, 5)]

    if stage == "v2bd":
        # B and D each win on one axis and have never been combined: B (the zA
        # bottleneck) leads within-cohort, D (FD-9's measured -31.1 mg/dL sensor
        # offset) leads cross-dataset. They touch different parts of the model,
        # so there is no reason they should not compose.
        return [{"id": f"E-v2-vib-simbias-s{s}", "stage": "V2", "model": "v2port",
                 "script": "run_v2port.py", "seed": s,
                 "extra": ["--corpus", "corpus_v2fmt_ov40.npz",
                           "--use-vib", "--w-vib", "1.0",
                           "--sim-bias", "measured"]}
                for s in (0, 1, 2)]

    raise SystemExit(f"unknown stage {stage!r}")


def load_state() -> dict:
    # utf-8-sig, because anything that edits this file from PowerShell
    # (`Set-Content -Encoding utf8`) leaves a BOM that json.loads rejects.
    p = state_path()
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else {}


def save_state(s: dict) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(s, indent=2))


def slug_for(run_id: str) -> str:
    return f"gp-{run_id.lower().replace('_', '-')}"


# -------------------------------------------------------------------- payload

def package(user: str) -> Path:
    """Bundle src + scripts + every processed shard into one Kaggle dataset."""
    out = BUILD / "dataset"
    if out.exists():
        shutil.rmtree(out)
    (out / "src").mkdir(parents=True)
    (out / "scripts").mkdir(parents=True)
    (out / "processed").mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "core" / "glucoprism", out / "src" / "glucoprism",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for s in (ROOT / "scripts").glob("*.py"):
        shutil.copy(s, out / "scripts" / s.name)
    # The v2 trainer is the sibling repo's own code. Its package is ALSO called
    # `glucoprism`, so it ships as a separate tree and is put on sys.path only
    # inside the v2 kernels.
    ref = ROOT / "external" / "glucoprism_v2_reference"
    if ref.exists():
        shutil.copytree(ref, out / "reference",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                      "weights", "*.pt"))
    for p in (ROOT / "data" / "processed").glob("*"):
        if p.is_file():
            shutil.copy(p, out / "processed" / p.name)
    # Content canary. Verifying by a DATA filename cannot detect a stale payload
    # whose scripts are out of date -- which is exactly what happened on the
    # first FD-8 push: the shards were current, `run_prism.py` was not, and 12
    # runs died on argparse while Kaggle reported them complete. The canary's
    # NAME encodes a hash of the code, so `datasets files` alone proves whether
    # the current source actually landed.
    h = hashlib.sha256()
    for f in sorted((out / "scripts").rglob("*.py")) + sorted((out / "src").rglob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    tag = h.hexdigest()[:12]
    (out / f"PAYLOAD_{tag}.txt").write_text(tag)

    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "GlucoPRISM public CGM corpus + code",
         "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    n = sum(1 for f in out.rglob("*") if f.is_file())
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"  packaged {n} files, {mb:.1f} MB, code hash {tag}")
    return out


def cmd_init(args) -> None:
    accs = accounts()
    print(f"{len(accs)} distinct accounts, {len(accs)*SLOTS_PER_ACCOUNT} slots\n")
    bad = []
    marker = None
    for a in accs:
        print(f"[{a['user']}]")
        d = package(a["user"])
        marker = next(p.name for p in d.glob("PAYLOAD_*.txt"))
        rc, out = kg(a, "datasets", "create", "-p", str(d), "-r", "zip", timeout=2400)
        if kg_failed(rc, out):
            print("  create rejected -> uploading a new version instead")
            rc, out = kg(a, "datasets", "version", "-p", str(d), "-m",
                         f"refresh {time.strftime('%Y-%m-%d %H:%M')}", "-r", "zip",
                         timeout=2400)
        last = out.strip().splitlines()[-1][:120] if out.strip() else ""
        ok = not kg_failed(rc, out)
        print(f"  {'ok' if ok else 'FAILED'}  rc={rc}  {last}")
        if not ok:
            bad.append(a["user"])

    # `datasets status` reports "ready" while a new version is still processing,
    # so it cannot confirm the upload landed. Verify by listing files and looking
    # for one that only exists in the NEW payload.
    print(f"\nverifying every account serves '{marker}' ...")
    for a in accs:
        rc, out = kg(a, "datasets", "files", f"{a['user']}/{DATASET_SLUG}",
                     "--page-size", "500", timeout=600)
        has = marker in out
        print(f"  {'ok  ' if has else 'MISS'}  {a['user']}")
        if not has and a["user"] not in bad:
            bad.append(a["user"])
    if bad:
        print(f"\n!! {len(bad)} account(s) not serving the new payload: {bad}")
        print("   re-run `init` for these before `step`, or their kernels will "
              "train on a stale corpus.")
    else:
        print("\nall accounts serving the new payload")


# ----------------------------------------------------------------- kernel body

KERNEL = '''\
"""{run_id} -- generated by experiments/scripts/orchestrate.py. Do not edit."""
import os, shutil, subprocess, sys, glob, zipfile, json
from pathlib import Path

WORK = Path("/kaggle/working"); INPUT = Path("/kaggle/input")
DATA = INPUT / "{dataset_slug}"
for z in DATA.glob("*.zip"):
    with zipfile.ZipFile(z) as f:
        f.extractall(DATA)

def find(sub):
    for c in [DATA / sub, Path(DATA.name) / sub, WORK / sub]:
        if c.exists():
            return c
    hits = [p for p in INPUT.rglob(sub) if p.is_dir()]
    if hits:
        return hits[0]
    raise FileNotFoundError(sub)

SRC, SCRIPTS, PROC = find("src"), find("scripts"), find("processed")
proc = WORK / "data" / "processed"; proc.mkdir(parents=True, exist_ok=True)
for p in PROC.glob("*"):
    if p.is_file():
        shutil.copy(p, proc / p.name)
(WORK / "scripts").mkdir(exist_ok=True)
for s in SCRIPTS.glob("*.py"):
    shutil.copy(s, WORK / "scripts" / s.name)
(WORK / "src").mkdir(exist_ok=True)
if not (WORK / "src" / "glucoprism").exists():
    shutil.copytree(SRC / "glucoprism", WORK / "src" / "glucoprism")

import torch
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", dev, flush=True)
# A P100 has no sm_60 kernels in Kaggle's torch: cuda.is_available() is True but
# every launch raises and the run silently finishes on CPU, ~6x slower and
# indistinguishable from success in the logs. Fail loudly instead.
if torch.cuda.is_available() and "T4" not in dev:
    print("!! expected a T4, got", dev, "-- results would be CPU-slow", flush=True)

SCRIPT = "{script}"
if SCRIPT == "run_v2port.py":
    # The sibling repo's trainer needs its own tree present where run_v2port.py
    # expects it: ROOT/external/glucoprism_v2_reference.
    REF = find("reference")
    dst = WORK / "external" / "glucoprism_v2_reference"
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists():
        shutil.copytree(REF, dst)
    print("reference tree:", dst, flush=True)

cmd = [sys.executable, str(WORK / "scripts" / SCRIPT),
       "--epochs", "{epochs}", "--seed", "{seed}",
       "--out", str(WORK / "checkpoints")]
if SCRIPT == "run_pretrain.py":
    cmd += ["--model", "{model}", "--batch-size", "{bs}", "--log-every", "10"]
cmd += {extra}
print("train:", " ".join(map(str, cmd)), flush=True)
rc = subprocess.run(cmd, cwd=str(WORK)).returncode
print("pretrain exit", rc, flush=True)

if SCRIPT == "run_v2port.py":
    # Their package is also named `glucoprism`, so it cannot be imported
    # alongside ours in one process. These checkpoints are scored locally by the
    # two-stage embed/score path instead.
    rc2 = 0
    print("eval skipped for v2port -- scored locally", flush=True)
else:
    rc2 = subprocess.run([sys.executable, str(WORK / "scripts" / "run_eval.py"),
                          "--checkpoints", str(WORK / "checkpoints"),
                          "--models", "{model}", "--out", str(WORK / "eval")],
                         cwd=str(WORK)).returncode
    print("eval exit", rc2, flush=True)
(WORK / "run_meta.json").write_text(json.dumps(
    {{"run_id": "{run_id}", "extra": {extra}, "device": dev,
      "pretrain_rc": rc, "eval_rc": rc2}}, indent=2))

for junk in ["src", "scripts", "data", "external", "reference"]:
    shutil.rmtree(WORK / junk, ignore_errors=True)
for p in WORK.rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)
print("=== output ===")
for f in sorted(glob.glob(str(WORK / "**" / "*"), recursive=True)):
    if os.path.isfile(f):
        print(f"{{os.path.getsize(f):>12,}}  {{f}}")
'''


def push_run(acc: dict, run: dict, epochs: int, bs: int) -> tuple[bool, str]:
    slug = slug_for(run["id"])
    kdir = BUILD / "kernels" / run["id"]
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / f"{slug}.py").write_text(KERNEL.format(
        run_id=run["id"], model=run["model"], epochs=epochs, bs=bs,
        script=run.get("script", "run_pretrain.py"),
        seed=run["seed"], extra=repr(list(run["extra"])),
        dataset_slug=DATASET_SLUG), encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{acc['user']}/{slug}", "title": f"GP {run['id']}",
        "code_file": f"{slug}.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "dataset_sources": [f"{acc['user']}/{DATASET_SLUG}"],
        "competition_sources": [], "kernel_sources": [],
    }, indent=2), encoding="utf-8")
    # --accelerator is a CLI FLAG, not a metadata field; a metadata key is
    # silently ignored and Kaggle then picks the machine freely.
    rc, out = kg(acc, "kernels", "push", "-p", str(kdir),
                 "--accelerator", ACCELERATOR)
    # `kernels push` exits 0 while printing "Kernel push error: Maximum weekly GPU
    # quota of 30.00 hours reached." Trusting rc marks the run pushed and it then
    # polls forever against a kernel that was never queued.
    ok = rc == 0 and "error" not in out.lower()
    return ok, out.strip().splitlines()[-1][:160] if out.strip() else ""


def quota_exhausted(msg: str) -> bool:
    return "weekly gpu quota" in (msg or "").lower()


def kernel_status(acc: dict, run_id: str) -> str:
    rc, out = kg(acc, "kernels", "status", f"{acc['user']}/{slug_for(run_id)}",
                 timeout=180)
    low = out.lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in low:
            return s
    return "unknown"


def pull_run(acc: dict, run_id: str) -> int:
    dest = OUTDIR / run_id
    if dest.exists():
        shutil.rmtree(dest)          # the CLI SKIPS existing files -> stale results
    dest.mkdir(parents=True, exist_ok=True)
    kg(acc, "kernels", "output", f"{acc['user']}/{slug_for(run_id)}",
       "-p", str(dest), timeout=900)
    return sum(1 for p in dest.rglob("*") if p.is_file())


def cmd_step(args) -> None:
    accs = accounts()
    if not accs:
        raise SystemExit("no usable accounts")
    # A reserved retry lane made sense at 8 accounts; at 6 it costs 17 % of
    # capacity to insure against a failure mode (a run needing an immediate
    # re-push) that the resumable queue already handles on the next pass.
    # Keep it only when the pool is large enough to spare one.
    retry_lane = accs[-1]["user"] if len(accs) >= 8 else None
    pool = [a for a in accs if a["user"] != retry_lane] or accs
    print(f"  pool: {len(pool)} accounts x {SLOTS_PER_ACCOUNT} slots"
          + (f", retry lane {retry_lane}" if retry_lane else ", no retry lane"))
    by_user = {a["user"]: a for a in accs}

    st = load_state()
    q = build_queue()
    for r in q:
        st.setdefault(r["id"], {"status": "pending"})

    # 1. poll everything in flight, pull what finished
    for rid, s in st.items():
        if rid.startswith("_") or not isinstance(s, dict):
            continue                       # bookkeeping keys, not runs
        if s.get("status") not in ("pushed", "running", "queued"):
            continue
        acc = by_user.get(s.get("account"))
        if acc is None:
            continue
        k = kernel_status(acc, rid)
        if k == "complete":
            n = pull_run(acc, rid)
            # Kaggle reports "complete" for any kernel that ran to the end of the
            # script -- including one whose training died on an argparse error and
            # whose eval then found no checkpoint. Trusting kernel status alone
            # silently banks empty runs as results. Check what the run actually
            # produced.
            meta = OUTDIR / rid / "run_meta.json"
            rc = None
            if meta.exists():
                try:
                    rc = json.loads(meta.read_text()).get("pretrain_rc")
                except Exception:  # noqa: BLE001
                    rc = None
            has_ck = (OUTDIR / rid / "checkpoints").exists()
            if rc not in (0, None) or not has_ck:
                s["status"] = "failed"
                s["fails"] = s.get("fails", 0) + 1
                s["why"] = f"pretrain_rc={rc} checkpoints={has_ck}"
                print(f"  [FAIL]  {rid:<16} {s['account']:<22} {s['why']}")
            else:
                s["status"], s["files"] = "complete", n
                print(f"  [done]  {rid:<16} {s['account']:<22} {n} files")
        elif k in ("error", "cancel"):
            s["status"] = "failed"
            s["fails"] = s.get("fails", 0) + 1
            print(f"  [FAIL]  {rid:<14} {s['account']:<22} ({k})")
        else:
            s["status"] = k
            print(f"  [{k:<7}] {rid:<14} {s['account']}")

    # 2. fill free slots
    inflight: dict[str, int] = {}
    for _k, s in st.items():
        if _k.startswith("_") or not isinstance(s, dict):
            continue
        if s.get("status") in ("pushed", "running", "queued"):
            inflight[s.get("account", "")] = inflight.get(s.get("account", ""), 0) + 1

    todo = [r for r in q if st[r["id"]]["status"] in ("pending", "failed")
            and st[r["id"]].get("fails", 0) < 3]
    # An account that has burned its 30 GPU-h for the week cannot take ANY run,
    # so retrying it per-run just wastes the queue. The exhausted set persists in
    # the state file with a timestamp -- Kaggle quota is weekly, so an entry
    # older than 7 days is stale and gets cleared. Without persistence every
    # fresh invocation re-tries the same dead accounts in pool order.
    quota = st.setdefault("_quota", {})
    now = time.time()
    for u in [u for u, t in quota.items() if now - t > 7 * 86400]:
        quota.pop(u)
    dead: set[str] = set(quota)
    if dead:
        print(f"  (skipping quota-exhausted: {sorted(dead)})")
    for run in todo:
        acc = next((a for a in pool
                    if a["user"] not in dead
                    and inflight.get(a["user"], 0) < SLOTS_PER_ACCOUNT), None)
        if acc is None:
            print("  (no free slot -- remaining runs stay queued)")
            break
        # A quota rejection is not the run's fault, so it is not counted against
        # it -- keep walking the pool until an account accepts it or none is
        # left. Trying a single alternate is not enough once several accounts
        # are exhausted, which is the normal state late in a week.
        ok, msg = push_run(acc, run, epochs=args.epochs, bs=args.batch_size)
        while not ok and quota_exhausted(msg):
            dead.add(acc["user"])
            quota[acc["user"]] = now
            print(f"  [quota] {acc['user']} out of GPU hours this week")
            nxt = next((a for a in pool if a["user"] not in dead
                        and inflight.get(a["user"], 0) < SLOTS_PER_ACCOUNT), None)
            if nxt is None:
                acc = None
                break
            acc = nxt
            ok, msg = push_run(acc, run, epochs=args.epochs, bs=args.batch_size)
        if acc is None:
            print(f"  [wait]  {run['id']:<14} no account with quota -- stays queued")
            save_state(st)
            continue
        st[run["id"]].update(status="pushed" if ok else "failed",
                             account=acc["user"], slug=slug_for(run["id"]),
                             pushed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        if not ok:
            st[run["id"]]["fails"] = st[run["id"]].get("fails", 0) + 1
        inflight[acc["user"]] = inflight.get(acc["user"], 0) + 1
        print(f"  [push{'' if ok else '-FAIL'}] {run['id']:<14} {acc['user']:<22} {msg}")

    save_state(st)
    cmd_status(args, st)


def cmd_status(args, st: dict | None = None) -> None:
    st = st if st is not None else load_state()
    q = build_queue()
    order = {"complete": 0, "running": 1, "queued": 2, "pushed": 3,
             "pending": 4, "failed": 5}
    counts: dict[str, int] = {}
    for r in q:
        s = st.get(r["id"], {"status": "pending"})
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    total = len(q)
    done = counts.get("complete", 0)
    print(f"\n  {done}/{total} complete  |  " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: order.get(x[0], 9))))


def cmd_reset(args) -> None:
    """Re-queue runs whose output is empty or whose training failed.

    Also clears the fail counter, because a fix to the payload invalidates the
    reason those runs failed -- otherwise three stale failures permanently retire
    a config that now works.
    """
    st = load_state()
    n = 0
    for rid, s in st.items():
        empty = not (OUTDIR / rid / "checkpoints").exists()
        if s.get("status") == "failed" or empty:
            s.update(status="pending", fails=0)
            s.pop("why", None)
            n += 1
            print(f"  requeued {rid}")
    save_state(st)
    print(f"\n{n} run(s) requeued")


def cmd_plan(args) -> None:
    q = build_queue()
    accs = accounts()
    print(f"{len(q)} runs, {len(accs)} accounts, {len(accs)*SLOTS_PER_ACCOUNT} slots\n")
    print(f"{'id':<16}{'model':<10}{'seed':>5}  extra")
    print("-" * 100)
    for r in q:
        print(f"{r['id']:<16}{r['model']:<10}{r['seed']:>5}  {' '.join(r['extra'])}")


def cmd_loop(args) -> None:
    while True:
        print(f"\n=== {time.strftime('%H:%M:%S')} ===")
        cmd_step(args)
        st = load_state()
        q = build_queue()
        if all(st.get(r["id"], {}).get("status") in ("complete", "failed") for r in q):
            print("\nall runs finished")
            return
        time.sleep(args.minutes * 60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["plan", "init", "step", "loop", "status",
                                       "reset"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--verify-file", default="stanford_pt_ov40m.npz",
                    help="a file that exists ONLY in the new payload, used to "
                         "confirm the upload actually landed")
    a = ap.parse_args()
    {"plan": cmd_plan, "init": cmd_init, "step": cmd_step,
     "loop": cmd_loop, "status": cmd_status, "reset": cmd_reset}[a.action](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


