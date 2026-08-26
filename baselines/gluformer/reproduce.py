"""Reproduce the GluFormer (tiny) baseline end to end.

Pretrained by us on our own corpus with the same schedule and the same frozen
folds as every other model, which is what makes the comparison paired.

Run at least three seeds for anything you intend to compare: the seed standard
deviation on this benchmark is close to 1.0 ROC-AUC, so a single-seed
difference below that is not interpretable.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "baselines" / "common"
SCRIPTS = ROOT / "src" / "scripts"
MODEL = 'gluformer'
SEEDS = (0, 1, 2)

for seed in SEEDS:
    cmd = [sys.executable, str(COMMON / "pretrain.py"),
           "--model", MODEL, "--seed", str(seed)]
    print(f"\n=== pretrain seed {seed}")
    if subprocess.run(cmd).returncode:
        sys.exit(f"pretraining failed at seed {seed}")

for script in ("embed_cohorts.py", "probe_pretrained_models.py"):
    cmd = [sys.executable, str(SCRIPTS / script)]
    print(f"\n=== {script}")
    if subprocess.run(cmd).returncode:
        sys.exit(f"{script} failed")
