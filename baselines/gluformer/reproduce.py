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
# `gluformer` is not a value `baselines/common/pretrain.py` accepts -- its
# choices are gluformer_tiny / gluformer_base. The paper reports the tiny
# variant (Table 7, "GluFormer-tiny"), so that is what this reproduces.
MODEL = 'gluformer_tiny'
SEEDS = (0, 1, 2)
# The corpus every arm in the paper trains on. Without this the baselines
# would load the no-overlap shards while GlucoPRISM trains on _ov40, and the
# comparison would no longer be paired -- see baselines/README.md.
SHARD = "_ov40"


def main() -> None:
    for seed in SEEDS:
        cmd = [sys.executable, str(COMMON / "pretrain.py"),
               "--model", MODEL, "--seed", str(seed),
               "--shard-suffix", SHARD]
        print(f"\n=== pretrain seed {seed}")
        if subprocess.run(cmd).returncode:
            sys.exit(f"pretraining failed at seed {seed}")

    for script in ("embed_cohorts.py", "probe_pretrained_models.py"):
        cmd = [sys.executable, str(SCRIPTS / script)]
        print(f"\n=== {script}")
        if subprocess.run(cmd).returncode:
            sys.exit(f"{script} failed")


if __name__ == "__main__":
    main()
