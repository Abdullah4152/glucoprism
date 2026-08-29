"""Reproduce the MOMENT-large and MOMENT-small baseline end to end.

Zero-shot: the published checkpoint is used as released and never retrained on
our corpus, so this fetches, embeds and probes. The probe is identical to the
one every other model in the paper gets -- that is the whole point of running
seven third-party models ourselves.
"""
import subprocess
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parent.parent / "common"
# Must match the keys of embed_zeroshot.py::MODELS exactly (see cgmformer).
MODELS = ['MOMENT-small', 'MOMENT-large']


def main() -> None:
    for step, script in (("fetch", "fetch_checkpoints.py"),
                         ("embed", "embed_zeroshot.py"),
                         ("probe", "probe_zeroshot.py")):
        cmd = [sys.executable, str(COMMON / script), "--models", *MODELS]
        print(f"\n=== {step}: {' '.join(cmd[1:])}")
        if subprocess.run(cmd).returncode:
            sys.exit(f"{step} failed")


if __name__ == "__main__":
    main()
