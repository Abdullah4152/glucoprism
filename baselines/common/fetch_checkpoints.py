"""Download the GlucoFM baseline checkpoints we do not already have.

All are used FROZEN and zero-shot -- none is retrained on our corpus, exactly as
GlucoFM App. B.1/B.2 specifies.
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\hf_cache")
from huggingface_hub import snapshot_download  # noqa: E402

REPOS = {
    "MOMENT-small": "AutonLab/MOMENT-1-small",
    "MOMENT-large": "AutonLab/MOMENT-1-large",
    "Mantis": "paris-noah/Mantis-8M",
    "MantisV2": "paris-noah/MantisV2",
    "Chronos-2": "amazon/chronos-2",
    "Chronos-2-small": "autogluon/chronos-2-small",
}

for name, repo in REPOS.items():
    try:
        p = snapshot_download(repo)
        mb = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6
        print(f"  {name:<18} {repo:<32} {mb:>8.1f} MB", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  {name:<18} {repo:<32} FAILED {type(e).__name__}: "
              f"{str(e)[:80]}", flush=True)
