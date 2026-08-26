"""Pretrain GlucoPRISM on the public-only corpus.

    python scripts/run_prism.py                              # faithful defaults
    python scripts/run_prism.py --hold-out --align-on block  # a sweep arm
    python scripts/run_prism.py --seed 1

Every proposal knob is exposed so a sweep arm is a command line, not an edit.
Defaults are the proposal's stated values (lambda = 1.0, dims 64/48/16), because
Phase 2 runs the method exactly as written before anything is tuned.

`--hold-out` drops the held-out pretraining subjects carved by
`scripts/freeze_splits.py`. Use it for every sweep arm; omit it for the final
headline runs. Downstream cohorts never inform model selection either way.
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
import sys
from pathlib import Path
sys.path.insert(0, str(ROOT / "src" / "core"))

from cgmkit.data.datasets import WindowShard          # noqa: E402
from cgmkit.data.views import PrismDataset            # noqa: E402
from cgmkit.models.glucofm import GlucoFMConfig       # noqa: E402
from cgmkit.models.prism import PrismConfig, prism_param_report  # noqa: E402
from cgmkit.train.pretrain import pretrain_prism      # noqa: E402

PROCESSED = ROOT / "data" / "processed"
DEFAULT_PT = ["replacebg", "stanford", "shanghait2dm", "colas", "bigideas"]
HOLDOUT = PROCESSED / "pretrain_holdout.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_PT)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "checkpoints"))
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--hold-out", action="store_true",
                    help="drop the held-out pretraining subjects (use for sweeps)")
    ap.add_argument("--shard-suffix", default="",
                    help="pretraining shard variant: '' = original (91%% REPLACE-BG), "
                         "'_bal' = uniform 12/subject cap, '_ctl' = size-matched "
                         "imbalanced control")
    # --- proposal knobs -------------------------------------------------
    ap.add_argument("--align-on", choices=["head", "block"], default="head")
    ap.add_argument("--lambda-sensor", type=float, default=1.0)
    ap.add_argument("--lambda-day", type=float, default=1.0)
    ap.add_argument("--lambda-indep", type=float, default=1.0)
    ap.add_argument("--beta-day", type=float, default=1.0)
    ap.add_argument("--dims", nargs=3, type=int, default=[64, 48, 16],
                    metavar=("dT", "dS", "dA"))
    ap.add_argument("--adversarial", action="store_true",
                    help="gradient-reversal zA head (E9 ablation)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--global-norm", action="store_true",
                    help="globally-scaled waveform view as a gated per-patch offset")
    # FD-6/FD-7: geometry and width, mirroring run_pretrain.py so a PRISM run and
    # a GlucoFM run at the same setting are genuinely the same backbone.
    ap.add_argument("--patch-k", type=int, default=12)
    ap.add_argument("--n-patches", type=int, default=24)
    ap.add_argument("--patch-stride", type=int, default=None)
    ap.add_argument("--width-scale", type=float, default=None,
                    help="scale every feature width (1.5 -> 2.10x params, "
                         "2.375 -> 4.97x). Block dims scale with it.")
    ap.add_argument("--head-layers", type=int, default=2, choices=[1, 2],
                    help="projection-head depth. A single linear head can only "
                         "enforce a LINEAR invariance, which is stricter than "
                         "Eq. 2 asks for; 2 is the SSL standard.")
    ap.add_argument("--use-cmp", action="store_true")
    ap.add_argument("--w-cmp", type=float, default=1.0)
    a = ap.parse_args()

    f = a.width_scale
    fm_kw = dict(global_norm=a.global_norm, K=a.patch_k, P=a.n_patches,
                 patch_stride=a.patch_stride, use_cmp=a.use_cmp, w_cmp=a.w_cmp)
    dims = list(a.dims)
    if f:
        fm_kw.update(d_model=int(128 * f), d_ff=int(256 * f), d_stream=int(64 * f),
                     n_heads=8 if f > 2 else 6,
                     state_wave_dim=int(64 * f), state_trend_dim=int(16 * f),
                     state_stats_dim=int(48 * f), event_wave_dim=int(48 * f),
                     event_roc_dim=int(48 * f), event_stats_dim=int(32 * f))
        # Blocked pooling must partition the token exactly, so the 4:3:1 split
        # scales with d_model. Any remainder goes to zT, the largest block.
        d = fm_kw["d_model"]
        dims = [d * 4 // 8, d * 3 // 8, d // 8]
        dims[0] += d - sum(dims)

    cfg = PrismConfig(
        fm=GlucoFMConfig(**fm_kw),
        d_trait=dims[0], d_state=dims[1], d_sensor=dims[2],
        align_on=a.align_on,
        lambda_sensor=a.lambda_sensor, lambda_day=a.lambda_day,
        lambda_indep=a.lambda_indep, beta_day=a.beta_day,
        adversarial_sensor=a.adversarial,
    )

    paths = [PROCESSED / f"{d}_pt{a.shard_suffix}.npz" for d in a.datasets]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"  [warn] missing shards: {missing}")
    shard = WindowShard([p for p in paths if p.exists()])

    excl = None
    if a.hold_out:
        if not HOLDOUT.exists():
            raise SystemExit("--hold-out needs data/processed/pretrain_holdout.json "
                             "(run scripts/freeze_splits.py first)")
        excl = json.loads(HOLDOUT.read_text(encoding="utf-8"))["subjects"]

    ds = PrismDataset(shard, cfg=cfg.fm, seed=a.seed,
                      augment_anchor=not a.no_augment, exclude_subjects=excl)

    n_v2 = sum(1 for i in range(len(ds)) if len(ds.v2[i]) > 0)
    print(f"  corpus: {len(ds):,} windows from {len(set(ds.subjects))} subjects"
          f"{f' (held out {len(excl)})' if excl else ''}")
    print(f"  V2-eligible windows: {n_v2:,} / {len(ds):,} "
          f"({100 * n_v2 / max(len(ds), 1):.1f}%)")
    print(f"  params: {prism_param_report(cfg)}")
    print(f"  align_on={cfg.align_on}  lambdas=({cfg.lambda_sensor}, {cfg.lambda_day}, "
          f"{cfg.lambda_indep})  beta={cfg.beta_day}  dims=({cfg.d_trait},"
          f"{cfg.d_state},{cfg.d_sensor})  adversarial={cfg.adversarial_sensor}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = pretrain_prism(ds, out, cfg=cfg, epochs=a.epochs, batch_size=a.batch_size,
                        lr=a.lr, seed=a.seed, num_workers=a.num_workers,
                        log_every=a.log_every)

    (out / "prism_run.json").write_text(json.dumps({
        "model": "prism", "datasets": a.datasets, "held_out": bool(a.hold_out),
        "shard_suffix": a.shard_suffix,
        "windows": len(ds), "subjects": len(set(ds.subjects)),
        "v2_eligible": n_v2,
        "epochs": a.epochs, "batch_size": a.batch_size, "seed": a.seed,
        "config": cfg.to_dict(), "checkpoint": str(ck),
    }, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
