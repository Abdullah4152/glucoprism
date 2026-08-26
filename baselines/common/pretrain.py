"""Pretrain one of the reproduced models on the GlucoPRISM public-only corpus.

    python scripts/run_pretrain.py --model glucofm      --epochs 120
    python scripts/run_pretrain.py --model cgm_jepa     --epochs 101
    python scripts/run_pretrain.py --model x_cgm_jepa   --epochs 101   # needs --gluco-cache
    python scripts/run_pretrain.py --model gluformer_tiny --epochs 100

Defaults reproduce each paper's stated recipe; every override is logged into the
checkpoint so a run can be traced back to its exact configuration.
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

import numpy as np
sys.path.insert(0, str(ROOT / "src" / "core"))

from cgmkit.data.datasets import (CGMJEPADataset, GlucoFMDataset,  # noqa: E402
                                      GluFormerDataset, WindowShard,
                                      make_gluco_cache_keys)
from cgmkit.data.windows import densify  # noqa: E402
from cgmkit.train.pretrain import (pretrain_cgm_jepa, pretrain_glucofm,  # noqa: E402
                                       pretrain_gluformer)
from cgmkit.train.pretrain import pretrain_cqp  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
# Proposal Table 1, PT role. CGMacros and Hall are downstream-only by design
# (CGMacros' real Dexcom/Libre pairs are the held-out V1 validation set).
DEFAULT_PT = ["replacebg", "stanford", "shanghait2dm", "colas", "bigideas"]


def load_shards(datasets: list[str], role: str = "pt", suffix: str = "") -> WindowShard:
    # `suffix` selects a pretraining corpus variant: '' original (91% REPLACE-BG),
    # '_bal' uniform 12-window/subject cap, '_ctl' size-matched imbalanced control.
    paths = [PROCESSED / f"{d}_{role}{suffix if role == 'pt' else ''}.npz"
             for d in datasets]
    have = [p for p in paths if p.exists()]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"  [warn] missing shards (run scripts/build_corpus.py): {missing}")
    if not have:
        raise FileNotFoundError(f"no {role} shards found for {datasets}")
    shard = WindowShard(have)
    print(f"  corpus: {len(shard):,} windows from {len(set(shard.subjects))} subjects "
          f"({', '.join(p.stem for p in have)})")
    return shard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["glucofm", "cgm_jepa", "x_cgm_jepa",
                             "gluformer_tiny", "gluformer_base", "cqp"])
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_PT)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "checkpoints"))
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--gluco-cache", default=str(PROCESSED / "gluco_cache.pkl"))
    ap.add_argument("--build-gluco-cache", action="store_true",
                    help="precompute the Glucodensity view before training (x_cgm_jepa)")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--shard-suffix", default="",
                    help="pretraining corpus variant: '' original, '_bal' rebalanced, "
                         "'_ctl' size-matched imbalanced control")
    # --- GlucoFM extensions; all defaults reproduce the paper unchanged ---
    ap.add_argument("--use-cmp", action="store_true",
                    help="add L_CMP: predict the day's glucometrics from the masked view")
    ap.add_argument("--w-cmp", type=float, default=1.0)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--n-queries", type=int, default=16)
    ap.add_argument("--global-norm", action="store_true",
                    help="add a globally-scaled waveform view as a gated per-patch offset")
    # FD-7 patch geometry. Defaults are the papers' 24 x 12 non-overlapping
    # tokenisation; patchify() keeps that path bit-identical to the reshape.
    ap.add_argument("--patch-k", type=int, default=12,
                    help="grid steps per patch (K). 12 = 1 hour, the papers' value")
    ap.add_argument("--n-patches", type=int, default=24,
                    help="patches per window (P). P * stride must equal L = 288")
    ap.add_argument("--patch-stride", type=int, default=None,
                    help="steps between patch starts; default K (non-overlapping). "
                         "stride < K gives each patch K-stride steps of lookback")
    # FD-6 scaling arm.
    ap.add_argument("--d-stream", type=int, default=64,
                    help="state/event token width before fusion")
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--sensor-aug", type=float, default=0.0,
                    help="probability of rendering the anchor through the "
                         "FD-9-calibrated second-sensor transform (FD-8 V6/V7)")
    ap.add_argument("--width-scale", type=float, default=None,
                    help="FD-6: scale every feature width by this factor "
                         "(1.5 -> 2.10x params, 2.375 -> 4.97x). Overrides the "
                         "individual width flags for the stream feature dims.")
    a = ap.parse_args()

    def fm_kwargs(**extra):
        """Shared GlucoFMConfig kwargs so every model path gets the same geometry."""
        f = a.width_scale
        kw = dict(n_layers=a.n_layers, d_model=a.d_model, d_ff=a.d_ff,
                  d_stream=a.d_stream, n_heads=a.n_heads,
                  K=a.patch_k, P=a.n_patches, patch_stride=a.patch_stride,
                  global_norm=a.global_norm)
        if f:
            kw.update(d_model=int(128 * f), d_ff=int(256 * f), d_stream=int(64 * f),
                      state_wave_dim=int(64 * f), state_trend_dim=int(16 * f),
                      state_stats_dim=int(48 * f), event_wave_dim=int(48 * f),
                      event_roc_dim=int(48 * f), event_stats_dim=int(32 * f))
        kw.update(extra)
        return kw

    shard = load_shards(a.datasets, "pt", a.shard_suffix)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.model == "cqp":
        from common.models.cqp import CQPConfig, cqp_param_report
        from cgmkit.models.glucofm import GlucoFMConfig
        cfg = CQPConfig(fm=GlucoFMConfig(**fm_kwargs()),
                        n_queries=a.n_queries, d_query=a.d_model // a.n_queries,
                        w_cmp=a.w_cmp)
        print(f"  params: {cqp_param_report(cfg)}")
        ds = GlucoFMDataset(shard, augment_prob=not a.no_augment, seed=a.seed or 0,
                            sensor_aug=a.sensor_aug)
        ck = pretrain_cqp(ds, out, cfg=cfg, epochs=a.epochs or 120,
                          batch_size=a.batch_size, lr=a.lr or 1e-4, seed=a.seed or 0,
                          num_workers=a.num_workers, log_every=a.log_every)
    elif a.model == "glucofm":
        from cgmkit.models.glucofm import GlucoFMConfig
        # Defaults reproduce the paper exactly; every flag below is off unless asked.
        fm_cfg = GlucoFMConfig(**fm_kwargs(use_cmp=a.use_cmp, w_cmp=a.w_cmp))
        from cgmkit.models.glucofm import glucofm_param_report
        print(f"  params: {glucofm_param_report(fm_cfg)}  "
              f"K={fm_cfg.K} P={fm_cfg.P} stride={fm_cfg.stride}")
        ds = GlucoFMDataset(shard, augment_prob=not a.no_augment, seed=a.seed or 0,
                            sensor_aug=a.sensor_aug)
        ck = pretrain_glucofm(ds, out, cfg=fm_cfg,
                              epochs=a.epochs or 120, batch_size=a.batch_size,
                              lr=a.lr or 1e-4, seed=a.seed or 0,
                              num_workers=a.num_workers, log_every=a.log_every)

    elif a.model in ("cgm_jepa", "x_cgm_jepa"):
        cross = a.model == "x_cgm_jepa"
        cache = None
        if cross:
            cache = Path(a.gluco_cache)
            if a.build_gluco_cache or not cache.exists():
                from cgmkit.data.glucodensity import precompute
                dense = np.stack([densify(shard.data["glucose"][i], shard.data["mask"][i])
                                  for i in range(len(shard))])
                print(f"  precomputing Glucodensity for {len(dense)} windows ...")
                precompute(dense, make_gluco_cache_keys(shard), cache, n_workers=4)
        ds = CGMJEPADataset(shard, gluco_cache=cache, seed=a.seed or 43)
        ck = pretrain_cgm_jepa(ds, out, cross_view=cross, epochs=a.epochs or 101,
                               batch_size=a.batch_size, lr=a.lr or 1e-4,
                               seed=a.seed or 43, num_workers=a.num_workers,
                               log_every=a.log_every)

    else:
        variant = a.model.split("_")[1]
        ds = GluFormerDataset(shard)
        ck = pretrain_gluformer(ds, out, variant=variant,
                                epochs=a.epochs or (100 if variant == "tiny" else 76),
                                batch_size=a.batch_size, lr=a.lr, seed=a.seed or 43,
                                num_workers=a.num_workers, log_every=a.log_every)

    meta = {"model": a.model, "datasets": a.datasets,
            "shard_suffix": a.shard_suffix, "windows": len(shard),
            "subjects": len(set(shard.subjects)), "epochs": a.epochs,
            "batch_size": a.batch_size, "lr": a.lr, "seed": a.seed,
            "patch_k": a.patch_k, "n_patches": a.n_patches,
            "patch_stride": a.patch_stride, "width_scale": a.width_scale,
            "d_model": a.d_model, "d_ff": a.d_ff, "d_stream": a.d_stream,
            "n_heads": a.n_heads, "n_layers": a.n_layers,
            "checkpoint": str(ck)}
    (out / f"{a.model}_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
