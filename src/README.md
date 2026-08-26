# Source

```
core/        library code -- imported, not run
scripts/     experiment drivers -- run these
ablations/   every ablation and diagnostic in the paper
```

## Why `core` has two trees

`core/cgmkit/` is our toolkit: the data pipeline, the evaluation probe, the
GlucoFM backbone and our own model.

`core/released_model/` is the implementation that actually trained the two
released checkpoints, vendored unchanged. `scripts/pretrain_glucoprism.py`
imports it rather than reimplementing it, because a reimplementation risks
silent divergence from the weights we ship.

The two used to collide: both exported a package named `glucoprism`, which
meant `sys.path` ordering decided which one you got. Ours is renamed `cgmkit`
so the collision cannot happen. The vendored tree keeps its original package
names (`glucoprism`, `glucofm`) because changing them would make it no longer a
faithful copy of what trained the weights.

## What is deliberately not here

No other paper's architecture. CGM-JEPA, X-CGM-JEPA, GluFormer and the
Glucodensity CQP model live in `baselines/common/models/`, next to the scripts
that reproduce them. `core/cgmkit/models/` holds three files: the shared
primitives, the backbone this paper builds on, and this paper's model.

## Paths

```python
ROOT   = os.environ.get("GLUCOPRISM_ROOT", <repo root>)
OUTDIR = os.environ.get("GLUCOPRISM_OUT",  ROOT / "artifacts")
```

Scripts put `src/core` and `baselines` on `sys.path` themselves, so `import
cgmkit` and `import common.models...` work from anywhere.

## One detail to read before changing anything

The predictive objectives operate on aligned mg/dL values while the pooled
representation is normalised. Conflating them makes the encoder level-blind: on
a +60 mg/dL shift, correct handling moves the representation by 0.136 and the
conflated variant by 8.7e-06. The model still trains and converges — it is
simply unable to represent hyperglycemia, which is fatal for every endpoint
here, and invisible in the loss curves. See `core/cgmkit/models/glucofm.py`.
