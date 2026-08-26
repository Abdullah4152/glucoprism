# Core

Library code. These modules are imported by `../scripts/` and `../ablations/`;
none of them is a program you run directly.

```
glucoprism/
  models/     GlucoFM backbone, CGM-JEPA, GluFormer, blocked pooling, CQP
  data/       harmonisation, windowing, labels, augmentation, protocol views
  eval/       the frozen-fold probe and subject-level aggregation
  train/      pretraining loops
reference/    the vendored implementation that trained the two released models
```

## glucoprism/models/

`glucofm.py` is the backbone every model here builds on: grid alignment with a
preserved observation mask, patchification with circadian encoding, a learnable
causal mask-aware Gaussian filter splitting a slow state stream from a residual
event stream, and a context encoder trained against an EMA target.

Two things in this file are load-bearing and easy to break:

- **Aligned mg/dL versus normalised values.** The predictive objectives operate
  on aligned mg/dL; the pooled representation is normalised. Conflating them
  makes the encoder level-blind — measured on a +60 mg/dL shift, correct
  handling moves the representation by 0.136 and the conflated variant by
  8.7e-06. Nothing in the loss curves reveals it, and every endpoint in this
  benchmark depends on absolute level.
- **`patchify` under lookback.** With no lookback, `P * K` must equal `L` and
  patches tile the day exactly. With lookback, patches overlap and the stride,
  not `K`, governs masking and circadian indexing. `../ablations/test_patch_geometry.py`
  checks both cases.

`blocks.py` is shared between models. Changing it silently alters the
reproductions, which is why `../../baselines/verify_cgm_jepa.py` is a
bit-exactness regression test rather than an approximate one.

## glucoprism/data/

`augment.py` holds the synthetic second-sensor generator. Its constants are
*measured* on real paired windows rather than assumed — prior synthetic views in
this literature carry no calibration offset at all, where the real one is
-31.1 mg/dL with 43 of 44 subjects agreeing in sign. `views.py` builds the
paired-sensor and repeated-day views.

Note that array index 0 is a window's own first reading, not midnight. Two
same-day windows from different devices generally start at different clock
times, and comparing them index-wise without aligning on `start_idx` produces
silently wrong results.

## glucoprism/eval/

`probe.py` implements the protocol: logistic regression, `l2`, `lbfgs`,
`max_iter=1000`, no inner search over `C`, no class weighting, on folds read
from disk rather than regenerated. `aggregate.py` does subject-level pooling.

## reference/

Vendored unchanged, and byte-identical to the tree that trained the released
checkpoints. Its package is also named `glucoprism`, so it must go on
`sys.path` *before* ours and only inside the runs that need it — see
`../scripts/run_v2port.py`. Do not merge the two trees.
