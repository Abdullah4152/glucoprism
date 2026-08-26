# Core

Library code. Imported by `../scripts/` and `../ablations/`; nothing here is a
program you run.

```
cgmkit/            our toolkit
  models/          blocks.py, glucofm.py (backbone), prism.py (this paper)
  data/            harmonisation, windowing, labels, augmentation, views
  eval/            frozen-fold probe and subject-level aggregation
  train/           pretraining loops
released_model/    vendored: the code that trained the released checkpoints
  glucoprism/      blocked pooling, the three objectives, the VIB, sensor_sim
  glucofm/         the backbone it wraps
```

## cgmkit/models/

Three files, deliberately. `blocks.py` holds primitives shared with the
baselines — changing it silently alters them, which is why
`baselines/common/verify_bit_exact.py` is a bit-exactness test rather than an
approximate one. `glucofm.py` is the backbone. `prism.py` is this paper's model.

Two things in `glucofm.py` are easy to break:

- **Aligned mg/dL versus normalised values** — see `../README.md`.
- **`patchify` under lookback.** With no lookback, `P * K` must equal `L` and
  patches tile the day exactly. With lookback they overlap, and the *stride*,
  not `K`, governs masking and circadian indexing.
  `../ablations/patchify_unit_tests.py` covers both.

## cgmkit/data/

`augment.py` holds the synthetic second-sensor generator. Its constants are
**measured** on real paired windows, not assumed: prior synthetic views in this
literature carry no calibration offset where the real one is -31.1 mg/dL with 43
of 44 subjects agreeing in sign. `views.py` builds the paired-sensor and
repeated-day views.

Array index 0 is a window's own first reading, **not midnight**. Two same-day
windows from different devices generally start at different clock times;
comparing them index-wise without aligning on `start_idx` gives silently wrong
answers.

## released_model/

Vendored unchanged and byte-identical to the tree that produced the released
checkpoints. It is placed on `sys.path` only inside the runs that need it — see
`../scripts/pretrain_glucoprism.py`.
