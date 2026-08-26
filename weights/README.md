# Weights

One folder per model. Everything here was trained by us on our public-only
corpus, and ships as `safetensors` rather than pickle: a pickled checkpoint
executes arbitrary code on load, and `safetensors` is a plain tensor container
that cannot.

```
glucoprism_c/   released model, within-cohort
glucoprism_e/   released model, cross-cohort
glucofm/        our reproduction of the backbone
cgm_jepa/       our reproduction
x_cgm_jepa/     our reproduction
gluformer/      our reproduction (tiny variant)
manifest.json   which training checkpoint each file came from
```

## What is deliberately absent

Chronos-2, MOMENT, Mantis, MantisV2 and CGMformer are evaluated in the paper but
not stored here. We did not train them, and redistributing them would mean
shipping other people's artefacts under our licence.
`baselines/<model>/reproduce.py` fetches them from their original sources.

## Which seed ships

One checkpoint per released model, so the choice needs a rule rather than an
eye. The rule, fixed before the transfer axis was examined: **the seed with the
best mean over the 14 within-cohort cells at window level**, which is the
paper's primary protocol.

```bash
python src/ablations/select_release_seed.py
```

That recomputes every seed on all three axes and reports whether the axes the
rule ignores would have chosen differently. For one of the two models they
would, and the script says so — we did not change rules to make them agree.

## Verifying

```bash
python src/scripts/verify_released.py
```

If the released weights do not reproduce the embeddings the paper was scored on,
every number in the paper is unverifiable, so this is checked rather than
asserted.

## The readout

The released models are read as `[z_T || z_S]` with the 16-dimensional Sensor
block `z_A` discarded — a slice, needing no retraining and no device labels.
Reading the full 128-dimensional vector is a different model with different
numbers; the paper reports both.
