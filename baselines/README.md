# Baselines

One folder per baseline. Each has a `README.md` and a `reproduce.py` that runs
that model end to end.

```
glucofm/     cgm_jepa/   x_cgm_jepa/   gluformer/     pretrained by us
cgmformer/   chronos/    moment/       mantis/        used zero-shot
common/      shared drivers and the baseline architectures
```

## Two kinds of baseline

**Pretrained by us.** GlucoFM, CGM-JEPA, X-CGM-JEPA and GluFormer-tiny are
reimplemented and pretrained on our corpus, with the same schedule and the same
frozen folds as our own models. That is what makes the comparison paired: every
model sees identical windows and is probed on identical splits. Our checkpoints
are in `../weights/`, so pretraining is optional.

**Used zero-shot.** Chronos-2, MOMENT, Mantis, MantisV2 and CGMformer are used
as published and never retrained. Their weights are **not** in this repository —
they belong to their authors and carry their own licences.
`reproduce.py` fetches them from the original sources.

## common/

```
models/               CGM-JEPA, GluFormer and CQP architectures
pretrain.py           shared driver for the models we pretrain
fetch_checkpoints.py  downloads third-party checkpoints
embed_zeroshot.py     embeds the four downstream cohorts
probe_zeroshot.py     frozen-fold probing, identical protocol
verify_bit_exact.py   regression test against the reference implementation
```

The architectures live here rather than in `core` because they are other
people's models; `core` holds only this paper's model and the backbone it
builds on.

## Why we ran seven third-party models ourselves

CGM papers rarely report general-purpose time-series foundation models, and when
they do the probe usually differs, which makes the numbers incomparable. Every
model here is embedded frozen and probed with exactly the protocol in the root
`README.md`. If you change the probe, change it for every model.
