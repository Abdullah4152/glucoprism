# Baselines

Two kinds of baseline appear in the paper, and they are reproduced differently.

## 1. CGM foundation models, pretrained by us on our corpus

GlucoFM, CGM-JEPA, X-CGM-JEPA and GluFormer-tiny are reimplemented in
`src/core/glucoprism/models/` and pretrained on the same corpus, with the same
schedule and the same frozen folds as our own models. This is what makes the
comparison paired: every model sees identical windows and is probed on
identical splits.

```bash
python ../src/scripts/run_pretrain.py --model glucofm        --seed 0
python ../src/scripts/run_pretrain.py --model cgm_jepa       --seed 0
python ../src/scripts/run_pretrain.py --model x_cgm_jepa     --seed 0
python ../src/scripts/run_pretrain.py --model gluformer_tiny --seed 0
```

Run three seeds for anything you intend to compare.

Our checkpoints for these are in `../weights/` (`*-ours.safetensors`), so you
can skip pretraining and go straight to scoring.

### Checking the reimplementations

```bash
python verify_cgm_jepa.py
```

This is a bit-exactness regression test against the reference implementation.
It guards the blocks shared between models: a change to `blocks.py` that
silently alters CGM-JEPA would invalidate every comparison in the paper, and
this catches it.

Our GlucoFM reproduction lands at 720,278 trainable parameters against the
720,241 reported in the original paper, and preserves the published ordering of
models on the benchmark. If your reproduction does not, something is wrong
before you start comparing.

## 2. Zero-shot time-series foundation models

Chronos-2, Chronos-2-small, MOMENT-large, MOMENT-small, Mantis, MantisV2 and
CGMformer are used as published, never retrained on our corpus. Their weights
are **not** redistributed here — they belong to their authors and carry their
own licences.

```bash
python fetch_baselines.py          # downloads from the original sources
python run_baselines.py            # embeds the four downstream cohorts
python score_baselines.py          # frozen-fold probing, identical protocol
```

`fetch_baselines.py` needs network access and, for some checkpoints, a
HuggingFace token in the usual environment variable. No credential is stored in
this repository.

## The protocol that makes these comparable

Every baseline is embedded frozen and probed with exactly the protocol in the
root `README.md` — same regression settings, same folds, same two reporting
levels. The point of reproducing seven third-party models under one probe is
that CGM papers rarely report them, and when the protocol differs the numbers
are not comparable. If you change the probe, change it for every model.
