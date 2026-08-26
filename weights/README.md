# Weights

All checkpoints here were trained by us on our public-only corpus. They ship as
`safetensors` rather than pickles: a pickled checkpoint executes arbitrary code
on load, and `safetensors` is a plain tensor container that cannot.

```
glucoprism-c.safetensors        released model, within-cohort
glucoprism-e.safetensors        released model, cross-cohort
glucofm-ours.safetensors        our reproduction of the backbone
cgm-jepa-ours.safetensors       our reproduction, our corpus
x-cgm-jepa-ours.safetensors     our reproduction, our corpus
gluformer-tiny-ours.safetensors our reproduction, our corpus
manifest.json                   which training checkpoint each file came from
```

## What is deliberately absent

Chronos-2, MOMENT, Mantis, MantisV2 and CGMformer are evaluated in the paper but
**not** included here. They are third-party checkpoints that we did not train;
redistributing them would mean shipping other people's artefacts under our
licence. `baselines/fetch_baselines.py` downloads them from their original
sources instead.

## Which seed ships

One checkpoint per released model, so the choice needs a rule rather than an
eye. The rule is: **the seed with the best mean over the 14 within-cohort cells
at window level**, which is the paper's primary protocol. It was fixed before
looking at the transfer axis.

To audit the choice, recompute every seed on both axes:

```bash
python src/ablations/select_release_seed.py
```

That prints window, subject and transfer means for every seed of both models
and names the seed the rule selects, so you can see whether the rule and the
alternatives disagree. They do for one of the two models, and the script says
so — we did not switch rules to make them agree.

## Loading

```python
from safetensors.torch import load_file
sd = load_file("weights/glucoprism-c.safetensors")
```

`src/scripts/export_inference_weights.py` shows the encoder/pool reconstruction,
and the released files are pruned to inference tensors only — the EMA target
branch, optimiser state and projection heads used solely by the training
objectives are not included, because none of them is needed to produce an
embedding.

## Verifying

```bash
python src/scripts/v2_embed_runs.py
python src/scripts/v2_score_npy.py
```

If the released weights do not reproduce the embeddings the paper was scored
on, every number in the paper is unverifiable, so this is checked rather than
asserted. The check compares embeddings tensor-by-tensor across every
model-seed-cohort combination.

## The readout

The released models are read as `[z_T || z_S]` with the 16-dimensional Sensor
block `z_A` discarded. That is a slice, not a retraining step, and it needs no
device labels at test time. Reading the full 128-dimensional vector instead is
a different model with different numbers — the paper reports both.
