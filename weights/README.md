# Weights

One folder per model. Everything here is `safetensors`; nothing is pickled.

```
glucoprism_c/    GlucoPRISM-C   -- the released model, within-cohort
glucoprism_e/    GlucoPRISM-E   -- the released model, cross-cohort
glucofm/         GlucoFM        -- the backbone, trained on our corpus
cgm_jepa/        CGM-JEPA       -- trained on our corpus
x_cgm_jepa/      X-CGM-JEPA     -- trained on our corpus
gluformer/       GluFormer-tiny -- trained on our corpus
```

Third-party checkpoints used zero-shot (Chronos-2, MOMENT, Mantis, MantisV2,
CGMformer) are **not** here. They belong to their authors and keep their own
licences; `baselines/common/fetch_checkpoints.py` fetches and stages them.

## One seed per model

Every arm is trained at three seeds and the paper reports the seed-matched mean,
because the seed standard deviation on this benchmark is close to 1.0 ROC-AUC.
What ships is a single checkpoint per model:

| model | seed | why |
|---|---|---|
| GlucoPRISM-C | 5 | chosen by `src/ablations/select_release_seed.py` on the held-out rule |
| GlucoPRISM-E | 1 | best of its three seeds under the same rule |
| GlucoFM | 0 | first seed; the baseline is not seed-selected |

A single checkpoint will not equal the paper's task-averaged numbers, which are
means over seeds. Use `probe_frozen_folds.py` across seeds to reproduce a table
row; use these weights to embed data.

## Inference tensors only

`glucoprism_c/` and `glucoprism_e/` hold **506,550** parameters — the encoder and
the pooled Trait/State readout, and nothing else. The EMA target branch, the
projection heads and the device head are used only during pretraining and are
pruned. `src/scripts/export_inference_weights.py` verifies that the pruned
checkpoint reproduces the full model's embeddings exactly before writing.

Training checkpoints (`.pt`) are not shipped: they store every parameter twice,
because the online and EMA branches share storage.

## Verifying

```bash
python src/scripts/verify_released.py
```

re-embeds four cohorts with each released checkpoint and compares the result
against `data/reference_embeddings.json` — a statistical signature of the
embeddings these weights are supposed to produce (shape, moments, per-column
means, per-row norms). Any changed tensor moves those. The signature ships
rather than the arrays, because this repository holds code and weights, not
model output; it is ~5 KB instead of 2.3 MB, and unlike an exact hash it
tolerates the last-bit float differences a different BLAS or GPU introduces.

## Licence

MIT (`../LICENSE`), same as the code.
