# Scripts

The reproduction path, in order. Every script takes `--help`.

## 1. Data and corpus

| Script | What it does |
|---|---|
| `download_datasets.py` | fetch and stage the public cohorts |
| `profile_datasets.py` | per-cohort device, sampling rate, coverage |
| `build_corpus.py` | window, align to the 288-point grid, write shards |
| `pack_corpus_for_trainer.py` | pack shards into the trainer's format |
| `freeze_evaluation_folds.py` | materialise the subject-to-fold assignment |
| `corpus_summary.py` | subject and hour counts, for verification |

**Do not run `freeze_evaluation_folds.py`** if you want to compare against our
numbers — `data/processed/splits_frozen.json` already holds the assignment every
model in the paper was scored on.

## 2. Pretraining

| Script | What it does |
|---|---|
| `pretrain_glucoprism.py` | trains the two released models |
| `pretrain_proposal_variant.py` | trains the proposal exactly as specified |
| `finetune_end_to_end.py` | optional end-to-end fine-tuning |
| `kaggle_orchestrator.py` | fans runs across Kaggle GPU sessions (optional) |
| `kaggle_submit.py` | submits a single run |

```bash
# GlucoPRISM-C
python pretrain_glucoprism.py --corpus corpus_v2fmt_ov40.npz --use-vib \
       --w-vib 0.1 --seed 0
# GlucoPRISM-E -- note the bottleneck weight differs
python pretrain_glucoprism.py --corpus corpus_v2fmt_ov40.npz --use-vib \
       --w-vib 1.0 --sim-bias measured --seed 0
```

Useful flags: `--no-protocol` switches the factorization objectives off while
keeping every other component, `--d-sensor` sets the Sensor block width,
`--w-vib` sets the KL price per nat, `--sim-bias measured` uses the measured
calibration offset in the paired-sensor view.

The released checkpoints' `config.json` is the authority on which value each
model used: C is `w_vib: 0.1`, E is `w_vib: 1.0`.

Baselines are in `baselines/<model>/reproduce.py`.

If you use `kaggle_orchestrator.py`, note that it verifies each run produced a
checkpoint rather than trusting the scheduler's exit code. The Kaggle CLI
reports success on several real failure modes, and a stale payload once killed
12 of 18 runs while every one was reported complete.

## 3. Embedding and probing

| Script | What it does |
|---|---|
| `embed_cohorts.py` | embed the four downstream cohorts, per block |
| `embed_cohorts_subset.py` | same, restricted to runs matching a substring |
| `probe_frozen_folds.py` | frozen-fold probing, window and subject level |
| `probe_pretrained_models.py` | probing for the non-blocked models |
| `evaluate_models.py` | end-to-end evaluation driver |
| `evaluate_external.py` | zero-shot evaluation on a cohort of your own |
| `collect_results.py` | gather run outputs into one table |

`evaluate_external.py` is the driver behind the paper's external-validation
table. It is cohort-agnostic by construction: it takes a window shard and a
subject-level label `.csv` and knows nothing about any particular dataset, so it
runs on the paper's fifth cohort and on yours identically. The paper's cohort is
governed by a data use agreement and is not distributed here, and neither is its
loader — see the repository README.

## 4. Headline results

| Script | What it does |
|---|---|
| `build_results_table.py` | assemble the 14-cell table |
| `significance_within_cohort.py` | paired Wilcoxon, Holm, Cliff's delta |
| `significance_transfer.py` | the same on the transfer axis |
| `cross_cohort_transfer.py` | all twelve transfer directions |
| `sensor_block_deletion.py` | the Sensor-block deletion, per level |

## 5. Numbers, tables, figures

| Script | What it does |
|---|---|
| `build_derived_inputs.py` | the aggregations the table generator consumes |
| `canonical_numbers.py` | recompute every quantity the paper cites |
| `make_all_tables.py` | **every table in the paper** |
| `make_paper_figures.py` | every result figure in the paper |
| `make_tables.py` | a reduced subset; superseded by `make_all_tables.py` |
| `make_architecture_figure.py` | the architecture figure |
| `lint_paper.py` | undefined macros and hand-typed literals in the paper |

Order matters: `build_derived_inputs.py` → `canonical_numbers.py` →
`make_all_tables.py` → `make_paper_figures.py`. The first writes the summary
tables the others read; the second writes `canonical.json`, which the table
generator loads.

Both generators write under `GLUCOPRISM_OUT`. Set `GLUCOPRISM_TEX_OUT` and
`GLUCOPRISM_FIG_OUT` to also write into a paper directory.

`canonical_numbers.py` is the single source of truth. It writes
`canonical.json` and `canonical.tex`, and the paper cites those macros instead
of typed numbers. It exists because an audit found 25 places where our own
documents stated different values for the same quantity — every one caused by a
number typed into prose by hand.

## 6. Weights

| Script | What it does |
|---|---|
| `export_inference_weights.py` | prune a checkpoint to inference tensors |
| `export_safetensors.py` | convert checkpoints to `safetensors` |
| `load_released.py` | load a released checkpoint |
| `verify_released.py` | released weights reproduce the paper's embeddings |
| `parameter_counts.py` | trainable and total parameter counts |

These checkpoints store every parameter twice, because the online and EMA
branches share storage. `safetensors` refuses aliased tensors, so the exporters
deduplicate on the storage pointer before writing.
