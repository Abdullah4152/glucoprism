# Scripts

The reproduction path, in order. Every script takes `--help`.

## 1. Data and corpus

| Script | What it does |
|---|---|
| `download_datasets.py` | fetch/stage the public cohorts (see `data/README.md`) |
| `profile_datasets.py` | per-cohort device, sampling rate, coverage summary |
| `build_corpus.py` | window, align to the 288-point grid, write pretraining shards |
| `build_v2_corpus.py` | pack the shards into the format the v2 trainer reads |
| `freeze_splits.py` | materialise the subject-to-fold assignment |
| `corpus_summary.py` | subject and hour counts for the corpus table |

`build_corpus.py` enforces the pretrain/downstream subject split and refuses to
write a shard whose subjects appear on both sides. An earlier version of this
pipeline leaked downstream subjects into pretraining; the guard exists because
of that, and it is checked on subject counts rather than trusted.

**Do not regenerate `data/splits_frozen.json`** if you want to compare against
our numbers. It was written before the first model was trained and is shared by
every model in the paper; that is what makes every comparison paired, and paired
tests are roughly an order of magnitude more sensitive here than unpaired ones.

## 2. Pretraining

| Script | What it does |
|---|---|
| `run_v2port.py` | trains the two released models |
| `run_pretrain.py` | trains the four CGM foundation-model reproductions |
| `run_prism.py` | trains the proposal-as-specified variant |
| `run_finetune.py` | optional end-to-end fine-tuning |
| `orchestrate.py` | fans runs out across Kaggle GPU sessions (convenience only) |

The two released configurations:

```bash
python run_v2port.py --corpus corpus_v2fmt_ov40.npz --use-vib --w-vib 0.1 --seed 0
python run_v2port.py --corpus corpus_v2fmt_ov40.npz --use-vib --w-vib 0.1 \
                     --sim-bias measured --seed 0
```

Useful flags: `--no-protocol` switches the factorization objectives off while
keeping every other component, `--d-sensor` sets the Sensor block width, and
`--w-vib` sets the KL price per nat.

`orchestrate.py` is optional. If you use it, note that it verifies each run
produced a checkpoint rather than trusting the scheduler's exit code — the
Kaggle CLI reports success on several real failure modes, and a stale payload
once killed 12 of 18 runs while every one of them was reported complete.

## 3. Evaluation

| Script | What it does |
|---|---|
| `v2_embed_runs.py` | embed the four downstream cohorts, per block |
| `embed_subset.py` | same, restricted to runs matching a substring |
| `v2_score_npy.py` | frozen-fold probing, window and subject level |
| `score_runs.py` | probing for the non-v2 models |
| `final_table.py` | assemble the 14-cell table |
| `significance.py` | paired Wilcoxon, Holm correction, Cliff's delta |
| `significance_transfer.py` | the same on the transfer axis |
| `fd3_cross_dataset.py` | cross-cohort transfer, all twelve directions |
| `fd3_drop_za.py` | the Sensor-block deletion, per level |

## 4. Numbers, tables, figures

| Script | What it does |
|---|---|
| `canonical_numbers.py` | recompute every quantity the paper cites |
| `make_all_tables.py` | every LaTeX table |
| `make_figures.py` | figures 2-8 |
| `make_arch_figure.py` | figure 1, the architecture |
| `lint_tex.py` | checks the paper for undefined macros and typed literals |

`canonical_numbers.py` is the single source of truth. It writes `canonical.json`
and `canonical.tex`, and the paper cites those macros instead of typed numbers.
This exists because an audit found 25 places where our own documents stated
different values for the same quantity — all of them caused by numbers being
typed into prose by hand.

## 5. Weights

| Script | What it does |
|---|---|
| `export_inference_weights.py` | prune a training checkpoint to inference tensors |
| `export_all_safetensors.py` | convert checkpoints to `safetensors` |
| `param_report.py` | trainable and total parameter counts |

Note that these checkpoints store every parameter twice, because the online and
EMA branches share storage. `safetensors` refuses aliased tensors, so the
exporters deduplicate on the storage pointer before writing.
