# GlucoPRISM

Code, weights and reproduction instructions for *Naming the Nuisance: Reserved
Subspaces Buy Control, Not Disentanglement, in CGM Foundation Models*.

This repository reproduces every number in the paper: the two released models,
our reproductions of four CGM foundation models on our own corpus, seven
zero-shot time-series baselines, and every ablation and analysis. It contains no
results — those are in the paper. What follows is how to regenerate them.

---

## What is here

```
baselines/     scripts that reproduce every baseline, ours and third-party
data/          frozen evaluation folds, corpus manifest, dataset fetchers
src/core/      the model, data and evaluation code
src/scripts/   corpus build, pretraining and evaluation drivers
src/ablations/ every ablation and analysis reported in the paper
weights/       one checkpoint per released model, plus our-corpus baselines
```

Each folder has its own `README.md` with the commands for that stage.

---

## Setup

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GLUCOPRISM_ROOT=$(pwd)                     # Windows: $env:GLUCOPRISM_ROOT=$PWD
```

Every script resolves paths from `GLUCOPRISM_ROOT`, defaulting to the repository
root, and writes intermediates to `$GLUCOPRISM_OUT` (default `./artifacts`). No
absolute paths are baked in.

A GPU is needed for pretraining (each run is roughly 35 minutes on a single
T4). Everything downstream of pretraining — probing, transfer, all ablations,
all tables and figures — runs on CPU from the released weights.

---

## The short path: verify the released models

If you only want to confirm the released checkpoints reproduce the paper's
embeddings and scores, you do not need to pretrain anything.

```bash
python src/scripts/v2_embed_runs.py            # embed the 4 downstream cohorts
python src/scripts/v2_score_npy.py             # frozen-fold probing, both levels
python src/scripts/final_table.py              # assemble the 14-cell table
python src/scripts/fd3_cross_dataset.py        # cross-cohort transfer
```

`data/splits_frozen.json` is the subject-to-fold assignment used for every
number in the paper. It was written before the first model was trained and is
shared by every model, which is what makes the comparisons paired. **Do not
regenerate it** if you intend to compare against our numbers.

---

## The full path: reproduce from raw data

### 1. Obtain the cohorts

The five pretraining cohorts and four downstream cohorts are public but each
requires its own access step (registration, a data-use agreement, or a direct
download). `data/README.md` lists each source and what it requires. We cannot
redistribute the raw data.

```bash
python src/scripts/download_datasets.py --help
```

### 2. Build the corpus

```bash
python src/scripts/build_corpus.py --day-overlap 40 --out corpus_v2fmt_ov40.npz
python src/scripts/build_v2_corpus.py
python src/scripts/freeze_splits.py            # only if starting a new benchmark
```

`build_corpus.py` enforces the pretrain/downstream subject split. It refuses to
write a shard whose subjects appear on both sides.

### 3. Pretrain

```bash
# the two released models
python src/scripts/run_v2port.py --corpus corpus_v2fmt_ov40.npz \
    --use-vib --w-vib 0.1 --seed 0                       # GlucoPRISM-C
python src/scripts/run_v2port.py --corpus corpus_v2fmt_ov40.npz \
    --use-vib --w-vib 0.1 --sim-bias measured --seed 0   # GlucoPRISM-E

# the backbone and the other CGM foundation models, on the same corpus
python src/scripts/run_pretrain.py --model glucofm
python src/scripts/run_pretrain.py --model cgm_jepa
python src/scripts/run_pretrain.py --model x_cgm_jepa
python src/scripts/run_pretrain.py --model gluformer_tiny
```

Run at least three seeds for anything you intend to compare. The seed standard
deviation on this benchmark is close to 1.0 ROC-AUC, so a single-seed
difference below that is not interpretable — including ours.

`src/scripts/orchestrate.py` fans these runs out across Kaggle GPU sessions if
you have accounts; set `GP_STAGE` and `GP_EXCLUDE`. It is convenience only,
not required.

### 4. Evaluate

```bash
python src/scripts/v2_embed_runs.py
python src/scripts/v2_score_npy.py
python src/scripts/final_table.py
python src/scripts/significance.py             # paired Wilcoxon + Holm
python src/scripts/fd3_cross_dataset.py
```

### 5. Regenerate every number, table and figure

```bash
python src/scripts/canonical_numbers.py        # one source of truth -> canonical.json
python src/scripts/make_all_tables.py          # every LaTeX table in the paper
python src/scripts/make_figures.py
python src/scripts/make_arch_figure.py
```

`canonical_numbers.py` recomputes every quantity the paper cites and writes
`canonical.tex`. The paper cites those macros rather than typed literals, so a
number cannot drift from the data behind it.

---

## Ablations

Everything reported as an ablation has a script in `src/ablations/`, including
the negative results. See `src/ablations/README.md` for the mapping from paper
section to script.

---

## Evaluation protocol

Reproduction depends on matching the protocol exactly:

- Logistic regression, `l2` penalty, `lbfgs`, `max_iter=1000`, **no inner search
  over `C`**, no class weighting.
- 5-fold subject-grouped cross-validation, 10 repeats, on the folds in
  `data/splits_frozen.json`.
- Window level **and** subject level reported together.
- Multiplicity correction over a family declared before results are read.

Any of these changed silently will produce numbers that are not comparable to
ours.

---

## Licence and citation

Code is MIT (`LICENSE`). Released weights are covered by the same licence.
Third-party checkpoints fetched by `baselines/fetch_baselines.py` carry their
own licences and are not redistributed here. See `CITATION.cff`.
