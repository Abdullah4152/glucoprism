# GlucoPRISM

Code, weights and reproduction instructions for *Naming the Nuisance: Reserved
Subspaces Buy Control, Not Disentanglement, in CGM Foundation Models*.

This repository reproduces every number in the paper. It contains no results —
those are in the paper. What follows is how to regenerate them.

```
src/core/cgmkit/          our toolkit: data pipeline, evaluation probe, our model
src/core/released_model/  the implementation that trained the released weights
src/scripts/              experiment drivers, named for what they do
src/ablations/            every ablation, named for the question it answers
baselines/<model>/        one folder per baseline, with how to reproduce it
weights/<model>/          one folder per model, with its checkpoint
data/                     frozen evaluation folds and corpus manifest
```

Every folder has a `README.md` with the commands for that stage.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GLUCOPRISM_ROOT=$(pwd)                  # Windows: $env:GLUCOPRISM_ROOT=$PWD
```

Scripts resolve paths from `GLUCOPRISM_ROOT` (default: the repository root) and
write intermediates to `GLUCOPRISM_OUT` (default: `./artifacts`). No absolute
paths are baked in.

A GPU is needed for pretraining — roughly 35 minutes per run on a single T4.
Everything downstream (probing, transfer, ablations, tables, figures) runs on
CPU from the released weights.

## Verify the released models without training anything

```bash
python src/scripts/embed_cohorts.py          # embed the four downstream cohorts
python src/scripts/probe_frozen_folds.py     # frozen-fold probing, both levels
python src/scripts/build_results_table.py    # the 14-cell table
python src/scripts/cross_cohort_transfer.py  # all twelve transfer directions
python src/scripts/verify_released.py        # weights reproduce paper embeddings
```

`data/splits_frozen.json` is the subject-to-fold assignment behind every number
in the paper. It was written before the first model was trained and is shared by
every model, which is what makes the comparisons paired. **Do not regenerate it**
if you intend to compare against our numbers.

## Reproduce from raw data

**1. Obtain the cohorts.** All nine are public; each has its own access step.
See `data/README.md`. We cannot redistribute them.

**2. Build the corpus.**

```bash
python src/scripts/build_corpus.py --day-overlap 40
python src/scripts/pack_corpus_for_trainer.py
python src/scripts/corpus_summary.py         # compare against data/corpus_report.json
```

**3. Pretrain.**

```bash
python src/scripts/pretrain_glucoprism.py --corpus corpus_v2fmt_ov40.npz \
       --use-vib --w-vib 0.1 --seed 0                        # GlucoPRISM-C
python src/scripts/pretrain_glucoprism.py --corpus corpus_v2fmt_ov40.npz \
       --use-vib --w-vib 0.1 --sim-bias measured --seed 0    # GlucoPRISM-E
```

Baselines have their own folders: `baselines/<model>/reproduce.py`.

Run at least three seeds for anything you intend to compare. The seed standard
deviation on this benchmark is close to 1.0 ROC-AUC, so a single-seed difference
below that is not interpretable — including ours.

**4. Evaluate and regenerate every number.**

```bash
python src/scripts/embed_cohorts.py
python src/scripts/probe_frozen_folds.py
python src/scripts/significance_within_cohort.py
python src/scripts/canonical_numbers.py      # one source of truth
python src/scripts/make_tables.py
python src/scripts/make_figures.py
```

`canonical_numbers.py` recomputes every quantity the paper cites and writes
`canonical.tex`. The paper cites those macros instead of typed literals, so a
number cannot drift from the data behind it.

## The protocol

Reproduction depends on matching it exactly:

- Logistic regression, `l2`, `lbfgs`, `max_iter=1000`, **no inner search over
  `C`**, no class weighting.
- 5-fold subject-grouped cross-validation, 10 repeats, on the frozen folds.
- Window level **and** subject level, always both.
- Multiplicity correction over a family declared before results are read.

Change any of these silently and your numbers stop being comparable to ours.

## Licence

MIT (`LICENSE`). Released weights are covered by the same terms. Third-party
checkpoints are fetched from their original sources by
`baselines/<model>/reproduce.py` and keep their own licences. See `CITATION.cff`.
