# GlucoPRISM

Code, weights and reproduction instructions for *Naming the Nuisance: Reserved
Subspaces Buy Addressability, Not Disentanglement in CGM Foundation Models*.

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

`data/processed/splits_frozen.json` is the subject-to-fold assignment behind
every number in the paper. It was written before the first model was trained and is shared by
every model, which is what makes the comparisons paired. **Do not regenerate it**
if you intend to compare against our numbers.

## Reproduce from raw data

**1. Obtain the cohorts.** All nine are public; each has its own access step.
See `data/README.md`. We cannot redistribute them.

**2. Build the corpus.**

```bash
python src/scripts/build_corpus.py --all --day-overlap 0.4 --out-suffix _ov40
python src/scripts/pack_corpus_for_trainer.py \
       --datasets replacebg stanford shanghait2dm colas bigideas --shard-suffix _ov40
python src/scripts/corpus_summary.py         # compare against data/corpus_report.json
```

`--day-overlap` is a **fraction**, not a percentage: consecutive 24 h windows
overlap by `r`, so the stride is `(1 - r) * 24 h`. Passing `40` makes the stride
negative and the build fails on every cohort.

Pass the five cohorts explicitly. `pack_corpus_for_trainer.py`'s default list
also contains `d1namo`, which is not part of the corpus the released models were
trained on and would add a sixth cohort.

This produces **10,952 windows from 514 subjects**, which is what
`data/corpus_report.json` records.

**3. Pretrain.**

```bash
python src/scripts/pretrain_glucoprism.py --corpus corpus_v2fmt_ov40.npz \
       --use-vib --w-vib 0.1 --seed 0                        # GlucoPRISM-C
python src/scripts/pretrain_glucoprism.py --corpus corpus_v2fmt_ov40.npz \
       --use-vib --w-vib 1.0 --sim-bias measured --seed 0    # GlucoPRISM-E
```

**The two released models use different bottleneck weights.** GlucoPRISM-C is
`--w-vib 0.1`, GlucoPRISM-E is `--w-vib 1.0`. Both are recorded in the shipped
configs (`weights/glucoprism_c/config.json`, `weights/glucoprism_e/config.json`)
and that is the authority. Training E at 0.1 produces a different model.

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

## External validation on your own cohort

The paper's external-validation table is a fifth cohort the models never saw:
frozen encoders, subject-level aggregation, one linear probe. That driver is
here and is deliberately cohort-agnostic:

```bash
python src/scripts/evaluate_external.py \
       --shard mycohort.npz --labels mycohort.csv --blocks
```

It takes only the two artefacts every cohort in this repository is reduced to
before probing — a window shard in the format `build_corpus.py` writes, and a
`.csv` with a `subject` column plus one column per endpoint. `--split-col`
additionally fits on one value of a shard column (a device, a site, a batch) and
scores on another, which is the cross-device test in the paper.

**The paper's external cohort is not distributed here, and neither is its
loader.** The Human Phenotype Project is owned by Pheno.AI, governed by a data
use agreement, and reachable only inside their trusted research environment;
access is arranged with them directly and is not ours to grant. A loader and
label construction for it would encode that dataset's internal schema, which is
not ours to publish, so those stay inside the enclave. What generalises — the
evaluation protocol — is the script above. With HPP access you can rebuild the
two inputs from the paper's appendix and reproduce the table; without it, you
can run the identical protocol on data of your own.

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
