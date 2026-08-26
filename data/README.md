# Data

We cannot redistribute the raw cohorts. All nine are public, but each has its
own access step. This folder holds what we can ship.

```
splits_frozen.json     the subject-to-fold assignment behind every number
corpus_report.json     what the built corpus should contain, for verification
pretrain_holdout.json  subjects held out of pretraining for hyperparameter work
```

## The frozen folds

`splits_frozen.json` is the most important file here. It was written **before
the first model was trained** and is shared by every model in the paper,
including all baselines. That is what licenses paired testing: seed and fold
variance is common to both arms of a comparison and cancels in the difference,
which makes paired tests roughly an order of magnitude more sensitive here than
unpaired ones.

Regenerate it and your numbers stop being comparable to ours. We release it
precisely so a new method can be compared against ours paired rather than
re-split.

## Cohorts

Pretraining: REPLACE-BG (Dexcom G4, 5 min), Stanford (Dexcom, 5 min),
ShanghaiT2DM (FreeStyle Libre, 15 min), Colás (Medtronic iPro, 5 min),
BIG IDEAs (Dexcom, 5 min).

Downstream: CGMacros, ShanghaiT2DM, Stanford, Hall — four cohorts, seven
endpoints, 14 task-cohort cells.

CGMacros and Hall never enter pretraining. CGMacros' real same-day Dexcom/Libre
paired windows are additionally reserved as a falsification set for the
synthetic paired-sensor view and are never trained on.

## Two composition choices that matter

REPLACE-BG is capped at 40 windows per subject. Uncapped it contributes roughly
28,000 windows and turns the corpus into a type 1 diabetes distribution.

ShanghaiT2DM's observation fraction is 0.333. That is not missingness to be
filtered out — it is the arithmetic consequence of a 15-minute sensor on a
five-minute grid. Coverage thresholds are applied relative to device rate, not
absolutely. A pipeline that filters on absolute coverage silently drops the
entire 15-minute cohort, and with it the only sampling-rate contrast in the
corpus.

## Staging

```bash
python ../src/scripts/download_datasets.py --help
python ../src/scripts/profile_datasets.py     # verify device, rate, coverage
python ../src/scripts/build_corpus.py --day-overlap 40
python ../src/scripts/corpus_summary.py       # compare against corpus_report.json
```

`build_corpus.py` enforces the pretrain/downstream subject split and refuses to
write a shard whose subjects appear on both sides. An earlier version of this
pipeline leaked downstream subjects into pretraining; the guard exists because
of that, and it checks subject counts rather than trusting the split.
