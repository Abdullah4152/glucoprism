# Data

We cannot redistribute the raw cohorts. All nine are public, but each has its
own access step. This folder holds what we *can* ship: the frozen evaluation
folds, the corpus manifest, and the scripts that stage the rest.

```
splits_frozen.json    the subject-to-fold assignment used for every number
corpus_report.json    what the built corpus contains, for verification
pretrain_holdout.json subjects held out of pretraining for hyperparameter work
```

## The frozen folds

`splits_frozen.json` is the most important file here. It was written **before
the first model was trained** and is shared by every model in the paper,
including all baselines. That is what licenses paired testing: seed and fold
variance is common to both arms of a comparison and cancels in the difference,
which makes paired tests roughly an order of magnitude more sensitive here than
unpaired ones.

If you regenerate it, your numbers stop being comparable to ours. We release it
precisely so that a new method can be compared against ours paired rather than
re-split.

## Pretraining cohorts

| Cohort | Device | Rate | Access |
|---|---|---|---|
| REPLACE-BG | Dexcom G4 | 5 min | JAEB Center public data request |
| Stanford | Dexcom | 5 min | public supplement |
| ShanghaiT2DM | FreeStyle Libre | 15 min | Figshare, open |
| Colás | Medtronic iPro | 5 min | journal supplement |
| BIG IDEAs | Dexcom | 5 min | public repository |

## Downstream cohorts

CGMacros, ShanghaiT2DM, Stanford and Hall, giving four cohorts and seven
endpoints for 14 task-cohort cells.

CGMacros and Hall never enter pretraining. CGMacros' real same-day
Dexcom/Libre paired windows are additionally reserved as a falsification set
for the synthetic paired-sensor view, and are never trained on.

## Two composition choices that matter

REPLACE-BG is capped at 40 windows per subject. Uncapped it contributes roughly
28,000 windows and turns the corpus into a type 1 diabetes distribution.

ShanghaiT2DM's observation fraction is 0.333, which is not missingness to be
filtered out — it is the arithmetic consequence of a 15-minute sensor on a
five-minute grid. Coverage thresholds are applied relative to device rate, not
absolutely. A pipeline that filters on absolute coverage will silently drop the
entire 15-minute cohort, and with it the only sampling-rate contrast in the
corpus.

## Staging

```bash
python ../src/scripts/download_datasets.py --help
python ../src/scripts/profile_datasets.py       # verify device/rate/coverage
python ../src/scripts/build_corpus.py --day-overlap 40
python ../src/scripts/corpus_summary.py         # compare against corpus_report.json
```

`corpus_summary.py` prints subject and hour counts. If they do not match
`corpus_report.json`, something upstream differs and downstream numbers will
not be comparable — check that before training anything.
