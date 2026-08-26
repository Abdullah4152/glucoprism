# Ablations

Every ablation, diagnostic and negative result in the paper. Scripts are named
for the question they answer — the originals were named after an internal
decision log (`fd3`, `fd7`, `rev_`), which meant nothing to a reader.

All of these run on CPU from already-embedded runs unless "needs" says otherwise.

| Script | Question it answers | Needs |
|---|---|---|
| `objectives_x_bottleneck_2x2.py` | do the objectives and the bottleneck work alone, or only together? | `--no-protocol` runs |
| `sensor_block_capacity_sweep.py` | does the benefit track the Sensor block's width and KL price? | width/beta runs |
| `model_capacity_x_factorization.py` | is the interference a capacity problem? | 1x and 5x runs |
| `corpus_composition.py` | is corpus volume or corpus diversity the constraint? | corpus-fraction runs |
| `window_patch_geometry.py` | which windowing choices matter? | geometry runs |
| `window_geometry_decision.py` | which geometry to keep, and why | geometry runs |
| `posthoc_factorization_fit.py` | can the blocks be fitted after training? | embeddings |
| `posthoc_factorization_collect.py` | collects the above into the reported comparison | — |
| `block_controls_matched_width.py` | is the gain just a narrower probe input? | embeddings |
| `shortcut_and_erasure_diagnostics.py` | five questions, listed below | embeddings |
| `sensor_generator_robustness.py` | is the synthetic sensor generator fragile? | paired measurements |
| `paired_sensor_measurement.py` | how do two real sensors actually disagree? | CGMacros pairs |
| `sensor_generator_fit.py` | fit the generator to the measurement | CGMacros pairs |
| `sensor_generator_validate.py` | does the synthetic partner match real pairs? | CGMacros pairs |
| `fewshot_multiday_and_stability.py` | label efficiency, multi-day pooling, trait stability | embeddings |
| `statistical_power.py` | what effect size can this protocol resolve? | scores |
| `patchify_unit_tests.py` | patchify correctness under lookback and overlap | — |
| `config_smoke_test.py` | does a config train at all, before queueing it? | GPU |
| `window_start_diversity_check.py` | window start-time spread, a leakage sanity check | shards |
| `embed_confound_arms.py` | embeds the confound arms through the identical path | checkpoints |
| `select_release_seed.py` | which seed of each released model ships | scores |

`shortcut_and_erasure_diagnostics.py` answers five separate questions in one
pass because they share the embedding load: post-hoc nullspace erasure as an
alternative to the reserved block; partial deletion; calibration under transfer;
how predictable the device is from the observation mask alone; and whether the
blocks separated, measured with HSIC as well as correlation.

## Reproducing the arms these need

```bash
# objectives off, everything else identical -- the 2x2
python ../scripts/pretrain_glucoprism.py --corpus corpus_v2fmt_ov40.npz \
       --no-protocol --seed 0
python ../scripts/pretrain_glucoprism.py --corpus corpus_v2fmt_ov40.npz \
       --no-protocol --use-vib --w-vib 0.1 --seed 0

# Sensor-block capacity: width, then KL price
python ../scripts/pretrain_glucoprism.py --corpus corpus_v2fmt_ov40.npz \
       --use-vib --w-vib 0.1 --d-sensor 8 --seed 0
python ../scripts/pretrain_glucoprism.py --corpus corpus_v2fmt_ov40.npz \
       --use-vib --w-vib 1.0 --seed 0
```

Then embed and score them like everything else:

```bash
python ../scripts/embed_cohorts_subset.py K-
python ../scripts/probe_frozen_folds.py --runs ... --blocks full zTzS
```

## Two analyses that exist because we got it wrong first

`posthoc_factorization_collect.py` reports the pre-specified comparison, not
"did any of five blocks win". The second reading returns a better-looking number
and is selection bias: the winning block is chosen after seeing the result and
is never the same block twice.

`shortcut_and_erasure_diagnostics.py` measures partial deletion by dimension
count, not by scaling. Scaling the block is a no-op under a standardising probe
— the scale divides out exactly — so a sweep over scale factors returns
identical numbers to three decimals while appearing to measure something.

Both are documented in the scripts. The paper's final appendix lists the full
set of negative and retracted results.
