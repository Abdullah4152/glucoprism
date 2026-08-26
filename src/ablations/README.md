# Ablations

Every ablation, diagnostic and negative result in the paper, mapped to the
section that reports it. All of these run on CPU from already-embedded runs
unless the "needs" column says otherwise.

| Script | Reports | Needs |
|---|---|---|
| `confound_analyse.py` | the 2x2 over objectives and bottleneck | `--no-protocol` runs |
| `capacity_analyse.py` | Sensor-block width and KL-weight sweep | width/beta runs |
| `fd8_analyse.py` | capacity crossed with how the factorization is obtained | 1x/5x runs |
| `rbg_fraction_study.py` | corpus composition and leave-one-cohort-out | corpus-fraction runs |
| `fd7_analyse.py`, `fd7_decide.py` | window and patch geometry | geometry runs |
| `run_posthoc_heads.py`, `posthoc_collect.py` | post-hoc factorization of a frozen encoder | embeddings |
| `fd3_block_controls.py` | block controls at matched width | embeddings |
| `reviewer_analyses.py` | erasure baseline, partial deletion, calibration, device predictability, block dependence | embeddings |
| `rev_generator_robustness.py` | paired-sensor generator stability under resampling | paired measurements |
| `fd9_sensor_analysis.py` | the real Dexcom/Libre disagreement | CGMacros pairs |
| `fd9_fit_generator.py`, `fd9_validate_generator.py` | fitting and validating the synthetic partner | CGMacros pairs |
| `power_analysis.py` | detectable effect size given the protocol | scores |
| `test_patch_geometry.py` | patchify correctness under lookback and overlap | — |
| `select_release_seed.py` | which seed of each released model ships | scores |
| `embed_confound.py` | embeds the confound arms through the identical path | checkpoints |
| `smoke_configs.py` | short runs that check a config trains before it is queued | GPU |
| `check_start_diversity.py` | window start-time spread, a leakage sanity check | shards |

## Reproducing the arms these need

Several scripts analyse runs that must be trained first:

```bash
# objectives off, everything else identical -- the 2x2
python ../scripts/run_v2port.py --corpus corpus_v2fmt_ov40.npz --no-protocol --seed 0
python ../scripts/run_v2port.py --corpus corpus_v2fmt_ov40.npz --no-protocol \
       --use-vib --w-vib 0.1 --seed 0

# Sensor-block capacity: width, then KL price
python ../scripts/run_v2port.py --corpus corpus_v2fmt_ov40.npz --use-vib \
       --w-vib 0.1 --d-sensor 8   --seed 0
python ../scripts/run_v2port.py --corpus corpus_v2fmt_ov40.npz --use-vib \
       --w-vib 1.0 --seed 0
```

Then embed and score them the same way as everything else:

```bash
python ../scripts/embed_subset.py K-          # or any substring
python ../scripts/v2_score_npy.py --runs ... --blocks full zTzS
```

## A note on reading these

Two of the analyses here exist because we got the answer wrong first, and the
scripts carry the corrected form:

- `posthoc_collect.py` reports the pre-specified comparison, not "did any of
  five blocks win". The second reading returns a better-looking number and is
  selection bias, because the winning block is chosen after seeing the result
  and is never the same block twice.
- `reviewer_analyses.py` measures partial deletion by dimension count, not by
  scaling. Scaling the block is a no-op under a standardising probe — the scale
  divides out exactly — and a sweep over scale factors returns identical numbers
  to three decimals while appearing to measure something.

Both are documented in the scripts themselves. The negative and retracted
results are listed in the paper's final appendix.
