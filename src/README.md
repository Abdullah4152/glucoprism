# Source

Three folders, split by what the code is for.

```
core/        the model, data and evaluation code -- imported, not run
scripts/     corpus build, pretraining and evaluation drivers -- run these
ablations/   every ablation and analysis reported in the paper
```

## core/

```
core/glucoprism/     our package: models, data pipeline, probing, training loops
core/reference/      the reference implementation that trained the two
                     released models, vendored unchanged
```

`core/reference/` is vendored deliberately. The two released models were trained
by `scripts/run_v2port.py`, which imports that tree rather than reimplementing
it, because a reimplementation risks silent divergence from the checkpoints we
actually ship. Its package is also called `glucoprism`, so it is kept as a
separate tree and placed on `sys.path` only inside the runs that need it — do
not merge the two.

## scripts/

Ordinary reproduction path: build a corpus, pretrain, embed, score, tabulate.
See `scripts/README.md`.

## ablations/

Every ablation, negative result and diagnostic in the paper, mapped to the
section that reports it. See `ablations/README.md`.

## Paths

Nothing here hardcodes an absolute path. Each script resolves:

```python
ROOT   = os.environ.get("GLUCOPRISM_ROOT", <repo root>)
OUTDIR = os.environ.get("GLUCOPRISM_OUT",  ROOT / "artifacts")
```

Set `GLUCOPRISM_ROOT` if you run scripts from outside the repository, and
`GLUCOPRISM_OUT` if you want intermediates somewhere other than `./artifacts`.

## One implementation detail worth reading before you change anything

The predictive objectives operate on aligned mg/dL values, while the pooled
representation is normalised. Conflating the two makes the encoder level-blind:
on a +60 mg/dL shift the correct handling moves the representation by 0.136 and
the conflated variant by 8.7e-06. A model that gets this wrong trains,
converges, and is silently unable to represent hyperglycemia — which is fatal
for every endpoint in this benchmark, because HbA1c, HOMA-IR and hypoglycemia
are all functions of absolute glucose level. It is invisible in the loss curves.
See `core/glucoprism/models/glucofm.py`.
