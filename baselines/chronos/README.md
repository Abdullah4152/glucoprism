# Chronos-2 and Chronos-2-small

Third-party checkpoint, used zero-shot.

General-purpose time-series foundation models, used zero-shot.

    python reproduce.py

Both sizes are evaluated. Chronos tokenises scaled values and runs a language
model over them, so a CGM window is fed as a plain univariate series.

No weights for this model are stored in this repository. It is not ours to redistribute; `reproduce.py` fetches it from the original source under its own licence.
