# CGMformer

Third-party checkpoint, used zero-shot.

CGM-specific, used as published. Not retrained on our corpus.

    python reproduce.py

Downloads the checkpoint from its original source, embeds the four downstream
cohorts, and probes on the frozen folds.

No weights for this model are stored in this repository. It is not ours to redistribute; `reproduce.py` fetches it from the original source under its own licence.
