# Mantis and MantisV2

Third-party checkpoint, used zero-shot.

Lightweight calibrated encoders for time-series classification, used zero-shot.

    python reproduce.py

Worth noting when reading the paper: Mantis is competitive at subject level
despite trailing at window level, which is one reason the paper reports both
protocols and draws no rank-level conclusion from the subject-level column.

No weights for this model are stored in this repository. It is not ours to redistribute; `reproduce.py` fetches it from the original source under its own licence.
