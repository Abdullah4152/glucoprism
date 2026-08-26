# glucoprism-e.safetensors

The cross-cohort released model. Identical to GlucoPRISM-C except that the
synthetic paired-sensor view carries the measured calibration offset rather
than none.

Same readout: `[z_T || z_S]`, discard `z_A`.

```python
from safetensors.torch import load_file
sd = load_file("weights/glucoprism_e/glucoprism-e.safetensors")
```

Trained by us on the public-only corpus described in `data/README.md`, and released under this repository's licence.
