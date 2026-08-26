# glucoprism-c.safetensors

The within-cohort released model: blocked pooling, protocol objectives at
lambda = (0.2, 0.2, 0.1), and a variational bottleneck on the Sensor block at
beta = 0.1.

Read it as `[z_T || z_S]` and discard the 16-dimensional Sensor block `z_A`.
That is a slice, not a retraining step, and it needs no device labels at test
time. Reading the full 128-dimensional vector is a different model with
different numbers; the paper reports both.

```python
from safetensors.torch import load_file
sd = load_file("weights/glucoprism_c/glucoprism-c.safetensors")
```

Trained by us on the public-only corpus described in `data/README.md`, and released under this repository's licence.
