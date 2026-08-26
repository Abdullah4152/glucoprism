# gluformer-tiny.safetensors

Our reproduction of GluFormer-tiny, pretrained on our corpus. Not comparable to
the published 135M-parameter model, which used a private corpus of roughly ten
million measurements.

Inference tensors: 51, parameters: 648,268.

```python
from safetensors.torch import load_file
sd = load_file("weights/gluformer/gluformer-tiny.safetensors")
```

Trained by us on the public-only corpus described in `data/README.md`, and released under this repository's licence.
