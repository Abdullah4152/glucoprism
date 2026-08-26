# x-cgm-jepa.safetensors

Our reproduction of X-CGM-JEPA, pretrained on our corpus.

Inference tensors: 136, parameters: 2,540,768.

```python
from safetensors.torch import load_file
sd = load_file("weights/x_cgm_jepa/x-cgm-jepa.safetensors")
```

Trained by us on the public-only corpus described in `data/README.md`, and released under this repository's licence.
