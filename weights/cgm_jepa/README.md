# cgm-jepa.safetensors

Our reproduction of CGM-JEPA, pretrained on our corpus.

Inference tensors: 71, parameters: 1,280,224.

```python
from safetensors.torch import load_file
sd = load_file("weights/cgm_jepa/cgm-jepa.safetensors")
```

Trained by us on the public-only corpus described in `data/README.md`, and released under this repository's licence.
