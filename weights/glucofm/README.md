# glucofm.safetensors

Our reproduction of the backbone, pretrained on our corpus. This is the
comparison every headline number in the paper is against, so it is the
checkpoint to load if you want to check a delta rather than an absolute.

```python
from safetensors.torch import load_file
sd = load_file("weights/glucofm/glucofm.safetensors")
```

Trained by us on the public-only corpus described in `data/README.md`, and released under this repository's licence.
