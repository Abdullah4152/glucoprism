# X-CGM-JEPA

Pretrained by us on our corpus.

CGM-JEPA plus a masked Glucodensity cross-view.

    python ../common/pretrain.py --model x_cgm_jepa --seed 0

Requires the Glucodensity features; `cgmkit/data/glucodensity.py` computes them
from the same windows, so no extra data is needed.

Our checkpoint is in `weights/x_cgm_jepa/`, so you can skip pretraining and go straight to probing.
