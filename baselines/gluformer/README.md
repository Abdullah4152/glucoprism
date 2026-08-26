# GluFormer (tiny)

Pretrained by us on our corpus.

Autoregressive next-token prediction over discretised glucose. We reproduce the
tiny variant; the published 135M-parameter model was trained on a private
corpus of roughly ten million measurements that we cannot match.

    python ../common/pretrain.py --model gluformer_tiny --seed 0

The tokeniser is `baselines/common/models/gluformer.py`, and glucose is clipped
to [40, 500] before binning. `cgmkit/data/gluformer_tokens.py` re-exports it for
the dataset layer.

Our checkpoint is in `weights/gluformer/`, so you can skip pretraining and go straight to probing.
