# GlucoFM

Pretrained by us on our corpus.

The backbone this paper builds on, and the strongest published model on this
benchmark. Ours is a reimplementation, not the authors' code.

    python ../common/pretrain.py --model glucofm --seed 0

Our reproduction lands at 720,278 trainable parameters against the 720,241
reported in the original paper, and preserves the published ordering of models
on the benchmark. If yours does not, fix that before comparing anything.

The implementation is in `src/core/cgmkit/models/glucofm.py` rather than here,
because it is also the backbone our own model wraps -- it is the one baseline
that is part of the method.

Our checkpoint is in `weights/glucofm/`, so you can skip pretraining and go straight to probing.
