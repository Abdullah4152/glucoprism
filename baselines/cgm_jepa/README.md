# CGM-JEPA

Pretrained by us on our corpus.

Masked latent prediction over day-level CGM windows.

    python ../common/pretrain.py --model cgm_jepa --seed 0

Check the implementation before trusting a comparison:

    python ../common/verify_bit_exact.py

That is a bit-exactness regression test, not an approximate one. It guards the
primitives in `cgmkit/models/blocks.py`, which CGM-JEPA shares with other
models here: a change there that silently altered CGM-JEPA would invalidate
every comparison in the paper.

Our checkpoint is in `weights/cgm_jepa/`, so you can skip pretraining and go straight to probing.
