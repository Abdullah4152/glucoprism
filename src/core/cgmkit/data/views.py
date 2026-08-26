"""The three protocol views GlucoPRISM trains on (proposal Sec. 3.2, 4.3).

The proposal's central claim is that the supervision needed to separate Trait,
State and Sensor is already contained in how CGM studies are run:

    V1  paired-sensor   same subject, SAME day, two devices   -> shares (t, s), varies a
    V2  repeated-day    same subject, DIFFERENT day           -> shares t,      varies s
    V3  challenge/free  in-clinic OGTT vs free-living         -> t fixed, s clamped

V1 is generated, not found. CGMacros is the only public paired-sensor cohort and
has just 45 subjects -- far too few to train on -- so, exactly as GlucoFM
synthesises augmented views during pretraining, we synthesise a **second-sensor
view for every window of every subject** in the corpus
(`augment.synthetic_libre_view`, Sec. 4.3). CGMacros' real pairs are held out
entirely and used only to check that the synthetic partner is a faithful proxy.

V3 is evaluation-only and drives E6; it lives in the experiment drivers, not here.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from ..models.glucofm import GlucoFMConfig, sample_patch_mask
from ..models.prism import DEVICE_ID
from .augment import augment, synthetic_libre_view


def build_v2_index(subjects: np.ndarray) -> dict[int, np.ndarray]:
    """window index -> the other window indices of the same subject.

    Subjects with a single window get an empty array and are flagged `has_v2 =
    False` at sampling time. They are skipped rather than paired with a copy of
    themselves: a self-pair makes the V2 alignment term trivially zero and would
    be a free reduction in the loss that teaches nothing.
    """
    order = {}
    for i, s in enumerate(subjects):
        order.setdefault(str(s), []).append(i)
    return {i: np.array([j for j in order[str(s)] if j != i], dtype=np.int64)
            for i, s in enumerate(subjects)}


class PrismDataset(Dataset):
    """Anchor window + its V1 partner + a V2 partner, with the patch mask.

    Both partners are drawn fresh on every `__getitem__`, so a window sees a
    different synthetic sensor and a different second day each epoch. That
    matches how GlucoFM applies augmentation on the fly and stops the model from
    memorising one fixed pairing.
    """

    def __init__(self, shard, cfg: GlucoFMConfig | None = None, seed: int = 0,
                 augment_anchor: bool = True, exclude_subjects=None):
        self.shard = shard
        self.cfg = cfg or GlucoFMConfig()
        self.seed = seed
        self.augment_anchor = augment_anchor

        d = shard.data
        subjects = np.asarray([str(s) for s in d["subject"]])
        # `exclude_subjects` carves out the held-out pretraining split used to
        # sweep loss weights and block dims (proposal Sec. 9). Filtering here
        # rather than at the shard keeps V2 pairing honest: a held-out subject's
        # windows must not be reachable as someone's second day either.
        keep = np.ones(len(subjects), dtype=bool)
        if exclude_subjects:
            keep = ~np.isin(subjects, np.asarray(list(exclude_subjects), dtype=str))

        self.glucose = np.nan_to_num(d["glucose"].astype(np.float32), nan=0.0)[keep]
        self.mask = d["mask"].astype(np.float32)[keep]
        self.start = d["start_idx"].astype(np.int64)[keep]
        self.subjects = subjects[keep]
        self.devices = np.asarray([str(x) for x in d["device"]])[keep]
        self.v2 = build_v2_index(self.subjects)

    def __len__(self) -> int:
        return len(self.glucose)

    def _device_id(self, i: int) -> int:
        # Unknown device families fall back to the Dexcom class rather than
        # crashing; the corpus only contains dexcom / libre / ipro today.
        return DEVICE_ID.get(self.devices[i], DEVICE_ID["dexcom"])

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, i, np.random.randint(1 << 30)))

        g, m = self.glucose[i], self.mask[i]
        if self.augment_anchor:
            g, m = augment(g, m, rng)                 # GlucoFM App. C.7, unchanged

        # --- V1: the synthetic second sensor (Sec. 4.3) ------------------
        # Same subject, same day, same underlying (t, s); only the device
        # changes. Built from the *un-augmented* window so the pair differs by
        # sensor effects alone rather than by two independent augmentations.
        g1, m1 = synthetic_libre_view(self.glucose[i], self.mask[i], rng)

        # --- V2: a different day of the same subject ---------------------
        others = self.v2[i]
        has_v2 = len(others) > 0
        j = int(rng.choice(others)) if has_v2 else i

        return {
            "x": g, "m": m, "s": self.start[i],
            "x1": g1.astype(np.float32), "m1": m1.astype(np.float32), "s1": self.start[i],
            "x2": self.glucose[j], "m2": self.mask[j], "s2": self.start[j],
            "device_id": self._device_id(i),
            "device_id1": DEVICE_ID["synthetic"],
            "has_v2": bool(has_v2),
        }


def collate(items: list[dict], cfg: GlucoFMConfig, device=None, generator=None) -> dict:
    """Stack a batch and draw its patch mask (GlucoFM App. C.5, ratio U[0.5, 0.6])."""
    import torch

    out = {}
    for k in ("x", "m", "x1", "m1", "x2", "m2"):
        out[k] = torch.as_tensor(np.stack([it[k] for it in items]), dtype=torch.float32)
    for k in ("s", "s1", "s2", "device_id", "device_id1"):
        out[k] = torch.as_tensor(np.array([it[k] for it in items]), dtype=torch.long)
    out["has_v2"] = torch.as_tensor(np.array([it["has_v2"] for it in items]), dtype=torch.bool)
    out["patch_mask"] = sample_patch_mask(len(items), cfg,
                                          torch.device("cpu"), generator=generator)
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


# --------------------------------------------------- V1 falsification test

def real_pair_index(shard) -> list[tuple[int, int]]:
    """CGMacros' real same-subject/same-day Dexcom+Libre window pairs.

    These are the held-out validation of the synthetic V1 view (Sec. 4.3): if zT
    is invariant across synthetic pairs but not across these, the augmentation is
    wrong and we report that. This is the paper's main internal falsification
    test, so these windows must never enter pretraining.
    """
    d = shard.data
    subj = np.asarray([str(s) for s in d["subject"]])
    dev = np.asarray([str(x) for x in d["device"]])
    day = np.asarray([str(t)[:10] for t in d["start_time"]])       # calendar date

    by_key: dict[tuple[str, str], dict[str, int]] = {}
    for i in range(len(subj)):
        by_key.setdefault((subj[i], day[i]), {})[dev[i]] = i

    pairs = []
    for (_s, _d), slot in by_key.items():
        devs = sorted(slot)
        if len(devs) >= 2:                    # one window per device on that day
            pairs.append((slot[devs[0]], slot[devs[1]]))
    return pairs
