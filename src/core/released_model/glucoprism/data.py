"""
GlucoPRISM pretraining data: the three protocol views.

V1 paired-sensor  synthetic partner from the CALIBRATED sensor simulator
                  (glucoprism/sensor_sim.py, fitted to the real CGMacros
                  Dexcom<->Libre gap). CGMacros' real pairs are never used in
                  training - they are the held-out validation of this proxy.
V2 repeated-day   a second window from the SAME subject on a DIFFERENT day.
V3 challenge/free is an evaluation-only view (Stanford OGTT vs home), used by
                  the E6 leakage audit rather than by pretraining.

Device labels come from the real cohort a window belongs to, so the zA
classification head gets genuine multi-device supervision (Dexcom / Libre /
iPro / Medtronic) without needing pairs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .sensor_sim import SensorSim, SensorSimParams

# The synthetic V1 partner is labelled `libre` - it is SIMULATING a Libre, and
# giving it its own class made the device head trivially solvable ("was this
# decimated?"), so L_sensor collapsed to ~0 without learning a real device
# manifold.
DEVICES = ["dexcom", "libre", "ipro", "medtronic", "unknown"]
DEV_ID = {d: i for i, d in enumerate(DEVICES)}


def load_corpus(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["x"] = np.nan_to_num(out["x"].astype(np.float32), nan=0.0)
    out["m"] = out["m"].astype(np.float32)
    return out


class PrismPretrainDataset(Dataset):
    """Emits an anchor window plus its V1 and V2 partners."""

    def __init__(self, corpus: dict, sim_params: SensorSimParams,
                 n_patches: int = 24, mask_low: float = 0.5, mask_high: float = 0.6,
                 seed: int = 0, train: bool = True, augment=None,
                 n_days: int = 1):
        self.c = corpus
        self.n = len(corpus["x"])
        self.P = n_patches
        # n_days > 1 additionally emits a SET of K windows from the same
        # subject, for the day-level JEPA objective (proposal 4.4 taken
        # seriously: the real hierarchy in CGM is day -> subject, not
        # hour -> 4-hour, which never paid across five configurations).
        self.n_days = n_days
        self.mask_low, self.mask_high = mask_low, mask_high
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.sim = SensorSim(sim_params)
        # GlucoFM applies CGM-aware augmentation to the anchor; GlucoPRISM
        # originally did not, which is a confound when the two are compared.
        # `augment` takes an AugConfig to switch it on.
        self.aug = None
        if augment is not None:
            from glucofm.data import CGMAugment
            self.aug = CGMAugment(augment, corpus["x"].shape[1], self.rng)

        # V2 index: subject -> {day -> [window ids]}
        self.by_subject: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        days = corpus.get("day")
        for i, s in enumerate(corpus["subj"]):
            d = str(days[i]) if days is not None else str(i)
            self.by_subject[str(s)][d].append(i)
        self.subject_days = {s: sorted(v.keys()) for s, v in self.by_subject.items()}

        dev = corpus.get("device")
        self.dev_id = np.array(
            [DEV_ID.get(str(v), DEV_ID["unknown"]) for v in dev] if dev is not None
            else [DEV_ID["unknown"]] * self.n, dtype=np.int64)

        n_v2 = sum(1 for s in self.subject_days if len(self.subject_days[s]) > 1)
        print(f"    dataset: {self.n} windows | {len(self.by_subject)} subjects | "
              f"{n_v2} subjects have >=2 days (V2 available)")

    def __len__(self):
        return self.n

    def _patch_mask(self) -> np.ndarray:
        ratio = self.rng.uniform(self.mask_low, self.mask_high)
        k = max(1, int(round(ratio * self.P)))
        pm = np.zeros(self.P, dtype=np.float32)
        pm[self.rng.choice(self.P, size=k, replace=False)] = 1.0
        return pm

    def _v2_partner(self, i: int) -> int | None:
        s = str(self.c["subj"][i])
        days = self.subject_days.get(s, [])
        if len(days) < 2:
            return None
        cur = str(self.c["day"][i]) if "day" in self.c else None
        other = [d for d in days if d != cur]
        if not other:
            return None
        d = other[self.rng.integers(len(other))]
        pool = self.by_subject[s][d]
        return int(pool[self.rng.integers(len(pool))])

    def __getitem__(self, i: int) -> dict:
        x = self.c["x"][i].astype(np.float32)
        m = self.c["m"][i].astype(np.float32)
        s = int(self.c["s"][i])

        # ---- V1: calibrated synthetic sensor view (from the CLEAN anchor) ----
        x1, m1 = self.sim(x, m, self.rng)

        if self.aug is not None and self.train:
            x, m = self.aug(x, m)
            x, m = x.astype(np.float32), m.astype(np.float32)

        # ---- V2: same subject, different day ----
        j = self._v2_partner(i) if self.train else None
        if j is None:
            x2, m2, s2, has_v2 = x, m, s, 0.0
        else:
            x2 = self.c["x"][j].astype(np.float32)
            m2 = self.c["m"][j].astype(np.float32)
            s2, has_v2 = int(self.c["s"][j]), 1.0

        out = {}
        if self.n_days > 1:
            xs, ms, ss, ok = self._day_set(i)
            out.update(xk=torch.from_numpy(xs), mk=torch.from_numpy(ms),
                       sk=torch.from_numpy(ss), kvalid=torch.from_numpy(ok))

        out.update({
            "x": torch.from_numpy(x * m), "m": torch.from_numpy(m),
            "s": torch.tensor(s, dtype=torch.long),
            "x1": torch.from_numpy(x1), "m1": torch.from_numpy(m1),
            "s1": torch.tensor(s, dtype=torch.long),
            "x2": torch.from_numpy(x2 * m2), "m2": torch.from_numpy(m2),
            "s2": torch.tensor(s2, dtype=torch.long),
            "patch_mask": torch.from_numpy(self._patch_mask()),
            "device": torch.tensor(int(self.dev_id[i]), dtype=torch.long),
            "device_v1": torch.tensor(DEV_ID["libre"], dtype=torch.long),
            "has_v2": torch.tensor(has_v2, dtype=torch.float32),
        })
        return out

    def _day_set(self, i: int):
        """K windows from subject(i), window i first, padded if it has fewer.

        `kvalid` marks real days; padded slots repeat window i and are excluded
        from the day-level objective, so a subject with one day contributes
        nothing to it rather than a degenerate self-prediction.
        """
        K = self.n_days
        s = str(self.c["subj"][i])
        pool = [j for d in self.subject_days.get(s, [])
                for j in self.by_subject[s][d] if j != i]
        take = [i]
        if pool:
            k = min(K - 1, len(pool))
            take += [int(v) for v in self.rng.choice(pool, size=k, replace=False)]
        ok = np.zeros(K, dtype=np.float32)
        ok[:len(take)] = 1.0
        while len(take) < K:
            take.append(i)
        xs = np.stack([self.c["x"][j].astype(np.float32) for j in take])
        ms = np.stack([self.c["m"][j].astype(np.float32) for j in take])
        ss = np.array([int(self.c["s"][j]) for j in take], dtype=np.int64)
        return (xs * ms).astype(np.float32), ms, ss, ok


class EvalWindows(Dataset):
    """Plain windows for frozen feature extraction (no pairs, no masking)."""

    def __init__(self, x: np.ndarray, m: np.ndarray, s: np.ndarray):
        self.x = np.nan_to_num(x.astype(np.float32), nan=0.0)
        self.m = m.astype(np.float32)
        self.s = s.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": torch.from_numpy(self.x[i] * self.m[i]),
                "m": torch.from_numpy(self.m[i]),
                "s": torch.tensor(int(self.s[i]), dtype=torch.long)}


def make_weighted_sampler(corpus: dict, alpha: float = 0.3):
    """Cohort-temperature sampler: p_c ~ n_c^alpha, uniform within cohort.

    Used with the FULL corpus, where REPLACE-BG is 97% of the windows and would
    otherwise leave the sensor block with ~1% Libre to learn a device manifold
    from.
    """
    coh = corpus["cohort"]
    names, counts = np.unique(coh, return_counts=True)
    p = counts.astype(float) ** alpha
    p /= p.sum()
    per = {n: p[i] / counts[i] for i, n in enumerate(names)}
    w = np.array([per[c] for c in coh], dtype=np.float64)
    return torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(w), num_samples=len(w), replacement=True)
