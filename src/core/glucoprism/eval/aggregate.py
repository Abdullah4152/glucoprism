"""Subject-level aggregation over a subject's days (proposal Sec. 4.4, Prop. 2).

Every clinical label in this benchmark is a subject property, so scoring one
24-hour window at a time measures the model through a single noisy day. E1b
aggregates a subject's windows first; E5 asks which aggregator is right.

Proposition 2 says the mean is the minimum-variance unbiased estimator for a
day-INVARIANT block, and a lossy one otherwise -- so the proposal prescribes

    z_subj = [ mean_k zT^(k)  ;  phi({zS^(k)}) ]                      (Sec. 4.4)

with `phi` a permutation-invariant set encoder that explicitly computes
dispersion statistics, and zA dropped at inference.

`phi` is fitted POST-HOC on frozen per-day embeddings (decision D11), which is
what lets E5 compare mean / concat(mean,max) / rich / phi over *identical*
inputs. It is fitted inside each training fold only -- never on test subjects --
so a learned aggregator gets no information a fixed one does not.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------- fixed aggregators

def _group_slices(subjects: np.ndarray):
    """Stable subject -> row-index map, ordered so outputs line up with `uniq`."""
    uniq = np.unique(subjects)
    return uniq, [np.flatnonzero(subjects == s) for s in uniq]


def agg_mean(X: np.ndarray, idx) -> np.ndarray:
    return np.stack([X[i].mean(0) for i in idx])


def agg_meanmax(X: np.ndarray, idx) -> np.ndarray:
    """GlucoFM's fallback for cohorts where the mean alone loses signal."""
    return np.stack([np.concatenate([X[i].mean(0), X[i].max(0)]) for i in idx])


def agg_rich(X: np.ndarray, idx) -> np.ndarray:
    """[mean | sd | p10 | p90] -- a fixed aggregator that carries dispersion.

    This is the control Prop. 2's learned aggregator has to beat: if a set
    encoder cannot improve on four hand-computed statistics, the proposition is
    not supported, and E5 says so.
    """
    out = []
    for i in idx:
        w = X[i]
        out.append(np.concatenate([w.mean(0), w.std(0),
                                   np.percentile(w, 10, axis=0),
                                   np.percentile(w, 90, axis=0)]))
    return np.stack(out)


FIXED = {"mean": agg_mean, "meanmax": agg_meanmax, "rich": agg_rich}


def aggregate(X: np.ndarray, subjects: np.ndarray, method: str = "mean"):
    """(n_windows, d) -> (n_subjects, d'), plus the subject order."""
    uniq, idx = _group_slices(np.asarray(subjects))
    if method not in FIXED:
        raise ValueError(f"unknown fixed aggregator {method!r}; use one of {sorted(FIXED)}")
    return FIXED[method](np.asarray(X, float), idx), uniq


# --------------------------------------------------- learned set aggregator

class DeepSetsAggregator:
    """Permutation-invariant set encoder over one subject's day embeddings.

    Input features are exactly what Sec. 4.4 asks for -- per-dimension mean,
    standard deviation and range across days -- so the encoder starts from the
    dispersion statistics rather than having to discover them. `rho` is a small
    MLP on top.

    Fitted with a supervised objective on the downstream label because that is
    the quantity E5 compares aggregators on; fitting it unsupervised would test
    a different claim. Fitting happens strictly inside a training fold.
    """

    def __init__(self, d_out: int = 64, hidden: int = 128, epochs: int = 200,
                 lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 0):
        self.d_out, self.hidden = d_out, hidden
        self.epochs, self.lr, self.weight_decay, self.seed = epochs, lr, weight_decay, seed
        self.rho = None
        self._d_in = None

    @staticmethod
    def set_features(X: np.ndarray, idx) -> np.ndarray:
        """{z^(k)} -> [mean | std | range] per dimension (Sec. 4.4)."""
        out = []
        for i in idx:
            w = X[i]
            out.append(np.concatenate([w.mean(0), w.std(0), w.max(0) - w.min(0)]))
        return np.stack(out)

    def fit(self, X: np.ndarray, subjects: np.ndarray, y_by_subject: dict):
        import torch
        import torch.nn as nn

        uniq, idx = _group_slices(np.asarray(subjects))
        feats = self.set_features(np.asarray(X, float), idx)
        y = np.array([y_by_subject[s] for s in uniq])
        classes = np.unique(y)
        if len(classes) < 2:
            self.rho = None                       # degenerate fold; fall back to mean
            return self

        torch.manual_seed(self.seed)
        self._d_in = feats.shape[1]
        self.rho = nn.Sequential(
            nn.Linear(self._d_in, self.hidden), nn.GELU(),
            nn.Linear(self.hidden, self.d_out), nn.GELU(),
            nn.Linear(self.d_out, len(classes)))
        opt = torch.optim.AdamW(self.rho.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        # Standardise on the training fold; the same statistics are reused at
        # transform time so no test-fold information reaches the fit.
        self._mu, self._sd = feats.mean(0), feats.std(0) + 1e-6
        xt = torch.tensor((feats - self._mu) / self._sd, dtype=torch.float32)
        remap = {c: i for i, c in enumerate(classes)}
        yt = torch.tensor([remap[v] for v in y], dtype=torch.long)

        self.rho.train()
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(self.rho(xt), yt)
            loss.backward()
            opt.step()
        self._grad_seen = any(p.grad is not None and float(p.grad.abs().max()) > 0
                              for p in self.rho.parameters())
        return self

    def transform(self, X: np.ndarray, subjects: np.ndarray):
        """-> (n_subjects, d_out) penultimate features, plus the subject order."""
        import torch

        uniq, idx = _group_slices(np.asarray(subjects))
        feats = self.set_features(np.asarray(X, float), idx)
        if self.rho is None:
            return agg_mean(np.asarray(X, float), idx), uniq
        xt = torch.tensor((feats - self._mu) / self._sd, dtype=torch.float32)
        self.rho.eval()
        with torch.no_grad():
            h = xt
            for layer in list(self.rho)[:-1]:     # everything before the classifier
                h = layer(h)
        return h.numpy(), uniq


def prism_subject_repr(zT: np.ndarray, zS: np.ndarray, subjects: np.ndarray,
                       phi: DeepSetsAggregator | None = None):
    """Sec. 4.4 exactly: mean-pool the trait block, set-encode the state block.

    `zA` is not an argument -- it is dropped at inference by construction, which
    is the point of giving the sensor factor a named place to live.
    """
    uniq, idx = _group_slices(np.asarray(subjects))
    t = agg_mean(np.asarray(zT, float), idx)
    if phi is None:
        s = agg_rich(np.asarray(zS, float), idx)
    else:
        s, _ = phi.transform(zS, subjects)
    return np.concatenate([t, s], axis=1), uniq


# ------------------------------------------------------- E4b block controls

def block_controls(X: np.ndarray, widths=(16, 48), seed: int = 0) -> dict:
    """Dimension-matched controls for the E4 block-routing claim.

    A block's absolute score is uninterpretable on its own: at 29-69 subjects a
    narrow probe input is better regularised than a wide one, so a 16-d block can
    beat a 128-d representation for reasons that have nothing to do with what it
    encodes. Each block is therefore compared against a control of IDENTICAL
    width. A block that does not beat its own control is not carrying
    block-specific information.

    Only DATA-INDEPENDENT controls are returned -- a random Gaussian projection
    and a random contiguous slice. Both are fixed given the seed, so neither can
    see a test fold.

    PCA was previously included here and it was WRONG: fitting the components on
    all rows lets the control peek at the evaluation fold, and it inflated the
    control by ~2 AUC. Refitted strictly inside each training fold, PCA scores the
    same as the full representation (66.9 vs 66.7 on GlucoFM), i.e. the apparent
    "PCA beats the blocks" result was leakage. Rank reduction is a legitimate
    thing to test, but it belongs in the probe where it can be fold-fitted --
    `pca_in_fold` below -- not in a control computed once over the whole matrix.
    """
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    out = {}
    for w in widths:
        if w > d:
            continue
        P = rng.normal(size=(d, w)) / np.sqrt(d)
        out[f"rand{w}"] = X @ P
        start = int(rng.integers(0, max(d - w, 1)))
        out[f"slice{w}"] = X[:, start:start + w]
    return out


def pca_in_fold(X_train: np.ndarray, X_test: np.ndarray, n_components: int):
    """Rank reduction fitted on the TRAINING fold only, then applied to both.

    Measured on GlucoFM over the 14 cells: PCA-24 gives 58.7 PR / 66.9 AUC against
    the full 128-d at 58.5 / 66.7 -- within noise. Rank reduction is not a win
    here; it is recorded so the leaked version is not mistaken for one.
    """
    from sklearn.decomposition import PCA
    k = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    p = PCA(n_components=k, random_state=0).fit(X_train)
    return p.transform(X_train), p.transform(X_test)


def null_floor(X: np.ndarray, subjects: np.ndarray, seed: int = 0) -> dict:
    """What a representation carrying NOTHING scores under the same probe.

    Without this the reader cannot tell whether "near chance" means 50 or 68 --
    at these cohort sizes a random projection can score well above 50 purely
    through subject-level structure the probe latches onto.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)
    perm = dict(zip(uniq, rng.permutation(uniq)))
    shuffled = np.stack([X[subjects == perm[s]].mean(0) for s in subjects])
    return {"noise": rng.normal(size=X.shape),
            "constant": np.ones_like(X),
            "subject_shuffled": shuffled}
