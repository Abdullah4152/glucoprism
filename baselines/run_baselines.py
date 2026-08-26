"""GlucoFM's remaining baselines, run exactly as GlucoFM App. B specifies.

All seven are used FROZEN and ZERO-SHOT -- none is retrained on our corpus.

  Chronos-2 / Chronos-2-small (App. B.1)
      "feed each CGM sequence into the model and extract the output hidden
       states. We aggregate valid hidden states with mean pooling while
       excluding the EOS token."
  MOMENT small / large (App. B.1)
      "remove null values, align each sequence to 288 points using linear
       interpolation, and extract frozen representations."
  Mantis / MantisV2 (App. B.1)
      "linearly interpolate each input sequence to 512 points and extract
       frozen encoder representations."
  CGMformer (App. B.2)
      "each CGM window is represented on a 288-point 5-minute grid; missing
       bins are mapped to CGMformer's <pad> token; glucose values are
       discretized using the released vocabulary; and a <cls> token is
       prepended to form a 289-token input ... mean-pooled hidden states ...
       the mean is weighted by the attention mask to exclude <pad> positions."

Writes one .npy per (model, cohort) into artifacts/baseline_emb/.
"""
from __future__ import annotations

import os as _os
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))


import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", r"D:\hf_cache")

ROOT = ROOT
sys.path.insert(0, str(ROOT / "src"))
from glucoprism.data.datasets import WindowShard          # noqa: E402
from glucoprism.data.labels import TASK_MATRIX            # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "experiments" / "artifacts" / "baseline_emb"
CGMF = ROOT / "external" / "cgmformer" / "ckpt" / "cgm_ckp"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def interp_to(g: np.ndarray, m: np.ndarray, n: int) -> np.ndarray:
    """Drop unobserved positions, then linearly interpolate onto `n` points.

    This is the "remove null values ... linear interpolation" step both MOMENT
    and Mantis are specified with. A window with fewer than 2 observations
    cannot be interpolated and is filled with its own mean (or the corpus
    median if it has none).
    """
    out = np.empty((len(g), n), np.float32)
    for i in range(len(g)):
        idx = np.flatnonzero(m[i] > 0)
        if idx.size >= 2:
            out[i] = np.interp(np.linspace(idx[0], idx[-1], n), idx, g[i][idx])
        elif idx.size == 1:
            out[i] = g[i][idx[0]]
        else:
            out[i] = 140.0
    return out


# ------------------------------------------------------------------ MOMENT

def embed_moment(g, m, repo: str, batch: int = 64) -> np.ndarray:
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(repo, model_kwargs={"task_name": "embedding"})
    model.init()
    model.to(DEV).eval()
    # MOMENT's encoder is built for seq_len=512; the paper interpolates CGM to
    # 288 and MOMENT pads internally to its own sequence length.
    x = interp_to(g, m, 512)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            t = torch.tensor(x[i:i + batch], device=DEV).unsqueeze(1)
            outs.append(model(x_enc=t).embeddings.float().cpu().numpy())
    return np.concatenate(outs)


# ------------------------------------------------------------------ Mantis

def embed_mantis(g, m, which: str, batch: int = 128) -> np.ndarray:
    from mantis.architecture import Mantis8M, MantisV2
    if which == "v1":
        net = Mantis8M(device=DEV).from_pretrained(str(ROOT / "external" / "Mantis"))
    else:
        net = MantisV2(device=DEV).from_pretrained(str(ROOT / "external" / "MantisV2"))
    net.to(DEV).eval()
    x = interp_to(g, m, 512)                       # App. B.1: 512 points
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            t = torch.tensor(x[i:i + batch], device=DEV).unsqueeze(1)
            outs.append(net(t).float().cpu().numpy())
    return np.concatenate(outs)


# ---------------------------------------------------------------- Chronos-2

def embed_chronos(g, m, repo: str, batch: int = 32) -> np.ndarray:
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(repo, device_map=DEV)
    x = interp_to(g, m, 288)
    outs = []
    for i in range(0, len(x), batch):
        ctx = [torch.tensor(row, dtype=torch.float32) for row in x[i:i + batch]]
        # embed() returns (list of per-series embeddings, list of scaling params).
        # Each embedding is (n_variates, n_positions, d_model); ours is univariate
        # and the final position is the EOS token, which App. B.1 excludes from
        # the mean pool.
        embs, _ = pipe.embed(ctx)
        for e in embs:
            e = e.float().cpu()
            if e.ndim == 3:
                e = e[0]
            outs.append(e[:-1].mean(0).numpy())
    return np.stack(outs)


# --------------------------------------------------------------- CGMformer

def _sinusoidal(n_pos: int, dim: int) -> torch.Tensor:
    """CGMformer replaces BERT's learnable position embedding with a sinusoidal
    one (their `CGMFormer/modeling_bert.py`), which is why the released
    checkpoint contains NO `position_embeddings.weight`. Loading it with stock
    HuggingFace BERT would silently leave that table randomly initialised."""
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(dim, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, (2 * (i // 2)) / dim)
    pe = torch.zeros(n_pos, dim)
    pe[:, 0::2] = torch.sin(ang[:, 0::2])
    pe[:, 1::2] = torch.cos(ang[:, 1::2])
    return pe


def embed_cgmformer(g, m, batch: int = 64) -> np.ndarray:
    from transformers import BertModel
    ckpt = CGMF / "checkpoint-30000"
    model = BertModel.from_pretrained(str(ckpt))
    with torch.no_grad():
        model.embeddings.position_embeddings.weight.copy_(
            _sinusoidal(model.config.max_position_embeddings,
                        model.config.hidden_size))
    model.to(DEV).eval()

    t2i = pickle.load(open(CGMF / "token2id.pkl", "rb"))
    PAD, CLS = t2i["<pad>"], t2i["<cls>"]
    lo = min(k for k in t2i if isinstance(k, (int, np.integer)))
    hi = max(k for k in t2i if isinstance(k, (int, np.integer)))

    vals = np.clip(np.rint(np.nan_to_num(g)), lo, hi).astype(np.int64)
    ids = np.vectorize(lambda v: t2i[int(v)])(vals)
    ids = np.where(m > 0, ids, PAD)
    n = len(ids)
    ids = np.concatenate([np.full((n, 1), CLS, np.int64), ids], axis=1)   # 289
    attn = np.concatenate([np.ones((n, 1), np.int64),
                           (m > 0).astype(np.int64)], axis=1)

    outs = []
    with torch.no_grad():
        for i in range(0, n, batch):
            a = torch.tensor(attn[i:i + batch], device=DEV)
            h = model(input_ids=torch.tensor(ids[i:i + batch], device=DEV),
                      attention_mask=a).last_hidden_state
            w = a.unsqueeze(-1).float()
            outs.append(((h * w).sum(1) / w.sum(1).clamp(min=1)).float().cpu().numpy())
    return np.concatenate(outs)


MODELS = {
    "MOMENT-small": lambda g, m: embed_moment(g, m, "AutonLab/MOMENT-1-small"),
    "MOMENT-large": lambda g, m: embed_moment(g, m, "AutonLab/MOMENT-1-large"),
    "Mantis": lambda g, m: embed_mantis(g, m, "v1"),
    "MantisV2": lambda g, m: embed_mantis(g, m, "v2"),
    "Chronos-2": lambda g, m: embed_chronos(g, m, "amazon/chronos-2"),
    "Chronos-2-small": lambda g, m: embed_chronos(g, m, "autogluon/chronos-2-small"),
    "CGMformer": lambda g, m: embed_cgmformer(g, m),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    print(f"device: {DEV}\n")
    shards = {c: WindowShard(PROC / f"{c}_ds.npz") for c in TASK_MATRIX}
    for name in a.models:
        for coh, sh in shards.items():
            dst = OUT / f"{name}__{coh}.npy"
            if dst.exists():
                continue
            g = np.nan_to_num(sh.data["glucose"].astype(np.float32))
            m = sh.data["mask"].astype(np.float32)
            try:
                e = MODELS[name](g, m)
                np.save(dst, e.astype(np.float32))
                print(f"  {name:<18}{coh:<15}{e.shape}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<18}{coh:<15}FAILED {type(exc).__name__}: "
                      f"{str(exc)[:140]}", flush=True)
                break


if __name__ == "__main__":
    main()
