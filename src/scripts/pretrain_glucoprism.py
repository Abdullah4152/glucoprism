"""Train the sibling repo's GlucoPRISM-v2 stack on OUR corpus, using THEIR code.

Head-to-head on our own benchmark (identical windows, folds and probe) showed
their v2 at 59.3 PR / 68.2 AUC against our GlucoFM at 57.7 / 65.3 -- a real
+2.9 AUC at roughly 3 sigma. Six waves of adding their components ONE AT A TIME to
a bare GlucoFM never reproduced it, which points at the combination rather than
any single part.

So rather than reimplement and risk a silent divergence, this runs their trainer
verbatim on our corpus. It isolates the remaining variable: their model, our data.

    python scripts/run_v2port.py --corpus corpus_v2fmt.npz --epochs 300 --seed 0
"""

from __future__ import annotations

import os as _os, sys as _sys
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))
RUNS = _P(_os.environ.get("GLUCOPRISM_RUNS", OUTDIR / "runs"))
EXTERNAL = _P(_os.environ.get("GLUCOPRISM_EXTERNAL", ROOT / "external"))
REFERENCE = ROOT / "src" / "core" / "released_model"
for _p in (ROOT / "src" / "core", ROOT / "baselines", ROOT / "src" / "scripts",
           ROOT / "src" / "ablations", REFERENCE,
           _P(__file__).resolve().parent):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))


import argparse
import json
import sys
from pathlib import Path
REF = REFERENCE
# Their package is also named `glucoprism`, so it must come FIRST on the path and
# this script must not import ours.
sys.path.insert(0, str(REF))

from glucofm.config import Config                                   # noqa: E402
from glucoprism.model import PrismConfig                            # noqa: E402
from glucoprism.pretrain import pretrain                            # noqa: E402
from glucoprism.sensor_sim import SensorSimParams                   # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def v2_config(epochs: int, seed: int) -> tuple[Config, PrismConfig]:
    """GlucoPRISM-v2 exactly as its checkpoint records it.

    Read from `weights/r1-gnorm-noscale.pt`:
      fm    : scale_inject=False, global_norm=True, mean=141.07, std=57.97,
              dropout=0.1
      prism : dims 64/48/16, w_sensor=0.2, w_day=0.2, w_indep=0.1,
              w_variance=1.0, beta_day_info=0.5, day_margin=0.3,
              use_cmp=True, w_cmp=1.0, stat_pool=True, n_devices=8
      300 epochs, sim_bias="zero"
    """
    fm = Config()
    fm.model.scale_inject = False
    fm.model.global_norm = True
    # Their constants were fitted on their corpus. Ours measures 156.90 / 63.87
    # over 2,745,507 observed readings, but keep THEIRS for the first run so this
    # is a pure model transplant; `--our-global-stats` swaps them.
    fm.model.global_mean = 141.07
    fm.model.global_std = 57.97
    fm.pretrain.epochs = epochs
    fm.pretrain.seed = seed

    pc = PrismConfig()
    pc.w_sensor, pc.w_day, pc.w_indep = 0.2, 0.2, 0.1
    pc.w_variance = 1.0
    pc.beta_day_info, pc.day_margin = 0.5, 0.3
    pc.use_cmp, pc.w_cmp = True, 1.0
    pc.stat_pool = True
    pc.n_devices = 8
    return fm, pc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus_v2fmt.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampler-alpha", type=float, default=0.3)
    ap.add_argument("--our-global-stats", action="store_true",
                    help="use our corpus's global mean/std (156.90 / 63.87) "
                         "instead of theirs (141.07 / 57.97)")
    # zA-drop-aware training. FD-3 measured that DISCARDING zA at inference is
    # worth +3.18 AUC on cross-dataset transfer, because zA collects the
    # device/cohort shortcut. That gain is currently accidental -- zA is never
    # asked to be clinically empty. The VIB makes it deliberate: zA becomes a
    # stochastic channel paying KL(q(zA|x) || N(0,I)) per nat, which upper-bounds
    # I(x; zA), so at small capacity it CANNOT carry clinical signal.
    #
    # Their own notes rule out the obvious alternative: a gradient-reversal
    # adversary was tried at w_adv=3 and BACKFIRED -- zA rose to 72.80 against a
    # 71.38 control, the documented failure mode where the encoder hides the
    # information from that particular head rather than discarding it
    # (Elazar & Goldberg 2018). Bounding information beats defeating a head.
    # The confound control. Our headline compares the v2 stack (global_norm +
    # stat_pool + L_CMP + variance floor, 300 epochs) WITH the factorization
    # against a bare GlucoFM. Those five components are therefore not separated
    # from the block structure. Zeroing the three protocol weights keeps every
    # component and switches the factorization objectives off, which is the arm
    # that separates them.
    ap.add_argument("--no-protocol", action="store_true",
                    help="set w_sensor = w_day = w_indep = 0: the v2 components "
                         "without the factorization objectives")
    ap.add_argument("--use-vib", action="store_true",
                    help="variational bottleneck on zA (zA-drop-aware training)")
    ap.add_argument("--w-vib", type=float, default=1.0,
                    help="beta: capacity price per nat on zA")
    # The three block widths must sum to embed_dim (128), so zA cannot be varied
    # in isolation -- something has to absorb the difference. zT is held at 64
    # because it is the block every downstream claim is about, so zS absorbs it
    # and the confound is zA-vs-zS width, which is stated in the paper.
    ap.add_argument("--d-sensor", type=int, default=None,
                    help="width of zA; zS absorbs the difference (zT stays 64)")
    ap.add_argument("--vib-free-bits", type=float, default=0.0)
    ap.add_argument("--sim-bias", choices=["zero", "measured"], default="zero",
                    help="'measured' restores the FD-9 calibration (-31.1 mg/dL "
                         "+- 15.8), which their sim_bias='zero' deliberately drops")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "checkpoints" / "v2port"))
    a = ap.parse_args()

    fm, pc = v2_config(a.epochs, a.seed)
    if a.our_global_stats:
        fm.model.global_mean, fm.model.global_std = 156.90, 63.87
    if a.no_protocol:
        pc.w_sensor = pc.w_day = pc.w_indep = 0.0
        for flag in ("use_sensor", "use_day", "use_indep"):
            if hasattr(pc, flag):
                setattr(pc, flag, False)
    if a.use_vib:
        pc.use_vib = True
        pc.w_vib = a.w_vib
        pc.vib_free_bits = a.vib_free_bits
    if a.d_sensor is not None:
        total = pc.d_trait + pc.d_state + pc.d_sensor
        pc.d_sensor = a.d_sensor
        pc.d_state = total - pc.d_trait - a.d_sensor
        if pc.d_state <= 0:
            raise SystemExit(f"--d-sensor {a.d_sensor} leaves d_state={pc.d_state}")
        print(f"  blocks     : zT={pc.d_trait} zS={pc.d_state} zA={pc.d_sensor}")

    corpus = PROCESSED / a.corpus
    if not corpus.exists():
        raise SystemExit(f"missing {corpus} -- run scripts/build_v2_corpus.py first")

    sim = SensorSimParams()
    if a.sim_bias == "zero":
        # Their V1 partner deliberately carries NO level shift.
        for attr in ("bias_mean", "bias_sd"):
            if hasattr(sim, attr):
                setattr(sim, attr, 0.0)
    else:
        # FD-9: real same-day Dexcom/Libre pairs differ by -31.1 +- 15.8 mg/dL,
        # and 43 of 44 subjects show Libre reading lower. The proposal specifies
        # a calibration offset; v2 removed it.
        for attr, val in (("bias_mean", -31.12), ("bias_sd", 15.77)):
            if hasattr(sim, attr):
                setattr(sim, attr, val)

    print(f"  corpus     : {corpus.name}")
    print(f"  global_norm: mean={fm.model.global_mean} std={fm.model.global_std}")
    print(f"  epochs     : {a.epochs}  seed: {a.seed}  sampler_alpha: {a.sampler_alpha}")
    print(f"  sim_bias   : {a.sim_bias}   vib: {a.use_vib}"
          + (f" (beta={a.w_vib}, free_bits={a.vib_free_bits})" if a.use_vib else ""))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = pretrain(corpus, out, fm, pc, sim, epochs=a.epochs,
                   sampler_alpha=a.sampler_alpha, log_every=10, augment=True)
    (out / "v2port_run.json").write_text(json.dumps(rep, indent=2, default=str),
                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "history"},
                     indent=2, default=str)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
