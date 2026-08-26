"""Figure 1: the architecture, and where the intervention happens.

The point is not to redraw a transformer. It is to make legible that
(a) everything up to blocked pooling is the unmodified backbone, (b) the only
structural change is carving the pooled vector into three named blocks, and
(c) the deployed intervention is a slice applied after training.

Layout rule used throughout: every box is sized from its longest line rather
than eyeballed, because the first version of this figure had text spilling out
of half of them. `fit()` returns the width a label needs at the given font size.
"""
from __future__ import annotations

import os as _os, sys as _sys
from pathlib import Path as _P
ROOT = _P(_os.environ.get("GLUCOPRISM_ROOT",
                          _P(__file__).resolve().parents[2]))
OUTDIR = _P(_os.environ.get("GLUCOPRISM_OUT", ROOT / "artifacts"))
for _p in (ROOT / "src" / "core", ROOT / "baselines"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))


from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = OUTDIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "figure.dpi": 240,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
C_OURS, C_BASE, C_BAD, C_GREY = "#1a5fb4", "#3a3a3a", "#c01c28", "#9a9996"
C_BG, C_NEW = "#eef3f8", "#dce8f7"

FIG_W, XLIM = 5.45, 100.0          # x-units per inch = XLIM / FIG_W
UPI = XLIM / FIG_W                 # ~18.3 x-units per inch
# Average glyph advance for this serif face is close to 0.52 em, and the math
# spans run wider, so budget generously: an underestimate here is what made the
# first two drafts of this figure spill text out of every box.
CHAR = 0.0098                      # inches per char per pt of font size


def fit(lines, fs, pad=3.4):
    """Width in x-units needed to hold the longest line at font size fs."""
    longest = max(len(s) for s in lines.split("\n"))
    return longest * fs * CHAR * UPI + pad


def box(ax, x, y, w, h, label, fc="white", ec=C_BASE, lw=0.85, fs=7.4,
        weight="normal", tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, zorder=4, fontweight=weight,
            color=tc if tc else C_BASE, linespacing=1.4)
    return x + w


def arrow(ax, x1, y1, x2, y2, color=C_BASE, lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7.5, lw=lw, color=color,
                                 linestyle=ls, zorder=2, shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(FIG_W, 3.55))
ax.set_xlim(0, XLIM)
ax.set_ylim(0, 62)
ax.axis("off")

# ============================================================ row 1: backbone
ax.add_patch(FancyBboxPatch((1.0, 38.5), 98.0, 20.0,
                            boxstyle="round,pad=0.4,rounding_size=1.2",
                            fc=C_BG, ec="none", zorder=0))
ax.text(2.6, 55.6, "GlucoFM backbone (unmodified)", fontsize=7.0,
        style="italic", color=C_GREY, ha="left", va="center", zorder=1)

Y, H, FS = 41.0, 11.2, 7.3
x = 12.0
seq = ["24 h CGM\n+ mask", "causal\nGauss. filter",
       "state /\nevent split", "3-layer\nencoder", "mean\npool"]
ends = []
for lbl in seq:
    w = fit(lbl, FS)
    box(ax, x, Y, w, H, lbl, fs=FS)
    ends.append((x, x + w))
    x += w + 3.4
for (_, a), (b, _) in zip(ends[:-1], ends[1:]):
    arrow(ax, a, Y + H / 2, b, Y + H / 2)

# ====================================================== row 2: blocked pooling
ax.add_patch(FancyBboxPatch((1.0, 22.0), 98.0, 14.0,
                            boxstyle="round,pad=0.4,rounding_size=1.2",
                            fc=C_NEW, ec="none", zorder=0))
ax.text(2.6, 33.6, "blocked pooling: the only structural change",
        fontsize=7.0, style="italic", color=C_OURS, ha="left", va="center",
        zorder=1)

BY, BH = 23.4, 8.4
blocks = [("$z_T$  Trait  (64-d)", C_OURS), ("$z_S$  State  (48-d)", C_OURS),
          ("$z_A$  Sensor  (16-d)", C_BAD)]
bw = 25.0
bx = 12.0
centres = []
for lbl, col in blocks:
    box(ax, bx, BY, bw, BH, lbl, ec=col, lw=1.3, fs=7.5, weight="bold", tc=col)
    centres.append(bx + bw / 2)
    bx += bw + 4.0

# pool -> the three blocks
arrow(ax, ends[-1][0] + (ends[-1][1] - ends[-1][0]) / 2, Y,
      centres[1], BY + BH, color=C_BASE)

# ========================================================= row 3: training use
ax.text(0.8, 15.4, "training", fontsize=6.2, style="italic", color=C_GREY,
        ha="left", va="center")
TY, TH = 11.6, 7.6
t1 = "$L_{\\mathrm{MCR}} + L_{\\mathrm{TD}}$\nbackbone"
t2 = ("$\\lambda_1 L_{\\mathrm{sensor}} + \\lambda_2 L_{\\mathrm{day}}"
      " + \\lambda_3 L_{\\mathrm{indep}}$\nprotocol objectives")
t3 = "VIB on $z_A$\nbounds $I(x; z_A)$"
w1 = fit("backbone", 7.0) + 6.0
w3 = fit("bounds $I(x; z_A)$", 7.0)
w2 = XLIM - 12.0 - w1 - w3 - 8.0 - 4.0
box(ax, 12.0, TY, w1, TH, t1, fs=7.0, ec=C_GREY)
box(ax, 12.0 + w1 + 4.0, TY, w2, TH, t2, fs=7.0, ec=C_OURS)
box(ax, 12.0 + w1 + w2 + 8.0, TY, w3, TH, t3, fs=7.0, ec=C_BAD)

arrow(ax, centres[0], BY, 12.0 + w1 / 2, TY + TH, color=C_GREY, lw=0.8,
      ls=(0, (2.5, 2)))
arrow(ax, centres[1], BY, 12.0 + w1 + 4.0 + w2 / 2, TY + TH, color=C_OURS,
      lw=0.8, ls=(0, (2.5, 2)))
arrow(ax, centres[2], BY, 12.0 + w1 + w2 + 8.0 + w3 / 2, TY + TH, color=C_BAD,
      lw=0.8, ls=(0, (2.5, 2)))

# ======================================================== row 4: the intervention
ax.text(0.8, 5.0, "inference", fontsize=6.2, style="italic", color=C_GREY,
        ha="left", va="center")
IY, IH = 1.2, 7.6
i1 = "read $[\\,z_T \\Vert z_S\\,]$, discard $z_A$"
i2 = "frozen linear probe"
wi1 = fit("read [z_T || z_S] - discard z_A", 7.6)
wi2 = fit(i2, 7.2)
box(ax, 12.0, IY, wi1, IH, i1, fs=7.6, ec=C_BAD, lw=1.35, fc="#fdf1f1",
    weight="bold", tc=C_BAD)
box(ax, 12.0 + wi1 + 4.0, IY, wi2, IH, i2, fs=7.2)
arrow(ax, 12.0 + wi1, IY + IH / 2, 12.0 + wi1 + 4.0, IY + IH / 2)

# The cut: zA leaves the graph. Routed down the right margin so the label does
# not land on top of the VIB box.
XCUT = XLIM - 3.0
arrow(ax, centres[2], BY, XCUT, BY - 3.0, color=C_BAD, lw=1.1)
arrow(ax, XCUT, BY - 3.0, XCUT, IY + IH / 2, color=C_BAD, lw=1.3)
# No label on this arrow: the inference box already reads "discard $z_A$", and
# a rotated word here lands on top of the VIB box.
arrow(ax, centres[0], BY, 12.0 + wi1 * 0.3, IY + IH, color=C_OURS, lw=1.0)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig0_architecture.{ext}")
plt.close(fig)
print(f"  fig0_architecture -> {OUT}")
