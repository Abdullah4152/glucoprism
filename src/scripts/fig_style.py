"""Shared figure style: one palette, one grid, one legend convention.

Modelled on the GlucoFM figure family the paper sits beside -- muted
categorical hues, a light y-grid, no top/right spines, and the legend BELOW the
panel row rather than inside any panel. Nothing is allowed to print on top of
the data: every legend and annotation goes in reserved space.

Import this from the figure scripts rather than re-declaring colours; the whole
point is that the reader sees one system across a dozen floats.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ palette
# Sampled from the reference figures. ORANGE is reserved for the released
# model, so "ours" is the same hue in every float; RED is reserved for the
# Sensor block and the one repelling relation, matching Figure 1.
ORANGE = "#f0871f"      # GlucoPRISM-C  (released, highlighted)
CORAL = "#e8646b"       # GlucoPRISM-E  (second released model)
NAVY = "#33518c"        # GlucoFM       (the backbone we build on)
SKY = "#5bc0e0"
PURPLE = "#8b7fc4"
TEAL = "#2eb39b"
PINK = "#f09aa4"
GREEN = "#4c9f70"
GREY = "#9aa0a6"
GREY_L = "#c8ccd0"
INK = "#33383d"
RED = "#c0392b"         # the deleted block / "pushes apart"

# A stable order for multi-model panels: ours first, backbone next, then the
# zero-shot field. Colour-blind safe on the pairs that ever sit adjacent.
SERIES = [ORANGE, CORAL, NAVY, TEAL, PURPLE, SKY, PINK, GREEN, GREY]

GRID = "#e6e8ea"
FACE = "#fbfbfc"


def use_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 6.4,
        "axes.linewidth": 0.6,
        "axes.edgecolor": GREY_L,
        "axes.facecolor": FACE,
        "axes.titlesize": 7.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 6.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 5.9,
        "ytick.labelsize": 5.9,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        "legend.frameon": False,
        "legend.fontsize": 5.9,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.3,
        "figure.dpi": 400,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
    })


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def letters(fig, axs, fs=7.0, pad=0.012):
    """Panel letters below each panel, placed in FIGURE coordinates.

    Axes coordinates do not work here: the drop below an axis varies with
    whether that panel has an x-label and how many lines its tick labels take,
    so a fixed axes-relative offset lands on the label in some panels and floats
    in others. Measuring each panel's tight bounding box puts every letter the
    same distance below whatever that panel actually ends with.

    Call after `tight_layout`, before `rowlegend`. Returns the lowest y used, so
    the legend can be placed just beneath.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bbs = [ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
           for ax in axs]
    # One shared baseline: panels in a row differ in whether they carry an
    # x-label, and letters stepped up and down across the row look like a
    # mistake even though each was correctly placed under its own panel.
    y = min(b.y0 for b in bbs) - pad
    for i, b in enumerate(bbs):
        fig.text(0.5 * (b.x0 + b.x1), y, f"({chr(97 + i)})", ha="center",
                 va="top", fontsize=fs, fontweight="bold", color=INK)
    return y


def below(fig, axs):
    """Lowest figure-y reached by these panels, labels included.

    For single-panel figures that want a legend underneath but no letter.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    return min(ax.get_tightbbox(r)
               .transformed(fig.transFigure.inverted()).y0 for ax in axs)


def rowlegend(fig, handles, labels, ncol=None, y=0.0, fs=5.9):
    """One legend for the whole row, under it, never over the data."""
    fig.legend(handles, labels, loc="upper center", ncol=ncol or len(labels),
               bbox_to_anchor=(0.5, max(y, 0.0)), frameon=False, fontsize=fs,
               handlelength=1.4, handletextpad=0.5, columnspacing=1.4)


def save(fig, name, out):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")
