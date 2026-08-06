"""Chart styling for the competition report.

One palette, applied everywhere, so every figure in the writeup reads as one
system. Values come from the validated reference palette (light surface).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, in fixed order — never cycled past slot 8.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Sequential blue, 100 -> 700. Used for magnitude (heatmaps), never for identity.
SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
CMAP_BLUE = LinearSegmentedColormap.from_list("ptcg_blue", SEQ_BLUE)


def use_style() -> None:
    """Apply the report style to matplotlib globally."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })


def despine(ax, keep=("left", "bottom")) -> None:
    """Strip chart junk — keep only the axes the reader needs."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def subtitle(ax, text: str) -> None:
    """A caption under the title that says what the reader should take away."""
    ax.set_title(ax.get_title(), pad=24)
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9,
            color=INK_MUTED, va="bottom", ha="left")


def label_bars_h(ax, values, labels=None, pad=0.01, fmt="{:.0f}") -> None:
    """Direct-label horizontal bars, so the reader never counts gridlines."""
    span = max(values) if len(values) else 1
    for i, v in enumerate(values):
        txt = fmt.format(v) if labels is None else labels[i]
        ax.text(v + span * pad, i, txt, va="center", ha="left",
                fontsize=9, color=INK_SECONDARY)
