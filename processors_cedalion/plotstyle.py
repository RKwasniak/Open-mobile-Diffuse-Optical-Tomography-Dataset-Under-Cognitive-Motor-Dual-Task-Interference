"""Shared, publication-quality matplotlib styling for the validation figures.

Call :func:`set_publication_style` once (e.g. at the top of the notebook) to give
every figure consistent typography, colours and spacing. The individual plotting
functions also call :func:`style_axes`, so they still look clean if the global
style was never set.

Chromophores use a red/teal pair; conditions and other categorical series use the
project palette below.
"""

from __future__ import annotations

import matplotlib as mpl

#: Project palette (warm → cool), used for conditions / categorical series.
CONDITION_PALETTE = (
    "#E84A00FF", "#BB1D2CFF", "#9B0C43FF", "#661F66FF",
    "#2C1F62FF", "#006289FF", "#004759FF",
)

#: Chromophore colours drawn from the palette (HbO warm red, HbR cool teal).
CHROMO_COLORS = {"HbO": "#BB1D2CFF", "HbR": "#006289FF"}

#: Neutral ink, axis-edge and grid colours used across figures.
INK = "#222222"
_EDGE = "#444444"
_GRID = "#9AA0A6"

#: Region labels for figure titles.
REGION_ABBR = {"FPA": "FPA"}

#: Short hemisphere labels.
HEMI_ABBR = {"left": "LH", "right": "RH", "both": ""}


def averaged_title(roi, hemisphere: str = "both", trial_types=("MC", "REST")) -> str:
    """Figure title ``region · [hemisphere ·] trial types``, e.g. ``"FPA · MC vs REST"``."""
    reg = REGION_ABBR.get(str(roi), str(roi))
    hemi = HEMI_ABBR.get(hemisphere, "")
    tt = " vs ".join(str(t) for t in trial_types)
    parts = [reg] + ([hemi] if hemi else []) + [tt]
    return " · ".join(parts)

_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica World", "Helvetica Neue", "Helvetica",
                        "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "figure.titleweight": "normal",
    "axes.titlepad": 10,
    "axes.labelsize": 11,
    "axes.labelcolor": INK,
    "axes.edgecolor": _EDGE,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=CONDITION_PALETTE),
    "grid.color": _GRID,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "legend.handlelength": 1.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
}


def set_publication_style() -> None:
    """Apply the shared publication style to matplotlib's global rcParams."""
    mpl.rcParams.update(_RC)


def style_axes(ax, grid_axis: str | None = "y"):
    """Despine, add a light grid and tidy ticks on a single Axes.

    ``grid_axis`` is ``"y"`` (default), ``"x"``, ``"both"`` or ``None``.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_EDGE)
        ax.spines[s].set_linewidth(0.9)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=_GRID, alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, colors=INK)
    return ax
