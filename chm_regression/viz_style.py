"""Shared figure style for the CHM regression stages (4, 6, 7).

PLAN.md requires that every chart in this project read as one system, and that
the mean / max / p98 / p90 colour encoding be identical wherever it appears.
This module is the single place that encoding is defined -- import it, never
re-declare colours in a stage script.

Palette provenance
------------------
Slots 1-4 of the `/dataviz` skill's validated categorical palette, in fixed
order. Validated as a 4-slot subset with the skill's own validator:

    light (surface #fcfcfb): all checks PASS
        worst adjacent CVD dE 9.1 (protan), normal-vision dE 22.9
        WARN contrast vs surface: aqua 2.74:1, yellow 2.11:1 -- below 3:1
    dark  (surface #1a1a19): all checks PASS, all slots >= 3:1

The light-mode contrast WARN is **not dismissable**: it obligates the "relief
rule" -- any figure using aqua or yellow as a line/area must carry visible
direct labels (or a table view). `annotate_line_ends()` below exists to
discharge that obligation; use it on every multi-series line figure.

Slot order is deliberate: p90 (yellow) and max (orange) are never adjacent in
the assigned order, which is what keeps the adjacent-pair CVD check passing.
Do not reorder.
"""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')  # headless, per PLAN.md

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# The fixed target -> colour encoding. Stages 4, 6 and 7 all use this.
# ---------------------------------------------------------------------------

TARGETS = ['mean', 'max', 'p98', 'p90']

TARGET_COLORS = {
    'mean': '#2a78d6',   # slot 1, blue
    'max':  '#eb6834',   # slot 2, orange
    'p98':  '#1baf7a',   # slot 3, aqua
    'p90':  '#eda100',   # slot 4, yellow
}

TARGET_COLORS_DARK = {
    'mean': '#3987e5',
    'max':  '#d95926',
    'p98':  '#199e70',
    'p90':  '#c98500',
}

# Slots that failed the 3:1 contrast check on the light surface. Figures using
# these as lines must carry direct labels -- see annotate_line_ends().
LOW_CONTRAST_TARGETS = {'p98', 'p90'}

TARGET_LABELS = {
    'mean': 'CHM mean',
    'max':  'CHM max',
    'p98':  'CHM p98',
    'p90':  'CHM p90',
}

SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
TEXT_MUTED = '#7a7973'
GRID = '#e3e2dd'

# A single-hue sequential ramp, for magnitude (e.g. hexbin density). One hue,
# light -> dark, never a rainbow.
SEQUENTIAL_CMAP = 'Blues'


def apply_style() -> None:
    """Recessive grid/axes, thin marks, text in ink tokens rather than series
    colour. Call once at the top of a stage script."""
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID,
        'axes.labelcolor': TEXT_SECONDARY,
        'axes.titlecolor': TEXT_PRIMARY,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'xtick.color': TEXT_SECONDARY,
        'ytick.color': TEXT_SECONDARY,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.frameon': False,
        'legend.fontsize': 9,
        'lines.linewidth': 2.0,       # 2px lines
        'lines.markersize': 8,        # >= 8px markers
        'figure.dpi': 130,
        'savefig.bbox': 'tight',
        'font.size': 10,
    })
    # Recessive axes: drop the top/right spines everywhere.
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


def color_for(target: str) -> str:
    return TARGET_COLORS[target]


def annotate_line_ends(ax, x, y, target, *, dx=0.15, fontsize=9):
    """Direct-label a series at its right-hand end.

    Discharges the light-mode relief obligation for the aqua/yellow slots, and
    satisfies the skill's rule that identity is never carried by colour alone.
    Text wears an ink token, not the series colour.
    """
    ax.annotate(
        TARGET_LABELS.get(target, target),
        xy=(x[-1], y[-1]),
        xytext=(x[-1] + dx, y[-1]),
        va='center', ha='left',
        fontsize=fontsize, color=TEXT_SECONDARY,
        annotation_clip=False,
    )
