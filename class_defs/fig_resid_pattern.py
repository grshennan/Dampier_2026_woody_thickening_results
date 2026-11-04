"""Figure: p98 residual time pattern 2019-2026 (resid_trend.py), 100 m, EPSG:7851.

Codes 1 consistently red / 2 consistently blue in strong red / blue (real: lift ~9-10 vs the shift null);
codes 3-6 (trend classes, near chance) in muted steps of the same two hues; 0 no pattern light grey;
255 outside the domain = surface. Legend carries class name + share of the domain (identity is never colour alone).
Colours: /dataviz reference palette (red slot 8, blue slot 1 + its sequential steps), ink from viz_style.
Output: outputs/resid_trend/fig_resid_pattern.png.
"""
import sys

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import common as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(C.PROJ / "chm_regression"))
import viz_style as V  # noqa: E402

SRC = r"E:\nisar_2026\regression\trend\p98_resid_pattern_2019_2026_100m.tif"
CLASSES = C.OUT / "resid_trend" / "classes_2019_2026.csv"
OUT = C.OUT / "resid_trend" / "fig_resid_pattern.png"
COLOURS = {0: V.GRID,       # no pattern
           1: "#e34948",    # consistently red (short for its cover)
           2: "#2a78d6",    # consistently blue (tall for its cover)
           3: "#6da7ec",    # red -> blue
           4: "#b7d3f6",    # toward blue
           5: "#ec8c8b",    # blue -> red
           6: "#f5c4c3"}    # toward red


def main():
    V.apply_style()
    with rasterio.open(SRC) as s:
        a = s.read(1)
        b = s.bounds
        epsg = s.crs.to_epsg()
    assert epsg == 7851, epsg
    m = np.ma.masked_equal(a, 255)
    cm = ListedColormap([COLOURS[k] for k in range(7)])
    cm.set_bad(V.SURFACE, 0)
    cl = pd.read_csv(CLASSES, index_col=0)
    fig, ax = plt.subplots(figsize=(9, 8.6))
    ax.imshow(m, cmap=cm, vmin=-0.5, vmax=6.5, extent=(b.left, b.right, b.bottom, b.top), interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    order = [1, 2, 3, 4, 5, 6, 0]
    handles = [Patch(facecolor=COLOURS[k], edgecolor=V.SURFACE,
                     label=f"{k} {n}  {100 * cl.loc[n, 'share']:.2f} %")
               for k in order for n in [cl.index[cl.code == k][0]]]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, title="class, share of domain",
                    title_fontsize=9, labelcolor=V.TEXT_PRIMARY)
    leg.get_frame().set_edgecolor(V.GRID)
    ax.set_title("p98 residual pattern 2019–2026, 100 m cells, EPSG:7851\n"
                 "red = short for its cover, blue = tall for its cover (per-year rank ≤ 25 / ≥ 75)",
                 fontsize=10, color=V.TEXT_PRIMARY)
    ls = cl.share / cl.null_shift
    lo = (cl.share / cl.null_year_order)[cl.code >= 3]
    fig.text(0.5, 0.02, f"Codes 1–2: lift {ls.iloc[1]:.1f} / {ls.iloc[2]:.1f} vs shift null (real). "
             f"Codes 3–6: lift {lo.min():.1f}–{lo.max():.1f} vs year-order null (near chance). Not a change map.",
             ha="center", fontsize=8.5, color=V.TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT, dpi=150, facecolor=V.SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
