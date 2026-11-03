"""Figure (p98_cover.py): direct RF p98 map and the 5 x 5 px residual p98 - E[p98 | cover].

Left: sequential (viz_style.SEQUENTIAL_CMAP), colour p1-p99. Right: diverging RdBu, symmetric at the p99 of |res|,
grey midpoint at 0; red = shorter than the label cover-height relation expects (thickening candidate).
Read decimated x4 (80 m display px). Output: outputs/p98_cover/fig_p98_resid.png.
"""
import sys

import matplotlib
import numpy as np
import rasterio
from rasterio.enums import Resampling

import common as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(C.PROJ / "chm_regression"))
import viz_style as V  # noqa: E402
from regression_2026 import MAPD, NODATA  # noqa: E402

OUT = C.OUT / "p98_cover" / "fig_p98_resid.png"


def read(name):
    with rasterio.open(MAPD / name) as s:
        a = s.read(1, out_shape=(s.height // 4, s.width // 4), resampling=Resampling.nearest)
        b = s.bounds
    return np.where(a == NODATA, np.nan, a), (b.left, b.right, b.bottom, b.top)


def main():
    V.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.4))
    p, ext = read("height_p98_ALL.tif")
    r, _ = read("p98_resid_w5_ALL.tif")
    lo, hi = np.nanpercentile(p, [1, 99])
    lim = float(np.nanpercentile(np.abs(r), 99))
    panels = [(p, V.SEQUENTIAL_CMAP, lo, hi, "p98 height (m): RF, ALL stack\ncolour p1–p99"),
              (r, "RdBu", -lim, lim, "p98 − E[p98 | cover] (m), 5 × 5 px mean\n"
                                    "red = short for its cover (thickening candidate)")]
    for a, (arr, cmap, vmin, vmax, title) in zip(ax, panels):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(V.GRID)
        im = a.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
        a.set_title(title, fontsize=10, color=V.TEXT_PRIMARY)
        a.set_xticks([]), a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=a, fraction=0.035, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=V.TEXT_SECONDARY, labelsize=8)
    fig.suptitle("Dampier Peninsula 2026, 20 m (shown at 80 m), EPSG:32751; single date, not a change map",
                 fontsize=10, color=V.TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=V.SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
