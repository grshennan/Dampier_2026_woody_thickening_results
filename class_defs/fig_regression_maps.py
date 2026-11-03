"""Figure (PLAN_regression.md): ALL-stack px-mean height and cover 1.37 maps side by side.

Sequential single hue (viz_style.SEQUENTIAL_CMAP), one colour bar per panel, colour clipped at p1-p99 of the valid
px, nodata = grid grey. Read decimated x4 (80 m display px). Output: outputs/regression/fig_maps_ALL.png.
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
from regression_2026 import MAPD, MAPS, NODATA, OUTR  # noqa: E402

TITLE = {"mean": "Height: px mean of 50 cm CHM (m)", "cover": "Cover at 1.37 m (fraction)"}


def main():
    V.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.4))
    for a, (t, (name, _)) in zip(ax, MAPS.items()):
        with rasterio.open(MAPD / name) as s:
            arr = s.read(1, out_shape=(s.height // 4, s.width // 4), resampling=Resampling.nearest)
            b = s.bounds
        arr = np.where(arr == NODATA, np.nan, arr)
        lo, hi = np.nanpercentile(arr, [1, 99])
        cm = plt.get_cmap(V.SEQUENTIAL_CMAP).copy()
        cm.set_bad(V.GRID)
        im = a.imshow(arr, cmap=cm, vmin=lo, vmax=hi, extent=(b.left, b.right, b.bottom, b.top),
                      interpolation="nearest")
        a.set_title(f"{TITLE[t]}\nRF, ALL stack (NISAR + S1 + S2); colour p1–p99", fontsize=10,
                    color=V.TEXT_PRIMARY)
        a.set_xticks([]), a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=a, fraction=0.035, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=V.TEXT_SECONDARY, labelsize=8)
    fig.suptitle("Dampier Peninsula 2026, 20 m (shown at 80 m), EPSG:32751", fontsize=10, color=V.TEXT_SECONDARY)
    fig.tight_layout()
    out = OUTR / "fig_maps_ALL.png"
    fig.savefig(out, dpi=150, facecolor=V.SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
