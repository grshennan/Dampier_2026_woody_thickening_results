"""Scatter of px mean against cover at h, with the linear cover fit and the local medians.

Panels: the gmm50_k4 heights (0.258, 4.143, 8.187 m) and the nfi heights (2, 5 m). Per panel: every label px
(grey), the OLS line over all px, medians of px mean in 0.1 cover bins (blue), and the chosen local median at
18-22 % cover (orange; the proposed cut). Cut values from the line (at 20 %) and the local median are in the title.
Output: outputs/fig_cover_fit.png.

    C:\\Users\\glens\\miniconda3\\Scripts\\conda.exe run -n general --no-capture-output python -u fig_cover_fit.py
"""
import sys

import numpy as np
import pandas as pd

from common import COVER, OUT, PROJ, col

sys.path.insert(0, str(PROJ / "chm_regression"))
import viz_style as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [(0.258, "gmm50_k4"), (4.143, "gmm50_k4"), (8.187, "gmm50_k4"), (2.0, "nfi"), (5.0, "nfi")]
BLUE, ORANGE = V.TARGET_COLORS["mean"], V.TARGET_COLORS["max"]


def main():
    V.apply_style()
    t = pd.read_parquet(COVER)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.5), sharey=True)
    for ax, (h, s) in zip(axs.flat, PANELS):
        c, y = t[col(h)].to_numpy(), t["mean"].to_numpy()
        k, a = np.polyfit(c, y, 1)
        bins = pd.cut(pd.Series(c), np.arange(0, 1.01, 0.1), include_lowest=True)
        med = pd.Series(y).groupby(bins, observed=True).median()
        near = (c >= 0.18) & (c <= 0.22)
        loc, fit = float(np.median(y[near])), a + k * 0.2
        ax.scatter(c, y, s=1, color=V.TEXT_MUTED, alpha=0.12, linewidths=0, rasterized=True, label="label px")
        ax.plot([0, 1], [a, a + k], color=V.TEXT_PRIMARY, lw=2, label="linear fit (all px)")
        ax.axvline(0.2, color=V.TEXT_SECONDARY, ls="--", lw=1)
        ax.scatter([i.mid for i in med.index], med, s=70, color=BLUE, edgecolors=V.SURFACE, linewidths=2,
                   zorder=3, label="median, 0.1 cover bins")
        ax.scatter([0.2], [loc], s=160, color=ORANGE, edgecolors=V.SURFACE, linewidths=2, zorder=4,
                   label="median at 18-22 % cover (chosen)")
        ax.set_title(f"{s}: h = {h:g} m\nfit at 20 %: {fit:.3f} m | local median: {loc:.3f} m (n {near.sum():,})",
                     loc="left", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"cover at h = {h:g} m (fraction of 50 cm cells >= h)")
        ax.set_ylabel("px mean of 50 cm p98 CHM (m)")
        ax.yaxis.set_tick_params(labelleft=True)
    axs.flat[-1].axis("off")
    axs.flat[0].set_ylim(bottom=0)
    h_, l_ = axs.flat[0].get_legend_handles_labels()
    axs.flat[-1].legend(h_, l_, loc="center", frameon=False, markerscale=1.2)
    fig.suptitle("Px mean against cover at h: linear cover fit vs local median at 20 % cover "
                 "(2026 LiDAR, 5 sites, 34,715 label px, NISAR 20 m)", x=0.01, ha="left")
    fig.tight_layout()
    out = OUT / "fig_cover_fit.png"
    fig.savefig(out, dpi=150)
    print(f"written: {out.resolve()}")


if __name__ == "__main__":
    main()
