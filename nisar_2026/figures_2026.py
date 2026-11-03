"""Stage 5 figures for the 2026 classification and regression (PLAN.md). Style and colours from
chm_regression/viz_style.py; the five height classes are ordered, so they take one sequential
ramp (light = bare, dark = tall forest), not categorical hues.

    fig_confusion.png        row-normalised confusion matrices, naive and K-fold
    fig_regression.png       predicted vs observed density, 4 targets x (naive, K-fold)
    fig_importance.png       top-20 classifier feature importance (gain)
    fig_maps.png             class map and RF mean-height map (8x decimated quick look)

Run from this directory:
    conda run -n general --no-capture-output python figures_2026.py [--quick]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chm_regression"))
import viz_style as V  # noqa: E402

sys.path.insert(0, str(HERE))
from labels_2026 import BOUNDS, CLASSES  # noqa: E402

OUT = HERE / "outputs"
QUICK = "--quick" in sys.argv          # figures of the --quick test outputs
SUB = "quick" if QUICK else ""
FIG = OUT / "figures" / SUB
TARGETS = ["mean", "p90", "p98", "max"]
EVALS = {"naive": "naive (resubstitution)", "kfold": "random 5-fold"}
CLASS_RAMP = ListedColormap(plt.get_cmap("Greens")(np.linspace(0.25, 1.0, len(CLASSES))))
DENSITY = ListedColormap(plt.get_cmap(V.SEQUENTIAL_CMAP)(np.linspace(0.3, 1.0, 256)))  # lowest bin visible
HEIGHT = ListedColormap(plt.get_cmap("Greens")(np.linspace(0.15, 1.0, 256)))  # 0 m stays off the surface
CLASS_TICKS = [f"{c}\n" + (f"< {BOUNDS[0]:g}" if i == 0 else f"≥ {BOUNDS[-1]:g}" if i == len(CLASSES) - 1
                            else f"{BOUNDS[i - 1]:g}–{BOUNDS[i]:g}") for i, c in enumerate(CLASSES)]


def confusion():
    met = pd.read_csv(OUT / "classify" / SUB / "metrics.csv").query("site == 'all'").set_index("eval")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    for ax, k in zip(axes, EVALS):
        cm = pd.read_csv(OUT / "classify" / SUB / f"confusion_{k}.csv", index_col=0).to_numpy()
        rec = cm / cm.sum(1, keepdims=True)
        ax.imshow(rec, cmap=V.SEQUENTIAL_CMAP, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{rec[i, j]:.0%}\n{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                        color="white" if rec[i, j] > 0.55 else V.TEXT_PRIMARY)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("LiDAR label")
        ax.grid(False)
        m = met.loc[k]
        ax.set_title(f"{EVALS[k]}: OA {m.oa:.3f}, balanced {m.balanced_acc:.3f}, κ {m.kappa:.3f}", loc="left")
    fig.suptitle("5-class XGBoost, 2026 (row-normalised: recall)", x=0.01, ha="left", color=V.TEXT_PRIMARY)
    fig.savefig(FIG / "fig_confusion.png")


def regression():
    pr = pd.read_parquet(OUT / "regress" / SUB / "predictions.parquet")
    met = pd.read_csv(OUT / "regress" / SUB / "metrics.csv").query("site == 'all'").set_index(["target", "eval"])
    targets = [t for t in TARGETS if f"pred_{t}_naive" in pr]
    fig, axes = plt.subplots(2, len(targets), figsize=(3.6 * len(targets), 8), squeeze=False, layout="constrained")
    for j, t in enumerate(targets):
        top = float(np.nanpercentile(pr[t], 99.9)) * 1.05
        for i, k in enumerate(EVALS):
            ax = axes[i, j]
            ax.hexbin(pr[t], pr[f"pred_{t}_{k}"], gridsize=50, extent=(0, top, 0, top), bins="log",
                      cmap=DENSITY, mincnt=1, linewidths=0)
            ax.plot([0, top], [0, top], color=V.TEXT_MUTED, lw=1, ls="--")
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.set_aspect("equal")
            m = met.loc[(t, k)]
            ax.set_title(f"{V.TARGET_LABELS[t]}, {k}\nRMSE {m.rmse:.2f} m, R² {m.r2:.2f}", loc="left", fontsize=10)
            ax.set_xlabel("LiDAR (m)")
            if j == 0:
                ax.set_ylabel("predicted (m)")
    fig.suptitle("RF regression, 2026 — density of 20 m pixels (log)", x=0.01, ha="left", color=V.TEXT_PRIMARY)
    fig.savefig(FIG / "fig_regression.png")


def importance():
    imp = pd.read_csv(OUT / "classify" / SUB / "importance.csv", index_col=0).iloc[:, 0].head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.barh(imp.index, imp.values, color=V.color_for("mean"), height=0.7)
    ax.set_xlabel("XGBoost importance (gain share)")
    ax.grid(axis="y", visible=False)
    ax.set_title("5-class XGBoost, 2026: top 20 features", loc="left")
    fig.savefig(FIG / "fig_importance.png")


def maps():
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), layout="constrained")
    with rasterio.open(OUT / "XGB_DP_NISAR_S2_2026_5class.tif") as s:
        c = s.read(1, out_shape=(s.height // 8, s.width // 8)).astype(float)
        ext = (s.bounds.left, s.bounds.right, s.bounds.bottom, s.bounds.top)
    c[c == 255] = np.nan
    axes[0].imshow(c, cmap=CLASS_RAMP, norm=BoundaryNorm(np.arange(-0.5, len(CLASSES)), CLASS_RAMP.N),
                   extent=ext, interpolation="nearest")
    axes[0].legend(handles=[Patch(color=CLASS_RAMP(i), label=l.replace("\n", "  ")) for i, l in enumerate(CLASS_TICKS)],
                   title="mean of p98 (m)", loc="upper left", bbox_to_anchor=(0, -0.01), ncol=3)
    axes[0].set_title("5-class XGBoost", loc="left")
    with rasterio.open(OUT / "RF_CHM_2026.tif") as s:
        h = s.read(list(s.descriptions).index("mean") + 1, out_shape=(s.height // 8, s.width // 8), masked=True)
    im = axes[1].imshow(h, cmap=HEIGHT, vmin=0, vmax=float(np.percentile(h.compressed(), 99)), extent=ext,
                        interpolation="nearest")
    fig.colorbar(im, ax=axes[1], shrink=0.6, label="CHM mean (m)")
    axes[1].set_title("RF regression, CHM mean", loc="left")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Dampier Peninsula mainland, 2026 (20 m, shown 8x decimated)", x=0.01, ha="left",
                 color=V.TEXT_PRIMARY)
    fig.savefig(FIG / "fig_maps.png")


if __name__ == "__main__":
    V.apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (confusion, regression, importance, maps):
        try:
            f()
            print(f"{f.__name__}: done", flush=True)
        except (FileNotFoundError, rasterio.errors.RasterioIOError) as e:
            print(f"{f.__name__}: skipped, missing input ({e})", flush=True)
    print(f"figures in {FIG.resolve()}", flush=True)
