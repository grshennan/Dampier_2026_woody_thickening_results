"""Stage 2 figures (PLAN.md). Style: chm_regression/viz_style.py (/dataviz rules).

fig_space_k.png     cover137 x p98_can density (single-hue ramp, log count) of the canopy cells; panel 1 has
                    the 4classTF cuts as iso-lines of the binned median CHM mean (1.22 / 2.35 / 5.381 m);
                    the other panels the GMM class boundaries for k = 2..6, class numbers at the class medians.
fig_k_selection.png BIC / ICL, stability ARI (bootstrap, leave-one-area-out; 0.8 rule), ARI vs 4classTF with
                    its block-shuffle null. Two series per panel: ink solid / ink dashed, direct-labelled.

Run from this directory:
    conda run -n general --no-capture-output python -u figures_2d.py [--classes outputs/classes]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'chm_regression'))
import viz_style as V  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (after viz_style sets Agg)
from matplotlib import patheffects  # noqa: E402

COVER_TICKS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
P98_TICKS = [1.5, 2, 3, 5, 8, 12, 20]
CUTS4 = [1.22, 2.35, 5.381]
HALO = [patheffects.withStroke(linewidth=3, foreground=V.SURFACE)]


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def axes_ticks(ax):
    ax.set_xticks(logit(np.array(COVER_TICKS)), [f'{t:.0%}' for t in COVER_TICKS])
    ax.set_yticks(np.log(P98_TICKS), [f'{t:g}' for t in P98_TICKS])
    ax.set_xlabel('cover at 1.37 m (logit axis)')
    ax.set_ylabel('p98 of canopy returns (m, log axis)')


def density(ax, x, y, ext):
    return ax.hexbin(x, y, gridsize=70, extent=ext, bins='log', cmap=V.SEQUENTIAL_CMAP, mincnt=1, linewidths=0)


def footnote(fig, src):
    fig.text(0.01, -0.01, f'Source: {src}. Canopy cells only (n_can ≥ 20, cover ≥ 1 %).', fontsize=8,
             color=V.TEXT_MUTED, ha='left', va='top')


def space_figure(c, mdl, out, src):
    x, y = logit(c.cover137.to_numpy()), np.log(c.p98_can.to_numpy())
    ext = (*np.percentile(x, [0.2, 99.8]), *np.percentile(y, [0.2, 99.8]))
    gx, gy = np.meshgrid(np.linspace(ext[0], ext[1], 300), np.linspace(ext[2], ext[3], 300))
    g = (np.c_[gx.ravel(), gy.ravel()] - mdl['med']) / mdl['iqr']
    ks = [k for k in (2, 3, 4, 5, 6) if k in mdl['models']]
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.4), constrained_layout=True, sharex=True, sharey=True)
    axs = axs.ravel()
    hb = density(axs[0], x, y, ext)
    # 4classTF cuts: iso-lines of the median CHM mean in 2-D bins
    bx, by = np.linspace(ext[0], ext[1], 41), np.linspace(ext[2], ext[3], 41)
    ix, iy = np.digitize(x, bx) - 1, np.digitize(y, by) - 1
    med = pd.Series(c.mean_chm.to_numpy()).groupby([iy, ix]).median()
    cnt = pd.Series(c.mean_chm.to_numpy()).groupby([iy, ix]).size()
    grid = np.full((40, 40), np.nan)
    for (r, q), v in med[cnt >= 10].items():
        if 0 <= r < 40 and 0 <= q < 40:
            grid[r, q] = v
    cs = axs[0].contour((bx[:-1] + bx[1:]) / 2, (by[:-1] + by[1:]) / 2, grid, levels=CUTS4,
                        colors=V.TEXT_PRIMARY, linewidths=1.2)
    axs[0].clabel(cs, fmt=lambda v: f'{v:g} m', fontsize=8, colors=V.TEXT_PRIMARY)
    axs[0].set_title('4classTF cuts (median CHM mean, iso-lines)')
    for ax, k in zip(axs[1:], ks):
        density(ax, x, y, ext)
        cls = mdl['rank'][k][mdl['models'][k].predict(g)].reshape(gx.shape)
        ax.contour(gx, gy, cls, levels=np.arange(k) + 0.5, colors=V.TEXT_PRIMARY, linewidths=1.0)
        lab = mdl['rank'][k][mdl['models'][k].predict((np.c_[x, y] - mdl['med']) / mdl['iqr'])]
        for i in range(k):
            ax.text(np.median(x[lab == i]), np.median(y[lab == i]), f'S{i + 1}', ha='center', va='center',
                    fontsize=10, color=V.TEXT_PRIMARY, path_effects=HALO)
        ax.set_title(f'GMM, k = {k}')
    for ax in axs[1 + len(ks):]:
        ax.set_visible(False)
    for ax in axs:
        ax.grid(False)
        axes_ticks(ax)
    fig.colorbar(hb, ax=axs.tolist(), shrink=0.6, label='cells per hexagon (log)')
    footnote(fig, src)
    fig.savefig(out)
    print(f'wrote {out.resolve()}', flush=True)


def k_figure(ks, ari, out, src):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    pairs = [(axs[0], [('bic', 'BIC', '-'), ('icl', 'ICL', '--')], 'criterion (lower is better)', ks),
             (axs[1], [('boot_ari_mean', 'block bootstrap', '-'), ('loso_ari_mean', 'leave one area out', '--')],
              'stability, mean ARI', ks[ks.k >= 2])]
    for ax, series, ylab, d in pairs:
        for col, name, ls in series:
            ax.plot(d.k, d[col], ls, color=V.TEXT_PRIMARY if ls == '-' else V.TEXT_SECONDARY, marker='o', markersize=4)
            ax.annotate(name, (d.k.iloc[-1], d[col].iloc[-1]), xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=8, color=V.TEXT_SECONDARY, annotation_clip=False)
        ax.set_ylabel(ylab)
    axs[1].axhline(0.8, color=V.TEXT_MUTED, lw=0.8)
    axs[1].annotate('support rule 0.8', (ks.k.min() + 1, 0.8), xytext=(0, 4), textcoords='offset points',
                    fontsize=8, color=V.TEXT_MUTED)
    axs[2].plot(ari.k, ari.ari_vs_4classTF, '-', color=V.TEXT_PRIMARY, marker='o', markersize=4)
    axs[2].plot(ari.k, ari.null_mean, '--', color=V.TEXT_SECONDARY)
    axs[2].annotate('observed', (ari.k.iloc[-1], ari.ari_vs_4classTF.iloc[-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=8, color=V.TEXT_SECONDARY, annotation_clip=False)
    axs[2].annotate('block-shuffle null', (ari.k.iloc[-1], ari.null_mean.iloc[-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=8, color=V.TEXT_SECONDARY, annotation_clip=False)
    axs[2].set_ylabel('ARI against 4classTF')
    for ax in axs:
        ax.set_xlabel('number of classes k')
        ax.set_xticks(ks.k)
    footnote(fig, src)
    fig.savefig(out)
    print(f'wrote {out.resolve()}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', type=Path, default=HERE / 'outputs' / 'classes')
    ap.add_argument('--cells', type=Path, default=HERE / 'outputs' / 'cells_2d.parquet')
    a = ap.parse_args()
    V.apply_style()
    fig_dir = a.classes / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    c = pd.read_parquet(a.cells)
    c = c[c.canopy].reset_index(drop=True)
    with open(a.classes / 'gmm_models.pkl', 'rb') as fh:
        mdl = pickle.load(fh)
    src = f'{a.cells.name}, {c.site.nunique()} areas, {len(c):,} cells'
    space_figure(c, mdl, fig_dir / 'fig_space_k.png', src)
    k_figure(pd.read_csv(a.classes / 'k_selection.csv'), pd.read_csv(a.classes / 'ari_vs_4classTF.csv'),
             fig_dir / 'fig_k_selection.png', src)


if __name__ == '__main__':
    main()
