"""Canopy gap fraction against canopy height, 2026 Dampier drone LiDAR, PALSAR grid.

Pairs, cell by cell, the outputs of two earlier scripts (no recomputation):
    outputs/gap_fraction/gap_<site>.tif   gap_pts_<h> (first-return gap below h)
    outputs/chm_targets_<site>.tif        CHM mean / max / p90 / p98, count
Both are on the same 24.2831 m PALSAR 2025 grid (asserted). Cells that are nodata
in either raster are dropped. MVT sites are excluded by the earlier scripts.

R2 is the squared Pearson r of a linear fit, all sites pooled and per site.

Outputs (outputs/gap_fraction/):
    gap_vs_height_r2.csv    R2, r, slope, intercept, n by site, h and CHM statistic
    gap_vs_height.png       scatter, rows = CHM mean / p98, columns = h

Usage
-----
conda run -n general --no-capture-output python gap_vs_height.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from chm_class_histograms import EXCLUDE_SITES, HERE
from chm_targets import STATS
from gap_fraction import BANDS, HEIGHTS
import viz_style

import matplotlib.pyplot as plt  # noqa: E402  (after viz_style sets Agg)

GAP_DIR = HERE / 'outputs' / 'gap_fraction'
PLOT_STATS = ['mean', 'p98']


def read(path) -> tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(path) as r:
        return r.read(masked=True).filled(np.nan), r.transform


def load_site(site: str) -> pd.DataFrame:
    g, tg = read(GAP_DIR / f'gap_{site}.tif')
    c, tc = read(HERE / 'outputs' / f'chm_targets_{site}.tif')
    assert tg == tc and g.shape[1:] == c.shape[1:], f'{site}: grids differ'
    df = pd.DataFrame(np.concatenate([g, c]).reshape(len(g) + len(c), -1).T,
                      columns=BANDS + STATS + ['count']).dropna()
    return df.assign(site=site)


def fit(x: pd.Series, y: pd.Series) -> dict:
    slope, icpt = np.polyfit(x, y, 1)
    r = x.corr(y)
    return {'n': len(x), 'r': r, 'r2': r * r, 'slope': slope, 'intercept': icpt}


def main():
    sites = sorted(f.stem[4:] for f in GAP_DIR.glob('gap_*.tif') if not f.stem.endswith('.partial'))
    sites = [s for s in sites if s not in EXCLUDE_SITES]
    cells = pd.concat([load_site(s) for s in sites])
    print(f'[gvh] {len(cells):,} cells, {len(sites)} sites')

    rows = [{'site': site, 'h': h, 'chm_stat': st, **fit(d[st], d[f'gap_pts_{h:g}'])}
            for site, d in [('all_sites', cells)] + list(cells.groupby('site'))
            for h in HEIGHTS for st in STATS]
    r2 = pd.DataFrame(rows)
    r2.to_csv(GAP_DIR / 'gap_vs_height_r2.csv', index=False, float_format='%.4f')
    print(r2[r2.site == 'all_sites'].to_string(index=False, float_format='%.3f'))
    print(r2.pivot_table(index='site', columns=['chm_stat', 'h'], values='r2')
          [PLOT_STATS].to_string(float_format='%.2f'))

    viz_style.apply_style()
    fig, axes = plt.subplots(len(PLOT_STATS), len(HEIGHTS), figsize=(4 * len(HEIGHTS), 3.9 * len(PLOT_STATS)),
                             constrained_layout=True, sharey=True, sharex='row')
    for row, st in zip(axes, PLOT_STATS):
        color = viz_style.TARGET_COLORS[st]
        for ax, h in zip(row, HEIGHTS):
            x, y = cells[st], cells[f'gap_pts_{h:g}']
            ax.scatter(x, y, s=2, color=color, alpha=0.15, linewidths=0, rasterized=True)
            f = r2[(r2.site == 'all_sites') & (r2.h == h) & (r2.chm_stat == st)].iloc[0]
            xs = np.array([0, x.max()])
            ax.plot(xs, f.intercept + f.slope * xs, color=viz_style.TEXT_PRIMARY, linewidth=1.2)
            ax.text(0.97, 0.97, f'R² = {f.r2:.2f}', transform=ax.transAxes, ha='right', va='top',
                    fontsize=10, color=viz_style.TEXT_PRIMARY)
            ax.set(ylim=(0, 1), xlim=(0, None), xlabel=f'{viz_style.TARGET_LABELS[st]} (m)')
            ax.set_title(f'Gap below {h:g} m', fontsize=11)
        row[0].set_ylabel('Gap fraction (first returns)')
    fig.suptitle(f'Gap fraction against canopy height: 2026 LiDAR, {len(sites)} sites, '
                 f'{len(cells):,} PALSAR cells (24.28 m)', color=viz_style.TEXT_PRIMARY, fontsize=12)
    fig.text(0.01, -0.01, 'Line: least-squares linear fit; R² of that fit, all sites pooled. '
             'Per-site values in gap_vs_height_r2.csv. MVT sites excluded.',
             fontsize=8, color=viz_style.TEXT_MUTED, va='top')
    out = GAP_DIR / 'gap_vs_height.png'
    fig.savefig(out)
    print(f'[gvh] wrote {out.resolve()}')


if __name__ == '__main__':
    main()
