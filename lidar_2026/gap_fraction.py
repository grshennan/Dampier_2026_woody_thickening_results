"""Canopy gap fraction on the SAR grid, 2026 Dampier drone LiDAR: point cloud vs CHM.

For each non-MVT site, on the PALSAR 2025 grid (24.2831 m; the same cells as
outputs/chm_targets_<site>.tif), two estimates of gap fraction at h = 0.5 / 1.37 / 2 / 5 m:

    gap_pts_h   first-return gap: first returns with height < h / all first returns
                (the standard ALS Pgap; equal to lascanopy -gap -cover_cutoff h)
    gap_chm_h   CHM gap: 50 cm p98 CHM pixels < h / all valid CHM pixels in the cell
                (the cover definition of class_vs_regression/translation.py, as 1 - cover)

The p98 CHM includes ground returns (lascanopy -height_cutoff 0), so a 50 cm cell
between crowns holds the ground height (~0.05 m). One return above h fills a whole
50 cm cell, so gap_chm is expected to be lower than gap_pts at small h.

Input cloud: <site>_ground_gndthin_height_denoise.las beside the CHM (z = height above
ground, lasheight -replace_z), read in chunks with laspy. Noise classes 7 and 18 and
withheld points are dropped; points with z < 0 count as gap at every h. EPSG:7851 is
used on the EPSG:32751 grid without reprojection (as in chm_class_histograms.py).
⚠ CRS 2026-11-03: withdrawn (user); reprojection is required. G:/python/crs_audit/CHECKPOINT.md.
Cells with fewer than LOW_COUNT_THRESHOLD 50 cm CHM pixels are nodata in all bands.

Outputs (outputs/gap_fraction/):
    gap_<site>.tif          10 bands: gap_pts_*, gap_chm_*, n_first, n_chm
    gap_summary.csv         pooled and cell-mean gap per site, plus CHM - points stats
    gap_chm_vs_pts.png      CHM gap against point gap, one panel per h

Usage
-----
conda run -n general --no-capture-output python -u gap_fraction.py
conda run -n general --no-capture-output python -u gap_fraction.py --sites "Bunda Bunda"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from chm_class_histograms import (EXCLUDE_SITES, HERE, ROOT_WIN, SAR_REF_WIN, load_chm,
                                  site_name, to_local)
from chm_targets import LOW_COUNT_THRESHOLD, target_grid_for_chm
import viz_style

import matplotlib.pyplot as plt  # noqa: E402  (after viz_style sets Agg)

HEIGHTS = [0.5, 1.37, 2.0, 5.0]
BANDS = ([f'gap_pts_{h:g}' for h in HEIGHTS] + [f'gap_chm_{h:g}' for h in HEIGHTS]
         + ['n_first', 'n_chm'])
NOISE_CLASSES = [7, 18]
CHUNK = 10_000_000
NODATA = -9999.0


def grid_index(transform, xs: np.ndarray, ys: np.ndarray, shape) -> np.ndarray:
    """Flat SAR cell index of each (x, y); -1 outside the grid. Same floor mapping as
    chm_targets.aggregate_chm_to_grid."""
    cols = np.floor((xs - transform.c) / transform.a).astype('int64')
    rows = np.floor((ys - transform.f) / transform.e).astype('int64')
    inside = (cols >= 0) & (cols < shape[1]) & (rows >= 0) & (rows < shape[0])
    return np.where(inside, rows * shape[1] + cols, -1)


def chm_counts(chm, transform, shape) -> np.ndarray:
    """(1 + len(HEIGHTS), ncell): valid CHM pixels, then pixels >= h."""
    v = chm.values
    cols = np.floor((chm.x.values - transform.c) / transform.a).astype('int64')
    rows = np.floor((chm.y.values - transform.f) / transform.e).astype('int64')
    idx = rows[:, None] * shape[1] + cols[None, :]
    inside = ((rows[:, None] >= 0) & (rows[:, None] < shape[0])
              & (cols[None, :] >= 0) & (cols[None, :] < shape[1]))
    valid = np.isfinite(v)
    assert (valid & inside).sum() == valid.sum(), 'valid CHM pixels outside the SAR grid'
    idx, v = idx[valid], v[valid]
    n = shape[0] * shape[1]
    return np.stack([np.bincount(idx, minlength=n)]
                    + [np.bincount(idx, weights=v >= h, minlength=n) for h in HEIGHTS])


def point_counts(las_path: Path, transform, shape) -> np.ndarray:
    """(1 + len(HEIGHTS), ncell): first returns, then first returns with z >= h."""
    n = shape[0] * shape[1]
    acc = np.zeros((1 + len(HEIGHTS), n))
    n_read = n_first = n_out = 0
    t0 = time.time()
    with laspy.open(las_path) as r:
        total = r.header.point_count
        for pts in r.chunk_iterator(CHUNK):
            n_read += len(pts)
            keep = ((np.asarray(pts.return_number) == 1) & ~np.asarray(pts.withheld, bool)
                    & ~np.isin(np.asarray(pts.classification), NOISE_CLASSES))
            idx = grid_index(transform, np.asarray(pts.x)[keep], np.asarray(pts.y)[keep], shape)
            z = np.asarray(pts.z)[keep]
            n_first += int(keep.sum())
            n_out += int((idx < 0).sum())
            z, idx = z[idx >= 0], idx[idx >= 0]
            acc[0] += np.bincount(idx, minlength=n)
            for i, h in enumerate(HEIGHTS, 1):
                acc[i] += np.bincount(idx, weights=z >= h, minlength=n)
            el = time.time() - t0
            print(f'[gap]   {n_read / 1e6:,.0f} / {total / 1e6:,.0f} M points, {el / 60:.1f} min', flush=True)
    assert n_read == total, f'{las_path}: read {n_read} of {total} points'
    assert acc[0].sum() + n_out == n_first, 'first returns lost in the grid mapping'
    print(f'[gap]   {n_first:,} first returns kept ({100 * n_first / n_read:.1f} %), '
          f'{100 * n_out / max(n_first, 1):.2f} % outside the SAR grid')
    return acc


def gap(counts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with np.errstate(invalid='ignore', divide='ignore'):
        g = 1 - counts[1:] / counts[0]
    g[:, mask] = np.nan
    return g


def process_site(chm_path: Path, ref, out_dir: Path) -> Path:
    site = site_name(chm_path)
    las_path = chm_path.with_name(chm_path.name.replace('_canopy_p98.tif', '.las'))
    out = out_dir / f'gap_{site}.tif'
    if out.exists() and out.stat().st_mtime > max(chm_path.stat().st_mtime, las_path.stat().st_mtime):
        print(f'[gap] {site}: {out.name} is current, reused')
        return out
    chm = load_chm(chm_path)
    target = target_grid_for_chm(chm, ref)
    tr, shape = target.rio.transform(), (target.sizes['y'], target.sizes['x'])
    assert tr.b == 0 and tr.d == 0, 'rotated grid'
    print(f'[gap] {site}: grid {shape[1]} x {shape[0]} cells, lower-left '
          f'{tr.c:.4f} {tr.f + shape[0] * tr.e:.4f}, step {tr.a:.4f}')

    c_chm = chm_counts(chm, tr, shape)
    print(f'[gap] {site}: reading {las_path.name}')
    c_pts = point_counts(las_path, tr, shape)

    mask = (c_chm[0] < LOW_COUNT_THRESHOLD) | (c_pts[0] == 0)
    g_pts, g_chm = gap(c_pts, mask), gap(c_chm, mask)
    for name, g in (('pts', g_pts), ('chm', g_chm)):  # gap(h) cannot fall as h rises
        assert (np.diff(g, axis=0)[:, ~mask] >= -1e-9).all(), f'{site}: gap_{name} not monotonic in h'
    n_first = np.where(mask, np.nan, c_pts[0])
    n_chm = np.where(mask, np.nan, c_chm[0])
    data = np.concatenate([g_pts, g_chm, n_first[None], n_chm[None]]).reshape(len(BANDS), *shape)

    da = (xr.DataArray(np.where(np.isfinite(data), data, NODATA).astype('float32'),
                       dims=('band', 'y', 'x'),
                       coords={'band': np.arange(1, len(BANDS) + 1), 'y': target.y, 'x': target.x},
                       attrs={'long_name': tuple(BANDS)})
          .rio.write_crs(target.rio.crs).rio.write_transform(tr).rio.write_nodata(NODATA))
    partial = out.with_name(out.stem + '.partial.tif')
    da.rio.to_raster(partial)
    partial.replace(out)
    print(f'[gap] wrote {out.resolve()} ({int((~mask).sum()):,} valid cells)')
    return out


def load_cells(out: Path) -> pd.DataFrame:
    da = rxr.open_rasterio(out, masked=True)
    df = pd.DataFrame(da.values.reshape(len(BANDS), -1).T, columns=BANDS)
    return df.dropna()


def summarise(cells: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for site, d in list(cells.items()) + [('all_sites', pd.concat(cells.values()))]:
        for h in HEIGHTS:
            p, c = d[f'gap_pts_{h:g}'], d[f'gap_chm_{h:g}']
            diff = c - p
            rows.append({'site': site, 'h': h, 'n_cells': len(d),
                         'pts_pooled': (p * d.n_first).sum() / d.n_first.sum(),
                         'chm_pooled': (c * d.n_chm).sum() / d.n_chm.sum(),
                         'pts_cell_mean': p.mean(), 'chm_cell_mean': c.mean(),
                         'diff_mean': diff.mean(), 'diff_rmse': np.sqrt((diff ** 2).mean()),
                         'pearson_r': p.corr(c)})
    return pd.DataFrame(rows)


def plot(all_cells: pd.DataFrame, summ: pd.DataFrame, n_sites: int, out: Path) -> None:
    fig, axes = plt.subplots(1, len(HEIGHTS), figsize=(4.2 * len(HEIGHTS), 4.6),
                             constrained_layout=True, sharey=True)
    for ax, h in zip(axes, HEIGHTS):
        x, y = all_cells[f'gap_pts_{h:g}'], all_cells[f'gap_chm_{h:g}']
        hb = ax.hexbin(x, y, gridsize=40, extent=(0, 1, 0, 1), mincnt=1, bins='log',
                       cmap=viz_style.SEQUENTIAL_CMAP, linewidths=0)
        ax.plot([0, 1], [0, 1], color=viz_style.TEXT_MUTED, linewidth=1, linestyle='--')
        s = summ[(summ.site == 'all_sites') & (summ.h == h)].iloc[0]
        ax.text(0.03, 0.97, f'CHM − points: mean {s.diff_mean:+.3f}\nRMSE {s.diff_rmse:.3f}, r {s.pearson_r:.2f}',
                transform=ax.transAxes, va='top', fontsize=8.5, color=viz_style.TEXT_SECONDARY)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='Point gap fraction (first returns)')
        ax.set_title(f'h = {h:g} m', fontsize=11)
    axes[0].set_ylabel('CHM gap fraction (50 cm p98)')
    fig.colorbar(hb, ax=axes, shrink=0.8, label='SAR cells per hexagon')
    fig.suptitle(f'Gap fraction below h, CHM against point cloud: 2026 LiDAR, {n_sites} sites, '
                 f'{len(all_cells):,} SAR cells (24.2831 m)', color=viz_style.TEXT_PRIMARY, fontsize=12)
    fig.text(0.01, -0.02, f'Dashed: 1:1. Cells with fewer than {LOW_COUNT_THRESHOLD} CHM pixels masked. '
             'MVT sites excluded.', fontsize=8, color=viz_style.TEXT_MUTED, va='top')
    fig.savefig(out)
    plt.close(fig)
    print(f'[gap] wrote {out.resolve()}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=ROOT_WIN)
    p.add_argument('--sites', nargs='*', help='site folder names (default: all non-MVT sites)')
    p.add_argument('--out-dir', default=str(HERE / 'outputs' / 'gap_fraction'))
    args = p.parse_args()

    root, out_dir = Path(to_local(args.root)), Path(args.out_dir)
    chms = sorted(f for f in root.glob('*/*_denoise_canopy_p98.tif')
                  if site_name(f) not in EXCLUDE_SITES and (not args.sites or site_name(f) in args.sites))
    print(f'[gap] {len(chms)} sites: {", ".join(site_name(f) for f in chms)}')
    if not chms:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = rxr.open_rasterio(to_local(SAR_REF_WIN))
    for f in chms:
        process_site(f, ref, out_dir)

    # Summary and figure over every site raster present, not only those run now.
    present = sorted(out_dir.glob('gap_*.tif'))
    cells = {f.stem[4:]: load_cells(f) for f in present if not f.stem.endswith('.partial')}
    summ = summarise(cells)
    csv = out_dir / 'gap_summary.csv'
    summ.to_csv(csv, index=False, float_format='%.4f')
    print(f'[gap] wrote {csv.resolve()}')
    print(summ[summ.site == 'all_sites'].to_string(index=False, float_format='%.3f'))
    viz_style.apply_style()
    plot(pd.concat(cells.values()), summ, len(cells), out_dir / 'gap_chm_vs_pts.png')


if __name__ == '__main__':
    main()
