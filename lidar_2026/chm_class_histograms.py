"""Canopy height histograms for the 2026 Dampier drone LiDAR.

For every canopy height raster (CHM, 50 cm) found under the 2026 survey root,
count pixels at two resolutions (50 cm CHM, and CHM mean on the SAR grid) in
two binnings, which gives four histogram sets. Each set is one file with a
panel per site; hist_all_sites.png holds the four all-sites histograms:

    hist_class_50cm.png        five height classes on the 50 cm CHM
    hist_class_sar_mean.png    five height classes on the SAR-resolution mean
    hist_bin05_50cm.png        0.5 m height bins on the 50 cm CHM
    hist_bin05_sar_mean.png    0.5 m height bins on the SAR-resolution mean

The five classes:

    class         50 cm CHM     SAR-resolution mean (PALSAR 2025 grid, 24.2831 m)
    Bare          < 0.2         < 0.2
    Low veg       0.2 - 2       0.2 - 1.22
    Low forest    2 - 5         1.22 - 2.35
    Forest        5 - 10        2.35 - 5.381
    Tall forest   >= 10         >= 5.381

The SAR-resolution bounds are the 20 % cover fit from
raster/CHM_pixel_statistics_for_resampling (v03/v04). 1.22 / 2.35 are the
operative published cuts (the fit gives 1.146 / 2.338 -- see
class_vs_regression/REPORT_threshold_translation.md, kept deliberately).
5.381 m is the fit at nominal 10 m, derived on the 2025 Yawuru sites.

The 0.5 m bins run from 0 to 25 m with an open top bin (>= 25 m); each figure
shows bins up to the highest bin that is occupied in any site, so all sites
share one x-axis per resolution.

The mean resampling imports target_grid_for_chm / aggregate_chm_to_grid from
chm_regression/chm_targets.py (no fork) and writes the same 5-band
chm_targets_<site>.tif (mean, max, p90, p98, count).

Divergences from the 2025 chm_targets procedure:
- No cut-area polygon exists for the 2026 sites, so no polygon clip; the
  valid footprint is the CHM's own non-nodata area.
- A CHM in EPSG:7851 (GDA2020 / MGA 51) is accepted against the EPSG:32751
  grid without reprojection: same projection, datum offset < 2 m (< 0.1 of a
  SAR pixel). Any other CRS aborts.
  ⚠ CRS 2026-11-03: this relabel is withdrawn (user). Reprojection is required to
  match grids. See G:/python/crs_audit/CHECKPOINT.md. Code not yet changed.
- The two MVT sites (JPP MVT, Quandong MVT) are excluded: they are coastal and
  need different processing.

Safe to run again as each CHM arrives: existing chm_targets rasters newer than
their CHM are reused, and the all-sites figures are rebuilt from what is present.

Usage
-----
conda run -n general --no-capture-output python chm_class_histograms.py
conda run -n general --no-capture-output python chm_class_histograms.py --pattern "*_chm*.tif"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'chm_regression'))
from chm_targets import (LOW_COUNT_THRESHOLD, STATS, aggregate_chm_to_grid,  # noqa: E402
                         target_grid_for_chm, win_to_linux)
import viz_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (after viz_style sets Agg)

ROOT_WIN = r'G:\Dampier\Dampier Lidar 2026'
SAR_REF_WIN = r'G:\Yawuru\WCM\SAR\PALSAR 2025 DP\HH_g_rflee.tif'
EXCLUDE_SITES = {'JPP MVT', 'Quandong MVT'}  # coastal; different processing, not in this EDA
N_SITES_EXPECTED = 6

CLASSES = ['Bare', 'Low veg', 'Low forest', 'Forest', 'Tall forest']
BOUNDS = {
    '50cm': [0.2, 2.0, 5.0, 10.0],
    'sar_mean': [0.2, 1.22, 2.35, 5.381],
}
BIN_W, BIN_TOP = 0.5, 25.0
BIN_EDGES = [round(float(e), 1) for e in np.arange(BIN_W, BIN_TOP + BIN_W / 2, BIN_W)]
BIN_LABELS = ([f'{lo:g}-{hi:g}' for lo, hi in zip([0.0] + BIN_EDGES[:-1], BIN_EDGES)]
              + [f'>={BIN_TOP:g}'])
SCHEMES = {  # scheme -> (labels, {resolution: inner bounds})
    'class': (CLASSES, BOUNDS),
    'bin05': (BIN_LABELS, {'50cm': BIN_EDGES, 'sar_mean': BIN_EDGES}),
}
RES_TITLES = {
    ('class', '50cm'): '50 cm LiDAR CHM, height classes (0.2 / 2 / 5 / 10 m)',
    ('class', 'sar_mean'): 'CHM mean at SAR resolution, height classes (0.2 / 1.22 / 2.35 / 5.381 m)',
    ('bin05', '50cm'): '50 cm LiDAR CHM, 0.5 m height bins',
    ('bin05', 'sar_mean'): 'CHM mean at SAR resolution, 0.5 m height bins',
}
OK_CRS = {32751, 7851}
SETTLE_S = 120  # skip a CHM modified more recently than this (may still be writing)


def to_local(path: str) -> str:
    """No-op on Windows; win_to_linux elsewhere (CLAUDE.md path convention)."""
    return path if sys.platform == 'win32' else win_to_linux(path)


def class_counts(values: np.ndarray, bounds: list[float]) -> np.ndarray:
    v = values[np.isfinite(values)]
    return np.histogram(v, bins=[-np.inf] + bounds + [np.inf])[0]


def site_name(chm_path: Path) -> str:
    return chm_path.parent.name


def load_chm(chm_path: Path):
    chm = rxr.open_rasterio(chm_path, masked=True).squeeze(drop=True)
    epsg = chm.rio.crs.to_epsg() if chm.rio.crs else None
    if epsg not in OK_CRS:
        raise ValueError(f'{chm_path}: CRS {chm.rio.crs} not in EPSG {sorted(OK_CRS)}')
    # ⚠ CRS 2026-11-03: relabel withdrawn (user); reprojection is required (G:\python\crs_audit\CHECKPOINT.md)
    if epsg != 32751:
        print(f'[hist] NOTE {chm_path.name}: EPSG:{epsg} used on the EPSG:32751 grid '
              f'without reprojection (datum offset < 2 m)')
        chm = chm.rio.write_crs('EPSG:32751')
    return chm.rio.write_nodata(np.nan)


def mean_raster(chm, chm_path: Path, ref, out_dir: Path) -> Path:
    out = out_dir / f'chm_targets_{site_name(chm_path)}.tif'
    if out.exists() and out.stat().st_mtime > chm_path.stat().st_mtime:
        return out
    target = target_grid_for_chm(chm, ref)
    assert target.size > 0, f'{chm_path}: empty target grid (CHM outside SAR reference?)'
    ds = aggregate_chm_to_grid(chm, target)
    n_valid = int(np.isfinite(chm.values).sum())
    n_counted = int(ds['count'].sum())
    assert n_counted == n_valid, (f'{chm_path}: {n_counted} of {n_valid} valid 50 cm pixels '
                                  f'mapped onto the SAR grid')
    bands = STATS + ['count']
    stacked = ds.to_array('band').sel(band=bands).astype('float32')
    stacked = (stacked.where(np.isfinite(stacked), -9999.0)
               .rio.write_crs(ds.rio.crs).rio.write_transform(ds.rio.transform())
               .rio.write_nodata(-9999.0))
    partial = out.with_name(out.stem + '.partial.tif')
    stacked.rio.to_raster(partial)
    partial.replace(out)
    return out


def counts_for_site(chm_path: Path, ref, out_dir: Path) -> pd.DataFrame:
    chm = load_chm(chm_path)
    tgt = rxr.open_rasterio(mean_raster(chm, chm_path, ref, out_dir), masked=True)
    mean = tgt.sel(band=1).values  # band 1 = mean
    count = np.nan_to_num(tgt.sel(band=5).values)  # band 5 = count
    values = {'50cm': chm.values.ravel(),
              'sar_mean': np.where(count >= LOW_COUNT_THRESHOLD, mean, np.nan).ravel()}
    rows = [{'site': site_name(chm_path), 'scheme': s, 'resolution': res, 'label': k, 'count': int(n)}
            for s, (labels, bounds) in SCHEMES.items()
            for res, v in values.items()
            for k, n in zip(labels, class_counts(v, bounds[res]))]
    return pd.DataFrame(rows)


def add_percent(df: pd.DataFrame) -> pd.DataFrame:
    tot = df.groupby(['site', 'scheme', 'resolution'])['count'].transform('sum')
    df['percent'] = 100 * df['count'] / tot
    return df


def draw(ax, d: pd.DataFrame, scheme: str, res: str, labels: list[str], title: str) -> None:
    d = d.set_index('label').loc[labels]
    n = int(d['count'].sum())
    pct, cnt = d['percent'].to_numpy(), d['count'].to_numpy()
    color = viz_style.TARGET_COLORS['mean']
    if scheme == 'class':
        x = np.arange(len(labels))
        ax.bar(x, pct, width=0.7, color=color, edgecolor=viz_style.SURFACE, linewidth=2)
        for xi, p, c in zip(x, pct, cnt):
            ax.annotate(f'{p:.1f}%\n{c:,}', (xi, p), xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color=viz_style.TEXT_SECONDARY)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, pct.max() * 1.22)  # headroom for the 2-line labels
        ax.grid(axis='x', visible=False)
    else:
        x = BIN_W * np.arange(len(labels))
        ax.bar(x, pct, width=BIN_W, align='edge', color=color, edgecolor=viz_style.SURFACE, linewidth=1)
        for xi, p, c in zip(x, pct, cnt):
            ax.annotate(f'{p:.1f}% ({c:,})', (xi + BIN_W / 2, p), xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', rotation=90, fontsize=6.5, color=viz_style.TEXT_SECONDARY,
                        bbox=dict(fc=viz_style.SURFACE, ec='none', pad=0.5))  # mask the bound lines
        for b in BOUNDS[res]:
            if b < x[-1] + BIN_W:
                ax.axvline(b, color=viz_style.TEXT_MUTED, linestyle=':', linewidth=1, zorder=0.5)
        ax.set_xlim(0, x[-1] + BIN_W)
        ax.set_xticks(np.arange(0, x[-1] + BIN_W + 1e-9, 1.0 if x[-1] < 8 else 2.0))
        ax.set_xlabel('Canopy height (m)')
        ax.set_ylim(0, pct.max() * 1.55)  # headroom for the rotated labels
        ax.grid(axis='x', visible=False)
    ax.set_ylabel('Percent of valid pixels')
    ax.set_title(f'{title}\nn = {n:,} pixels', fontsize=10)


def save(fig, suptitle: str, sar: bool, out: Path) -> None:
    fig.suptitle(suptitle, color=viz_style.TEXT_PRIMARY, fontsize=13)
    note = 'Labels: percent and pixel count. Dotted lines on the 0.5 m bin histograms: class bounds.'
    if sar:
        note += (f'\nSAR resolution: PALSAR 2025 grid, 24.2831 m; cells with fewer than '
                 f'{LOW_COUNT_THRESHOLD} 50 cm samples masked. 5.381 m bound from the 2025 Yawuru '
                 f'cover fit, not refitted on 2026 data.')
    fig.text(0.01, -0.012, note, fontsize=8, color=viz_style.TEXT_MUTED, va='top')
    fig.savefig(out)
    plt.close(fig)
    print(f'[hist] wrote {out.resolve()}')


def plot_sets(df: pd.DataFrame, sites: list[str], n_chms: int, out_dir: Path) -> None:
    """One file per (scheme, resolution) with a panel per site, plus one all-sites 2x2 file."""
    labels = {}
    for (scheme, res), dsr in df.groupby(['scheme', 'resolution']):
        lab = SCHEMES[scheme][0]
        if scheme == 'bin05':  # trim to the highest bin occupied in any site
            occupied = dsr.loc[(dsr['site'] == 'all_sites') & (dsr['count'] > 0), 'label']
            lab = lab[:max(lab.index(k) for k in occupied) + 1]
        labels[(scheme, res)] = lab
        ncols = 3 if scheme == 'class' else 2
        nrows = -(-len(sites) // ncols)
        size = (6, 4.2) if scheme == 'class' else (0.24 * len(lab) + 1.5, 4.6)
        fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows),
                                 squeeze=False, constrained_layout=True)
        for ax, site in zip(axes.flat, sites):
            draw(ax, dsr[dsr['site'] == site], scheme, res, lab, site)
        for ax in axes.flat[len(sites):]:
            ax.set_visible(False)
        save(fig, f'2026 Dampier LiDAR by site: {RES_TITLES[(scheme, res)]}', res == 'sar_mean',
             out_dir / f'hist_{scheme}_{res}.png')

    keys = [('class', '50cm'), ('class', 'sar_mean'), ('bin05', '50cm'), ('bin05', 'sar_mean')]
    w = max(0.24 * len(labels[k]) + 1.5 for k in keys[2:])
    fig, axes = plt.subplots(2, 2, figsize=(2 * w, 10), constrained_layout=True,
                             gridspec_kw={'height_ratios': [1, 1.15]})
    for ax, (scheme, res) in zip(axes.flat, keys):
        d = df[(df['site'] == 'all_sites') & (df['scheme'] == scheme) & (df['resolution'] == res)]
        draw(ax, d, scheme, res, labels[(scheme, res)], RES_TITLES[(scheme, res)])
    save(fig, f'All 2026 Dampier LiDAR sites ({n_chms} of {N_SITES_EXPECTED})', True,
         out_dir / 'hist_all_sites.png')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=ROOT_WIN)
    p.add_argument('--pattern', default='*canopy_p98*.tif', help='CHM file name glob in each site folder')
    p.add_argument('--out-dir', default=str(HERE / 'outputs'))
    args = p.parse_args()

    root, out_dir = Path(to_local(args.root)), Path(args.out_dir)
    now = time.time()
    chms = sorted(f for f in root.glob(f'*/{args.pattern}')
                  if not f.name.endswith('.partial.tif') and now - f.stat().st_mtime > SETTLE_S
                  and site_name(f) not in EXCLUDE_SITES)
    print(f'[hist] {len(chms)} CHMs found under {root} matching {args.pattern} '
          f'(excluded: {", ".join(sorted(EXCLUDE_SITES))})')
    if not chms:
        return
    dup = pd.Series([site_name(f) for f in chms]).duplicated(keep=False)
    if dup.any():
        sys.exit('[hist] more than one CHM per site folder, use a narrower --pattern:\n  '
                 + '\n  '.join(str(f) for f, d in zip(chms, dup) if d))
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_style.apply_style()
    ref = rxr.open_rasterio(to_local(SAR_REF_WIN))

    per_site = []
    for f in chms:
        print(f'[hist] {site_name(f)}: {f.name}')
        per_site.append(counts_for_site(f, ref, out_dir))
    df = pd.concat(per_site, ignore_index=True)
    allsites = (df.groupby(['scheme', 'resolution', 'label'], as_index=False, sort=False)['count'].sum()
                .assign(site='all_sites'))
    df = add_percent(pd.concat([df, allsites], ignore_index=True))
    csv = out_dir / 'height_counts.csv'
    df.to_csv(csv, index=False)
    print(f'[hist] wrote {csv.resolve()}')

    plot_sets(df, [site_name(f) for f in chms], len(chms), out_dir)


if __name__ == '__main__':
    main()
