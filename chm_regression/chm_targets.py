"""Stage 1 -- CHM multi-statistic resampling.

For each of the 13 drone-LiDAR CHM rasters (50 cm, already p98-canopy-height
per pixel), compute a per-SAR-pixel 4-band target raster (mean, max, p90,
p98) plus a valid-pixel-count band, on the SAR reference geobox for a given
PALSAR-2 year.

This is a faithful, vectorised port of ``calculate_lidar_chm_metrics`` /
``get_CHM_pixel`` from
``raster/CHM_pixel_statistics_for_resampling/Pixel_CHM_resampling_statistics v04.ipynb``,
and of ``raster_align`` (cell 7) and the ``sar_masks`` construction (cell 51)
from ``Dampier_Peninsula_strat_4class_YEAR.ipynb``. Instead of the notebook's
double Python ``for x / for y`` loop (one ``.sel`` per SAR pixel), the 50 cm
pixel centres are mapped in one vectorised pass to target row/col via the
inverse of the reference affine, then aggregated with ``pandas.groupby``.
This is mathematically the same operation -- "which 50 cm centres fall
within this target pixel's footprint" -- just not evaluated one target
pixel at a time.

No notebooks are read at runtime; paths below are transcribed from cell 4 of
the classification notebook.

Usage
-----
conda run -n general --no-capture-output python chm_targets.py --year 2025
conda run -n general --no-capture-output python chm_targets.py --year 2025 --check-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from affine import Affine

# ---------------------------------------------------------------------------
# Paths -- transcribed verbatim (as G:\ strings) from cell 4 of
# Dampier_Peninsula_strat_4class_YEAR.ipynb. Do not mass-rewrite; translate
# with win_to_linux() at run time only.
# ---------------------------------------------------------------------------

LIDAR_PATHS_WIN = [
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Apr_2025\YA25_1\YF25_1_height_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Apr_2025\YA25_2\YA25_2_height_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Apr_2025\YA25_3\YF25_3_height_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Apr_2025\YA25_4\YF25_4_height_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Apr_2025\YA25_5\YA25_5_height_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\20250407_govcowwell\reprocess\20250407_govcowwell_height_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\YJ25_1\YJ25_1_height_clean_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\YJ25_2\YJ25_2_height_clean_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\YJ25_3\YJ25_3_height_clean_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\YJ25_4\YJ25_4_height_clean_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\YJ25_5\YJ25_5_height_clean_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\YJ25_6\YJ25_6_height_clean_canopy_p98.tif',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Oct_2025\YOct25_1\YOct25_1_height_clean_canopy_p98.tif',
]

LIDAR_POLYS_WIN = [
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Apr_2025\cut_areas\YA25_1_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Apr_2025\cut_areas\YA25_2_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Apr_2025\cut_areas\YA25_3_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Apr_2025\cut_areas\YA25_4_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Apr_2025\cut_areas\YA25_5_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\20250407_govcowwell\LASTools\20250407_govcowwell_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\cut_areas\YJ25_1_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\cut_areas\YJ25_2_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\cut_areas\YJ25_3_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\cut_areas\YJ25_4_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\cut_areas\YJ25_5_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_July_2025\cut_areas\YJ25_6_cut_area.gpkg',
    r'G:\Yawuru\Drone Data 2025\Yawuru_Lidar_Oct_2025\cut_areas\YOct25_1_cut_area.gpkg',
]

DAMPIER_CUTBOX_WIN = r'G:\Australia_shapefile\Dampier_peninsula_cutbox.gpkg'

# Area names, one per index in LIDAR_PATHS_WIN / LIDAR_POLYS_WIN.
AREA_NAMES = [
    'YA25_1', 'YA25_2', 'YA25_3', 'YA25_4', 'YA25_5',
    '20250407_govcowwell',
    'YJ25_1', 'YJ25_2', 'YJ25_3', 'YJ25_4', 'YJ25_5', 'YJ25_6',
    'YOct25_1',
]

STATS = ['mean', 'max', 'p90', 'p98']

# Below this many 50 cm CHM samples in a target pixel, flag the cell as
# low-confidence / maskable downstream. A full ~24 m pixel at 50 cm
# resolution nominally holds ~2300 samples (24/0.5)**2; anything under a
# few dozen means the pixel only clips a sliver of the drone survey edge.
LOW_COUNT_THRESHOLD = 30


def win_to_linux(path: str) -> str:
    """Translate a G:\\... workspace path to the Linux mount, per CLAUDE.md.
    No-op on Windows (path returned unchanged)."""
    if sys.platform == 'win32':
        return path
    p = path.replace('\\', '/')
    if p.startswith('G:/'):
        p = '/media/glen/CDU_Storage/' + p[len('G:/'):]
    return p


def find_palsar_hh_hv(year: str) -> tuple[Path, Path]:
    """Return (HH, HV) gamma-0 refined-Lee-filtered PALSAR paths for a year.

    Prefers the notebook's expected ``HH_g_rflee.tif`` / ``HV_g_rflee.tif``
    naming (cell 14). 2025's reprocessing landed as ``HH_g_rflee_w5.tif`` /
    ``HV_g_rflee_w5.tif`` instead (window-5 Refined Lee, same processing
    step, new suffix) -- this is a naming drift versus the notebook that
    predates this script; fall back to it and warn.
    """
    base = Path(win_to_linux(rf'G:\Yawuru\WCM\SAR\PALSAR {year} DP'))
    candidates = [
        (base / 'HH_g_rflee.tif', base / 'HV_g_rflee.tif'),
        (base / 'HH_g_rflee_w5.tif', base / 'HV_g_rflee_w5.tif'),
    ]
    for hh, hv in candidates:
        if hh.exists() and hv.exists():
            if 'w5' in hh.name:
                print(f'[chm_targets] NOTE: {base} has no HH_g_rflee.tif (the name the '
                      f'classification notebook cell 14 expects); using {hh.name} / '
                      f'{hv.name} instead. Same processing step (Refined Lee, window 5), '
                      f'different filename -- naming drift versus the notebook, not a '
                      f'data problem.')
            return hh, hv
    raise FileNotFoundError(f'No HH/HV gamma-0 refined-Lee pair found under {base}')


def build_sar_reference(year: str) -> xr.DataArray:
    """Reproduce cells 14-17: PALSAR HH/HV gamma-0, clipped to the Dampier
    cutbox in EPSG:32751. Only the grid (CRS / transform / shape) matters
    here -- band values are never used by Stage 1.
    """
    hh_path, hv_path = find_palsar_hh_hv(year)
    hh_g = rxr.open_rasterio(hh_path)
    hv_g = rxr.open_rasterio(hv_path)
    sar_g0 = xr.concat([hh_g, hv_g], dim='band').assign_coords(
        band=['gamma0_HH', 'gamma0_HV']
    )

    dampier_path = win_to_linux(DAMPIER_CUTBOX_WIN)
    dampier_geom = gpd.read_file(dampier_path).to_crs('epsg:32751')
    sar_g0 = sar_g0.rio.clip([dampier_geom.iloc[0].geometry])
    return sar_g0


def target_grid_for_chm(chm: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Reproduce raster_align's bounds intersection (cell 7) to get the
    target-grid subset (x/y coords, transform, shape) that a given CHM
    raster maps onto. Returns an empty (all-NaN) single-band DataArray on
    that grid, matching ``lidar_metrics_ds['empty']`` in the source
    notebook.
    """
    in_bounds = chm.rio.bounds()
    ref_bounds = ref.rio.bounds()
    minx = max(in_bounds[0], ref_bounds[0])
    miny = max(in_bounds[1], ref_bounds[1])
    maxx = min(in_bounds[2], ref_bounds[2])
    maxy = min(in_bounds[3], ref_bounds[3])
    ref_clipped = ref.isel(band=0).drop_vars('band').rio.clip_box(
        minx=minx, miny=miny, maxx=maxx, maxy=maxy
    )
    empty = xr.full_like(ref_clipped, np.nan, dtype='float64')
    return empty


def aggregate_chm_to_grid(chm: xr.DataArray, target: xr.DataArray) -> xr.Dataset:
    """Core of Stage 1: map every 50 cm CHM pixel centre to its target
    row/col (vectorised inverse-affine, replacing the notebook's per-target-
    pixel ``get_CHM_pixel`` loop) and aggregate with pandas.groupby into
    mean / max / p90 / p98 / count bands on ``target``'s grid.
    """
    transform: Affine = target.rio.transform()
    height, width = target.sizes['y'], target.sizes['x']
    inv = ~transform

    values = chm.values.astype('float64').ravel()
    valid = np.isfinite(values)
    if chm.rio.nodata is not None and not np.isnan(chm.rio.nodata):
        valid &= values != chm.rio.nodata
    if not valid.any():
        empty_stats = {s: xr.full_like(target, np.nan) for s in STATS}
        empty_stats['count'] = xr.full_like(target, 0, dtype='int32')
        return xr.Dataset(empty_stats)

    xs, ys = np.meshgrid(chm.x.values, chm.y.values)
    xs = xs.ravel()[valid]
    ys = ys.ravel()[valid]
    values = values[valid]

    cols_f, rows_f = inv * (xs, ys)
    cols = np.floor(cols_f).astype('int64')
    rows = np.floor(rows_f).astype('int64')

    in_grid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    if not in_grid.all():
        cols, rows, values = cols[in_grid], rows[in_grid], values[in_grid]

    df = pd.DataFrame({'row': rows, 'col': cols, 'value': values})
    agg = df.groupby(['row', 'col'])['value'].agg(
        mean='mean', max='max',
        p90=lambda s: np.percentile(s, 90),
        p98=lambda s: np.percentile(s, 98),
        count='count',
    )

    out = {}
    for stat in STATS:
        arr = np.full((height, width), np.nan, dtype='float32')
        arr[agg.index.get_level_values('row'), agg.index.get_level_values('col')] = agg[stat].values
        out[stat] = xr.DataArray(arr, dims=('y', 'x'), coords={'y': target.y, 'x': target.x})

    count_arr = np.zeros((height, width), dtype='int32')
    count_arr[agg.index.get_level_values('row'), agg.index.get_level_values('col')] = agg['count'].values
    out['count'] = xr.DataArray(count_arr, dims=('y', 'x'), coords={'y': target.y, 'x': target.x})

    ds = xr.Dataset(out)
    ds = ds.rio.write_crs(target.rio.crs)
    ds = ds.rio.write_transform(transform)
    return ds


@dataclass
class AreaResult:
    name: str
    n_cells: int
    n_low_count: int
    low_count_frac: float
    out_path: Path


def process_area(idx: int, year: str, ref: xr.DataArray, out_dir: Path) -> AreaResult:
    name = AREA_NAMES[idx]
    chm_path = win_to_linux(LIDAR_PATHS_WIN[idx])
    poly_path = win_to_linux(LIDAR_POLYS_WIN[idx])

    chm = rxr.open_rasterio(chm_path).squeeze(drop=True)
    nodata = chm.rio.nodata
    if nodata is not None:
        chm = chm.where(chm != nodata, np.nan)
    chm = chm.rio.write_nodata(np.nan)

    target = target_grid_for_chm(chm, ref)
    ds = aggregate_chm_to_grid(chm, target)

    ds.attrs['area_name'] = name
    ds.attrs['year'] = year
    ds.attrs['low_count_threshold'] = LOW_COUNT_THRESHOLD
    for v in ds.data_vars:
        ds[v].attrs['low_count_threshold'] = LOW_COUNT_THRESHOLD

    out_path = out_dir / f'chm_targets_{name}.tif'
    band_order = STATS + ['count']
    stacked = xr.concat(
        [ds[v].astype('float32') for v in band_order], dim='band'
    ).assign_coords(band=band_order)
    stacked = stacked.rio.write_crs(ds.rio.crs)
    stacked = stacked.rio.write_nodata(np.nan)

    # Clip to the survey's cut-area polygon. This reproduces the final step of
    # notebook cell 51 --
    #     lidar_classified_clipped = [r.rio.clip(c, drop=True) for r, c in ...]
    # -- and it is not optional: without it the targets cover each CHM's full
    # rectangular extent, including margins outside the area the established
    # classification was ever trained on (8,709 vs 5,791 cells across the 13
    # surveys, i.e. a third of the pixels are outside the cut areas).
    poly = gpd.read_file(poly_path).to_crs(stacked.rio.crs)
    stacked = stacked.rio.clip(poly.geometry, drop=True)

    stacked.rio.to_raster(out_path)

    # Counted after the clip so the report matches what was written.
    count_band = stacked.sel(band='count')
    n_cells = int((count_band > 0).sum())
    n_low = int(((count_band > 0) & (count_band < LOW_COUNT_THRESHOLD)).sum())
    low_frac = n_low / n_cells if n_cells else float('nan')

    print(f'[chm_targets] {name}: {n_cells} target cells, '
          f'{n_low} ({low_frac:.1%}) below count threshold {LOW_COUNT_THRESHOLD} '
          f'-> {out_path}')

    return AreaResult(name, n_cells, n_low, low_frac, out_path)


def run(year: str, out_dir: Path) -> list[AreaResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = build_sar_reference(year)
    results = [process_area(i, year, ref, out_dir) for i in range(len(LIDAR_PATHS_WIN))]

    summary = pd.DataFrame(
        [{'area': r.name, 'n_cells': r.n_cells, 'n_low_count': r.n_low_count,
          'low_count_frac': r.low_count_frac, 'raster': str(r.out_path)} for r in results]
    )
    summary_path = out_dir / 'chm_targets_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f'[chm_targets] wrote summary table -> {summary_path}')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', default='2025', help='PALSAR year for the SAR reference geobox (default 2025)')
    parser.add_argument('--out-dir', default=None,
                         help='Output directory (default: chm_regression/outputs/<year>)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / 'outputs' / args.year
    )
    run(args.year, out_dir)


if __name__ == '__main__':
    main()
