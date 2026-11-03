"""Stage 1 -- shared NAFI fire data layer for the fire-history study.

Everything the later stages need to know about *fire* lives here: the
month-resolved scar polygons, the 2016-2025 fire-frequency raster, and the
timing convention that ties a fire to the composite it can affect.

Timing convention (PLAN_woody_thickening_splice_1.md section 2)
--------------------------------------------------
The canopy-height composites are **July**. So:

* a fire in **Jan-Jul of year Y** suppresses the **year-Y** height;
* a fire in **Aug-Dec of year Y** postdates the year-Y composite, and the first
  height it can affect is **Y+1**.

Every derived variable here carries that convention, and the figures state it.

Data notes
----------
The NAFI scar shapefiles are national, ~34 MB .shp / ~44 MB .dbf and ~270 k
features per year, on a slow NTFS-over-USB volume. They are therefore always
read with a ``bbox=`` filter and **once** per year -- never inside a per-point
loop. Column naming drifts across years (``Month``/``month``, plus
``Update_no``/``Update_No``/``Id``/``ID``/``FSID``), so the month column is
matched case-insensitively and nothing else but the geometry is relied on.
This handling is inherited from ``../chm_regression/thickening_fire.py``.

Self-test::

    conda run -n general --no-capture-output python fire_data.py
"""

from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'outputs'
FIG_ROOT = Path('/media/glen/CDU_Storage/python/dampier_veg_timeseries')

NAFI_DIR_WIN = r'G:\python\fire_class_forest_stats\NAFI'
NAFI_DIR = Path(NAFI_DIR_WIN if sys.platform == 'win32'
                else '/media/glen/CDU_Storage/' + NAFI_DIR_WIN[3:].replace('\\', '/'))
NAFI_SCARS = NAFI_DIR / 'fire scars'
FF_RASTER = NAFI_DIR / 'FF_2016_25_gda94.tif'
FF_NODATA = 255

PINDAN_GPKG = '/media/glen/CDU_Storage/Veg and Soil Mapping/Dampier_pindan_main.gpkg'
POINTS_GPKG = FIG_ROOT / 'chm_gedi_fusion' / 'gedi_2025_points_for_fire_histories.gpkg'

# The complete canopy-height series. 2020 and 2022 are absent because the input
# data do not exist -- they are not a gap to be filled. See PLAN section 1.1.
YEARS = [2019, 2021, 2023, 2024, 2025]
CHM_THRESHOLD = 'T4'          # the only threshold produced for every year
REF_YEAR = 2025               # grid the FF raster is warped onto
CHM_NODATA = -9999.0

HISTORY_START = 2015          # earliest NAFI year on the drive
FIRE_YEARS = list(range(2019, 2026))
EARLY_MONTHS = list(range(1, 8))    # Jan-Jul: before the July composite
LATE_MONTHS = list(range(8, 13))    # Aug-Dec: after it
TSB_CAP = 10


def log(msg: str) -> None:
    print(f'[1] {msg}', flush=True)


def chm_path(year: int) -> Path:
    return FIG_ROOT / f'figsDP{year}' / f'RF_HYBRID_{CHM_THRESHOLD}_{year}.tif'


# ---------------------------------------------------------------------------
# Fire scar polygons
# ---------------------------------------------------------------------------

def scar_path(year: int) -> Path:
    # '{year}*firescar*' also matches the part-year folder, e.g. '2026 to Oct18 firescar shape files'
    hits = sorted(glob.glob(str(NAFI_SCARS / f'{year}*firescar*' / '*.shp')))
    if not hits:
        raise FileNotFoundError(f'no NAFI shapefile for {year} under {NAFI_SCARS}')
    return Path(hits[0])


def load_scars(year: int, bounds, crs) -> gpd.GeoDataFrame:
    """One year's scars, clipped to ``bounds`` (given in ``crs``) and reprojected.

    The bbox is converted to the shapefile's native EPSG:4283 and passed to
    ``read_file`` so only the Dampier subset is ever parsed or reprojected.
    """
    box = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt([
            f'POLYGON(({bounds[0]} {bounds[1]},{bounds[2]} {bounds[1]},'
            f'{bounds[2]} {bounds[3]},{bounds[0]} {bounds[3]},'
            f'{bounds[0]} {bounds[1]}))']),
        crs=crs)
    bbox_native = box.to_crs('EPSG:4283').total_bounds
    g = gpd.read_file(scar_path(year), bbox=tuple(bbox_native))
    if len(g) == 0:
        return g.to_crs(crs) if g.crs is not None else g
    month_col = next((c for c in g.columns if c.lower() == 'month'), None)
    if month_col is None:
        raise ValueError(f'{year}: no month column, got {g.columns.tolist()}')
    g['Month'] = pd.to_numeric(g[month_col], errors='coerce')
    return g.to_crs(crs)


def fire_events_at_points(points: gpd.GeoDataFrame,
                          years=None,
                          buffer_m: float = 2000.0) -> pd.DataFrame:
    """One row per (point, fire) for every scar intersecting a point.

    ``points`` must carry a ``point_id`` column. Returns columns
    ``point_id, year, month, frac_year, season, area_ha``. ``frac_year`` is
    ``year + (month - 0.5) / 12`` -- the mid-month position used to place the
    vertical bands in F1.

    A point may be hit more than once in a year; every hit is kept, with only
    exact (point_id, year, month) duplicates collapsed (the same burn split
    across several scar polygons).
    """
    years = list(FIRE_YEARS if years is None else years)
    crs = points.crs
    minx, miny, maxx, maxy = points.total_bounds
    bounds = (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)

    rows = []
    for year in years:
        t = time.time()
        g = load_scars(year, bounds, crs)
        if len(g) == 0:
            log(f'{year}: no scars in bbox')
            continue
        g = g[g['Month'].notna()].copy()
        g['area_ha'] = g.geometry.area / 1e4
        hit = gpd.sjoin(points[['point_id', 'geometry']], g[['Month', 'area_ha', 'geometry']],
                        how='inner', predicate='intersects')
        for _, r in hit.iterrows():
            month = int(r['Month'])
            rows.append({
                'point_id': int(r['point_id']),
                'year': year,
                'month': month,
                'frac_year': year + (month - 0.5) / 12.0,
                'season': 'early' if month in EARLY_MONTHS else 'late',
                'area_ha': float(r['area_ha']),
            })
        log(f'{year}: {len(g):>6,} scars in bbox, {len(hit):>3} point hits '
            f'({time.time() - t:.0f} s)')

    if not rows:
        return pd.DataFrame(columns=['point_id', 'year', 'month', 'frac_year',
                                     'season', 'area_ha'])
    df = pd.DataFrame(rows)
    df = (df.sort_values(['point_id', 'year', 'month', 'area_ha'])
            .drop_duplicates(subset=['point_id', 'year', 'month'], keep='last')
            .reset_index(drop=True))
    return df


def prefire_context(points: gpd.GeoDataFrame,
                    start: int = HISTORY_START,
                    end: int = 2018) -> pd.DataFrame:
    """Last burn year in ``start``-``end``, per point.

    Without this a point that never burnt inside the plotted 2019-2025 window
    reads as long-unburnt when it may have burnt in 2018. Returns columns
    ``point_id, last_burn_before_2019`` (NaN when unburnt across the window).
    """
    ev = fire_events_at_points(points, years=list(range(start, end + 1)))
    out = pd.DataFrame({'point_id': points['point_id'].astype(int).to_numpy()})
    if len(ev) == 0:
        out['last_burn_before_2019'] = np.nan
        return out
    last = ev.groupby('point_id')['year'].max().rename('last_burn_before_2019')
    return out.merge(last, on='point_id', how='left')


def tsb_at(events: pd.DataFrame, point_ids, year: int,
           pre_2019: pd.DataFrame | None = None) -> pd.Series:
    """Whole years since the last burn as at **July of ``year``**, capped at 10.

    Only fires that predate the July composite count: Jan-Jul of ``year``
    itself gives 0, Aug-Dec of ``year`` does not count for ``year`` at all.
    """
    tsb = pd.Series(float(TSB_CAP), index=pd.Index(point_ids, name='point_id'))
    if len(events):
        eff = events[(events['year'] < year)
                     | ((events['year'] == year) & (events['month'] <= 7))]
        if len(eff):
            # A fire's "effective year" is the composite year it precedes.
            last = eff.groupby('point_id')['year'].max()
            age = (year - last).astype(float)
            tsb.loc[age.index] = np.minimum(age, TSB_CAP)
    if pre_2019 is not None:
        pre = pre_2019.dropna(subset=['last_burn_before_2019'])
        for pid, by in zip(pre['point_id'], pre['last_burn_before_2019']):
            if pid in tsb.index:
                tsb.loc[pid] = min(tsb.loc[pid], min(year - by, TSB_CAP))
    return tsb


# ---------------------------------------------------------------------------
# Fire frequency raster
# ---------------------------------------------------------------------------

def ff_at_points(points: gpd.GeoDataFrame) -> pd.Series:
    """Sample FF_2016_25 at each point. Nodata (255) becomes NaN."""
    with rasterio.open(FF_RASTER) as src:
        pts = points.to_crs(src.crs)
        coords = [(g.x, g.y) for g in pts.geometry]
        vals = np.array([v[0] for v in src.sample(coords)], dtype='float32')
    vals[vals == FF_NODATA] = np.nan
    return pd.Series(vals, index=points.index, name='ff_2016_25')


def ff_on_grid(ref_raster: Path | None = None,
               out_path: Path | None = None,
               overwrite: bool = False) -> Path:
    """Warp the 250 m FF raster onto the reference CHM grid.

    ``Resampling.nearest`` is mandatory: FF is a categorical count of years
    burnt, and any averaging kernel would invent fractional fire frequencies.
    """
    ref_raster = Path(ref_raster or chm_path(REF_YEAR))
    out_path = Path(out_path or (OUT_DIR / 'ff_strata_pindan.tif'))
    if out_path.exists() and not overwrite:
        log(f'{out_path.name} already written (pass overwrite=True to rebuild)')
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ref_raster) as ref:
        profile = ref.profile.copy()
        profile.update(dtype='uint8', count=1, nodata=FF_NODATA,
                       compress='deflate', tiled=True,
                       blockxsize=512, blockysize=512, predictor=1)
        dst = np.full((ref.height, ref.width), FF_NODATA, dtype='uint8')
        with rasterio.open(FF_RASTER) as src:
            reproject(source=rasterio.band(src, 1),
                      destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref.transform, dst_crs=ref.crs,
                      src_nodata=FF_NODATA, dst_nodata=FF_NODATA,
                      resampling=Resampling.nearest)
        ref_transform, ref_crs = ref.transform, ref.crs

    # Mask to pindan: the strata only ever apply inside the land system.
    from rasterio.features import geometry_mask
    pindan = gpd.read_file(PINDAN_GPKG).to_crs(ref_crs)
    outside = geometry_mask([g for g in pindan.geometry],
                            out_shape=dst.shape, transform=ref_transform,
                            invert=False)
    n_before = int((dst != FF_NODATA).sum())
    dst[outside] = FF_NODATA
    n_after = int((dst != FF_NODATA).sum())
    if n_after == 0 or n_after >= n_before:
        raise RuntimeError(f'pindan clip did nothing or removed everything '
                           f'({n_before:,} -> {n_after:,}) -- check the CRS')

    tmp = out_path.with_name(f'{out_path.stem}.partial.tif')
    with rasterio.open(tmp, 'w', **profile) as dstf:
        dstf.write(dst, 1)
        dstf.set_band_description(1, 'NAFI fire frequency 2016-2025 (years burnt)')
    tmp.replace(out_path)
    out_path.with_suffix('.tif.aux.xml').unlink(missing_ok=True)
    log(f'wrote {out_path}  ({n_after:,} pindan pixels of {n_before:,})')
    return out_path


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------

def load_points(path: Path | None = None) -> gpd.GeoDataFrame:
    """Sample points with a stable ``point_id`` and a human ``label``.

    ``point_id`` is file order. The file has no name column, so the label is
    built from the coordinates.
    """
    g = gpd.read_file(path or POINTS_GPKG).reset_index(drop=True)
    g['point_id'] = np.arange(len(g), dtype=int)
    if 'lat' in g.columns and 'lon' in g.columns:
        g['label'] = [f'P{i} ({la:.3f}, {lo:.3f})'
                      for i, la, lo in zip(g['point_id'], g['lat'], g['lon'])]
    else:
        ll = g.to_crs('EPSG:4326')
        g['label'] = [f'P{i} ({p.y:.3f}, {p.x:.3f})'
                      for i, p in zip(g['point_id'], ll.geometry)]
    return g


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    log(f'points     {POINTS_GPKG}')
    pts = load_points()
    log(f'  {len(pts)} points, CRS {pts.crs.to_string()}')
    for _, r in pts.iterrows():
        log(f'    {r["label"]}')

    log(f'FF raster  {FF_RASTER}')
    with rasterio.open(FF_RASTER) as src:
        log(f'  CRS {src.crs.to_string()}  {src.width}x{src.height}  '
            f'res {src.res[0]:.6f}  nodata {src.nodata}  dtype {src.dtypes[0]}')

    for year in YEARS:
        p = chm_path(year)
        with rasterio.open(p) as src:
            log(f'CHM {year}   {src.width}x{src.height} @ {src.res[0]:.4f} m  '
                f'CRS {src.crs.to_string()}  nodata {src.nodata}  '
                f'band "{src.descriptions[0]}"')

    log('--- scars in the points bbox (+2 km) ---')
    ev = fire_events_at_points(pts)
    log(f'{len(ev)} point-fire events 2019-2025')
    if len(ev):
        for pid, grp in ev.groupby('point_id'):
            hits = ', '.join(f'{int(r.year)}-{int(r.month):02d}' for r in grp.itertuples())
            log(f'  P{pid}: {len(grp)} fires -- {hits}')
    pre = prefire_context(pts)
    log(f'pre-2019 last burn: {pre.to_dict(orient="records")}')

    ff = ff_at_points(pts)
    log(f'FF at points: {ff.tolist()}')

    out = ff_on_grid()
    with rasterio.open(out) as src:
        a = src.read(1)
    vals, counts = np.unique(a[a != FF_NODATA], return_counts=True)
    log('FF histogram over pindan (years burnt 2016-2025):')
    tot = counts.sum()
    for v, c in zip(vals, counts):
        log(f'  {v}: {c:>9,} px ({100 * c / tot:5.1f} %)')

    log(f'self-test done ({time.time() - t0:.0f} s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
