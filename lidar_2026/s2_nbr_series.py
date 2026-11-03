"""EXPLORATORY, separate from the main fire analysis (fire_vs_sar.py, NAFI only). Sentinel-2 NBR time
series on the NISAR 20 m grid of each site, and fire timing / severity per cell.

The user (2026-10-21): good Sentinel-2 burn maps need much manual work, otherwise they contain
artefacts. This mapping has no manual checking. Its rules (the regional baseline, the fire-season
windows) were set by comparison with NAFI at the 5 sites. Use the results as a pointer only.
Analysis: s2_fire_vs_sar.py. Outputs: outputs/structure_sar/s2_fire/.

NAFI is on a 250 m grid, even where it is mapped from Sentinel-2. This script tries to give the same
information at 20 m:

    load     DEA STAC S2A/B/C ARD (2025-04-01 .. 2026-10-21), nbart_nir_2 (B8A), nbart_swir_3 (B12) and
             oa_fmask, read straight onto each site's 20 m window of the NISAR grid (EPSG:32751; the
             struct_nisar20m_<site>.tif window) with odc.stac (windowed COG reads, 2 threads,
             solar_day groups, average resampling, fmask nearest). NBR = (B8A - B12) / (B8A + B12);
             fmask 0 / 2 / 3 (nodata, cloud, shadow) -> NaN. -> outputs/structure_sar/s2_fire/s2_nbr_<site>.zarr
    baseline the median NBR of all cells of the 5 sites on each date -> outputs/structure_sar/s2_fire/s2_nbr_baseline.csv.
             The cell series minus this baseline (the anomaly) removes the seasonal fall of NBR as the
             grass cures (March-June) and the whole-scene jumps of about +-0.1 between dates
             (atmosphere, view angle). Without it, the test of Bunda Extra marked 29 % of the site as
             burnt in March-April 2026, where NAFI has no fire. When 9 Mile and South Dampier burn
             (July 2026, about 35 % of all cells), the baseline falls a little, so these drops are
             slightly too small.
    metrics  per cell, the largest robust drop of the anomaly in each fire year: step = median of the 3
             valid observations before - median of the 3 from that date on. dnbr_<y> = the step, doy_<y> =
             day of year of the first observation after the drop (the fire is between it and the
             observation before). Burnt if dnbr >= BURNT_DNBR (0.1, the usual "unburnt" limit of dNBR;
             no site-specific tuning). nbr_now = median NBR of the last 3 valid observations.
             -> outputs/structure_sar/s2_fire/s2_burn_<site>.tif (bands named in the descriptions; nodata NaN)

Fire seasons searched: 2025-05-01 .. 2025-10-31 and 2026-05-01 .. 2026-10-21. Drops in the wet season
(November-April) are not used: with the year-round search, 33 % of Country Downs and 10-30 % of Biota
track and Bunda Extra "burnt" in January-April, and 8 % of Country Downs in November 2025, where NAFI has
fire in only 1 % of one site (Country Downs, January 2026). Wet-season green-up and cloud residue make
these drops. All NAFI fire months at the 5 sites (2023-2026) are May-October. The series starts in April 2025, so 3 valid observations are
available before the first May fire.

Usage (<= 2 cores; set OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1)
-----
conda run -n general --no-capture-output python -u s2_nbr_series.py              (all sites: load + metrics)
conda run -n general --no-capture-output python -u s2_nbr_series.py --sites "9 Mile"   (load one site)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pystac_client
import rasterio
import xarray as xr

from structure_grids import EXCLUDE_SAR, OUT_DIR

STAC_URL = 'https://explorer.dea.ga.gov.au/stac'
COLLECTIONS = ['ga_s2am_ard_3', 'ga_s2bm_ard_3', 'ga_s2cm_ard_3']
S3 = ('s3://dea-public-data/', 'https://dea-public-data.s3.ap-southeast-2.amazonaws.com/')
START, END = '2025-04-01', '2026-10-21'
YEARS = {2025: ('2025-05-01', '2025-10-31'), 2026: ('2026-05-01', END)}  # fire seasons only (see above)
BANDS = ['nbart_nir_2', 'nbart_swir_3', 'oa_fmask']
FMASK_BAD = [0, 2, 3]  # nodata, cloud, shadow
K = 3  # valid observations on each side of a step
BURNT_DNBR = 0.1
FIRE_DIR = OUT_DIR / 's2_fire'


def windows(sites=None):
    for f in sorted(OUT_DIR.glob('struct_nisar20m_*.tif')):
        site = f.stem[len('struct_nisar20m_'):]
        if site.endswith('.partial') or site in EXCLUDE_SAR or (sites and site not in sites):
            continue
        with rasterio.open(f) as s:
            yield site, s.transform, (s.height, s.width)


def slug(site):
    return site.lower().replace(' ', '_')


def load(site, tf, shape) -> xr.DataArray:
    out = FIRE_DIR / f's2_nbr_{slug(site)}.zarr'
    if out.exists():
        print(f'[s2] {site}: {out.name} exists, reused', flush=True)
        return xr.open_zarr(out).nbr.load()
    import odc.stac
    from odc.geo.geobox import GeoBox
    gbox = GeoBox(shape, tf, 'EPSG:32751')
    bb = gbox.extent.to_crs('EPSG:4326').boundingbox
    items = pystac_client.Client.open(STAC_URL).search(
        collections=COLLECTIONS, bbox=[bb.left, bb.bottom, bb.right, bb.top], datetime=f'{START}/{END}').item_collection()
    t0 = time.time()
    ds = odc.stac.load(items, bands=BANDS, geobox=gbox, groupby='solar_day', pool=2,
                       resampling={'oa_fmask': 'nearest', '*': 'average'}, patch_url=lambda u: u.replace(*S3))
    nir, swir = ds.nbart_nir_2.astype('float32'), ds.nbart_swir_3.astype('float32')
    bad = ds.oa_fmask.isin(FMASK_BAD) | (nir <= 0) | (swir <= 0)
    nbr = ((nir - swir) / (nir + swir)).where(~bad).rename('nbr')
    clear = float(np.isfinite(nbr).mean())
    print(f'[s2] {site}: {len(items)} items, {nbr.sizes["time"]} solar days, {100 * clear:.0f} % clear pixel-dates, '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)
    nbr.to_dataset().drop_vars('spatial_ref', errors='ignore').to_zarr(out, mode='w')
    return nbr


def steps(v: np.ndarray, t: np.ndarray) -> dict:
    """Largest robust NBR drop per fire year for one cell (v: NBR with NaN, t: dates)."""
    ok = np.isfinite(v)
    v, t = v[ok], t[ok]
    out = {}
    for y, (a, b) in YEARS.items():
        best, when = np.nan, np.nan
        for k in range(K, len(v) - K + 1):
            if np.datetime64(a) <= t[k] <= np.datetime64(b):
                s = np.median(v[k - K:k]) - np.median(v[k:k + K])
                if not best >= s:
                    best, when = s, (t[k] - np.datetime64(f'{y}-01-01')).astype('timedelta64[D]').astype(int) + 1
        out[f'dnbr_{y}'], out[f'doy_{y}'] = best, when
    out['nbr_now'] = np.median(v[-K:]) if len(v) >= K else np.nan
    out['n_valid'] = len(v)
    return out


def baseline(nbrs: dict) -> pd.Series:
    """Median NBR of all cells of all sites per date."""
    vals = {}
    for nbr in nbrs.values():
        t = nbr.time.values.astype('datetime64[D]')
        a = nbr.values.reshape(len(t), -1)
        for d, row in zip(t, a):
            vals.setdefault(d, []).append(row[np.isfinite(row)])
    b = pd.Series({d: np.median(np.concatenate(v)) if sum(map(len, v)) >= 1000 else np.nan for d, v in vals.items()})
    b.index.name = 'date'
    b.rename('nbr_median').to_csv(FIRE_DIR / 's2_nbr_baseline.csv')
    print(f'[s2] baseline: {b.notna().sum()} dates with >= 1000 clear cells', flush=True)
    return b


def metrics(site, tf, shape, nbr: xr.DataArray, base: pd.Series) -> None:
    t = nbr.time.values.astype('datetime64[D]')
    a = nbr.values.reshape(len(t), -1)
    anom = a - base.reindex(t).to_numpy()[:, None]
    rows = [{**steps(anom[:, i], t), 'nbr_now': steps(a[:, i], t)['nbr_now']} for i in range(a.shape[1])]
    df = pd.DataFrame(rows)
    names = list(df.columns)
    arr = df.to_numpy('float32').T.reshape(len(names), *shape)
    out = FIRE_DIR / f's2_burn_{slug(site)}.tif'
    prof = dict(driver='GTiff', width=shape[1], height=shape[0], count=len(names), dtype='float32',
                crs='EPSG:32751', transform=tf, nodata=np.nan, compress='deflate')
    with rasterio.open(out, 'w', **prof) as d:
        d.write(arr)
        d.descriptions = tuple(names)
    for y in YEARS:
        b = df[f'dnbr_{y}'] >= BURNT_DNBR
        m = pd.to_datetime(f'{y}-01-01') + pd.to_timedelta(df.loc[b, f'doy_{y}'] - 1, 'D')
        print(f'[s2] {site} {y}: {100 * b.mean():.0f} % of window cells with dNBR >= {BURNT_DNBR}; '
              f'months {m.dt.month.value_counts(normalize=True).round(2).sort_index().to_dict()}', flush=True)
    print(f'[s2] {site}: wrote {out.resolve()}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sites', nargs='*', help='load these sites only (implies --load-only)')
    p.add_argument('--load-only', action='store_true')
    args = p.parse_args()
    args.load_only = args.load_only or bool(args.sites)
    FIRE_DIR.mkdir(parents=True, exist_ok=True)
    w = list(windows(args.sites))
    nbrs = {site: load(site, tf, shape) for site, tf, shape in w}
    if args.load_only:
        return
    assert args.sites is None, 'the baseline needs all sites: run metrics without --sites'
    base = baseline(nbrs)
    for site, tf, shape in w:
        metrics(site, tf, shape, nbrs[site], base)


if __name__ == '__main__':
    main()
