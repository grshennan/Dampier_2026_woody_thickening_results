"""Stage 1 -- threshold translation between height statistics.

Builds the per-SAR-pixel table the whole study rests on: for each of the 13
2025 drone-LiDAR CHMs, aggregated onto the **2025 SAR grid**, the mean / p90 /
p98 of the 50 cm p98-canopy values plus **canopy cover at each nominal height**,
clipped to the survey cut areas and flagged for pindan membership.

Why cover is here and not in ``chm_regression/chm_targets.py``: Stage 1a
established that the operative class boundaries were never defined as "the mean
height corresponding to a nominal canopy height". They were defined by a
**cover fit** -- regress mean height on canopy-cover-at-h over SAR pixels, read
the mean height at **20 % cover** (the Australian NFI forest definition at
h = 2 m). Reproducing that fit against p90 instead of mean is therefore the
*native* translation, not an analogy, and it needs cover per SAR pixel.

Reuses ``chm_targets``' vectorised inverse-affine binning; the only extension is
the cover statistics in ``aggregate_with_cover``.

Usage
-----
conda run -n general --no-capture-output python translation.py --stage extract
conda run -n general --no-capture-output python translation.py --stage derive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr  # noqa: F401  -- registers the .rio accessor
import xarray as xr
from affine import Affine

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'chm_regression'))

from chm_targets import (  # noqa: E402
    AREA_NAMES, LIDAR_PATHS_WIN, LIDAR_POLYS_WIN, LOW_COUNT_THRESHOLD,
    build_sar_reference, target_grid_for_chm, win_to_linux,
)

# The 2025 grid is the declared common grid for this study (user decision
# 2026-08-11): coarser than the other four years, so reprojecting onto it never
# upsamples, and 2025's 74.7 % footprint sets the intersection mask anyway.
GRID_YEAR = '2025'

# v04's nominal 50 cm cover heights. Superset of the operative 4-class scheme,
# so the 9-class ladder for the Stage 6c resolution probe falls out of the same
# extraction.
COVER_HEIGHTS = [0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

# The cover fraction the class boundaries are read at. 20 % at h = 2 m is the
# Australian NFI forest definition; the same fraction was applied at every
# other height for consistency.
COVER_FRAC = 0.20

# Statistics to derive translated thresholds for.
STATISTICS = ['mean', 'p90', 'p98']

# The operative 4-class scheme, on the mean-of-p98 scale.
OPERATIVE_CUTS = {'bare': 0.2, 'low_forest': 1.22, 'forest': 2.35}

# Stage 1a: what the cover fit actually produces for the same nominal heights,
# per v03 and v04 (identical -- same method, same data). 1.22 is NOT this
# number; see REPORT_threshold_translation.md.
PUBLISHED_FIT = {2.0: 1.146, 5.0: 2.338, 10.0: 5.381}

PINDAN_WIN = r'G:\Veg and Soil Mapping\Dampier_pindan_main.gpkg'

OUT = HERE / 'outputs' / 'translation'


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def aggregate_with_cover(chm: xr.DataArray, target: xr.DataArray,
                         heights: list[float] | None = None,
                         cond_heights: list[float] = ()) -> xr.Dataset:
    """``chm_targets.aggregate_chm_to_grid`` plus cover-at-h bands.

    Cover at h is the fraction of the SAR pixel's 50 cm CHM pixels whose
    canopy height is >= h -- the same definition the v04 notebook uses.
    cond_heights (optional, class_defs 2026-11-02): bands mhi_<h> / mlo_<h> =
    mean of the 50 cm cells >= h / < h (NaN where there are none), so that
    mean = cover_h * mhi_h + (1 - cover_h) * mlo_h.
    """
    if heights is None:
        heights = COVER_HEIGHTS
    transform: Affine = target.rio.transform()
    height, width = target.sizes['y'], target.sizes['x']
    inv = ~transform

    values = chm.values.astype('float64').ravel()
    valid = np.isfinite(values)
    if chm.rio.nodata is not None and not np.isnan(chm.rio.nodata):
        valid &= values != chm.rio.nodata

    stat_names = (['mean', 'p90', 'p98', 'count'] + [f'cover_{h}' for h in heights]
                  + [f'{s}_{h}' for h in cond_heights for s in ('mhi', 'mlo')])
    if not valid.any():
        out = {s: xr.full_like(target, np.nan) for s in stat_names}
        out['count'] = xr.full_like(target, 0, dtype='int32')
        return xr.Dataset(out)

    xs, ys = np.meshgrid(chm.x.values, chm.y.values)
    xs = xs.ravel()[valid]
    ys = ys.ravel()[valid]
    values = values[valid]

    cols_f, rows_f = inv * (xs, ys)
    cols = np.floor(cols_f).astype('int64')
    rows = np.floor(rows_f).astype('int64')

    in_grid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cols, rows, values = cols[in_grid], rows[in_grid], values[in_grid]

    df = pd.DataFrame({'row': rows, 'col': cols, 'value': values})
    for h in heights:
        df[f'cover_{h}'] = (df['value'] >= h).astype('float64')

    agg_spec = {
        'mean': ('value', 'mean'),
        'p90': ('value', lambda s: np.percentile(s, 90)),
        'p98': ('value', lambda s: np.percentile(s, 98)),
        'count': ('value', 'count'),
    }
    for h in heights:
        agg_spec[f'cover_{h}'] = (f'cover_{h}', 'mean')
    for h in cond_heights:
        hi = df['value'] >= h
        df[f'mhi_{h}'] = df['value'].where(hi)
        df[f'mlo_{h}'] = df['value'].where(~hi)
        agg_spec[f'mhi_{h}'] = (f'mhi_{h}', 'mean')  # NaN-skipping mean
        agg_spec[f'mlo_{h}'] = (f'mlo_{h}', 'mean')

    agg = df.groupby(['row', 'col']).agg(**agg_spec)

    r_idx = agg.index.get_level_values('row')
    c_idx = agg.index.get_level_values('col')

    out = {}
    for stat in stat_names:
        dtype = 'int32' if stat == 'count' else 'float32'
        fill = 0 if stat == 'count' else np.nan
        arr = np.full((height, width), fill, dtype=dtype)
        arr[r_idx, c_idx] = agg[stat].values
        out[stat] = xr.DataArray(arr, dims=('y', 'x'),
                                 coords={'y': target.y, 'x': target.x})

    ds = xr.Dataset(out).rio.write_crs(target.rio.crs).rio.write_transform(transform)
    return ds


def extract() -> pd.DataFrame:
    """Build the per-cell table over all 13 areas on the 2025 grid."""
    ref = build_sar_reference(GRID_YEAR)
    pindan = gpd.read_file(win_to_linux(PINDAN_WIN))

    frames = []
    for idx, name in enumerate(AREA_NAMES):
        chm_path = win_to_linux(LIDAR_PATHS_WIN[idx])
        poly_path = win_to_linux(LIDAR_POLYS_WIN[idx])

        chm = rxr.open_rasterio(chm_path).squeeze(drop=True)
        nodata = chm.rio.nodata
        if nodata is not None:
            chm = chm.where(chm != nodata, np.nan)
        chm = chm.rio.write_nodata(np.nan)

        target = target_grid_for_chm(chm, ref)
        ds = aggregate_with_cover(chm, target)

        # The Stage 1 clip fix: the cut area is load-bearing. Without it a
        # third of the cells are margin the classification never trained on.
        cut = gpd.read_file(poly_path)
        if cut.crs != ds.rio.crs:
            cut = cut.to_crs(ds.rio.crs)
        ds = ds.rio.clip(cut.geometry, drop=True)

        df = ds.to_dataframe().reset_index()
        df = df[df['count'] >= LOW_COUNT_THRESHOLD].copy()
        df = df[np.isfinite(df['mean'])]
        df['area'] = name

        pts = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df['x'], df['y']), crs=ds.rio.crs)
        pin = pindan.to_crs(ds.rio.crs)
        df['in_pindan'] = pts.geometry.within(pin.union_all()).values

        log(f'  {name:22s} {len(df):6,d} cells  '
            f'pindan {int(df["in_pindan"].sum()):6,d}')
        frames.append(df.drop(columns=[c for c in ('spatial_ref',) if c in df]))

    out = pd.concat(frames, ignore_index=True)
    log(f'\ntotal {len(out):,} cells across {out["area"].nunique()} areas; '
        f'pindan {int(out["in_pindan"].sum()):,} across '
        f'{out.loc[out["in_pindan"], "area"].nunique()} areas')
    return out


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

def cover_fit(df: pd.DataFrame, stat: str, h: float) -> dict:
    """Replicate the v04 derivation: linear fit of `stat` on cover-at-h over
    SAR cells, evaluated at COVER_FRAC. Returns the threshold and fit quality.
    """
    x = df[f'cover_{h}'].to_numpy()
    y = df[stat].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30:
        return {'threshold': np.nan, 'n': len(x), 'r2': np.nan, 'se': np.nan}

    coeffs = np.polyfit(x, y, deg=1)
    fn = np.poly1d(coeffs)
    resid = y - fn(x)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan

    # SE of the fitted value at COVER_FRAC (ordinary least squares).
    n = len(x)
    s2 = (resid ** 2).sum() / (n - 2)
    sxx = ((x - x.mean()) ** 2).sum()
    se = float(np.sqrt(s2 * (1 / n + (COVER_FRAC - x.mean()) ** 2 / sxx))) if sxx > 0 else np.nan

    return {'threshold': float(fn(COVER_FRAC)), 'n': n, 'r2': float(r2),
            'se': se, 'slope': float(coeffs[0]), 'intercept': float(coeffs[1])}


def quantile_match(df: pd.DataFrame, stat: str, mean_cut: float) -> float:
    """Threshold on `stat` reproducing the areal fraction that `mean_cut`
    gives on the mean. Areal-equivalent by construction."""
    frac = float((df['mean'] < mean_cut).mean())
    return float(np.quantile(df[stat].dropna(), frac))


def agreement_max(df: pd.DataFrame, stat: str, mean_cuts: list[float]) -> tuple[list[float], float]:
    """Thresholds on `stat` maximising per-cell 4-class agreement with the
    mean-based labels. Coordinate ascent from the quantile-matched start."""
    y_ref = np.digitize(df['mean'].to_numpy(), mean_cuts)
    v = df[stat].to_numpy()
    cuts = [quantile_match(df, stat, c) for c in mean_cuts]

    def score(c):
        return float((np.digitize(v, c) == y_ref).mean())

    best = score(cuts)
    for _ in range(12):
        improved = False
        for i in range(len(cuts)):
            lo = cuts[i - 1] + 1e-3 if i > 0 else float(np.nanmin(v))
            hi = cuts[i + 1] - 1e-3 if i < len(cuts) - 1 else float(np.nanmax(v))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            grid = np.linspace(lo, hi, 200)
            for g in grid:
                trial = list(cuts)
                trial[i] = float(g)
                s = score(trial)
                if s > best + 1e-9:
                    best, cuts, improved = s, trial, True
        if not improved:
            break
    return cuts, best


def derive(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """All translation methods, for one population."""
    rows = []

    # M1 -- cover fit, the native method.
    for stat in STATISTICS:
        for h in COVER_HEIGHTS:
            r = cover_fit(df, stat, h)
            rows.append({'population': label, 'method': 'cover_fit', 'stat': stat,
                         'nominal_h': h, **r})

    # M2/M3 -- against the operative mean cuts, for the p90/p98 arms.
    mean_cuts = [OPERATIVE_CUTS['bare'], OPERATIVE_CUTS['low_forest'],
                 OPERATIVE_CUTS['forest']]
    for stat in STATISTICS:
        for cut_name, cut in OPERATIVE_CUTS.items():
            rows.append({'population': label, 'method': 'quantile_match',
                         'stat': stat, 'operative_cut': cut_name,
                         'threshold': quantile_match(df, stat, cut),
                         'n': int(df[stat].notna().sum())})
        cuts, acc = agreement_max(df, stat, mean_cuts)
        for cut_name, c in zip(OPERATIVE_CUTS, cuts):
            rows.append({'population': label, 'method': 'agreement_max',
                         'stat': stat, 'operative_cut': cut_name,
                         'threshold': c, 'agreement': acc,
                         'n': int(df[stat].notna().sum())})

    return pd.DataFrame(rows)


def sensitivity(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """d(class fraction)/d(threshold) at each operative cut, and the areal
    cost of the 1.22-vs-1.146 discrepancy Stage 1a found."""
    rows = []
    for cut_name, cut in OPERATIVE_CUTS.items():
        eps = 0.05
        f_lo = float((df['mean'] < cut - eps).mean())
        f_hi = float((df['mean'] < cut + eps).mean())
        rows.append({'population': label, 'cut': cut_name, 'value': cut,
                     'd_fraction_per_m': (f_hi - f_lo) / (2 * eps)})

    n = len(df)
    moved = int(((df['mean'] >= PUBLISHED_FIT[2.0]) & (df['mean'] < 1.22)).sum())
    rows.append({'population': label, 'cut': 'low_forest_1.146_vs_1.22',
                 'value': 1.22 - PUBLISHED_FIT[2.0],
                 'cells_reclassified': moved,
                 'frac_reclassified': moved / n if n else np.nan})
    moved5 = int(((df['mean'] >= PUBLISHED_FIT[5.0]) & (df['mean'] < 2.35)).sum())
    rows.append({'population': label, 'cut': 'forest_2.338_vs_2.35',
                 'value': 2.35 - PUBLISHED_FIT[5.0],
                 'cells_reclassified': moved5,
                 'frac_reclassified': moved5 / n if n else np.nan})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=['extract', 'derive', 'all'], default='all')
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cells_path = OUT / 'cells_2025grid.parquet'

    if args.stage in ('extract', 'all'):
        log(f'[extract] 13 areas onto the {GRID_YEAR} SAR grid, count >= {LOW_COUNT_THRESHOLD}')
        df = extract()
        df.to_parquet(cells_path)
        log(f'[extract] wrote {cells_path}')
    else:
        df = pd.read_parquet(cells_path)

    if args.stage in ('derive', 'all'):
        all13 = df
        pin4 = df[df['in_pindan']]

        thr = pd.concat([derive(all13, 'all13'), derive(pin4, 'pindan4')],
                        ignore_index=True)
        thr.to_csv(OUT / 'thresholds.csv', index=False)

        sens = pd.concat([sensitivity(all13, 'all13'), sensitivity(pin4, 'pindan4')],
                         ignore_index=True)
        sens.to_csv(OUT / 'sensitivity.csv', index=False)

        # The control: does the cover fit on `mean` recover the published
        # 1.146 / 2.338? If not, the machinery is wrong before it is applied
        # to p90.
        ctrl = {}
        for h, published in PUBLISHED_FIT.items():
            if h not in COVER_HEIGHTS:
                continue
            got = cover_fit(all13, 'mean', h)['threshold']
            ctrl[f'nominal_{h}m'] = {'published_v04': published,
                                     'recovered': got,
                                     'diff': got - published}
        (OUT / 'control_check.json').write_text(json.dumps(ctrl, indent=2))

        log('\n[control] cover fit on `mean`, all 13 areas, vs published v04:')
        for k, v in ctrl.items():
            log(f'  {k}: published {v["published_v04"]:.3f}  '
                f'recovered {v["recovered"]:.3f}  diff {v["diff"]:+.3f}')

        log('\n[thresholds] cover fit, all 13 areas:')
        cf = thr[(thr.method == 'cover_fit') & (thr.population == 'all13')]
        for stat in STATISTICS:
            s = cf[cf.stat == stat].set_index('nominal_h')['threshold']
            log(f'  {stat:5s}: ' + '  '.join(f'{h:g}m->{s.get(h, np.nan):.3f}'
                                             for h in COVER_HEIGHTS))
        log(f'\n[done] wrote {OUT}/thresholds.csv, sensitivity.csv, control_check.json')


if __name__ == '__main__':
    main()
