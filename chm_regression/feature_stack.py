"""Stage 2 -- Feature stack builder.

Faithful port of cells 1-3 and 14-47 of
``Dampier_Peninsula_strat_4class_YEAR.ipynb``: PALSAR-2 gamma-0 + power
decomposition load, the full Sentinel-1 Stokes / entropy / DOP / alpha /
power-decomposition block, Sentinel-2 8-band geomedian, all reprojected onto
the PALSAR gamma-0 geobox and concatenated into one ``band``-dimensioned
``xr.DataArray``.

This module changes NOTHING about the notebook's maths. In particular the S1
decomposition (cells 25-37: ``ws = 5`` box-filter convolution, Stokes vector,
entropy, DOP, the 5-95 percentile ``S_norm`` clip, ``NESZ = -16`` valid/noise
branching, Verma et al. 2024 dual-pol alpha) is transcribed verbatim. Do not
"improve" or refactor it -- divergence here silently invalidates comparison
with the existing classification series.

Usage
-----
conda run -n general --no-capture-output python feature_stack.py --year 2019
conda run -n general --no-capture-output python feature_stack.py --year 2019 --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import xarray as xr
from dask.distributed import Client, LocalCluster
from odc.geo.xr import xr_reproject
from rasterio.enums import Resampling
from scipy.signal import convolve2d

# ---------------------------------------------------------------------------
# Paths -- transcribed verbatim (as G:\ strings) from cells 12/14/21/23 of
# Dampier_Peninsula_strat_4class_YEAR.ipynb. Translate with win_to_linux() at
# run time only; do not mass-rewrite.
# ---------------------------------------------------------------------------

DAMPIER_CUTBOX_WIN = r'G:\Australia_shapefile\Dampier_peninsula_cutbox.gpkg'
S2_ZARR_TMPL_WIN = r'G:\python\odc-stac\ds_median_DP_{year}_07.zarr'
S1_ZARR_TMPL_WIN = r'G:\python\dampier_veg_timeseries\s1_july_{year}_DP.zarr'
PALSAR_DIR_TMPL_WIN = r'G:\Yawuru\WCM\SAR\PALSAR {year} DP'

# S2 bands kept, and their renamed labels (cell 21).
S2_BANDS = ['nbart_blue', 'nbart_green', 'nbart_red', 'nbart_red_edge_1',
            'nbart_red_edge_2', 'nbart_red_edge_3', 'nbart_nir_1', 'nbart_nir_2']
S2_BAND_LABELS = ['Blue', 'Green', 'Red', 'Red-edge 1', 'Red-edge 2',
                   'Red-edge 3', 'NIR 1', 'NIR 2']

CRS = 'epsg:32751'


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
    """Resolve (HH, HV) gamma-0 refined-Lee-filtered PALSAR paths for a year.

    Prefers the notebook's expected ``HH_g_rflee.tif`` / ``HV_g_rflee.tif``
    naming (cell 14). Falls back to ``HH_g_rflee_w5.tif`` / ``HV_g_rflee_w5.tif``
    (older SNAP window-5 route, e.g. pre-regen 2025) and prints which one it
    used. Raises if neither filtered pair exists -- never silently falls back
    to the unfiltered ``HH_g.tif``/``HV_g.tif``, since mixing filtered and
    unfiltered gamma-0 across years is exactly the artefact this project
    exists to eliminate.
    """
    base = Path(win_to_linux(PALSAR_DIR_TMPL_WIN.format(year=year)))
    candidates = [
        (base / 'HH_g_rflee.tif', base / 'HV_g_rflee.tif'),
        (base / 'HH_g_rflee_w5.tif', base / 'HV_g_rflee_w5.tif'),
    ]
    for hh, hv in candidates:
        if hh.exists() and hv.exists():
            print(f'[feature_stack] resolved gamma-0 speckle-filtered pair for {year}: '
                  f'{hh.name} / {hv.name}')
            return hh, hv
    raise FileNotFoundError(
        f'No speckle-filtered gamma-0 HH/HV pair found under {base} '
        f'(looked for HH_g_rflee.tif and HH_g_rflee_w5.tif, and matching HV). '
        f'Refusing to fall back to unfiltered HH_g.tif/HV_g.tif.'
    )


def start_cluster(n_workers: int = 2, threads_per_worker: int = 2,
                   memory_limit: str = '16GB') -> tuple[LocalCluster, Client]:
    """Reproduce cell 3's LocalCluster pattern, capped at 2 workers per the
    concurrent-I/O restriction on the NTFS/USB drive while PALSAR
    reprocessing is running elsewhere.
    """
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )
    client = cluster.get_client()
    print(f'[feature_stack] dask dashboard: {client.dashboard_link}')
    return cluster, client


# ---------------------------------------------------------------------------
# PALSAR-2 (cells 14-19)
# ---------------------------------------------------------------------------

def load_palsar(year: str, dask_chunk_size: int) -> xr.DataArray:
    hh_path, hv_path = find_palsar_hh_hv(year)
    palsar_dir = Path(win_to_linux(PALSAR_DIR_TMPL_WIN.format(year=year)))
    sar_decomp_path = palsar_dir / 'img.tif'
    sar_alpha_path = palsar_dir / 'Alpha_dp_grd.tif'

    hh_g = rxr.open_rasterio(hh_path, chunks=dask_chunk_size)
    hv_g = rxr.open_rasterio(hv_path, chunks=dask_chunk_size)
    sar_g0 = xr.concat([hh_g, hv_g], dim='band').assign_coords(
        band=[r'$\gamma^{0}_{HH}$', r'$\gamma^{0}_{HV}$']
    )
    hh_g = None
    hv_g = None
    sar_decomp = rxr.open_rasterio(sar_decomp_path, chunks=dask_chunk_size).assign_coords(
        band=['Dihedral', 'Surface', 'Unpolarised']
    )
    sar_alpha = rxr.open_rasterio(sar_alpha_path, chunks=dask_chunk_size).assign_coords(
        band=[r'$\alpha$']
    )

    # cell 17: clip gamma-0 to the Dampier cutbox
    dampier_path = win_to_linux(DAMPIER_CUTBOX_WIN)
    dampier_geom = gpd.read_file(dampier_path).to_crs(CRS)
    sar_g0 = sar_g0.rio.clip(dampier_geom.iloc[0:1].geometry)

    # cell 18: reproject decomp + alpha onto the (now clipped) gamma-0 geobox
    sar_decomp = xr_reproject(sar_decomp, sar_g0.odc.geobox, resampling=Resampling.nearest, nodata=np.nan)
    sar_alpha = xr_reproject(sar_alpha, sar_g0.odc.geobox, resampling=Resampling.nearest, nodata=np.nan)

    # cell 19
    sar_data = xr.concat([sar_g0, sar_decomp, sar_alpha], dim='band', join='override')
    return sar_data


# ---------------------------------------------------------------------------
# Sentinel-2 (cell 21)
# ---------------------------------------------------------------------------

def load_s2(year: str, dask_chunk_size: int, sar_data: xr.DataArray) -> xr.Dataset:
    s2_path = win_to_linux(S2_ZARR_TMPL_WIN.format(year=year))
    s2 = xr.open_zarr(s2_path).rio.write_crs(CRS).chunk(dask_chunk_size)
    s2 = s2.rename(dict(zip(S2_BANDS, S2_BAND_LABELS)))
    s2 = s2[S2_BAND_LABELS]
    s2 = xr_reproject(s2, sar_data.odc.geobox, resampling=Resampling.nearest, nodata=np.nan)
    return s2


# ---------------------------------------------------------------------------
# Sentinel-1 load + decomposition (cells 23-42) -- TRANSCRIBED VERBATIM.
# Variable names kept close to the notebook so the port is auditable
# line-by-line against cells 25-37. Do not vectorise/refactor the maths.
# ---------------------------------------------------------------------------

def load_s1_decomposed(year: str, sar_data: xr.DataArray) -> xr.DataArray:
    s1_path = win_to_linux(S1_ZARR_TMPL_WIN.format(year=year))
    s1_da = xr.open_zarr(s1_path).squeeze().drop_vars(['time', 'spatial_ref']).to_dataarray(dim='band')
    s1_da = s1_da.rio.write_nodata(np.nan).rio.write_crs(CRS)

    # cell 25
    C11 = s1_da.sel(band='VV').values
    C22 = s1_da.sel(band='VH').values

    # cell 26 -- ws = 5 box-filter convolution, verbatim
    ws = 5
    kernel = np.ones((ws, ws), np.float32) / (ws * ws)
    pad = int((ws - 1) / 2)

    C11_av = convolve2d(C11, kernel, mode='valid')
    C11_av = np.pad(C11_av, pad_width=pad, mode='constant', constant_values=0)
    C11_av_db = 10 * np.log10(C11_av)

    C22_av = convolve2d(C22, kernel, mode='valid')
    C22_av = np.pad(C22_av, pad_width=pad, mode='constant', constant_values=0)

    # cell 27 -- Stokes vector elements + entropy eigenvalues
    s0 = C11_av + C22_av
    s1 = C11_av - C22_av

    prob1 = C11_av / (C11_av + C22_av)
    prob2 = C22_av / (C11_av + C22_av)

    # cell 28 -- entropy
    ent = -prob1 * np.log2(prob1) - prob2 * np.log2(prob2)

    # cell 29 -- DOP, beta
    dop = (C11_av - C22_av) / (C11_av + C22_av)
    dop = np.abs(dop)
    beta = C11_av / (C11_av + C22_av)

    # cell 30 -- abs of Stokes vector elements
    s0 = np.abs(s0)  # Already postive
    s1 = np.abs(s1)

    # cell 31 -- slope mask stub (kept as in notebook: always all-ones)
    Slope_msk = np.where(False, 0, 1)

    # cell 32 -- slope-mask the Stokes vector elements
    C11_mask = np.multiply(C11_av, Slope_msk)
    C22_mask = np.multiply(C22_av, Slope_msk)
    s1_mask = np.multiply(s1, Slope_msk)

    # cell 33 -- S_norm: 5-95 percentile clip then max-normalise
    def S_norm(S_array):
        S_5 = np.nanpercentile(S_array, 5)
        S_95 = np.nanpercentile(S_array, 95)
        S_cln = np.where(S_array > S_95, S_95, S_array)
        S_cln = np.where(S_cln < S_5, S_5, S_cln)
        S_cln_max = np.nanmax(S_cln)
        S_norm_array = np.divide(S_cln, S_cln_max)
        return S_norm_array

    # cell 34
    C11_norm = S_norm(C11_mask)
    C22_norm = S_norm(C22_mask)
    s1_norm = S_norm(s1_mask)
    s1_s_norm = S_norm(s1)  # S1 normalized for DpRSI, no slope mask

    # cell 35 -- power calculation
    dprbi = np.sqrt(np.square(C11_norm) + np.square(C22_norm)) / np.sqrt(2)
    dprbi = dprbi * s1_norm

    dprsi_con1 = (1 - ent) * np.sqrt(1 - np.square(s1_s_norm))  # For Valid pixels
    dprsi_con2 = np.sqrt(1 - np.square(s1_s_norm))  # For Noise pixels

    NESZ = -16  # For Sentinel-1
    dprsi = np.where(C11_av_db > NESZ, dprsi_con1, dprsi_con2)

    alpha1 = np.arctan2(dprbi, 1 - dprbi)
    alpha1 = np.degrees(alpha1)

    alpha2 = np.arctan2(1 - dprsi, dprsi)
    alpha2 = np.degrees(alpha2)

    alpha_dp = (alpha1 + alpha2) / 2  # Dual-pol target characteristic parameter (Verma et al. 2024)

    # cell 36 -- alpha as geometric factor, power components
    alpha_dp_rad = np.radians(2 * alpha_dp)
    cos_a = np.cos(alpha_dp_rad)

    # Power components for valid pixels (VV > NESZ)
    Pu_v = (1 - dop) * s0
    Pd_v = (1 / 2) * dop * s0 * (1 - cos_a)
    Ps_v = (1 / 2) * dop * s0 * (1 + cos_a)

    # Power components for noise pixels (VV < NESZ)
    Pu_n = (1 - beta) * s0
    Pd_n = (1 / 2) * beta * s0 * (1 - cos_a)
    Ps_n = (1 / 2) * beta * s0 * (1 + cos_a)

    # Dual-pol scattering power
    Pu = np.where(C11_av_db > NESZ, Pu_v, Pu_n)  # Unpolarized power
    Pd = np.where(C11_av_db > NESZ, Pd_v, Pd_n)  # "Dihedral-like" power
    Ps = np.where(C11_av_db > NESZ, Ps_v, Ps_n)  # "Surface-like" power

    dprbi = dprbi.astype(np.float32)
    dprsi = dprsi.astype(np.float32)
    Pd = Pd.astype(np.float32)
    Ps = Ps.astype(np.float32)
    Pu = Pu.astype(np.float32)

    # cell 37
    new_arrays = {
        's1_alpha': alpha_dp,
        's1_Pd': Pd,
        's1_Ps': Ps,
        's1_Pu': Pu,
    }
    new_das = [
        xr.DataArray(arr, dims=['y', 'x'], coords={'y': s1_da.y, 'x': s1_da.x}).assign_coords(band=name)
        for name, arr in new_arrays.items()
    ]
    s1_da = xr.concat([s1_da, *new_das], dim='band').assign_coords(
        band=list(s1_da.band.values) + list(new_arrays.keys())
    )

    # cell 39 -- reproject onto PALSAR gamma-0 geobox
    s1_da = xr_reproject(s1_da, sar_data.odc.geobox, resampling=Resampling.nearest, nodata=np.nan)

    # cell 41 -- snap to exact coords
    s1_da = s1_da.sel({'x': sar_data.x, 'y': sar_data.y}, method='nearest').assign_coords(
        {'x': sar_data.x, 'y': sar_data.y}
    )

    return s1_da


# ---------------------------------------------------------------------------
# Full stack assembly
# ---------------------------------------------------------------------------

def build_stack(year: str, dask_chunk_size: int = 4096) -> xr.DataArray:
    """Reproduce cells 14-47: PALSAR + S1 decomposition + S2, concatenated
    into one band-dimensioned DataArray on the PALSAR gamma-0 geobox.
    """
    year = str(year)

    sar_data = load_palsar(year, dask_chunk_size)          # cells 14-19
    s2 = load_s2(year, dask_chunk_size, sar_data)           # cell 21
    s1_da = load_s1_decomposed(year, sar_data)              # cells 23-41

    # cell 42
    sar_data = xr.concat([sar_data, s1_da], dim='band', join='exact').rio.write_nodata(np.nan)

    # cell 45
    sar_data = sar_data.chunk(dask_chunk_size)

    # cell 47
    data_stack = xr.concat(
        [sar_data, s2.to_dataarray(dim='band').assign_coords({'x': sar_data.x, 'y': sar_data.y})],
        dim='band', join='exact',
    )
    data_stack.name = 'data_stack'
    return data_stack


def cache_path(year: str, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (Path(__file__).resolve().parent / 'outputs' / str(year))
    return out_dir / f'feature_stack_{year}.zarr'


def build_and_cache(year: str, dask_chunk_size: int = 4096, out_dir: Path | None = None,
                     overwrite: bool = False) -> Path:
    path = cache_path(year, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f'[feature_stack] cache already exists, skipping build: {path}')
        return path

    stack = build_stack(year, dask_chunk_size=dask_chunk_size)

    # The three concats leave the band dim ragged -- PALSAR arrives as one
    # 12-band chunk, then S1/S2 bands one per chunk -- which zarr rejects
    # ("uniform chunk sizes except for final chunk"). Band is a tiny dim, so
    # collapse it to a single chunk; x/y keep dask_chunk_size. Values untouched.
    stack = stack.chunk({'band': -1, 'y': dask_chunk_size, 'x': dask_chunk_size})

    ds = stack.to_dataset(name='data_stack')
    ds.to_zarr(path, mode='w')
    print(f'[feature_stack] wrote cache -> {path}')
    return path


def verify(year: str, dask_chunk_size: int = 4096) -> dict:
    stack = build_stack(year, dask_chunk_size=dask_chunk_size)
    stack = stack.compute()

    bands = list(stack.band.values)
    shape = stack.shape
    crs = stack.rio.crs
    res = stack.rio.resolution()

    vals = stack.values
    all_nan_pixels = np.all(np.isnan(vals), axis=0)
    total_pixels = all_nan_pixels.size
    frac_all_nan = float(all_nan_pixels.sum()) / total_pixels

    print(f'[feature_stack] verify({year}): {len(bands)} bands, shape {shape}, '
          f'crs {crs}, pixel size {res}, all-NaN pixel fraction {frac_all_nan:.4f}')
    for b in bands:
        print(f'    band: {b}')

    return {
        'bands': bands,
        'shape': shape,
        'crs': str(crs),
        'resolution': res,
        'frac_all_nan': frac_all_nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', default='2019', help='Year to build the feature stack for')
    parser.add_argument('--dask-chunk-size', type=int, default=4096)
    parser.add_argument('--out-dir', default=None, help='Output directory for the zarr cache')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verify', action='store_true',
                         help='Compute the stack and print band list / shape / CRS / pixel '
                              'size / all-NaN fraction instead of caching to zarr')
    parser.add_argument('--n-workers', type=int, default=2)
    parser.add_argument('--memory-limit', default='16GB')
    args = parser.parse_args()

    cluster, client = start_cluster(n_workers=args.n_workers, memory_limit=args.memory_limit)
    try:
        if args.verify:
            verify(args.year, dask_chunk_size=args.dask_chunk_size)
        else:
            out_dir = Path(args.out_dir) if args.out_dir else None
            build_and_cache(args.year, dask_chunk_size=args.dask_chunk_size,
                             out_dir=out_dir, overwrite=args.overwrite)
    finally:
        client.close()
        cluster.close()


if __name__ == '__main__':
    main()
