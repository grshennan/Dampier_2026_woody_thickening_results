"""Stage 5 -- predict canopy height for every year.

Port of cells 80-88 of ``Dampier_Peninsula_strat_4class_YEAR.ipynb``, with the
class raster replaced by four continuous canopy-height rasters per year.

Applies the Stage 4 models (trained on 2025 features + 2025 drone-LiDAR labels)
to each year's own feature stack, producing one 4-band raster per year:

    figsDP<year>/RF_CHM_<year>.tif    bands 1-4 = mean, max, p98, p90

``PLAN.md`` Stage 5 specified 20 single-band files instead; the user chose the
4-band layout, which holds identical data in one file per year with band
descriptions set, and which Stages 6/7 read just as easily.

float32 metres, CRS and **nodata written explicitly**. The explicit nodata is
not cosmetic: ``nodata=None`` is the exact condition behind the out-of-grid
sampling bug documented in ``GEDI/PLAN.md``, where an out-of-grid sample
silently returns a finite 0.0 instead of failing. Stage 6 samples these rasters,
so leaving nodata unset would corrupt the validation rather than error.

Prediction runs in row blocks: a year's stack is ~20 x 9648 x 7537 float32
(~5.8 GB) and a RandomForest predict on all pixels at once would hold several
multiples of that.

Usage
-----
conda run -n general --no-capture-output python predict_year.py --years 2019 2021 2023 2024 2025
conda run -n general --no-capture-output python predict_year.py --years 2025 --targets mean
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import rioxarray as rxr  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from feature_stack import build_stack, cache_path

HERE = Path(__file__).parent
MODEL_DIR = HERE / 'outputs' / 'models'
FIG_ROOT = Path('/media/glen/CDU_Storage/python/dampier_veg_timeseries')

TARGETS = ['mean', 'max', 'p98', 'p90']
NODATA = np.float32(-9999.0)
YEARS = ['2019', '2021', '2023', '2024', '2025']


def load_model(target: str, model_dir: Path, n_jobs: int = 8):
    path = model_dir / f'rf_{target}.pkl'
    if not path.exists():
        raise FileNotFoundError(f'No Stage 4 model at {path} -- run train_regression.py first')
    with open(path, 'rb') as fh:
        model = pickle.load(fh)
    # The models were pickled with n_jobs=1 (the search parallelised over folds,
    # not within a fit), which would make a 73 Mpx predict single-threaded.
    # 8 is the project cap -- the machine has 24 but must stay usable.
    model.n_jobs = n_jobs
    return model


def load_stack(year: str, dask_chunk_size: int = 4096) -> xr.DataArray:
    path = cache_path(year)
    if path.exists():
        print(f'[predict] using cached stack: {path}', flush=True)
        ds = xr.open_zarr(path)
        da = ds['data_stack']
        # See train_regression.load_stack: rioxarray's scalar spatial_ref coord
        # comes back from zarr as a (band,) data variable and collides with
        # write_crs. Drop and re-declare; the geotransform comes from x/y.
        da = da.drop_vars('spatial_ref', errors='ignore')
        return da.rio.write_crs('epsg:32751')
    print(f'[predict] no cache for {year}; building in memory', flush=True)
    return build_stack(year, dask_chunk_size=dask_chunk_size)


def predict_stack(model, stack_vals: np.ndarray, block_rows: int) -> np.ndarray:
    """Predict over a (band, y, x) array in row blocks.

    Pixels with any non-finite feature are left as NODATA rather than imputed --
    the training table dropped those rows, so the model has never seen them and
    filling them would fabricate canopy height outside the data.
    """
    n_bands, ny, nx = stack_vals.shape
    out = np.full((ny, nx), NODATA, dtype='float32')

    for r0 in range(0, ny, block_rows):
        r1 = min(r0 + block_rows, ny)
        block = stack_vals[:, r0:r1, :]
        X = block.reshape(n_bands, -1).T
        good = np.isfinite(X).all(axis=1)
        if good.any():
            pred = model.predict(X[good])
            flat = np.full(X.shape[0], NODATA, dtype='float32')
            flat[good] = pred.astype('float32')
            out[r0:r1, :] = flat.reshape(r1 - r0, nx)
        print(f'[predict]     rows {r0}-{r1} ({good.sum()} valid px)', flush=True)
    return out


def run_year(year: str, targets: list[str], model_dir: Path, block_rows: int,
             overwrite: bool, n_jobs: int = 8) -> None:
    out_dir = FIG_ROOT / f'figsDP{year}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'RF_CHM_{year}.tif'

    if out.exists() and not overwrite:
        print(f'[predict] {year}: {out} already written (use --overwrite)', flush=True)
        return

    # Keep canonical band order regardless of the order given on the command line,
    # so band 1 is always `mean` in every year's file.
    wanted = [t for t in TARGETS if t in targets]
    if not wanted:
        raise ValueError(f'no valid targets in {targets}; choose from {TARGETS}')

    # Fail before the expensive stack load if a model is missing.
    models = {t: load_model(t, model_dir, n_jobs) for t in wanted}

    stack = load_stack(year)
    print(f'[predict] {year}: stack {stack.shape}, loading into memory', flush=True)
    t0 = time.time()
    vals = np.asarray(stack.values, dtype='float32')
    print(f'[predict] {year}: loaded in {time.time() - t0:.0f}s', flush=True)

    ref = stack.isel(band=0).drop_vars('band', errors='ignore')

    bands = []
    for target in wanted:
        print(f'[predict] {year} / {target}: predicting', flush=True)
        t0 = time.time()
        arr = predict_stack(models[target], vals, block_rows)
        bands.append(arr)

        valid = arr != NODATA
        print(f'[predict] {year} / {target}: done in {time.time() - t0:.0f}s '
              f'| valid {valid.mean():.1%} | mean {arr[valid].mean():.2f} m '
              f'| p98 {np.percentile(arr[valid], 98):.2f} m', flush=True)

    del vals

    da = xr.DataArray(np.stack(bands), dims=('band', 'y', 'x'),
                      coords={'band': np.arange(1, len(bands) + 1),
                              'y': ref.y, 'x': ref.x})
    da = da.rio.write_crs(stack.rio.crs)
    da = da.rio.write_transform(stack.rio.transform())
    da = da.rio.write_nodata(NODATA)          # explicit -- see module docstring
    da.attrs['long_name'] = tuple(wanted)     # rioxarray -> per-band descriptions

    # Write via a temp name and rename: a killed run must not leave a truncated
    # file that a later --resume/skip-existing pass would trust (see CHECKPOINT.md).
    # Keep the .tif extension -- rasterio picks its driver from the extension.
    tmp = out.with_name(f'{out.stem}.partial.tif')
    da.rio.to_raster(tmp, dtype='float32', compress='DEFLATE')
    tmp.replace(out)
    print(f'[predict] {year}: wrote {out} '
          f'({len(bands)} bands: {", ".join(wanted)})', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--years', nargs='+', default=YEARS)
    p.add_argument('--targets', nargs='+', default=TARGETS)
    p.add_argument('--model-dir', default=None)
    p.add_argument('--block-rows', type=int, default=512)
    p.add_argument('--n-jobs', type=int, default=8,
                   help='threads for RF predict; do not raise above 8 (see CHECKPOINT.md)')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--log', default=None)
    args = p.parse_args()

    if args.log:
        import os
        import sys
        logf = open(args.log, 'a', buffering=1)
        os.dup2(logf.fileno(), sys.stdout.fileno())
        os.dup2(logf.fileno(), sys.stderr.fileno())

    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    for year in args.years:
        run_year(year, args.targets, model_dir, args.block_rows, args.overwrite,
                 args.n_jobs)


if __name__ == '__main__':
    main()
