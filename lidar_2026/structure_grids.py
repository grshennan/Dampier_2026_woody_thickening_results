"""Canopy structure statistics from the point clouds on the two NISAR grids, 2026 Dampier LiDAR.

For each non-MVT site except Bunda Bunda (EXCLUDE_SAR), the cloud <site>_ground_gndthin_height_denoise.las (z = height above
ground) is read once, and first returns (not withheld, not noise class 7 / 18; the filter of
gap_fraction.py) are binned onto the NISAR grid, so that no SAR is resampled:

    nisar20m   NISAR GCOV grid, EPSG:32751, 20 m (the feature_stack_2026 grid). EPSG:7851 point
               coordinates are used without reprojection (project convention, datum offset < 2 m).
               ⚠ CRS 2026-11-03: withdrawn (user); reprojection is required. G:/python/crs_audit/CHECKPOINT.md.
    (albers30m, the EPSG:3577 30 m grid, was removed 2026-10-22: all NISAR data are native 20 m UTM.)

No CHM raster is used. Per cell (z < 0 is clipped to 0, so it counts as gap):
    cover137          first returns with z >= 1.37 / all first returns (= 1 - gap)
    *_all             p98, p90, mean of all first returns (mean_all = mean first-return height)
    *_can             p98, p90, mean of canopy first returns (z >= 1.37); nodata if n_can < 20
    c_<stat>          cover137 x <stat>; c_*_can = 0 where n_can = 0
    *_h02, *_h1,      cover_h<h>, *_can_h<h>, c_*_h<h>: the same with the cover cutoff at
    *_h2, *_h5        0.2 m / 1 m / 2 m / 5 m (no 0 m: z < 0 is clipped, so cover would be 1)
    n_first, n_can    return counts
    fill              share of the cell's 1 m sub-cells that hold a first return
Percentiles come from a per-cell histogram (0.05 m bins, extra edge at 1.37 m), linear inside a
bin; means are exact sums. A few cells per site are checked against np.percentile on raw z.

Outputs: outputs/structure_sar/struct_nisar20m_<site>.tif (56 bands, nodata -9999).

Usage (<= 2 cores: set OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1)
-----
conda run -n general --no-capture-output python -u structure_grids.py [--sites "9 Mile"]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.transform import Affine

from chm_class_histograms import EXCLUDE_SITES, HERE, ROOT_WIN, to_local
from gap_fraction import CHUNK, NOISE_CLASSES

H_COVER = 1.37
CUTS = {'': H_COVER, '_h2': 2.0, '_h5': 5.0, '_h02': 0.2, '_h1': 1.0}  # band suffix -> cover cutoff (m)
EXCLUDE_SAR = {'Bunda Bunda'}  # floats on a spring: ground moves, wetter, other species (user, 2026-10-21)
EDGES = np.unique(np.r_[np.round(np.arange(0, 50.0001, 0.05), 2), H_COVER])
NB = len(EDGES) - 1
I_CUT = {k: int(np.searchsorted(EDGES, h)) for k, h in CUTS.items()}  # first canopy bin per cutoff
assert all(EDGES[I_CUT[k]] == h for k, h in CUTS.items())
MIN_CAN = 20
SUB = 1.0  # occupancy sub-cell (m)
NODATA = -9999.0
CUT_BANDS = ['p98_can', 'p90_can', 'mean_can', 'c_p98_all', 'c_p90_all', 'c_mean_all',
             'c_p98_can', 'c_p90_can', 'c_mean_can']
BANDS = (['cover137', 'p98_all', 'p90_all', 'mean_all', 'p98_can', 'p90_can', 'mean_can',
          'c_p98_all', 'c_p90_all', 'c_mean_all', 'c_p98_can', 'c_p90_can', 'c_mean_can',
          'n_first', 'n_can', 'fill']
         + [b + k for k in CUTS if k for b in ['cover'] + CUT_BANDS])
GRID_REFS = {
    'nisar20m': r'G:\python\nisar\geotiff\NISAR_GCOV_20260928T095935_t147_D_freqA_HHHH.tif',
}
OUT_DIR = HERE / 'outputs' / 'structure_sar'
FIG_DIR = OUT_DIR / 'figures'  # checks/, pooled/, by_site/
N_CHECK = 3  # cells per site checked against np.percentile
N_CHECK_EXTRA = 1000  # canopy returns a check at another cutoff needs (a sparse tail can miss a 0.05 m bin)


class Grid:
    """A whole-pixel window of a reference grid around the cloud, plus its accumulators."""

    def __init__(self, name, ref_path, bounds_xy, to_crs):
        self.name, self.to_crs = name, to_crs
        with rasterio.open(to_local(ref_path)) as s:
            tf, self.crs = s.transform, s.crs
        xs, ys = bounds_xy
        if to_crs is not None:
            xs, ys = to_crs.transform(xs, ys)
        c0, r0 = np.floor(~tf * (min(xs), max(ys))).astype(int)
        c1, r1 = np.ceil(~tf * (max(xs), min(ys))).astype(int)
        self.tf = tf * Affine.translation(c0, r0)
        self.shape = (r1 - r0, c1 - c0)
        self.k = round(tf.a / SUB)
        n = self.shape[0] * self.shape[1]
        self.hist = np.zeros(n * NB, 'int64')
        self.sum_all = np.zeros(n)
        self.sum_can = {k: np.zeros(n) for k in CUTS}
        self.occ = np.zeros((self.shape[0] * self.k, self.shape[1] * self.k), bool)
        self.n_out = 0
        mid = n // 2 + self.shape[1] // 2  # cells near the window centre for the percentile check
        self.check_cells = np.arange(mid, mid + N_CHECK)
        self.check_z = [[] for _ in range(N_CHECK)]

    def add(self, x, y, z, b):
        if self.to_crs is not None:
            x, y = self.to_crs.transform(x, y)
        cf = (x - self.tf.c) / self.tf.a
        rf = (y - self.tf.f) / self.tf.e
        cols, rows = np.floor(cf).astype('int64'), np.floor(rf).astype('int64')
        inside = (cols >= 0) & (cols < self.shape[1]) & (rows >= 0) & (rows < self.shape[0])
        self.n_out += int((~inside).sum())
        idx = rows[inside] * self.shape[1] + cols[inside]
        z, b = z[inside], b[inside]
        n = len(self.sum_all)
        self.hist += np.bincount(idx * NB + b, minlength=n * NB)
        self.sum_all += np.bincount(idx, weights=z, minlength=n)
        for k, i in I_CUT.items():
            self.sum_can[k] += np.bincount(idx, weights=np.where(b >= i, z, 0), minlength=n)
        self.occ[np.floor(rf[inside] * self.k).astype('int64'), np.floor(cf[inside] * self.k).astype('int64')] = True
        for i, c in enumerate(self.check_cells):
            self.check_z[i].append(z[idx == c])

    def bands(self) -> np.ndarray:
        h = self.hist.reshape(-1, NB)
        n_first = h.sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = {'mean_all': self.sum_all / n_first}
            for q in (98, 90):
                out[f'p{q}_all'] = hist_quantile(h, EDGES, q / 100)
            for k, i in I_CUT.items():
                n_can = h[:, i:].sum(1)
                cover = n_can / n_first
                out['cover137' if k == '' else 'cover' + k] = cover
                out['mean_can' + k] = self.sum_can[k] / n_can
                for q in (98, 90):
                    out[f'p{q}_can{k}'] = hist_quantile(h[:, i:], EDGES[i:], q / 100)
                for s in ('p98', 'p90', 'mean'):
                    out[f'{s}_can{k}'] = np.where(n_can >= MIN_CAN, out[f'{s}_can{k}'], np.nan)
                    out[f'c_{s}_all{k}'] = cover * out[f'{s}_all']
                    out[f'c_{s}_can{k}'] = np.where(n_can == 0, 0.0, cover * out[f'{s}_can{k}'])
                if k == '':
                    out['n_can'] = n_can.astype(float)
        sk = self.k
        out['fill'] = self.occ.reshape(self.shape[0], sk, self.shape[1], sk).mean((1, 3)).ravel()
        out['n_first'] = n_first.astype(float)
        a = np.stack([out[b] for b in BANDS])
        a[:, n_first == 0] = np.nan
        a[BANDS.index('fill'), n_first == 0] = 0
        return a.reshape(len(BANDS), *self.shape)

    def check(self, a: np.ndarray) -> float:
        """Largest |histogram - np.percentile| over the check cells (p98/p90, _all/_can at every cutoff)."""
        err = 0.0
        for c, zs in zip(self.check_cells, self.check_z):
            z = np.concatenate(zs)
            r, col = divmod(int(c), self.shape[1])
            for q in (98, 90):
                if len(z):
                    err = max(err, abs(a[BANDS.index(f'p{q}_all'), r, col] - np.percentile(z, q)))
                for k, hc in CUTS.items():
                    zc = z[z >= hc]
                    if len(zc) >= (MIN_CAN if k == '' else N_CHECK_EXTRA):
                        err = max(err, abs(a[BANDS.index(f'p{q}_can{k}'), r, col] - np.percentile(zc, q)))
        return err


def hist_quantile(h: np.ndarray, edges: np.ndarray, q: float) -> np.ndarray:
    """Quantile q per row of histogram h (bins edges[i]..edges[i+1]), linear inside a bin."""
    cum = np.cumsum(h, 1)
    n = cum[:, -1]
    t = q * n
    i = np.minimum((cum < t[:, None]).sum(1), h.shape[1] - 1)
    rows = np.arange(len(h))
    before = np.where(i > 0, cum[rows, i - 1], 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.clip((t - before) / h[rows, i], 0, 1)
    v = edges[i] + frac * (edges[i + 1] - edges[i])
    return np.where(n > 0, v, np.nan)


def write(a: np.ndarray, g: Grid, out: Path) -> None:
    partial = out.with_name(out.stem + '.partial.tif')
    prof = dict(driver='GTiff', width=g.shape[1], height=g.shape[0], count=len(BANDS), dtype='float32',
                crs=g.crs, transform=g.tf, nodata=NODATA, compress='deflate')
    with rasterio.open(partial, 'w', **prof) as d:
        d.write(np.where(np.isfinite(a), a, NODATA).astype('float32'))
        d.descriptions = tuple(BANDS)
    partial.replace(out)


def process_site(las_path: Path, site: str, laslib_first: bool = False) -> None:
    """laslib_first: first return = return_number <= 1 (LASlib is_first(), as LAStools -keep_first), for clouds
    without return numbers (Yawuru Apr 2025: all 0). Default: return_number == 1."""
    outs = {name: OUT_DIR / f'struct_{name}_{site}.tif' for name in GRID_REFS}
    def current(o: Path) -> bool:
        if not (o.exists() and o.stat().st_mtime > las_path.stat().st_mtime):
            return False
        with rasterio.open(o) as s:
            return list(s.descriptions) == BANDS

    if all(current(o) for o in outs.values()):
        print(f'[struct] {site}: outputs are current, reused', flush=True)
        return
    t0 = time.time()
    with laspy.open(las_path) as r:
        hd = r.header
        bx = np.r_[np.linspace(hd.mins[0], hd.maxs[0], 50), np.full(50, hd.maxs[0]),
                   np.linspace(hd.maxs[0], hd.mins[0], 50), np.full(50, hd.mins[0])]
        by = np.r_[np.full(50, hd.mins[1]), np.linspace(hd.mins[1], hd.maxs[1], 50),
                   np.full(50, hd.maxs[1]), np.linspace(hd.maxs[1], hd.mins[1], 50)]
        grids = [Grid(name, ref, (bx, by), None)
                 for name, ref in GRID_REFS.items()]
        for g in grids:
            print(f'[struct] {site}: {g.name} window {g.shape[1]} x {g.shape[0]} cells', flush=True)
        total, n_read, n_first = hd.point_count, 0, 0
        for pts in r.chunk_iterator(CHUNK):
            n_read += len(pts)
            rn = np.asarray(pts.return_number)
            keep = (((rn <= 1) if laslib_first else (rn == 1)) & ~np.asarray(pts.withheld, bool)
                    & ~np.isin(np.asarray(pts.classification), NOISE_CLASSES))
            x, y = np.asarray(pts.x)[keep], np.asarray(pts.y)[keep]
            z = np.clip(np.asarray(pts.z)[keep], 0, EDGES[-1] - 1e-6)
            b = np.searchsorted(EDGES, z, side='right') - 1
            n_first += int(keep.sum())
            for g in grids:
                g.add(x, y, z, b)
            print(f'[struct]   {n_read / 1e6:,.0f} / {total / 1e6:,.0f} M points, '
                  f'{(time.time() - t0) / 60:.1f} min', flush=True)
    assert n_read == total, f'{las_path}: read {n_read} of {total} points'
    print(f'[struct] {site}: {n_first:,} first returns kept', flush=True)
    for g in grids:
        a = g.bands()
        n_in = int(np.nansum(a[BANDS.index('n_first')]))
        assert n_in + g.n_out == n_first, f'{site} {g.name}: first returns lost in the grid mapping'
        err = g.check(a)
        assert err <= 0.05 + 1e-6, f'{site} {g.name}: histogram percentile error {err:.3f} m'
        write(a, g, outs[g.name])
        print(f'[struct] {site} {g.name}: {n_in:,} in grid, {100 * g.n_out / n_first:.3f} % outside; '
              f'percentile check max error {err:.4f} m; wrote {outs[g.name].resolve()}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sites', nargs='*', help='site folder names (default: all non-MVT sites)')
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(to_local(ROOT_WIN))
    las = sorted(f for f in root.glob('*/*_ground_gndthin_height_denoise.las')
                 if f.parent.name not in EXCLUDE_SITES | EXCLUDE_SAR and (not args.sites or f.parent.name in args.sites))
    print(f'[struct] {len(las)} sites: {", ".join(f.parent.name for f in las)}', flush=True)
    for f in las:
        process_site(f, f.parent.name)


if __name__ == '__main__':
    main()
