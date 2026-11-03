"""GEDI L2A shots of one calendar year (--year, default 2025) over the six 2026 drone LiDAR sites, v002 and v003.

AOI: each in-scope site's CHM extent (EPSG:7851) + BUFFER_M, as lon/lat polygons.
Granules: CMR search GEDI02_A (bbox of the AOI, the calendar year).

    v002  local granules in G:\\python\\GEDI\\gedi_l2a_dampier\\ (h5py, no download);
          a granule missing locally is read over HTTPS like v003
    v003  HTTPS byte-range reads, capped at --rate MB/s in total over --workers. Session,
          rate cap and metadata cache are _init / throttle / MetaCache from
          G:\\python\\GDV\\GEDI\\gedi_l2a_download.py (imported, not forked).

Divergences from gedi_l2a_download.extract(), both forced by the v003 layout:
- extract() accepts only 1-D chunked datasets; v003 'rh' is chunked (3351, 4), so
  read_rows() here reads gzip chunks of any width.
- v003 has no top-level quality_flag. It has l2a_quality_flag_rel2 (the release-2
  definition, used here as quality_flag so v002 and v003 are filtered alike) and
  l2a_quality_flag_rel3 (kept as quality_flag_rel3).

Columns as gedi_l2a_download.extract() (shot_number, beam, beam_type, time, lat/lon
lowest mode, VARS, rh0..rh100 in m) plus granule, year, month, x/y (EPSG:7851),
utm_x/utm_y (EPSG:32751), in_mainland/in_pindan, site and qc_pass. All shots are
kept; qc_pass = GEDI/geoloc_extract.load_qc_shots (quality_flag == 1,
degrade_flag <= 0, sensitivity >= 0.95, the calendar year).

Check: v002 QC shots inside the unbuffered site boxes must equal the count from
chm_regression/outputs/gedi/shots_<year>_calendar.parquet in the same boxes.

Outputs: outputs/gedi/shots_<year>_sites_v002.parquet, ..._v003.parquet

Usage
-----
conda run -n general --no-capture-output python -u gedi_site_shots.py
conda run -n general --no-capture-output python -u gedi_site_shots.py --versions 003 --limit 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from shapely.ops import transform as shp_transform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'chm_regression'))
for _d in (r'G:\python\GEDI', r'G:\python\GDV\GEDI'):
    _p = _d if os.name == 'nt' else '/media/glen/CDU_Storage/' + _d[3:].replace('\\', '/')
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gedi_l2a_download as gl  # noqa: E402
import geoloc_boundaries as gb  # noqa: E402
import geoloc_extract as ge  # noqa: E402
from chm_class_histograms import EXCLUDE_SITES, ROOT_WIN, site_name, to_local  # noqa: E402

YEAR = 2025
BUFFER_M = 50.0
CHM_GLOB = '*/*_denoise_canopy_p98.tif'
LOCAL_V002_WIN = r'G:\python\GEDI\gedi_l2a_dampier'
OLD_SHOTS_TMPL = str(HERE.parent / 'chm_regression' / 'outputs' / 'gedi' / 'shots_{year}_calendar.parquet')
OUT_DIR = HERE / 'outputs' / 'gedi'
CRS_CHM = 'EPSG:7851'


def site_boxes(root: Path) -> dict[str, tuple[float, float, float, float]]:
    """Site -> CHM bounds (EPSG:7851) for the in-scope sites."""
    out = {}
    for f in sorted(root.glob(CHM_GLOB)):
        if site_name(f) in EXCLUDE_SITES:
            continue
        with rasterio.open(f) as r:
            out[site_name(f)] = tuple(r.bounds)
    assert len(out) == 6, f'expected 6 in-scope sites, found {sorted(out)}'
    return out


def aoi_lonlat(boxes) -> shapely.Geometry:
    to_ll = Transformer.from_crs(CRS_CHM, 'EPSG:4326', always_xy=True).transform
    polys = [shp_transform(to_ll, shapely.box(*b).buffer(BUFFER_M, join_style='mitre')) for b in boxes.values()]
    return shapely.union_all(polys)


def list_granules(version: str, bbox) -> list[dict]:
    import earthaccess
    res = earthaccess.search_data(short_name='GEDI02_A', version=version, bounding_box=tuple(bbox),
                                  temporal=(f'{YEAR}-01-01', f'{YEAR}-12-31'), count=-1)
    items = [{'id': g['meta']['native-id'],
              'href': [u for u in g.data_links(access='external') if u.endswith('.h5')][0]} for g in res]
    return sorted(items, key=lambda d: d['id'])


def read_local(path: str, poly) -> pd.DataFrame:
    """Same rows and columns as gedi_l2a_download.extract(), from a local granule."""
    import h5py
    minx, miny, maxx, maxy = poly.bounds
    parts = []
    with h5py.File(path, 'r') as h:
        for beam in sorted(k for k in h.keys() if k.startswith('BEAM')):
            g = h[beam]
            lat, lon = g['lat_lowestmode'][()], g['lon_lowestmode'][()]
            m = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
            m[m] = shapely.contains_xy(poly, lon[m], lat[m])
            if not m.any():
                continue
            j = np.where(m)[0]
            i0, i1 = int(j.min()), int(j.max()) + 1
            loc = m[i0:i1]
            desc = g.attrs.get('description', b'')
            desc = desc.decode('utf-8', 'ignore') if isinstance(desc, bytes) else str(desc)
            cols = {'beam': beam, 'beam_type': 'full' if 'full' in desc.lower() else 'coverage',
                    'time': gl.GEDI_EPOCH + pd.to_timedelta(g['delta_time'][i0:i1][loc], unit='s'),
                    'lat': lat[i0:i1][loc], 'lon': lon[i0:i1][loc]}
            for name, p in gl.VARS.items():
                cols[name] = g[p][i0:i1][loc]
            rh = g['rh'][i0:i1][loc].astype('float32')
            parts.append(pd.concat([pd.DataFrame(cols), pd.DataFrame(rh, columns=gl.RH_COLS)], axis=1))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


VARS_V003 = {**gl.VARS, 'quality_flag': 'l2a_quality_flag_rel2', 'quality_flag_rel3': 'l2a_quality_flag_rel3'}


def read_rows(get, d, i0: int, i1: int, pool) -> np.ndarray:
    """Rows i0:i1 of h5py dataset d: contiguous, or gzip-chunked (any chunk width)."""
    if d.chunks is None:
        row = d.dtype.itemsize * int(np.prod(d.shape[1:]))
        off = d.id.get_offset()
        return np.frombuffer(get(off + i0 * row, off + i1 * row), d.dtype).reshape((-1,) + d.shape[1:])
    if set(d._filters) - {'gzip'}:
        raise ValueError(f'{d.name}: unsupported filters {d._filters}')
    cr = d.chunks[0]
    need = [c for c in (d.id.get_chunk_info(k) for k in range(d.id.get_num_chunks()))
            if c.chunk_offset[0] < i1 and c.chunk_offset[0] + cr > i0]
    if any(c.filter_mask for c in need):
        raise ValueError(f'{d.name}: filter mask set')
    out = np.empty((i1 - i0,) + d.shape[1:], d.dtype)
    blobs = pool.map(lambda c: zlib.decompress(get(c.byte_offset, c.byte_offset + c.size)), need)
    for c, b in zip(need, blobs):
        a = np.frombuffer(b, d.dtype).reshape(d.chunks)
        r0 = c.chunk_offset[0]
        rs, re = max(i0, r0), min(i1, r0 + cr)
        if d.ndim == 1:
            out[rs - i0:re - i0] = a[rs - r0:re - r0]
        else:
            k0 = c.chunk_offset[1]
            ke = min(k0 + d.chunks[1], d.shape[1])
            out[rs - i0:re - i0, k0:ke] = a[rs - r0:re - r0, :ke - k0]
    return out


def extract_v003(href: str) -> pd.DataFrame:
    """AOI shots of one remote granule; same rows and columns as read_local()."""
    import h5py
    import requests
    from concurrent.futures import ThreadPoolExecutor
    poly = gl._W['poly']
    minx, miny, maxx, maxy = gl._W['bbox']
    r = gl._W['session'].get(href, headers={'Range': 'bytes=0-0'}, timeout=120, stream=True)
    r.raise_for_status()
    url, size = r.url, int(r.headers['Content-Range'].split('/')[1])  # signed URL, resolved once
    r.close()
    http = requests.Session()

    def get(a, b):
        for k in range(4):
            try:
                resp = http.get(url, headers={'Range': f'bytes={a}-{b - 1}'}, timeout=(60, 180))
                resp.raise_for_status()
                if len(resp.content) != b - a:
                    raise IOError(f'short read {len(resp.content)} != {b - a}')
                gl.throttle(len(resp.content))
                return resp.content
            except Exception:
                if k == 3:
                    raise
                time.sleep(5 * (k + 1))

    parts = []
    with h5py.File(gl.MetaCache(get, size), 'r') as h, ThreadPoolExecutor(16) as pool:
        for beam in sorted(k for k in h.keys() if k.startswith('BEAM')):
            g = h[beam]
            n = g['lat_lowestmode'].shape[0]
            lat = read_rows(get, g['lat_lowestmode'], 0, n, pool)
            w = np.where((lat >= miny) & (lat <= maxy))[0]
            if not w.size:
                continue
            a0, a1 = int(w.min()), int(w.max()) + 1
            lon = read_rows(get, g['lon_lowestmode'], a0, a1, pool)
            m = (lon >= minx) & (lon <= maxx) & (lat[a0:a1] >= miny) & (lat[a0:a1] <= maxy)
            m[m] = shapely.contains_xy(poly, lon[m], lat[a0:a1][m])
            if not m.any():
                continue
            j = np.where(m)[0]
            i0, i1 = a0 + int(j.min()), a0 + int(j.max()) + 1
            loc = m[i0 - a0:i1 - a0]
            desc = g.attrs.get('description', b'')
            desc = desc.decode('utf-8', 'ignore') if isinstance(desc, bytes) else str(desc)
            cols = {'beam': beam, 'beam_type': 'full' if 'full' in desc.lower() else 'coverage',
                    'time': gl.GEDI_EPOCH + pd.to_timedelta(read_rows(get, g['delta_time'], i0, i1, pool)[loc], unit='s'),
                    'lat': lat[i0:i1][loc], 'lon': lon[i0 - a0:i1 - a0][loc]}
            for name, p in VARS_V003.items():
                cols[name] = read_rows(get, g[p], i0, i1, pool)[loc]
            rh = read_rows(get, g['rh'], i0, i1, pool)[loc].astype('float32')
            parts.append(pd.concat([pd.DataFrame(cols), pd.DataFrame(rh, columns=gl.RH_COLS)], axis=1))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _worker_v003(it: dict):
    for attempt in range(1, 4):
        try:
            return it, extract_v003(it['href'])
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                return it, f'ERROR: {type(exc).__name__}: {exc}'
            time.sleep(10 * attempt)


def fetch_remote(items: list[dict], poly, rate: float, workers: int, version: str) -> list[tuple[dict, pd.DataFrame]]:
    out = []
    worker = _worker_v003 if version == '003' else gl._worker
    with ProcessPoolExecutor(max_workers=workers, initializer=gl._init,
                             initargs=(poly.wkb, rate * 1e6 / workers)) as ex:
        t0 = time.time()
        for k, (it, res) in enumerate(ex.map(worker, items), 1):
            if isinstance(res, str):
                raise RuntimeError(f'{it["id"]}: {res}')
            out.append((it, res))
            print(f'[shots]   remote {k}/{len(items)} {it["id"]}: {len(res)} shots, '
                  f'{(time.time() - t0) / 60:.1f} min', flush=True)
    return out


def add_columns(df: pd.DataFrame, boxes) -> pd.DataFrame:
    df = df[df['time'].dt.year == YEAR].reset_index(drop=True)
    df['year'], df['month'] = df['time'].dt.year, df['time'].dt.month
    x, y = Transformer.from_crs('EPSG:4326', CRS_CHM, always_xy=True).transform(df['lon'], df['lat'])
    ux, uy = Transformer.from_crs('EPSG:4326', 'EPSG:32751', always_xy=True).transform(df['lon'], df['lat'])
    df['x'], df['y'], df['utm_x'], df['utm_y'] = x, y, ux, uy
    df['site'] = None
    for s, (x0, y0, x1, y1) in boxes.items():
        inb = (df.x >= x0 - BUFFER_M) & (df.x <= x1 + BUFFER_M) & (df.y >= y0 - BUFFER_M) & (df.y <= y1 + BUFFER_M)
        df.loc[inb, 'site'] = s
    return gb.flag_boundaries(df)


def in_boxes(x, y, boxes) -> np.ndarray:
    """Unbuffered site boxes, applied to 32751 coordinates (as the planning count was)."""
    return np.any([(x >= x0) & (x < x1) & (y >= y0) & (y < y1) for x0, y0, x1, y1 in boxes.values()], axis=0)


def run_version(version: str, boxes, poly, args) -> Path:
    items = list_granules(version, poly.bounds)
    if args.limit:
        items = items[:args.limit]
    local_dir = Path(to_local(LOCAL_V002_WIN))
    local = [it for it in items if version == '002' and (local_dir / f'{it["id"]}.h5').exists()]
    remote = [it for it in items if it not in local]
    print(f'[shots] v{version}: {len(items)} granules over the AOI in {YEAR} '
          f'({len(local)} local, {len(remote)} remote at <= {args.rate} MB/s)', flush=True)

    frames = []
    for k, it in enumerate(local, 1):
        d = read_local(str(local_dir / f'{it["id"]}.h5'), poly)
        frames.append(d.assign(granule=it['id']))
        print(f'[shots]   local {k}/{len(local)} {it["id"]}: {len(d)} shots', flush=True)
    if remote:
        frames += [d.assign(granule=it['id']) for it, d in fetch_remote(remote, poly, args.rate, args.workers, version)]
    df = add_columns(pd.concat([f for f in frames if len(f)], ignore_index=True), boxes)
    assert len(df) > 0, f'v{version}: no shots in the AOI'
    assert df['shot_number'].is_unique, f'v{version}: duplicate shot_number'

    out = OUT_DIR / f'shots_{YEAR}_sites_v{version}.parquet'
    partial = out.with_name(out.stem + '.partial.parquet')
    df.to_parquet(partial, index=False)
    df['qc_pass'] = df.index.isin(ge.load_qc_shots(str(partial), YEAR, time_window='calendar').index)
    df.to_parquet(partial, index=False)
    partial.replace(out)
    print(f'[shots] v{version}: {len(df):,} shots, {int(df.qc_pass.sum()):,} pass QC; '
          f'per site (QC): {df[df.qc_pass].groupby("site").size().to_dict()}')
    print(f'[shots] wrote {out.resolve()}', flush=True)

    if version == '002' and not args.limit:
        old = pd.read_parquet(to_local(OLD_SHOTS_TMPL.format(year=YEAR)))
        n_old = int(in_boxes(old.utm_x, old.utm_y, boxes).sum())
        q = df[df.qc_pass]
        n_new = int(in_boxes(q.utm_x, q.utm_y, boxes).sum())
        assert n_new == n_old, f'v002 QC shots in the site boxes: {n_new} here, {n_old} in {OLD_SHOTS_TMPL.format(year=YEAR)}'
        print(f'[shots] v002 check: {n_new} QC shots in the site boxes = {n_old} in shots_{YEAR}_calendar.parquet')
    return out


def main():
    global YEAR
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--year', type=int, default=YEAR)
    p.add_argument('--versions', nargs='+', default=['002', '003'])
    p.add_argument('--limit', type=int, help='first N granules per version (smoke test)')
    p.add_argument('--rate', type=float, default=4.0, help='total MB/s cap (1 MB = 1e6 B)')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    YEAR = args.year

    boxes = site_boxes(Path(to_local(ROOT_WIN)))
    poly = aoi_lonlat(boxes)
    print(f'[shots] AOI: {len(boxes)} sites + {BUFFER_M:g} m, lon/lat bounds {np.round(poly.bounds, 4).tolist()}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for v in args.versions:
        run_version(v, boxes, poly, args)


if __name__ == '__main__':
    main()
