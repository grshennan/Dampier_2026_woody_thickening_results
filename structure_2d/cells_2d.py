"""Stage 1 (PLAN.md): one row per 20 m cell with LiDAR structure, for 5 Dampier 2026 sites + 13 Yawuru 2025 areas.

Structure grids: lidar_2026/outputs/structure_sar/struct_nisar20m_<site>.tif (laspy, Dampier) and
outputs/yawuru/struct_nisar20m_<area>.tif (LAStools, yawuru_struct.py; method checked against 9 Mile).
Cell rule as structure_vs_sar.load_grid: fill >= FILL_MIN (0.9) and cover137 finite.

The 4classTF label of each cell (mean of the 50 cm CHM p98 over the cell; cuts 1.22 / 2.35 / 5.381;
0 Non-forest, 1 Low forest, 2 Forest, 3 Tall forest) is the baseline for Stage 3:
- Dampier: nisar_2026/outputs/labels_2026.parquet, joined by (row, col) on the GCOV grid.
- Yawuru: the CHMs of chm_targets.LIDAR_PATHS_WIN aggregated onto the same grid with
  chm_class_histograms.mean_raster (as labels_2026.py), count >= LOW_COUNT_THRESHOLD.
Cells without a label keep cls4 = -1.
"No canopy" cells (canopy False): n_can < MIN_CAN (p98_can NaN) or cover137 < COVER_MIN (1 %).

Run from this directory:
    conda run -n general --no-capture-output python -u cells_2d.py
Output: outputs/cells_2d.parquet
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in ('lidar_2026', 'chm_regression', 'nisar_2026'):
    sys.path.insert(0, str(ROOT / p))
from chm_class_histograms import load_chm, mean_raster  # noqa: E402
from chm_targets import AREA_NAMES, LIDAR_PATHS_WIN, LOW_COUNT_THRESHOLD  # noqa: E402
from labels_2026 import GRID_REF_WIN, SITES  # noqa: E402
from structure_grids import EXCLUDE_SAR, OUT_DIR as DAMPIER_DIR  # noqa: E402
from structure_vs_sar import FILL_MIN, read_window  # noqa: E402

YAWURU_DIR = HERE / 'outputs' / 'yawuru'
OUT = HERE / 'outputs' / 'cells_2d.parquet'
LABELS = ROOT / 'nisar_2026' / 'outputs' / 'labels_2026.parquet'
STACK_OLD = Path(r'E:\nisar_2026\feature_stack_2026.tif')  # the stack of REPORT_structure_sar.md (rho check)
CUTS4 = [1.22, 2.35, 5.381]
COVER_MIN = 0.01  # below 1 % cover, p98_can describes a few shrubs (dense Yawuru clouds: 20 returns can be < 0.1 %)
COLS = ['cover137', 'p98_can', 'p90_can', 'mean_can', 'c_p98_can', 'p98_all', 'mean_all',
        'cover_h02', 'cover_h1', 'cover_h2', 'cover_h5', 'n_first', 'n_can', 'fill']


def grid_cells(path, site, source, ref_tf):
    with rasterio.open(path) as s:
        a = s.read(masked=True).filled(np.nan).astype('float64')
        tf, desc = s.transform, list(s.descriptions)
    c_off, r_off = ~ref_tf * (tf.c, tf.f)
    assert tf.a == ref_tf.a and tf.e == ref_tf.e and abs(c_off - round(c_off)) < 1e-6 and abs(r_off - round(r_off)) < 1e-6, \
        f'{site}: not a whole-pixel window of the GCOV grid ({tf})'
    keep = (a[desc.index('fill')] >= FILL_MIN) & np.isfinite(a[desc.index('cover137')])
    rr, cc = np.nonzero(keep)
    d = pd.DataFrame({b: a[desc.index(b)][rr, cc] for b in COLS})
    d.insert(0, 'site', site)
    d.insert(1, 'source', source)
    d.insert(2, 'row', rr + round(r_off))
    d.insert(3, 'col', cc + round(c_off))
    d['x'], d['y'] = ref_tf.c + (d['col'] + 0.5) * ref_tf.a, ref_tf.f + (d['row'] + 0.5) * ref_tf.e
    d['lrow'], d['lcol'] = rr, cc  # in the site window (lidar_2026 fire_cells.parquet uses these)
    return d, (tf, (a.shape[1], a.shape[2]))


def yawuru_labels(ref):
    parts = []
    for area, chm in zip(AREA_NAMES, LIDAR_PATHS_WIN):
        out_dir = HERE / 'outputs' / 'chm_mean' / area  # mean_raster names the file by the CHM's folder
        out_dir.mkdir(parents=True, exist_ok=True)
        tif = mean_raster(load_chm(Path(chm)), Path(chm), ref, out_dir)
        with rasterio.open(tif) as s:
            a = s.read().astype('float64')
            tf = s.transform
        a[a == -9999.0] = np.nan
        c_off, r_off = ~ref.rio.transform() * (tf.c, tf.f)
        rr, cc = np.nonzero(np.nan_to_num(a[4]) >= LOW_COUNT_THRESHOLD)
        parts.append(pd.DataFrame({'site': area, 'row': rr + round(r_off), 'col': cc + round(c_off),
                                   'mean_chm': a[0][rr, cc], 'count': a[4][rr, cc]}))
    return pd.concat(parts, ignore_index=True)


def main(test=False):
    """test=True (--test): skip missing Yawuru grids; write outputs/test/cells_2d_test.parquet."""
    ref = rxr.open_rasterio(GRID_REF_WIN, masked=True)
    ref_tf = ref.rio.transform()
    assert (ref_tf.c, ref_tf.f, ref_tf.a) == (393160.0, 8197760.0, 20.0), f'GCOV grid {ref_tf}'
    with rasterio.open(STACK_OLD) as s:
        assert s.transform == ref_tf, 'stack grid != GCOV grid'

    dam, win = [], {}
    for site in SITES:
        assert site not in EXCLUDE_SAR
        d, win[site] = grid_cells(DAMPIER_DIR / f'struct_nisar20m_{site}.tif', site, 'Dampier 2026', ref_tf)
        dam.append(d)
    dam = pd.concat(dam, ignore_index=True)
    n_can_f, n_c = int(dam.p98_can.notna().sum()), int(dam.c_p98_can.notna().sum())
    print(f'Dampier: {len(dam):,} cells; p98_can finite {n_can_f:,}, c_p98_can finite {n_c:,}', flush=True)
    assert (n_can_f, n_c) == (33021, 33119), 'Dampier cell counts differ from REPORT_structure_sar.md'

    # rho check: c_p98_can against 168 A HV of the stack used in REPORT_structure_sar.md (0.76)
    names = json.loads(STACK_OLD.with_suffix('.json').read_text())['features']
    hv = []
    for s in SITES:  # dam is in SITES order
        tf, shape = win[s]
        c0, r0 = (round(v) for v in ~ref_tf * (tf.c, tf.f))
        band = read_window(STACK_OLD, tf, shape, bands=[names.index('168_A_HV_dB') + 1])[0]
        g = dam[dam.site == s]
        hv.append(band[g.row - r0, g.col - c0])
    hv = np.concatenate(hv)
    ok = np.isfinite(hv) & dam.c_p98_can.notna().to_numpy()
    rho = spearmanr(dam.c_p98_can[ok], hv[ok]).statistic
    print(f'check: Spearman rho c_p98_can vs 168 A HV = {rho:.3f} (report 0.76), n {ok.sum():,}', flush=True)
    assert abs(rho - 0.76) < 0.006, 'rho check failed'

    lab = pd.read_parquet(LABELS)
    dam = dam.merge(lab[['row', 'col', 'mean']].rename(columns={'mean': 'mean_chm'}), on=['row', 'col'], how='left')

    yaw = []
    for area in AREA_NAMES:
        p = YAWURU_DIR / f'struct_nisar20m_{area}.tif'
        if test and not p.exists():
            print(f'TEST: {area} missing, skipped', flush=True)
            continue
        assert p.exists(), f'{p}: run yawuru_struct.py grids first'
        yaw.append(grid_cells(p, area, 'Yawuru 2025', ref_tf)[0])
    yaw = pd.concat(yaw, ignore_index=True)
    dup = yaw.duplicated(['row', 'col'], keep=False)
    print(f'Yawuru: {len(yaw):,} cells; {int(dup.sum())} rows in two areas '
          f'({", ".join(sorted(yaw.site[dup].unique()))}); keep the row with more first returns', flush=True)
    yaw = yaw.sort_values('n_first', ascending=False).drop_duplicates(['row', 'col']).sort_index()
    ylab = yawuru_labels(ref).sort_values('count', ascending=False).drop_duplicates(['row', 'col'])
    yaw = yaw.merge(ylab[['row', 'col', 'mean_chm']], on=['row', 'col'], how='left')

    cells = pd.concat([dam, yaw], ignore_index=True)
    assert not cells.duplicated(['row', 'col']).any(), 'Dampier and Yawuru cells overlap'
    assert test or set(cells.site) == set(SITES) | set(AREA_NAMES)
    cells['cls4'] = np.where(cells.mean_chm.notna(), np.digitize(cells.mean_chm, CUTS4), -1).astype('int8')
    cells['canopy'] = cells.p98_can.notna() & (cells.cover137 >= COVER_MIN)  # False -> fixed class "no canopy"
    out = OUT.parent / 'test' / 'cells_2d_test.parquet' if test else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out, index=False)
    t = cells.groupby(['source', 'site']).agg(n=('cover137', 'size'), no_canopy=('canopy', lambda v: int((~v).sum())),
                                              labelled=('cls4', lambda v: int((v >= 0).sum())),
                                              cover137=('cover137', 'median'), p98_can=('p98_can', 'median'))
    print(t.round(3).to_string(), flush=True)
    print(f'total {len(cells):,} cells, {int((~cells.canopy).sum()):,} no canopy, '
          f'{int((cells.cls4 < 0).sum()):,} without a 4classTF label; ' + f'wrote {out.resolve()}' + (' (TEST: YJ25_3 missing)' if test else ''), flush=True)


if __name__ == '__main__':
    main(test='--test' in sys.argv)
