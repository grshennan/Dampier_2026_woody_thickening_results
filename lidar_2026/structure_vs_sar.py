"""Canopy structure statistics (structure_grids.py) against NISAR SAR, 2026 Dampier LiDAR.

Cells: struct_nisar20m_<site>.tif with fill >= 0.9 (edge cells dropped); Bunda Bunda is excluded
(EXCLUDE_SAR, from 2026-10-21). SAR values are read at the cell row / col, all on the same 20 m
EPSG:32751 grid, with no resampling (from 2026-10-22; the 30 m EPSG:3577 series is deleted):
    stack      E:\\nisar_2026\\feature_stack_2026.tif, the 26 NISAR bands (GCOV gamma0 + GSLC
               decomposition, Sep-Oct, Refined Lee 5 x 5)
    dcmp       E:\\NISAR\\Dampier\\GSLC_dcmp_20m_rflee5\\dcmp_<date>_<cycle>_<trk>.tif (single-date
               images, sigma0, Jun-Oct, Refined Lee 5 x 5)
    coh        E:\\NISAR\\Dampier\\GSLC_coh_20m\\coh_<d1>_<d2>_<trk>.tif (pairs, unfiltered)
    dcmp_mean  mean over all dates of each dcmp band per track; dcmp_sepoct: mean over the stack's two
               Sep-Oct dates (SEPOCT). Power bands (*_db) are averaged in linear power, then put back
               in dB (2026-10-22: Ps = 0 is floored at -60 dB, and a mean in dB of -60 and -25 is a
               false -42.5); the other bands are averaged as they are.
For each structure statistic x SAR feature: n, Pearson r, linear R2, Spearman rho, rho2, pooled
and per site. Descriptive only (cells are spatially autocorrelated).

Outputs (outputs/structure_sar/; PNGs in figures/checks/ and figures/pooled/):
    check_backscatter_vs_height.png   HV / HH against p98_all and mean_all (sanity check)
    structure_sar_stats.csv           long table, all metrics
    r2_by_site.csv                    rho2 per site, main statistics x best stack features
    r2_matrix.png                     R2 and rho2, statistics x best 20 m stack features
    scatter_main.png                  cover x p98 against the best features, beside p98 and cover
    season.png                        rho2 by date, HV / HH / windowed HH coherence, per track (single dates)

Usage
-----
conda run -n general --no-capture-output python -u structure_vs_sar.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.stats import rankdata

from chm_class_histograms import HERE, to_local
from structure_grids import BANDS, CUTS, EXCLUDE_SAR, FIG_DIR, OUT_DIR
import viz_style

import matplotlib.pyplot as plt  # noqa: E402  (after viz_style sets Agg)
import matplotlib.dates as mdates  # noqa: E402

FILL_MIN = 0.9
MIN_N = 30
STACK = Path(r'E:\nisar_2026\feature_stack_2026.tif')
NISAR_DIR = Path(r'E:\NISAR\Dampier')
DCMP_DIR, COH_DIR = NISAR_DIR / 'GSLC_dcmp_20m_rflee5', NISAR_DIR / 'GSLC_coh_20m'
DCMP_RE = re.compile(r'dcmp_(\d{8})_\d{3}_(\d{3}_[AD])\.tif$')
COH_RE = re.compile(r'coh_(\d{8})_(\d{8})_(\d{3}_[AD])\.tif$')
COH_BANDS = ['coh_HH', 'coh_HV', 'coh_HH_win', 'coh_HV_win']
STATS = [b for b in BANDS if b not in ('n_first', 'n_can', 'fill') and not b.endswith(tuple(k for k in CUTS if k))]  # 1.37 m only
MAIN = ['c_p98_can', 'c_p98_all', 'mean_all', 'p98_all', 'p98_can', 'cover137']
STAT_LABELS = {
    'cover137': 'cover (1.37 m)', 'p98_all': 'p98, all', 'p90_all': 'p90, all', 'mean_all': 'mean first-return h',
    'p98_can': 'p98, canopy', 'p90_can': 'p90, canopy', 'mean_can': 'mean, canopy',
    'c_p98_all': 'cover × p98, all', 'c_p90_all': 'cover × p90, all', 'c_mean_all': 'cover × mean, all',
    'c_p98_can': 'cover × p98, canopy', 'c_p90_can': 'cover × p90, canopy', 'c_mean_can': 'cover × mean, canopy',
}
TRACKS = ['147_D', '168_A']
CHECK_DATES = {'147_D': '20261010', '168_A': '20261011'}  # single dates for the checks (stack dates)
SEPOCT = {'147_D': ['20260928', '20261010'], '168_A': ['20260929', '20261011']}  # the stack's two dates per track


def read_window(path, tf, shape, margin=0, bands=None) -> np.ndarray:
    """Bands of path on the window (tf, shape) grown by margin cells; nan outside / nodata."""
    with rasterio.open(path) as s:
        c, r = ~s.transform * (tf.c, tf.f)
        assert abs(c - round(c)) < 1e-6 and abs(r - round(r)) < 1e-6 and s.transform.a == tf.a, \
            f'{Path(path).name}: window is not on the file grid'
        win = Window(round(c) - margin, round(r) - margin, shape[1] + 2 * margin, shape[0] + 2 * margin)
        a = s.read(bands, window=win, boundless=True, fill_value=np.nan).astype('float64')
        if s.nodata is not None:  # boundless reads do not apply the nodata mask
            a[a == s.nodata] = np.nan
    return a


def sar_sources(grid: str) -> list[tuple[Path, list[int], list[dict]]]:
    """(file, band indexes, feature descriptors) for every SAR feature on the 20 m grid."""
    assert grid == 'nisar20m', f'{grid}: only the 20 m grid exists (albers30m removed 2026-10-22)'
    names = json.loads(STACK.with_suffix('.json').read_text())['features']
    idx = [i for i, n in enumerate(names, 1) if n[:5] in TRACKS]
    out = [(STACK, idx, [{'kind': 'stack', 'track': names[i - 1][:5], 'date': '', 'date2': '',
                          'band': names[i - 1][6:]} for i in idx])]
    for f in sorted(f for f in DCMP_DIR.glob('dcmp_*.tif') if DCMP_RE.search(f.name)):
        d, trk = DCMP_RE.search(f.name).groups()
        with rasterio.open(f) as s:
            desc = s.descriptions
        idx = [i for i, n in enumerate(desc, 1) if n != 'n_looks']
        out.append((f, idx, [{'kind': 'dcmp', 'track': trk, 'date': d, 'date2': '', 'band': desc[i - 1]}
                             for i in idx]))
    for f in sorted(f for f in COH_DIR.glob('coh_*.tif') if COH_RE.search(f.name)):
        d1, d2, trk = COH_RE.search(f.name).groups()
        with rasterio.open(f) as s:
            desc = s.descriptions
        idx = [desc.index(b) + 1 for b in COH_BANDS]
        out.append((f, idx, [{'kind': 'coh', 'track': trk, 'date': d1, 'date2': d2, 'band': b} for b in COH_BANDS]))
    return out


def feature_id(m: dict) -> str:
    return '_'.join(v for v in (m['track'], m['date'], m['date2'], m['band']) if v)


def load_grid(grid: str, exclude=EXCLUDE_SAR) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Cells (structure bands), SAR features at the cells, feature metadata, per-site window.

    Bands are read by their TIFF descriptions, so an older file with fewer bands still loads
    (its missing bands are NaN)."""
    sources = sar_sources(grid)
    cells, feats, windows = [], [], {}
    meta = {feature_id(m): m for _, _, ms in sources for m in ms}
    for f in sorted(OUT_DIR.glob(f'struct_{grid}_*.tif')):
        if f.stem.endswith('.partial'):
            continue
        site = f.stem[len(f'struct_{grid}_'):]
        if site in exclude:
            continue
        with rasterio.open(f) as s:
            a = s.read(masked=True).filled(np.nan).astype('float64')
            tf, shape, desc = s.transform, (s.height, s.width), list(s.descriptions)
        windows[site] = (tf, shape)
        fill = a[desc.index('fill')]
        keep = (fill >= FILL_MIN) & np.isfinite(a[desc.index('cover137')])
        n_drop = int(((fill > 0) & ~keep).sum())
        rr, cc = np.nonzero(keep)
        d = pd.DataFrame({b: a[i][rr, cc] for i, b in enumerate(desc)}).assign(site=site, row=rr, col=cc)
        x = {}
        for path, idx, ms in sources:
            s_ = read_window(path, tf, shape, bands=idx)
            for j, m in enumerate(ms):
                x[feature_id(m)] = s_[j][rr, cc]
        cells.append(d)
        feats.append(pd.DataFrame(x))
        print(f'[svs] {grid} {site}: {len(d):,} cells kept, {n_drop:,} edge cells (fill < {FILL_MIN}) dropped',
              flush=True)
    cells = pd.concat(cells, ignore_index=True)
    feats = pd.concat(feats, ignore_index=True)
    dc = [k for k, m in meta.items() if m['kind'] == 'dcmp']  # date means of each dcmp band, per track
    for trk in TRACKS:
        for b in sorted({meta[k]['band'] for k in dc}):
            cols = [k for k in dc if meta[k]['track'] == trk and meta[k]['band'] == b]
            sep = [k for k in cols if meta[k]['date'] in SEPOCT[trk]]
            assert len(sep) == len(SEPOCT[trk]), f'{trk} {b}: Sep-Oct dates missing'
            for name, sel in (('mean', cols), ('sepoct', sep)):
                k = f'{trk}_{name}_{b}'
                if b.endswith('_db'):  # powers: mean in linear units
                    feats[k] = 10 * np.log10((10 ** (feats[sel] / 10)).mean(axis=1, skipna=False))
                else:
                    feats[k] = feats[sel].mean(axis=1, skipna=False)
                meta[k] = {'kind': f'dcmp_{name}', 'track': trk, 'date': '', 'date2': '', 'band': b}
    return cells, feats, meta, windows


def corr(x: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < MIN_N or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {'n': n, 'r': np.nan, 'r2': np.nan, 'rho': np.nan, 'rho2': np.nan}
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    rho = np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]
    return {'n': n, 'r': r, 'r2': r * r, 'rho': rho, 'rho2': rho * rho}


def metrics(grid, cells, feats, meta) -> pd.DataFrame:
    rows = []
    groups = [('all_sites', cells.index)] + [(s, g.index) for s, g in cells.groupby('site')]
    for site, ix in groups:
        F = feats.loc[ix]
        for st in STATS:
            x = cells.loc[ix, st].to_numpy()
            for k in feats.columns:
                rows.append({'grid': grid, 'feature': k, **meta[k], 'stat': st, 'site': site,
                             **corr(x, F[k].to_numpy())})
    return pd.DataFrame(rows)


def shift_check(grid, cells, windows, path, band, stat='mean_all'):
    """|rho| of stat against one SAR band with the SAR shifted by -1..1 cells (alignment check)."""
    out = np.zeros((3, 3))
    for i, dr in enumerate((-1, 0, 1)):
        for j, dc in enumerate((-1, 0, 1)):
            xs, ys = [], []
            for site, g in cells.groupby('site'):
                tf, shape = windows[site]
                a = read_window(path, tf, shape, margin=1, bands=[band])[0]
                xs.append(g[stat].to_numpy())
                ys.append(a[g.row.to_numpy() + 1 + dr, g.col.to_numpy() + 1 + dc])
            out[i, j] = abs(corr(np.concatenate(xs), np.concatenate(ys))['rho'])
    print(f'[svs] alignment {grid} {Path(path).name} band {band}: |rho({stat})| by shift (rows dy -1..1, cols dx -1..1)\n'
          + np.array2string(out, precision=3), flush=True)
    assert out[1, 1] == out.max(), f'{grid}: |rho| is not largest at zero shift'


def consistency_checks(cells20: pd.DataFrame, windows20: dict) -> None:
    """p98_all against the CHM p98 of nisar_2026 labels; site cover against gap_fraction."""
    labels = HERE.parent / 'nisar_2026' / 'outputs' / 'labels'
    for site, g in cells20.groupby('site'):
        f = labels / f'chm_targets_{site}.tif'
        if f.exists():
            tf, shape = windows20[site]
            chm_p98 = read_window(f, tf, shape, bands=[4])[0][g.row, g.col]  # bands mean, max, p90, p98, count
            chm_p98[chm_p98 == -9999] = np.nan
            c = corr(g.p98_all.to_numpy(), chm_p98)
            print(f'[svs] {site}: points p98_all vs CHM p98 r {c["r"]:.3f} (n {c["n"]:,}), '
                  f'mean difference {np.nanmean(g.p98_all - chm_p98):+.2f} m', flush=True)
    gs = pd.read_csv(HERE / 'outputs' / 'gap_fraction' / 'gap_summary.csv')
    gs = gs[np.isclose(gs.h, 1.37)].set_index('site')
    for site, g in cells20.groupby('site'):
        cov = g.n_can.sum() / g.n_first.sum()
        print(f'[svs] {site}: pooled cover137 {cov:.3f}, 1 - gap_fraction pts_pooled '
              f'{1 - gs.loc[site, "pts_pooled"]:.3f}', flush=True)


def scatter(ax, x, y, color, label=True):
    ax.scatter(x, y, s=2, color=color, alpha=0.15, linewidths=0, rasterized=True)
    if label:
        c = corr(np.asarray(x, float), np.asarray(y, float))
        ax.text(0.03, 0.97, f'R² {c["r2"]:.2f}\nρ {c["rho"]:.2f}', transform=ax.transAxes, va='top',
                fontsize=9, color=viz_style.TEXT_PRIMARY)


def plot_check(c20, f20, out):
    panels = [('20 m stack', '147_D', f20, c20, '147_D_{}_dB'), ('20 m stack', '168_A', f20, c20, '168_A_{}_dB'),
              ('20 m single date', '147_D', f20, c20, '147_D_' + CHECK_DATES['147_D'] + '_{}_sigma0_db'),
              ('20 m single date', '168_A', f20, c20, '168_A_' + CHECK_DATES['168_A'] + '_{}_sigma0_db')]
    rows = [(pol, st) for pol in ('HV', 'HH') for st in ('p98_all', 'mean_all')]
    fig, axes = plt.subplots(len(rows), len(panels), figsize=(3.6 * len(panels), 3.1 * len(rows)),
                             constrained_layout=True, sharex='row')
    for i, (pol, st) in enumerate(rows):
        color = viz_style.TARGET_COLORS['p98' if st == 'p98_all' else 'mean']
        for j, (gname, trk, F, C, pat) in enumerate(panels):
            ax = axes[i, j]
            scatter(ax, C[st], F[pat.format(pol)], color)
            ax.set_xlim(0, None)
            if i == 0:
                date = '' if gname.endswith('stack') else f', {CHECK_DATES[trk][:4]}-{CHECK_DATES[trk][4:6]}-{CHECK_DATES[trk][6:]}'
                ax.set_title(f'{gname}, {trk.replace("_", " ")}{date}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{pol} (dB)')
            ax.set_xlabel(f'{STAT_LABELS[st]} (m)')
    fig.suptitle('Check: L-band backscatter against canopy height (AGB proxy), 2026 LiDAR, 5 sites',
                 color=viz_style.TEXT_PRIMARY, fontsize=12)
    fig.text(0.01, -0.01, 'Stack: GCOV γ0, Sep–Oct composite. Single date: GSLC σ0, one image. 20 m, Refined Lee 5×5. '
             f'Cells with LiDAR fill ≥ {FILL_MIN}. R²: linear fit; ρ: Spearman.',
             fontsize=8, color=viz_style.TEXT_MUTED, va='top')
    fig.savefig(out)
    plt.close(fig)
    print(f'[svs] wrote {out.resolve()}', flush=True)


def plot_matrix(m20: pd.DataFrame, cols: list[str], out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True, sharey=True)
    for ax, metric, title in zip(axes, ('r2', 'rho2'), ('Linear R²', 'Spearman ρ²')):
        p = m20.pivot(index='stat', columns='feature', values=metric).loc[STATS, cols]
        ax.imshow(p.values, cmap=viz_style.SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect='auto')
        for (i, j), v in np.ndenumerate(p.values):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.5,
                    color='white' if v > 0.55 else viz_style.TEXT_PRIMARY)
        ax.set_xticks(range(len(cols)), [c.replace('_', ' ') for c in cols], rotation=60, ha='right', fontsize=8)
        ax.set_yticks(range(len(STATS)), [STAT_LABELS[s] for s in STATS], fontsize=9)
        ax.grid(False)
        ax.set_title(title)
    fig.suptitle('Structure statistics against NISAR features, 20 m stack, all sites pooled',
                 color=viz_style.TEXT_PRIMARY, fontsize=12)
    fig.savefig(out)
    plt.close(fig)
    print(f'[svs] wrote {out.resolve()}', flush=True)


def plot_scatter_main(cells, feats, cols, out):
    rows = ['c_p98_can', 'c_p98_all', 'p98_all', 'cover137']
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.6 * len(cols), 3.0 * len(rows)),
                             constrained_layout=True, sharey='col', sharex='row')
    color = viz_style.TARGET_COLORS['mean']
    for i, st in enumerate(rows):
        for j, k in enumerate(cols):
            ax = axes[i, j]
            scatter(ax, cells[st], feats[k], color)
            if i == 0:
                ax.set_title(k.replace('_', ' '), fontsize=10)
            if j == 0:
                ax.set_ylabel('SAR value')
            ax.set_xlabel(STAT_LABELS[st] + ('' if st == 'cover137' else ' (m)'))
    fig.suptitle('Cover × p98 against the best 20 m NISAR features, beside p98 and cover alone',
                 color=viz_style.TEXT_PRIMARY, fontsize=12)
    fig.savefig(out)
    plt.close(fig)
    print(f'[svs] wrote {out.resolve()}', flush=True)


def plot_season(m30: pd.DataFrame, out):
    series = [('c_p98_can', 'mean'), ('p98_all', 'max'), ('cover137', 'p98')]  # stat, viz_style colour slot
    feats = [('dcmp', 'HV_sigma0_db', 'HV σ0'), ('dcmp', 'HH_sigma0_db', 'HH σ0'), ('coh', 'coh_HH_win', 'HH coherence (window)')]
    fig, axes = plt.subplots(len(TRACKS), len(feats), figsize=(4.6 * len(feats), 3.4 * len(TRACKS)),
                             constrained_layout=True, sharey=True)
    for i, trk in enumerate(TRACKS):
        for j, (kind, band, title) in enumerate(feats):
            ax = axes[i, j]
            for st, slot in series:
                d = m30[(m30.kind == kind) & (m30.band == band) & (m30.track == trk) & (m30.stat == st)].copy()
                d1 = pd.to_datetime(d.date)
                d['t'] = d1 + (pd.to_datetime(d.date2) - d1) / 2 if kind == 'coh' else d1
                d = d.sort_values('t')
                ax.plot(d.t, d.rho2, color=viz_style.TARGET_COLORS[slot], marker='o', markersize=5,
                        label=STAT_LABELS[st])
                if len(d) and kind != 'coh':  # coherence lines end together: legend only
                    ax.annotate(STAT_LABELS[st], xy=(d.t.iloc[-1], d.rho2.iloc[-1]), xytext=(6, 0),
                                textcoords='offset points', va='center', fontsize=8, color=viz_style.TEXT_SECONDARY,
                                annotation_clip=False)
            ax.set_title(f'{trk.replace("_", " ")}: {title}', fontsize=10)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.set_ylim(0, 1)
            if j == 0:
                ax.set_ylabel('Spearman ρ²')
    axes[0, 0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Season: structure against NISAR by date, 20 m single-date images, all sites pooled',
                 color=viz_style.TEXT_PRIMARY, fontsize=12)
    fig.text(0.01, -0.01, 'Coherence plotted at the mid-date of each pair (unfiltered). Backscatter: Refined Lee 5×5.',
             fontsize=8, color=viz_style.TEXT_MUTED, va='top')
    fig.savefig(out)
    plt.close(fig)
    print(f'[svs] wrote {out.resolve()}', flush=True)


def main():
    viz_style.apply_style()
    for d in ('checks', 'pooled'):
        (FIG_DIR / d).mkdir(parents=True, exist_ok=True)
    c20, f20, meta20, w20 = load_grid('nisar20m')
    consistency_checks(c20, w20)
    shift_check('nisar20m', c20, w20, STACK,
                json.loads(STACK.with_suffix('.json').read_text())['features'].index('168_A_HV_dB') + 1)
    shift_check('nisar20m', c20, w20, DCMP_DIR / f'dcmp_{CHECK_DATES["168_A"]}_032_168_A.tif', 2)
    plot_check(c20, f20, FIG_DIR / 'checks' / 'check_backscatter_vs_height.png')

    m = metrics('nisar20m', c20, f20, meta20)
    m.to_csv(OUT_DIR / 'structure_sar_stats.csv', index=False, float_format='%.4f')
    print(f'[svs] wrote {(OUT_DIR / "structure_sar_stats.csv").resolve()} ({len(m):,} rows)', flush=True)

    pooled = m[m.site == 'all_sites']
    m20 = pooled[pooled.kind == 'stack']
    best = (m20.groupby('feature').rho2.max().sort_values(ascending=False))
    cols = list(best.index[:12])
    print('[svs] 20 m stack, pooled rho2 (best 12 features):')
    print(m20.pivot(index='stat', columns='feature', values='rho2').loc[STATS, cols].to_string(float_format='%.2f'))
    ms = pooled[pooled.kind != 'stack']
    print('[svs] 20 m single-date images, pairs and date means, best features by pooled rho2 for each main statistic:')
    for st in MAIN:
        top = ms[ms.stat == st].nlargest(3, 'rho2')
        print(f'  {st:10s} ' + '; '.join(f'{r.feature} {r.rho2:.2f} (R2 {r.r2:.2f})' for r in top.itertuples()))
    by_site = (m[(m.kind == 'stack') & m.stat.isin(MAIN) & m.feature.isin(cols[:4])]
               .pivot_table(index=['feature', 'site'], columns='stat', values='rho2')[MAIN])
    by_site.to_csv(OUT_DIR / 'r2_by_site.csv', float_format='%.4f')
    print(by_site.to_string(float_format='%.2f'))

    plot_matrix(m20, cols, FIG_DIR / 'pooled' / 'r2_matrix.png')
    top4 = list(m20[m20.stat == 'c_p98_can'].nlargest(4, 'rho2').feature)
    plot_scatter_main(c20, f20, top4, FIG_DIR / 'pooled' / 'scatter_main.png')
    plot_season(ms, FIG_DIR / 'pooled' / 'season.png')


if __name__ == '__main__':
    main()
