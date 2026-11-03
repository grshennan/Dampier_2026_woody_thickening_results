"""Fire effects on GEDI RH98 of one year Y (--year, default 2025) against the 2026 drone LiDAR CHM.

Shots, footprint statistics, strata and offsets: those of gedi_wiggle.py (read from its
cache outputs/gedi/wiggle_footprints_v<ver>.parquet; no CHM is read here).
Fire: NAFI monthly burnt-area polygons (250 m), read with woody_thickening_splice_1/
fire_data.load_scars (EPSG:4283 bbox read, reprojected to EPSG:7851).

Rules, fixed before any result was seen:
- One reference date for all shots, T0 = Y-01-01 (2025: GEDI shots 2025-02-13..07-09;
  LiDAR 2026-10-05..09):
    pre    any scar in calendar Y-1 -> regrowth at GEDI time
    inter  any scar Y-01..2026-10 -> burnt between GEDI and LiDAR
  Sub-periods of inter, reported, not used for grouping: inter_<Y>_early (Jan-Jul, overlaps
  the shot dates, so the fire can precede the shot), inter_<Y>_late (Aug-Dec), inter_<y>
  for each later year, inter_2026 (Jan-Oct; October overlaps the flights and month
  resolution cannot order them). inter_n1 / inter_n2 / inter_n3plus: shots by the number
  of these sub-periods with a scar (added with the earlier years, where the gap is long).
- A shot is burnt in a window if its 25 m disc (at the offset of the level) intersects a scar.
- Groups: clean (neither), pre_only, inter_only, both; pre_any, inter_any; clean_strict =
  no scar within 250 m (one NAFI pixel) of the disc in either window.
- Levels: nominal and the global offset chosen on all shots (gedi_wiggle.search on the
  stratum). No re-search per group: it would inflate r.
- Per stratum x level x group x site (and pindan / mainland, never blended): n, bias
  (stat - RH98), RMSE, r, R2, for p90 (headline) and p98.
- Null for "clean r is higher": N_NULL random subsets of the headline shots with the same
  per-site counts as the group (fire is site-clustered); report the share of null r >= the
  observed r, and of null RMSE <= the observed RMSE.
- Differences clean - fire group: bootstrap (N_BOOT, each group resampled by itself),
  95 % interval of the r and bias differences.
- Expected signs: inter -> LiDAR lower -> bias more negative than clean; pre -> regrowth
  after the shot -> bias more positive. Groups with n < MIN_INTERPRET are reported only.
- Added after the first run, which showed the groups are nearly the sites (9 Mile is all
  'both'; clean is Bunda Extra and Country Downs): within-site differences for every pair
  of clean / pre_only / inter_only / both with >= MIN_PAIR shots each in that site, by the
  same bootstrap. These are the only comparisons not confounded with site.
- Added with the earlier years (every 2021 shot burnt in between, so no clean group): the
  within-site OLS slope of (stat - RH98) on n_inter, per site and pooled with site means
  removed; bootstrap within site. Expected sign: negative (more fires, lower LiDAR).

Outputs (gedi_wiggle.out_dir(year): outputs/gedi/ for 2025, outputs/gedi/<year>/ otherwise):
    fire_shots_v<ver>.csv   per shot: flags, burnt fractions, scar distances, RH98, p90, p98
    fire_summary.csv        version x stratum x level x group x site x stat
    fire_null.csv           headline: observed against the site-stratified subset null
    fire_diff.csv           headline: clean - fire group, bootstrap intervals
    fire_trend.csv          headline: within-site slope of the error on the fire count
    fire_scatter.png, fire_map.png

Usage
-----
conda run -n general --no-capture-output python -u gedi_fire.py --self-test
conda run -n general --no-capture-output python -u gedi_fire.py
"""

from __future__ import annotations

import argparse
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from chm_class_histograms import HERE
from gedi_wiggle import (DX, DY, HEADLINE, K0, MIN_VALID, RADIUS_M, SEED, STRATA,
                         load_footprints, out_dir, metrics, search, site_chms, stat_matrix)
import viz_style

import matplotlib.pyplot as plt  # noqa: E402  (after viz_style sets Agg)

sys.path.insert(0, str(HERE.parent / 'woody_thickening_splice_1'))
import fire_data  # noqa: E402

CRS = 'EPSG:7851'
YEAR = 2025
LAST_FIRE = (2026, 10)  # the LiDAR month; NAFI 2026 is to 18 Oct
STRICT_M = 250.0
BBOX_PAD_M = 1000.0
STATS = ['p90', 'p98']
GROUPS = ['all', 'clean', 'clean_strict', 'pre_only', 'inter_only', 'both', 'pre_any', 'inter_any']
FIRE_GROUPS = ['pre_only', 'inter_only', 'both', 'pre_any', 'inter_any']
MIN_INTERPRET = 10
PAIR_GROUPS = ['clean', 'pre_only', 'inter_only', 'both']
MIN_PAIR = 5
N_NULL = 10000
N_BOOT = 2000


def windows(year: int) -> dict[str, dict[int, range]]:
    w = {'pre': {year - 1: range(1, 13)}, f'inter_{year}_early': {year: range(1, 8)},
         f'inter_{year}_late': {year: range(8, 13)}}
    w |= {f'inter_{y}': {y: range(1, 13)} for y in range(year + 1, LAST_FIRE[0])}
    return w | {f'inter_{LAST_FIRE[0]}': {LAST_FIRE[0]: range(1, LAST_FIRE[1] + 1)}}


def inter_parts(cols) -> list[str]:
    return [c[6:] for c in cols if c.startswith('burnt_inter_')]


def load_windows(bounds, year: int) -> tuple[dict, dict]:
    """Union of scars per window, and per month 1-7 of the GEDI year (for the shot-order check)."""
    scars, spec_all = {}, windows(year)
    for y in sorted({y for w in spec_all.values() for y in w}):
        g = fire_data.load_scars(y, bounds, CRS)
        scars[y] = g[g.Month.notna()]
        print(f'[fire] {fire_data.scar_path(y).name}: {len(scars[y])} scars in the bbox, months '
              f'{scars[y].Month.astype(int).value_counts().sort_index().to_dict()}', flush=True)
    uni = lambda g: shapely.union_all(shapely.make_valid(g.geometry.to_numpy())) if len(g) else shapely.Polygon()
    win = {w: uni(pd.concat([scars[y][scars[y].Month.isin(list(ms))] for y, ms in spec.items()]))
           for w, spec in spec_all.items()}
    win['inter'] = shapely.union_all([g for w, g in win.items() if w.startswith('inter_')])
    month_y = {m: uni(scars[year][scars[year].Month == m]) for m in range(1, 8)}
    return win, month_y


def fire_flags(xs, ys, win: dict, month_y: dict | None = None, shot_month=None) -> pd.DataFrame:
    """Per shot: burnt flag and disc fraction per window, distance disc edge -> scar, group."""
    pts = shapely.points(xs, ys)
    discs = shapely.buffer(pts, RADIUS_M, quad_segs=32)
    out = {}
    for w, geom in win.items():
        shapely.prepare(geom)
        out[f'burnt_{w}'] = shapely.intersects(geom, discs)
        out[f'frac_{w}'] = shapely.area(shapely.intersection(discs, geom)) / shapely.area(discs)
        out[f'dist_{w}'] = np.maximum(shapely.distance(pts, geom) - RADIUS_M, 0) if not geom.is_empty \
            else np.full(len(xs), np.inf)
    f = pd.DataFrame(out)
    p, i = f.burnt_pre.to_numpy(), f.burnt_inter.to_numpy()
    f['group'] = np.select([~p & ~i, p & ~i, ~p & i], ['clean', 'pre_only', 'inter_only'], 'both')
    f['clean_strict'] = (f.dist_pre > STRICT_M) & (f.dist_inter > STRICT_M)
    f['n_inter'] = f[[f'burnt_{w}' for w in inter_parts(f.columns)]].sum(axis=1)
    if month_y is not None:  # GEDI-year Jan-Jul fire month against the shot month (diagnostic only)
        hit = np.column_stack([shapely.intersects(month_y[m], discs) for m in range(1, 8)])
        months = np.arange(1, 8)
        sm = np.asarray(shot_month)[:, None]
        f['early_before_shot'] = (hit & (months < sm)).any(axis=1)
        f['early_same_month'] = (hit & (months == sm)).any(axis=1)
        f['early_after_shot'] = (hit & (months > sm)).any(axis=1)
    return f


def group_masks(f: pd.DataFrame) -> dict[str, np.ndarray]:
    g = f.group.to_numpy()
    return {'all': np.ones(len(f), bool), 'clean': g == 'clean', 'clean_strict': f.clean_strict.to_numpy(),
            'pre_only': g == 'pre_only', 'inter_only': g == 'inter_only', 'both': g == 'both',
            'pre_any': f.burnt_pre.to_numpy(), 'inter_any': f.burnt_inter.to_numpy(),
            'inter_n1': (f.n_inter == 1).to_numpy(), 'inter_n2': (f.n_inter == 2).to_numpy(),
            'inter_n3plus': (f.n_inter >= 3).to_numpy()} \
        | {f'{w}_any': f[f'burnt_{w}'].to_numpy() for w in inter_parts(f.columns)}


def fast_r(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson r for (n_draw, n) prediction and target matrices."""
    Pc, yc = P - P.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    return (Pc * yc).sum(axis=1) / np.sqrt((Pc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))


def site_null(pred, y, site, mask, rng, n_draw=N_NULL) -> dict:
    """Random subsets with the group's per-site counts: r and RMSE distributions."""
    idx = np.empty((n_draw, int(mask.sum())), int)
    col = 0
    for s in np.unique(site[mask]):
        pool = np.flatnonzero(site == s)
        k = int((mask & (site == s)).sum())
        idx[:, col:col + k] = pool[np.argsort(rng.random((n_draw, len(pool))), axis=1)[:, :k]]
        col += k
    P, Y = pred[idx], y[idx]
    return {'r': fast_r(P, Y), 'rmse': np.sqrt(((P - Y) ** 2).mean(axis=1))}


def boot_diff(pred, y, ma, mb, rng, n_boot=N_BOOT) -> dict:
    """Bootstrap of r(a) - r(b) and bias(a) - bias(b); each group resampled by itself."""
    def draw(m):
        i = np.flatnonzero(m)[rng.integers(0, m.sum(), (n_boot, m.sum()))]
        P, Y = pred[i], y[i]
        return fast_r(P, Y), (P - Y).mean(axis=1)
    (ra, ba), (rb, bb) = draw(ma), draw(mb)
    dr, db = ra - rb, ba - bb
    return {'dr_lo': np.nanpercentile(dr, 2.5), 'dr_hi': np.nanpercentile(dr, 97.5),
            'dbias_lo': np.percentile(db, 2.5), 'dbias_hi': np.percentile(db, 97.5)}


def count_trend(e, n, site, rng, n_boot=N_BOOT) -> dict:
    """Within-site OLS slope of the error e on the fire count n (site-demeaned), bootstrap within site."""
    def slope(idx):
        ec, nc = e[idx], n[idx].astype(float)
        for s in np.unique(site[idx]):
            m = site[idx] == s
            ec[m], nc[m] = ec[m] - ec[m].mean(), nc[m] - nc[m].mean()
        return (nc * ec).sum() / (nc ** 2).sum() if (nc ** 2).sum() > 0 else np.nan
    pools = [np.flatnonzero(site == s) for s in np.unique(site)]
    b = [slope(np.concatenate([p[rng.integers(0, len(p), len(p))] for p in pools])) for _ in range(n_boot)]
    return {'n': len(e), 'n_levels': len(np.unique(n)), 'slope': slope(np.arange(len(e))),
            'slope_lo': np.nanpercentile(b, 2.5), 'slope_hi': np.nanpercentile(b, 97.5)}


def analyse(version: str, df: pd.DataFrame, win: dict, month_y: dict, rng):
    full = df[df[[f'frac_{k}' for k in range(len(DX))]].min(axis=1) >= MIN_VALID].reset_index(drop=True)
    summary, nulls, diffs, trends, shots = [], [], [], [], None
    rng_t = np.random.default_rng(SEED + 1)  # separate stream: keeps the null / diff draws as first run
    for stratum, (lo, hi) in STRATA.items():
        d = full[(full.rh98 >= lo) & (full.rh98 <= hi)].reset_index(drop=True)
        if len(d) < 3:
            continue
        y, site = d.rh98.to_numpy(float), d.site.to_numpy()
        g = int(search(stat_matrix(d, 'p98'), y, d.track.to_numpy())['global'][0])
        for level, k in (('nominal', K0), ('global', g)):
            f = fire_flags(d.x.to_numpy() + DX[k], d.y.to_numpy() + DY[k], win, month_y, d.time.dt.month)
            masks = group_masks(f)
            where = {'all_sites': np.ones(len(d), bool), 'pindan': d.in_pindan.to_numpy(),
                     'mainland': d.in_mainland.to_numpy()} | {s: site == s for s in np.unique(site)}
            for st in STATS:
                pred = stat_matrix(d, st)[k]
                for gname, gm in masks.items():
                    for wname, wm in where.items():
                        m = gm & wm
                        row = {'version': version, 'stratum': stratum, 'level': level, 'dx': DX[k], 'dy': DY[k],
                               'group': gname, 'area': wname, 'stat': st}
                        if m.sum() < 3:
                            summary.append(row | {'n': int(m.sum())})
                            continue
                        summary.append(row | metrics(pred[m], y[m]) | {'interpret': bool(m.sum() >= MIN_INTERPRET)})
                if stratum != HEADLINE:
                    continue
                for gname in ('clean', 'clean_strict') + tuple(FIRE_GROUPS):
                    gm = masks[gname]
                    if gm.sum() < 3:
                        continue
                    obs = metrics(pred[gm], y[gm])
                    nd = site_null(pred, y, site, gm, rng)
                    nulls.append({'version': version, 'level': level, 'stat': st, 'group': gname, 'n': obs['n'],
                                  'r': obs['r'], 'rmse': obs['rmse'], 'r_all': metrics(pred, y)['r'],
                                  'null_r_median': np.nanmedian(nd['r']), 'null_r_lo': np.nanpercentile(nd['r'], 2.5),
                                  'null_r_hi': np.nanpercentile(nd['r'], 97.5),
                                  'p_null_r_ge_obs': np.nanmean(nd['r'] >= obs['r']),
                                  'null_rmse_median': np.median(nd['rmse']),
                                  'p_null_rmse_le_obs': np.mean(nd['rmse'] <= obs['rmse'])})
                pairs = [('all_sites', 'clean', g, 3) for g in FIRE_GROUPS] + \
                        [(s, a, b, MIN_PAIR) for s in np.unique(site) for i, a in enumerate(PAIR_GROUPS)
                         for b in PAIR_GROUPS[i + 1:]]
                for area, ga, gb in ((p[0], p[1], p[2]) for p in pairs
                                     if all(((masks[g] & where[p[0]]).sum() >= p[3]) for g in p[1:3])):
                    ma, mb = masks[ga] & where[area], masks[gb] & where[area]
                    a, b = metrics(pred[ma], y[ma]), metrics(pred[mb], y[mb])
                    diffs.append({'version': version, 'level': level, 'stat': st, 'area': area, 'a': ga, 'b': gb,
                                  'n_a': a['n'], 'n_b': b['n'], 'dr': a['r'] - b['r'], 'dbias': a['bias'] - b['bias'],
                                  'drmse': a['rmse'] - b['rmse'], **boot_diff(pred, y, ma, mb, rng)})
                e, n_int = pred - y, f.n_inter.to_numpy()
                for area in ['all_sites'] + list(np.unique(site)):
                    m = where[area]
                    if m.sum() >= MIN_INTERPRET and len(np.unique(n_int[m])) > 1:
                        trends.append({'version': version, 'level': level, 'stat': st, 'area': area,
                                       **count_trend(e[m], n_int[m], site[m], rng_t)})
            if stratum == HEADLINE:
                part = pd.concat([d[['shot_number', 'site', 'time', 'beam_type', 'in_pindan', 'rh98']],
                                  pd.DataFrame({'level': level, 'dx': DX[k], 'dy': DY[k],
                                                'p90': stat_matrix(d, 'p90')[k], 'p98': stat_matrix(d, 'p98')[k],
                                                'x': d.x + DX[k], 'y': d.y + DY[k]}), f], axis=1)
                shots = part if shots is None else pd.concat([shots, part], ignore_index=True)
                c = f.group.value_counts().to_dict()
                print(f'[fire] v{version} {level} ({DX[k]:+.1f}, {DY[k]:+.1f}) m, n {len(d)}: {c}, '
                      f'clean_strict {int(f.clean_strict.sum())}; inter parts '
                      + ', '.join(f'{w} {int(f[f"burnt_{w}"].sum())}' for w in inter_parts(f.columns))
                      + f'; n_inter {f.n_inter.value_counts().sort_index().to_dict()}'
                      + f'; GEDI-year Jan-Jul fire before / same / after shot month: '
                        f'{int(f.early_before_shot.sum())} / {int(f.early_same_month.sum())} / '
                        f'{int(f.early_after_shot.sum())}', flush=True)
    shots['version'] = version
    return summary, nulls, diffs, trends, shots


def check_against_wiggle(summ: pd.DataFrame, year: int) -> None:
    """The 'all' group must reproduce gedi_wiggle's numbers (same population, same offsets)."""
    ref = pd.read_csv(out_dir(year) / 'wiggle_summary.csv', dtype={'version': str})
    ref = ref[ref.version.isin(summ.version.unique())]
    ref = ref[ref.level.isin(['nominal', 'global']) & ref.stat.isin(STATS) & (ref.group == 'all_sites')]
    got = summ[(summ.group == 'all') & (summ.area == 'all_sites')]
    m = ref.merge(got, on=['version', 'stratum', 'level', 'stat'], suffixes=('_ref', ''))
    assert len(m) == len(ref), f'{len(ref) - len(m)} wiggle rows not matched'
    for c in ('n', 'bias', 'rmse', 'r'):
        assert np.allclose(m[f'{c}_ref'], m[c], atol=1e-4), f'{c} differs from wiggle_summary.csv'
    h = m[(m.version == m.version.max()) & (m.stratum == HEADLINE) & (m.level == 'nominal') & (m.stat == 'p90')]
    if not len(h):
        print(f'[fire] check: "all" = wiggle_summary.csv for {len(m)} rows')
        return
    h = h.iloc[0]
    print(f'[fire] check: "all" = wiggle_summary.csv for {len(m)} rows; v{h.version} nominal p90 n {int(h.n)} '
          f'bias {h.bias:+.2f} RMSE {h.rmse:.2f} r {h.r:.3f}')


def plot_scatter(shots: pd.DataFrame, summ: pd.DataFrame, out, year: int) -> None:
    panels = ['all', 'clean', 'clean_strict', 'pre_only', 'inter_only', 'both']
    d = shots[(shots.version == '003') & (shots.level == 'nominal')]
    masks = group_masks(d.reset_index(drop=True))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4), constrained_layout=True, sharex=True, sharey=True)
    lim = (0, max(d.rh98.max(), d.p90.max()) * 1.05)
    for ax, gname in zip(axes.ravel(), panels):
        g = d[masks[gname]]
        ax.plot(lim, lim, color=viz_style.TEXT_MUTED, linewidth=1, linestyle='--')
        ax.scatter(g.p90, g.rh98, s=16, color=viz_style.color_for('p90'), alpha=0.85,
                   edgecolor=viz_style.SURFACE, linewidth=0.5)
        m = summ[(summ.version == '003') & (summ.stratum == HEADLINE) & (summ.level == 'nominal')
                 & (summ.group == gname) & (summ.area == 'all_sites') & (summ.stat == 'p90')].iloc[0]
        txt = f'n {int(m.n)}' if m.n < 3 or pd.isna(m.get('rmse')) else f'n {int(m.n)}  bias {m.bias:+.2f} m\nRMSE {m.rmse:.2f} m  r {m.r:.2f}'
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', fontsize=8.5, color=viz_style.TEXT_SECONDARY)
        ax.set(xlim=lim, ylim=lim, aspect='equal')
        ax.set_title(gname.replace('_', ' '), fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel('GEDI RH98 (m)')
    for ax in axes[-1]:
        ax.set_xlabel('Footprint CHM p90, 2026 LiDAR (m)')
    fig.suptitle(f'{year} GEDI (v003, nominal) against 2026 LiDAR by NAFI fire group, RH98 3–20 m\n'
                 f'pre: scar in {year - 1}; inter: scar {year}-01 to 2026-10 (dashed: 1:1)',
                 color=viz_style.TEXT_PRIMARY, fontsize=11)
    fig.savefig(out)
    plt.close(fig)
    print(f'[fire] wrote {out.resolve()}')


def plot_map(shots: pd.DataFrame, win: dict, out, year: int) -> None:
    d = shots[(shots.version == '003') & (shots.level == 'nominal')]
    sites = d.site.value_counts().index.tolist()
    nc = min(3, len(sites))
    nr = -(-len(sites) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 4.5 * nr), constrained_layout=True, squeeze=False)
    marker = {'clean': 'o', 'pre_only': '^', 'inter_only': 's', 'both': 'D'}
    fill = {'pre': (viz_style.TARGET_COLORS['mean'], f'scar {year - 1} (pre)'),
            'inter': (viz_style.TARGET_COLORS['max'], f'scar {year}-01 to 2026-10 (inter)')}
    for ax, s in zip(axes.ravel(), sites):
        g = d[d.site == s]
        x0, x1, y0, y1 = g.x.min() - 300, g.x.max() + 300, g.y.min() - 300, g.y.max() + 300
        clip = box(x0, y0, x1, y1)
        for w, (c, lab) in fill.items():
            geom = shapely.intersection(win[w], clip)
            if not geom.is_empty:  # GeoSeries.plot draws polygon holes; ax.fill of the exterior does not
                gpd.GeoSeries([geom]).plot(ax=ax, color=c, alpha=0.35, linewidth=0)
            ax.fill([], [], color=c, alpha=0.35, label=lab)
        for grp, mk in marker.items():
            h = g[g.group == grp]
            ax.scatter(h.x, h.y, marker=mk, s=22, facecolor=viz_style.TEXT_PRIMARY, edgecolor=viz_style.SURFACE,
                       linewidth=0.6, label=f'{grp.replace("_", " ")} ({len(h)})')
        ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect='equal')
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_title(f'{s} ({len(g)} shots)', fontsize=10)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.legend(loc='best', fontsize=6.5, framealpha=0.9)
    for ax in axes.ravel()[len(sites):]:
        ax.axis('off')
    fig.suptitle(f'{year} GEDI shots (v003, nominal, RH98 3–20 m) on NAFI burnt area, {CRS} (m)',
                 color=viz_style.TEXT_PRIMARY, fontsize=12)
    fig.savefig(out)
    plt.close(fig)
    print(f'[fire] wrote {out.resolve()}')


def self_test() -> None:
    rng = np.random.default_rng(SEED)
    xs, ys = np.array([0.0, 1000, 2000, 3000, 4000, 5000]), np.zeros(6)
    win = {'pre': shapely.union_all([box(-50, -50, 50, 50), box(2950, -50, 3050, 50)]),
           'inter_2025_early': box(1005, -50, 1100, 50),          # 7.5 m inside the disc edge -> burnt
           'inter_2025_late': box(2950, -50, 3050, 50),
           'inter_2026': box(4000 + 112.5, -50, 4200, 50)}        # 100 m beyond the disc edge
    win['inter'] = shapely.union_all([g for w, g in win.items() if w.startswith('inter_')])
    month_y = {m: (box(1005, -50, 1100, 50) if m == 3 else shapely.Polygon()) for m in range(1, 8)}
    f = fire_flags(xs, ys, win, month_y, [5, 5, 5, 5, 5, 2])
    assert f.group.tolist() == ['pre_only', 'inter_only', 'clean', 'both', 'clean', 'clean'], f.group.tolist()
    assert f.clean_strict.tolist() == [False, False, True, False, False, True], f.clean_strict.tolist()
    assert np.isclose(f.frac_pre[0], 1.0) and 0 < f.frac_inter[1] < 0.5
    assert np.isclose(f.dist_inter[4], 100.0, atol=0.01)
    assert f.n_inter.tolist() == [0, 1, 0, 1, 0, 0], f.n_inter.tolist()
    assert list(windows(2025)) == ['pre', 'inter_2025_early', 'inter_2025_late', 'inter_2026']
    assert list(windows(2019))[-2:] == ['inter_2025', 'inter_2026'] and windows(2019)['pre'] == {2018: range(1, 13)}
    assert f.early_before_shot.tolist() == [False, True, False, False, False, False]
    print('[self-test] fire_flags: groups, clean_strict, fractions, distance and shot order as planted')

    n = 150
    site = np.repeat(np.array(['a', 'b', 'c']), n // 3)
    y = rng.uniform(3, 15, n) + (site == 'c') * 3
    pred = y + rng.normal(0, 2, n)
    mask = rng.random(n) < 0.4
    nd = site_null(pred, y, site, mask, rng, 4000)
    r_all = np.corrcoef(pred, y)[0, 1]
    assert abs(np.median(nd['r']) - r_all) < 0.03, (np.median(nd['r']), r_all)
    for s in 'abc':  # per-site counts preserved: a site-only mask can only draw that site
        m = site == s
        nd1 = site_null(pred, y, site, m, rng, 50)
        assert np.allclose(nd1['r'], np.corrcoef(pred[m], y[m])[0, 1])
    b = boot_diff(pred, y, mask, mask, rng, 2000)
    assert b['dr_lo'] < 0 < b['dr_hi'] and b['dbias_lo'] < 0 < b['dbias_hi']
    nf = rng.integers(0, 5, n) + (site == 'c') * 2  # site c burns more and has a +3 m offset
    t = count_trend(-0.5 * nf + (site == 'c') * 3 + rng.normal(0, 0.5, n), nf, site, rng)
    assert t['slope_lo'] < -0.5 < t['slope_hi'] and abs(t['slope'] + 0.5) < 0.1, t
    print(f'[self-test] count_trend: planted -0.5 m per fire with a site offset, got {t["slope"]:+.3f} '
          f'[{t["slope_lo"]:+.3f}, {t["slope_hi"]:+.3f}]')
    print(f'[self-test] site_null median r {np.median(nd["r"]):.3f} = all-shot r {r_all:.3f}; '
          f'per-site counts preserved; boot_diff of a group with itself spans 0')
    print('[self-test] PASS')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--versions', nargs='+', default=['002', '003'])
    p.add_argument('--year', type=int, default=YEAR)
    args = p.parse_args()
    if args.self_test:
        self_test()
        return

    chms = site_chms()
    rng = np.random.default_rng(SEED)
    od = out_dir(args.year)
    dfs = {v: load_footprints(v, chms, args.year) for v in args.versions}
    allxy = pd.concat(dfs.values())
    bounds = (allxy.x.min() - BBOX_PAD_M, allxy.y.min() - BBOX_PAD_M,
              allxy.x.max() + BBOX_PAD_M, allxy.y.max() + BBOX_PAD_M)
    win, month_y = load_windows(bounds, args.year)
    summary, nulls, diffs, trends, shots = [], [], [], [], []
    for v, df in dfs.items():
        a, b, c, t, d = analyse(v, df, win, month_y, rng)
        summary += a
        nulls += b
        diffs += c
        trends += t
        shots.append(d)
        d.to_csv(od / f'fire_shots_v{v}.csv', index=False, float_format='%.3f')
    summ = pd.DataFrame(summary)
    check_against_wiggle(summ, args.year)
    shots = pd.concat(shots, ignore_index=True)
    for name, t in (('fire_summary.csv', summ), ('fire_null.csv', pd.DataFrame(nulls)),
                    ('fire_diff.csv', pd.DataFrame(diffs)), ('fire_trend.csv', pd.DataFrame(trends))):
        t.to_csv(od / name, index=False, float_format='%.4f')
        print(f'[fire] wrote {(od / name).resolve()}')

    head = summ[(summ.stratum == HEADLINE) & (summ.area == 'all_sites')]
    print(head.pivot_table(index=['version', 'level', 'stat', 'group'], values=['n', 'bias', 'rmse', 'r', 'r2'],
                           sort=False).round(3)[['n', 'bias', 'rmse', 'r', 'r2']].to_string())
    print(pd.DataFrame(nulls).round(3).to_string())
    print(pd.DataFrame(diffs).round(3).to_string())
    print(pd.DataFrame(trends).round(3).to_string())

    viz_style.apply_style()
    plot_scatter(shots, summ, od / 'fire_scatter.png', args.year)
    plot_map(shots, win, od / 'fire_map.png', args.year)


if __name__ == '__main__':
    main()
