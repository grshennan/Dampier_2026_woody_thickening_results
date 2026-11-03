"""GEDI RH98 of one year (--year, default 2025) against the 2026 drone LiDAR CHM, with a
geolocation offset search.

Shots: outputs/gedi/shots_<year>_sites_v<ver>.parquet (gedi_site_shots.py), qc_pass only.
CHM: the six in-scope 50 cm p98 CHMs (EPSG:7851; shots are projected to the same CRS).

Rules, fixed before any result was seen:
- Footprint statistic: every 50 cm CHM pixel whose centre is within RADIUS_M = 12.5 m
  of the footprint centre (the 25 m footprint, ~1,963 px). Statistics p90, p95, p98,
  max and mean of those pixels.
- Offsets: the footprint centre moves by (dx, dy), each in -30..+30 m, step 2.5 m
  (625 offsets; +-3 sigma of the 10 m nominal GEDI error). dy > 0 is north.
- Fixed population: a shot is used only if >= MIN_VALID of its disc is valid CHM at
  every offset, so every offset scores the same shots.
- Strata on RH98: all, above_floor (>= 3 m), plausible (<= 20 m),
  above_floor_plausible (3-20 m, the headline). Each stratum is searched by itself.
- Comparison as Stage 6: y = RH98, prediction = footprint CHM statistic,
  bias = statistic - RH98, R2 = 1 - SS_res / SS_tot(RH98).
- Levels:
    nominal  (0, 0) for every shot
    global   one offset for all shots: max Pearson r(RH98, p98). r ignores a constant
             bias, so the offset cannot trade against the bias
    track    one offset per track (granule x beam) with >= MIN_TRACK shots: min mean
             (p98 - RH98 - b)^2, b = the pooled p98 bias at the global offset;
             tracks with fewer shots keep the global offset
    shot     one offset per shot, min (p98 - RH98 - b)^2. Optimistic upper bound only
    global_loo, track_loo
             the same two searches, leave-one-out: each shot's offset is chosen
             without it. These are the honest gains; the in-sample levels overfit
- Global offset uncertainty: bootstrap over shots (N_BOOT), 95 % interval of dx and
  dy and the share of resamples that choose (0, 0).
- A within-site RH98 permutation null was tried first and rejected: it removes the
  nominal correlation, so its "gain" starts from r ~ 0, not from the real nominal r,
  and is not comparable (self-test: null gain 0.19 against a planted gain of 0.22).
- Offsets are chosen on the pooled shots; per site and per boundary (pindan,
  mainland) metrics use those offsets. Boundaries are never blended.

Outputs (out_dir(year): outputs/gedi/ for 2025, the first run; outputs/gedi/<year>/ otherwise):
    wiggle_footprints_v<ver>.parquet   shot x offset footprint statistics
    wiggle_surface_v<ver>.csv          stratum x offset: n, r, bias, RMSE (p98)
    wiggle_tracks_v<ver>.csv           per track: shots, chosen offset
    wiggle_shots_v<ver>.csv            per shot: RH98, p98 nominal / track / shot
    wiggle_summary.csv                 version x stratum x level x group x statistic
    wiggle_v002_v003_pairs.csv         v003 position and RH98 against v002, per shot
    wiggle_surface.png, wiggle_scatter.png

Usage
-----
conda run -n general --no-capture-output python -u gedi_wiggle.py --self-test
conda run -n general --no-capture-output python -u gedi_wiggle.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chm_class_histograms import EXCLUDE_SITES, HERE, ROOT_WIN, load_chm, site_name, to_local
import viz_style

import matplotlib.pyplot as plt  # noqa: E402  (after viz_style sets Agg)

RADIUS_M = 12.5
OFFSETS_M = np.arange(-30.0, 30.0 + 1e-9, 2.5)
MIN_VALID = 0.95
MIN_TRACK = 5
N_BOOT = 1000
STATS = ['p90', 'p95', 'p98', 'max', 'mean']
STRATA = {'all': (-np.inf, np.inf), 'above_floor': (3.0, np.inf), 'plausible': (-np.inf, 20.0),
          'above_floor_plausible': (3.0, 20.0)}
HEADLINE = 'above_floor_plausible'
LEVELS = ['nominal', 'global', 'track', 'shot', 'global_loo', 'track_loo']
GEDI_DIR = HERE / 'outputs' / 'gedi'
YEAR = 2025
SEED = 0

DX, DY = (a.ravel() for a in np.meshgrid(OFFSETS_M, OFFSETS_M))  # offset k -> (DX[k], DY[k])
K0 = int(np.flatnonzero((DX == 0) & (DY == 0))[0])


def site_chms() -> dict[str, Path]:
    root = Path(to_local(ROOT_WIN))
    return {site_name(f): f for f in sorted(root.glob('*/*_denoise_canopy_p98.tif'))
            if site_name(f) not in EXCLUDE_SITES}


def footprints(chm, xs: np.ndarray, ys: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Footprint statistics at every offset: {stat: (n_shot, n_off)}, valid fraction (n_shot, n_off)."""
    tr = chm.rio.transform()
    assert tr.b == 0 and tr.d == 0 and abs(tr.a) == abs(tr.e), 'rotated or non-square CHM grid'
    res, v = tr.a, chm.values
    ny, nx = v.shape
    off_c = np.rint(DX / res).astype(int)
    off_r = np.rint(-DY / res).astype(int)
    assert np.allclose(off_c * res, DX) and np.allclose(-off_r * res, DY), 'offsets not on the pixel grid'
    half = int(np.ceil(RADIUS_M / res)) + 1
    d = np.arange(-half, half + 1)
    out = {s: np.full((len(xs), len(DX)), np.nan) for s in STATS}
    frac = np.zeros((len(xs), len(DX)))
    for i, (x, y) in enumerate(zip(xs, ys)):
        cf, rf = (x - tr.c) / tr.a - 0.5, (y - tr.f) / tr.e - 0.5  # fractional pixel-centre index
        c0, r0 = int(np.rint(cf)), int(np.rint(rf))
        dc, dr = np.meshgrid(d, d)
        inside = np.hypot((c0 + dc - cf) * res, (r0 + dr - rf) * res) <= RADIUS_M
        dc, dr = dc[inside], dr[inside]
        rows = r0 + dr[None, :] + off_r[:, None]
        cols = c0 + dc[None, :] + off_c[:, None]
        ok = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
        vals = np.where(ok, v[np.clip(rows, 0, ny - 1), np.clip(cols, 0, nx - 1)], np.nan)
        fin = np.isfinite(vals)
        frac[i] = fin.mean(axis=1)
        good = fin.any(axis=1)
        if not good.any():
            continue
        g = vals[good]
        p = np.nanpercentile(g, [90, 95, 98], axis=1)
        for s, row in zip(['p90', 'p95', 'p98'], p):
            out[s][i, good] = row
        out['max'][i, good] = np.nanmax(g, axis=1)
        out['mean'][i, good] = np.nanmean(g, axis=1)
    return out, frac


def direct_disc(chm, x: float, y: float) -> np.ndarray:
    """Reference for the strided gather: distance mask over the whole CHM."""
    w = 2 * RADIUS_M
    sub = chm.sel(x=slice(x - w, x + w), y=slice(y + w, y - w))
    xs, ys = np.meshgrid(sub.x.values, sub.y.values)
    v = sub.values[np.hypot(xs - x, ys - y) <= RADIUS_M]
    return v[np.isfinite(v)]


def corr_rows(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    Pc = P - P.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    return (Pc @ yc) / np.sqrt((Pc ** 2).sum(axis=1) * (yc ** 2).sum())


def search(P: np.ndarray, y: np.ndarray, track: np.ndarray) -> dict[str, np.ndarray]:
    """Chosen offset index per shot for each level. P: (n_off, n) p98, y: (n,) RH98."""
    n = len(y)
    g = int(np.nanargmax(corr_rows(P, y)))
    b = float((P[g] - y).mean())
    e2 = (P - y - b) ** 2
    trk = np.full(n, g)
    for t in np.unique(track):
        m = track == t
        if m.sum() >= MIN_TRACK:
            trk[m] = int(np.argmin(e2[:, m].mean(axis=1)))
    return {'nominal': np.full(n, K0), 'global': np.full(n, g), 'track': trk, 'shot': np.argmin(e2, axis=0)}


def search_loo(P: np.ndarray, y: np.ndarray, track: np.ndarray) -> dict[str, np.ndarray]:
    """Global and track offsets for each shot, chosen without that shot."""
    n = len(y)
    glob, trk = np.empty(n, int), np.empty(n, int)
    for i in range(n):
        m = np.arange(n) != i
        c = search(P[:, m], y[m], track[m])
        glob[i] = c['global'][0]
        same = track[m] == track[i]
        trk[i] = c['track'][same][0] if same.any() else glob[i]
    return {'global_loo': glob, 'track_loo': trk}


def boot_global(P: np.ndarray, y: np.ndarray, rng) -> dict:
    ks = np.array([int(np.nanargmax(corr_rows(P[:, b], y[b])))
                   for b in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))])
    return {'boot_dx_lo': np.percentile(DX[ks], 2.5), 'boot_dx_hi': np.percentile(DX[ks], 97.5),
            'boot_dy_lo': np.percentile(DY[ks], 2.5), 'boot_dy_hi': np.percentile(DY[ks], 97.5),
            'boot_share_nominal': (ks == K0).mean()}


def r_at(P, y, k) -> float:
    return float(np.corrcoef(P[k, np.arange(len(y))], y)[0, 1])


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    e = pred - y
    return {'n': len(y), 'bias': e.mean(), 'rmse': np.sqrt((e ** 2).mean()),
            'r': np.corrcoef(pred, y)[0, 1] if len(y) > 2 else np.nan,
            'r2': 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum() if len(y) > 2 else np.nan}


def out_dir(year: int) -> Path:
    return GEDI_DIR if year == 2025 else GEDI_DIR / str(year)


def load_footprints(version: str, chms: dict[str, Path], year: int = YEAR) -> pd.DataFrame:
    """Per shot (wide): shot columns + '<stat>_<k>' and 'frac_<k>' for every offset k."""
    shots_path = GEDI_DIR / f'shots_{year}_sites_v{version}.parquet'
    cache = out_dir(year) / f'wiggle_footprints_v{version}.parquet'
    cache.parent.mkdir(exist_ok=True)
    if cache.exists() and cache.stat().st_mtime > shots_path.stat().st_mtime:
        return pd.read_parquet(cache)
    s = pd.read_parquet(shots_path)
    s = s[s.qc_pass & s.site.notna()].reset_index(drop=True)
    s['track'] = s.granule.str.rsplit('_V', n=1).str[0] + '_' + s.beam  # same key in v002 and v003
    parts = []
    for site, f in chms.items():
        d = s[s.site == site].reset_index(drop=True)
        if not len(d):
            continue
        chm = load_chm(f)
        stats, frac = footprints(chm, d.x.to_numpy(), d.y.to_numpy())
        wide = {f'{st}_{k}': stats[st][:, k] for st in STATS for k in range(len(DX))}
        wide |= {f'frac_{k}': frac[:, k] for k in range(len(DX))}
        parts.append(pd.concat([d, pd.DataFrame(wide)], axis=1))
        print(f'[wiggle] v{version} {site}: {len(d)} QC shots, '
              f'{int((frac.min(axis=1) >= MIN_VALID).sum())} with a full disc at every offset', flush=True)
    df = pd.concat(parts, ignore_index=True)
    partial = cache.with_name(cache.stem + '.partial.parquet')
    df.to_parquet(partial, index=False)
    partial.replace(cache)
    return df


def stat_matrix(df: pd.DataFrame, stat: str) -> np.ndarray:
    return df[[f'{stat}_{k}' for k in range(len(DX))]].to_numpy().T  # (n_off, n)


def analyse(version: str, df: pd.DataFrame, rng) -> tuple[list, list, list, list]:
    full = df[df[[f'frac_{k}' for k in range(len(DX))]].min(axis=1) >= MIN_VALID].reset_index(drop=True)
    print(f'[wiggle] v{version}: {len(full)} of {len(df)} QC shots have >= {MIN_VALID:.0%} valid disc at every offset')
    summary, surface, tracks, shots = [], [], [], []
    for stratum, (lo, hi) in STRATA.items():
        d = full[(full.rh98 >= lo) & (full.rh98 <= hi)].reset_index(drop=True)
        if len(d) < 3:
            print(f'[wiggle] v{version} {stratum}: {len(d)} shots, skipped')
            continue
        y, trk = d.rh98.to_numpy(float), d.track.to_numpy()
        P = stat_matrix(d, 'p98')
        r_off = corr_rows(P, y)
        for k in range(len(DX)):
            m = metrics(P[k], y)
            surface.append({'version': version, 'stratum': stratum, 'dx': DX[k], 'dy': DY[k],
                            'n': m['n'], 'r': r_off[k], 'bias': m['bias'], 'rmse': m['rmse']})
        choice = search(P, y, trk) | search_loo(P, y, trk)
        gain = {lv: r_at(P, y, choice[lv]) - r_at(P, y, choice['nominal']) for lv in LEVELS}
        boot = boot_global(P, y, rng)

        site = d.site.to_numpy()
        groups = {'all_sites': np.ones(len(d), bool), 'pindan': d.in_pindan.to_numpy(),
                  'mainland': d.in_mainland.to_numpy()} | {s_: site == s_ for s_ in np.unique(site)}
        g = choice['global'][0]
        for lv in LEVELS:
            idx = choice[lv]
            for st in STATS:
                pred = stat_matrix(d, st)[idx, np.arange(len(d))]
                for gname, gm in groups.items():
                    if gm.sum() < 3:
                        continue
                    summary.append({'version': version, 'stratum': stratum, 'level': lv, 'group': gname,
                                    'stat': st, **metrics(pred[gm], y[gm]),
                                    'global_dx': DX[g], 'global_dy': DY[g],
                                    'r_gain': gain[lv], **boot})
        if stratum == HEADLINE:
            for t in np.unique(trk):
                m_ = trk == t
                k = choice['track'][m_][0]
                tracks.append({'version': version, 'track': t, 'site': ','.join(sorted(set(site[m_]))),
                               'n': int(m_.sum()), 'own_offset': bool(m_.sum() >= MIN_TRACK),
                               'dx': DX[k], 'dy': DY[k]})
            P_n = np.arange(len(d))
            shots.append(pd.DataFrame({
                'version': version, 'shot_number': d.shot_number, 'site': site, 'track': trk,
                'time': d.time, 'beam_type': d.beam_type, 'rh98': y,
                'p98_nominal': P[K0], 'p98_global': P[g], 'p98_track': P[choice['track'], P_n],
                'p98_shot': P[choice['shot'], P_n], 'p98_track_loo': P[choice['track_loo'], P_n],
                'track_dx': DX[choice['track']], 'track_dy': DY[choice['track']],
                'shot_dx': DX[choice['shot']], 'shot_dy': DY[choice['shot']]}))
        print(f'[wiggle] v{version} {stratum}: n {len(d)}, r nominal {r_at(P, y, choice["nominal"]):.3f}, '
              f'global ({DX[g]:+.1f}, {DY[g]:+.1f}) m r {r_at(P, y, choice["global"]):.3f} '
              f'(gain {gain["global"]:+.3f}, LOO {gain["global_loo"]:+.3f}; boot dx '
              f'[{boot["boot_dx_lo"]:+.1f}, {boot["boot_dx_hi"]:+.1f}] dy [{boot["boot_dy_lo"]:+.1f}, '
              f'{boot["boot_dy_hi"]:+.1f}], (0,0) in {boot["boot_share_nominal"]:.1%}), '
              f'track gain {gain["track"]:+.3f} (LOO {gain["track_loo"]:+.3f}), '
              f'shot gain {gain["shot"]:+.3f} (in-sample bound)', flush=True)
    return summary, surface, tracks, shots


def pairs_v002_v003(year: int) -> pd.DataFrame | None:
    p2, p3 = (GEDI_DIR / f'shots_{year}_sites_v{v}.parquet' for v in ('002', '003'))
    if not (p2.exists() and p3.exists()):
        return None
    cols = ['shot_number', 'granule', 'beam', 'site', 'x', 'y', 'rh98', 'qc_pass', 'quality_flag', 'degrade_flag']
    a = pd.read_parquet(p2, columns=cols)
    b = pd.read_parquet(p3, columns=cols + ['quality_flag_rel3'])
    m = a.merge(b, on='shot_number', suffixes=('_v002', '_v003'))
    m['dx'], m['dy'] = m.x_v003 - m.x_v002, m.y_v003 - m.y_v002
    m['shift_m'] = np.hypot(m.dx, m.dy)
    m['rh98_diff'] = m.rh98_v003 - m.rh98_v002
    print(f'[wiggle] v002/v003: {len(a)} / {len(b)} shots, {len(m)} paired on shot_number')
    for name, q in (('all paired', m), ('QC pass in both', m[m.qc_pass_v002 & m.qc_pass_v003])):
        if len(q):
            print(f'[wiggle]   {name}: n {len(q)}, shift median {q.shift_m.median():.2f} m, '
                  f'max {q.shift_m.max():.2f} m; mean dx {q.dx.mean():+.2f}, dy {q.dy.mean():+.2f} m; '
                  f'|rh98 diff| max {q.rh98_diff.abs().max():.3f} m')
    return m


def plot_surface(surf: pd.DataFrame, tracks: pd.DataFrame, out: Path, year: int) -> None:
    vers = sorted(surf.version.unique())
    fig, axes = plt.subplots(1, len(vers), figsize=(5.4 * len(vers), 4.8), constrained_layout=True, squeeze=False)
    n = len(OFFSETS_M)
    for ax, v in zip(axes[0], vers):
        s = surf[(surf.version == v) & (surf.stratum == HEADLINE)].sort_values(['dy', 'dx'])
        grid = s.r.to_numpy().reshape(n, n)
        step = OFFSETS_M[1] - OFFSETS_M[0]
        ext = (OFFSETS_M[0] - step / 2, OFFSETS_M[-1] + step / 2) * 2
        im = ax.imshow(grid, origin='lower', extent=ext, cmap=viz_style.SEQUENTIAL_CMAP)
        t = tracks[(tracks.version == v) & tracks.own_offset]
        ax.scatter(t.dx, t.dy, s=40, facecolor='none', edgecolor=viz_style.TEXT_SECONDARY, linewidth=1.5,
                   label=f'best offset per track ({len(t)} tracks)')
        k = int(s.r.to_numpy().argmax())
        ax.plot(0, 0, marker='+', ms=14, mew=2, color=viz_style.TEXT_PRIMARY, linestyle='none', label='nominal (0, 0)')
        ax.plot(s.dx.iloc[k], s.dy.iloc[k], marker='x', ms=12, mew=2.5, color='#c2410c', linestyle='none',
                label=f'best global ({s.dx.iloc[k]:+.1f}, {s.dy.iloc[k]:+.1f}) m')
        ax.set(xlabel='dx (m, east +)', ylabel='dy (m, north +)', aspect='equal')
        ax.grid(False)
        ax.set_title(f'GEDI02_A v{v}: r(RH98, footprint CHM p98)\nn = {int(s.n.iloc[0])} shots, RH98 3–20 m',
                     fontsize=10)
        ax.legend(loc='lower left', fontsize=7.5, framealpha=0.9)
        fig.colorbar(im, ax=ax, shrink=0.85, label='Pearson r')
    fig.suptitle(f'Geolocation offset search: {year} GEDI against 2026 drone LiDAR (6 sites)',
                 color=viz_style.TEXT_PRIMARY, fontsize=12)
    fig.savefig(out)
    plt.close(fig)
    print(f'[wiggle] wrote {out.resolve()}')


def plot_scatter(shots: pd.DataFrame, summ: pd.DataFrame, out: Path, year: int) -> None:
    vers = sorted(shots.version.unique())
    cols = [('p98_nominal', 'nominal'), ('p98_global', 'global'), ('p98_track_loo', 'track_loo')]
    fig, axes = plt.subplots(len(vers), len(cols), figsize=(4.3 * len(cols), 4.3 * len(vers)),
                             constrained_layout=True, squeeze=False, sharex=True, sharey=True)
    lim = (0, max(shots.rh98.max(), shots[[c for c, _ in cols]].max().max()) * 1.05)
    for row, v in zip(axes, vers):
        d = shots[shots.version == v]
        for ax, (c, lv) in zip(row, cols):
            ax.plot(lim, lim, color=viz_style.TEXT_MUTED, linewidth=1, linestyle='--')
            ax.scatter(d[c], d.rh98, s=14, color=viz_style.color_for('p98'), alpha=0.75, linewidth=0)
            m = summ[(summ.version == v) & (summ.stratum == HEADLINE) & (summ.level == lv)
                     & (summ.group == 'all_sites') & (summ.stat == 'p98')].iloc[0]
            ax.text(0.03, 0.97, f'n {m.n}  bias {m.bias:+.2f} m\nRMSE {m.rmse:.2f} m  r {m.r:.2f}'
                    + ('' if lv == 'nominal' else f'\nr gain over nominal {m.r_gain:+.3f}'),
                    transform=ax.transAxes, va='top', fontsize=8.5, color=viz_style.TEXT_SECONDARY)
            ax.set(xlim=lim, ylim=lim, aspect='equal')
            ax.set_title(f'v{v}, {lv} offset: footprint CHM p98', fontsize=10)
        row[0].set_ylabel('GEDI RH98 (m)')
    for ax in axes[-1]:
        ax.set_xlabel('Footprint CHM p98, 2026 LiDAR (m)')
    fig.suptitle(f'{year} GEDI RH98 against 2026 drone LiDAR, RH98 3–20 m (dashed: 1:1)',
                 color=viz_style.TEXT_PRIMARY, fontsize=12)
    fig.savefig(out)
    plt.close(fig)
    print(f'[wiggle] wrote {out.resolve()}')


def self_test() -> None:
    """Plant shots whose 'RH98' is the true footprint p98 at a known offset; the search must find it."""
    rng = np.random.default_rng(SEED)
    site, f = next((s, f) for s, f in site_chms().items() if s == 'Biota track')
    chm = load_chm(f)
    tr = chm.rio.transform()
    yy, xx = np.nonzero(np.isfinite(chm.values))
    pick = rng.choice(len(xx), 4000, replace=False)
    xs = tr.c + (xx[pick] + rng.uniform(0, 1, len(pick))) * tr.a
    ys = tr.f + (yy[pick] + rng.uniform(0, 1, len(pick))) * tr.e
    stats, frac = footprints(chm, xs, ys)
    keep = np.flatnonzero(frac.min(axis=1) >= 1.0)[:120]
    assert len(keep) >= 60, f'only {len(keep)} planted shots fully inside {site}'
    k_true = int(np.flatnonzero((DX == 10.0) & (DY == -5.0))[0])
    P = stats['p98'][keep].T
    trk = np.repeat(np.arange(len(keep) // 10 + 1), 10)[:len(keep)]
    for label, k, noise in (('offset (+10, -5) m', k_true, 0.0), ('offset (0, 0) + 1 m noise', K0, 1.0)):
        y = P[k] + rng.normal(0, noise, len(keep))
        ch = search(P, y, trk) | search_loo(P, y, trk)
        gain = {lv: r_at(P, y, ch[lv]) - r_at(P, y, ch['nominal']) for lv in ('global', 'global_loo', 'track_loo')}
        boot = boot_global(P, y, rng)
        print(f'[self-test] planted {label}: global ({DX[ch["global"][0]]:+.1f}, {DY[ch["global"][0]]:+.1f}), '
              f'r gain {gain["global"]:+.3f}, LOO {gain["global_loo"]:+.3f}, track LOO {gain["track_loo"]:+.3f}; '
              f'boot dx [{boot["boot_dx_lo"]:+.1f}, {boot["boot_dx_hi"]:+.1f}] '
              f'dy [{boot["boot_dy_lo"]:+.1f}, {boot["boot_dy_hi"]:+.1f}]')
        assert ch['global'][0] == k, 'global offset not recovered'
        assert (ch['global_loo'] == k).all(), 'leave-one-out global offset not recovered for every shot'
        assert boot['boot_dx_lo'] <= DX[k] <= boot['boot_dx_hi'] and boot['boot_dy_lo'] <= DY[k] <= boot['boot_dy_hi']
        if noise == 0:
            assert (ch['track'] == k).all(), 'per-track offsets missed the planted offset'
            assert gain['global_loo'] > 0.1, 'no gain for a planted offset'
        else:
            assert gain['track_loo'] <= 0.02, 'leave-one-out finds a gain where there is none'
    for i in keep[:3]:
        ref = direct_disc(chm, xs[i], ys[i])
        got = stats['p98'][i, K0]
        assert np.isclose(np.percentile(ref, 98), got, atol=0), f'gather p98 {got} != direct {np.percentile(ref, 98)}'
        assert np.isclose(ref.max(), stats['max'][i, K0]) and np.isclose(ref.mean(), stats['mean'][i, K0])
    print(f'[self-test] strided gather = direct distance mask for 3 shots (p98, max, mean; 0 difference)')
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
    od = out_dir(args.year)
    od.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary, surface, tracks, shots = [], [], [], []
    for v in args.versions:
        df = load_footprints(v, chms, args.year)
        a, b, c, d = analyse(v, df, rng)
        summary += a
        surface += b
        tracks += c
        shots += d
        pd.DataFrame(b).to_csv(od / f'wiggle_surface_v{v}.csv', index=False, float_format='%.4f')
        pd.DataFrame(c).to_csv(od / f'wiggle_tracks_v{v}.csv', index=False)
        pd.concat(d).to_csv(od / f'wiggle_shots_v{v}.csv', index=False, float_format='%.3f')
    summ = pd.DataFrame(summary)
    summ.to_csv(od / 'wiggle_summary.csv', index=False, float_format='%.4f')
    pairs = pairs_v002_v003(args.year)
    if pairs is not None:
        pairs.to_csv(od / 'wiggle_v002_v003_pairs.csv', index=False, float_format='%.3f')
    for name in ('wiggle_summary.csv', 'wiggle_v002_v003_pairs.csv'):
        print(f'[wiggle] wrote {(od / name).resolve()}')
    head = summ[(summ.stratum == HEADLINE) & (summ.group == 'all_sites')]
    print(head.pivot_table(index=['version', 'level'], columns='stat', values='rmse').round(3).to_string())

    viz_style.apply_style()
    plot_surface(pd.DataFrame(surface), pd.DataFrame(tracks), od / 'wiggle_surface.png', args.year)
    plot_scatter(pd.concat(shots), summ, od / 'wiggle_scatter.png', args.year)


if __name__ == '__main__':
    main()
