"""Map figure and class areas of the 2-D k = 4 + no canopy map over the Dampier Peninsula (REPORT "Map").

fig_map_2d.png      left: the 5-class map (no canopy = neutral grey; S1-S4 validated all-pairs, light mode);
                    right: one small multiple per class, the class share of each 1 km block (single-hue ramp),
                    class identity in the panel title.
map_areas_2d.csv    km2 and share per class, per boundary (mainland / pindan), never blended.

Run from this directory:
    conda run -n general --no-capture-output python -u map_figure_2d.py
"""
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'chm_regression'))
import viz_style as V  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MAP = HERE / 'outputs' / 'XGB_DP_NISAR_S2_2026_2D_k4_all_nocan.tif'
OUT = HERE / 'outputs' / 'figures'
BOUNDARIES_WIN = {'mainland': r'G:\Australia_shapefile\Dampier_peninsula_mainland.gpkg',
                  'pindan': r'G:\Veg and Soil Mapping\Dampier_pindan_main.gpkg'}
NAMES = ['no canopy', 'S1 short, sparse', 'S2 open, mid height', 'S3 open, tall', 'S4 dense, shorter']
# no canopy: neutral (absence); S1-S4: dataviz slots yellow / green / blue / magenta, all-pairs PASS (light)
COLORS = ['#d6d3ca', '#eda100', '#008300', '#2a78d6', '#e87ba4']
NODATA = 255
BLOCK = 50  # 50 x 20 m = 1 km


def assert_clip_effective(n_px, poly_km2, res, name):
    """Pixels inside the polygon must match its own area (project rule: never assume a clip)."""
    expect = poly_km2 * 1e6 / res ** 2
    assert abs(n_px / expect - 1) < 0.01, f'{name}: {n_px} px vs {expect:.0f} expected from area'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(MAP) as src:
        a, tr, crs, res = src.read(1), src.transform, src.crs, src.res[0]
        b = src.bounds
    valid = a != NODATA

    rows = []
    for name, p in BOUNDARIES_WIN.items():
        g = gpd.read_file(p).to_crs(crs)
        inside = ~geometry_mask(g.geometry, a.shape, tr)
        assert_clip_effective(int(inside.sum()), g.area.sum() / 1e6, res, name)
        m = inside & valid
        cnt = np.bincount(a[m], minlength=len(NAMES))[:len(NAMES)]
        for k, n in enumerate(NAMES):
            rows.append({'boundary': name, 'class': n, 'km2': cnt[k] * res ** 2 / 1e6, 'share': cnt[k] / m.sum()})
        rows.append({'boundary': name, 'class': 'no data (inside boundary)',
                     'km2': (inside & ~valid).sum() * res ** 2 / 1e6, 'share': np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(OUT.parent / 'map_areas_2d.csv', index=False, float_format='%.4f')
    print(t.to_string(index=False, float_format=lambda v: f'{v:,.3f}'))

    # 1 km block shares per class (blocks with < 50 % valid pixels left blank)
    h, w = (a.shape[0] // BLOCK) * BLOCK, (a.shape[1] // BLOCK) * BLOCK
    blk = a[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    nvalid = (blk != NODATA).sum(axis=(1, 3))
    shares = [np.where(nvalid >= BLOCK ** 2 / 2, (blk == k).sum(axis=(1, 3)) / np.maximum(nvalid, 1), np.nan)
              for k in range(len(NAMES))]
    ext_blk = [b.left, b.left + w * res, b.top - h * res, b.top]
    ext = [b.left, b.right, b.bottom, b.top]
    main_share = t[t.boundary == 'mainland'].set_index('class')['share']

    V.apply_style()
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[2.2, 1, 1, 1], wspace=0.08, hspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    cmap = ListedColormap(COLORS)
    cmap.set_bad(V.SURFACE)
    ax.imshow(np.ma.masked_equal(a[::2, ::2], NODATA), cmap=cmap, extent=ext, interpolation='nearest',
              norm=BoundaryNorm(np.arange(-0.5, len(NAMES)), cmap.N))
    gpd.read_file(BOUNDARIES_WIN['mainland']).to_crs(crs).boundary.plot(ax=ax, color=V.TEXT_MUTED, lw=0.4)
    ax.legend(handles=[Patch(color=c, label=f'{n}  ({main_share[n]:.1%})') for c, n in zip(COLORS, NAMES)],
              loc='upper left', bbox_to_anchor=(0, -0.01), ncol=2, frameon=False, fontsize=8,
              title='class (share of mainland)', title_fontsize=8, alignment='left')
    ax.set_title('2-D structure classes, NISAR + S2, 2026 (20 m)', loc='left', fontsize=10, color=V.TEXT_PRIMARY)

    ramp = ListedColormap(plt.get_cmap(V.SEQUENTIAL_CMAP)(np.linspace(0.15, 1.0, 256)))
    ramp.set_bad(V.SURFACE)
    for k, n in enumerate(NAMES):
        sax = fig.add_subplot(gs[k // 3, 1 + k % 3])
        im = sax.imshow(np.ma.masked_invalid(shares[k]), cmap=ramp, vmin=0, vmax=1, extent=ext_blk,
                        interpolation='nearest')
        sax.set_title(n, loc='left', fontsize=9, color=V.TEXT_PRIMARY)
    host = fig.add_subplot(gs[1, 3])
    host.axis('off')
    cax = host.inset_axes([0.05, 0.45, 0.9, 0.06])
    fig.colorbar(im, cax=cax, orientation='horizontal', label='class share of 1 km block')
    for axx in fig.axes:
        if axx not in (cax, host):
            axx.set_xticks([]), axx.set_yticks([])
            axx.set_xlabel(''), axx.set_ylabel('')
            axx.set_aspect('equal')
            for s in axx.spines.values():
                s.set_visible(False)
    fig.text(0.01, 0.01, f'{MAP.name}; EPSG:32751. Model: run (a) of REPORT_structure_2d.md (K-fold kappa 0.613). '
             'No canopy extrapolates (3.4 % of training cells, 88 % Yawuru).', fontsize=7, color=V.TEXT_SECONDARY)
    f = OUT / 'fig_map_2d.png'
    fig.savefig(f, dpi=200, bbox_inches='tight', facecolor=V.SURFACE)
    print('written:', f, OUT.parent / 'map_areas_2d.csv')


if __name__ == '__main__':
    main()
