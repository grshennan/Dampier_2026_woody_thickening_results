"""Stage 2: 2026 LiDAR labels on the NISAR GCOV 20 m grid (PLAN.md).

For each training site, the 50 cm p98 CHM is aggregated onto the GCOV mosaic grid
(EPSG:32751, 20 m, 11625 x 9081) by lidar_2026/chm_class_histograms.mean_raster, which
calls chm_regression/chm_targets.aggregate_chm_to_grid (no fork). Output per site:
outputs/labels/chm_targets_<site>.tif (mean, max, p90, p98, count; nodata -9999).

The training table outputs/labels_2026.parquet has one row per 20 m pixel with
count >= LOW_COUNT_THRESHOLD (30): site, row, col (on the full GCOV grid), x, y, the four
CHM statistics, count, fill (= count / 1600 cells of 50 cm) and cls (0..4 from `mean`).

Sites: 9 Mile, Biota track, Bunda Extra, Country Downs, South Dampier.
Excluded: Bunda Bunda (user decision 2026-10-19), JPP MVT and Quandong MVT (coastal,
README_analysis_scope.md).

Classes on the mean of p98 per pixel (cuts from the 20 % cover fit, see
lidar_2026/chm_class_histograms.py; fitted on a 10 m grid, applied here at 20 m):
    0 Low (bare) < 0.2 | 1 Non-forest 0.2-1.22 | 2 Low forest 1.22-2.35 |
    3 Forest 2.35-5.381 | 4 Tall forest >= 5.381

Run from this directory:
    conda run -n general --no-capture-output python labels_2026.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lidar_2026"))
sys.path.insert(0, str(HERE.parent / "chm_regression"))
from chm_class_histograms import load_chm, mean_raster, to_local  # noqa: E402
from chm_targets import LOW_COUNT_THRESHOLD, STATS  # noqa: E402

ROOT_WIN = r"G:\Dampier\Dampier Lidar 2026"
GRID_REF_WIN = r"G:\python\nisar\geotiff\NISAR_GCOV_20260928T095935_t147_D_freqA_HHHH.tif"
SITES = ["9 Mile", "Biota track", "Bunda Extra", "Country Downs", "South Dampier"]
EXCLUDED = {"Bunda Bunda", "JPP MVT", "Quandong MVT"}
BOUNDS = [0.2, 1.22, 2.35, 5.381]
CLASSES = ["Low", "Non-forest", "Low forest", "Forest", "Tall forest"]
OUT = HERE / "outputs"
LABEL_DIR = OUT / "labels"
TABLE = OUT / "labels_2026.parquet"
TALL_MIN = 100            # gate: report before training if Tall forest has fewer pixels


def site_table(site, tif, ref_tf):
    """One row per labelled pixel; row/col are indices on the full GCOV grid."""
    with rasterio.open(tif) as src:
        a = src.read().astype("float64")
        tf = src.transform
    a[a == -9999.0] = np.nan
    # the site grid must be a whole-pixel window of the reference grid
    c_off, r_off = ~ref_tf * (tf.c, tf.f)
    assert tf.a == ref_tf.a and tf.e == ref_tf.e, f"{site}: pixel size {tf.a} != {ref_tf.a}"
    assert abs(c_off - round(c_off)) < 1e-6 and abs(r_off - round(r_off)) < 1e-6, \
        f"{site}: grid offset ({c_off}, {r_off}) is not a whole pixel"
    count = np.nan_to_num(a[4])
    rr, cc = np.nonzero(count >= LOW_COUNT_THRESHOLD)
    df = pd.DataFrame({s: a[i][rr, cc] for i, s in enumerate(STATS)})
    df.insert(0, "site", site)
    df.insert(1, "row", rr + round(r_off))
    df.insert(2, "col", cc + round(c_off))
    df.insert(3, "x", tf.c + (cc + 0.5) * tf.a)
    df.insert(4, "y", tf.f + (rr + 0.5) * tf.e)
    df["count"] = count[rr, cc].astype("int32")
    df["fill"] = df["count"] / round((tf.a / 0.5) ** 2)
    df["cls"] = np.digitize(df["mean"], BOUNDS).astype("int8")
    n_low = int(((count > 0) & (count < LOW_COUNT_THRESHOLD)).sum())
    print(f"{site}: {len(df)} pixels (count >= {LOW_COUNT_THRESHOLD}), {n_low} low-count dropped, "
          f"{(df['fill'] < 0.8).mean():.1%} of kept are < 80 % filled", flush=True)
    return df


def main():
    assert not EXCLUDED & set(SITES)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    ref = rxr.open_rasterio(to_local(GRID_REF_WIN), masked=True)
    assert ref.rio.crs.to_epsg() == 32751 and ref.rio.resolution()[0] == 20.0, "unexpected GCOV grid"
    parts = []
    for site in SITES:
        chms = sorted(Path(to_local(ROOT_WIN), site).glob("*canopy_p98*.tif"))
        assert len(chms) == 1, f"{site}: expected one CHM, found {chms}"
        tif = mean_raster(load_chm(chms[0]), chms[0], ref, LABEL_DIR)
        parts.append(site_table(site, tif, ref.rio.transform()))
    df = pd.concat(parts, ignore_index=True)
    assert not df.duplicated(["row", "col"]).any(), "sites overlap on the 20 m grid"
    df.to_parquet(TABLE, index=False)

    counts = pd.crosstab(df["site"], df["cls"].map(dict(enumerate(CLASSES))), margins=True)
    print(counts.reindex(columns=CLASSES + ["All"], fill_value=0).to_string(), flush=True)
    print(f"written: {TABLE.resolve()}", flush=True)
    n_tall = int((df["cls"] == 4).sum())
    if n_tall < TALL_MIN:
        print(f"GATE: Tall forest has only {n_tall} pixels (< {TALL_MIN}); report before training.")


if __name__ == "__main__":
    main()
