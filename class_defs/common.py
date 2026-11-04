"""Shared helpers for class_defs (PLAN: C:\\Users\\glens\\.claude\\plans\\previously-we-ve-stuck-to-abstract-pixel.md).

cover_table(heights): one row per label px (same px as labels_2026.parquet) with mean, count and cover at each
nominal height h (fraction of the px's 50 cm p98 CHM cells >= h). Built by class_vs_regression/translation
.aggregate_with_cover (no fork) and cached in outputs/cover_table.parquet; new heights add columns.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent  # moved up from nisar_2026\ on 2026-11-02
NISAR = PROJ / "nisar_2026"
for p in (NISAR, PROJ / "lidar_2026", PROJ / "chm_regression", PROJ / "class_vs_regression"):
    sys.path.insert(0, str(p))
import labels_2026 as L  # noqa: E402
from chm_class_histograms import load_chm, to_local  # noqa: E402
from chm_targets import LOW_COUNT_THRESHOLD, target_grid_for_chm  # noqa: E402
import translation  # noqa: E402

OUT = HERE / "outputs"
COVER = OUT / "cover_table.parquet"
OLD_CUTS = [0.2, 1.22, 2.35]
CLASSES = ["Bare", "Low veg", "Low forest", "Forest"]


def col(h):
    return f"cover_{round(float(h), 3)}"


def labels():
    lab = pd.read_parquet(L.TABLE)
    assert set(lab.site) == set(L.SITES) and not set(lab.site) & L.EXCLUDED
    return lab


def chm_path(site, variant="filt"):
    """variant 'filt' = the DJI Terra 5 cm filtered CHM (default, classification track); 'unf' = the unfiltered
    re-export CHM in <Site>\\<Site> unf\\ (PLAN_unf_reexport.md, 2026-11-04)."""
    if variant == "unf":
        chms = sorted(Path(to_local(L.ROOT_WIN), site, f"{site} unf").glob("*_unf_*canopy_p98.tif"))
    else:
        assert variant == "filt", variant
        chms = sorted(Path(to_local(L.ROOT_WIN), site).glob("*canopy_p98*.tif"))
    assert len(chms) == 1, f"{site} ({variant}): expected one CHM, found {chms}"
    return chms[0]


def ref_grid():
    ref = rxr.open_rasterio(to_local(L.GRID_REF_WIN), masked=True)
    assert ref.rio.crs.to_epsg() == 32751 and ref.rio.resolution()[0] == 20.0
    return ref


def mhi(h):
    return f"mhi_{round(float(h), 3)}"


def mlo(h):
    return f"mlo_{round(float(h), 3)}"


def _site_cover(site, heights, ref, cond=()):
    chm = load_chm(chm_path(site))
    ds = translation.aggregate_with_cover(chm, target_grid_for_chm(chm, ref), heights=heights, cond_heights=cond)
    df = ds.drop_vars("spatial_ref", errors="ignore").to_dataframe().reset_index()
    df = df[df["count"] >= LOW_COUNT_THRESHOLD]
    tf = ref.rio.transform()
    df["col"] = np.round((df["x"] - tf.c) / tf.a - 0.5).astype("int64")
    df["row"] = np.round((df["y"] - tf.f) / tf.e - 0.5).astype("int64")
    df["site"] = site
    return df[["site", "row", "col", "mean", "count"] + [f"cover_{h}" for h in heights]
              + [f"{s}_{h}" for h in cond for s in ("mhi", "mlo")]]


def cover_table(heights, cond=False):
    """Cached per-px cover table with a cover column for each h (rounded to 1 mm).

    cond=True also adds mhi_<h> / mlo_<h> (mean of the px's 50 cm cells >= h / < h; NaN where none) and asserts
    mean = cover * mhi + (1 - cover) * mlo per px.
    """
    heights = sorted({round(float(h), 3) for h in heights})
    lab = labels()
    tab = pd.read_parquet(COVER) if COVER.exists() else None
    need = [h for h in heights if tab is None or col(h) not in tab]
    need_c = [h for h in heights if cond and (tab is None or mhi(h) not in tab)]
    if need or need_c:
        ref = ref_grid()
        new = pd.concat([_site_cover(s, need, ref, need_c) for s in L.SITES], ignore_index=True)
        new = new.rename(columns={f"cover_{h}": col(h) for h in need})
        new = new.rename(columns={f"{s}_{h}": f"{s}_{round(float(h), 3)}" for h in need_c for s in ("mhi", "mlo")})
        added = [col(h) for h in need] + [f(h) for h in need_c for f in (mhi, mlo)]
        if tab is None:
            tab = new
        else:
            tab = tab.merge(new[["site", "row", "col"] + added], on=["site", "row", "col"],
                            how="left", validate="1:1")
        # same px and same mean as the label table
        m = lab[["site", "row", "col", "mean"]].merge(tab[["site", "row", "col", "mean"]],
                                                     on=["site", "row", "col"], how="outer", indicator=True)
        assert (m["_merge"] == "both").all(), m["_merge"].value_counts()
        assert np.allclose(m["mean_x"], m["mean_y"], atol=1e-4), "cover table mean != label mean"
        assert tab[[col(h) for h in need]].notna().all().all()
        for h in need_c:
            c, hi, lo = tab[col(h)], tab[mhi(h)].fillna(0), tab[mlo(h)].fillna(0)
            assert ((c > 0) == tab[mhi(h)].notna()).all() and ((c < 1) == tab[mlo(h)].notna()).all()
            assert np.allclose(c * hi + (1 - c) * lo, tab["mean"], atol=1e-4), f"mean decomposition fails at h={h}"
        part = COVER.with_name(COVER.stem + ".partial.parquet")
        tab.to_parquet(part, index=False)
        part.replace(COVER)
    return tab


COVER_Y = OUT / "cover_table_yawuru.parquet"


def cover_table_yawuru(heights, cond=True, path=None):
    """Yawuru 2025 (13 areas, chm_targets.LIDAR_PATHS_WIN) on the NISAR 20 m grid: same aggregate_with_cover path
    as _site_cover; count >= LOW_COUNT_THRESHOLD; cells whose centre is in the area polygon; one row per cell
    (overlap: keep the area with more 50 cm cells). Cached in `path` (default outputs/cover_table_yawuru.parquet;
    rebuilt if a height is missing). Asserts the mean decomposition and the clip (cells x 400 m2 vs polygon area,
    +-10 %).
    """
    import geopandas as gpd
    from chm_targets import AREA_NAMES, LIDAR_PATHS_WIN, LIDAR_POLYS_WIN
    COVER_Y = Path(path) if path else globals()["COVER_Y"]  # noqa: N806
    heights = sorted({round(float(h), 3) for h in heights})
    want = [col(h) for h in heights] + ([f(h) for h in heights for f in (mhi, mlo)] if cond else [])
    if COVER_Y.exists():
        tab = pd.read_parquet(COVER_Y)
        if all(c in tab for c in want):
            return tab
    ref = ref_grid()
    parts = []
    for name, cp, pp in zip(AREA_NAMES, LIDAR_PATHS_WIN, LIDAR_POLYS_WIN):
        chm = load_chm(Path(to_local(cp)))
        ds = translation.aggregate_with_cover(chm, target_grid_for_chm(chm, ref), heights=heights,
                                              cond_heights=heights if cond else ())
        df = ds.drop_vars("spatial_ref", errors="ignore").to_dataframe().reset_index()
        df = df[df["count"] >= LOW_COUNT_THRESHOLD]
        poly = gpd.read_file(to_local(pp))
        epsg = poly.crs.to_epsg()
        # ⚠ CRS 2026-11-03: relabel withdrawn (user); reprojection is required (G:\python\crs_audit\CHECKPOINT.md)
        poly = poly.set_crs(32751, allow_override=True) if epsg == 7851 else poly.to_crs(32751)  # as load_chm
        inside = gpd.points_from_xy(df["x"], df["y"]).within(poly.union_all())
        df = df[inside].copy()
        area, n = float(poly.area.sum()), len(df)
        ratio = n * 400.0 / area
        print(f"[yawuru] {name}: {n:,} cells, cells*400/poly = {ratio:.3f} (poly EPSG:{epsg})", flush=True)
        assert 0.9 <= ratio <= 1.1, f"{name}: clip not effective ({ratio:.3f})"
        tf = ref.rio.transform()
        df["col"] = np.round((df["x"] - tf.c) / tf.a - 0.5).astype("int64")
        df["row"] = np.round((df["y"] - tf.f) / tf.e - 0.5).astype("int64")
        df["site"] = name
        parts.append(df)
    tab = pd.concat(parts, ignore_index=True)
    tab = tab.rename(columns={f"cover_{h}": col(h) for h in heights})
    tab = tab.rename(columns={f"{s}_{h}": f"{s}_{round(float(h), 3)}" for h in heights for s in ("mhi", "mlo")})
    dup = tab.duplicated(["row", "col"], keep=False)
    print(f"[yawuru] {int(dup.sum())} rows in two areas ({', '.join(sorted(tab.site[dup].unique()))}); "
          f"keep the row with more 50 cm cells", flush=True)
    tab = tab.sort_values("count", ascending=False).drop_duplicates(["row", "col"]).sort_values(["site", "row", "col"])
    if cond:
        for h in heights:
            c, hi, lo = tab[col(h)], tab[mhi(h)].fillna(0), tab[mlo(h)].fillna(0)
            assert ((c > 0) == tab[mhi(h)].notna()).all() and ((c < 1) == tab[mlo(h)].notna()).all()
            assert np.allclose(c * hi + (1 - c) * lo, tab["mean"], atol=1e-4), f"Yawuru decomposition fails at h={h}"
    tab = tab[["site", "row", "col", "x", "y", "mean", "count"] + want].reset_index(drop=True)
    part = COVER_Y.with_name(COVER_Y.stem + ".partial.parquet")
    tab.to_parquet(part, index=False)
    part.replace(COVER_Y)
    return tab
