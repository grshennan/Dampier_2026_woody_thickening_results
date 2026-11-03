"""Stage 1: 2026 feature stack on the NISAR GCOV 20 m grid (PLAN.md).

Grid: G:\\python\\nisar\\geotiff GCOV mosaics, EPSG:32751, 20 m, 11625 x 9081 (the cutbox).
Pixels outside the Dampier mainland polygon are NaN (clip asserted against the polygon area).

Per track (147_D, 168_A), the two passes that bracket the LiDAR flights (5-9 Oct 2026):
Speckle (user, 2026-10-20): Refined Lee 5 x 5 (GDV\\GEDI\\rflee.py) on every acquisition before the two
passes are pooled; decomposition ws 5. The unfiltered stack is kept as feature_stack_2026_nospeckle.tif.
- GCOV: HH_dB, HV_dB, HVHH_dB (= HV_dB - HH_dB), RVI (= 4 HV / (HH + HV)) from the mean of the
  linear gamma0 of the two passes (one pass where the other has no data), each pass filtered by
  G:\\python\\nisar\\gcov_rflee.py (geotiff_rflee5\\). GCOV mask 1..254 = valid.
- GSLC (complex): the native GSLC frames of each pass (EPSG:32751, 10 x 5 m, which nest exactly
  in the 20 m grid: 2 x 4 pixels per cell) -> C2 by GDV\\GEDI\\gslc_c2_composite.add_granule() with
  chm_config's grid rebound to the GCOV grid (nisar_grid()): an exact block mean, no reprojection,
  one C2 per pass (c2_20m_<trk>_<day>.tif) -> Refined Lee 5 x 5 per pass -> look-weighted mean of the
  passes (c2_20m_rflee5_*.tif) -> GDV\\GEDI\\gslc_decomposition.run() (Verma 2024 from C11, C22, C12,
  ws 5, min looks 8 = one pass) -> the 9 bands in DCMP_BANDS, already on the grid.
  (Until 2026-10-20 the C2 came from the 30 m EPSG:3577 per-pass rasters of gslc_pass_dcmp.py,
  reprojected bilinear to 20 m: that stack is kept as feature_stack_2026_gslc30m.tif.)
Every intermediate file is skipped when it is already complete, so a restart redoes only what is missing.
Sentinel-2: the 2026-09-20..10-10 geomedian (s2_geomedian_2026.py), 10 bands averaged 10 -> 20 m,
then NDVI, EVI, BSI from the 20 m bands.

--sar gslc (user, 2026-10-27; run tag 4classTF_GSLC): all SAR from GSLC. The 4 GCOV features per track are
replaced by gHH_dB, gHV_dB, gHVHH_dB, gRVI, the same formulas on C11 / C22 of the filtered, pooled C2
(c2_20m_rflee5_*.tif). That C2 is sigma0 (gslc_c2_composite: beta0 x sin(incidence), ellipsoid), not
terrain-flattened like the GCOV gamma0 (RTC). Valid where n_looks > 0. The 18 dcmp and 13 S2 bands are
the same as the GCOV stack. Output E:\\nisar_2026\\feature_stack_2026_gslc.tif.
One acquisition per stack (user, 2026-10-27): --sar gslc --passes 147_D=20260928 --out
E:\\nisar_2026\\feature_stack_2026_gslc_147D_0928.tif, etc. (26 bands; the same chain on one pass).

Accuracy diagnosis Stage 1.1 (PLAN_stage1_nisar.md, user 2026-10-28: sigma0 for the decompositions, gamma0 for
the backscatter features; GSLC decomposition with phase, GCOV decomposition without phase). Two new modes; the
gcov and gslc modes are unchanged:
- --sar gcov_nophase (33 features): per track HH_dB, HV_dB, HVHH_dB, RVI from the GCOV gamma0 (look-weighted pool
  of the dates, not the plain mean of gcov_track()) + the 6-band GCOV no-phase decomposition of the pooled GCOV
  sigma0 (GDV\\GEDI\\gcov_decomposition.py, bands <trk>_n_<band>) + 13 S2. Output feature_stack_2026_gcovnp.tif.
  Per date: gcov_g0s0_20m_rflee<w>_<trk>_<date>.tif = gamma0 HH / HV (Refined Lee, geotiff_rflee5 reused for
  w 5), sigma0 = gamma0 x rtcGammaToSigmaFactor, factor, numberOfLooks, NESZ HH / HV (linear; the GCOV
  metadata noiseEquivalentBackscatter, bilinear from its 1 km grid to the cell centres, frames first-valid:
  it changes by ~4.6 dB across a frame, so one value per date is not used). Pooled: the same bands,
  look-weighted (gcov_g0s0_20m_rflee<w>_<trk>_<d1>_<d2>.tif).
- --sar gslc_gamma (39 features): per track g0HH_dB, g0HV_dB, g0HVHH_dB, g0RVI from the GSLC gamma0 = pooled
  C2 C11, C22 / the pooled GCOV factor of the same dates and track (weights = the GSLC n_looks of each date;
  checked against dividing each date before pooling, gslc_g0_track()) + the 9-band GSLC decomposition (sigma0
  C2, unchanged) + 13 S2. Output feature_stack_2026_gslcg0.tif. GSLC gamma0 file:
  gslc_g0_20m_rflee<w>_<trk>_<d1>_<d2>.tif.
- --rflee-win, --dcmp-ws (default 5, 5; Stage 1.2): window of the Refined Lee filter and of the decomposition,
  in the file names (c2_20m_rflee<w>_*, dcmp_20m_rflee<w>_ws<ws>_*, gcovdcmp_20m_rflee<w>_ws<ws>_*) and in the
  stack name (_rflee<w>_ws<ws> when not 5, 5). Only for the two new modes.
- --s1 (Stage 1.4, user 2026-10-28: source = the CDSE Sentinel-1 monthly global mosaic, VV / VH linear, 20 m,
  downloaded by diagnosis\\s1_mosaic_2026.py; only September 2026 was published on 2026-10-28): adds S1_VV_dB,
  S1_VH_dB, S1_VHVV_dB, S1_RVI and the 6-band no-phase decomposition (gcov_decomposition.run, C11 = VV,
  C22 = VH, ws = --dcmp-ws, NESZ -20 dB as the script) as S1_n_<band>. The mosaic has no sigma0, so the
  decomposition takes the mosaic values (gamma0 by the CDSE product description; a stated deviation from the plan).
  The MGRS tiles nest in the 20 m grid; they are joined first-valid (average resampling = a copy). Stack name +_s1.

Output (float32, NaN nodata, band descriptions = FEATURES): E:\\nisar_2026\\feature_stack_2026.tif
(+ .json with the feature list and inputs). Written *.partial.tif, then renamed.

Run from this directory:
    conda run -n general --no-capture-output python feature_stack_2026.py
    conda run -n general --no-capture-output python feature_stack_2026.py --check     # QA of the output
    ... --passes 147_D=20260916,20260928 --s2 G:\\python\\odc-stac\\ds_median_DP_2025_07.zarr \\
        --out E:\\nisar_2026\\test_stack.tif     # end-to-end test stack, not for results
"""
import argparse
import json
import re
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import h5py
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from rasterio.windows import Window

sys.path.insert(0, r"G:\python\GDV\GEDI")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chm_regression"))
import chm_config as C  # noqa: E402
import gslc_c2_composite as G2  # noqa: E402
import gcov_decomposition as G4  # noqa: E402
import gslc_decomposition as G3  # noqa: E402
import rflee  # noqa: E402

C.set_region("Dampier")

PASSES = {"147_D": ["20260928", "20261010"], "168_A": ["20260929", "20261011"]}
GCOV_DIR = Path(r"G:\python\nisar\geotiff")
GCOV_RFLEE_DIR = Path(r"G:\python\nisar\geotiff_rflee5")     # gcov_rflee.py: per-date Refined Lee 5 x 5
GRID_REF = GCOV_DIR / "NISAR_GCOV_20260928T095935_t147_D_freqA_HHHH.tif"
WORK = Path(r"E:\nisar_2026")
OUT_TIF = WORK / "feature_stack_2026.tif"
OUT_TIF_GSLC = WORK / "feature_stack_2026_gslc.tif"
S2_ZARR = r"G:\python\odc-stac\ds_geomedian_DP_2026_0920_1010.zarr"
MAINLAND_WIN = r"G:\Australia_shapefile\Dampier_peninsula_mainland.gpkg"

GCOV_FEATS = ["HH_dB", "HV_dB", "HVHH_dB", "RVI"]
GSLC_BS_FEATS = ["gHH_dB", "gHV_dB", "gHVHH_dB", "gRVI"]     # --sar gslc
DCMP_BANDS = ["Pd_db", "Ps_db", "Pu_db", "DpRBI", "DpRSI", "alpha_dp", "H", "DoP", "alpha_cloude"]
S2_BANDS = ["nbart_blue", "nbart_green", "nbart_red", "nbart_red_edge_1", "nbart_red_edge_2",
            "nbart_red_edge_3", "nbart_nir_1", "nbart_nir_2", "nbart_swir_2", "nbart_swir_3"]
S2_INDICES = ["NDVI", "EVI", "BSI"]
GSLC_GRID = "20m_rflee5"     # tag of this stack version (importance record); older stacks "30m", "20m"
DCMP_MIN_LOOKS = 8           # one pass = 2 x 4 native looks per 20 m cell (GCOV also keeps one-pass px)
DCMP_WS = 5                  # decomposition window after the filter (settled project chain, user 2026-10-20)
GSLC_RE = re.compile(r"GSLC_\d{3}_(\d{3}_[AD])_\d{3}_.*?_(\d{8})T\d{6}_")
IMPORTANCE_RECORD = Path(__file__).resolve().parent / "outputs" / "importance_record.csv"

# Stage 1.1 modes (gcov_nophase, gslc_gamma)
GCOV_H5 = Path(r"E:\NISAR\Dampier\GCOV")
NESZ_H5 = "science/LSAR/GCOV/metadata/calibrationInformation/frequencyA/noiseEquivalentBackscatter"
G0S0_BANDS = ["g0_HH", "g0_HV", "s0_HH", "s0_HV", "factor", "n_looks", "nesz_HH", "nesz_HV"]
GSLC_G0_BANDS = ["g0_HH", "g0_HV", "factor", "n_looks"]
GSLC_G0_FEATS = ["g0HH_dB", "g0HV_dB", "g0HVHH_dB", "g0RVI"]
GCOV_DCMP_FEATS = [f"n_{b}" for b in G4.BANDS]
BS_FEATS = {"gcov": GCOV_FEATS, "gslc": GSLC_BS_FEATS, "gcov_nophase": GCOV_FEATS, "gslc_gamma": GSLC_G0_FEATS}
NEW_MODES = {"gcov_nophase": "gcovnp", "gslc_gamma": "gslcg0"}
S1_DIR = WORK / "s1_mosaic"
S1_MONTH = "2026_M09"
S1_NESZ_DB = -20.0              # Dual_pol_powers_decomposition_grd_tif.py (Sentinel-1)
S1_BANDS = ["VV", "VH", "n_looks", "nesz"]
S1_FEATS = ["S1_VV_dB", "S1_VH_dB", "S1_VHVV_dB", "S1_RVI"] + [f"S1_n_{b}" for b in G4.BANDS]


def sp_tag(win):
    return f"rflee{win}x{win}"


def features(passes, sar="gcov", s1=False):
    dc = GCOV_DCMP_FEATS if sar == "gcov_nophase" else DCMP_BANDS
    return ([f"{t}_{f}" for t in passes for f in BS_FEATS[sar]] + [f"{t}_{b}" for t in passes for b in dc]
            + (S1_FEATS if s1 else []) + S2_BANDS + S2_INDICES)


def grid():
    with rasterio.open(GRID_REF) as s:
        return s.transform, s.height, s.width, s.crs


def mainland_mask(tf, h, w):
    """True inside the mainland; the clip is asserted against the polygon area (CLAUDE.md)."""
    geom = gpd.read_file(MAINLAND_WIN).to_crs(32751).union_all()
    inside = ~geometry_mask([geom], out_shape=(h, w), transform=tf)
    ratio = inside.sum() * abs(tf.a * tf.e) / geom.area
    assert 0.99 < ratio < 1.01, f"mainland clip not effective: pixel area / polygon area = {ratio:.4f}"
    print(f"mainland: {inside.sum():,} px, pixel/polygon area {ratio:.4f}", flush=True)
    return inside


def read_on_grid(path, tf, h, w, band=1):
    with rasterio.open(path) as s:
        assert s.transform == tf and (s.height, s.width) == (h, w), f"{path.name}: not on the GCOV grid"
        return s.read(band)


def gcov_track(trk, days, tf, h, w):
    """Mean of the per-date Refined Lee 5 x 5 GCOV (G:\\python\\nisar\\gcov_rflee.py) of the two passes."""
    hh, hv = [], []
    for day in days:
        f = {lay: sorted(GCOV_RFLEE_DIR.glob(f"NISAR_GCOV_{day}T*_t{trk}_freqA_{lay}.tif")) for lay in ("HHHH", "HVHV")}
        f["mask"] = sorted(GCOV_DIR.glob(f"NISAR_GCOV_{day}T*_t{trk}_freqA_mask.tif"))
        assert all(len(v) == 1 for v in f.values()), f"GCOV {trk} {day}: {f}"
        for lay in ("HHHH", "HVHV"):
            with rasterio.open(f[lay][0]) as s:
                assert s.tags().get("speckle") == rflee.TAG, f"{f[lay][0].name}: not filtered"
        m = read_on_grid(f["mask"][0], tf, h, w)
        ok = (m >= 1) & (m <= 254)
        hh.append(np.where(ok, read_on_grid(f["HHHH"][0], tf, h, w), np.nan))
        hv.append(np.where(ok, read_on_grid(f["HVHV"][0], tf, h, w), np.nan))
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN means outside the swath
        n = np.isfinite(np.stack(hh)).sum(0)
        HH, HV = np.nanmean(hh, 0), np.nanmean(hv, 0)
        out = {"HH_dB": 10 * np.log10(HH), "HV_dB": 10 * np.log10(HV)}
        out["HVHH_dB"] = out["HV_dB"] - out["HH_dB"]
        out["RVI"] = 4 * HV / (HH + HV)
    print(f"GCOV {trk} {days}: {(n == 2).sum():,} px with both passes, {(n == 1).sum():,} with one", flush=True)
    return {k: v.astype("float32") for k, v in out.items()}


def gslc_bs_track(trk, days, tf, h, w):
    """--sar gslc: the gcov_track() features from C11 (HH) / C22 (HV) of the filtered, pooled C2 (sigma0)."""
    c2f = WORK / f"c2_20m_rflee5_{trk}_{'_'.join(days)}.tif"
    with rasterio.open(c2f) as s:
        assert s.tags().get("speckle") == rflee.TAG, f"{c2f.name}: not filtered"
        assert s.descriptions[:2] == ("C11", "C22") and s.descriptions[4] == "n_looks", s.descriptions
    a = {b: read_on_grid(c2f, tf, h, w, i) for i, b in ((1, "C11"), (2, "C22"), (5, "n"))}
    ok = (a["n"] > 0) & (a["C11"] != C.NODATA)
    HH, HV = np.where(ok, a["C11"], np.nan), np.where(ok, a["C22"], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"gHH_dB": 10 * np.log10(HH), "gHV_dB": 10 * np.log10(HV)}
        out["gHVHH_dB"] = out["gHV_dB"] - out["gHH_dB"]
        out["gRVI"] = 4 * HV / (HH + HV)
    print(f"GSLC backscatter {trk} {days}: {ok.sum():,} px with looks", flush=True)
    return {k: v.astype("float32") for k, v in out.items()}


@contextmanager
def nisar_grid(tf, h, w, crs):
    """Rebind chm_config's grid globals to the GCOV grid, so that G2.add_granule() bins the native GSLC
    pixels into it and G3.run() masks by the mainland on it (C.boundary_geom keeps EPSG:3577 as its
    default argument, so C.boundary_mask is replaced too). Restored on exit."""
    keep = {k: getattr(C, k) for k in ("GRID_CRS", "GRID_RES", "X0", "X1", "Y0", "Y1", "boundary_mask")}
    assert tf.a == -tf.e and tf.b == tf.d == 0 and tf.a == int(tf.a), f"grid not square / integer: {tf}"
    inside = mainland_mask(tf, h, w)
    C.GRID_CRS, C.GRID_RES, C.X0, C.Y1 = str(crs), int(tf.a), int(tf.c), int(tf.f)
    C.X1, C.Y0 = C.X0 + w * C.GRID_RES, C.Y1 - h * C.GRID_RES
    C.boundary_mask = lambda r0=0, c0=0, hh=None, ww=None: inside[r0:r0 + (hh or h), c0:c0 + (ww or w)]
    try:
        yield
    finally:
        for k, v in keep.items():
            setattr(C, k, v)


def gslc_frames(trk, days):
    """Latest-version GSLC granules of the track's passes; asserts that each nests in the grid."""
    files = [p for p in G2.granules() if (m := GSLC_RE.search(p.name)) and m.group(1) == trk and m.group(2) in days]
    assert {GSLC_RE.search(p.name).group(2) for p in files} == set(days), f"GSLC {trk} {days}: {files}"
    for p in files:
        with h5py.File(p, "r") as f:
            g = f[G2.GRID]
            dx, dy = float(g["xCoordinateSpacing"][()]), float(g["yCoordinateSpacing"][()])
            x0, y0 = float(g["xCoordinates"][0]) - dx / 2, float(g["yCoordinates"][0]) - dy / 2   # pixel edge
            assert int(g["projection"][()]) == 32751 and (x0 - C.X0) % C.GRID_RES == 0 \
                and (y0 - C.Y1) % C.GRID_RES == 0 and C.GRID_RES % dx == 0 and C.GRID_RES % -dy == 0, \
                f"{p.name}: does not nest in the {C.GRID_RES} m grid"
    return files


def c2_prof(tf, h, w, crs):
    return dict(driver="GTiff", width=w, height=h, count=5, dtype="float32", crs=crs, transform=tf, nodata=C.NODATA,
                tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=3, BIGTIFF="YES")


def tags_of(path):
    if not Path(path).exists():
        return {}
    with rasterio.open(path) as s:
        return s.tags()


def c2_pass(trk, day, files, tf, h, w, crs):
    """Native GSLC frames of one pass -> C2 on the GCOV grid (unfiltered; cached per pass: a restart reuses it).
    Call inside nisar_grid(). -> path."""
    out = WORK / f"c2_20m_{trk}_{day}.tif"
    stems = ",".join(p.stem for p in files)
    if tags_of(out).get("granules") == stems:
        print(f"  reuse {out.name}", flush=True)
        return out
    acc, nesz = G2.new_acc(h, w), []
    for p in files:
        t = time.time()
        looks, nz = G2.add_granule(p, acc, (0, 0, h, w), "beta0")       # gslc_pass_dcmp.py default norm
        nesz.append(nz)
        print(f"  {p.name[17:60]}: {looks / 1e6:.1f} M looks, NESZ {nz:.1f} dB, {(time.time() - t) / 60:.1f} min",
              flush=True)
    n = acc["looks"]
    c2 = np.full((5, h, w), C.NODATA, "float32")
    for b, k in enumerate(G2.SUMS):
        np.divide(acc[k], n, out=c2[b], where=n > 0)
    c2[4][n > 0] = n[n > 0]
    del acc
    partial = out.with_name(out.stem + ".partial.tif")
    with rasterio.open(partial, "w", **c2_prof(tf, h, w, crs)) as d:
        d.write(c2)
        d.descriptions = ("C11", "C22", "C12_real", "C12_imag", "n_looks")
        d.update_tags(nesz_db=f"{np.nanmedian(nesz):.3f}", nesz_source="median of the per-frame NISAR metadata",
                      granules=stems, date=day,
                      source="NISAR L2 GSLC freqA HH/HV -> C2 (sigma0), one pass, exact 2 x 4 block mean on the GCOV grid")
    partial.replace(out)
    print(f"  C2 {trk} {day}: n_looks in mainland p1/p50 {np.percentile(n[C.boundary_mask()], [1, 50])}", flush=True)
    return out


def c2_track(trk, days, tf, h, w, crs, win=rflee.WIN, ws=DCMP_WS):
    """Per pass: C2 on the GCOV grid -> Refined Lee win x win (rflee.py) -> look-weighted pool of the filtered
    passes -> decomposition ws. Each file is skipped when it is already complete (restart-safe). -> dcmp path."""
    WORK.mkdir(parents=True, exist_ok=True)
    key, tag = f"{trk}_{'_'.join(days)}", sp_tag(win)
    c2f, dcmp = WORK / f"c2_20m_rflee{win}_{key}.tif", WORK / f"dcmp_20m_rflee{win}_ws{ws}_{key}.tif"
    with nisar_grid(tf, h, w, crs):
        files = gslc_frames(trk, days)
        per = [c2_pass(trk, d, [p for p in files if GSLC_RE.search(p.name).group(2) == d], tf, h, w, crs)
               for d in days]
        src = ",".join(f"{p.name}:{tags_of(p)['granules']}" for p in per)
        t = tags_of(dcmp)
        if t.get("speckle") == tag and t.get("inputs") == src:
            print(f"  reuse {dcmp.name}", flush=True)
            return dcmp
        t = tags_of(c2f)
        if not (t.get("speckle") == tag and t.get("inputs") == src):
            sums, n, nesz = np.zeros((4, h, w), "float64"), np.zeros((h, w), "float64"), []
            for p in per:
                with rasterio.open(p) as s:
                    a = s.read()
                    nesz.append(float(s.tags()["nesz_db"]))
                ok = (a[0] != C.NODATA) & (a[4] > 0)
                ch = [np.where(ok, b, np.nan) for b in a[:4]]
                f = rflee.refined_lee(ch, ch[0] + ch[1], np.where(ok, a[4], 0), win)
                for b in range(4):
                    sums[b] += np.where(ok, f[b].astype("float64") * a[4], 0)
                n += np.where(ok, a[4], 0)
                print(f"  {p.name} filtered: mean ratio C11 {np.nanmean(f[0][ok]) / a[0][ok].mean():.4f} "
                      f"C22 {np.nanmean(f[1][ok]) / a[1][ok].mean():.4f}", flush=True)
                del a, ch, f, ok
            c2 = np.full((5, h, w), C.NODATA, "float32")
            for b in range(4):
                np.divide(sums[b], n, out=c2[b], where=n > 0, casting="unsafe")
            c2[4][n > 0] = n[n > 0]
            del sums
            partial = c2f.with_name(c2f.stem + ".partial.tif")
            with rasterio.open(partial, "w", **c2_prof(tf, h, w, crs)) as d:
                d.write(c2)
                d.descriptions = ("C11", "C22", "C12_real", "C12_imag", "n_looks")
                d.update_tags(nesz_db=f"{np.nanmedian(nesz):.3f}", nesz_source="median of the per-pass C2 tags",
                              speckle=tag, side="centre", window=win, inputs=src,
                              source="per-pass C2 -> rflee.py [C11, C22, C12_real, C12_imag], span C11+C22, looks "
                                     "n_looks -> look-weighted mean of the passes")
            partial.replace(c2f)
            ok = c2[4] > 0
            psd = (c2[0][ok] * c2[1][ok] >= c2[2][ok] ** 2 + c2[3][ok] ** 2).mean()
            print(f"  C2 {key} filtered + pooled: n_looks in mainland p1/p50 "
                  f"{np.percentile(n[C.boundary_mask()], [1, 50])}, PSD {psd:.5f}", flush=True)
            del c2, n, ok
        partial = dcmp.with_name(dcmp.stem + ".partial.tif")
        assert G3.run(c2f, partial, ws=ws, aux=WORK / "dcmp_aux", min_looks=DCMP_MIN_LOOKS,
                      tags=dict(speckle=tag, inputs=src)) is not None, f"{trk}: no dcmp"
        partial.replace(dcmp)
    return dcmp


def one(pattern, folder=GCOV_DIR):
    f = sorted(folder.glob(pattern))
    assert len(f) == 1, f"{folder}\\{pattern}: {f}"
    return f[0]


def nesz_grid(trk, day, tf, h, w, block=1024):
    """GCOV metadata noiseEquivalentBackscatter HH, HV (linear, ~1 km grid per frame) -> bilinear at the cell
    centres of the grid. NaN in the metadata grid is first filled from the nearest valid node; each frame covers
    its own node extent + half a node; frames are joined first-valid in name order (mosaic_frames.py "first").
    -> {"HH": float32 (h, w), "HV": ...}"""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    files = sorted(GCOV_H5.glob(f"NISAR_L2_PR_GCOV_*_{trk}_*_{day}T*.h5"))
    assert len(files) == 2, f"GCOV h5 {trk} {day}: {files}"
    xs, ys = tf.c + tf.a * (np.arange(w) + 0.5), tf.f + tf.e * (np.arange(h) + 0.5)
    out = {p: np.full((h, w), np.nan, "float32") for p in ("HH", "HV")}
    for f in files:
        with h5py.File(f, "r") as hf:
            m = hf[NESZ_H5]
            assert int(m["projection"][()]) == 32751, f.name
            x, y = m["xCoordinates"][:], m["yCoordinates"][:]
            dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
            ci = np.flatnonzero((xs >= x.min() - dx / 2) & (xs <= x.max() + dx / 2))
            ri = np.flatnonzero((ys >= y.min() - dy / 2) & (ys <= y.max() + dy / 2))
            for p in out:
                v = m[p][:].astype("float64")
                bad = ~(np.isfinite(v) & (v > 0))
                if bad.all():
                    continue
                if bad.any():
                    v = v[tuple(distance_transform_edt(bad, return_distances=False, return_indices=True))]
                order = np.argsort(y)
                rgi = RegularGridInterpolator((y[order], x), v[order], bounds_error=False, fill_value=None)
                for b0 in range(0, len(ri), block):
                    rr = ri[b0:b0 + block]
                    Y, X = np.meshgrid(ys[rr], xs[ci], indexing="ij")
                    val = rgi((Y, X)).astype("float32")
                    sub = out[p][rr[0]:rr[-1] + 1, ci[0]:ci[-1] + 1]
                    fill = np.isnan(sub) & (val > 0)
                    sub[fill] = val[fill]
    return out


def gcov_pass_g0s0(trk, day, tf, h, w, crs, win=rflee.WIN):
    """One GCOV date -> gcov_g0s0_20m_rflee<win>_<trk>_<day>.tif (G0S0_BANDS; nodata C.NODATA). gamma0 Refined
    Lee win x win (geotiff_rflee5 reused for win 5: gcov_rflee.py, the same filter and grid), sigma0 = gamma0 x
    rtcGammaToSigmaFactor. Valid = mask 1..254, filtered gamma0 > 0, factor > 0. Skipped when complete. -> path."""
    out, tag = WORK / f"gcov_g0s0_20m_rflee{win}_{trk}_{day}.tif", sp_tag(win)
    if tags_of(out).get("speckle") == tag:
        print(f"  reuse {out.name}", flush=True)
        return out
    t0 = time.time()
    lay = {k: one(f"NISAR_GCOV_{day}T*_t{trk}_freqA_{k}.tif")
           for k in ("HHHH", "HVHV", "mask", "numberOfLooks", "rtcGammaToSigmaFactor")}
    m = read_on_grid(lay["mask"], tf, h, w)
    looks = read_on_grid(lay["numberOfLooks"], tf, h, w)
    fac = read_on_grid(lay["rtcGammaToSigmaFactor"], tf, h, w)
    if win == rflee.WIN:
        g = []
        for k in ("HHHH", "HVHV"):
            p = one(f"NISAR_GCOV_{day}T*_t{trk}_freqA_{k}.tif", GCOV_RFLEE_DIR)
            assert tags_of(p).get("speckle") == tag, f"{p.name}: not {tag}"
            g.append(read_on_grid(p, tf, h, w))
        src = f"{GCOV_RFLEE_DIR} (gcov_rflee.py)"
    else:
        hh, hv = read_on_grid(lay["HHHH"], tf, h, w), read_on_grid(lay["HVHV"], tf, h, w)
        ok = (m >= 1) & (m <= 254) & (hh > 0) & (hv > 0)
        hh, hv = np.where(ok, hh, np.nan), np.where(ok, hv, np.nan)
        g = rflee.refined_lee([hh, hv], hh + hv, np.where(ok, looks, 0), win)
        del hh, hv
        src = f"{GCOV_DIR} -> rflee.py [HHHH, HVHV], span HHHH+HVHV, looks numberOfLooks"
    with np.errstate(invalid="ignore"):
        ok = (m >= 1) & (m <= 254) & (g[0] > 0) & (g[1] > 0) & (fac > 0)
    nz = nesz_grid(trk, day, tf, h, w)
    prof = c2_prof(tf, h, w, crs) | dict(count=len(G0S0_BANDS))
    partial = out.with_name(out.stem + ".partial.tif")
    with rasterio.open(partial, "w", **prof) as d:
        for i, a in enumerate((g[0], g[1], g[0] * fac, g[1] * fac, fac, looks, nz["HH"], nz["HV"]), 1):
            d.write(np.where(ok, a, C.NODATA).astype("float32"), i)
        d.descriptions = tuple(G0S0_BANDS)
        d.update_tags(speckle=tag, window=win, side="centre", source=src, date=day,
                      granules=lay["HHHH"].name, radiometry="g0 = GCOV gamma0 (RTC); s0 = g0 x rtcGammaToSigmaFactor",
                      nesz="noiseEquivalentBackscatter HH / HV (linear, DN^2), bilinear from the 1 km metadata grid to "
                           "the cell centres, frames first-valid; compared with sigma0, as for the GSLC NESZ")
    partial.replace(out)
    nzdb = 10 * np.log10(nz["HH"][ok])
    print(f"  GCOV g0/s0 {trk} {day}: {ok.sum():,} valid px, factor p50 {10 * np.log10(np.median(fac[ok])):+.2f} dB, "
          f"NESZ HH p1/p50/p99 {np.percentile(nzdb, [1, 50, 99]).round(2)} dB, {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return out


def pool(paths, bands, weight, out, tags):
    """Look-weighted mean of the named bands of rasters on one grid (weight = the band `weight` of each); the
    weight band of the output is the sum. Nodata C.NODATA. Written *.partial.tif, then renamed. -> out."""
    with rasterio.open(paths[0]) as s:
        prof, desc = s.profile, s.descriptions
    assert all(b in desc for b in bands) and weight in desc, (desc, bands)
    n = None
    for p in paths:
        with rasterio.open(p) as s:
            assert s.descriptions == desc and s.transform == prof["transform"], p
            wi = s.read(desc.index(weight) + 1).astype("float64")
            wi = np.where((wi != C.NODATA) & (wi > 0), wi, 0)
            n = wi if n is None else n + wi
    partial = out.with_name(out.stem + ".partial.tif")
    with rasterio.open(partial, "w", **(prof | dict(count=len(bands)))) as d:
        for i, b in enumerate(bands, 1):
            if b == weight:
                d.write(np.where(n > 0, n, C.NODATA).astype("float32"), i)
                continue
            acc = np.zeros(n.shape, "float64")
            for p in paths:
                with rasterio.open(p) as s:
                    wi = s.read(desc.index(weight) + 1).astype("float64")
                    x = s.read(desc.index(b) + 1).astype("float64")
                ok = (wi != C.NODATA) & (wi > 0) & (x != C.NODATA)
                acc += np.where(ok, x * wi, 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                d.write(np.where(n > 0, acc / n, C.NODATA).astype("float32"), i)
        d.descriptions = tuple(bands)
        d.update_tags(**tags)
    partial.replace(out)
    return out


def gcov_track_g0s0(trk, days, tf, h, w, crs, win=rflee.WIN):
    """Per date gcov_pass_g0s0() -> look-weighted pool (numberOfLooks) of every band. Skipped when complete."""
    per = [gcov_pass_g0s0(trk, d, tf, h, w, crs, win) for d in days]
    out, tag = WORK / f"gcov_g0s0_20m_rflee{win}_{trk}_{'_'.join(days)}.tif", sp_tag(win)
    src = ",".join(p.name for p in per)
    if len(per) == 1:
        return per[0]
    t = tags_of(out)
    if t.get("speckle") == tag and t.get("inputs") == src:
        print(f"  reuse {out.name}", flush=True)
        return out
    pool(per, G0S0_BANDS, "n_looks", out, dict(speckle=tag, window=win, inputs=src,
         source="per-date gcov_g0s0 -> look-weighted mean (numberOfLooks) of every band, n_looks = sum"))
    print(f"  pooled {out.name}", flush=True)
    return out


def gcov_dcmp_track(trk, days, tf, h, w, crs, win=rflee.WIN, ws=DCMP_WS):
    """GCOV no-phase decomposition (gcov_decomposition.run) of the pooled GCOV sigma0, S_norm over the mainland
    of the grid (nisar_grid). Skipped when complete. -> path."""
    s0 = gcov_track_g0s0(trk, days, tf, h, w, crs, win)
    out = WORK / f"gcovdcmp_20m_rflee{win}_ws{ws}_{trk}_{'_'.join(days)}.tif"
    t = tags_of(out)
    if t.get("speckle") == sp_tag(win) and t.get("source") == str(s0) and t.get("ws") == str(ws):
        print(f"  reuse {out.name}", flush=True)
        return out
    partial = out.with_name(out.stem + ".partial.tif")
    with nisar_grid(tf, h, w, crs):
        assert G4.run(s0, partial, ws=ws, aux=WORK / "dcmp_aux", tags=dict(speckle=sp_tag(win))) is not None, trk
    partial.replace(out)
    return out


def gslc_g0_track(trk, days, tf, h, w, crs, win=rflee.WIN):
    """GSLC gamma0 = pooled filtered C2 C11, C22 / pooled GCOV factor of the same dates and track; the factor is
    pooled with the GSLC n_looks of each date (the C2 pool weights). Where the per-date filtered C2 exist
    (c2_20m_rflee<win>_<trk>_<day>.tif), the result is checked against dividing each date before pooling
    (tags check_*). -> gslc_g0_20m_rflee<win>_<trk>_<days>.tif (GSLC_G0_BANDS)."""
    key, tag = f"{trk}_{'_'.join(days)}", sp_tag(win)
    c2f, out = WORK / f"c2_20m_rflee{win}_{key}.tif", WORK / f"gslc_g0_20m_rflee{win}_{key}.tif"
    assert tags_of(c2f).get("speckle") == tag, f"{c2f.name}: missing or not {tag} (run c2_track first)"
    if tags_of(out).get("speckle") == tag and tags_of(out).get("source") == c2f.name:
        print(f"  reuse {out.name}", flush=True)
        return out
    num, n = np.zeros((h, w), "float64"), np.zeros((h, w), "float64")
    facs, nds = [], []
    for d in days:
        nd = read_on_grid(WORK / f"c2_20m_{trk}_{d}.tif", tf, h, w, 5).astype("float64")    # unfiltered: pool weights
        nd = np.where((nd != C.NODATA) & (nd > 0), nd, 0)
        fd = read_on_grid(one(f"NISAR_GCOV_{d}T*_t{trk}_freqA_rtcGammaToSigmaFactor.tif"), tf, h, w).astype("float64")
        okf = np.isfinite(fd) & (fd > 0) & (nd > 0)
        num += np.where(okf, fd * nd, 0)
        n += np.where(okf, nd, 0)
        facs.append(np.where(okf, fd, np.nan).astype("float32"))
        nds.append(nd.astype("float32"))
    with np.errstate(invalid="ignore", divide="ignore"):
        fac = np.where(n > 0, num / n, np.nan)
    del num, n
    with rasterio.open(c2f) as s:
        c11, c22, nl = (s.read(i).astype("float64") for i in (1, 2, 5))
    ok = (c11 != C.NODATA) & (nl > 0) & np.isfinite(fac)
    g0 = [np.where(ok, c / fac, np.nan) for c in (c11, c22)]
    del c11, c22
    chk = {}
    per = [WORK / f"c2_20m_rflee{win}_{trk}_{d}.tif" for d in days]
    if len(days) > 1 and all(tags_of(p).get("speckle") == tag for p in per):
        for b, name in ((1, "HH"), (2, "HV")):
            num, n = np.zeros((h, w), "float64"), np.zeros((h, w), "float64")
            for p, fd, nd in zip(per, facs, nds):
                cd = read_on_grid(p, tf, h, w, b).astype("float64")
                okd = (cd != C.NODATA) & np.isfinite(fd)
                num += np.where(okd, cd / np.where(okd, fd, 1) * nd, 0)
                n += np.where(okd, nd, 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                gb = np.where(n > 0, num / n, np.nan)
                dd = np.abs(10 * np.log10(g0[b - 1] / gb))
            dd = dd[np.isfinite(dd) & mainland_mask(tf, h, w)]
            chk[f"check_{name}_db_p50_p99_p999_max"] = " ".join(f"{v:.5f}" for v in
                                                                 np.percentile(dd, [50, 99, 99.9, 100]))
            print(f"  GSLC g0 {trk} {name}: pool-then-divide vs divide-then-pool |dB| p50/p99/p99.9/max "
                  f"{chk[f'check_{name}_db_p50_p99_p999_max']}", flush=True)
            assert np.percentile(dd, 99.9) < 0.01, f"{trk} {name}: the two gamma0 pools differ by >= 0.01 dB"
    prof = c2_prof(tf, h, w, crs) | dict(count=len(GSLC_G0_BANDS))
    partial = out.with_name(out.stem + ".partial.tif")
    with rasterio.open(partial, "w", **prof) as d:
        for i, a in enumerate((g0[0], g0[1], fac, nl), 1):
            d.write(np.where(ok, a, C.NODATA).astype("float32"), i)
        d.descriptions = tuple(GSLC_G0_BANDS)
        d.update_tags(speckle=tag, window=win, source=c2f.name, **chk,
                      radiometry="g0 = GSLC sigma0 C2 (beta0 x sin ellipsoid incidence) / GCOV rtcGammaToSigmaFactor",
                      factor="look-weighted (GSLC n_looks per date) mean of the GCOV factor of the same dates")
    partial.replace(out)
    print(f"  GSLC g0 {trk}: {ok.sum():,} valid px -> {out.name}", flush=True)
    return out


def s1_grid(tf, h, w, crs, month=S1_MONTH):
    """The S1 mosaic tiles of one month -> s1_g0_20m_<month>.tif on the grid (S1_BANDS: VV, VH linear; n_looks 1;
    NESZ linear). Tiles joined first-valid in name order. Skipped when complete. -> path."""
    out = WORK / f"s1_g0_20m_{month}.tif"
    tiles = sorted(S1_DIR.glob(f"Sentinel-1_IW_mosaic_{month}_*_VV.tif"))
    assert tiles and all(p.with_name(p.name.replace("_VV.tif", "_VH.tif")).exists() for p in tiles), \
        f"S1 {month}: missing tiles in {S1_DIR}"
    src = ",".join(p.name[:-7] for p in tiles)
    if tags_of(out).get("tiles") == src:
        print(f"  reuse {out.name}", flush=True)
        return out
    a = {b: np.full((h, w), np.nan, "float32") for b in ("VV", "VH")}
    for p in tiles:
        for b in a:
            with rasterio.open(p.with_name(p.name.replace("_VV.tif", f"_{b}.tif"))) as s:
                assert s.res == (20.0, 20.0) and (s.transform.c - tf.c) % 20 == 0 and (s.transform.f - tf.f) % 20 == 0, \
                    f"{p.name}: does not nest in the 20 m grid"
                dst = np.full((h, w), np.nan, "float32")
                reproject(rasterio.band(s, 1), dst, src_nodata=s.nodata, dst_transform=tf, dst_crs=crs,
                          dst_nodata=np.nan, resampling=Resampling.average)
            fill = np.isnan(a[b]) & (dst > 0)
            a[b][fill] = dst[fill]
    ok = np.isfinite(a["VV"]) & np.isfinite(a["VH"])
    prof = c2_prof(tf, h, w, crs) | dict(count=len(S1_BANDS))
    partial = out.with_name(out.stem + ".partial.tif")
    with rasterio.open(partial, "w", **prof) as d:
        for i, x in enumerate((a["VV"], a["VH"], np.ones((h, w), "float32"),
                               np.full((h, w), 10 ** (S1_NESZ_DB / 10), "float32")), 1):
            d.write(np.where(ok, x, C.NODATA).astype("float32"), i)
        d.descriptions = tuple(S1_BANDS)
        d.update_tags(tiles=src, month=month, source="CDSE sentinel-1-global-mosaics (S1SAR_L3_IW_MCM), VV / VH linear",
                      nesz_db=S1_NESZ_DB, n_looks="placeholder 1 (monthly mosaic)")
    partial.replace(out)
    print(f"  S1 {month}: {ok.sum():,} valid px -> {out.name}", flush=True)
    return out


def s1_dcmp(tf, h, w, crs, ws=DCMP_WS, month=S1_MONTH):
    """No-phase decomposition of the S1 mosaic (gcov_decomposition.run, C11 = VV, C22 = VH), S_norm over the
    mainland of the grid. Skipped when complete. -> (grid file, dcmp file)."""
    g = s1_grid(tf, h, w, crs, month)
    out = WORK / f"s1dcmp_20m_ws{ws}_{month}.tif"
    t = tags_of(out)
    if t.get("source") == str(g) and t.get("ws") == str(ws):
        print(f"  reuse {out.name}", flush=True)
        return g, out
    partial = out.with_name(out.stem + ".partial.tif")
    with nisar_grid(tf, h, w, crs):
        assert G4.run(g, partial, ws=ws, aux=WORK / "dcmp_aux", min_looks=0, in_bands=tuple(S1_BANDS),
                      radiometry="S1 monthly mosaic values (gamma0 by the CDSE product description), not sigma0") \
            is not None, "S1: no dcmp"
    partial.replace(out)
    return g, out


def bands_to_grid(path, names, tf, h, w):
    """Named bands of a raster on the grid, C.NODATA -> NaN. -> {name: float32}."""
    out = {}
    with rasterio.open(path) as s:
        assert s.transform == tf and (s.height, s.width) == (h, w), f"{Path(path).name}: not on the GCOV grid"
        for b in names:
            a = s.read(s.descriptions.index(b) + 1)
            out[b] = np.where(a == C.NODATA, np.nan, a).astype("float32")
    return out


def g0_feats(path, names, tf, h, w):
    """Backscatter features (dB, HV - HH dB, RVI) from bands g0_HH, g0_HV (linear gamma0) of path."""
    a = bands_to_grid(path, ["g0_HH", "g0_HV"], tf, h, w)
    HH, HV = (np.where(a[k] > 0, a[k], np.nan) for k in ("g0_HH", "g0_HV"))
    with np.errstate(invalid="ignore", divide="ignore"):
        hh, hv = 10 * np.log10(HH), 10 * np.log10(HV)
        out = dict(zip(names, (hh, hv, hv - hh, 4 * HV / (HH + HV))))
    print(f"gamma0 backscatter {Path(path).name}: {np.isfinite(hh).sum():,} valid px", flush=True)
    return {k: v.astype("float32") for k, v in out.items()}


def dcmp_to_grid(dcmp, tf, h, w):
    out = {}
    with rasterio.open(dcmp) as s:
        assert s.transform == tf and (s.height, s.width) == (h, w), f"{dcmp.name}: not on the GCOV grid"
        for b in DCMP_BANDS:
            a = s.read(G3.BANDS.index(b) + 1)
            out[b] = np.where(a == C.NODATA, np.nan, a).astype("float32")
    return out


def s2_to_grid(tf, h, w, crs, s2_zarr=S2_ZARR):
    ds = xr.open_zarr(s2_zarr)
    ds = ds.rio.write_crs(ds.rio.crs or "EPSG:32751")
    stf = ds.rio.transform()
    out = {}
    for b in S2_BANDS:
        dst = np.full((h, w), np.nan, dtype="float32")
        reproject(ds[b].values.astype("float32"), dst, src_transform=stf, src_crs=ds.rio.crs, src_nodata=np.nan,
                  dst_transform=tf, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
        out[b] = dst
    cnt = np.full((h, w), np.nan, dtype="float32")        # stays NaN for a zarr without `count`
    if "count" in ds:
        reproject(ds["count"].values.astype("float32"), cnt, src_transform=stf, src_crs=ds.rio.crs,
                  dst_transform=tf, dst_crs=crs, resampling=Resampling.min)
    blue, red, nir, swir = out["nbart_blue"], out["nbart_red"], out["nbart_nir_1"], out["nbart_swir_2"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["NDVI"] = (nir - red) / (nir + red)
        out["EVI"] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        out["BSI"] = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue))
    return out, cnt


def build(passes, out_tif, s2_zarr=S2_ZARR, sar="gcov", win=rflee.WIN, ws=DCMP_WS, s1=False):
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    tf, h, w, crs = grid()
    inside = mainland_mask(tf, h, w)
    feats = features(passes, sar, s1)
    partial = out_tif.with_name(out_tif.stem + ".partial.tif")
    prof = dict(driver="GTiff", width=w, height=h, count=len(feats), dtype="float32", crs=crs, transform=tf,
                nodata=np.nan, tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=3,
                BIGTIFF="YES", interleave="band")
    with rasterio.open(partial, "w", **prof) as dst:
        def put(name, arr):
            dst.write(np.where(inside, arr, np.nan).astype("float32"), feats.index(name) + 1)
            dst.set_band_description(feats.index(name) + 1, name)
            print(f"  {name}: {np.isfinite(arr[inside]).mean():.1%} valid in mainland "
                  f"({time.time() - t0:.0f} s)", flush=True)

        for trk, days in passes.items():
            if sar == "gcov_nophase":
                bs = g0_feats(gcov_track_g0s0(trk, days, tf, h, w, crs, win), GCOV_FEATS, tf, h, w)
                dc = bands_to_grid(gcov_dcmp_track(trk, days, tf, h, w, crs, win, ws), G4.BANDS, tf, h, w)
                dc = {f"n_{k}": v for k, v in dc.items()}
            else:
                dcmp = c2_track(trk, days, tf, h, w, crs, win, ws)   # also makes the pooled C2 the GSLC bs reads
                dc = dcmp_to_grid(dcmp, tf, h, w)
                bs = (gcov_track(trk, days, tf, h, w) if sar == "gcov" else
                      gslc_bs_track(trk, days, tf, h, w) if sar == "gslc" else
                      g0_feats(gslc_g0_track(trk, days, tf, h, w, crs, win), GSLC_G0_FEATS, tf, h, w))
            for k, v in bs.items():
                put(f"{trk}_{k}", v)
            for k, v in dc.items():
                put(f"{trk}_{k}", v)
        if s1:
            g, dcmp = s1_dcmp(tf, h, w, crs, ws)
            a = bands_to_grid(g, ["VV", "VH"], tf, h, w)
            VV, VH = (np.where(a[k] > 0, a[k], np.nan) for k in ("VV", "VH"))
            with np.errstate(invalid="ignore", divide="ignore"):
                vv, vh = 10 * np.log10(VV), 10 * np.log10(VH)
                for k, v in zip(S1_FEATS[:4], (vv, vh, vh - vv, 4 * VH / (VV + VH))):
                    put(k, v.astype("float32"))
            for k, v in bands_to_grid(dcmp, G4.BANDS, tf, h, w).items():
                put(f"S1_n_{k}", v)
        s2, cnt = s2_to_grid(tf, h, w, crs, s2_zarr)
        print(f"S2 clear observations in mainland: min {cnt[inside].min():.0f}, "
              f"{(cnt[inside] == 0).mean():.2%} px with none", flush=True)
        for k, v in s2.items():
            put(k, v)
    partial.replace(out_tif)
    ver = GSLC_GRID + {"gcov": "", "gslc": "_gslcbs"}.get(sar, "_" + NEW_MODES.get(sar, ""))
    ver += f"_rflee{win}_ws{ws}" if (win, ws) != (rflee.WIN, DCMP_WS) else ""
    meta = {"features": feats, "passes": passes, "s2": str(s2_zarr), "grid_ref": str(GRID_REF), "sar": sar,
            "gslc_grid": ver}
    if sar in NEW_MODES:
        meta |= {"rflee_win": win, "dcmp_ws": ws}
    if s1:
        meta |= {"s1": S1_MONTH}
        meta["gslc_grid"] += "_s1"
    out_tif.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"written: {out_tif} ({time.time() - t0:.0f} s)", flush=True)


def record_importance(run, model, target, importance, features, importance_type, n_iter, path=OUT_TIF):
    """Append one model's feature importances to outputs/importance_record.csv (long format, one row per
    feature, never overwritten), so that runs can be compared later (user, 2026-10-20)."""
    gslc = json.loads(Path(path).with_suffix(".json").read_text()).get("gslc_grid", "30m")
    imp = pd.Series(np.asarray(importance, "float64"), index=features)
    df = pd.DataFrame({"run": run, "model": model, "target": target, "gslc_grid": gslc, "n_iter": n_iter,
                       "importance_type": importance_type, "feature": imp.index, "importance": imp.values,
                       "rank": imp.rank(ascending=False, method="first").astype(int).values,
                       "written": f"{datetime.now():%Y-%m-%d %H:%M}"})
    df.to_csv(IMPORTANCE_RECORD, mode="a", header=not IMPORTANCE_RECORD.exists(), index=False)


def feature_names(path=OUT_TIF):
    return json.loads(Path(path).with_suffix(".json").read_text())["features"]


def sample(df, path=OUT_TIF):
    """Feature values at df.row / df.col (indices on the GCOV grid), one window per site."""
    feats = feature_names(path)
    X = np.full((len(df), len(feats)), np.nan, dtype="float32")
    with rasterio.open(path) as s:
        for _, g in df.groupby("site"):
            r0, c0 = g["row"].min(), g["col"].min()
            win = Window(c0, r0, g["col"].max() - c0 + 1, g["row"].max() - r0 + 1)
            a = s.read(window=win)
            X[df.index.get_indexer(g.index)] = a[:, g["row"] - r0, g["col"] - c0].T
    return pd.DataFrame(X, columns=feats, index=df.index)


def training_table(path=OUT_TIF, quick=False, need=None, extra=None):
    """labels_2026.parquet rows with every feature finite -> (labels, X). need = the features that must be
    finite (default all); the others may be NaN (classify_2026.py --keep-nan). extra = more label rows (site, row,
    col, mean, ...; classify_2026.py --extra-labels), appended after the labels_2026 rows; adds column `source`
    (Dampier / Extra)."""
    from labels_2026 import TABLE
    lab = pd.read_parquet(TABLE)
    if extra is not None:
        assert not set(extra["site"]) & set(lab["site"]), "extra label sites overlap labels_2026 sites"
        lab = pd.concat([lab.assign(source="Dampier"), extra.assign(source="Extra")], ignore_index=True)
        assert not lab.duplicated(["row", "col"]).any(), "extra label px duplicate a labels_2026 px"
    X = sample(lab, path)
    ok = np.isfinite(X[need or X.columns].to_numpy()).all(1)
    if need:
        print(f"rows with a NaN feature, kept: {int((~np.isfinite(X[ok].to_numpy()).all(1)).sum())}", flush=True)
    print(f"rows with a non-finite {'required ' if need else ''}feature, dropped:",
          lab.loc[~ok, "site"].value_counts().to_dict() or 0, flush=True)
    lab, X = lab[ok], X[ok]
    if quick:                                  # small stratified subset for a fast end-to-end test
        keep = lab.groupby("cls").sample(n=min(300, lab["cls"].value_counts().min()), random_state=42).index
        lab, X = lab.loc[keep], X.loc[keep]
    print(f"training table: {len(lab):,} rows x {X.shape[1]} features", flush=True)
    return lab.reset_index(drop=True), X.reset_index(drop=True)


def predict_raster(models, names, out_tif, dtype="float32", nodata=-9999.0, path=OUT_TIF, block_rows=512,
                   bands=None, need=None):
    """Predict each model over the stack in row blocks -> one band per model (predict_year.predict_stack:
    a pixel is predicted only where every feature is finite). bands = the 1-based stack bands the models
    use, in their feature order (default all). need = 0-based positions in bands that must be finite; if
    given, the other features may be NaN and go to the model as missing (classify_2026.py --keep-nan).
    Written *.partial.tif, then renamed."""
    from predict_year import NODATA, predict_stack
    if need is not None:
        def predict_stack(m, a, n):                            # noqa: F811 -- NaN-tolerant variant
            X = a.reshape(a.shape[0], -1).T
            good = np.isfinite(X[:, need]).all(1)
            flat = np.full(X.shape[0], NODATA, dtype="float32")
            if good.any():
                flat[good] = m.predict(X[good]).astype("float32")
            return flat.reshape(n, -1)
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    partial = out_tif.with_name(out_tif.stem + ".partial.tif")
    with rasterio.open(path) as s:
        prof = dict(driver="GTiff", width=s.width, height=s.height, count=len(models), dtype=dtype, crs=s.crs,
                    transform=s.transform, nodata=nodata, tiled=True, compress="deflate", BIGTIFF="IF_SAFER")
        with rasterio.open(partial, "w", **prof) as d:
            for r0 in range(0, s.height, block_rows):
                n = min(block_rows, s.height - r0)
                a = s.read(bands, window=Window(0, r0, s.width, n))
                for i, m in enumerate(models, 1):
                    p = predict_stack(m, a, n)
                    d.write(np.where(p == NODATA, nodata, p).astype(dtype), i, window=Window(0, r0, s.width, n))
            d.descriptions = tuple(names)
    partial.replace(out_tif)
    print(f"written: {out_tif}", flush=True)


def read_at(tif, df):
    """Values of every band of tif at df.row / df.col -> (n, bands)."""
    with rasterio.open(tif) as s:
        out = np.empty((len(df), s.count), dtype="float64")
        for _, g in df.groupby("site"):
            r0, c0 = g["row"].min(), g["col"].min()
            a = s.read(window=Window(c0, r0, g["col"].max() - c0 + 1, g["row"].max() - r0 + 1))
            out[df.index.get_indexer(g.index)] = a[:, g["row"] - r0, g["col"] - c0].T
    return out


def check(path):
    feats = feature_names(path)
    with rasterio.open(path) as s:
        print(f"{path}: {s.count} bands, {s.height} x {s.width}, {s.crs}, res {s.res}")
        assert list(s.descriptions) == feats, "band descriptions != feature list"
        for i, f in enumerate(feats, 1):
            a = s.read(i, out_shape=(s.height // 8, s.width // 8))          # decimated QA
            v = a[np.isfinite(a)]
            print(f"  {f:22s} valid {v.size / a.size:6.1%}  p1 {np.percentile(v, 1):9.3f}  "
                  f"p50 {np.percentile(v, 50):9.3f}  p99 {np.percentile(v, 99):9.3f}")


def compare_gslc(path, ref=OUT_TIF):
    """--sar gslc stack against the GCOV stack: shared bands bit-identical (S2 only, if the stack's passes are
    not the GCOV stack's); gHH / gHV against HH / HV (Pearson r of dB, largest |r| at zero shift); valid px."""
    fp, fr = feature_names(path), feature_names(ref)
    same = json.loads(Path(path).with_suffix(".json").read_text())["passes"] == PASSES
    shared = [f for f in fp if f in fr and (same or f in S2_BANDS + S2_INDICES)]
    with rasterio.open(path) as s, rasterio.open(ref) as r:
        for f in shared:
            a, b = s.read(fp.index(f) + 1), r.read(fr.index(f) + 1)
            assert np.array_equal(a, b, equal_nan=True), f"{f}: differs from {Path(ref).name}"
        print(f"{len(shared)} shared bands bit-identical to {Path(ref).name}", flush=True)
        for trk in [t for t in PASSES if f"{t}_gHH_dB" in fp]:
            for g, c in (("gHH_dB", "HH_dB"), ("gHV_dB", "HV_dB")):
                a, b = s.read(fp.index(f"{trk}_{g}") + 1), r.read(fr.index(f"{trk}_{c}") + 1)
                print(f"  {trk} valid px: {g} {np.isfinite(a).sum():,}  {c} {np.isfinite(b).sum():,}", flush=True)
                rs = {}
                for dy, dx in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    aa, bb = a[::4], np.roll(b, (dy, dx), (0, 1))[::4]     # every 4th row: enough px
                    ok = np.isfinite(aa) & np.isfinite(bb)
                    rs[(dy, dx)] = np.corrcoef(aa[ok], bb[ok])[0, 1]
                d = np.median(a[np.isfinite(a) & np.isfinite(b)] - b[np.isfinite(a) & np.isfinite(b)])
                print(f"  {trk} {g} vs {c}: r {', '.join(f'{k} {v:.3f}' for k, v in rs.items())}; "
                      f"median dB difference {d:+.2f}", flush=True)
                assert max(rs, key=lambda k: abs(rs[k])) == (0, 0), f"{trk} {g}: largest |r| not at zero shift"


def shift_r(a, b, step=4):
    """Pearson r of a and b at 0 and +-1 px shifts (every step-th row) and the median a - b. -> (rs, median)."""
    rs = {}
    for dy, dx in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        aa, bb = a[::step], np.roll(b, (dy, dx), (0, 1))[::step]
        ok = np.isfinite(aa) & np.isfinite(bb)
        rs[(dy, dx)] = np.corrcoef(aa[ok], bb[ok])[0, 1]
    ok = np.isfinite(a) & np.isfinite(b)
    return rs, float(np.median(a[ok] - b[ok]))


def compare_new(path, ref=OUT_TIF):
    """Stage 1.1 checks of a gcov_nophase / gslc_gamma stack: S2 bands bit-identical to ref; for gslc_gamma, the
    GSLC gamma0 against the GCOV gamma0 of the matching gcov_nophase stack (if it exists): |median| < 0.2 dB,
    largest r at zero shift, valid px counts."""
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    fp, fr = meta["features"], feature_names(ref)
    with rasterio.open(path) as s, rasterio.open(ref) as r:
        for f in S2_BANDS + S2_INDICES:
            assert np.array_equal(s.read(fp.index(f) + 1), r.read(fr.index(f) + 1), equal_nan=True), \
                f"{f}: differs from {Path(ref).name}"
    print(f"13 S2 bands bit-identical to {Path(ref).name}", flush=True)
    other = Path(str(path).replace("_gslcg0", "_gcovnp"))
    if meta["sar"] != "gslc_gamma" or not other.exists():
        return
    fo = feature_names(other)
    with rasterio.open(path) as s, rasterio.open(other) as o:
        for trk in meta["passes"]:
            for g, c in (("g0HH_dB", "HH_dB"), ("g0HV_dB", "HV_dB")):
                a, b = s.read(fp.index(f"{trk}_{g}") + 1), o.read(fo.index(f"{trk}_{c}") + 1)
                rs, d = shift_r(a, b)
                print(f"  {trk} {g} (GSLC) vs {c} (GCOV): valid px {np.isfinite(a).sum():,} / {np.isfinite(b).sum():,} "
                      f"(diff {np.isfinite(a).sum() - np.isfinite(b).sum():+,}); r "
                      f"{', '.join(f'{k} {v:.3f}' for k, v in rs.items())}; median dB difference {d:+.3f}", flush=True)
                assert max(rs, key=lambda k: abs(rs[k])) == (0, 0), f"{trk} {g}: largest |r| not at zero shift"
                if abs(d) >= 0.2:      # user 2026-10-28: the -0.19..-0.20 dB GSLC - GCOV offset is accepted (report §1.1)
                    print(f"  WARNING {trk} {g}: |median| {d:+.3f} dB >= 0.2 (accepted offset, report §1.1)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", nargs="+", default=None, metavar="TRK=DAY1,DAY2")
    ap.add_argument("--out", type=Path, default=None, help="default feature_stack_2026[_gslc].tif")
    ap.add_argument("--check", action="store_true", help="QA of an existing stack")
    ap.add_argument("--s2", default=S2_ZARR, help="S2 geomedian zarr (another one only for a test stack)")
    ap.add_argument("--sar", choices=["gcov", "gslc", *NEW_MODES], default="gcov",
                    help="backscatter features from GCOV (default) or from the GSLC C2 (run tag 4classTF_GSLC); "
                         "gcov_nophase / gslc_gamma: Stage 1.1 (module docstring)")
    ap.add_argument("--rflee-win", type=int, default=rflee.WIN, help="Refined Lee window (new modes only)")
    ap.add_argument("--dcmp-ws", type=int, default=DCMP_WS, help="decomposition window (new modes only)")
    ap.add_argument("--s1", action="store_true", help="add the S1 mosaic features (Stage 1.4; new modes only)")
    a = ap.parse_args()
    if a.sar not in NEW_MODES and ((a.rflee_win, a.dcmp_ws) != (rflee.WIN, DCMP_WS) or a.s1):
        ap.error("--rflee-win / --dcmp-ws / --s1 only with --sar gcov_nophase or gslc_gamma")
    sfx = f"_rflee{a.rflee_win}_ws{a.dcmp_ws}" if (a.rflee_win, a.dcmp_ws) != (rflee.WIN, DCMP_WS) else ""
    sfx += "_s1" if a.s1 else ""
    a.out = a.out or (OUT_TIF if a.sar == "gcov" else OUT_TIF_GSLC if a.sar == "gslc" else
                      WORK / f"feature_stack_2026_{NEW_MODES[a.sar]}{sfx}.tif")
    if not a.check:
        p = dict(x.split("=") for x in a.passes) if a.passes else PASSES
        build({k: v.split(",") if isinstance(v, str) else v for k, v in p.items()}, a.out, a.s2, a.sar,
              a.rflee_win, a.dcmp_ws, a.s1)
    check(a.out)
    if a.sar == "gslc":
        compare_gslc(a.out)
    if a.sar in NEW_MODES:
        compare_new(a.out)
