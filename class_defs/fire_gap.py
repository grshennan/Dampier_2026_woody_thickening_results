"""Part A (PLAN_fire_gap.md): fires between the LiDAR flight and the NISAR stack, Yawuru 2025 (13 areas) + Dampier 2026.

Sub-commands (run in order; outputs in outputs/fire_gap/):
    dates   flight date per area from point gps_time (seek-sampled, LAS header adjusted-GPS bit) -> flight_dates.csv
    cells   label cells on the NISAR 20 m grid (common.cover_table_yawuru, h 1.37) -> cells.parquet
    nafi    NAFI 2025 + 2026 scars per cell, classed vs flight date F and gap end G -> fire_cells.parquet
    s2      S2 NBR drop between F and G (pointer only) -> s2_cells.parquet
    effect  counts per area; features at the Yawuru cells (-> outputs/features_yawuru_s1.parquet); feature
            offsets gap - not burnt at fixed structure (cover_1.37 x mhi_1.37 bins)
    fit     pooled RF (Dampier + Yawuru, 49 features) for cover_1.37 and mhi_1.37, 5 folds on site x cover
            quintile (seed 42); configs: all / no_gap / dampier_only (no_gap_amb = no_gap: no ambiguous
            cells); out-of-fold residuals by fire class; R2 / RMSE per source. Detach (~20-30 min).
    report  REPORT_fire_gap.txt + fig_fire_gap.png

conda.exe run -n general --no-capture-output python -u fire_gap.py <cmd>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

OUT = C.OUT / "fire_gap"
OUT.mkdir(parents=True, exist_ok=True)
H = 1.37
G_END = pd.Timestamp("2026-10-11")       # last NISAR date (168 A)
IMG_START = pd.Timestamp("2026-09-20")   # first image date (S2 geomedian)
N_JOBS = 10                              # user 2026-11-02

YW = r"G:\Yawuru\Drone Data 2025"
YAW_LAS = {  # structure_2d/PLAN.md source table
    "YA25_1": r"Yawuru_Lidar_Apr_2025\YA25_1\YF25_1_height.las",
    "YA25_2": r"Yawuru_Lidar_Apr_2025\YA25_2\YA25_2_height_clean.las",
    "YA25_3": r"Yawuru_Lidar_Apr_2025\YA25_3\YA25_3_height_thin.las",
    "YA25_4": r"Yawuru_Lidar_Apr_2025\YA25_4\YF25_4_height.las",
    "YA25_5": r"Yawuru_Lidar_Apr_2025\YA25_5\YA25_5_height.las",
    "20250407_govcowwell": r"20250407_govcowwell\reprocess\20250407_govcowwell_height_clean.las",
    **{f"YJ25_{i}": rf"Yawuru_Lidar_July_2025\YJ25_{i}\YJ25_{i}_height_clean.las" for i in range(1, 7)},
    "YJ25_3": r"Yawuru_Lidar_July_2025\YJ25_3\YJ25_3.las",  # _height_clean is a copy of YJ25_4 (structure_2d REPORT)
    "YOct25_1": r"Yawuru_Lidar_Oct_2025\YOct25_1\YOct25_1_height_clean.las",
}
FOLDER_MONTH = {"YA": 4, "20": 4, "YJ": 7, "YO": 10}
DAMPIER_LAS = {
    "9 Mile": "9_mile_ground_gndthin_height_denoise.las",
    "Biota track": "biota_track_ground_gndthin_height_denoise.las",
    "Bunda Extra": "bunda_extra_ground_gndthin_height_denoise.las",
    "Country Downs": "country_downs_ground_gndthin_height_denoise.las",
    "South Dampier": "south_dampier_ground_gndthin_height_denoise.las",
}
GPS_EPOCH = pd.Timestamp("1980-01-06")
LEAP = 18


def las_dates(path, n_seek=400, n_pts=2000):
    """UTC times of n_seek x n_pts points spread evenly through the file (no full read)."""
    import laspy
    with laspy.open(path) as f:
        h = f.header
        adjusted = bool(int(h.global_encoding.value) & 1) if hasattr(h.global_encoding, "value") \
            else bool(h.global_encoding.gps_time_type)
        n = h.point_count
        t = []
        for i in np.linspace(0, max(n - n_pts, 0), n_seek).astype(np.int64):
            f.seek(int(i))
            t.append(np.asarray(f.read_points(n_pts).gps_time))
    t = np.concatenate(t)
    if not adjusted:
        return None, n, t
    utc = GPS_EPOCH + pd.to_timedelta(t + 1e9 - LEAP, unit="s")
    return pd.DatetimeIndex(utc), n, t


def cmd_dates(_):
    rows = []
    items = [(k, "Yawuru 2025", Path(YW) / v) for k, v in YAW_LAS.items()]
    items += [(k, "Dampier 2026", Path(C.L.ROOT_WIN) / k / v) for k, v in DAMPIER_LAS.items()]
    for site, src, p in items:
        utc, n, t = las_dates(p)
        r = dict(site=site, source=src, las=str(p), n_points=n)
        if utc is None:  # week-seconds: fall back to the folder / file-name date
            r.update(adjusted=False, flag="week-seconds gps_time: folder date")
            print(f"{site}: week-seconds gps_time ({t.min():.0f}..{t.max():.0f})", flush=True)
        else:
            d = utc.normalize()
            r.update(adjusted=True, first=utc.min(), last=utc.max(), mode=d.value_counts().idxmax(),
                     n_days=d.nunique(), flag="")
            print(f"{site}: {utc.min()} .. {utc.max()} UTC, mode {r['mode'].date()}, {r['n_days']} day(s)", flush=True)
            if src == "Yawuru 2025":
                m = FOLDER_MONTH[site[:2]] if site != "20250407_govcowwell" else 3
                assert r["mode"].year == 2025 and r["mode"].month == m, f"{site}: month {r['mode']} != folder {m}"
                if site == "20250407_govcowwell" and r["mode"] != pd.Timestamp("2025-04-07"):
                    r["flag"] = f"gps_time {r['mode'].date()} != name 2025-04-07; gps date used"
            else:
                assert pd.Timestamp("2026-10-05") <= r["mode"] <= pd.Timestamp("2026-10-09"), site
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "flight_dates.csv", index=False)
    print(df.drop(columns="las").to_string(index=False))


def cmd_cells(_):
    y = C.cover_table_yawuru([H], cond=True)
    y["source"] = "Yawuru 2025"
    d = C.cover_table([H], cond=True)
    d["source"] = "Dampier 2026"
    keep = ["source", "site", "row", "col", "mean", "count", C.col(H), C.mhi(H), C.mlo(H)]
    cells = pd.concat([d[keep], y[keep]], ignore_index=True)
    assert not cells.duplicated(["row", "col"]).any(), "Dampier and Yawuru cells overlap"
    cells.to_parquet(OUT / "cells.parquet", index=False)
    print(cells.groupby(["source", "site"]).size().to_string())
    print(f"Yawuru total {int((cells.source == 'Yawuru 2025').sum()):,} (structure_2d: 8,419 at fill >= 0.9)")


SCARS = Path(r"G:\python\fire_class_forest_stats\NAFI\fire scars")
EDGE_M = 250.0


def cell_xy(cells):
    tf = C.ref_grid().rio.transform()
    return tf.c + (cells["col"] + 0.5) * tf.a, tf.f + (cells["row"] + 0.5) * tf.e


def flight(cells):
    fd = pd.read_csv(OUT / "flight_dates.csv", parse_dates=["first", "last", "mode"])
    return cells.merge(fd[["site", "first", "last"]], on="site", how="left", validate="m:1")


def _date(s):
    """'YYYY/MM/DD' with day clipped to the month (NAFI has 2025/09/31)."""
    y, m, d = map(int, s.split("/"))
    return pd.Timestamp(y, m, 1) + pd.Timedelta(days=min(d, pd.Timestamp(y, m, 1).days_in_month) - 1)


def read_scars(bbox_ll):
    import geopandas as gpd
    import pyogrio
    out = []
    for p in sorted(SCARS.glob("20*/*.shp")):
        g = pyogrio.read_dataframe(p, bbox=bbox_ll)
        pc = next(c for c in g.columns if c.lower() in ("map_period", "mapping_pe"))
        mc = next(c for c in g.columns if c.lower() == "month")
        year = int(p.parent.name[:4])
        # older files hold free text ("(a) 2018/05/26-2018/05/31  (b) ..."): start = min, end = max of the dates
        tok = g[pc].fillna("").astype(str).str.findall(r"\d{4}/\d\d/\d\d")
        start = tok.map(lambda d: min(map(_date, d)) if isinstance(d, list) and d else pd.NaT)
        end = tok.map(lambda d: max(map(_date, d)) if isinstance(d, list) and d else pd.NaT)
        mon = pd.to_datetime(dict(year=year, month=g[mc].fillna(1).astype(int).clip(1, 12), day=1))
        start = start.fillna(mon)
        end = end.fillna(mon + pd.offsets.MonthEnd(0))
        g = gpd.GeoDataFrame({"start": start, "end": end + pd.Timedelta(hours=23, minutes=59), "file": p.name},
                             geometry=g.geometry, crs=g.crs)
        g = g.dissolve(by=["start", "end", "file"]).reset_index()
        out.append(g.to_crs(32751))
    s = pd.concat(out, ignore_index=True)
    print(f"[nafi] {len(s)} scar periods in the bbox ({', '.join(f.name for f in sorted(SCARS.glob('20*/*.shp')))})")
    return gpd.GeoDataFrame(s, crs=32751)


def cmd_nafi(_):
    import geopandas as gpd
    cells = flight(pd.read_parquet(OUT / "cells.parquet"))
    cells["x"], cells["y"] = cell_xy(cells)
    pts = gpd.GeoDataFrame(cells[["source", "site", "row", "col"]],
                           geometry=gpd.points_from_xy(cells.x, cells.y), crs=32751)
    bb = pts.to_crs(4283).total_bounds + np.array([-0.01, -0.01, 0.01, 0.01])  # ~1 km
    scars = read_scars(tuple(bb))
    # window (w0, w1]: Yawuru flight -> last NISAR date; Dampier first image -> last NISAR date (flight inside)
    yaw = cells.source == "Yawuru 2025"
    cells["w0"] = cells["last"].where(yaw, IMG_START)
    cells["w1"] = G_END + pd.Timedelta(hours=23, minutes=59)
    j = gpd.sjoin(pts.reset_index(), scars, predicate="within")[["index", "start", "end"]]
    j = j.merge(cells[["w0", "w1", "first"]], left_on="index", right_index=True)
    j["cls"] = np.select([j.end <= j.w0, (j.start > j.w0) & (j.end <= j.w1), j.start > j.w1],
                         ["before", "gap", "after"], "ambiguous")
    g = j[j.cls == "gap"].groupby("index")
    cells["n_gap_fires"] = g.size().reindex(cells.index, fill_value=0)
    cells["first_gap_start"] = g.start.min().reindex(cells.index)
    cells["first_gap_end"] = g.end.min().reindex(cells.index)
    cells["ambiguous"] = j[j.cls == "ambiguous"].groupby("index").size().reindex(cells.index, fill_value=0) > 0
    before = j[(j.cls == "before") & (j.end < j["first"])]
    cells["last_before"] = before.groupby("index").end.max().reindex(cells.index)
    cells["fire"] = np.select([cells.n_gap_fires > 0, cells.ambiguous], ["gap", "ambiguous"], "none")
    # distance to the nearest edge of the scars inside each area's own window (NAFI edge uncertainty)
    cells["edge_dist"] = np.inf
    for site, idx in cells.groupby("site").groups.items():
        w0, w1 = cells.loc[idx[0], "w0"], cells.loc[idx[0], "w1"]
        near = scars[(scars.end > w0) & (scars.start <= w1)]
        if len(near):
            cells.loc[idx, "edge_dist"] = pts.loc[idx].distance(near.union_all().boundary).to_numpy()
    cells["edge"] = cells.edge_dist < EDGE_M
    cells["gap_month"] = cells.first_gap_end.dt.month
    keep = ["source", "site", "row", "col", "x", "y", "first", "last", "w0", "w1", "n_gap_fires", "first_gap_start",
            "first_gap_end", "gap_month", "ambiguous", "fire", "edge_dist", "edge", "last_before"]
    cells[keep].to_parquet(OUT / "fire_cells.parquet", index=False)
    j.merge(cells[["site", "row", "col"]], left_on="index", right_index=True).to_parquet(OUT / "fire_scar_hits.parquet")
    t = cells.groupby(["source", "site"]).agg(n=("row", "size"), F=("last", "first"),
                                              gap=("fire", lambda s: (s == "gap").mean()),
                                              amb=("fire", lambda s: (s == "ambiguous").mean()),
                                              edge=("edge", "mean"), last_before=("last_before", "median"))
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    jj = j.merge(cells[["site"]], left_on="index", right_index=True).reset_index(drop=True)
    jj = jj[jj.cls != "before"].assign(s=lambda d: d.start.dt.date, e=lambda d: d.end.dt.date)
    per = jj.groupby(["site", "cls", "s", "e"]).size().rename("cells")
    per.reset_index().to_csv(OUT / "gap_periods.csv", index=False)
    print("\nScar periods after the window start, per area (cells):\n" + per.to_string())
    # hand check: scar periods for one gap cell and one unburnt cell
    for f in ("gap", "none"):
        sel = cells[(cells.fire == f) & (cells.source == "Yawuru 2025")]
        if len(sel):
            i = sel.index[len(sel) // 2]
            print(f"\n[check] {f} cell {cells.loc[i, ['site', 'row', 'col']].to_dict()}, F {cells.loc[i, 'last']}")
            print(j[j["index"] == i][["start", "end", "cls"]].sort_values("start").to_string(index=False))


S2_DIR = OUT / "s2"
S2_START, S2_END = "2025-03-01", "2026-10-21"
SEASONS = [("2025-05-01", "2025-10-31"), ("2026-05-01", S2_END)]  # dry-season search only (s2_nbr_series rule)


class RateCap:
    """Sleep while the bytes received (system-wide counter: conservative) run ahead of rate x elapsed."""

    def __init__(self, mb_s):
        import psutil
        import time
        self.ps, self.time, self.rate = psutil, time, mb_s * 1e6
        self.b0, self.t0 = psutil.net_io_counters().bytes_recv, time.time()

    def wait(self):
        while True:
            got = self.ps.net_io_counters().bytes_recv - self.b0
            ahead = got / self.rate - (self.time.time() - self.t0)
            if ahead <= 0:
                return got
            self.time.sleep(min(ahead, 5))


def s2_load(name, rows, cols, cap):
    """NBR on the NISAR 20 m window of one area (method of lidar_2026/s2_nbr_series.load, rate-capped)."""
    import xarray as xr
    out = S2_DIR / f"s2_nbr_{name}.zarr"
    if out.exists():
        return xr.open_zarr(out).nbr.load()
    import dask
    import odc.stac
    import pystac_client
    from odc.geo.geobox import GeoBox
    from rasterio.transform import Affine
    sys.path.insert(0, str(C.PROJ / "lidar_2026"))
    import s2_nbr_series as S
    tf = C.ref_grid().rio.transform()
    r0, c0 = int(rows.min()) - 2, int(cols.min()) - 2
    shape = (int(rows.max()) - r0 + 3, int(cols.max()) - c0 + 3)
    gbox = GeoBox(shape, tf * Affine.translation(c0, r0), "EPSG:32751")
    bb = gbox.extent.to_crs("EPSG:4326").boundingbox
    items = pystac_client.Client.open(S.STAC_URL).search(
        collections=S.COLLECTIONS, bbox=[bb.left, bb.bottom, bb.right, bb.top],
        datetime=f"{S2_START}/{S2_END}").item_collection()
    ds = odc.stac.load(items, bands=S.BANDS, geobox=gbox, groupby="solar_day", chunks={"time": 8},
                       resampling={"oa_fmask": "nearest", "*": "average"}, patch_url=lambda u: u.replace(*S.S3))
    slabs = []
    for i in range(0, ds.sizes["time"], 8):
        with dask.config.set(scheduler="threads", num_workers=2):
            slabs.append(ds.isel(time=slice(i, i + 8)).compute())
        got = cap.wait()
    ds = xr.concat(slabs, "time")
    nir, swir = ds.nbart_nir_2.astype("float32"), ds.nbart_swir_3.astype("float32")
    bad = ds.oa_fmask.isin(S.FMASK_BAD) | (nir <= 0) | (swir <= 0)
    nbr = ((nir - swir) / (nir + swir)).where(~bad).rename("nbr")
    nbr = nbr.assign_coords(row0=r0, col0=c0)
    print(f"[s2] {name}: {len(items)} items, {nbr.sizes['time']} days, {100 * float(np.isfinite(nbr).mean()):.0f} % clear, "
          f"{got / 1e9:.2f} GB received so far", flush=True)
    part = S2_DIR / f"s2_nbr_{name}.partial.zarr"
    nbr.to_dataset().drop_vars("spatial_ref", errors="ignore").to_zarr(part, mode="w")
    part.rename(out)
    return nbr


def s2_step(v, t, w0, w1):
    """Largest robust drop (3 before / 3 after) with the first post-drop date in (w0, w1] and in a dry season."""
    ok = np.isfinite(v)
    v, t = v[ok], t[ok]
    best, d0, d1 = np.nan, pd.NaT, pd.NaT
    for k in range(3, len(v) - 2):
        if not (w0 < t[k] <= w1 and any(pd.Timestamp(a) <= t[k] <= pd.Timestamp(b) for a, b in SEASONS)):
            continue
        s = np.median(v[k - 3:k]) - np.median(v[k:k + 3])
        if not best >= s:
            best, d0, d1 = s, t[k - 1], t[k]
    return best, d0, d1


def cmd_s2(a):
    import xarray as xr
    S2_DIR.mkdir(exist_ok=True)
    fc = pd.read_parquet(OUT / "fire_cells.parquet")
    cap = RateCap(a.rate)
    res = []
    # Yawuru: own windows, baseline = median of all Yawuru cells per date
    yaw = fc[fc.source == "Yawuru 2025"]
    series = {}
    for name, g in yaw.groupby("site"):
        nbr = s2_load(name, g.row, g.col, cap)
        r0, c0 = int(nbr.row0), int(nbr.col0)
        t = pd.DatetimeIndex(nbr.time.values).normalize()
        series[name] = (g, pd.DataFrame(nbr.values[:, g.row.values - r0, g.col.values - c0], index=t))
    allv = pd.concat([s for _, s in series.values()], axis=1)
    base = allv.median(axis=1).where(allv.notna().sum(axis=1) >= 1000)
    base.rename("nbr_median").to_csv(OUT / "s2_baseline_yawuru.csv")
    # Dampier: reuse lidar_2026 zarrs (read only) and their 5-site baseline
    sys.path.insert(0, str(C.PROJ / "lidar_2026"))
    import rasterio
    import s2_nbr_series as S
    dbase = pd.read_csv(S.FIRE_DIR / "s2_nbr_baseline.csv", index_col=0, parse_dates=True).iloc[:, 0]
    tf_ref = C.ref_grid().rio.transform()
    for name, g in fc[fc.source == "Dampier 2026"].groupby("site"):
        nbr = xr.open_zarr(S.FIRE_DIR / f"s2_nbr_{S.slug(name)}.zarr").nbr.load()
        with rasterio.open(S.OUT_DIR / f"struct_nisar20m_{name}.tif") as s:
            r0 = round((s.transform.f - tf_ref.f) / tf_ref.e)
            c0 = round((s.transform.c - tf_ref.c) / tf_ref.a)
        t = pd.DatetimeIndex(nbr.time.values).normalize()
        series[name] = (g, pd.DataFrame(nbr.values[:, g.row.values - r0, g.col.values - c0], index=t))
    for name, (g, df) in series.items():
        b = (base if g.source.iloc[0] == "Yawuru 2025" else dbase).reindex(df.index).to_numpy()
        an = df.to_numpy() - b[:, None]
        t = df.index
        for j, (_, r) in enumerate(g.iterrows()):
            s, d0, d1 = s2_step(an[:, j], t, pd.Timestamp(r.w0), pd.Timestamp(r.w1))
            res.append(dict(site=name, row=r.row, col=r.col, dnbr_gap=s, date_before=d0, date_gap=d1,
                            n_valid=int(np.isfinite(an[:, j]).sum())))
    out = pd.DataFrame(res)
    out["s2_burnt"] = out.dnbr_gap >= S.BURNT_DNBR
    out.to_parquet(OUT / "s2_cells.parquet", index=False)
    m = fc.merge(out, on=["site", "row", "col"])
    print(m.groupby(["source", "site"]).agg(nafi_gap=("fire", lambda s: (s == "gap").mean()),
                                            s2_burnt=("s2_burnt", "mean"),
                                            agree=("s2_burnt", lambda s: (s == (m.loc[s.index, "fire"] == "gap")).mean()),
                                            s2_month=("date_gap", lambda d: d[m.loc[d.index, "s2_burnt"]].dt.month.median())
                                            ).round(3).to_string())


STACK = Path(r"E:\nisar_2026\feature_stack_2026_gslcg0_rflee9_ws5_s1.tif")
FEAT_Y = C.OUT / "features_yawuru_s1.parquet"
OFF_BANDS = ["168_A_g0HV_dB", "168_A_g0HH_dB", "147_D_g0HV_dB", "147_D_g0HH_dB", "S1_VH_dB", "NDVI", "EVI", "BSI"]
COVER_EDGES = [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
MHI_EDGES = [0, 3, 5, 8, 100]  # mhi NaN (cover 0) -> bin 0
MIN_BIN = 10
MIN_AREA = 30


def table():
    """All label rows with fire flags and the 49 features (Dampier: features_2026_s1.parquet; Yawuru: stack)."""
    cells = pd.read_parquet(OUT / "cells.parquet")
    fc = pd.read_parquet(OUT / "fire_cells.parquet")
    s2 = pd.read_parquet(OUT / "s2_cells.parquet") if (OUT / "s2_cells.parquet").exists() else None
    t = cells.merge(fc.drop(columns=["source"]), on=["site", "row", "col"], validate="1:1")
    if s2 is not None:
        t = t.merge(s2, on=["site", "row", "col"], how="left", validate="1:1")
    fd = pd.read_parquet(C.OUT / "features_2026_s1.parquet").drop(columns="mean")
    if not FEAT_Y.exists():
        sys.path.insert(0, str(C.NISAR))
        import feature_stack_2026 as F
        y = t[t.source == "Yawuru 2025"][["site", "row", "col"]].reset_index(drop=True)
        y.join(F.sample(y, STACK)).to_parquet(FEAT_Y, index=False)
    fy = pd.read_parquet(FEAT_Y)
    feats = [c for c in fd.columns if c not in ("site", "row", "col")]
    assert list(fy.columns[3:]) == feats, "feature order differs"
    t = t.merge(pd.concat([fd, fy], ignore_index=True), on=["site", "row", "col"], how="left", validate="1:1")
    ok = np.isfinite(t[feats].to_numpy()).all(1)
    print(f"rows with a non-finite feature, dropped: {t.loc[~ok, 'site'].value_counts().to_dict() or 0}", flush=True)
    return t[ok].reset_index(drop=True), feats


def cmd_effect(_):
    t, feats = table()
    y = t[t.source == "Yawuru 2025"].copy()
    G = G_END
    # 1. counts per area
    rows = []
    for site, g in t.groupby("site", sort=False):
        b = g[g.fire == "gap"]
        r = dict(source=g.source.iloc[0], site=site, n=len(g), F=g["last"].iloc[0].date(),
                 gap=(g.fire == "gap").mean(), amb=(g.fire == "ambiguous").mean(), edge=g.edge.mean())
        if "s2_burnt" in g:
            r["s2_burnt"] = g.s2_burnt.mean()
            r["s2_burnt_in_gap"] = b.s2_burnt.mean() if len(b) else np.nan
            r["s2_burnt_in_none"] = g[g.fire == "none"].s2_burnt.mean()
        if len(b):
            r["fire_end"] = ", ".join(sorted({str(d.date()) for d in b.first_gap_end}))
            r["days_to_G"] = int((G - b.first_gap_end).dt.days.median())
            m = b.first_gap_end.dt.month
            r["season"] = "early (Apr-Jul)" if m.median() <= 7 else "late (Aug-Nov)"
        rows.append(r)
    cnt = pd.DataFrame(rows)
    cnt.to_csv(OUT / "counts.csv", index=False)
    print(cnt.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    # 2. offsets gap - not burnt at the same structure, per area
    y["cb"] = pd.cut(y[C.col(H)], COVER_EDGES, labels=False)
    y["hb"] = pd.cut(y[C.mhi(H)].fillna(0.5), MHI_EDGES, labels=False)
    off = []
    for site, g in y.groupby("site"):
        g = g[g.fire.isin(["gap", "none"])]
        n_gap, n_none = (g.fire == "gap").sum(), (g.fire == "none").sum()
        if n_gap < MIN_AREA or n_none < MIN_AREA:
            continue
        for band in OFF_BANDS:
            s = g.groupby(["cb", "hb", "fire"])[band].agg(["median", "size"]).reset_index()
            m = s[s["size"] >= MIN_BIN].pivot_table(index=["cb", "hb"], columns="fire", values="median").dropna()
            nb = s[s["size"] >= MIN_BIN].pivot_table(index=["cb", "hb"], columns="fire", values="size").dropna()
            off.append(dict(site=site, band=band, n_gap=n_gap, n_none=n_none, bins=len(m),
                            n_gap_bins=int(nb.get("gap", pd.Series(dtype=float)).sum()),
                            n_none_bins=int(nb.get("none", pd.Series(dtype=float)).sum()),
                            offset=(m["gap"] - m["none"]).mean() if len(m) else np.nan,
                            raw=g[g.fire == "gap"][band].median() - g[g.fire == "none"][band].median()))
    off = pd.DataFrame(off)
    off.to_csv(OUT / "offsets.csv", index=False)
    if len(off):
        print("\nOffset gap - not burnt (median diff over common cover x mhi bins, >= 10 cells each):")
        print(off.pivot(index="site", columns="band", values="offset")[OFF_BANDS].round(2).join(
            off.groupby("site")[["n_gap", "n_none", "bins", "n_gap_bins", "n_none_bins"]].first()).to_string())
        print("\nRaw offset (no structure matching):")
        print(off.pivot(index="site", columns="band", values="raw")[OFF_BANDS].round(2).to_string())
    # structure of gap vs none (are burnt cells structurally different?)
    print("\nStructure by fire class (Yawuru areas with gap cells): median cover_1.37 / mhi_1.37")
    print(y[y.site.isin(off.site.unique() if len(off) else [])].groupby(["site", "fire"])[[C.col(H), C.mhi(H)]]
          .median().round(2).to_string())


def folds_for(t):
    from sklearn.model_selection import StratifiedKFold
    q = t.groupby("site")[C.col(H)].rank(method="first", pct=True).mul(5).clip(upper=4.999).astype(int)
    strata = t.site + "_" + q.astype(str)
    return [te for _, te in StratifiedKFold(5, shuffle=True, random_state=42).split(t, strata)]


def boot_diff(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(n)]
    return np.median(a) - np.median(b), *np.percentile(d, [2.5, 97.5])


def cmd_fit(_):
    import time
    from sklearn.ensemble import RandomForestRegressor
    t, feats = table()
    folds = folds_for(t)
    yaw = (t.source == "Yawuru 2025").to_numpy()
    gap = (t.fire == "gap").to_numpy()
    amb = (t.fire == "ambiguous").to_numpy()
    configs = {"all": np.ones(len(t), bool), "no_gap": ~gap, "dampier_only": ~yaw}
    if amb.any():
        configs["no_gap_amb"] = ~gap & ~amb
    targets = {"cover": (C.col(H), np.ones(len(t), bool)), "mhi": (C.mhi(H), (t[C.col(H)] >= 0.05).to_numpy())}
    X = t[feats].to_numpy("float32")
    pred_path = OUT / "fit_oof.parquet"
    oof = pd.read_parquet(pred_path) if pred_path.exists() else t[["source", "site", "row", "col", "fire"]].copy()
    imps = []
    for tn, (col, rows) in targets.items():
        yv = t[col].to_numpy()
        for cn, use in configs.items():
            key = f"{tn}__{cn}"
            if key in oof:
                print(f"[fit] {key}: exists, skipped", flush=True)
                continue
            p = np.full(len(t), np.nan)
            t0 = time.time()
            for k, te in enumerate(folds):
                tr = np.setdiff1d(np.arange(len(t)), te)
                tr = tr[use[tr] & rows[tr]]
                tes = te[rows[te]]
                m = RandomForestRegressor(500, max_features=0.33, min_samples_leaf=5, n_jobs=N_JOBS, random_state=k)
                m.fit(X[tr], yv[tr])
                p[tes] = m.predict(X[tes])
                if cn == "all":
                    imps.append(pd.DataFrame({"target": tn, "fold": k, "feature": feats,
                                              "impurity": m.feature_importances_}))
            oof[key] = p
            part = pred_path.with_name("fit_oof.partial.parquet")
            oof.to_parquet(part, index=False)
            part.replace(pred_path)
            print(f"[fit] {key}: {(time.time() - t0) / 60:.1f} min", flush=True)
    if imps:
        pd.concat(imps).to_csv(OUT / "fit_importance.csv", index=False)
    fit_summary(t, oof, targets, configs)


def fit_summary(t, oof, targets, configs):
    from sklearn.metrics import r2_score
    rows, res = [], []
    groups = {"Dampier": t.source == "Dampier 2026", "Yawuru": t.source == "Yawuru 2025",
              "Yawuru not burnt": (t.source == "Yawuru 2025") & (t.fire == "none"),
              "Yawuru gap": (t.source == "Yawuru 2025") & (t.fire == "gap")}
    for tn, (col, _) in targets.items():
        for cn in configs:
            p = oof[f"{tn}__{cn}"]
            for gn, g in groups.items():
                m = g & p.notna()
                if m.sum() < 10:
                    continue
                e = p[m] - t.loc[m, col]
                rows.append(dict(target=tn, config=cn, rows=gn, n=int(m.sum()), r2=r2_score(t.loc[m, col], p[m]),
                                 rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean())))
            # residual gap - not burnt within each burnt area
            for site, g in t[t.source == "Yawuru 2025"].groupby("site"):
                e = (p - t[col])[g.index]
                a, b = e[g.fire == "gap"].dropna().to_numpy(), e[g.fire == "none"].dropna().to_numpy()
                if len(a) >= MIN_AREA and len(b) >= MIN_AREA:
                    d, lo, hi = boot_diff(a, b)
                    res.append(dict(target=tn, config=cn, site=site, n_gap=len(a), n_none=len(b), diff=d, lo=lo, hi=hi))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "fit_metrics.csv", index=False)
    r = pd.DataFrame(res)
    r.to_csv(OUT / "fit_residuals.csv", index=False)
    print("\nK-fold metrics per source (never blended):")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nResidual (pred - label) median, gap - not burnt, within area (bootstrap 95 % CI):")
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


REPORT = C.HERE / "REPORT_fire_gap.txt"
READING = """\
| Item | Result |
|---|---|
| Dampier 2026 | 0 cells burnt in the image window (NAFI); S2 < 1 % (pointer). Labels OK. |
| Yawuru, burnt in the gap (NAFI) | 6 of 13 areas: YA25_1 55 %, YA25_4 68 %, YJ25_2 56 %, YJ25_3 100 %, YJ25_4 99 %, YJ25_5 67 % (3,020 of 8,419 cells) |
| Fire dates | 2025-05-18..22 (YA25_1); 2025-08-01..09-14 (YA25_4, YJ25_2..4); 2026-04..06 (YJ25_5); 131-506 days before G |
| S2 cross-check | confirms the NAFI dates; YA25_1 99 % and YJ25_2 92 % burnt by S2: NAFI's 250 m edges miss part of each fire |
| S2 only | govcowwell 65 % + YA25_3 58 % (adjacent), 2026-04-29..05-02, dNBR 0.17-0.20, no NAFI scar: uncertain |
| S2 artefact | YOct25_1 43 %: uniform dNBR 0.10 +- 0.005 on the first date after the flight -> not a fire |
| Images vs labels | at the same cover x height, gap cells read lower: NISAR -0.4 to -1.1 dB (YA25_1, YA25_4, YJ25_5), S1 VH -0.3 to -2.4 dB, NDVI up to -0.09; YJ25_2 ~0 (its controls are S2-burnt) |
| Label fit | gap rows not in training -> predicted cover below the pre-fire label (bias -0.045 to -0.11 in YA25_1, YJ25_2, YJ25_5, YA25_4; unburnt areas +-0.02) |
| Gap rows in training | the RF fits them (random K-fold, autocorrelated): the mismatch becomes label noise, not a visible error |
| Dampier rows | unchanged by adding Yawuru (cover R2 0.773 vs 0.777; mhi 0.699 vs 0.700) |
| Ambiguous | 0 cells: "drop gap + ambiguous" = "drop gap" |

Recommendation: drop the 6 burnt areas whole (YA25_1, YA25_4, YJ25_2, YJ25_3, YJ25_4, YJ25_5), not only the
NAFI gap cells. Reason: the areas are ~0.25 km2 against a 250 m NAFI grid (88-100 % of cells in burnt areas are
< 250 m from a scar edge), and S2 shows the "not burnt" rest of YA25_1 and YJ25_2 also burnt. Keeps 7 areas,
4,308 cells. Flag (keep, or user decision) govcowwell + YA25_3 for the S2-only 2026-05 drop.
Alternatives: drop NAFI gap cells only (5,399 cells kept); keep all with a fire flag."""


def fig_timeline(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(C.PROJ / "chm_regression"))
    import viz_style as V
    V.apply_style()
    fd = pd.read_csv(OUT / "flight_dates.csv", parse_dates=["last"])
    per = pd.read_csv(OUT / "gap_periods.csv", parse_dates=["s", "e"])
    cnt = pd.read_csv(OUT / "counts.csv").set_index("site")
    blue, orange = V.TARGET_COLORS["mean"], V.TARGET_COLORS["max"]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(fd) + 1.4))
    ax.axvspan(IMG_START, G_END, color=V.GRID, lw=0, zorder=0)
    ax.text(IMG_START - pd.Timedelta(days=4), len(fd) - 0.5, "images 09-20..10-11", color=V.TEXT_SECONDARY,
            fontsize=8, ha="right", va="center")
    for i, r in fd.iterrows():
        for _, p in per[(per.site == r.site) & (per.cls == "gap")].iterrows():
            ax.barh(i, (p.e - p.s).days + 1, left=p.s, height=0.5, color=orange, zorder=2)
        ax.plot(r["last"], i, "o", ms=8, color=blue, mec=V.SURFACE, mew=2, zorder=3)
        g = cnt.loc[r.site, "gap"]
        if g > 0:
            ax.text(pd.Timestamp("2026-10-25"), i, f"{100 * g:.0f} % burnt", va="center", fontsize=8,
                    color=V.TEXT_PRIMARY)
    ax.set_yticks(range(len(fd)), fd.site)
    ax.invert_yaxis()
    ax.set_xlim(pd.Timestamp("2025-03-01"), pd.Timestamp("2027-01-15"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=8, color=blue, label="LiDAR flight (gps_time)"),
                       Patch(color=orange, label="NAFI fire scar period, in the gap")],
              loc="center", bbox_to_anchor=(0.45, 0.2), frameon=False, fontsize=8)
    ax.set_ylim(len(fd), -0.7)
    ax.set_title("Fires between LiDAR flight and the 2026 image stack (NAFI; % = cells in a gap scar)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_report(_):
    f = lambda v: f"{v:.3f}"  # noqa: E731
    fd = pd.read_csv(OUT / "flight_dates.csv")
    cnt = pd.read_csv(OUT / "counts.csv")
    off = pd.read_csv(OUT / "offsets.csv")
    met = pd.read_csv(OUT / "fit_metrics.csv")
    res = pd.read_csv(OUT / "fit_residuals.csv")
    per = pd.read_csv(OUT / "gap_periods.csv")
    fig = OUT / "fig_fire_gap.png"
    fig_timeline(fig)
    o = pd.read_parquet(OUT / "fit_oof.parquet").merge(pd.read_parquet(OUT / "cells.parquet"),
                                                      on=["source", "site", "row", "col"])
    pa = []
    for s, g in o[o.source == "Yawuru 2025"].groupby("site"):
        r = dict(site=s, gap=(g.fire == "gap").mean())
        for tn, col in [("cover", C.col(H)), ("mhi", C.mhi(H))]:
            for cn in ["all", "no_gap"]:
                e = (g[f"{tn}__{cn}"] - g[col]).dropna()
                r[f"{tn}_{cn}_bias"], r[f"{tn}_{cn}_rmse"] = e.mean(), np.sqrt((e ** 2).mean())
        pa.append(r)
    pa = pd.DataFrame(pa)
    pa.to_csv(OUT / "fit_per_area.csv", index=False)
    s2a = None
    if (OUT / "s2_cells.parquet").exists():
        m = pd.read_parquet(OUT / "fire_cells.parquet").merge(pd.read_parquet(OUT / "s2_cells.parquet"),
                                                              on=["site", "row", "col"])
        m["nafi_gap"] = m.fire == "gap"
        s2a = m.groupby(["source", "site"]).agg(nafi_gap=("nafi_gap", "mean"), s2_burnt=("s2_burnt", "mean"),
                                                 agree=("s2_burnt", lambda x: (x == m.loc[x.index, "nafi_gap"]).mean()),
                                                 s2_date_median=("date_gap", lambda d: d[m.loc[d.index, "s2_burnt"]].median()))
        s2a.to_csv(OUT / "s2_agreement.csv")
    cnt_cols = [c for c in ["source", "site", "n", "F", "gap", "s2_burnt_in_gap", "s2_burnt_in_none", "edge",
                            "fire_end", "days_to_G", "season"] if c in cnt]
    L = [
        "REPORT - fire gap between LiDAR flight and the 2026 image stack (Part A, PLAN_fire_gap.md)",
        f"Date: {pd.Timestamp.now():%Y-%m-%d}. Script: {C.HERE / 'fire_gap.py'}",
        f"Outputs: {OUT}",
        f"Figure: {fig}",
        "",
        "READING AND RECOMMENDATION",
        READING,
        "",
        "1. FLIGHT DATES (UTC, point gps_time)",
        fd[["site", "source", "first", "last", "flag"]].fillna("").to_string(index=False),
        "",
        "2. FIRES IN THE GAP PER AREA",
        "Window: Yawuru = flight .. 2026-10-11 (last NISAR date); Dampier = 2026-09-20 .. 2026-10-11 (image window).",
        "gap = NAFI scar period wholly in the window; ambiguous = period spans a window end (0 cells here).",
        "edge = cell centre < 250 m from a window-scar edge (NAFI edge uncertainty).",
        cnt[cnt_cols].to_string(index=False, float_format=f),
        "",
        "NAFI scar periods in the window (cells per period):",
        per.to_string(index=False),
        "",
        "S2 NBR cross-check (pointer only): share of cells with dNBR >= 0.1 in the window (dry season only);",
        "agree = S2 burnt == NAFI gap; s2_date_median = first clear date after the drop.",
        s2a.to_string(float_format=f) if s2a is not None else "(not run)",
        "",
        "3. FEATURE OFFSET, GAP - NOT BURNT, SAME STRUCTURE",
        "Median difference in common bins of cover_1.37 x mhi_1.37 (>= 10 cells per class per bin), mean over bins.",
        "Only areas with >= 30 gap and >= 30 not-burnt cells (YJ25_3, YJ25_4: all burnt -> no comparison).",
        "SAR in dB; NDVI / EVI / BSI in index units.",
        off.pivot(index="site", columns="band", values="offset")[OFF_BANDS].to_string(float_format=lambda v: f"{v:.2f}"),
        "",
        off.groupby("site")[["n_gap", "n_none", "bins", "n_gap_bins", "n_none_bins"]].first().to_string(),
        "",
        "4. LABEL FIT (pooled RF 500, max_features 0.33, leaf 5; 49 features; 5 folds site x cover quintile, seed 42)",
        "Configs: all = Dampier + all Yawuru; no_gap = Yawuru gap rows removed from training (still predicted);",
        "dampier_only = Dampier rows only in training. Out-of-fold, per source (never blended).",
        "mhi rows: cover_1.37 >= 0.05.",
        met.to_string(index=False, float_format=f),
        "",
        "5. RESIDUAL (pred - label), GAP - NOT BURNT, WITHIN AREA (median diff, bootstrap 95 % CI)",
        res.to_string(index=False, float_format=f),
        "",
        "6. OUT-OF-FOLD BIAS / RMSE PER YAWURU AREA (pred - label), configs all and no_gap",
        "NOTE: YJ25_3 and YJ25_4 are ~100 % gap, so under no_gap they have no training rows: their no_gap",
        "numbers are in effect leave-area-out (project rule: not used as evidence; shown for completeness).",
        pa.to_string(index=False, float_format=f),
        "",
        "CAVEATS",
        "- NAFI is 250 m; the areas are ~0.25 km2, so most burnt-area cells are near a scar edge.",
        "- S2 NBR drop is a pointer only (no manual checking).",
        "- 12-18 months of growth without fire also changes structure; not measurable here.",
        "- YA25_1-5 are single-return clouds. K-fold is optimistic (spatial autocorrelation).",
    ]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}\nwrote {fig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dates", "cells", "nafi", "s2", "effect", "fit", "report"])
    ap.add_argument("--rate", type=float, default=4.0, help="download cap, MB/s total (user rule)")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
