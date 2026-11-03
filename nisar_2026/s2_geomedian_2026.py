"""Stage 0d: Sentinel-2 geomedian 2026-09-20 .. 2026-10-10 over the Dampier cutbox (PLAN.md).

Two steps:
  download   DEA STAC (S2A/B/C ARD), every item over the cutbox in the window with
             eo:cloud_cover < --max-cloud and >= --min-km2 inside the mainland polygon
             (drops 5 edge tiles with < 10 km2 of land); the 10 bands + oa_fmask are downloaded whole at
             <= --rate MB/s in total to E:\\S2_DP_2026\\raw\\<item id>\\<band>.tif (resumable).
             The selected items are saved as items.json.
  composite  odc.stac.load of the local copies (EPSG:32751, 10 m, solar_day) -> fmask [0,2,3]
             + mask_cleanup -> to_f32 -> xr_geomedian, the chain of ../download_s2_july.py.
             Also writes `count` = clear observations per pixel.
             -> G:\\python\\odc-stac\\ds_geomedian_DP_2026_0920_1010.zarr

Download code is imported from G:\\python\\GDV\\GEDI (s2_download.fetch_retry, chm_config.Pacer).

Run from this directory:
    conda run -n general --no-capture-output python s2_geomedian_2026.py download --dry-run
    conda run -n general --no-capture-output python s2_geomedian_2026.py download --rate 2
    conda run -n general --no-capture-output python s2_geomedian_2026.py composite
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pystac
import pystac_client

sys.path.insert(0, r"G:\python\GDV\GEDI")
from chm_config import Pacer  # noqa: E402
from s2_download import S3, fetch_retry  # noqa: E402

STAC_URL = "https://explorer.dea.ga.gov.au/stac"
COLLECTIONS = ["ga_s2am_ard_3", "ga_s2bm_ard_3", "ga_s2cm_ard_3"]
START, END = "2026-09-20", "2026-10-10"
AOI_WIN = r"G:\Australia_shapefile\Dampier_peninsula_cutbox.gpkg"
MAINLAND_WIN = r"G:\Australia_shapefile\Dampier_peninsula_mainland.gpkg"
RAW = Path(r"E:\S2_DP_2026\raw")
ITEMS_JSON = RAW.parent / "items.json"
OUT_ZARR = r"G:\python\odc-stac\ds_geomedian_DP_2026_0920_1010.zarr"
CRS, RES = "EPSG:32751", 10

BANDS = ["nbart_blue", "nbart_green", "nbart_red", "nbart_red_edge_1", "nbart_red_edge_2",
         "nbart_red_edge_3", "nbart_nir_1", "nbart_nir_2", "nbart_swir_2", "nbart_swir_3"]
FMASK_BAD = [0, 2, 3]                       # nodata, cloud, shadow
MASK_FILTERS = [("opening", 2), ("dilation", 3)]


def aoi_bbox():
    return gpd.read_file(AOI_WIN).to_crs(4326).total_bounds.tolist()


def select(max_cloud, min_km2):
    from shapely.geometry import shape
    land = gpd.read_file(MAINLAND_WIN).to_crs(32751).union_all()
    items = pystac_client.Client.open(STAC_URL).search(
        collections=COLLECTIONS, bbox=aoi_bbox(), datetime=f"{START}/{END}").item_collection()
    on_land = lambda i: gpd.GeoSeries([shape(i.geometry)], crs=4326).to_crs(32751).iloc[0].intersection(land).area / 1e6
    return pystac.ItemCollection([i for i in items if i.properties.get("eo:cloud_cover", 100) < max_cloud
                                  and on_land(i) >= min_km2])


def download(args):
    RAW.mkdir(parents=True, exist_ok=True)
    if ITEMS_JSON.exists() and not args.refresh:
        items = pystac.ItemCollection.from_file(str(ITEMS_JSON))
    else:
        items = select(args.max_cloud, args.min_km2)
        items.save_object(dest_href=str(ITEMS_JSON))
    tiles = sorted({i.properties["odc:region_code"] for i in items})
    dates = sorted({i.datetime.date().isoformat() for i in items})
    print(f"{len(items)} items, tiles {tiles}, {len(dates)} dates {dates[0]} .. {dates[-1]}", flush=True)
    jobs = [(i.assets[b].href.replace(*S3), RAW / i.id / f"{b}.tif") for i in items for b in BANDS + ["oa_fmask"]]
    todo = [j for j in jobs if not j[1].exists()]
    print(f"{len(todo)}/{len(jobs)} files to get at <= {args.rate} MB/s -> {RAW}", flush=True)
    if args.dry_run:
        return
    for _, dst in todo:
        dst.parent.mkdir(exist_ok=True)
    pacer, t0, done, errors = Pacer(args.rate), time.monotonic(), 0, []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_retry, url, dst, pacer): dst for url, dst in todo}
        for fu in as_completed(futs):
            try:
                fu.result()
                done += 1
            except Exception as e:           # keep going; a re-run retries the file
                errors.append(futs[fu])
                print(f"  ERROR {futs[fu]}: {e}", flush=True)
            if done % 20 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} files, {pacer.got / 1e9:.2f} GB, "
                      f"{(time.monotonic() - t0) / 60:.0f} min, {pacer.mbps():.2f} MB/s", flush=True)
    print(f"done: {done} files, {len(errors)} errors (re-run to retry)", flush=True)


def composite(args):
    import odc.stac
    import rioxarray  # noqa: F401  (registers .rio)
    from dask.distributed import Client, LocalCluster
    from odc.algo import erase_bad, mask_cleanup, to_f32, xr_geomedian

    items = pystac.ItemCollection.from_file(str(ITEMS_JSON))
    for i in items:                          # point every asset at the local copy
        for b in BANDS + ["oa_fmask"]:
            p = RAW / i.id / f"{b}.tif"
            assert p.exists(), f"missing {p}: run the download step first"
            i.assets[b].href = p.as_uri()    # a bare E:\ path is joined to the item's https self link
    client = Client(LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="8GB"))
    print(client.dashboard_link, flush=True)
    ds = odc.stac.load(items, bands=BANDS + ["oa_fmask"], groupby="solar_day", bbox=aoi_bbox(),
                       crs=CRS, resolution=RES, chunks={"x": args.chunk, "y": args.chunk})
    print(f"{ds.sizes['time']} solar days: {[str(t)[:10] for t in ds.time.values]}", flush=True)
    bad = mask_cleanup(ds["oa_fmask"].isin(FMASK_BAD), mask_filters=MASK_FILTERS)
    count = (~bad).sum("time").astype("uint8")
    gm = xr_geomedian(to_f32(erase_bad(ds[BANDS], bad), scale=1 / 10000, offset=0),
                      num_threads=1, eps=1e-7)
    gm["count"] = count
    gm = gm.rio.write_crs(CRS)
    gm.to_zarr(OUT_ZARR, mode="w")
    print(f"written: {OUT_ZARR}", flush=True)
    client.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    d = sub.add_parser("download")
    d.add_argument("--rate", type=float, default=4.0, help="max MB/s in total over all threads")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--max-cloud", type=float, default=80.0)
    d.add_argument("--min-km2", type=float, default=10.0, help="min item area inside the mainland")
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--refresh", action="store_true", help="redo the STAC search")
    c = sub.add_parser("composite")
    c.add_argument("--chunk", type=int, default=1000)
    a = ap.parse_args()
    download(a) if a.step == "download" else composite(a)
