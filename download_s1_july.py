"""
Download the Sentinel-1 July global mosaic (VV + VH) for one year over the
Dampier Peninsula, into a dask-chunked Zarr store.

Source: Copernicus Data Space (CDSE) STAC, collection "sentinel-1-global-mosaics"
(monthly 20 m VV/VH backscatter mosaics). This is the same product the
Dampier 4-class classification notebooks consume as `s1_july_<year>_DP.zarr`.

Run it straight from a terminal (`python download_s1_july.py`) or paste the
cells into a notebook. Change YEAR, run, repeat.
"""

import os
import numpy as np
import geopandas as gpd
import pystac_client
import odc.stac
import dask

# ----------------------------------------------------------------------------
# Config -- edit these
# ----------------------------------------------------------------------------
YEAR = 2024
AOI_PATH = r"G:\Australia_shapefile\Dampier_peninsula_cutbox.gpkg"
OUT_PATH = rf"s1_july_{YEAR}_DP.zarr"   # written into this folder

CRS = "EPSG:32751"    # WGS84 / UTM zone 51S -- matches the classification grid
RESOLUTION = 20       # metres (native mosaic resolution)
CHUNK = 4000          # dask chunk size in pixels (x and y)

# CDSE S3 credentials. Your working keys live in
# G:\python\Karajarri\2026 Forest four-class with Yawuru training\download_S1.ipynb
os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
os.environ["AWS_ACCESS_KEY_ID"] = "ZENXKDK096O00XZWJA6A"
os.environ["AWS_SECRET_ACCESS_KEY"] = "zPgGuMtcOjxhupahVZnMxT5mkSsdxkFR71KT3w1G"
os.environ["AWS_HTTPS"] = "YES"
os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
# eodata S3/TLS drops reads under load; without this GDAL silently returns
# nodata for failed tiles (-> NaN holes in the mosaic). Harden + retry.
os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["GDAL_HTTP_MAX_RETRY"] = "10"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"

# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # No dask.distributed cluster: the default threaded scheduler is more robust
    # for this I/O-bound S3 -> local-zarr job (no workers to be killed on slow reads).

    # AOI bounds in lon/lat for the STAC search
    aoi = gpd.read_file(AOI_PATH).to_crs("EPSG:4326")
    bbox = tuple(aoi.total_bounds)   # (minx, miny, maxx, maxy)

    catalog = pystac_client.Client.open("https://stac.dataspace.copernicus.eu/v1")
    catalog.add_conforms_to("ITEM_SEARCH")

    search = catalog.search(
        collections=["sentinel-1-global-mosaics"],
        bbox=bbox,
        datetime=f"{YEAR}-07-01/{YEAR}-07-31",
    )
    items = list(search.items())
    print(f"{len(items)} items found for July {YEAR}")

    ds = odc.stac.load(
        items,
        bands=["VV", "VH"],
        bbox=bbox,
        crs=CRS,
        resolution=RESOLUTION,
        chunks={"x": CHUNK, "y": CHUNK},
        fail_on_error=False,
        nodata=np.nan,
    )
    print(ds)

    # Serial reads: parallel /vsis3 GETs trip eodata's throttling (503 -> NaN holes).
    with dask.config.set(scheduler="single-threaded"):
        ds.to_zarr(OUT_PATH, mode="w")
    print(f"written: {OUT_PATH}")
