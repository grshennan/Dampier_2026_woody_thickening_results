"""Accuracy diagnosis Stage 1.4 (PLAN_stage1_nisar.md; source chosen by the user 2026-10-28): download the CDSE
Sentinel-1 monthly global mosaic (collection "sentinel-1-global-mosaics", VV + VH, 20 m, the source of
download_s1_july.py and the 2019-2025 series) for 2026 over the Dampier cutbox, at <= --rate MB/s in total
(1 MB = 1e6 B; sleep when the bytes received run ahead of rate x elapsed, as GDV\\SARGDE\\sargde_download.fetch).

Whole COG files over S3 (credentials: the CDSE S3 keys set by ..\\download_s1_july.py on import), one at a time,
to E:\\nisar_2026\\s1_mosaic\\<item id>_<band>.tif (*.part, size-checked, then renamed; a complete file is skipped).

    conda run -n general --no-capture-output python diagnosis\\s1_mosaic_2026.py [--months 2026-09] [--rate 4]
"""
import argparse
import sys
import time
from pathlib import Path

import boto3
import geopandas as gpd
import pystac_client

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
import download_s1_july as S1  # noqa: E402  (sets the CDSE S3 environment)

OUT = Path(r"E:\nisar_2026\s1_mosaic")
CHUNK = 1 << 20


def items(month):
    bbox = tuple(gpd.read_file(S1.AOI_PATH).to_crs("EPSG:4326").total_bounds)
    cat = pystac_client.Client.open("https://stac.dataspace.copernicus.eu/v1")
    cat.add_conforms_to("ITEM_SEARCH")
    y, m = map(int, month.split("-"))
    end = f"{y + m // 12}-{m % 12 + 1:02d}-01"
    return list(cat.search(collections=["sentinel-1-global-mosaics"], bbox=bbox,
                           datetime=f"{month}-01T00:00:00Z/{end}T00:00:00Z").items())


def fetch(s3, href, out, rate):
    bucket, key = href[len("s3://"):].split("/", 1)
    total = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    if out.exists() and out.stat().st_size == total:
        print(f"  skip {out.name} (complete)", flush=True)
        return 0
    part = out.with_name(out.name + ".part")
    got, t0 = 0, time.monotonic()
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    with open(part, "wb") as f:
        for chunk in body.iter_chunks(CHUNK):
            f.write(chunk)
            got += len(chunk)
            ahead = got / rate - (time.monotonic() - t0)
            if ahead > 0:
                time.sleep(ahead)
    if got != total:
        raise IOError(f"{key}: short read {got}/{total} bytes")
    part.replace(out)
    print(f"  {out.name}: {got / 1e6:.0f} MB in {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["2026-09", "2026-10"])
    ap.add_argument("--rate", type=float, default=4.0, help="max MB/s in total (1 MB = 1e6 B)")
    a = ap.parse_args()
    s3 = boto3.client("s3", endpoint_url="https://" + S1.os.environ["AWS_S3_ENDPOINT"], region_name="default",
                      aws_access_key_id=S1.os.environ["AWS_ACCESS_KEY_ID"],
                      aws_secret_access_key=S1.os.environ["AWS_SECRET_ACCESS_KEY"])
    try:
        s3.head_object(Bucket="eodata", Key=items(a.months[0])[0].assets["VV"].href.split("/", 3)[3])
    except s3.exceptions.ClientError as e:
        raise SystemExit(f"CDSE S3 refused the keys in download_s1_july.py ({e}); put working keys there")
    OUT.mkdir(parents=True, exist_ok=True)
    for month in a.months:
        its = items(month)
        print(f"{month}: {len(its)} items {[i.id for i in its]}", flush=True)
        for it in its:
            for b in ("VV", "VH"):
                fetch(s3, it.assets[b].href, OUT / f"{it.id}_{b}.tif", a.rate * 1e6)
    print("done", flush=True)


if __name__ == "__main__":
    main()
