"""Regression track: MTE regression 2019-2026, one model per year (PLAN_mte_retro.md 2026-11-03; unf re-run
2026-11-04, plan C:\\Users\\glens\\.claude\\plans\\we-re-doing-the-regression-snazzy-graham.md).
2019 / 2021 / 2023 / 2024 / 2025: PALSAR + S1 + S2 stack (20 bands). 2026: NISAR + S1 + S2 stack (49 bands).
Labels: 5 Dampier 2026 sites (unfiltered CHMs, --lidar unf) + 13 Yawuru 2025 areas. 2019-2025: burnt areas kept
(fire = noise, user 2026-11-03). 2026: burnt areas dropped (as the old 2026 run, Gate A).
Targets per stack px from the 50 cm CHM: mean, p98, cover 1.37 (translation.aggregate_with_cover); count >= 30.
Project CRS EPSG:7851 (user 2026-11-04): every input is asserted 7851; products are written in 7851.

  prep    (env general)  per year: label cells on the year's stack grid (cached, provenance-checked); 20 / 49
          features read from the 7851 stack at the label px; folds -> outputs/mte_retro/<year>/rows_<year>.parquet
  kfold   (env pytorch)  5 members (seeds 0-4) x 5 folds -> outputs/mte_retro/<year>/*.parquet / *.csv
  maps    (env pytorch)  5 members on all rows -> <map dir>/*_MTE[_<year>].tif + residuals
          map dir: E:/nisar_2026/regression/retro/<year>/ (2019-2025), E:/nisar_2026/regression/ (2026)
  report  -> REPORT_mte_retro.txt
  --quick prep: 2025, 9 Mile + YJ25_1; unf != filt; code path = retro_2d chm_targets; writes nothing.
          kfold: fold 0, 2 members, 30 epochs, writes nothing.

Divergences from mlp_mte_2026.py / class_defs tables:
 1. CRS (user 2026-11-04): project CRS EPSG:7851. Inputs are asserted 7851, never relabelled here: stacks and
    boundaries were converted by G:/python/crs_audit/mga51.py (stacks: ITRF2014 to GDA2020 (1), EPSG:8048, at the
    epoch of the year; origin moved, values unchanged); Yawuru CHMs / polygons are *_mga51 copies (label corrected
    to the verified frame GDA2020). (Replaces the 2026-11-03 rule: attach 32751 after a null 7851 -> 32751 check.)
 2. Stack: 20 bands (PALSAR 6, S1 6, S2 8) on each year's PALSAR grid; no NISAR before 2026.
 3. Labels: all 13 Yawuru areas; the 6 burnt areas are group 'Yawuru burnt' (reported, not removed; 2026: removed).
 4. Config fixed = 2026 best (outputs/mlp_mte/best_config.json); no grid, no no-clamp ablation, no RF / MLP.
 5. Yawuru clip guard: cells x px area / polygon area in [0.8, 1.2] (24 m px; struct_palsar rule).
 6. Folds: fire_gap.folds_for rule (site x per-site cover quintile, StratifiedKFold 5, seed 42), per year.
 7. Label px with p98 < mean are kept (1 px in 2019 / 2023 / 2025); stop only if >= 0.5 %.
 8. LiDAR: --lidar unf (default) = common.chm_path(site, 'unf'), the unfiltered re-export CHMs; each path logged.
 9. Caches (cells_<y>.parquet, members *.pt, map rasters) carry a provenance JSON (CHM paths + size + mtime; rows
    SHA-1). A cache is re-used only if its provenance matches; else the step stops.
10. Year 2026: NISAR 49-band stack (read, not re-built); sensors NISAR / S1 / S2 by band name; old 2026 map names;
    valid mask = finite stack and mainland. One script, one report, 6 years.
11. The stack is not re-built: features are read from the 7851 stack copy (stack_<y>_mga51.tif) at the label px;
    its grid = the source stack grid + the logged origin move (asserted).

    C:\\Users\\glens\\miniconda3\\Scripts\\conda.exe run -n general --no-capture-output python -u mte_retro.py prep --quick
    C:\\Users\\glens\\miniconda3\\Scripts\\conda.exe run -n pytorch --no-capture-output python -u mte_retro.py kfold --years 2025 --quick
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import common as C
from cover_height_report import reg

YEARS = ["2025", "2024", "2023", "2021", "2019", "2026"]
OUTT = C.OUT / "mte_retro"
MAPR = Path(r"E:\nisar_2026\regression\retro")
MAP26 = Path(r"E:\nisar_2026\regression")
STACK26 = Path(r"E:\nisar_2026\feature_stack_2026_gslcg0_rflee9_ws5_s1_mga51.tif")
MGA_LOG = Path(r"G:\python\crs_audit\apply_mga51_2026-11-04.json")
REPORT = C.HERE / "REPORT_mte_retro.txt"
EPSG = 7851
H = 1.37
CV = C.col(H)
T3 = ["mean", "p98", "cover"]
YC = ["mean", "p98", CV]
MAPS = {"mean": "height_mean", "p98": "height_p98", "cover": "cover137"}
KEY = ["site", "row", "col"]
GROUPS = ["pooled", "Dampier", "Yawuru kept", "Yawuru burnt"]
BURNT = ["YA25_1", "YA25_4", "YJ25_2", "YJ25_3", "YJ25_4", "YJ25_5"]  # cover_height_rows.BURNT (Gate A)
MSEEDS = [0, 1, 2, 3, 4]
NODATA = -9999.0
BOUNDARIES_WIN = {"mainland": r"G:\Australia_shapefile\Dampier_peninsula_mainland_utm51S_mga51.gpkg",
                  "pindan": r"G:\Veg and Soil Mapping\Dampier_pindan_main_mga51.gpkg"}


def ydir(year):
    return OUTT / year


def mdir(year):
    return MAP26 if year == "2026" else MAPR / year


def tag(year):
    return "" if year == "2026" else f"_{year}"


def stack_path(year):
    return STACK26 if year == "2026" else MAPR / year / f"stack_{year}_mga51.tif"


def member_path(year, s):
    return mdir(year) / (f"mte_ALL_s{s}.pt" if year == "2026" else f"mte_{year}_s{s}.pt")


def mga(p):
    p = Path(p)
    return p.with_name(p.stem + "_mga51" + p.suffix)


# ------------------------------------------------------------------------------------------- CRS / provenance

def assert_crs(obj, name):
    """Divergence 1: every input is in the project CRS EPSG:7851. No relabel here."""
    crs = obj.crs if hasattr(obj, "geometry") else obj.rio.crs
    e = crs.to_epsg() if crs else None
    assert e == EPSG, f"{name}: CRS EPSG:{e}; project CRS is EPSG:{EPSG} (convert with crs_audit/mga51.py)"
    return obj


def file_sig(p):
    st = Path(p).stat()
    return dict(path=str(p), size=st.st_size, mtime=st.st_mtime)


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def prov_path(p):
    return Path(p).with_name(Path(p).name + ".prov.json")


def cache_ok(targets, want):
    """Divergence 9: True if all targets exist with matching provenance; False if none exist; else stop."""
    targets = [Path(t) for t in targets]
    have = [t.exists() for t in targets]
    if not any(have):
        return False
    for t in targets:
        pj = prov_path(t)
        if not t.exists() or not pj.exists() or json.loads(pj.read_text()) != json.loads(json.dumps(want)):
            raise SystemExit(f"[stale] {t}: cache without matching provenance ({pj.name}). Move the old outputs "
                             f"(pre_unf) or delete them, then re-run.")
    return True


def prov_write(target, want):
    prov_path(target).write_text(json.dumps(want, indent=1))


def boundary(name):
    import geopandas as gpd
    from chm_class_histograms import to_local
    return assert_crs(gpd.read_file(to_local(BOUNDARIES_WIN[name])), BOUNDARIES_WIN[name])


def inside_mask(name, tf, shape):
    """Pixel centres inside the boundary; clip guard (CLAUDE.md): px area / polygon area in [0.95, 1.05]."""
    from rasterio.features import geometry_mask
    g = boundary(name)
    m = geometry_mask(g.geometry, out_shape=shape, transform=tf, invert=True)
    ratio = m.sum() * abs(tf.a * tf.e) / g.area.sum()
    assert 0.95 < ratio < 1.05, f"{name}: clip px area / polygon area = {ratio:.3f}"
    return m


def stack_info(year):
    """Divergence 11: the 7851 stack; grid = source grid + logged origin move; band names checked."""
    import rasterio
    from feature_stack import S2_BAND_LABELS
    p = stack_path(year)
    log = json.loads(MGA_LOG.read_text())[str(p)]
    with rasterio.open(p) as s:
        assert s.crs.to_epsg() == EPSG, f"{p}: {s.crs}"
        tf, shape, bands = s.transform, (s.height, s.width), list(s.descriptions)
    with rasterio.open(log["src"]) as s:
        stf = s.transform
        assert list(s.descriptions) == bands and (s.height, s.width) == shape
    exp = (stf.a, stf.b, stf.c + log["dE"], stf.d, stf.e, stf.f + log["dN"])
    assert np.allclose(tuple(tf)[:6], exp, atol=1e-6), f"{year}: stack grid != source grid + logged move"
    if year == "2026":
        assert len(bands) == 49 and all(bands), bands
    else:
        assert len(bands) == 20 and bands[12:20] == S2_BAND_LABELS, bands
    print(f"[crs] {year}: {p.name} EPSG:{EPSG}; {log['op']}; origin move dE {log['dE']:+.4f} dN {log['dN']:+.4f} m",
          flush=True)
    return SimpleNamespace(path=p, tf=tf, shape=shape, bands=bands, move=(log["dE"], log["dN"]), op=log["op"])


def sensor_groups(bands):
    if len(bands) == 20:  # build_stack order (asserted by stack_info)
        return {"PALSAR": list(range(0, 6)), "S1": list(range(6, 12)), "S2": list(range(12, 20))}
    import cover_height_rf as R
    g = R.groups(bands)
    assert set(g) == {"NISAR", "S1", "S2"} and sum(map(len, g.values())) == len(bands), g.keys()
    return {k: [bands.index(c) for c in v] for k, v in g.items()}


# ------------------------------------------------------------------------------------------- prep (env general)

def sites(lidar):
    from chm_class_histograms import to_local
    from chm_targets import AREA_NAMES, LIDAR_PATHS_WIN, LIDAR_POLYS_WIN
    out = [("Dampier 2026", s, Path(C.chm_path(s, lidar)), None) for s in C.L.SITES]
    out += [("Yawuru 2025", n, mga(to_local(c)), mga(to_local(p)))
            for n, c, p in zip(AREA_NAMES, LIDAR_PATHS_WIN, LIDAR_POLYS_WIN)]
    assert len(out) == 18, len(out)
    return out


def ref_grid(year):
    import rioxarray as rxr
    return assert_crs(rxr.open_rasterio(stack_path(year), masked=True).isel(band=[0]), str(stack_path(year)))


def site_cells(source, site, chm, ref, poly):
    """aggregate_with_cover on the year's stack grid (as common._site_cover / p98_cover.rows); count >= 30."""
    from chm_targets import LOW_COUNT_THRESHOLD, target_grid_for_chm
    tgt = target_grid_for_chm(chm, ref)
    ds = C.translation.aggregate_with_cover(chm, tgt, heights=[H])
    df = ds.drop_vars("spatial_ref", errors="ignore").to_dataframe().reset_index()
    tf = tgt.rio.transform()
    pa = abs(tf.a * tf.e)
    cnt = df["count"].fillna(0)
    ratio = (cnt > 0).sum() * pa / (cnt.sum() * 0.25)  # classify_retro.label_table guard (CRS no-op -> 0 or huge)
    assert 0.8 < ratio < 2.0, f"{site}: label cell area / CHM area = {ratio:.2f}"
    df = df[df["count"] >= LOW_COUNT_THRESHOLD]
    if poly is not None:
        import geopandas as gpd
        inside = gpd.points_from_xy(df["x"], df["y"]).within(poly.geometry.union_all())
        r2 = inside.sum() * pa / poly.area.sum()
        assert 0.8 < r2 < 1.2, f"{site}: cells in the cut polygon x px area / polygon area = {r2:.2f}"
        df = df[inside]
    return pd.DataFrame({"source": source, "site": site, "x": df["x"].to_numpy(), "y": df["y"].to_numpy(),
                         "mean": df["mean"].to_numpy(), "p98": df["p98"].to_numpy(), CV: df[f"cover_{H}"].to_numpy(),
                         "count": df["count"].to_numpy()})


def load_site(chm_p, poly_p):
    import geopandas as gpd
    import rioxarray as rxr
    chm = assert_crs(rxr.open_rasterio(chm_p, masked=True).squeeze(drop=True), str(chm_p)).rio.write_nodata(np.nan)
    poly = assert_crs(gpd.read_file(poly_p), str(poly_p)) if poly_p else None
    return chm, poly


def label_cells(years, ss):
    """One read per CHM, aggregated onto every year's grid. -> {year: DataFrame}."""
    refs = {y: ref_grid(y) for y in years}
    out = {y: [] for y in years}
    for source, site, chm_p, poly_p in ss:
        t0 = time.time()
        print(f"[chm] {site}: {chm_p}" + (f" | polygon {poly_p}" if poly_p else ""), flush=True)
        chm, poly = load_site(chm_p, poly_p)
        for y in years:
            out[y].append(site_cells(source, site, chm, refs[y], poly))
        print(f"[cells] {site}: {', '.join(f'{y} {len(out[y][-1]):,}' for y in years)} cells "
              f"({time.time() - t0:.0f} s)", flush=True)
        del chm
    return {y: pd.concat(v, ignore_index=True) for y, v in out.items()}


def on_stack(lab, tf, shape, year):
    """x / y -> stack row / col (classify_retro.label_table asserts); one row per stack px."""
    lab["col"] = np.rint((lab.x - tf.c) / tf.a - 0.5).astype(int)
    lab["row"] = np.rint((lab.y - tf.f) / tf.e - 0.5).astype(int)
    assert lab.col.between(0, shape[1] - 1).all() and lab.row.between(0, shape[0] - 1).all(), f"{year}: off the stack"
    assert (np.abs(tf.c + (lab.col + 0.5) * tf.a - lab.x) < 0.01 * abs(tf.a)).all(), f"{year}: x off the stack grid"
    assert (np.abs(tf.f + (lab.row + 0.5) * tf.e - lab.y) < 0.01 * abs(tf.e)).all(), f"{year}: y off the stack grid"
    dup = lab.duplicated(["row", "col"], keep=False)
    assert (lab[dup].source == "Yawuru 2025").all(), f"{year}: Dampier px shared: {lab[dup].site.unique()}"
    if dup.any():
        print(f"[labels] {year}: {int(dup.sum())} rows on shared px ({lab[dup].groupby('site').size().to_dict()}); "
              f"keep the higher count", flush=True)
        lab = lab.sort_values("count", ascending=False).drop_duplicates(["row", "col"]).sort_index()
    return lab.reset_index(drop=True)


def sample_stack(path, lab, n_bands):
    """Features at the label px: one window per site (bbox of its px), all bands."""
    import rasterio
    from rasterio.windows import Window
    X = np.full((len(lab), n_bands), np.nan, "float32")
    with rasterio.open(path) as s:
        for _, d in lab.groupby("site"):
            r0, r1, c0, c1 = d.row.min(), d.row.max(), d.col.min(), d.col.max()
            a = s.read(window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
            X[d.index.to_numpy()] = a[:, d.row.to_numpy() - r0, d.col.to_numpy() - c0].T
    return X


def folds(t):
    """fire_gap.folds_for: strata = site x per-site cover quintile; StratifiedKFold(5, shuffle, 42)."""
    from sklearn.model_selection import StratifiedKFold
    q = t.groupby("site")[CV].rank(method="first", pct=True).mul(5).clip(upper=4.999).astype(int)
    f = np.full(len(t), -1)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(t, t.site + "_" + q.astype(str))):
        f[te] = k
    assert (f >= 0).all()
    return f


def cells_prov(y, lidar, ss, si):
    return dict(lidar=lidar, grid=list(tuple(si.tf)[:6]), shape=list(si.shape), crs=EPSG,
                chms=[file_sig(c) for _, _, c, _ in ss], polys=[file_sig(p) for _, _, _, p in ss if p])


def prep_quick(lidar):
    """F1 + code-path check. Writes nothing."""
    import rioxarray as rxr
    from chm_targets import build_sar_reference
    si = stack_info("2025")
    ss = sites(lidar)
    ss = [ss[0], next(s for s in ss if s[1] == "YJ25_1")]
    cells = label_cells(["2025"], ss)["2025"]
    # F1: unf != filtered at the same px
    filt = site_cells("Dampier 2026", ss[0][1], load_site(C.chm_path(ss[0][1], "filt"), None)[0], ref_grid("2025"), None)
    m = cells[cells.site == ss[0][1]].merge(filt, on=["x", "y"], suffixes=("", "_f"))
    dmean, dcov = (m["mean"] - m["mean_f"]).abs(), (m[CV] - m[f"{CV}_f"]).abs()
    print(f"[quick] {ss[0][1]} {lidar} vs filt: {len(m):,} px matched; |d mean| median {dmean.median():.4f} max "
          f"{dmean.max():.3f} m; |d cover| median {dcov.median():.4f}; mean d cover {(m[CV] - m[f'{CV}_f']).mean():+.4f}",
          flush=True)
    if lidar == "unf":
        assert (dmean > 1e-4).mean() > 0.5, "unf CHM gives the filtered values: chm_path variant not used"
    # code path: the original 32751 Yawuru CHM on the 32751 PALSAR grid (same CRS, no relabel) = retro_2d chm_targets
    lab_dir = C.PROJ / "retro_2d" / "outputs" / "2025" / "labels"
    from chm_class_histograms import to_local
    from chm_targets import AREA_NAMES, LIDAR_PATHS_WIN, LIDAR_POLYS_WIN
    i = AREA_NAMES.index("YJ25_1")
    import geopandas as gpd
    c32 = rxr.open_rasterio(to_local(LIDAR_PATHS_WIN[i]), masked=True).squeeze(drop=True).rio.write_nodata(np.nan)
    p32 = gpd.read_file(to_local(LIDAR_POLYS_WIN[i]))
    assert c32.rio.crs.to_epsg() == p32.crs.to_epsg() == 32751
    old = site_cells("Yawuru 2025", "YJ25_1", c32, build_sar_reference("2025"), p32)
    r = rxr.open_rasterio(lab_dir / "chm_targets_YJ25_1.tif", masked=True).isel(band=0)
    ref = np.array([float(r.sel(x=x, y=y, method="nearest")) for x, y in zip(old.x[:2000], old.y[:2000])])
    diff = np.abs(ref - old["mean"].to_numpy()[:2000])
    print(f"[quick] YJ25_1 code path (32751 CHM on the 32751 PALSAR grid) vs retro_2d chm_targets band 1: max |d| "
          f"{np.nanmax(diff):.2e}", flush=True)
    assert np.nanmax(diff) < 1e-4, "YJ25_1: code path differs from chm_targets"
    # CRS effect at Yawuru: same CHM, 7851 grid (moved by the epoch transform) vs the 32751 grid, matched by row / col
    new = on_stack(cells[cells.site == "YJ25_1"].copy(), si.tf, si.shape, "2025")
    rtf = build_sar_reference("2025").rio.transform()
    stf = rasterio_tf(json.loads(MGA_LOG.read_text())[str(si.path)]["src"])
    assert np.allclose(tuple(stf)[:6], tuple(rtf)[:6], atol=1e-6 * abs(rtf.a)), "source stack grid != PALSAR grid"
    old = on_stack(old.copy(), stf, si.shape, "2025")
    mm = new.merge(old, on=["row", "col"], suffixes=("", "_o"))
    d = (mm["mean"] - mm["mean_o"]).abs()
    print(f"[quick] YJ25_1 7851 grid vs 32751 grid (move {si.move[0]:+.3f} / {si.move[1]:+.3f} m): {len(new):,} / "
          f"{len(old):,} px, {len(mm):,} matched; |d mean| median {d.median():.4f}, p99 {d.quantile(0.99):.4f} m",
          flush=True)
    assert len(mm) > 0.95 * len(old) and d.median() < 0.05
    print("[quick] prep passed (nothing written)", flush=True)


def rasterio_tf(p):
    import rasterio
    with rasterio.open(p) as s:
        return s.transform


def prep(years, lidar):
    OUTT.mkdir(parents=True, exist_ok=True)
    ss = sites(lidar)
    for s in ss:
        print(f"[chm] {s[1]}: {s[2]}", flush=True)
    info = {y: stack_info(y) for y in years}
    cache = {y: OUTT / f"cells_{y}.parquet" for y in years}
    want = {y: cells_prov(y, lidar, ss, info[y]) for y in years}
    reuse = {y: cache_ok([cache[y]], want[y]) for y in years}  # stops on a stale cache before any work
    todo = [y for y in years if not reuse[y]]
    cells = {y: pd.read_parquet(cache[y]) for y in years if reuse[y]}
    if reuse and any(reuse.values()):
        print(f"[prep] cells re-used (provenance matches): {[y for y in years if reuse[y]]}", flush=True)
    if todo:
        new = label_cells(todo, ss)
        for y in todo:
            new[y].to_parquet(cache[y], index=False)
            prov_write(cache[y], want[y])
        cells |= new
    for y in years:
        t0 = time.time()
        si = info[y]
        c = cells[y]
        bad = (c.p98 < c["mean"] - 1e-6) | ~c[CV].between(0, 1)
        if bad.any():
            print(f"[prep] {y}: {int(bad.sum())} cells fail p98 >= mean or cover in [0, 1]:\n"
                  f"{c[bad].assign(d=c.p98 - c['mean']).sort_values('d').head(10).to_string()}", flush=True)
        inside = inside_mask("mainland", si.tf, si.shape)
        lab = on_stack(c.copy(), si.tf, si.shape, y)
        X = sample_stack(si.path, lab, len(si.bands))
        good = np.isfinite(X).all(1) & inside[lab.row.to_numpy(), lab.col.to_numpy()]
        print(f"[prep] {y}: {len(lab):,} label px, {int((~good).sum()):,} dropped for a non-finite feature or outside "
              f"the mainland ({lab[~good].groupby('site').size().to_dict()})", flush=True)
        lab = pd.concat([lab, pd.DataFrame(X, columns=si.bands)], axis=1)[good].reset_index(drop=True)
        lab.insert(2, "group", np.where(lab.source == "Dampier 2026", "Dampier",
                                        np.where(lab.site.isin(BURNT), "Yawuru burnt", "Yawuru kept")))
        if y == "2026":  # divergence 3 / 10: as the old 2026 run (Gate A)
            lab = lab[lab.group != "Yawuru burnt"].reset_index(drop=True)
        # A cell can have p98 < mean (a thin tall tail lifts the mean; 1 cell in 2019 / 2023 / 2025, -3 to -6 mm).
        # Kept; the delta head (p98 >= mean) is valid while the share is < 0.5 % (PLAN_mlp_mte.md rule).
        lt = float((lab.p98 < lab["mean"] - 1e-6).mean())
        print(f"[prep] {y}: label px with p98 < mean {lt:.4%}", flush=True)
        assert lt < 0.005, f"{y}: label p98 < mean in {lt:.2%} of px: drop the delta head"
        assert lab[CV].between(0, 1).all(), f"{y}: cover outside [0, 1]"
        lab["fold"] = folds(lab)
        ydir(y).mkdir(parents=True, exist_ok=True)
        rows = ydir(y) / f"rows_{y}.parquet"
        lab.to_parquet(rows, index=False)
        json.dump(dict(year=y, bands=si.bands, stack=str(si.path), n_rows=len(lab), pixel=float(si.tf.a),
                       shape=list(si.shape), crs=EPSG, crs_op=si.op, origin_move=list(si.move), lidar=lidar,
                       rows_sha1=sha1(rows), chms=want[y]["chms"]),
                  open(ydir(y) / f"meta_{y}.json", "w"), indent=1)
        print(f"[prep] {y}: wrote {rows} ({time.time() - t0:.0f} s)\n"
              f"{lab.groupby('group')[YC].describe().T.round(3).to_string()}", flush=True)
        if y == "2026":
            check_2026_features(lab, si.bands)
        del X


def check_2026_features(lab, bands):
    """Verification: features at the label px = the old 2026 feature tables (same stack values, same px index)."""
    old = pd.concat([pd.read_parquet(C.OUT / "features_2026_s1.parquet").drop(columns="mean"),
                     pd.read_parquet(C.OUT / "features_yawuru_s1.parquet")])
    m = lab[KEY + bands].merge(old, on=KEY, suffixes=("", "_o"))
    err = max(float(np.abs(m[b].to_numpy(float) - m[f"{b}_o"].to_numpy(float)).max()) for b in bands)
    print(f"[prep] 2026: {len(m):,} of {len(lab):,} label px in the old feature tables; max |d feature| {err:.2e}",
          flush=True)
    assert err < 1e-6, "2026 features at the label px != old feature tables"


# ------------------------------------------------------------------------------------------- kfold (env pytorch)

def load_year(y):
    meta = json.load(open(ydir(y) / f"meta_{y}.json"))
    rows = ydir(y) / f"rows_{y}.parquet"
    assert sha1(rows) == meta["rows_sha1"], f"{rows}: SHA-1 != meta (rows changed after prep)"
    d = pd.read_parquet(rows)
    b = meta["bands"]
    return d, b, d[b].to_numpy("float64"), d[YC].to_numpy("float64"), d.fold.to_numpy(), meta


def best_cfg():
    import mlp_mte_2026 as E
    cfg = json.load(open(E.OUTE / "best_config.json"))["cfg"]
    cfg["w"] = tuple(cfg["w"])
    return cfg


def fm_rows(model, t, seed, d, y, p):
    x = d[["group", "fold"]].assign(y=y, p=p)
    rec = []
    for g in GROUPS:
        xx = x if g == "pooled" else x[x.group == g]
        for k, xf in xx.groupby("fold"):
            rec.append(dict(model=model, target=t, seed=seed, group=g, fold=k, **reg(xf.y.to_numpy(), xf.p.to_numpy())))
    return rec


def kfold(years, quick=False):
    import mlp_mte_2026 as E
    import mlp_regression_2026 as M
    import p98_cover as P
    cfg = best_cfg()
    for y in years:
        t0 = time.time()
        d, bands, X, Y, fold, meta = load_year(y)
        lab_lt = float((d.p98 < d["mean"]).mean())
        print(f"[kfold] {y}: {len(d):,} rows x {len(bands)} features; cfg {cfg}; label p98 < mean {lab_lt:.3%}; "
              f"groups {d.group.value_counts().to_dict()}", flush=True)
        seeds, kw = (MSEEDS[:2], dict(folds=[0], max_ep=30)) if quick else (MSEEDS, {})
        mem = {}
        for s in seeds:
            mem[s] = E.kfold_mt(X, Y, fold, cfg, s, **kw)
            te = ~np.isnan(mem[s]["oof"][:, 0])
            print(f"[kfold {(time.time() - t0) / 60:5.1f} min] {y} member {s}: R2 "
                  f"{np.round(E.score3(Y[te], mem[s]['oof'][te]), 4)}, best ep {mem[s]['best_ep']}", flush=True)
        Pm = np.stack([mem[s]["oof"] for s in seeds])
        oof, sd = Pm.mean(0), Pm.std(0, ddof=1)
        if quick:
            te = fold == 0
            p = oof[te]
            assert (p[:, :2] > 0).all() and ((p[:, 2] > 0) & (p[:, 2] < 1)).all(), "outputs outside bounds"
            assert (p[:, 1] >= p[:, 0]).all() or cfg["head"] != "delta", "delta head: p98 < mean"
            assert not np.allclose(Pm[0][te], Pm[1][te]), "member seeds give the same preds"
            e = E.Ens([mem[s]["models"][0] for s in seeds])
            assert np.allclose(e.predict3(X[te])[0], p, atol=1e-10), "ensemble != member mean"
            ridge = [r2_ridge(M, X, Y[:, j], fold, te) for j in range(3)]
            print(f"[quick] {y} fold 0 ensemble R2 {np.round(E.score3(Y[te], p), 3)}; ridge {np.round(ridge, 3)}; "
                  f"test rows clamped {(e.n_clamped(X[te]) > 0).mean():.1%}; bounds, p98 >= mean, seeds differ, "
                  f"ensemble = member mean: ok", flush=True)
            continue
        od = ydir(y)
        for j, t in enumerate(T3):
            d[KEY + ["group", "fold"]].assign(y=Y[:, j], pred=oof[:, j], sd=sd[:, j]).to_parquet(od / f"pred_MTE_{t}.parquet")
        fm = []
        for j, t in enumerate(T3):
            fm += fm_rows("MTE", t, 0, d, Y[:, j], oof[:, j])
            fm += fm_rows("ridge", t, 0, d, Y[:, j], M.ridge_oof(X, Y[:, j], fold))
            for s in seeds:
                fm += fm_rows("member", t, s, d, Y[:, j], Pm[s][:, j])
        pd.DataFrame(fm).to_csv(od / "fold_metrics.csv", index=False)
        # residual r = p98 - g(cover), g = E[p98 | cover] on the training-fold labels (p98_cover.kfold)
        rt, rh = np.full(len(d), np.nan), np.full(len(d), np.nan)
        for k in range(5):
            tr, te = fold != k, fold == k
            g = P.hgb(Y[tr, 2:3], Y[tr, 1])
            rt[te] = Y[te, 1] - g.predict(Y[te, 2:3])
            rh[te] = oof[te, 1] - g.predict(oof[te, 2:3])
        d[KEY + ["group", "fold"]].assign(y=rt, pred=rh).to_parquet(od / "pred_MTE_resid.parquet")
        pd.DataFrame(fm_rows("MTE", "resid", 0, d, rt, rh)).to_csv(od / "resid_fold_metrics.csv", index=False)
        ens_f = [E.Ens([mem[s]["models"][k] for s in seeds]) for k in range(5)]
        grp = sensor_groups(bands)
        imp = []
        for j, t in enumerate(T3):
            imp += [dict(target=t, **x) for x in M.perm_sensor([E.View(ens_f[k], j) for k in range(5)], X, Y[:, j],
                                                               fold, grp, np.random.default_rng(0))]
        pd.DataFrame(imp).to_csv(od / "importance.csv", index=False)
        eps = sum((mem[s]["best_ep"] for s in seeds), [])
        ran = sum((mem[s]["ran"] for s in seeds), [])
        ncl = np.zeros(len(d), int)
        for k in range(5):
            ncl[fold == k] = ens_f[k].n_clamped(X[fold == k])
        misc = dict(year=y, n_rows=len(d), rows_sha1=meta["rows_sha1"], label_p98_lt_mean=lab_lt,
                    oof_p98_lt_mean=float((oof[:, 1] < oof[:, 0]).mean()),
                    best_ep_mean=float(np.mean(eps)), best_ep_max=int(max(eps)), fits=len(ran),
                    fits_at_max_ep=int(sum(n >= M.MAX_EP for n in ran)), test_rows_clamped=float((ncl > 0).mean()),
                    train_r2=dict(zip(T3, np.mean([mem[s]["train_r2"] for s in seeds], 0).tolist())),
                    kfold_r2=dict(zip(T3, E.score3(Y, oof))),
                    sd_spearman={t: float(pd.Series(sd[:, j]).corr(pd.Series(np.abs(oof[:, j] - Y[:, j])), "spearman"))
                                 for j, t in enumerate(T3)})
        json.dump(misc, open(od / "misc.json", "w"), indent=1)
        print(f"[kfold] {y} done {(time.time() - t0) / 60:.1f} min: pooled OOF R2 "
              f"{ {t: round(v, 3) for t, v in misc['kfold_r2'].items()} }; fits at max epoch {misc['fits_at_max_ep']}"
              f" of {misc['fits']}", flush=True)


def r2_ridge(M, X, y, fold, te):
    from sklearn.metrics import r2_score
    return r2_score(y[te], M.ridge_oof(X, y, fold)[te])


# ------------------------------------------------------------------------------------------- maps (env pytorch)

def maps(years):
    import rasterio
    import mlp_mte_2026 as E
    import mlp_regression_2026 as M
    import p98_cover as P
    assert P.NODATA == NODATA
    cfg = best_cfg()
    for y in years:
        t0 = time.time()
        d, bands, X, Y, fold, meta = load_year(y)
        md, tg = mdir(y), tag(y)
        want = dict(rows_sha1=meta["rows_sha1"], cfg={k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
                    seeds=MSEEDS)
        pts = [member_path(y, s) for s in MSEEDS]
        if cache_ok(pts, want):
            members = [E.MT.load(p, len(bands)) for p in pts]
            print(f"[maps] {y}: members loaded (provenance matches)", flush=True)
        else:
            members = []
            for s, p in zip(MSEEDS, pts):
                m, h, be = E.fit(X, Y, cfg, s)
                m.save(p)
                members.append(m)
                print(f"[maps {(time.time() - t0) / 60:5.1f} min] {y} member {s}: best ep {be + 1}, ran {len(h)} -> {p}",
                      flush=True)
            for p in pts:
                prov_write(p, want)
        ens = E.Ens(members)
        outs = {**{t: md / f"{MAPS[t]}_MTE{tg}.tif" for t in T3}, **{t + "sd": md / f"{MAPS[t]}_MTEsd{tg}.tif" for t in T3},
                "ncl": md / f"n_clamped_MTE{tg}.tif"}
        with rasterio.open(meta["stack"]) as s:
            assert s.crs.to_epsg() == EPSG and list(s.descriptions) == bands, "stack CRS / band order != training"
            tf, shape = s.transform, (s.height, s.width)
            fin = np.ones(shape, bool)
            for _, w in s.block_windows(1):
                a = s.read(window=w)
                fin[w.row_off:w.row_off + w.height, w.col_off:w.col_off + w.width] = np.isfinite(a).all(0)
        masks = {b: inside_mask(b, tf, shape) for b in BOUNDARIES_WIN}
        valid = fin & masks["mainland"]
        print(f"[maps] {y}: finite stack {fin.sum():,} px, finite and mainland {valid.sum():,} px"
              + (" (old 2026 RF mask 64,392,072)" if y == "2026" else ""), flush=True)
        if cache_ok(list(outs.values()), want):
            print(f"[maps] {y}: skip rasters (exist, provenance matches)", flush=True)
        else:
            E.predict_multi(ens, list(range(1, len(bands) + 1)), SimpleNamespace(STACK_TIF=meta["stack"]), outs,
                            mask=valid)
            for p in outs.values():
                prov_write(p, want)
        # checks: map = ensemble at the label px; valid mask = finite stack and mainland; CRS 7851
        mu = ens.predict3(X)[0]
        rr, cc = d.row.to_numpy(), d.col.to_numpy()
        for j, t in enumerate(T3):
            for k in (t, t + "sd"):
                a, prof = M.read_nan(outs[k])
                assert (np.isfinite(a) == valid).all(), f"{y} {k}: valid mask != finite stack and mainland"
                if k == t:
                    assert np.allclose(a[rr, cc], mu[:, j].astype("float32"), atol=1e-5), f"{y} {t}: map != ensemble"
                del a
        with rasterio.open(outs["mean"]) as s:
            assert s.crs.to_epsg() == EPSG, f"{outs['mean']}: {s.crs}"
        print(f"[maps {(time.time() - t0) / 60:5.1f} min] {y}: map = ensemble at {len(d):,} label px; valid mask = "
              f"finite stack and mainland ({valid.sum():,} px); CRS EPSG:{EPSG}", flush=True)
        res, resw, prof, gy = M.resid_maps(outs["p98"], outs["cover"])
        P.write(res, md / f"p98_resid_MTE{tg}.tif", prof)
        P.write(resw, md / f"p98_resid_w{P.WIN}_MTE{tg}.tif", prof)
        del res, resw
        rec = []
        prods = {**{t: outs[t] for t in T3}, "resid": md / f"p98_resid_MTE{tg}.tif",
                 f"resid_w{P.WIN}": md / f"p98_resid_w{P.WIN}_MTE{tg}.tif"}
        lab = dict(zip(T3, YC))
        for name, p in prods.items():
            a = M.read_nan(p)[0]
            v = a[np.isfinite(a)]
            r = dict(year=y, product=name, n_valid=v.size, map_min=v.min(), map_max=v.max(),
                     **{f"mean_{b}": float(np.nanmean(a[m])) for b, m in masks.items()})
            if name in lab:
                yl = d[lab[name]]
                r.update(lt_005=np.mean(v < 0.05), label_lt_005=np.mean(yl < 0.05), above_label_max=np.mean(v > yl.max()),
                         label_max=yl.max(), label_mean=yl.mean())
            rec.append(r)
            del a, v
        a, b = M.read_nan(outs["p98"])[0], M.read_nan(outs["mean"])[0]
        ok = np.isfinite(a) & np.isfinite(b)
        p_lt = float((a[ok] < b[ok]).mean())
        del a, b
        mc = pd.DataFrame(rec)
        mc["map_p98_lt_mean"] = p_lt
        mc.to_csv(ydir(y) / "map_checks.csv", index=False)
        print(f"[maps] {y} done {(time.time() - t0) / 60:.1f} min; g(cover) on the map: g(0) {gy[0]:.2f}, g(0.5) "
              f"{gy[500]:.2f}, g(1) {gy[-1]:.2f} m\n{mc.round(3).to_string(index=False)}", flush=True)


# ------------------------------------------------------------------------------------------- report

def report(years):
    from cover_height_report import summarise
    head, res, imp, eff, mc = [], [], [], [], []
    for y in sorted(years):
        od = ydir(y)
        if not (od / "fold_metrics.csv").exists():
            continue
        fm = pd.read_csv(od / "fold_metrics.csv")
        for val in ["r2", "rmse", "bias"]:
            s = summarise(fm[fm.model.isin(["MTE", "ridge"])], ["model", "target", "group"], val).m.rename(val)
            head.append(s.reset_index().assign(year=y))
        rr = pd.read_csv(od / "resid_fold_metrics.csv")
        res.append(summarise(rr, ["group"], "r2").m.rename(y))
        im = pd.read_csv(od / "importance.csv").groupby(["target", "sensor"]).dR2.mean().unstack("sensor")
        imp.append(im.assign(year=y).reset_index().set_index(["year", "target"]))
        m = json.load(open(od / "misc.json"))
        eff.append(dict(year=y, rows=m["n_rows"], fits_at_max=f"{m['fits_at_max_ep']}/{m['fits']}",
                        best_ep=round(m["best_ep_mean"]), clamped_rows=round(m["test_rows_clamped"], 3),
                        oof_p98_lt_mean=round(m["oof_p98_lt_mean"], 4),
                        **{f"train-kfold {t}": round(m["train_r2"][t] - m["kfold_r2"][t], 3) for t in T3},
                        **{f"rho(sd,|e|) {t}": round(m["sd_spearman"][t], 2) for t in T3}))
        if (od / "map_checks.csv").exists():
            mc.append(pd.read_csv(od / "map_checks.csv"))
    h = pd.concat(head).groupby(["year", "target", "group", "model"])[["r2", "rmse", "bias"]].first()
    mte = h.xs("MTE", level="model").unstack("group")
    rdg = h.xs("ridge", level="model").r2.unstack("group")
    cols = [g for g in GROUPS if g in mte.r2.columns]
    L = ["REPORT — MTE regression 2019-2026, unfiltered Dampier 2026 LiDAR, project CRS EPSG:7851 (mte_retro.py)",
         "Plan: C:\\Users\\glens\\.claude\\plans\\we-re-doing-the-regression-snazzy-graham.md (old: PLAN_mte_retro.md;",
         "pre-unf result: REPORT_mte_retro_pre_unf.txt). Change vs pre-unf: REPORT_unf_change.txt.",
         "Method = mlp_mte_2026.py (multi-task bounded MLP, 5 members, input clamp), 2026 best config, one model per year.",
         "Stacks: 2019-2025 PALSAR + S1 + S2 (20 bands); 2026 NISAR + S1 + S2 (49 bands). All in EPSG:7851 (stack origin",
         "moved by ITRF2014 to GDA2020 (1), EPSG:8048, at the year's epoch; values unchanged).",
         "Labels: 5 Dampier 2026 sites (unfiltered CHMs) + 13 Yawuru 2025 areas. 2019-2025: burnt areas kept (user",
         "2026-11-03: fire = noise). 2026: 'Yawuru burnt' dropped (as the old 2026 run).",
         "Targets per stack px from the 50 cm CHM: mean, p98, cover 1.37. K-fold = stratified 5-fold (site x cover",
         "quintile). Random K-fold under spatial autocorrelation: optimistic.", "",
         "1. K-FOLD R2 (fold mean), MTE", mte.r2[cols].round(3).to_string(), "",
         "   ridge R2 (floor)", rdg.reindex(columns=cols).round(3).to_string(), "",
         "   RMSE (m; cover fraction)", mte.rmse[cols].round(3).to_string(), "",
         "   bias (pred - label)", mte.bias[cols].round(3).to_string(), "",
         "2. p98 RESIDUAL r = p98 - g(cover), K-fold R2", pd.concat(res, axis=1).T.reindex(columns=cols).round(3).to_string(), "",
         "3. IMPORTANCE — permutation R2 drop by sensor (fold mean)", pd.concat(imp).round(3).to_string(), "",
         "4. EFFECTIVENESS", pd.DataFrame(eff).to_string(index=False), ""]
    if mc:
        L += ["5. MAPS (E:\\nisar_2026\\regression\\retro\\<year>\\, 2026: E:\\nisar_2026\\regression\\; map = ensemble at",
              "   the label px, valid mask = finite stack and mainland, CRS 7851: asserted)",
              pd.concat(mc).round(3).to_string(index=False), ""]
    L += ["CAVEATS", "- Acquisition year is recoverable from the bands: every cross-year statement is confounded.",
          "- Years are on different grids (2025: 24.28 m, own shape; 2026: 20 m NISAR). Reproject or aggregate before",
          "  differencing.",
          "- 2025 / 2026 LiDAR labels train every year: accuracy degrades backwards in time.",
          "- Fire between each image year and the LiDAR flight is not handled (user: noise). 'Yawuru burnt' = the 6 areas",
          "  burnt 2025 -> 2026 (Gate A); for earlier years other fires (Dampier and Yawuru) are also unflagged.",
          "- Stack frame epoch = year + 0.54 (S1 / S2 July composites; PALSAR annual mosaic date not resolved), 2026.75.",
          "  Stack move 0.03 m (2019) to 0.47 m (2026), <= 2.4 % of a px.",
          "- Yawuru 2025 frame = GDA2020: user statement + AUSPOS reports (GDA2020 values); the base entry in DJI Terra is not",
          "  on disk.",
          "- 2020 and 2022: no PALSAR mosaic; not in the series."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    print(f"\nwrote {REPORT}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "kfold", "maps", "report"])
    ap.add_argument("--years", nargs="+", default=YEARS, choices=YEARS)
    ap.add_argument("--lidar", default="unf", choices=["unf", "filt"], help="Dampier 2026 CHM variant (default unf)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=10, help="CPU threads for torch (cap 10)")
    a = ap.parse_args()
    if a.step in ("kfold", "maps"):
        import torch
        torch.set_num_threads(min(a.threads, 10))
        print(f"device {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    if a.step == "prep":
        return prep_quick(a.lidar) if a.quick else prep(a.years, a.lidar)
    if a.step == "kfold":
        return kfold(a.years, a.quick)
    {"maps": maps, "report": report}[a.step](a.years)


if __name__ == "__main__":
    main()
