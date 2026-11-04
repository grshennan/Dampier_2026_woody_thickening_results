"""p98 residual time pattern 2019-2026 (user 2026-11-03): consistently red / blue, and trends red <-> blue.

  grid    per-year MTE p98 residual (per px) -> common 100 m EPSG:7851 grid (project CRS), area mean (same CRS: resample, no
          relabel); cell valid if >= 50 % of its area is valid -> trend/p98_resid_MTE_stack_2019_2026_100m.tif
  stats   domain = mainland cells valid in all 6 years. Per-year percentile rank over the domain; red <= 25,
          blue >= 75. Mann-Kendall S (exact one-sided p) on rank vs year, Theil-Sen slope on the true years.
          -> trend/p98_resid_pattern_2019_2026_100m.tif (uint8 classes), p98_resid_trend_2019_2026_100m.tif
          (float32 metrics + ranks), p98_resid_pattern_2019_2025_100m.tif (PALSAR years only, sensitivity).
          Nulls: per-cell random year order (trend classes), random toroidal shift per year (consistency).
  report  -> REPORT_resid_trend.txt

Inputs: E:\\nisar_2026\\regression\\retro\\<year>\\p98_resid_MTE_<year>.tif (2019-2025, PALSAR + S1 + S2) and
E:\\nisar_2026\\regression\\p98_resid_MTE.tif (2026, NISAR + S1 + S2). r = p98 - g(cover), g refit on each map.

    C:\\Users\\glens\\miniconda3\\Scripts\\conda.exe run -n general --no-capture-output python -u resid_trend.py grid
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import mte_retro as MR

YEARS = [2019, 2021, 2023, 2024, 2025, 2026]
SRC = {y: MR.MAPR / str(y) / f"p98_resid_MTE_{y}.tif" for y in YEARS[:-1]}
SRC[2026] = Path(r"E:\nisar_2026\regression\p98_resid_MTE.tif")
TD = Path(r"E:\nisar_2026\regression\trend")
OD = C.OUT / "resid_trend"
REPORT = C.HERE / "REPORT_resid_trend.txt"
RES = 100.0
MIN_VALID = 0.5
RED, BLUE = 25.0, 75.0
NODATA = -9999.0
STACK = TD / "p98_resid_MTE_stack_2019_2026_100m.tif"
CLASSES = {0: "No pattern", 1: "Consistently red", 2: "Consistently blue", 3: "Red -> blue", 4: "Toward blue",
           5: "Blue -> red", 6: "Toward red"}
SEED = 42


def out_names(years):
    tag = f"{years[0]}_{years[-1]}"
    return TD / f"p98_resid_pattern_{tag}_100m.tif", TD / f"p98_resid_trend_{tag}_100m.tif"


# ------------------------------------------------------------------------------------------- grid

def common_grid():
    import rasterio
    from rasterio.transform import from_origin
    b = []
    for y, p in SRC.items():
        with rasterio.open(p) as s:
            assert s.crs.to_epsg() == 7851, f"{p}: {s.crs}"
            b.append(s.bounds)
    x0 = np.floor(min(v.left for v in b) / RES) * RES
    x1 = np.ceil(max(v.right for v in b) / RES) * RES
    y0 = np.floor(min(v.bottom for v in b) / RES) * RES
    y1 = np.ceil(max(v.top for v in b) / RES) * RES
    return from_origin(x0, y1, RES, RES), (int(round((y1 - y0) / RES)), int(round((x1 - x0) / RES)))


def write(path, arr, tf, nodata, dtype, names):
    import rasterio
    arr = arr if arr.ndim == 3 else arr[None]
    tmp = path.with_name(path.stem + ".partial.tif")
    prof = dict(driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0], dtype=dtype,
                crs="EPSG:7851", transform=tf, nodata=nodata, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(tmp, "w", **prof) as s:
        s.write(arr.astype(dtype))
        for i, n in enumerate(names, 1):
            s.set_band_description(i, n)
    tmp.replace(path)
    print(f"wrote {path}", flush=True)


def grid():
    import rasterio
    from rasterio.warp import Resampling, reproject
    TD.mkdir(parents=True, exist_ok=True)
    OD.mkdir(parents=True, exist_ok=True)
    tf, shape = common_grid()
    print(f"grid: EPSG:7851, {RES:.0f} m, {shape[1]} x {shape[0]}, origin ({tf.c:.0f}, {tf.f:.0f})", flush=True)
    stack = np.full((len(YEARS), *shape), np.nan, "float32")
    rec = []
    for i, y in enumerate(YEARS):
        t0 = time.time()
        with rasterio.open(SRC[y]) as s:
            a, stf, scrs = s.read(1), s.transform, s.crs
            assert s.nodata == NODATA, f"{y}: nodata {s.nodata}"
        v = a != NODATA
        a = np.where(v, a, np.nan).astype("float32")
        mean = np.full(shape, np.nan, "float32")
        frac = np.zeros(shape, "float32")
        reproject(a, mean, src_transform=stf, src_crs=scrs, src_nodata=np.nan, dst_transform=tf,
                  dst_crs="EPSG:7851", dst_nodata=np.nan, resampling=Resampling.average)
        reproject(v.astype("float32"), frac, src_transform=stf, src_crs=scrs, dst_transform=tf,
                  dst_crs="EPSG:7851", resampling=Resampling.average)
        ok = np.isfinite(mean) & (frac >= MIN_VALID)
        stack[i][ok] = mean[ok]
        # check: area mean = native px mean over the same cells (px centre -> cell)
        rr, cc = np.nonzero(v)
        xs, ys = stf * (cc + 0.5, rr + 0.5)
        ci = np.floor((np.asarray(xs) - tf.c) / RES).astype(int)
        ri = np.floor((tf.f - np.asarray(ys)) / RES).astype(int)
        keep = ok[ri, ci]
        nat = float(a[rr[keep], cc[keep]].mean())
        agg = float(np.average(stack[i][ok], weights=frac[ok]))
        rec.append(dict(year=y, src=str(SRC[y]), src_res=stf.a, n_cells=int(ok.sum()), native_mean=nat,
                        grid_mean_wtd=agg, d=agg - nat))
        print(f"[grid] {y}: {ok.sum():,} cells; native mean {nat:.4f}, 100 m area-weighted mean {agg:.4f} m "
              f"({time.time() - t0:.0f} s)", flush=True)
        assert abs(agg - nat) < 0.01, f"{y}: 100 m mean != native mean"
        del a, v, rr, cc, xs, ys, ci, ri, keep
    pd.DataFrame(rec).to_csv(OD / "grid_checks.csv", index=False)
    write(STACK, np.where(np.isfinite(stack), stack, NODATA), tf, NODATA, "float32", [str(y) for y in YEARS])


# ------------------------------------------------------------------------------------------- stats

def mk_threshold(n, alpha=0.05):
    """Exact one-sided Mann-Kendall: smallest S with P(S >= s) <= alpha under random order (no ties)."""
    s = np.array([sum(np.sign(p[j] - p[i]) for i, j in itertools.combinations(range(n), 2))
                  for p in itertools.permutations(range(n))])
    for t in range(int(s.max()) + 1):
        if (s >= t).mean() <= alpha:
            return t, s
    raise ValueError


def mk_p(S, sdist):
    """One-sided p in the direction of S (P(S' >= S) for S > 0, P(S' <= S) for S < 0)."""
    vals, cnt = np.unique(sdist, return_counts=True)
    ge = np.array([cnt[vals >= v].sum() for v in vals]) / sdist.size
    le = np.array([cnt[vals <= v].sum() for v in vals]) / sdist.size
    idx = np.searchsorted(vals, S)
    return np.where(S >= 0, ge[idx], le[idx]).astype("float32")


def mk_S(R):
    n = R.shape[1]
    return sum(np.sign(R[:, j] - R[:, i]) for i, j in itertools.combinations(range(n), 2)).astype("int16")


def sen(R, t):
    sl = [(R[:, j] - R[:, i]) / (t[j] - t[i]) for i, j in itertools.combinations(range(len(t)), 2)]
    return np.median(np.stack(sl, 1), 1).astype("float32")


def classify(R, S, s_thr, n_start=2, n_end=2):
    n = R.shape[1]
    red, blue = (R <= RED).sum(1), (R >= BLUE).sum(1)
    st, en = R[:, :n_start].mean(1), R[:, -n_end:].mean(1)
    c = np.zeros(len(R), "uint8")
    up, dn = S >= s_thr, S <= -s_thr
    c[dn] = 6
    c[dn & (st >= BLUE) & (en <= RED)] = 5
    c[up] = 4
    c[up & (st <= RED) & (en >= BLUE)] = 3
    c[(blue >= n - 1) & (red == 0)] = 2
    c[(red >= n - 1) & (blue == 0)] = 1
    return c, red, blue


def shares(c):
    return pd.Series(np.bincount(c, minlength=7)[:7] / c.size, index=list(CLASSES)).rename(lambda k: CLASSES[k])


def stats():
    import rasterio
    from scipy.stats import kendalltau, rankdata, theilslopes
    with rasterio.open(STACK) as s:
        st, tf = s.read(), s.transform
        assert [int(d) for d in s.descriptions] == YEARS
    st = np.where(st == NODATA, np.nan, st)
    shape = st.shape[1:]
    main = MR.inside_mask("mainland", tf, shape)
    pin = MR.inside_mask("pindan", tf, shape)
    allv = np.isfinite(st).all(0)
    dom = main & allv
    n_main = int(main.sum())
    anyv = main & np.isfinite(st).any(0)
    print(f"[stats] mainland {n_main:,} cells; valid any year {anyv.sum():,}; all 6 years (domain) {dom.sum():,} "
          f"({dom.sum() / anyv.sum():.1%} of any-year)", flush=True)
    V = st[:, dom].T  # cells x years, m
    R = np.stack([(rankdata(V[:, i]) - 0.5) / len(V) * 100 for i in range(V.shape[1])], 1).astype("float32")
    rng = np.random.default_rng(SEED)
    rows, res, out = [], {}, {}
    for yrs in (YEARS, YEARS[:-1]):
        k = [YEARS.index(y) for y in yrs]
        Rk, Vk, t = R[:, k], V[:, k], np.array(yrs, float)
        thr, sdist = mk_threshold(len(yrs))
        S = mk_S(Rk)
        c, nred, nblue = classify(Rk, S, thr)
        p = mk_p(S, sdist)
        sr, sm = sen(Rk, t), sen(Vk, t)
        # check: S, Sen vs scipy on 100 random cells
        for i in rng.choice(len(Rk), 100, replace=False):
            tau = kendalltau(t, Rk[i]).statistic
            n2 = len(t) * (len(t) - 1) / 2
            assert abs(tau * n2 - S[i]) < 1e-6, (i, tau * n2, S[i])
            assert abs(theilslopes(Rk[i], t).slope - sr[i]) < 1e-4
        assert np.bincount(c, minlength=7).sum() == dom.sum()
        # null 1: random year order per cell (trend classes)
        nt = []
        for _ in range(50):
            perm = np.argsort(rng.random(Rk.shape), 1)
            Rp = np.take_along_axis(Rk, perm, 1)
            nt.append(shares(classify(Rp, mk_S(Rp), thr)[0]))
        null_order = pd.concat(nt, axis=1).mean(axis=1)
        # null 2: random shift per year (10-30 km, random sign), cross-year correspondence destroyed (consistency)
        full = np.full((len(yrs), *shape), np.nan, "float32")
        for j in range(len(yrs)):
            full[j][dom] = Rk[:, j]
        ns = []
        for _ in range(20):
            sh = np.stack([np.roll(full[j], tuple(rng.integers(100, 300, 2) * rng.choice([-1, 1], 2)), (0, 1))
                           if j else full[j] for j in range(len(yrs))])
            ok = np.isfinite(sh).all(0)
            Rs = sh[:, ok].T
            ns.append(shares(classify(Rs, mk_S(Rs), thr)[0]))
        null_shift = pd.concat(ns, axis=1).mean(axis=1)
        tag = f"{yrs[0]}-{yrs[-1]}"
        obs = shares(c)
        pin_d = pin[dom]
        tab = pd.DataFrame({"cells": np.bincount(c, minlength=7)[:7], "km2": np.bincount(c, minlength=7)[:7] * RES ** 2 / 1e6,
                            "share": obs.values, "null_year_order": null_order.values, "null_shift": null_shift.values,
                            "share_pindan": shares(c[pin_d]).values, "share_not_pindan": shares(c[~pin_d]).values},
                           index=obs.index)
        tab.insert(0, "code", list(CLASSES))
        tab.to_csv(OD / f"classes_{yrs[0]}_{yrs[-1]}.csv")
        print(f"[stats] {tag}: MK threshold S >= {thr} of {len(yrs) * (len(yrs) - 1) // 2} "
              f"(p {(sdist >= thr).mean():.4f})\n{tab.round(4).to_string()}", flush=True)
        res[tag] = dict(thr=int(thr), p_thr=float((sdist >= thr).mean()), n_dom=int(dom.sum()))
        out[tag] = c
        # rasters
        pat, trd = out_names(yrs)
        cl = np.full(shape, 255, "uint8")
        cl[dom] = c
        write(pat, cl, tf, 255, "uint8", [f"pattern {tag} (" + ", ".join(f"{k} {v}" for k, v in CLASSES.items()) + ")"])
        if yrs == YEARS:
            bands = {"sen_rank_per_yr": sr, "sen_m_per_yr": sm, "mk_S": S.astype("float32"), "mk_p_one_sided": p,
                     "mean_rank": Rk.mean(1), "n_red": nred.astype("float32"), "n_blue": nblue.astype("float32"),
                     **{f"rank_{y}": Rk[:, i] for i, y in enumerate(yrs)}}
            arr = np.full((len(bands), *shape), NODATA, "float32")
            for i, v in enumerate(bands.values()):
                arr[i][dom] = v
            write(trd, arr, tf, NODATA, "float32", list(bands))
    a, b = out["2019-2026"], out["2019-2025"]
    agree = pd.crosstab(pd.Series(a, name="2019-2026").map(CLASSES), pd.Series(b, name="2019-2025").map(CLASSES))
    agree.to_csv(OD / "crosstab_2026_vs_2025.csv")
    res["same_class_share"] = float((a == b).mean())
    res.update(n_mainland=n_main, n_any_year=int(anyv.sum()), n_domain=int(dom.sum()),
               n_domain_pindan=int((dom & pin).sum()),
               per_year_resid_sd=[float(np.nanstd(V[:, i])) for i in range(V.shape[1])],
               rank_corr=pd.DataFrame(R, columns=YEARS).corr("spearman").round(3).to_dict())
    (OD / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[stats] same class 2019-2026 vs 2019-2025: {res['same_class_share']:.3f}\n{agree}", flush=True)


# ------------------------------------------------------------------------------------------- report

def report():
    s = json.loads((OD / "summary.json").read_text())
    g = pd.read_csv(OD / "grid_checks.csv")
    corr = pd.DataFrame(s["rank_corr"])
    L = ["REPORT - p98 residual time pattern 2019-2026 (resid_trend.py; user 2026-11-03)",
         "r = p98 - g(cover), MTE maps; 2019-2025 PALSAR + S1 + S2, 2026 NISAR + S1 + S2. Red = short for its cover.",
         f"Grid EPSG:7851 {RES:.0f} m, area mean of per-px r, cell valid if >= {MIN_VALID:.0%} valid.",
         f"Per-year percentile rank over the domain: red <= {RED:.0f}, blue >= {BLUE:.0f}.",
         "Trend: Mann-Kendall S on rank vs year (exact one-sided p <= 0.05); start = mean rank of the first 2 years,",
         "end = mean rank of the last 2.", ""]
    t = pd.read_csv(OD / "classes_2019_2026.csv", index_col=0)
    lo, ls = t.share / t.null_year_order, t.share / t.null_shift
    L += ["RESULT 2019-2026 (share of domain; lift vs its null)",
          *[f" {k:<18} {t.share[k]:6.2%}  lift {(ls if t.code[k] <= 2 else lo)[k]:5.2f}"
            f" ({'shift' if t.code[k] <= 2 else 'year order'} null)" for k in t.index if t.code[k] > 0],
          " Consistency is real (lift >> 1). Trend classes are near chance (lift ~1): a single cell's trend is",
          " not evidence; use trend cells only as candidates, in clusters, after the fire overlay.", "",
          "1. DOMAIN (100 m cells)",
         f"mainland {s['n_mainland']:,} | valid any year {s['n_any_year']:,} | valid all 6 years (domain) "
         f"{s['n_domain']:,} ({s['n_domain'] / s['n_any_year']:.1%}) | domain in pindan {s['n_domain_pindan']:,}", "",
         "grid check (100 m area mean vs native px mean, same cells, m)",
         g[["year", "src_res", "n_cells", "native_mean", "grid_mean_wtd", "d"]].round(4).to_string(index=False), "",
         "per-year residual SD at 100 m (m): " + " ".join(f"{y} {v:.2f}" for y, v in zip(YEARS, s["per_year_resid_sd"])),
         "", "2. RANK CORRELATION BETWEEN YEARS (Spearman, 100 m cells)", corr.to_string(), ""]
    codes = ("class codes: " + ", ".join(f"{k} {v}" for k, v in CLASSES.items()))
    for i, tag in enumerate(["2019-2026", "2019-2025"], 3):
        t = pd.read_csv(OD / f"classes_{tag.replace('-', '_')}.csv", index_col=0)
        t["lift_order"] = t.share / t.null_year_order.replace(0, np.nan)
        t["lift_shift"] = t.share / t.null_shift.replace(0, np.nan)
        L += [f"{i}. CLASSES {tag} (MK S >= {s[tag]['thr']}, exact p {s[tag]['p_thr']:.4f}; consistent = "
              f"red / blue in >= n-1 years, never the other)",
              t.round(4).to_string(), ""]
    L += [codes, "null_year_order = random year order per cell (50 x): trend classes by chance.",
          "null_shift = each year's map shifted by a random offset (10-30 km, 20 x): consistency by chance.",
          "lift = share / null. Use lift_shift for codes 1-2, lift_order for codes 3-6.", "",
          f"5. 2026 POINT: same class with and without 2026: {s['same_class_share']:.3f} of domain cells",
          "   (crosstab: outputs/resid_trend/crosstab_2026_vs_2025.csv)", "",
          "6. FILES (E:\\nisar_2026\\regression\\trend\\)",
          " p98_resid_MTE_stack_2019_2026_100m.tif  6 bands, r (m) per year",
          " p98_resid_pattern_2019_2026_100m.tif    uint8 class codes, 255 = outside domain",
          " p98_resid_trend_2019_2026_100m.tif      sen_rank_per_yr, sen_m_per_yr, mk_S, mk_p_one_sided, mean_rank,",
          "                                          n_red, n_blue, rank_<year> x 6",
          " p98_resid_pattern_2019_2025_100m.tif    sensitivity, PALSAR years only", "",
          "CAVEATS",
          "- Acquisition year is recoverable from the bands. Ranks remove year-wide shifts only, not spatial artefacts.",
          "- PALSAR mosaic strip seams step the maps (retro_2d/REPORT_palsar_date.txt): straight edges = suspect.",
          "- 2026 is another sensor (NISAR L-band) and another model; see section 5.",
          "- Red = short for its cover also after fire (resprouting); red -> blue can be post-fire recovery.",
          "  Fire history (NAFI) not yet overlaid.",
          "- All labels are 2025 / 2026 LiDAR: accuracy degrades backwards in time.",
          "- Per-px r has K-fold R2 ~0.5; 100 m cells are better but not validated over time."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    {"grid": grid, "stats": stats, "report": report}[sys.argv[1]]()
