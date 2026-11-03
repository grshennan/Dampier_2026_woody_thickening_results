"""Regression track, follow-on (user 2026-11-02): (A) p98 from the mean and cover maps; (B) height-cover decorrelation.

  rows    p98 / p90 of the 50 cm CHM cells per 20 m label px (translation.aggregate_with_cover, as common._site_cover)
          for the 39,023 rows (Dampier + Yawuru kept). Asserts mean and count = rows.parquet.
          -> outputs/p98_cover/rows_p98.parquet
  kfold   same folds (folds_cover_height.json), seed 0. p98 by:
            direct    RF ALL (49) -> p98 (RF 500, max_features 0.33, leaf 5, as Part B)
            transfer  f(mean, cover) -> p98, f = HGB fitted on the TRAINING-fold labels; applied to the Part B OOF
                      preds pred_ALL_{mean,cover}_s0 (same folds)
            transfer_mean  f(mean) only (what cover adds)
            oracle    f(true mean, true cover) (ceiling of the transfer)
          Decorrelation: residual r = p98 - g(cover), g = E[p98 | cover] (HGB, training-fold labels).
          r_hat from each p98 approach + cover_hat. Also 5 x 5 px window Pearson r(p98, cover) in the label sites.
          -> outputs/p98_cover/kfold_*.csv, REPORT_p98_cover.txt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

import common as C
import cover_height_rf as R

OUTP = C.OUT / "p98_cover"
REPORT = C.HERE / "REPORT_p98_cover.txt"
KEY = ["site", "row", "col"]
CV = C.col(1.37)
GROUPS = ["pooled", "Dampier", "Yawuru kept"]
WIN = 5


def rows():
    from chm_class_histograms import load_chm, to_local
    from chm_targets import AREA_NAMES, LIDAR_PATHS_WIN, target_grid_for_chm
    d = pd.read_parquet(R.OUTD / "rows.parquet")
    d = d[d.group != "Yawuru burnt"].reset_index(drop=True)
    ref = C.ref_grid()
    tf = ref.rio.transform()
    paths = {s: C.chm_path(s) for s in C.L.SITES}
    paths.update({n: Path(to_local(p)) for n, p in zip(AREA_NAMES, LIDAR_PATHS_WIN)})
    parts = []
    for site in d.site.unique():
        t0 = time.time()
        chm = load_chm(paths[site])
        ds = C.translation.aggregate_with_cover(chm, target_grid_for_chm(chm, ref), heights=[1.37])
        df = ds.drop_vars("spatial_ref", errors="ignore").to_dataframe().reset_index()
        df["col"] = np.round((df["x"] - tf.c) / tf.a - 0.5).astype("int64")
        df["row"] = np.round((df["y"] - tf.f) / tf.e - 0.5).astype("int64")
        df["site"] = site
        parts.append(df[KEY + ["mean", "count", "p90", "p98"]].rename(columns={"mean": "mean_chk", "count": "n_chk"}))
        print(f"{site}: {time.time() - t0:.0f} s", flush=True)
    a = d.merge(pd.concat(parts), on=KEY, how="left", validate="1:1")
    assert a.p98.notna().all(), "label px without p98"
    assert np.allclose(a["mean"], a.mean_chk, atol=1e-4) and (a["count"] == a.n_chk).all(), "re-aggregation differs"
    assert (a.p98 >= a["mean"] - 1e-6).all()
    OUTP.mkdir(exist_ok=True)
    a.drop(columns=["mean_chk", "n_chk"]).to_parquet(OUTP / "rows_p98.parquet")
    print(a.groupby("group")[["mean", "p90", "p98", CV]].describe().T.round(3).to_string())


def hgb(X, y):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=0).fit(X, y)


def load_all():
    d, base, _ = R.load(with_glcm=False)
    d = d[d.group != "Yawuru burnt"].reset_index(drop=True)
    p = pd.read_parquet(OUTP / "rows_p98.parquet")[KEY + ["p90", "p98"]]
    d = d.merge(p, on=KEY, how="left", validate="1:1")
    for t in ["mean", "cover"]:
        q = pd.read_parquet(R.OUTD / f"pred_ALL_{t}_s0.parquet")
        assert (q[KEY + ["fold"]].to_numpy() == d[KEY + ["fold"]].to_numpy()).all(), "row order / folds differ"
        d[f"{t}_hat"] = q.pred.to_numpy()
    assert d.p98.notna().all() and len(d) == 39023
    return d, base


def win_corr(d, a, b):
    """Pearson r of columns a, b over the WIN x WIN px window around each label px (>= 20 label px in it)."""
    out = np.full(len(d), np.nan)
    h = WIN // 2
    for _, g in d.groupby("site"):
        r0, c0 = g.row.min(), g.col.min()
        H, W = g.row.max() - r0 + 1, g.col.max() - c0 + 1
        A, B = np.full((H, W), np.nan), np.full((H, W), np.nan)
        A[g.row - r0, g.col - c0], B[g.row - r0, g.col - c0] = g[a], g[b]
        for i, (r, c) in zip(g.index, zip(g.row - r0, g.col - c0)):
            x, y = A[max(r - h, 0):r + h + 1, max(c - h, 0):c + h + 1].ravel(), B[max(r - h, 0):r + h + 1,
                                                                                  max(c - h, 0):c + h + 1].ravel()
            k = np.isfinite(x) & np.isfinite(y)
            if k.sum() >= 20 and x[k].std() > 0 and y[k].std() > 0:
                out[i] = np.corrcoef(x[k], y[k])[0, 1]
    return out


def reg(y, p):
    return dict(n=len(y), r2=r2_score(y, p), rmse=float(np.sqrt(np.mean((p - y) ** 2))), bias=float(np.mean(p - y)),
                pearson=float(np.corrcoef(y, p)[0, 1]))


def table(d, pairs):
    rec = []
    for name, (yc, pc) in pairs.items():
        for g in GROUPS:
            x = d if g == "pooled" else d[d.group == g]
            for k, xf in x.groupby("fold"):
                rec.append(dict(what=name, group=g, fold=k, **reg(xf[yc].to_numpy(), xf[pc].to_numpy())))
    f = pd.DataFrame(rec)
    s = f.groupby(["what", "group"], sort=False)[["r2", "rmse", "bias", "pearson"]].agg(["mean", "std"])
    return f, s


def fmt(s, order):
    out = pd.DataFrame(index=s.index)
    for m in ["r2", "rmse", "bias", "pearson"]:
        out[m] = [f"{a:.3f} ({b:.3f})" for a, b in zip(s[(m, "mean")], s[(m, "std")])]
    return out.reindex(pd.MultiIndex.from_product([order, GROUPS])).to_string()


def kfold(n_jobs):
    t0 = time.time()
    d, base = load_all()
    fold = d.fold.to_numpy()
    for c in ["p98_direct", "p98_transfer", "p98_transfer_mean", "p98_oracle", "g_true", "g_hat"]:
        d[c] = np.nan
    imp = []
    for k in range(5):
        tr, te = fold != k, fold == k
        m = RandomForestRegressor(n_estimators=500, max_features=0.33, min_samples_leaf=5, n_jobs=n_jobs,
                                  random_state=0).fit(d.loc[tr, base], d.loc[tr, "p98"])
        d.loc[te, "p98_direct"] = m.predict(d.loc[te, base])
        imp.append(pd.Series(m.feature_importances_, base))
        f2 = hgb(d.loc[tr, ["mean", CV]].to_numpy(), d.loc[tr, "p98"])
        d.loc[te, "p98_transfer"] = f2.predict(d.loc[te, ["mean_hat", "cover_hat"]].to_numpy())
        d.loc[te, "p98_oracle"] = f2.predict(d.loc[te, ["mean", CV]].to_numpy())
        f1 = hgb(d.loc[tr, ["mean"]].to_numpy(), d.loc[tr, "p98"])
        d.loc[te, "p98_transfer_mean"] = f1.predict(d.loc[te, ["mean_hat"]].to_numpy())
        g = hgb(d.loc[tr, [CV]].to_numpy(), d.loc[tr, "p98"])
        d.loc[te, "g_true"] = g.predict(d.loc[te, [CV]].to_numpy())
        d.loc[te, "g_hat"] = g.predict(d.loc[te, ["cover_hat"]].to_numpy())
        print(f"[{time.time() - t0:5.0f} s] fold {k}", flush=True)
    d["res_true"] = d.p98 - d.g_true
    for a in ["direct", "transfer"]:
        d[f"res_{a}"] = d[f"p98_{a}"] - d.g_hat
    d["res_mean_true"] = d["mean"] - hgb(d[[CV]].to_numpy(), d["mean"]).predict(d[[CV]].to_numpy())  # in-sample
    print("window correlations ...", flush=True)
    d["wr_true"] = win_corr(d, "p98", CV)
    d["wr_direct"] = win_corr(d, "p98_direct", "cover_hat")
    d["wr_transfer"] = win_corr(d, "p98_transfer", "cover_hat")
    d["wr_mean_true"] = win_corr(d, "mean", CV)
    d.to_parquet(OUTP / "kfold_preds.parquet")

    fA, sA = table(d, {"direct": ("p98", "p98_direct"), "transfer": ("p98", "p98_transfer"),
                       "transfer_mean": ("p98", "p98_transfer_mean"), "oracle": ("p98", "p98_oracle"),
                       "mean_hat (ref)": ("mean", "mean_hat")})
    w = d.dropna(subset=["wr_true", "wr_direct", "wr_transfer"])
    fB, sB = table(d, {"res direct": ("res_true", "res_direct"), "res transfer": ("res_true", "res_transfer")})
    fW, sW = table(w, {"win r direct": ("wr_true", "wr_direct"), "win r transfer": ("wr_true", "wr_transfer")})
    pd.concat([fA, fB, fW]).to_csv(OUTP / "kfold_fold_metrics.csv", index=False)
    fi = pd.concat(imp, axis=1).mean(1).nlargest(8)

    q = [1, 5, 25, 50, 75, 95, 99]
    lab = d.groupby("group")[["mean", "p90", "p98", CV]].mean()
    L = ["REPORT — p98 from mean + cover, and height-cover decorrelation (p98_cover.py)",
         "Labels: 39,023 px (Dampier 2026 + Yawuru kept). Folds: folds_cover_height.json. Seed 0 only "
         "(seed SD 0.000 in REPORT_regression).",
         "p98 = 98th percentile of the 50 cm CHM cells in the 20 m px. Values: fold mean (fold SD).",
         "Optimistic: random K-fold under spatial autocorrelation.", "",
         "0. LABELS (mean per group)", lab.round(3).to_string(),
         f"r(p98, mean) {np.corrcoef(d.p98, d['mean'])[0, 1]:.3f}; r(p98, cover) {np.corrcoef(d.p98, d[CV])[0, 1]:.3f};"
         f" r(mean, cover) {np.corrcoef(d['mean'], d[CV])[0, 1]:.3f}", "",
         "1. p98 K-FOLD (direct RF vs transfer f(mean_hat, cover_hat))", fmt(sA, list(dict.fromkeys(sA.index.get_level_values(0)))),
         "", "direct RF top 8 impurity (fold mean): " + ", ".join(f"{n} {v:.3f}" for n, v in fi.items()), "",
         "2. DECORRELATION — residual r = p98 - E[p98 | cover]; skill of r_hat against the LiDAR r",
         fmt(sB, ["res direct", "res transfer"]),
         f"residual SD (LiDAR) {d.res_true.std():.3f} m; p98 SD {d.p98.std():.3f} m", "",
         f"3. DECORRELATION — {WIN} x {WIN} px window r(p98, cover); skill against the LiDAR window r "
         f"({len(w):,} px with >= 20 label px in the window)", fmt(sW, ["win r direct", "win r transfer"]),
         "LiDAR window r percentiles " + str(dict(zip(q, np.percentile(w.wr_true, q).round(3)))),
         "map window r percentiles   " + str(dict(zip(q, np.percentile(w.wr_transfer, q).round(3)))),
         "mean vs cover window r (LiDAR) " + str(dict(zip(q, np.nanpercentile(d.wr_mean_true, q).round(3))))]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {REPORT} ({(time.time() - t0) / 60:.1f} min)")


def write(arr, out, prof):
    import rasterio
    tmp = out.with_name(out.stem + ".partial.tif")
    with rasterio.open(tmp, "w", **prof) as s:
        s.write(np.where(np.isfinite(arr), arr, NODATA).astype("float32"), 1)
    tmp.replace(out)


def maps(n_jobs):
    """Direct RF p98 map (ALL, all 39,023 px, seed 0); residual = p98 - g(cover map), g = E[p98 map | cover map] on the
    map, every 7th px (HGB, applied via a 1001-step lookup); residual window mean WIN x WIN (>= 80 % valid px)."""
    import pickle
    import warnings

    import rasterio
    from scipy.ndimage import uniform_filter
    import regression_2026 as RG
    warnings.filterwarnings("ignore", "X does not have valid feature names")
    t0 = time.time()
    d, base, bands, RR = RG.train_table()
    d = d.merge(pd.read_parquet(OUTP / "rows_p98.parquet")[KEY + ["p98"]], on=KEY, how="left", validate="1:1")
    assert d.p98.notna().all()
    out, pkl = RG.MAPD / "height_p98_ALL.tif", RG.MAPD / "rf_ALL_p98.pkl"
    if out.exists() and pkl.exists():
        m = pickle.load(open(pkl, "rb"))
        print(f"skip {out} (exists)", flush=True)
    else:
        m = RG.fit(d, base, "p98", n_jobs)
        pickle.dump(m, open(pkl, "wb"))
        print(f"[{(time.time() - t0) / 60:5.1f} min] fit p98; predicting {out}", flush=True)
        RR.FS.predict_raster([m], ["p98_ALL"], out, path=RR.STACK_TIF, bands=bands)
    back = RR.FS.read_at(out, d)[:, 0]
    assert np.allclose(back, m.predict(d[base]).astype("float32"), atol=1e-4), "p98 map != model at label px"
    print(f"[{(time.time() - t0) / 60:5.1f} min] p98 map at label px = model (asserted)", flush=True)

    with rasterio.open(out) as s:
        p98, prof = s.read(1), s.profile
    with rasterio.open(RG.MAPD / "cover137_ALL.tif") as s:
        cov = s.read(1)
        assert s.transform == prof["transform"] and s.shape == p98.shape
    valid = (p98 != NODATA) & (cov != NODATA)
    # g fitted on the MAP (every 7th px): the label relation sits 1-2 m above the scene at cover 0.2-0.5 and left
    # the scene median at -1.3 m; K-fold skill of the two choices is equal (R2 0.461 vs 0.468, p98_cover.py kfold)
    sk = valid[::7, ::7]
    g = hgb(cov[::7, ::7][sk][:, None], p98[::7, ::7][sk])
    gx = np.linspace(0, 1, 1001)
    gy = g.predict(gx[:, None])
    assert valid.sum() == ((p98 != NODATA).sum()), "p98 and cover masks differ"
    res = np.full(p98.shape, np.nan, "float32")
    res[valid] = p98[valid] - np.interp(np.clip(cov[valid], 0, 1), gx, gy)
    v = valid.astype("float32")
    num = uniform_filter(np.where(valid, res, 0).astype("float64"), WIN, mode="constant")
    den = uniform_filter(v.astype("float64"), WIN, mode="constant")
    resw = np.where(valid & (den >= 0.8), num / np.maximum(den, 1e-9), np.nan).astype("float32")
    prof.update(dtype="float32", nodata=NODATA, count=1)
    write(res, RG.MAPD / "p98_resid_ALL.tif", prof)
    write(resw, RG.MAPD / f"p98_resid_w{WIN}_ALL.tif", prof)
    q = [1, 5, 25, 50, 75, 95, 99]
    L = [f"p98 map {out}", f"valid {valid.sum():,} px", "p98 map percentiles " +
         str(dict(zip(q, np.percentile(p98[valid], q).round(2)))),
         "p98 labels percentiles  " + str(dict(zip(q, np.percentile(d.p98, q).round(2)))),
         f"g(cover) = E[p98 map | cover map]: g(0) {gy[0]:.2f}, g(0.25) {gy[250]:.2f}, g(0.5) {gy[500]:.2f}, "
         f"g(0.75) {gy[750]:.2f}, g(1) {gy[-1]:.2f} m",
         "residual percentiles " + str(dict(zip(q, np.percentile(res[valid], q).round(2)))),
         f"residual w{WIN} percentiles " + str(dict(zip(q, np.nanpercentile(resw, q).round(2)))),
         f"residual w{WIN} valid {np.isfinite(resw).sum():,} px"]
    (OUTP / "map_checks.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    print(f"done {(time.time() - t0) / 60:.1f} min", flush=True)


NODATA = -9999.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["rows", "kfold", "maps"])
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()
    {"rows": rows, "kfold": lambda: kfold(a.n_jobs), "maps": lambda: maps(a.n_jobs)}[a.step]()
    sys.exit(0)
