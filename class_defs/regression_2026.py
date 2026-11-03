"""Regression track (PLAN_regression.md): 2026 height (px mean) and cover (1.37 m) maps.

  score    R1: score the Part B K-fold preds (no new fits). Stacks ALL / SAR / MS, targets mean / cover.
           Pooled first, then Dampier, then Yawuru kept. Fold mean (fold SD, seed SD), 3 seeds.
           Paired gain ALL - SAR, ALL - MS; real if > 2 x fold SD and > seed SD.
           Class check: px-mean preds -> classes with the cut set in outputs/cut_sets.json (read at run time).
           Importances: imp_<stack>_<target>.csv (Part B, seed 0).
  maps     R2: fit RF (ALL stack only, user 2026-11-02) on all 39,023 px, seed 0, for px mean and cover 1.37;
           predict every px of the stack with all 49 features finite (the stack is NaN outside the mainland).
           Asserts: stack bands at the label px = training features (atol 1e-4); map at the label px = model
           predict. Writes E:/nisar_2026/regression/{height_mean,cover137}_ALL.tif (float32, nodata -9999,
           .partial.tif then rename; existing maps skipped) + rf_ALL_<t>.pkl; checks -> map_checks_*.csv.
           --quick: fit px mean only, time a 1 M px tile, write nothing.
  --test   synthetic checks of the block shuffle and map metrics (R3; dropped 2026-11-02, kept for reuse).

Outputs: outputs/regression/score_*.csv, map_checks_*.csv, class_defs/REPORT_regression.txt.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

import common as C
from cover_height_report import paired, reg, summarise
from cover_height_rf import OUTD, SEEDS

STACKS = ["ALL", "SAR", "MS"]
TGT = {"mean": "px mean (m)", "cover": "cover 1.37"}
GROUPS = ["pooled", "Dampier", "Yawuru kept"]
CUTSET = "gmm50_k4_loc_y"
OUTR = C.OUT / "regression"
REPORT = C.HERE / "REPORT_regression.txt"
BLOCK = 8


# ------------------------------------------------------------------------------------------- R1

def load_preds():
    out = {}
    for s in STACKS:
        for t in TGT:
            for seed in SEEDS:
                p = pd.read_parquet(OUTD / f"pred_{s}_{t}_s{seed}.parquet")
                assert p.use.all() and p.pred.notna().all() and len(p) == 39023, (s, t, seed)
                out[(s, t, seed)] = p
    k0 = out[(STACKS[0], "mean", 0)][["site", "row", "col", "fold"]]
    for p in out.values():
        assert (p[["site", "row", "col", "fold"]].to_numpy() == k0.to_numpy()).all(), "row order / folds differ"
    return out


def with_pooled(p):
    return pd.concat([p.assign(g="pooled"), p.assign(g=p.group)])


def fold_metrics(preds):
    rec = []
    for (s, t, seed), p in preds.items():
        for (g, k), x in with_pooled(p).groupby(["g", "fold"]):
            rec.append(dict(stack=s, target=t, seed=seed, group=g, fold=k, **reg(x.y, x.pred)))
    return pd.DataFrame(rec)


def class_check(preds, cuts):
    rec = []
    for (s, t, seed), p in preds.items():
        if t != "mean":
            continue
        for g, x in with_pooled(p).groupby("g"):
            a, b = np.digitize(x.y, cuts), np.digitize(x.pred, cuts)
            P, R, F, N = precision_recall_fscore_support(a, b, labels=range(4), zero_division=0)
            rec.append(dict(stack=s, seed=seed, group=g, n=len(a), macro_f1=F.mean(), worst_f1=F.min(),
                            f1w=f1_score(a, b, average="weighted", zero_division=0),
                            **{f"{m}{i}": v[i] for m, v in zip("PRF", (P, R, F)) for i in range(4)},
                            **{f"n{i}": N[i] for i in range(4)}))
    return pd.DataFrame(rec)


def importances():
    perm, top = [], []
    for s in STACKS:
        for t in TGT:
            ii = pd.read_csv(OUTD / f"imp_{s}_{t}.csv")
            pm = ii[ii.kind == "perm_dR2"].groupby("name").value.agg(["mean", "std"])
            for n, r in pm.iterrows():
                perm.append(dict(stack=s, target=t, sensor=n, dR2=r["mean"], fold_sd=r["std"]))
            fi = ii[ii.kind == "impurity"].groupby("name").value.mean().nlargest(5)
            top.append(f"{s:3s} {t:5s} " + ", ".join(f"{n} {v:.3f}" for n, v in fi.items()))
    return pd.DataFrame(perm), top


def fmt(df, nd=3):
    return df.round(nd).to_string()


def section(L, title, body):
    L += ["", title, "-" * len(title), body]


def score():
    OUTR.mkdir(exist_ok=True)
    preds = load_preds()
    fm = fold_metrics(preds)
    fm.to_csv(OUTR / "score_fold_metrics.csv", index=False)

    # Part B per-source reference (REPORT_cover_height.txt section 1)
    ref = {("mean", "Dampier"): 0.773, ("mean", "Yawuru kept"): 0.879,
           ("cover", "Dampier"): 0.775, ("cover", "Yawuru kept"): 0.903}
    s = summarise(fm[fm["stack"] == "ALL"], ["target", "group"], "r2")
    for k, v in ref.items():
        assert abs(s.loc[k].m - v) < 0.0006, f"R1 does not reproduce Part B {k}: {s.loc[k].m:.4f} vs {v}"
    print("Part B reference R2 reproduced (ALL, per source)", flush=True)

    L = ["REPORT — 2026 height and cover regression (PLAN_regression.md)",
         "Labels: Dampier 2026 (5 sites, 34,715 px) + Yawuru 2025 kept (7 areas, 4,308 px) = 39,023 px.",
         "RF 500, max_features 0.33, leaf 5; stratified 5-fold (outputs/folds_cover_height.json); seeds 0-2.",
         "Values: fold mean (fold SD, seed SD). Pooled = all 39,023 px; then per source.",
         "Optimistic: random K-fold under spatial autocorrelation."]

    tabs = []
    for val in ["r2", "rmse", "bias"]:
        x = summarise(fm, ["group", "target", "stack"], val)
        x["val"] = val
        tabs.append(x.reset_index())
        cell = x.apply(lambda r: f"{r.m:.3f} ({r.fold_sd:.3f}, {r.seed_sd:.3f})", axis=1)
        tab = cell.unstack("stack")[STACKS].reindex(pd.MultiIndex.from_product([GROUPS, list(TGT)]))
        section(L, f"1. K-FOLD {val.upper()}" + (" (m for px mean)" if val != "r2" else ""), tab.to_string())
    pd.concat(tabs).to_csv(OUTR / "score_summary.csv", index=False)

    gains = []
    for b in ["SAR", "MS"]:
        for val in ["r2", "rmse"]:
            p = paired(fm, b, "ALL", "stack", ["group", "target"], val=val)
            gains.append(p.assign(gain=f"ALL - {b}", val=val).reset_index())
    g = pd.concat(gains)
    g.to_csv(OUTR / "score_gains.csv", index=False)
    gt = g.set_index(["val", "gain", "group", "target"])[["m", "fold_sd", "seed_sd", "real"]]
    gt = gt.reindex(pd.MultiIndex.from_product([["r2", "rmse"], ["ALL - SAR", "ALL - MS"], GROUPS, list(TGT)]))
    section(L, "1b. PAIRED GAIN (per fold, seed); real if |gain| > 2 x fold SD and > seed SD",
            fmt(gt.rename(columns={"m": "gain"})))

    cs = json.load(open(C.OUT / "cut_sets.json"))[CUTSET]
    cuts = cs["cuts"]
    cm = class_check(preds, cuts)
    cm.to_csv(OUTR / "score_class.csv", index=False)
    a = cm.groupby(["group", "stack"])[["macro_f1", "f1w", "worst_f1"]].agg(["mean", "std"])
    head = pd.DataFrame({m: a[(m, "mean")] for m in ["macro_f1", "f1w", "worst_f1"]})
    head["macro_seed_sd"] = a[("macro_f1", "std")]
    head = head.reindex(pd.MultiIndex.from_product([GROUPS, STACKS]))
    pc = cm.groupby(["group", "stack"])[[f"{m}{i}" for i in range(4) for m in "PRF"]].mean()
    pc = pc.reindex(head.index)
    nn = cm[cm.seed == 0].set_index(["group", "stack"])[[f"n{i}" for i in range(4)]].reindex(head.index)
    section(L, f"2. CLASS CHECK — px-mean preds -> 4 classes, cut set {CUTSET} {cuts} "
            f"(cuts_50cm {cs.get('cuts_50cm')}); pooled folds, mean of 3 seeds",
            fmt(head) + "\n\nper class P / R / F1 (c0 Bare -> c3 Forest)\n" + fmt(pc, 2)
            + "\n\ntrue px per class\n" + nn.loc[(slice(None), "ALL"), :].to_string())

    perm, top = importances()
    perm.to_csv(OUTR / "score_importance.csv", index=False)
    pt = perm.set_index(["target", "stack", "sensor"]).dR2.unstack("sensor")
    section(L, "3. IMPORTANCE — permutation R2 drop by sensor (seed 0, fold mean, pooled test fold)", fmt(pt)
            + "\n\ntop 5 impurity (seed 0, fold mean)\n" + "\n".join(top))
    section(L, "4. MAP CHECKS", map_section())
    section(L, "5. MAP COMPARISON", "dropped (user 2026-11-02): ALL maps only; the K-fold tables above compare stacks")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {REPORT}")


# ------------------------------------------------------------------------------------------- R2

MAPD = Path(r"E:\nisar_2026\regression")
MAPS = {"mean": ("height_mean_ALL.tif", "mean"), "cover": ("cover137_ALL.tif", C.col(1.37))}
NODATA = -9999.0


def map_section():
    L = []
    for t in TGT:
        fn = OUTR / f"map_checks_{t}.txt"
        L.append(fn.read_text(encoding="utf-8") if fn.exists() else f"{t}: R2 not run yet")
    return "\n\n".join(L)


def train_table():
    import cover_height_rf as R
    d, base, _ = R.load(with_glcm=False)
    d = d[d.group != "Yawuru burnt"].reset_index(drop=True)
    assert len(d) == 39023 and len(base) == 49 and np.isfinite(d[base].to_numpy(float)).all()
    feats = R.FS.feature_names(R.STACK_TIF)
    Xs = R.FS.sample(d[["site", "row", "col"]], R.STACK_TIF)[base].to_numpy(float)
    err = float(np.abs(Xs - d[base].to_numpy(float)).max())
    assert err < 1e-4, f"stack at label px != training features (max |d| {err})"
    print(f"label-px assert: stack bands = training features, max |d| {err:.1e} (39,023 px x 49)", flush=True)
    return d, base, [feats.index(c) + 1 for c in base], R


def fit(d, base, ycol, n_jobs, n_trees=500):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n_trees, max_features=0.33, min_samples_leaf=5, n_jobs=n_jobs,
                                 random_state=0).fit(d[base], d[ycol])


def time_tile(m, bands, R):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(R.STACK_TIF) as s:
        tf, h, w = s.transform, s.height, s.width
        a = s.read(bands, window=Window(0, h // 2 - 512, w, 1024)).reshape(len(bands), -1).T
    X = a[np.isfinite(a).all(1)][:1_000_000]
    t1 = time.time()
    m.predict(X)
    dt = time.time() - t1
    n_main = int(R.FS.mainland_mask(tf, h, w).sum())
    print(f"tile: {len(X):,} finite px in {dt:.0f} s; mainland {n_main:,} px -> about "
          f"{dt * n_main / len(X) / 60:.0f} min per map (upper bound)", flush=True)


def map_checks(t, out, m, d, base, ycol, R):
    import rasterio
    back = R.FS.read_at(out, d)[:, 0]
    assert np.allclose(back, m.predict(d[base]).astype("float32"), atol=1e-5), f"{t}: map != model at label px"
    with rasterio.open(out) as s:
        arr, tf = s.read(1), s.transform
        assert s.nodata == NODATA
    inside = R.FS.mainland_mask(tf, *arr.shape)
    valid = arr != NODATA
    assert not np.isnan(arr).any(), f"{t}: NaN in map"
    assert not (valid & ~inside).any(), f"{t}: valid px outside the mainland"
    v = arr[valid]
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    hist = pd.DataFrame({"map (all valid px)": np.percentile(v, q), "labels (39,023 px)": np.percentile(d[ycol], q)},
                        index=[f"p{x}" for x in q]).T
    hist["mean"] = [v.mean(), d[ycol].mean()]
    site = d.assign(map=back).groupby(["group", "site"]).agg(n=(ycol, "size"), label=(ycol, "mean"),
                                                             map=("map", "mean"))
    site["map - label"] = site["map"] - site["label"]
    L = [f"{t} map: {out}",
         f"valid {valid.sum():,} of {inside.sum():,} mainland px ({valid.sum() / inside.sum():.1%}); no NaN; "
         f"none outside the mainland",
         "map at the label px = model predict (asserted); in-sample, so the site check is not an accuracy",
         "", "distribution", fmt(hist), "", "per site, mean at the label px", fmt(site)]
    txt = "\n".join(L)
    (OUTR / f"map_checks_{t}.txt").write_text(txt, encoding="utf-8")
    pd.concat([hist.assign(kind="hist"), site.reset_index().assign(kind="site")]).to_csv(
        OUTR / f"map_checks_{t}.csv", index=False)
    print(txt, flush=True)


def maps(a):
    import warnings
    warnings.filterwarnings("ignore", "X does not have valid feature names")  # stack blocks are numpy, same order
    t0 = time.time()
    OUTR.mkdir(exist_ok=True)
    d, base, bands, R = train_table()
    if a.quick:
        m = fit(d, base, "mean", a.n_jobs)
        print(f"fit px mean, 500 trees, n_jobs {a.n_jobs}: {time.time() - t0:.0f} s", flush=True)
        return time_tile(m, bands, R)
    MAPD.mkdir(exist_ok=True)
    for t, (name, ycol) in MAPS.items():
        out, pkl = MAPD / name, MAPD / f"rf_ALL_{t}.pkl"
        if out.exists() and pkl.exists():
            print(f"skip {out} (exists)", flush=True)
            m = pickle.load(open(pkl, "rb"))
        else:
            m = fit(d, base, ycol, a.n_jobs)
            pickle.dump(m, open(pkl, "wb"))
            print(f"[{(time.time() - t0) / 60:5.1f} min] fit {t}; predicting {out}", flush=True)
            R.FS.predict_raster([m], [f"{t}_ALL"], out, path=R.STACK_TIF, bands=bands)
        print(f"[{(time.time() - t0) / 60:5.1f} min] {t} map done; checks", flush=True)
        map_checks(t, out, m, d, base, ycol, R)
    print(f"done {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------------------------------- R3 helpers

def block_shuffle(a, rng, block=BLOCK):
    """Permute block x block tiles of a 2-D float map; the remainder strip becomes NaN."""
    h, w = a.shape
    H, W = (h // block) * block, (w // block) * block
    t = a[:H, :W].reshape(H // block, block, W // block, block).transpose(0, 2, 1, 3).reshape(-1, block, block)
    t = t[rng.permutation(len(t))]
    out = np.full(a.shape, np.nan, a.dtype)
    out[:H, :W] = t.reshape(H // block, W // block, block, block).transpose(0, 2, 1, 3).reshape(H, W)
    return out


def map_metrics(a, b, m):
    k = m & np.isfinite(a) & np.isfinite(b)
    x, y = a[k].astype(float), b[k].astype(float)
    return dict(n=int(k.sum()), r=float(np.corrcoef(x, y)[0, 1]), rmsd=float(np.sqrt(np.mean((x - y) ** 2))),
                mean_diff=float(np.mean(x - y)))


def test():
    rng = np.random.default_rng(0)
    # smooth field: structure at ~5-block scale, so the null destroys correspondence
    z = rng.normal(size=(40, 40))
    a = np.kron(z, np.ones((BLOCK // 2, BLOCK // 2)))[:160, :160].astype(np.float32)
    b = a + rng.normal(0, 0.3, a.shape).astype(np.float32)
    m = np.ones(a.shape, bool)
    s = block_shuffle(a, rng)
    assert np.array_equal(np.sort(s.ravel()), np.sort(a.ravel())), "multiset changed"
    i = map_metrics(a, a, m)
    assert abs(i["r"] - 1) < 1e-9 and i["rmsd"] == 0 and i["mean_diff"] == 0, i
    o = map_metrics(a, b, m)
    nr = [map_metrics(block_shuffle(a, rng), b, m)["r"] for _ in range(20)]
    print(f"identical r {i['r']:.3f} rmsd {i['rmsd']}; observed r {o['r']:.3f}; "
          f"null r {np.mean(nr):.3f} (SD {np.std(nr):.3f})")
    assert o["r"] > 0.9 and abs(np.mean(nr)) < 0.05, "null r not about 0"
    # remainder strip -> NaN, ignored by map_metrics
    s2 = block_shuffle(a[:157, :155], rng)
    assert np.isnan(s2[156:, :]).all() and np.isfinite(s2[:152, :152]).all()
    print("test passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", choices=["score", "maps"])
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()
    if a.test:
        return test()
    score() if a.step == "score" else maps(a)


if __name__ == "__main__":
    main()
