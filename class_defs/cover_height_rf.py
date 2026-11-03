"""Part B step B3 / B3b (PLAN_cover_height.md): RF K-fold regressions of cover, mhi, mlo and mean at h = 1.37 m.

Stacks: MS (S2 13), MS+G (+ S2 GLCM 24), SAR (NISAR 26 + S1 10), SAR+G (+ NISAR GLCM 32 + S1 GLCM 16),
ALL (49), ALL+G (121). Targets: cover (all rows), mhi (c >= 0.05), mlo (c <= 0.95), mean (baseline).
RF 500, max_features 0.33, min_samples_leaf 5, n_jobs 10, random_state = seed; fixed, not tuned.
Folds: outputs/folds_cover_height.json (built once over all 43,134 rows; random stratified K-fold).
Rows: main run = Dampier + Yawuru kept; --with-burnt = all rows (B3b, default stack ALL), same folds.
Same rows for every stack: rows with any non-finite feature of the run's stacks are dropped (printed per site).

Outputs outputs/cover_height/ (restart-safe: .partial then rename, existing files skipped):
  pred_<stack>_<target>_s<seed>[_burnt].parquet  site, row, col, group, fold, y, pred, use: test-fold pred at
                                                 EVERY row (model trained on its target rows only); metrics on use
  imp_<stack>_<target>[_burnt].csv               seed 0: impurity per feature + permutation R2 drop by group,
                                                 per fold; impurity also to nisar_2026 importance_record.csv
  --quick: (1) reference check = cover_split_features --quick (Dampier, folds_old fold 0, 50 trees, ALL, mean:
           R2 0.772); (2) 1 fold, 50 trees, every stack x target, seed 0 -> quick_<...>.csv only.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import common as C
import feature_stack_2026 as FS

H = 1.37
OUTD = C.OUT / "cover_height"
KEY = ["site", "row", "col"]
N_JOBS = 10
SEEDS = [0, 1, 2]
STACK_TIF = r"E:\nisar_2026\feature_stack_2026_gslcg0_rflee9_ws5_s1.tif"
TARGETS = {"cover": (C.col(H), lambda c: np.ones(len(c), bool)), "mhi": (C.mhi(H), lambda c: c >= 0.05),
           "mlo": (C.mlo(H), lambda c: c <= 0.95), "mean": ("mean", lambda c: np.ones(len(c), bool))}


def groups(cols):
    """Feature groups: S2, S2 GLCM, NISAR, NISAR GLCM, S1, S1 GLCM."""
    def base(c):
        return c[5:] if c.startswith("glcm_") else c
    def src(c):
        b = base(c)
        if b[:5] in ("147_D", "168_A"):
            return "NISAR"
        if b.startswith("S1_"):
            return "S1"
        if b.startswith("nbart_") or b.split("_")[0] in ("NDVI", "EVI", "BSI"):
            return "S2"
        raise ValueError(c)
    out = {}
    for c in cols:
        out.setdefault(src(c) + (" GLCM" if c.startswith("glcm_") else ""), []).append(c)
    return out


def stacks(base, glcm):
    g, gg = groups(base), groups(glcm)
    return {"MS": g["S2"], "MS+G": g["S2"] + gg["S2 GLCM"],
            "SAR": g["NISAR"] + g["S1"], "SAR+G": g["NISAR"] + g["S1"] + gg["NISAR GLCM"] + gg["S1 GLCM"],
            "ALL": base, "ALL+G": base + glcm}


def load(with_glcm=True):
    rows = pd.read_parquet(OUTD / "rows.parquet")
    fd = pd.read_parquet(C.OUT / "features_2026_s1.parquet").drop(columns="mean")
    fy = pd.read_parquet(C.OUT / "features_yawuru_s1.parquet")
    assert list(fd.columns) == list(fy.columns)
    base = [c for c in fd.columns if c not in KEY]
    d = rows.merge(pd.concat([fd, fy]), on=KEY, how="left", validate="1:1")
    glcm = []
    if with_glcm:
        g = pd.read_parquet(C.OUT / "glcm_rows.parquet")
        glcm = [c for c in g.columns if c.startswith("glcm_")]
        d = d.merge(g, on=KEY, how="left", validate="1:1")
    assert len(d) == len(rows) and (d[KEY] == rows[KEY]).all().all()
    fo = json.load(open(C.OUT / "folds_cover_height.json"))
    assert fo["n"] == len(d)
    d["fold"] = -1
    for k, te in enumerate(fo["test_folds"]):
        d.loc[te, "fold"] = k
    assert (d.fold >= 0).all()
    return d, base, glcm


def perm_drop(m, X, y, grp, rng, n_rep=3):
    base = r2_score(y, m.predict(X))
    out = {}
    for g, cols in grp.items():
        dd = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            dd.append(base - r2_score(y, m.predict(Xp)))
        out[g] = float(np.mean(dd))
    return out


def fit_one(d, cols, tname, seed, folds, n_trees, imp):
    ycol, sel = TARGETS[tname]
    use = sel(d[C.col(H)].to_numpy())
    y = d[ycol].to_numpy()
    assert np.isfinite(y[use]).all(), tname
    pred = np.full(len(d), np.nan)
    rng = np.random.default_rng(seed)
    grp = groups(cols)
    fold = d.fold.to_numpy()
    for k in folds:
        tr, te = use & (fold != k), use & (fold == k)
        m = RandomForestRegressor(n_estimators=n_trees, max_features=0.33, min_samples_leaf=5, n_jobs=N_JOBS,
                                  random_state=seed).fit(d.loc[tr, cols], y[tr])
        pred[fold == k] = m.predict(d.loc[fold == k, cols])  # every test row: combine needs mhi/mlo everywhere
        if imp is not None:
            imp += [dict(kind="impurity", name=c, fold=k, value=v) for c, v in zip(cols, m.feature_importances_)]
            imp += [dict(kind="perm_dR2", name=g, fold=k, value=v)
                    for g, v in perm_drop(m, d.loc[te, cols], y[te], grp, rng).items()]
    ok = np.isin(fold, list(folds))
    return d.loc[ok, KEY + ["group", "fold"]].assign(y=y[ok], pred=pred[ok], use=use[ok])


def r2_by(p, by="group"):
    """R2 on the target's own rows (use)."""
    return {g: r2_score(x.y, x.pred) for g, x in p[p.use].groupby(by)}


def quick(a):
    t0 = time.time()
    # (1) reference: same as cover_split_features --quick (mean, set all, R2 0.7718)
    f = pd.read_parquet(C.OUT / "features_2026_s1.parquet")
    feats = [c for c in f.columns if c not in KEY + ["mean"]]
    te = np.zeros(len(f), bool)
    te[json.load(open(C.OUT / "folds_old.json"))["test_folds"][0]] = True
    m = RandomForestRegressor(n_estimators=50, max_features=0.33, min_samples_leaf=5, n_jobs=N_JOBS,
                              random_state=0).fit(f.loc[~te, feats], f.loc[~te, "mean"])
    r = r2_score(f.loc[te, "mean"], m.predict(f.loc[te, feats]))
    print(f"reference (folds_old fold 0, 50 trees, ALL, mean): R2 {r:.4f} (cover_split_features_quick 0.7718)",
          flush=True)
    assert abs(r - 0.7718) < 0.03
    # (2) every stack x target, 1 fold, 50 trees
    d, base, glcm = prep(a)
    res = []
    for sname, cols in run_stacks(a, base, glcm).items():
        for tname in TARGETS:
            p = fit_one(d, cols, tname, 0, [0], 50, None)
            assert p.pred.notna().all()
            res.append(dict(stack=sname, target=tname, n=int(p.use.sum()), **r2_by(p)))
            print(f"[{time.time() - t0:5.0f} s] {sname:6s} {tname:5s} n {p.use.sum():6,} "
                  + " ".join(f"{g} {v:.3f}" for g, v in r2_by(p).items()), flush=True)
    sfx = "_burnt" if a.with_burnt else ""
    pd.DataFrame(res).to_csv(OUTD / f"quick{sfx}.csv", index=False)
    print(pd.DataFrame(res).round(3).to_string(index=False))


def run_stacks(a, base, glcm):
    st = stacks(base, glcm) if glcm else {"MS": groups(base)["S2"], "SAR": groups(base)["NISAR"] + groups(base)["S1"],
                                          "ALL": base}
    names = a.stacks or (["ALL"] if a.with_burnt else list(st))
    return {n: st[n] for n in names}


def prep(a):
    d, base, glcm = load(with_glcm=(C.OUT / "glcm_rows.parquet").exists())
    if not a.with_burnt:
        d = d[d.group != "Yawuru burnt"].reset_index(drop=True)
    cols = sorted(set(sum(run_stacks(a, base, glcm).values(), [])), key=(base + glcm).index)
    bad = ~np.isfinite(d[cols].to_numpy(float)).all(1)
    print(f"rows {len(d):,}; features {len(cols)}; dropped (non-finite) {bad.sum():,}"
          + (f": {d.loc[bad].site.value_counts().to_dict()}" if bad.any() else ""), flush=True)
    print(d.loc[~bad].group.value_counts().to_dict(), flush=True)
    return d.loc[~bad].reset_index(drop=True), base, glcm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--with-burnt", action="store_true", help="B3b: all rows incl. Yawuru burnt")
    ap.add_argument("--stacks", nargs="+", help="default: all 6 (main), ALL (--with-burnt)")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    a = ap.parse_args()
    OUTD.mkdir(exist_ok=True)
    if a.quick:
        return quick(a)
    d, base, glcm = prep(a)
    assert glcm, "glcm_rows.parquet missing (run glcm_2026.py)"
    sfx = "_burnt" if a.with_burnt else ""
    t0 = time.time()
    for seed in a.seeds:
        for sname, cols in run_stacks(a, base, glcm).items():
            for tname in TARGETS:
                fn = OUTD / f"pred_{sname}_{tname}_s{seed}{sfx}.parquet"
                if fn.exists():
                    print(f"skip {fn.name}", flush=True)
                    continue
                t1 = time.time()
                imp = [] if seed == 0 else None
                p = fit_one(d, cols, tname, seed, range(5), 500, imp)
                if imp is not None:
                    ii = pd.DataFrame(imp)
                    ii.to_csv(OUTD / f"imp_{sname}_{tname}{sfx}.csv", index=False)
                    fi = ii[ii.kind == "impurity"].groupby("name", sort=False).value.mean().reindex(cols)
                    FS.record_importance(f"cover_height_{sname}{sfx}", "RF", tname, fi.values, cols, "impurity",
                                         500, path=STACK_TIF)
                tmp = fn.with_name(fn.stem + ".partial.parquet")
                p.to_parquet(tmp)
                tmp.replace(fn)
                r2f = [r2_score(x.y, x.pred) for _, x in p[p.use & (p.group == "Dampier")].groupby("fold")]
                print(f"[{(time.time() - t0) / 60:6.1f} min] s{seed} {sname:6s} {tname:5s} {time.time() - t1:5.0f} s"
                      f"  Dampier R2 {np.mean(r2f):.3f} (fold SD {np.std(r2f):.3f})  "
                      + " ".join(f"{g} {v:.3f}" for g, v in r2_by(p).items() if g != "Dampier"), flush=True)
    print(f"done {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
