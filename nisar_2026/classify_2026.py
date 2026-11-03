"""Stage 3: 5-class XGBoost classification, 2026 (PLAN.md).

Labels: labels_2026.parquet `cls` (0 Low, 1 Non-forest, 2 Low forest, 3 Forest, 4 Tall forest; cuts
0.2 / 1.22 / 2.35 / 5.381 m on the mean of p98 per 20 m pixel). Features: feature_stack_2026.tif.

Follows classify_year.train() (same model, search space and scoring) with these divergences:
1. No 70/30 split. All labelled pixels train; accuracy is reported two ways:
   - naive: the refitted best model predicts its own training data (resubstitution);
   - K-fold: cross_val_predict with the best parameters on the 5 folds of the search
     (StratifiedKFold, shuffled, random -- no spatial blocking, user decision 2026-10-19).
   Both are optimistic: random folds under spatial autocorrelation, and the folds also chose
   the parameters.
2. objective multi:softprob with no num_class (XGBoost takes it from the labels; a stale
   num_class breaks set_params before predict, see classify_year.py divergence 5).
3. Balanced sample weights are passed to the search, the refit and every K-fold fit.
4. --merge-low (user, 2026-10-20): 4 classes, Low merged into Non-forest (< 1.22 m); outputs in
   outputs/classify_4class/ and outputs/XGB_DP_NISAR_S2_2026_4class.tif.
5. --exclude-site NAME (user, 2026-10-20; Bunda Extra has white sandy soil, unlike the red pindan of
   the other sites): the site is dropped from training and accuracy; outputs get a _no<Site> suffix.
6. --sar-only (user, 2026-10-20): NISAR features only (GCOV + GSLC of both tracks, no S2), to test
   why S2 led the importances; outputs get a _NISARonly suffix and an XGB_DP_NISAR_2026_ map.
7. --tag NAME (user, 2026-10-27): overrides the run tag (output folder, map name, importance record).
   4classTF = Low merged into Non-forest, Tall forest kept (the --merge-low scheme), named so to keep it
   apart from the 2019-2025 4-class (Bare / Low veg / Low forest / Forest). The GSLC-only SAR run:
   --merge-low --stack E:\\nisar_2026\\feature_stack_2026_gslc.tif --tag 4classTF_GSLC.
8. --seed N (accuracy diagnosis Stage 0, 2026-10-28): random_state of the folds, the search and the model
   (default 42 = RANDOM_STATE); without --tag, a seed other than 42 adds _seed<N> to the tag.
9. --s2-only (Stage 0): the 13 S2 bands only (no NISAR); the rows are the same as the full stack (rows with
   a non-finite SAR feature are still dropped), so the folds are the same; _S2only suffix, XGB_DP_S2_2026_ map.
10. --no-map (Stage 0): no full-scene prediction (the importance record is still written).
11. --no-nisar (Stage 1.4): drop the NISAR features (prefix 147_D / 168_A), keep the rest (S1 + S2 of a stack
   made with feature_stack_2026.py --s1); the rows are the same as the full stack; _noNISAR suffix,
   XGB_DP_S1_S2_2026_ map. --sar-only keeps only NISAR (the S1_ bands are dropped).
12. --site NAME (accuracy diagnosis Stage 2): train and K-fold inside one site only; _site<Name> suffix. A class
   with no px at the site is left out of the fit (labels re-coded to 0..k-1, predictions coded back), so
   --no-map is then required.
13. --subsample N (Stage 2): a stratified random subset of N rows (seeded by --seed) before the search; _n<N>
   suffix. Accuracy is on the subset only.
14. --extra-features PATH (Stage 2): a float32 raster on the stack grid (NaN nodata, band descriptions = names),
   added after the stack bands through a VRT (outputs/diagnosis/<stack>+<extra>.vrt with its .json); _cov
   suffix. The rows must stay the same as the stack alone (asserted), so the folds are the same.
15. --keep-nan (per-acquisition test, user 2026-11-01): rows and map pixels need only the S2 features finite; NaN
   SAR features go to XGBoost as missing (its learned default direction). _keepnan suffix.
16. --nan-augment (implies --keep-nan): each label row gets one copy with one acquisition (bands named
   <trk>_<mmdd>_*, per_acq_features_2026.py perdate stack) set to NaN, the acquisition drawn at random (--seed).
   Folds are made on the original rows; each copy is in its original's fold (training side only for the
   search, so the search scores the same rows as other runs). K-fold is reported for the originals (kfold) and,
   in cross_val_predict, for the copies (kfold_masked). This is not leave-site-out. _nanaug suffix.
17. --cuts c1 c2 ... (class_defs, user 2026-11-01): classes = np.digitize(mean, cuts), names from the cuts
   (`<c1`, `c1-c2`, ..., `>=cn`); not with --merge-low. --cuts 1.22 2.35 5.381 is asserted equal to --merge-low.
   Without --tag the tag is <k>class_cuts_<c1>_<c2>... Without --cuts nothing changes.
18. --extra-labels PATH (class_defs, user 2026-11-02; needs --cuts): a parquet of extra label px (site, row, col,
   mean; e.g. class_defs\\outputs\\labels_yawuru_kept.parquet, the 7 unburnt Yawuru 2025 areas) appended to
   labels_2026 before the features are read; same rules (rows with a non-finite feature dropped). Column `source`
   = Dampier / Extra. metrics.csv and the confusion files add per-source rows (site src_Dampier / src_Extra);
   report those -- the `all` row blends the sources. Without it nothing changes.
Every run does its own RandomizedSearchCV (no parameters carried over); random_state 42 gives every
run the same 200 candidate sets, so runs differ only in the data that chooses between them.

Outputs in outputs/classify/ and outputs/XGB_DP_NISAR_S2_2026_5class.tif (uint8, nodata 255).

Run from this directory (detached for the full run):
    conda run -n general --no-capture-output python classify_2026.py --quick      # n_iter 5, subset, no map
    conda run -n general --no-capture-output python -u classify_2026.py --n-iter 200 --n-jobs 10
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.utils.class_weight import compute_sample_weight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feature_stack_2026 as F  # noqa: E402
from labels_2026 import CLASSES  # noqa: E402

RANDOM_STATE = 42
OUT = HERE / "outputs" / "classify"
MAP = HERE / "outputs" / "XGB_DP_NISAR_S2_2026_5class.tif"
PARAM_DIST = {                                    # classify_year.train(), incl. the 0.5-1.0 band fix
    "max_depth": randint(3, 12),
    "learning_rate": uniform(0.01, 0.3),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0, 0.5),
}


def class_metrics(y, p):
    labels = list(range(len(CLASSES)))
    pr, rc, f1, n = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {"oa": accuracy_score(y, p), "balanced_acc": balanced_accuracy_score(y, p),
            "kappa": cohen_kappa_score(y, p), "f1_weighted": float(np.average(f1, weights=n)) if n.sum() else np.nan,
            **{f"{k}_{c}": float(v[i]) for k, v in (("precision", pr), ("recall", rc), ("f1", f1), ("n", n))
               for i, c in enumerate(CLASSES)}}


def report(lab, preds):
    """Overall and per-site metrics for each prediction set -> (DataFrame, {name: confusion matrix})."""
    rows, cms = [], {}
    for name, p in preds.items():
        rows.append({"eval": name, "site": "all", **class_metrics(lab["cls"], p)})
        for site, g in lab.groupby("site"):
            rows.append({"eval": name, "site": site, **class_metrics(g["cls"], p[g.index])})
        cms[name] = pd.DataFrame(confusion_matrix(lab["cls"], p, labels=range(len(CLASSES))),
                                 index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES])
        if "source" in lab:                                   # divergence 18: per source, never blended
            for src, g in lab.groupby("source"):
                rows.append({"eval": name, "site": f"src_{src}", **class_metrics(g["cls"], p[g.index])})
                cms[f"{name}_src_{src}"] = pd.DataFrame(
                    confusion_matrix(g["cls"], p[g.index], labels=range(len(CLASSES))),
                    index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES])
    return pd.DataFrame(rows), cms


def stack_with_extra(stack, extra):
    """VRT of the stack bands, then the extra bands (same grid, asserted), with its .json -> VRT path."""
    from osgeo import gdal
    gdal.UseExceptions()
    with rasterio.open(stack) as s, rasterio.open(extra) as e:
        assert (s.crs, s.transform, s.shape) == (e.crs, e.transform, e.shape), "extra features not on the stack grid"
        assert all(e.descriptions) and np.isnan(e.nodata), "extra features need band names and NaN nodata"
        n, names = s.count + e.count, list(e.descriptions)
    out = HERE / "outputs" / "diagnosis" / f"{Path(stack).stem}+{Path(extra).stem}.vrt"
    out.parent.mkdir(parents=True, exist_ok=True)
    gdal.BuildVRT(str(out), [str(stack), str(extra)], separate=True).Close()
    with rasterio.open(out) as v:
        assert v.count == n, f"VRT has {v.count} bands, expected {n}"
    meta = json.loads(Path(stack).with_suffix(".json").read_text())
    meta.update(features=F.feature_names(stack) + names, extra_features=str(extra))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return out


def main():
    global CLASSES, OUT, MAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="n_iter 5, 300 px per class, no map")
    ap.add_argument("--stack", type=Path, default=F.OUT_TIF)
    ap.add_argument("--merge-low", action="store_true", help="4 classes: Low merged into Non-forest (< 1.22 m)")
    ap.add_argument("--exclude-site", nargs="*", default=[], help="sites dropped from training; _ for space (Bunda_Extra)")
    ap.add_argument("--sar-only", action="store_true", help="NISAR features only (GCOV + GSLC), no S2")
    ap.add_argument("--tag", help="run tag instead of the generated one (e.g. 4classTF_GSLC)")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE, help="random_state of folds, search and model")
    ap.add_argument("--s2-only", action="store_true", help="the 13 S2 bands only, no NISAR")
    ap.add_argument("--no-map", action="store_true", help="no full-scene prediction")
    ap.add_argument("--no-nisar", action="store_true", help="drop the NISAR features (S1 + S2 run, Stage 1.4)")
    ap.add_argument("--site", help="train and K-fold inside this site only; _ for space (Stage 2)")
    ap.add_argument("--subsample", type=int, help="stratified random subset of N rows, seeded by --seed (Stage 2)")
    ap.add_argument("--extra-features", type=Path, help="raster of extra bands on the stack grid (Stage 2)")
    ap.add_argument("--keep-nan", action="store_true", help="only S2 must be finite; NaN SAR goes to XGBoost as missing")
    ap.add_argument("--nan-augment", action="store_true", help="one copy per row with one acquisition set to NaN")
    ap.add_argument("--cuts", type=float, nargs="+", help="class cuts on `mean` (np.digitize); class_defs")
    ap.add_argument("--extra-labels", type=Path, help="parquet of extra label px (site, row, col, mean); needs --cuts")
    a = ap.parse_args()
    a.keep_nan = a.keep_nan or a.nan_augment
    if a.cuts:
        if a.merge_low:
            ap.error("--cuts and --merge-low cannot be used together")
        assert a.cuts == sorted(a.cuts), "--cuts must increase"
        CLASSES = ([f"<{a.cuts[0]:g}"] + [f"{lo:g}-{hi:g}" for lo, hi in zip(a.cuts, a.cuts[1:]) ]
                   + [f">={a.cuts[-1]:g}"])
        a.tag = a.tag or f"{len(CLASSES)}class_cuts_" + "_".join(f"{c:g}" for c in a.cuts)
    if sum((a.s2_only, a.sar_only, a.no_nisar)) > 1:
        ap.error("--s2-only, --sar-only and --no-nisar cannot be used together")
    a.exclude_site = [s.replace("_", " ") for s in a.exclude_site]   # conda run splits args at spaces
    a.site = a.site and a.site.replace("_", " ")
    if a.merge_low:
        CLASSES = CLASSES[1:]
    tag = a.tag or (f"{len(CLASSES)}class" + "".join(f"_no{s.replace(' ', '')}" for s in a.exclude_site)
                    + ("_NISARonly" if a.sar_only else "") + ("_S2only" if a.s2_only else "") + ("_noNISAR" if a.no_nisar else "")
                    + (f"_site{a.site.replace(' ', '')}" if a.site else "") + (f"_n{a.subsample}" if a.subsample else "")
                    + ("_cov" if a.extra_features else "") + (f"_seed{a.seed}" if a.seed != RANDOM_STATE else "")
                    + ("_nanaug" if a.nan_augment else "_keepnan" if a.keep_nan else ""))
    if (a.merge_low or a.exclude_site or a.sar_only or a.s2_only or a.no_nisar or a.seed != RANDOM_STATE or a.tag
            or a.site or a.subsample or a.extra_features or a.keep_nan):
        sensors = "NISAR" if a.sar_only else "S2" if a.s2_only else "S1_S2" if a.no_nisar else "NISAR_S2"
        OUT, MAP = HERE / "outputs" / f"classify_{tag}", HERE / "outputs" / f"XGB_DP_{sensors}_2026_{tag}.tif"
    if a.n_jobs > 10:
        ap.error("--n-jobs must be <= 10 (nisar_2026 core cap, user 2026-10-20)")
    n_iter = 5 if a.quick else a.n_iter
    out = OUT / "quick" if a.quick else OUT
    out.mkdir(parents=True, exist_ok=True)

    if a.extra_features:
        n_stack = len(F.training_table(a.stack)[0])
        a.stack = stack_with_extra(a.stack, a.extra_features)
    need = (F.S2_BANDS + F.S2_INDICES) if a.keep_nan else None
    extra = None
    if a.extra_labels:                             # divergence 18
        if not a.cuts:
            ap.error("--extra-labels needs --cuts")
        extra = pd.read_parquet(a.extra_labels)[["site", "row", "col", "mean"]]
        extra["cls"] = np.digitize(extra["mean"], a.cuts).astype("int8")   # only for the --quick subset
    lab, X = F.training_table(a.stack, quick=a.quick, need=need, extra=extra)
    if a.extra_features and not a.quick:
        assert len(lab) == n_stack, f"extra features drop rows: {len(lab)} != {n_stack}"
    if a.merge_low:
        lab["cls"] = lab["cls"].clip(lower=1) - 1
    if a.cuts:
        new = np.digitize(lab["mean"], a.cuts).astype("int8")
        if a.cuts == [1.22, 2.35, 5.381]:
            assert (new == lab["cls"].clip(lower=1) - 1).all(), "--cuts 1.22 2.35 5.381 != --merge-low"
        lab["cls"] = new
    if a.exclude_site:
        assert set(a.exclude_site) <= set(lab["site"]), f"unknown site in {a.exclude_site}"
        keep = ~lab["site"].isin(a.exclude_site).to_numpy()
        lab, X = lab[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
        print(f"excluded {a.exclude_site}: {len(lab):,} rows left", flush=True)
    if a.sar_only:
        X = X[[c for c in X.columns if c[:5] in F.PASSES]]
        print(f"NISAR only: {X.shape[1]} features", flush=True)
    if a.s2_only:
        X = X[F.S2_BANDS + F.S2_INDICES]
        assert X.shape[1] == 13, X.columns
        print(f"S2 only: {X.shape[1]} features", flush=True)
    if a.no_nisar:
        X = X[[c for c in X.columns if c[:5] not in F.PASSES]]
        print(f"no NISAR: {X.shape[1]} features ({', '.join(c for c in X.columns if c.startswith('S1_'))})", flush=True)
    if a.site:
        assert a.site in set(lab["site"]), f"unknown site {a.site}"
        keep = (lab["site"] == a.site).to_numpy()
        lab, X = lab[keep].reset_index(drop=True), X[keep].reset_index(drop=True)
        print(f"site {a.site} only: {len(lab):,} rows", flush=True)
    if a.subsample:
        from sklearn.model_selection import train_test_split
        keep = np.sort(train_test_split(lab.index, train_size=a.subsample, stratify=lab["cls"], random_state=a.seed)[0])
        lab, X = lab.loc[keep].reset_index(drop=True), X.loc[keep].reset_index(drop=True)
        print(f"subsample: {len(lab):,} rows (stratified, seed {a.seed})", flush=True)
    present = np.unique(lab["cls"])            # --site: a class can be absent; XGBoost needs codes 0..k-1
    if len(present) < len(CLASSES) and not (a.quick or a.no_map):
        ap.error(f"classes {sorted(set(range(len(CLASSES))) - set(present))} absent: use --no-map")
    y, w = np.searchsorted(present, lab["cls"]), compute_sample_weight("balanced", lab["cls"])
    print(lab.groupby("cls").size().rename(dict(enumerate(CLASSES))).to_string(), flush=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(X, y))
    model = xgb.XGBClassifier(objective="multi:softprob", n_estimators=500, random_state=a.seed, n_jobs=1)
    Xf, yf, wf, cv_folds, n = X.to_numpy(), y, w, folds, len(y)
    if a.nan_augment:                              # divergence 16
        import re
        groups = sorted({m.group(1) for c in X.columns if (m := re.match(r"(\d{3}_[AD]_\d{4})_", c))})
        assert len(groups) >= 2, f"--nan-augment needs per-acquisition bands, found {groups}"
        drop = np.random.default_rng(a.seed).integers(len(groups), size=n)
        Xa = X.copy()
        for k, g in enumerate(groups):
            Xa.loc[drop == k, [c for c in X.columns if c.startswith(g + "_")]] = np.nan
        Xf, yf, wf = np.vstack([Xf, Xa.to_numpy()]), np.concatenate([y, y]), np.concatenate([w, w])
        folds = [(np.concatenate([tr, tr + n]), te) for tr, te in cv_folds]           # search: originals scored
        cv_folds = [(np.concatenate([tr, tr + n]), np.concatenate([te, te + n])) for tr, te in cv_folds]
        lab["masked_acq"] = np.array(groups)[drop]
        print(f"nan-augment: {n:,} copies, one of {groups} set to NaN in each "
              f"({pd.Series(lab['masked_acq']).value_counts().to_dict()})", flush=True)

    t0 = time.time()
    print(f"search: {n_iter} iterations x 5 folds, n_jobs {a.n_jobs}, seed {a.seed}", flush=True)
    search = RandomizedSearchCV(model, PARAM_DIST, n_iter=n_iter, cv=folds, scoring="f1_weighted",
                                n_jobs=a.n_jobs, random_state=a.seed, verbose=1)
    search.fit(Xf, yf, sample_weight=wf)
    print(f"search done in {time.time() - t0:.0f} s; best {search.best_score_:.4f} {search.best_params_}", flush=True)

    best = search.best_estimator_                  # refitted on all rows, with the weights
    naive = present[best.predict(X.to_numpy())]
    oof_all = present[cross_val_predict(clone(best), Xf, yf, cv=cv_folds, n_jobs=min(a.n_jobs, 5),
                                        params={"sample_weight": wf})]
    oof = oof_all[:n]
    preds = {"naive": naive, "kfold": oof} | ({"kfold_masked": oof_all[n:]} if a.nan_augment else {})
    print(f"K-fold done ({time.time() - t0:.0f} s)", flush=True)
    met, cms = report(lab, preds)

    met.to_csv(out / "metrics.csv", index=False)
    for k, cm in cms.items():
        cm.to_csv(out / f"confusion_{k}.csv")
    pd.DataFrame(search.cv_results_).drop(columns="params").to_csv(out / "search_cv_results.csv", index=False)
    pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False).to_csv(out / "importance.csv")
    if not a.quick:
        F.record_importance(tag, "XGBClassifier", "class", best.feature_importances_, list(X.columns),
                            best.importance_type or "gain", n_iter, a.stack)
    lab.assign(**{f"pred_{k}": v for k, v in preds.items()}).to_parquet(out / "predictions.parquet", index=False)
    (out / "best_params.json").write_text(json.dumps({k: float(v) for k, v in search.best_params_.items()}, indent=1))
    with open(out / f"xgb_{len(CLASSES)}class.pkl", "wb") as fh:
        pickle.dump({"model": best, "features": list(X.columns), "classes": CLASSES}, fh)

    s = met[met.site == "all"].set_index("eval")[["oa", "balanced_acc", "kappa", "f1_weighted"]]
    print(s.round(4).to_string(), flush=True)
    if "source" in lab:
        print("per source (report these, not `all`):\n"
              + met[met.site.str.startswith("src_")].set_index(["eval", "site"])[s.columns].round(4).to_string(),
              flush=True)
    for k, cm in cms.items():
        print(f"confusion {k}:\n{cm.to_string()}", flush=True)
    if a.quick or a.no_map:
        return

    best.set_params(n_jobs=a.n_jobs)
    bands = [F.feature_names(a.stack).index(c) + 1 for c in X.columns]
    F.predict_raster([best], ["class"], MAP, dtype="uint8", nodata=255, path=a.stack, bands=bands,
                     need=[list(X.columns).index(c) for c in need] if need else None)
    back = F.read_at(MAP, lab)[:, 0]
    assert (back == naive).all(), f"map != in-memory naive prediction at {(back != naive).sum()} training px"
    print("check: map equals the naive prediction at every training pixel", flush=True)
    with rasterio.open(MAP) as src:
        cls = src.read(1)
    frac = pd.Series(np.bincount(cls[cls != 255], minlength=len(CLASSES)), index=CLASSES)
    print(f"mainland class fractions:\n{(frac / frac.sum()).round(4).to_string()}", flush=True)
    print(f"total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
