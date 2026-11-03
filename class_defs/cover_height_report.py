"""Part B step B4 (PLAN_cover_height.md): combine cover x height predictions and report.

Reads outputs/cover_height/pred_*.parquet (cover_height_rf.py). Combined mean = c^ * mhi^ + (1 - c^) * mlo^
(c^ clipped to 0..1); variants: mlo = training-fold median of mlo; mlo = true mlo. Direct = the mean model.
Every metric per group (Dampier / Yawuru kept / Yawuru burnt), never blended. Fold mean, fold SD (mean over
seeds), seed SD (SD of the seed fold-means). A difference is "real" if > 2 x paired-fold SD and > seed SD.
Classes: np.digitize(pred mean, cuts) for cut sets old / nfi / gmm50_k4 (outputs/cut_sets.json), pooled over folds.

Outputs: class_defs/REPORT_cover_height.txt, outputs/cover_height/report_*.csv, fig_r2_stacks.png, fig_obs_pred.png.
  --partial   allow missing pred files (progress check while B3 runs).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, r2_score)

import common as C
from cover_height_rf import KEY, OUTD, SEEDS, TARGETS, groups

STACKS = ["MS", "MS+G", "SAR", "SAR+G", "ALL", "ALL+G"]
T = list(TARGETS)  # cover, mhi, mlo, mean
MEANS = ["mean", "comb", "comb_lo_med", "comb_lo_true"]  # direct + 3 combined variants
CUTSETS = ["old", "nfi", "gmm50_k4"]
REPORT = C.HERE / "REPORT_cover_height.txt"
XGB = {"old": 0.607, "nfi": 0.614, "gmm50_k4": 0.612}  # CHECKPOINT: XGB K-fold kappa, 39 features, Dampier

# Hand-written from the 2026-11-02 run (B3 18 runs, B3b ALL + ALL+G); numbers are in the sections below.
READING = """
0. READING (Dampier / Yawuru kept; R2 unless stated)
----------------------------------------------------
| Finding                         | Value                                                         |
|---------------------------------|---------------------------------------------------------------|
| Best stack                      | ALL+G: mean 0.785 / 0.893; cover 0.793 / 0.913; mhi 0.717 / 0.638 |
| GLCM gain on ALL (real)         | mean +0.012 / +0.013; cover +0.018 / +0.010; mhi +0.018 / +0.123 |
| GLCM gain on SAR (real)         | mean +0.007 / +0.014; mhi +0.018 / +0.210; mlo +0.040 / +0.084 |
| GLCM gain on MS (real)          | mean +0.027 / +0.044; mhi +0.049 / +0.214                      |
| Combined vs direct mean         | no real gain: |dR2| <= 0.009 in every stack; oracle mlo no gain |
| SAR alone (NISAR + S1)          | mean 0.687 / 0.841; SAR+G 0.693 / 0.855                        |
| Add MS to SAR / add SAR to MS   | mean +0.086 / +0.038 (Dampier); +0.038 / +0.090 (Yawuru kept)  |
| SAR stack permutation           | mhi: NISAR 1.21, S1 0.13; mlo: S1 1.49, NISAR 0.82; cover 0.53 / 0.46 |
| Classes, ALL+G, kappa (Dampier) | old 0.635 / nfi 0.633 / gmm50_k4 0.609 vs XGB 0.607 / 0.614 / 0.612 |
| Bare recall (old, nfi)          | 0.25-0.26 (ALL+G, Dampier): the regression pulls Bare up      |
| B3b: burnt rows added           | shared rows: Dampier mean dR2 -0.000; kept mean -0.002 (not real) |
| B3b: burnt rows                 | mean 0.754, bias -0.114 m (pred below pre-fire label, in training) |
| B3b ALL+G                       | shared rows: Dampier mean dR2 -0.000; burnt rows mean 0.777, bias -0.091 m |

- GLCM gain is optimistic: 100-110 m windows overlap test px under random K-fold.
- Yawuru mhi has the largest fold SD (0.08 for MS); read Yawuru mhi gains as direction, not size.
- Classes vs XGB are not paired (XGB: 39 features, no S1, own folds); +0.02-0.03 kappa for old / nfi.
- Combining adds no skill but gives cover and mhi as separate products at the same px-mean accuracy.
""".strip("\n").split("\n")
READING = [""] + READING


# ------------------------------------------------------------------------------------------- load

def load_run(stack, seed, sfx, partial):
    ps = {}
    for t in T:
        fn = OUTD / f"pred_{stack}_{t}_s{seed}{sfx}.parquet"
        if not fn.exists():
            if partial:
                return None
            raise FileNotFoundError(fn)
        ps[t] = pd.read_parquet(fn)
    w = ps["cover"][KEY + ["group", "fold"]].copy()
    for t, p in ps.items():
        assert (p[KEY].to_numpy() == w[KEY].to_numpy()).all() and (p.fold.to_numpy() == w.fold.to_numpy()).all()
        assert p.pred.notna().all(), (stack, t, seed)
        w[f"y_{t}"], w[f"p_{t}"], w[f"u_{t}"] = p.y.to_numpy(), p.pred.to_numpy(), p.use.to_numpy()
    # combine
    c = w.p_cover.clip(0, 1)
    w["p_comb"] = c * w.p_mhi + (1 - c) * w.p_mlo
    lo_med = np.zeros(len(w))
    for k in range(5):
        tr = (w.fold != k) & w.u_mlo
        lo_med[w.fold == k] = np.median(w.loc[tr, "y_mlo"])
    w["p_comb_lo_med"] = c * w.p_mhi + (1 - c) * lo_med
    w["p_comb_lo_true"] = c * w.p_mhi + (1 - c) * w.y_mlo.fillna(0)
    # true components reproduce mean
    tc, thi, tlo = w.y_cover, w.y_mhi.fillna(0), w.y_mlo.fillna(0)
    assert np.allclose(tc * thi + (1 - tc) * tlo, w.y_mean, atol=1e-4), "true-component combination"
    for m in MEANS[1:]:
        w[f"y_{m}"], w[f"u_{m}"] = w.y_mean, True
    w["stack"], w["seed"] = stack, seed
    return w


def load_all(sfx, stacks, partial):
    out = {}
    for s in stacks:
        for seed in SEEDS:
            w = load_run(s, seed, sfx, partial)
            if w is not None:
                out[(s, seed)] = w
    return out


def add_flag(w, rows):
    f = rows.set_index(KEY).s2_flag
    w = w.join(f, on=KEY)
    w["sub"] = np.where(w.group != "Yawuru kept", "", np.where(w.s2_flag, "kept: S2 flag", "kept: other"))
    return w


# ------------------------------------------------------------------------------------------- metrics

def reg(y, p):
    return dict(r2=r2_score(y, p), rmse=float(np.sqrt(np.mean((p - y) ** 2))), bias=float(np.mean(p - y)), n=len(y))


def fold_metrics(runs, gcol="group"):
    rec = []
    for (s, seed), w in runs.items():
        for t in T + MEANS[1:]:
            u = w[f"u_{t}"].to_numpy(bool)
            for (g, k), x in w[u].groupby([gcol, "fold"]):
                if g == "":
                    continue
                rec.append(dict(stack=s, seed=seed, target=t, group=g, fold=k, **reg(x[f"y_{t}"], x[f"p_{t}"])))
    return pd.DataFrame(rec)


def summarise(fm, keys, val):
    a = fm.groupby(keys + ["seed"])[val].agg(["mean", "std"])
    b = a.groupby(keys).agg(m=("mean", "mean"), fold_sd=("std", "mean"), seed_sd=("mean", "std"))
    return b.fillna(0)


def paired(fm, a_key, b_key, col, keys, val="r2"):
    """b - a per fold and seed, paired on keys + seed + fold."""
    A = fm[fm[col] == a_key].set_index(keys + ["seed", "fold"])[val]
    B = fm[fm[col] == b_key].set_index(keys + ["seed", "fold"])[val]
    d = (B - A).dropna().rename("d").reset_index()
    s = summarise(d, keys, "d")
    s["real"] = (s.m.abs() > 2 * s.fold_sd) & (s.m.abs() > s.seed_sd)
    return s


def class_metrics(runs, gcol="group", conf_seed=0):
    cuts = json.load(open(C.OUT / "cut_sets.json"))
    rec, conf = [], {}
    for (s, seed), w in runs.items():
        for cs in CUTSETS:
            cu = cuts[cs]["cuts"]
            yt = np.digitize(w.y_mean, cu)
            for v in ["mean", "comb"]:
                yp = np.digitize(w[f"p_{v}"], cu)
                for g, idx in w.groupby(gcol).indices.items():
                    if g == "":
                        continue
                    a, b = yt[idx], yp[idx]
                    p, r, f, n = precision_recall_fscore_support(a, b, labels=range(4), zero_division=0)
                    rec.append(dict(stack=s, seed=seed, cutset=cs, var=v, group=g, n=len(a), oa=(a == b).mean(),
                                    kappa=cohen_kappa_score(a, b), ba=balanced_accuracy_score(a, b),
                                    f1w=f1_score(a, b, average="weighted", zero_division=0),
                                    **{f"{m}{i}": x[i] for m, x in zip("PRF", (p, r, f)) for i in range(4)},
                                    **{f"n{i}": n[i] for i in range(4)}))
                    if seed == conf_seed:
                        conf[(s, cs, v, g)] = confusion_matrix(a, b, labels=range(4))
    return pd.DataFrame(rec), conf


def class_summary(cm, keys):
    a = cm.groupby(keys)[["kappa", "ba", "oa", "f1w"]].agg(["mean", "std"])
    out = pd.DataFrame({m: a[(m, "mean")] for m in ["kappa", "ba", "oa", "f1w"]})
    out["kappa_seed_sd"] = a[("kappa", "std")]
    return out


def importances(stacks, sfx):
    imp, perm = [], []
    for s in stacks:
        for t in T:
            fn = OUTD / f"imp_{s}_{t}{sfx}.csv"
            if not fn.exists():
                continue
            ii = pd.read_csv(fn)
            fi = ii[ii.kind == "impurity"].groupby("name").value.mean().nlargest(5)
            imp.append(f"  {s:6s} {t:5s} " + ", ".join(f"{n} {v:.3f}" for n, v in fi.items()))
            pp = ii[ii.kind == "perm_dR2"].groupby("name").value.mean()
            perm.append(pd.Series(pp, name=(s, t)))
    pm = pd.DataFrame(perm) if perm else pd.DataFrame()
    if len(pm):
        pm.index = pd.MultiIndex.from_tuples(pm.index, names=["stack", "target"])
    return imp, pm


# ------------------------------------------------------------------------------------------- figures

def figures(rs, runs, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(C.PROJ / "chm_regression"))
    import viz_style as V
    V.apply_style()
    c_base, c_glcm = V.TARGET_COLORS["mean"], V.TARGET_COLORS["max"]  # palette slots 1, 2 (validated pair)
    tg = ["cover", "mhi", "mlo", "mean", "comb"]
    title = {"cover": "cover 1.37 m", "mhi": "mean ≥ 1.37 m (μhi)", "mlo": "mean < 1.37 m (μlo)",
             "mean": "px mean, direct", "comb": "px mean, combined"}
    grps = ["Dampier", "Yawuru kept"]
    fig, ax = plt.subplots(2, 5, figsize=(14, 5.6), sharey="row")
    x = np.arange(3)
    for i, g in enumerate(grps):
        for j, t in enumerate(tg):
            a = ax[i, j]
            for k, (sfx, col, lab, dx) in enumerate([("", c_base, "base", -0.09), ("+G", c_glcm, "+ GLCM", 0.09)]):
                v = [rs.loc[(b + sfx, t, g)] for b in ["MS", "SAR", "ALL"]]
                a.errorbar(x + dx, [r.m for r in v], yerr=[r.fold_sd for r in v], fmt="o", ms=6, color=col,
                           elinewidth=1.2, capsize=0, label=lab, mec=V.SURFACE, mew=1.5)
            a.set_xticks(x, ["MS", "SAR", "ALL"])
            a.set_title(f"{title[t]}\n{g}", fontsize=9, color=V.TEXT_PRIMARY)
            a.grid(axis="y", color=V.GRID, lw=0.6)
        ax[i, 0].set_ylabel("K-fold R²", color=V.TEXT_SECONDARY)
    ax[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("RF R² by stack, without / with GLCM (bars: fold SD; 3 seeds)", fontsize=10, color=V.TEXT_PRIMARY)
    fig.tight_layout()
    f1 = OUTD / "fig_r2_stacks.png"
    fig.savefig(f1, dpi=150, facecolor=V.SURFACE)
    plt.close(fig)

    w = runs[(best, 0)]
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))
    for i, g in enumerate(grps):
        for j, t in enumerate(["cover", "mhi", "mean", "comb"]):
            a = ax[i, j]
            x_ = w[(w.group == g) & w[f"u_{t}"]]
            lim = (0, 1) if t == "cover" else (0, np.nanpercentile(x_[f"y_{t}"], 99.5))
            a.hexbin(x_[f"y_{t}"], x_[f"p_{t}"], gridsize=45, extent=lim * 2, cmap=V.SEQUENTIAL_CMAP, bins="log",
                     mincnt=1, linewidths=0)
            a.plot(lim, lim, color=V.TEXT_MUTED, lw=1)
            a.set_xlim(lim), a.set_ylim(lim)
            r = reg(x_[f"y_{t}"], x_[f"p_{t}"])
            a.set_title(f"{title[t]} · {g}\nR² {r['r2']:.3f}  RMSE {r['rmse']:.3f}", fontsize=9, color=V.TEXT_PRIMARY)
            a.set_xlabel("LiDAR", color=V.TEXT_SECONDARY), a.set_ylabel("predicted", color=V.TEXT_SECONDARY)
    fig.suptitle(f"Observed vs predicted, stack {best}, seed 0 (log density)", fontsize=10, color=V.TEXT_PRIMARY)
    fig.tight_layout()
    f2 = OUTD / "fig_obs_pred.png"
    fig.savefig(f2, dpi=150, facecolor=V.SURFACE)
    plt.close(fig)
    return f1, f2


# ------------------------------------------------------------------------------------------- report

def fmt(df, nd=3):
    return df.round(nd).to_string()


def section(L, title, body=""):
    L += ["", title, "-" * len(title)] + ([body] if body else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    rows = pd.read_parquet(OUTD / "rows.parquet")
    runs = {k: add_flag(w, rows) for k, w in load_all("", STACKS, a.partial).items()}
    assert runs, "no complete runs"
    have = sorted({s for s, _ in runs}, key=STACKS.index)
    fm = fold_metrics(runs)
    fs = fold_metrics(runs, gcol="sub")
    fm.to_csv(OUTD / "report_fold_metrics.csv", index=False)
    rs = summarise(fm, ["stack", "target", "group"], "r2")
    L = ["REPORT — cover x height regression, h = 1.37 m (Part B, PLAN_cover_height.md)",
         f"Runs: {len(runs)} (stack x seed) of {len(STACKS) * len(SEEDS)}; stacks {have}.",
         "RF 500, max_features 0.33, leaf 5; 5 random stratified folds (site x cover quintile); seeds 0-2.",
         "Optimistic: random K-fold under spatial autocorrelation; GLCM windows (100-110 m) overlap test px.",
         "Values: fold mean (fold SD, seed SD). Groups never blended."] + READING

    for g in ["Dampier", "Yawuru kept"]:
        for val in ["r2", "rmse", "bias"]:
            s = summarise(fm[fm.group == g], ["stack", "target"], val)
            tab = s.m.unstack("target").reindex(index=have, columns=T + MEANS[1:])
            section(L, f"1. {val.upper()} — {g}", fmt(tab))
            if val == "r2":
                sd = pd.DataFrame({f"{k} SD (max over stacks)": s[k].unstack("target").reindex(columns=T + ["comb"])
                                   .max() for k in ["fold_sd", "seed_sd"]}).T
                L.append(fmt(sd))

    # GLCM gain
    gg = []
    for b in ["MS", "SAR", "ALL"]:
        if b in have and b + "+G" in have:
            p = paired(fm, b, b + "+G", "stack", ["target", "group"])
            gg.append(p.assign(stack=b).reset_index())
    if gg:
        g = pd.concat(gg).set_index(["group", "stack", "target"])[["m", "fold_sd", "seed_sd", "real"]]
        section(L, "2. GLCM GAIN — R2(+G) - R2(base), paired folds", fmt(g.rename(columns={"m": "dR2"})))

    # combined vs direct
    cd = []
    for m in MEANS[1:]:
        p = paired(fm, "mean", m, "target", ["stack", "group"])
        cd.append(p.assign(variant=m).reset_index())
    cd = pd.concat(cd).set_index(["group", "variant", "stack"])[["m", "fold_sd", "seed_sd", "real"]]
    section(L, "3. COMBINED - DIRECT mean, R2, paired folds (comb_lo_true uses the true mlo: oracle)",
            fmt(cd.rename(columns={"m": "dR2"})))

    # source contribution both ways
    sc = []
    for sfx in ["", "+G"]:
        for one, other in [("MS", "SAR"), ("SAR", "MS")]:
            if one + sfx in have and "ALL" + sfx in have:
                p = paired(fm, one + sfx, "ALL" + sfx, "stack", ["target", "group"])
                sc.append(p.assign(added=f"{other} to {one}{sfx}").reset_index())
    if sc:
        sc = pd.concat(sc).set_index(["group", "added", "target"])[["m", "fold_sd", "real"]]
        section(L, "4. SOURCE CONTRIBUTION BOTH WAYS — R2(ALL) - R2(one source)", fmt(sc.rename(columns={"m": "dR2"})))

    # classes
    cm, conf = class_metrics(runs)
    cm.to_csv(OUTD / "report_class_metrics.csv", index=False)
    cs = class_summary(cm, ["group", "cutset", "var", "stack"])
    section(L, "5. CLASSES from predicted mean (direct = mean model, comb = combined); pooled folds, mean of seeds",
            "XGB K-fold kappa (CHECKPOINT, 39 features, Dampier): " + ", ".join(f"{k} {v}" for k, v in XGB.items())
            + "\n" + fmt(cs))
    for st in [s for s in ["ALL+G", "SAR+G"] if s in have]:
        pc = cm[cm["stack"] == st].groupby(["group", "cutset", "var"])[[f"{m}{i}" for i in range(4) for m in "PRF"]].mean()
        section(L, f"5b. PER-CLASS P / R / F1 (c0 -> c3), stack {st}", fmt(pc, 2))
        for g in ["Dampier", "Yawuru kept"]:
            for cs_ in CUTSETS:
                L.append(f"confusion {st} {cs_} comb {g} (rows true c0..c3, seed 0):\n"
                         + str(conf.get((st, cs_, "comb", g))))

    # importances
    imp, pm = importances(have, "")
    section(L, "6. IMPORTANCE — permutation R2 drop by group (seed 0, fold mean)", fmt(pm) if len(pm) else "none")
    section(L, "6b. IMPORTANCE — top 5 impurity (seed 0, fold mean)", "\n".join(imp))

    # per site, descriptive
    best = max([s for s in have], key=lambda s: rs.loc[(s, "mean", "Dampier")].m)
    ps = []
    for (s, seed), w in runs.items():
        if s != best:
            continue
        for t in ["cover", "mhi", "mean", "comb"]:
            u = w[f"u_{t}"].to_numpy(bool)
            for site, x in w[u].groupby("site"):
                ps.append(dict(seed=seed, site=site, group=x.group.iloc[0], target=t,
                               **reg(x[f"y_{t}"], x[f"p_{t}"])))
    ps = pd.DataFrame(ps).groupby(["group", "site", "target"])[["rmse", "bias"]].mean().unstack("target")
    section(L, f"7. PER SITE (descriptive), stack {best}, RMSE / bias, mean of seeds", fmt(ps))
    s2 = summarise(fs, ["stack", "target", "group"], "r2").m.unstack("group")
    section(L, "7b. Yawuru kept split: S2-flag sites (govcowwell, YA25_3) vs other, R2 (descriptive)",
            fmt(s2.loc[[(best, t) for t in T + ["comb"] if (best, t) in s2.index]]))

    # B3b
    bur = {k: add_flag(w, rows) for k, w in load_all("_burnt", ["ALL", "ALL+G"], True).items()}
    if bur:
        L += b3b(runs, bur)
    else:
        section(L, "8. B3b (burnt areas added back)", "not run yet")

    f1, f2 = figures(rs, runs, best) if all(s in have for s in STACKS) else (None, None)
    L += ["", f"Figures: {f1}, {f2}", f"Tables: {OUTD}\\report_*.csv"]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {REPORT}")


def b3b(runs, bur):
    L = []
    have = sorted({s for s, _ in bur})
    # shared rows: same test fold
    for (s, seed), wb in bur.items():
        if (s, seed) in runs:
            m = runs[(s, seed)][KEY + ["fold"]].merge(wb[KEY + ["fold"]], on=KEY, suffixes=("", "_b"))
            assert len(m) == len(runs[(s, seed)]) and (m.fold == m.fold_b).all(), "B3b folds differ from B3"
    fb = fold_metrics(bur)
    fb.to_csv(OUTD / "report_fold_metrics_burnt.csv", index=False)
    section(L, f"8. B3b — burnt areas added back (stacks {have}); all 43,134 rows, same folds",
            "Shared-row test folds asserted equal to B3. Burnt rows are fitted (random K-fold).")
    for val in ["r2", "rmse", "bias"]:
        s = summarise(fb, ["stack", "group", "target"], val).m.unstack("target").reindex(columns=T + MEANS[1:])
        section(L, f"8a. B3b {val.upper()} per group", fmt(s))
    # paired B3b - B3 on shared rows (Dampier, Yawuru kept: rows identical, folds identical)
    rec = []
    for (s, seed), wb in bur.items():
        if (s, seed) not in runs:
            continue
        fa = fold_metrics({(s, seed): runs[(s, seed)]}).assign(run="B3")
        f_ = fb[(fb["stack"] == s) & (fb.seed == seed)].assign(run="B3b")
        rec.append(pd.concat([fa, f_]))
    if rec:
        both = pd.concat(rec)
        both = both[both.group != "Yawuru burnt"]
        for val in ["r2", "rmse"]:
            p = paired(both, "B3", "B3b", "run", ["stack", "group", "target"], val=val)
            section(L, f"8b. B3b - B3 on shared rows, {val.upper()} (paired folds)",
                    fmt(p.rename(columns={"m": f"d{val}"})))
    cm, conf = class_metrics(bur)
    cm.to_csv(OUTD / "report_class_metrics_burnt.csv", index=False)
    section(L, "8c. B3b classes per group (pooled folds, mean of seeds)",
            fmt(class_summary(cm, ["group", "cutset", "var", "stack"])))
    pc = cm.groupby(["stack", "group", "cutset", "var"])[[f"{m}{i}" for i in range(4) for m in "PRF"]].mean()
    section(L, "8d. B3b per-class P / R / F1", fmt(pc, 2))
    for k, v in conf.items():
        if k[2] == "comb" and k[1] == "nfi":
            L.append(f"confusion B3b {k[0]} nfi comb {k[3]} (seed 0):\n{v}")
    imp, pm = importances(have, "_burnt")
    section(L, "8e. B3b importance — permutation R2 drop by group (seed 0)", fmt(pm) if len(pm) else "none")
    section(L, "8f. B3b top 5 impurity", "\n".join(imp))
    return L


if __name__ == "__main__":
    main()
