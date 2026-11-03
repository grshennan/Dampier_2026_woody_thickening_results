"""Regression track, follow-on M (user 2026-11-03): MLP vs RF, ALL stack (49 features), env pytorch (GPU).

Same test as the classification track (mlp_2026.py): can a per-pixel MLP improve on the RF?
Targets: px mean, p98, cover 1.37; then the p98 residual r = p98 - g(cover).
Rows: 39,023 (Dampier 2026 + Yawuru kept); folds: folds_cover_height.json (same as RF).
RF comparator from disk (no refit): pred_ALL_{mean,cover}_s{0,1,2}.parquet; p98_cover/kfold_preds.parquet (p98_direct,
res_direct, seed 0 only; RF seed SD 0.000, so seed 0 is paired with every MLP seed).

Model (from mlp_2026.py net / fit): 49 -> h1 -> h2 -> 1, ReLU, dropout; AdamW lr 1e-3, wd 1e-4; batch 512; early stop on
a 15 % split of the training fold (stratified on y deciles), patience 20, max 300 epochs, best epoch restored; inputs
standardised on the training part. Divergences from mlp_2026.py:
  1. MSE loss on y standardised on the training part (not cross-entropy).
  2. ReduceLROnPlateau (factor 0.5, patience 8) on the validation loss (mlp_2026 had a fixed lr).
  3. Predictions clipped: heights >= 0, cover to [0, 1]; the clipped fraction is reported.
  4. Inference in float64 (map = model at the label px to 1e-5).
Grid per target: 128-64 / 256-128 / 512-256 x dropout 0.1 / 0.2 / 0.4 (9). Chosen by pooled K-fold R2 on the test
folds (optimistic; the RF is not tuned). Best config: seeds 0, 1, 2. Stratified K-fold only; no leave-site-out.

  kfold   -> outputs/mlp_regression/*.csv, pred_MLP_<t>_s<seed>.parquet; REPORT_mlp_regression.txt
  maps    fit on all 39,023 px (best config, seed 0) -> E:/nisar_2026/regression/{height_mean,height_p98,cover137}_MLP.tif,
          p98_resid_MLP.tif, p98_resid_w5_MLP.tif, mlp_ALL_<t>.pt; RF vs MLP map agreement with an 8 x 8 block-shuffle null
  --quick fold 0, config 0, 30 epochs per target; g reproduction check; 1 M px tile time. Writes nothing.

    C:\\Users\\glens\\miniconda3\\Scripts\\conda.exe run -n pytorch --no-capture-output python -u mlp_regression_2026.py kfold --quick
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")  # HGB / BLAS: regression track core cap

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, precision_recall_fscore_support, r2_score
from sklearn.model_selection import train_test_split

import common as C
import cover_height_rf as R
import p98_cover as P
from cover_height_report import paired, reg, summarise

OUTM = C.OUT / "mlp_regression"
REPORT = C.HERE / "REPORT_mlp_regression.txt"
KEY = ["site", "row", "col"]
CV = C.col(1.37)
GROUPS = ["pooled", "Dampier", "Yawuru kept"]
TGT = {"mean": ("mean", 0.0, None), "p98": ("p98", 0.0, None), "cover": (CV, 0.0, 1.0)}
GRID = [dict(w=w, drop=dr, wd=1e-4) for w in [(128, 64), (256, 128), (512, 256)] for dr in [0.1, 0.2, 0.4]]
SEEDS = [0, 1, 2]
MAX_EP, PATIENCE = 300, 20
RF_R2 = {"mean": 0.795, "cover": 0.811, "p98": 0.724}  # pooled fold-mean K-fold R2 (REPORT_regression / _p98_cover)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------------------- model

def net(n_in, w, drop):
    return torch.nn.Sequential(torch.nn.Linear(n_in, w[0]), torch.nn.ReLU(), torch.nn.Dropout(drop),
                               torch.nn.Linear(w[0], w[1]), torch.nn.ReLU(), torch.nn.Dropout(drop),
                               torch.nn.Linear(w[1], 1))


class Reg:
    """Fitted MLP + scalers; .predict(X) -> clipped float64 (sklearn-like, for predict_raster / map_checks)."""

    def __init__(self, m, mu, sd, ym, ys, lo, hi, cfg):
        self.m = copy.deepcopy(m).double().to(DEV).eval()
        self.mu, self.sd = torch.tensor(mu, dtype=torch.float64, device=DEV), torch.tensor(sd, dtype=torch.float64,
                                                                                          device=DEV)
        self.ym, self.ys, self.lo, self.hi, self.cfg = float(ym), float(ys), lo, hi, cfg

    def raw(self, X):
        X = np.asarray(X, dtype="float64")
        out = np.empty(len(X))
        with torch.no_grad():
            for i in range(0, len(X), 1 << 20):
                xb = torch.tensor(X[i:i + (1 << 20)], device=DEV)
                out[i:i + len(xb)] = (self.m((xb - self.mu) / self.sd).squeeze(1) * self.ys + self.ym).cpu().numpy()
        return out

    def predict(self, X):
        return np.clip(self.raw(X), self.lo, self.hi)

    def save(self, path):
        torch.save(dict(state=self.m.state_dict(), mu=self.mu.cpu().numpy(), sd=self.sd.cpu().numpy(), ym=self.ym,
                        ys=self.ys, lo=self.lo, hi=self.hi, cfg=self.cfg), path)

    @staticmethod
    def load(path, n_in):
        z = torch.load(path, weights_only=False)
        m = net(n_in, z["cfg"]["w"], z["cfg"]["drop"]).double()
        m.load_state_dict(z["state"])
        return Reg(m, z["mu"], z["sd"], z["ym"], z["ys"], z["lo"], z["hi"], z["cfg"])


def fit(X, y, cfg, seed, lo, hi, max_ep=MAX_EP):
    torch.manual_seed(seed)
    strat = pd.qcut(y, 10, labels=False, duplicates="drop")
    tr, va = train_test_split(np.arange(len(y)), test_size=0.15, stratify=strat, random_state=seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    ym, ys = y[tr].mean(), y[tr].std() + 1e-9
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=DEV)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32, device=DEV)
    m = net(X.shape[1], cfg["w"], cfg["drop"]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    loss_fn = torch.nn.MSELoss()
    tr_t, va_t = torch.tensor(tr, device=DEV), torch.tensor(va, device=DEV)
    best, best_state, best_ep, wait, hist = np.inf, None, -1, 0, []
    g = torch.Generator(device=DEV).manual_seed(seed)
    for ep in range(max_ep):
        m.train()
        perm = tr_t[torch.randperm(len(tr_t), device=DEV, generator=g)]
        tl = 0.0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = loss_fn(m(Xt[b]).squeeze(1), yt[b])
            loss.backward()
            opt.step()
            tl += loss.item() * len(b)
        m.eval()
        with torch.no_grad():
            vl = loss_fn(m(Xt[va_t]).squeeze(1), yt[va_t]).item()
        sch.step(vl)
        hist.append((ep, tl / len(tr), vl, opt.param_groups[0]["lr"]))
        if vl < best - 1e-4:
            best, best_ep, wait = vl, ep, 0
            best_state = {n: t.detach().clone() for n, t in m.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.load_state_dict(best_state)
    return Reg(m, mu, sd, ym, ys, lo, hi, cfg), hist, best_ep


def kfold_one(X, y, fold, cfg, seed, lo, hi, folds=range(5), max_ep=MAX_EP):
    oof, trr2, eps, ran, models, hist, clip = np.full(len(y), np.nan), [], [], [], [], [], 0
    for k in folds:
        trn, te = fold != k, fold == k
        m, h, be = fit(X[trn], y[trn], cfg, seed, lo, hi, max_ep)
        raw = m.raw(X[te])
        clip += int(((raw < lo) | ((raw > hi) if hi is not None else False)).sum())
        oof[te] = np.clip(raw, lo, hi)
        trr2.append(r2_score(y[trn], m.predict(X[trn])))
        eps.append(be + 1)
        ran.append(len(h))
        models.append(m)
        hist += [dict(fold=k, epoch=e, train_loss=a, val_loss=b, lr=c) for e, a, b, c in h]
    return dict(oof=oof, train_r2=float(np.mean(trr2)), best_ep=eps, ran=ran, models=models, hist=hist, clip=clip)


def ridge_oof(X, y, fold):
    oof = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = fold != k, fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        oof[te] = Ridge(alpha=1.0).fit((X[tr] - mu) / sd, y[tr]).predict((X[te] - mu) / sd)
    return oof


def perm_sensor(models, X, y, fold, grp, rng, n_rep=3):
    """Permutation R2 drop per sensor on each test fold (rows of the group's columns permuted jointly; as RF perm_drop)."""
    rec = []
    for k, m in enumerate(models):
        te = fold == k
        Xte, yte = X[te], y[te]
        base = r2_score(yte, m.predict(Xte))
        for g, idx in grp.items():
            dd = []
            for _ in range(n_rep):
                Xp = Xte.copy()
                Xp[:, idx] = Xte[rng.permutation(len(Xte))][:, idx]
                dd.append(base - r2_score(yte, m.predict(Xp)))
            rec.append(dict(sensor=g, fold=k, dR2=float(np.mean(dd))))
    return rec


# ------------------------------------------------------------------------------------------- data

def load():
    d, base, _ = R.load(with_glcm=False)
    d = d[d.group != "Yawuru burnt"].reset_index(drop=True)
    d = d.merge(pd.read_parquet(P.OUTP / "rows_p98.parquet")[KEY + ["p98"]], on=KEY, how="left", validate="1:1")
    assert len(d) == 39023 and len(base) == 49 and d.p98.notna().all()
    assert np.isfinite(d[base].to_numpy(float)).all()
    return d, base


def rf_preds(d):
    out = {}
    for t in ["mean", "cover"]:
        for s in SEEDS:
            q = pd.read_parquet(R.OUTD / f"pred_ALL_{t}_s{s}.parquet")
            assert (q[KEY + ["fold"]].to_numpy() == d[KEY + ["fold"]].to_numpy()).all(), "RF row order / folds differ"
            out[(t, s)] = q.pred.to_numpy()
    k = pd.read_parquet(P.OUTP / "kfold_preds.parquet")
    assert (k[KEY + ["fold"]].to_numpy() == d[KEY + ["fold"]].to_numpy()).all(), "p98 row order / folds differ"
    for s in SEEDS:
        out[("p98", s)] = k.p98_direct.to_numpy()  # RF p98 seed 0 only (seed SD 0.000)
    return out, k


def fm_rows(model, t, seed, d, y, p):
    x = d[["group", "fold"]].assign(y=y, p=p)
    rec = []
    for g in GROUPS:
        xx = x if g == "pooled" else x[x.group == g]
        for k, xf in xx.groupby("fold"):
            rec.append(dict(model=model, target=t, seed=seed, group=g, fold=k, **reg(xf.y.to_numpy(), xf.p.to_numpy())))
    return rec


def g_fold(d, k, cov_pred):
    """g = E[p98 | cover] per fold (HGB, training-fold labels, as p98_cover.kfold); applied to cov_pred at the test rows.
    Also returns g applied to the RF cover_hat, to check against kfold_preds.g_hat."""
    fold = d.fold.to_numpy()
    g_rf, g_x = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for f in sorted(set(fold)):
        tr, te = fold != f, fold == f
        g = P.hgb(d.loc[tr, [CV]].to_numpy(), d.loc[tr, "p98"])
        g_rf[te] = g.predict(k.loc[te, ["cover_hat"]].to_numpy())
        g_x[te] = g.predict(cov_pred[te][:, None])
    return g_rf, g_x


# ------------------------------------------------------------------------------------------- kfold

def quick():
    import regression_2026 as RG
    d, base = load()
    rf, k = rf_preds(d)
    X, fold = d[base].to_numpy("float64"), d.fold.to_numpy()
    te = fold == 0
    for t, (yc, lo, hi) in TGT.items():
        y = d[yc].to_numpy("float64")
        t0 = time.time()
        r = kfold_one(X, y, fold, GRID[0], 0, lo, hi, folds=[0], max_ep=30)
        ro = ridge_oof(X, y, fold)
        print(f"{t:5s} fold 0: MLP R2 {r2_score(y[te], r['oof'][te]):.3f} (30 ep, cfg 0, train {r['train_r2']:.3f}, "
              f"best ep {r['best_ep']}) | RF {r2_score(y[te], rf[(t, 0)][te]):.3f} | ridge "
              f"{r2_score(y[te], ro[te]):.3f} | clipped {r['clip']} | {time.time() - t0:.0f} s", flush=True)
    g_rf, _ = g_fold(d, k, k.cover_hat.to_numpy())
    err = float(np.abs(g_rf - k.g_hat).max())
    print(f"g reproduction (RF cover_hat -> g_hat): max |d| {err:.2e}", flush=True)
    assert err < 1e-3
    d2, base2, bands, RR = RG.train_table()
    assert base2 == base
    RG.time_tile(r["models"][0], bands, RR)
    print("quick passed", flush=True)


def kfold():
    t0 = time.time()
    OUTM.mkdir(exist_ok=True)
    d, base = load()
    rf, k = rf_preds(d)
    X, fold = d[base].to_numpy("float64"), d.fold.to_numpy()
    grp = {g: [base.index(c) for c in cols] for g, cols in R.groups(base).items()}
    fm, eff, imp, mlp = [], [], [], {}
    for t, (yc, lo, hi) in TGT.items():
        y = d[yc].to_numpy("float64")
        for s in SEEDS:
            fm += fm_rows("RF", t, s, d, y, rf[(t, s)])
        rows, res = [], {}
        for i, cfg in enumerate(GRID):
            t1 = time.time()
            r = kfold_one(X, y, fold, cfg, 0, lo, hi)
            res[i] = r
            rows.append(dict(cfg=i, w=str(cfg["w"]), drop=cfg["drop"], wd=cfg["wd"], r2_pooled=r2_score(y, r["oof"]),
                             train_r2=r["train_r2"], best_ep_mean=np.mean(r["best_ep"]), ran_max=max(r["ran"]),
                             s=time.time() - t1))
            print(f"[{(time.time() - t0) / 60:5.1f} min] {t:5s} cfg {i} {cfg}: K-fold R2 {rows[-1]['r2_pooled']:.4f}, "
                  f"train {r['train_r2']:.4f}, best ep {r['best_ep']}", flush=True)
        cf = pd.DataFrame(rows)
        cf.to_csv(OUTM / f"configs_{t}.csv", index=False)
        b = int(cf.loc[cf.r2_pooled.idxmax(), "cfg"])
        runs = {0: res[b]}
        for s in SEEDS[1:]:
            runs[s] = kfold_one(X, y, fold, GRID[b], s, lo, hi)
        pd.DataFrame(runs[0]["hist"]).to_csv(OUTM / f"history_{t}.csv", index=False)
        for s, r in runs.items():
            fm += fm_rows("MLP", t, s, d, y, r["oof"])
            mlp[(t, s)] = r["oof"]
            d[KEY + ["group", "fold"]].assign(y=y, pred=r["oof"], use=True).to_parquet(
                OUTM / f"pred_MLP_{t}_s{s}.parquet")
        ro = ridge_oof(X, y, fold)
        fm += fm_rows("ridge", t, 0, d, y, ro)
        rng = np.random.default_rng(0)
        imp += [dict(target=t, **x) for x in perm_sensor(runs[0]["models"], X, y, fold, grp, rng)]
        kf = [r2_score(y, r["oof"]) for r in runs.values()]
        eff.append(dict(target=t, best_cfg=b, w=str(GRID[b]["w"]), drop=GRID[b]["drop"],
                        r2_kfold=np.mean(kf), seed_sd=np.std(kf, ddof=1), r2_train=np.mean([r["train_r2"] for r in
                                                                                            runs.values()]),
                        ridge_r2=r2_score(y, ro), best_ep_mean=np.mean(sum((r["best_ep"] for r in runs.values()), [])),
                        best_ep_max=max(sum((r["best_ep"] for r in runs.values()), [])),
                        folds_at_max_ep=sum(n >= MAX_EP for r in runs.values() for n in r["ran"]),
                        clipped_frac=np.mean([r["clip"] for r in runs.values()]) / len(y)))
        print(f"[{(time.time() - t0) / 60:5.1f} min] {t}: best cfg {b} {GRID[b]}; {eff[-1]}", flush=True)
    fm = pd.DataFrame(fm)
    fm.to_csv(OUTM / "fold_metrics.csv", index=False)
    pd.DataFrame(eff).to_csv(OUTM / "effectiveness.csv", index=False)
    pd.DataFrame(imp).to_csv(OUTM / "importance.csv", index=False)

    s = summarise(fm[(fm.model == "RF") & (fm.group == "pooled")], ["target"], "r2")
    for t, v in RF_R2.items():
        assert abs(s.loc[t].m - v) < 0.0006, f"RF {t} pooled R2 {s.loc[t].m:.4f} != {v}"
    print("RF pooled K-fold R2 reproduced (mean 0.795, cover 0.811, p98 0.724)", flush=True)

    # residual r = p98 - g(cover): RF (res_direct) vs MLP, seed 0
    g_rf, g_m = g_fold(d, k, mlp[("cover", 0)])
    err = float(np.abs(g_rf - k.g_hat).max())
    assert err < 1e-3, f"g not reproduced: max |d| {err}"
    rr = fm_rows("RF", "resid", 0, d, k.res_true.to_numpy(), k.res_direct.to_numpy())
    rr += fm_rows("MLP", "resid", 0, d, k.res_true.to_numpy(), mlp[("p98", 0)] - g_m)
    pd.DataFrame(rr).to_csv(OUTM / "resid_fold_metrics.csv", index=False)

    # class check: px-mean preds -> classes, cut set in cut_sets.json (read at run time)
    name, cs = next(iter(json.load(open(C.OUT / "cut_sets.json")).items()))
    yt = np.digitize(d["mean"], cs["cuts"])
    cc = []
    for model, src in (("RF", rf), ("MLP", mlp)):
        for s in SEEDS:
            yp = np.digitize(src[("mean", s)], cs["cuts"])
            for g in GROUPS:
                i = np.ones(len(d), bool) if g == "pooled" else (d.group == g).to_numpy()
                F = precision_recall_fscore_support(yt[i], yp[i], labels=range(len(cs["cuts"]) + 1),
                                                    zero_division=0)[2]
                cc.append(dict(model=model, seed=s, group=g, cut_set=name, macro_f1=F.mean(), worst_f1=F.min(),
                               f1w=f1_score(yt[i], yp[i], average="weighted", zero_division=0)))
    pd.DataFrame(cc).to_csv(OUTM / "class_check.csv", index=False)
    report()
    print(f"kfold done {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------------------------------- report

def cell(r, nd=3):
    return f"{r.m:.{nd}f} ({r.fold_sd:.{nd}f}, {r.seed_sd:.{nd}f})"


def report():
    fm = pd.read_csv(OUTM / "fold_metrics.csv")
    eff = pd.read_csv(OUTM / "effectiveness.csv").set_index("target")
    rr = pd.read_csv(OUTM / "resid_fold_metrics.csv")
    cc = pd.read_csv(OUTM / "class_check.csv")
    imp = pd.read_csv(OUTM / "importance.csv")
    order = pd.MultiIndex.from_product([list(TGT), GROUPS], names=["target", "group"])

    head = pd.DataFrame(index=order)
    for val in ["r2", "rmse"]:
        sm = summarise(fm, ["model", "target", "group"], val)
        g = paired(fm[fm.model != "ridge"], "RF", "MLP", "model", ["target", "group"], val=val)
        head[f"RF {val}"] = sm.loc["RF"].reindex(order).m.round(3)
        head[f"MLP {val}"] = sm.loc["MLP"].reindex(order).m.round(3)
        head[f"gain {val}"] = g.reindex(order).m.map(lambda v: f"{v:+.3f}")
        head[f"real {val}"] = g.reindex(order).real.map({True: "yes", False: "no"})
    gr2 = paired(fm[fm.model != "ridge"], "RF", "MLP", "model", ["target", "group"])
    pooled = gr2.xs("pooled", level="group")

    rs = summarise(rr, ["model", "group"], "r2").m.unstack("model")[["RF", "MLP"]].reindex(GROUPS)
    rg = paired(rr, "RF", "MLP", "model", ["group"]).reindex(GROUPS)
    ccs = cc.groupby(["group", "model"])[["macro_f1", "worst_f1", "f1w"]].mean().unstack("model").reindex(GROUPS)
    cut_set = cc.cut_set.iloc[0]

    def verdict(t):
        r = pooled.loc[t]
        return ("MLP better" if r.m > 0 else "RF better") if r.real else "tie (noise)"

    read = [f"| {'Question':34s} | Answer",
            f"| {'-' * 34} | {'-' * 60}"]
    for t in TGT:
        r = pooled.loc[t]
        read.append(f"| {('MLP vs RF, ' + t):34s} | {verdict(t)}: R2 gain {r.m:+.3f} pooled (fold SD {r.fold_sd:.3f})")
    read.append(f"| {'MLP vs RF, p98 residual':34s} | R2 RF {rs.loc['pooled', 'RF']:.3f} / MLP "
                f"{rs.loc['pooled', 'MLP']:.3f}; gain {rg.loc['pooled'].m:+.3f} "
                f"({'real' if rg.loc['pooled'].real else 'noise'})")
    read.append(f"| {'Classes from px mean (' + cut_set + ')':34s} | macro F1 pooled RF "
                f"{ccs.loc['pooled', ('macro_f1', 'RF')]:.3f} / MLP {ccs.loc['pooled', ('macro_f1', 'MLP')]:.3f}")
    read.append(f"| {'MLP training effective?':34s} | beats ridge every target: "
                f"{bool((eff.r2_kfold > eff.ridge_r2).all())}; folds at max epoch: {int(eff.folds_at_max_ep.sum())}")
    if (OUTM / "map_range.csv").exists():
        z = pd.read_csv(OUTM / "map_range.csv").set_index(["target", "model"]).at_zero
        read.append(f"| {'MLP map extrapolation':34s} | px clipped to 0: mean {z['mean', 'MLP']:.1%}, p98 "
                    f"{z['p98', 'MLP']:.1%}, cover {z['cover', 'MLP']:.1%} (RF 0 %); K-fold cannot see this")

    ef = eff[["w", "drop", "r2_kfold", "seed_sd", "r2_train", "ridge_r2", "best_ep_mean", "best_ep_max",
              "folds_at_max_ep", "clipped_frac"]].copy()
    ef["train - kfold"] = ef.r2_train - ef.r2_kfold
    rf_imp = []
    for t in ["mean", "cover"]:
        ii = pd.read_csv(R.OUTD / f"imp_ALL_{t}.csv")
        rf_imp += [dict(model="RF", target=t, sensor=n, dR2=v) for n, v in
                   ii[ii.kind == "perm_dR2"].groupby("name").value.mean().items()]
    im = pd.concat([pd.DataFrame(rf_imp), imp.groupby(["target", "sensor"]).dR2.mean().reset_index().assign(
        model="MLP")]).set_index(["target", "model", "sensor"]).dR2.unstack("sensor").round(3)

    L = ["REPORT — MLP vs RF regression, ALL stack (mlp_regression_2026.py)",
         "Labels 39,023 px (Dampier 2026 + Yawuru kept); folds_cover_height.json; 49 features (NISAR + S1 + S2).",
         "RF: 500 trees, max_features 0.33, leaf 5, not tuned (from disk). MLP: 2 hidden layers, config chosen on the",
         "test folds (optimistic for the MLP). Values: fold mean; gain = MLP - RF per fold and seed;",
         "real if |gain| > 2 x fold SD and > seed SD. Random K-fold under spatial autocorrelation: optimistic.", "",
         "1. HEADLINE — K-fold R2 and RMSE (m; cover as fraction)", head.to_string(), "",
         "2. READING", *read, "",
         "3. MLP EFFECTIVENESS (best config per target; seeds 0-2)", ef.round(3).to_string(),
         "ridge = linear baseline on the same folds; folds_at_max_ep = fold fits that ran to 300 epochs.", "",
         "4. p98 RESIDUAL r = p98 - g(cover), g = E[p98 | cover] on the training-fold labels; skill vs LiDAR r (seed 0)",
         pd.DataFrame({"RF R2": rs.RF.round(3), "MLP R2": rs.MLP.round(3), "gain": rg.m.round(3),
                       "fold SD": rg.fold_sd.round(3), "real": rg.real}).to_string(), "",
         f"5. CLASS CHECK — px-mean preds -> classes, cut set {cut_set}; mean of 3 seeds", ccs.round(3).to_string(), "",
         "6. IMPORTANCE — permutation R2 drop by sensor (seed 0, fold mean; RF p98 has impurity only)", im.to_string()]
    mc = OUTM / "map_compare.txt"
    L += ["", "7. MAPS", mc.read_text(encoding="utf-8") if mc.exists() else "not run yet (mlp_regression_2026.py maps)"]
    mr = OUTM / "map_range.txt"
    if mr.exists():
        L += ["8. MAP RANGE (extrapolation; mlp_regression_2026.py map_range)", mr.read_text(encoding="utf-8")]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    print(f"\nwrote {REPORT}", flush=True)


# ------------------------------------------------------------------------------------------- maps

MAPS = {"mean": "height_mean", "p98": "height_p98", "cover": "cover137"}


def read_nan(path):
    import rasterio
    with rasterio.open(path) as s:
        a, prof = s.read(1), s.profile
    return np.where(a == P.NODATA, np.nan, a).astype("float32"), prof


def resid_maps(p98_tif, cov_tif):
    """Transcribed from p98_cover.maps: g = E[p98 map | cover map] (HGB, every 7th px, 1001-step lookup);
    residual + WIN x WIN window mean (>= 80 % valid px)."""
    import rasterio
    from scipy.ndimage import uniform_filter
    with rasterio.open(p98_tif) as s:
        p98, prof = s.read(1), s.profile
    with rasterio.open(cov_tif) as s:
        cov = s.read(1)
        assert s.transform == prof["transform"] and s.shape == p98.shape
    valid = (p98 != P.NODATA) & (cov != P.NODATA)
    assert valid.sum() == (p98 != P.NODATA).sum(), "p98 and cover masks differ"
    sk = valid[::7, ::7]
    g = P.hgb(cov[::7, ::7][sk][:, None], p98[::7, ::7][sk])
    gx = np.linspace(0, 1, 1001)
    gy = g.predict(gx[:, None])
    res = np.full(p98.shape, np.nan, "float32")
    res[valid] = p98[valid] - np.interp(np.clip(cov[valid], 0, 1), gx, gy)
    num = uniform_filter(np.where(valid, res, 0).astype("float64"), P.WIN, mode="constant")
    den = uniform_filter(valid.astype("float64"), P.WIN, mode="constant")
    resw = np.where(valid & (den >= 0.8), num / np.maximum(den, 1e-9), np.nan).astype("float32")
    prof.update(dtype="float32", nodata=P.NODATA, count=1)
    return res, resw, prof, gy


def maps():
    import warnings
    import regression_2026 as RG
    warnings.filterwarnings("ignore", "X does not have valid feature names")
    t0 = time.time()
    d, base, bands, RR = RG.train_table()
    d = d.merge(pd.read_parquet(P.OUTP / "rows_p98.parquet")[KEY + ["p98"]], on=KEY, how="left", validate="1:1")
    assert d.p98.notna().all()
    X = d[base].to_numpy("float64")
    L = []
    for t, (yc, lo, hi) in TGT.items():
        out, pt = RG.MAPD / f"{MAPS[t]}_MLP.tif", RG.MAPD / f"mlp_ALL_{t}.pt"
        if out.exists() and pt.exists():
            m = Reg.load(pt, len(base))
            print(f"skip {out} (exists)", flush=True)
        else:
            cf = pd.read_csv(OUTM / f"configs_{t}.csv")
            cfg = GRID[int(cf.loc[cf.r2_pooled.idxmax(), "cfg"])]
            m, h, be = fit(X, d[yc].to_numpy("float64"), cfg, 0, lo, hi)
            m.save(pt)
            print(f"[{(time.time() - t0) / 60:5.1f} min] fit {t} {cfg}: best ep {be + 1}, ran {len(h)}; predicting "
                  f"{out}", flush=True)
            RR.FS.predict_raster([m], [f"{t}_MLP"], out, path=RR.STACK_TIF, bands=bands)
        RG.map_checks(f"{t}_MLP", out, m, d, base, yc, RR)
        print(f"[{(time.time() - t0) / 60:5.1f} min] {t} map done + checked", flush=True)

    res, resw, prof, gy = resid_maps(RG.MAPD / "height_p98_MLP.tif", RG.MAPD / "cover137_MLP.tif")
    P.write(res, RG.MAPD / "p98_resid_MLP.tif", prof)
    P.write(resw, RG.MAPD / f"p98_resid_w{P.WIN}_MLP.tif", prof)
    L.append(f"MLP g(cover) on the map: g(0) {gy[0]:.2f}, g(0.25) {gy[250]:.2f}, g(0.5) {gy[500]:.2f}, "
             f"g(0.75) {gy[750]:.2f}, g(1) {gy[-1]:.2f} m")
    del res, resw

    # RF vs MLP map agreement + 8 x 8 block-shuffle null (20 reps)
    q = [1, 5, 25, 50, 75, 95, 99]
    rec = []
    prods = {"mean": "height_mean", "p98": "height_p98", "cover": "cover137", "resid": "p98_resid",
             f"resid_w{P.WIN}": f"p98_resid_w{P.WIN}"}
    rng = np.random.default_rng(0)
    for name, stem in prods.items():
        a, _ = read_nan(RG.MAPD / f"{stem}_ALL.tif")
        b, _ = read_nan(RG.MAPD / f"{stem}_MLP.tif")
        va, vb = np.isfinite(a), np.isfinite(b)
        if name in MAPS:
            assert (va == vb).all(), f"{name}: RF and MLP valid masks differ"
        m = va & vb
        o = RG.map_metrics(a, b, m)
        nr = [RG.map_metrics(a, RG.block_shuffle(b, rng), m)["r"] for _ in range(20)]
        rec.append(dict(product=name, n=o["n"], r=o["r"], r_null=np.mean(nr), r_null_sd=np.std(nr),
                        lift=o["r"] - np.mean(nr), rmsd=o["rmsd"], mean_mlp_minus_rf=-o["mean_diff"],
                        **{f"RF p{x}": v for x, v in zip(q, np.nanpercentile(a, q))},
                        **{f"MLP p{x}": v for x, v in zip(q, np.nanpercentile(b, q))}))
        print(f"[{(time.time() - t0) / 60:5.1f} min] compare {name}: r {o['r']:.3f}, null {np.mean(nr):.3f}", flush=True)
        del a, b
    mc = pd.DataFrame(rec).set_index("product")
    mc.to_csv(OUTM / "map_compare.csv")
    L = [f"Maps: {RG.MAPD}\\{{height_mean,height_p98,cover137,p98_resid,p98_resid_w{P.WIN}}}_MLP.tif",
         "Checks per map (outputs/regression/map_checks_<t>_MLP.txt): map = model at the label px, nodata -9999,",
         "no NaN, none outside the mainland; valid mask = RF map mask (asserted).", *L, "",
         "RF vs MLP map agreement (all valid px); null = 8 x 8 px block shuffle of the MLP map, 20 reps",
         mc[["n", "r", "r_null", "r_null_sd", "lift", "rmsd", "mean_mlp_minus_rf"]].round(3).to_string(), "",
         "percentiles, RF vs MLP map", mc[[c for c in mc.columns if " p" in c]].round(2).T.to_string()]
    (OUTM / "map_compare.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    report()
    print(f"maps done {(time.time() - t0) / 60:.1f} min", flush=True)


def map_range():
    """Map px outside the label range: clipped to 0 (MLP raw < 0) and above the label max. K-fold cannot see these."""
    import regression_2026 as RG
    d, _ = load()
    rec = []
    for t, (yc, lo, hi) in TGT.items():
        for m in ["ALL", "MLP"]:
            a, _ = read_nan(RG.MAPD / f"{MAPS[t]}_{m}.tif")
            v = a[np.isfinite(a)]
            rec.append(dict(target=t, model="RF" if m == "ALL" else "MLP", at_zero=np.mean(v == 0),
                            below_label_min=np.mean(v < d[yc].min()), above_label_max=np.mean(v > d[yc].max()),
                            map_max=v.max(), label_max=d[yc].max()))
    r = pd.DataFrame(rec).set_index(["target", "model"])
    r.to_csv(OUTM / "map_range.csv")
    txt = ("Map px outside the label range (fraction of valid px); MLP clips raw < 0 to 0 (divergence 3)\n"
           + r.round(4).to_string())
    (OUTM / "map_range.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt, flush=True)
    report()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["kfold", "maps", "map_range", "report"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=10, help="CPU threads (cap 10)")
    a = ap.parse_args()
    torch.set_num_threads(min(a.threads, 10))
    print(f"device {DEV}", flush=True)
    if a.quick:
        return quick()
    {"kfold": kfold, "maps": maps, "map_range": map_range, "report": report}[a.step]()


if __name__ == "__main__":
    main()
