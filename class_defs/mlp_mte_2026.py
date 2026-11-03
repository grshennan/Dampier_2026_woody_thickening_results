"""Regression track, follow-on E (PLAN_mlp_mte.md, user 2026-11-03): multi-task bounded MLP, 5-seed ensemble (MTE).

One net, 3 bounded outputs (px mean, p98, cover 1.37), inputs clamped to the training range, 5 members.
Rows, folds, RF comparator and helpers from mlp_regression_2026.py (imported as M; not edited).

Model: trunk = M.net with the last layer Linear(h2, 3) -> z. Heads (s_t = training SD):
  mean = s_mean * softplus(z0); p98 = mean + s_d * softplus(z1) (head 'delta', d = p98 - mean) or s_p98 * softplus(z1)
  ('indep'); cover = sigmoid(z2). Last-layer bias set so the start output = training means.
  Loss: sum over targets of mean(((yhat - y) / s_t)^2); early stop on the same sum (15 % split, mean deciles).
  Training as M.fit (AdamW 1e-3, wd 1e-4, batch 512, ReduceLROnPlateau 0.5 / 8, patience 20, max 300, best epoch).
Divergences from M:
  1. Input clamp: per-feature p0.5 / p99.5 of the X given to fit (the whole training fold: all members share them),
     before scaling, in train and predict. clamp=False for the ablation.
  2. No output clipping (heads are bounded by construction).
  3. Ensemble = mean of 5 members (seeds 0-4); member SD (ddof 1) = uncertainty.

  kfold    -> outputs/mlp_mte/*.csv, pred_MTE_<t>.parquet; REPORT_mlp_mte.txt
  maps     5 members on all 39,023 px -> E:/nisar_2026/regression/mte_ALL_s<k>.pt + 7 rasters (*_MTE.tif, *_MTEsd.tif,
           n_clamped_MTE.tif), residual maps, map agreement + map range
  --quick  fold 0, config 0, 2 members, 30 epochs; asserts; 1 M px tile time with 5 members. Writes nothing.

    C:\\Users\\glens\\miniconda3\\Scripts\\conda.exe run -n pytorch --no-capture-output python -u mlp_mte_2026.py kfold --quick
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import argparse
import contextlib
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_recall_fscore_support, r2_score
from sklearn.model_selection import train_test_split

import common as C
import cover_height_rf as R
import mlp_regression_2026 as M
import p98_cover as P
from cover_height_report import paired, reg, summarise

OUTE = C.OUT / "mlp_mte"
REPORT = C.HERE / "REPORT_mlp_mte.txt"
KEY, GROUPS, CV, DEV = M.KEY, M.GROUPS, M.CV, M.DEV
T3 = ["mean", "p98", "cover"]
YC = ["mean", "p98", CV]
GRID = [dict(w=w, drop=0.2, wd=1e-4, head=h) for w in [(512, 256), (1024, 512)] for h in ["delta", "indep"]]
MSEEDS = [0, 1, 2, 3, 4]
MAPS = {"mean": "height_mean", "p98": "height_p98", "cover": "cover137"}
B = 1 << 20


# ------------------------------------------------------------------------------------------- model

class MTNet(torch.nn.Module):
    def __init__(self, n_in, w, drop, head, hs):
        super().__init__()
        self.f = M.net(n_in, w, drop)
        self.f[-1] = torch.nn.Linear(w[1], 3)
        self.delta = head == "delta"
        self.register_buffer("hs", torch.as_tensor(hs, dtype=torch.float32))

    def forward(self, x):
        z = self.f(x)
        m = self.hs[0] * F.softplus(z[:, 0])
        p = (m if self.delta else 0) + self.hs[1] * F.softplus(z[:, 1])
        return torch.stack([m, p, torch.sigmoid(z[:, 2])], 1)


def _t(a):
    return torch.as_tensor(np.asarray(a, dtype="float64"), device=DEV)


class MT:
    """Fitted multi-task net (float64 copy on DEV). raw(X) -> (n, 3) physical units."""

    def __init__(self, net, mu, sd, qlo, qhi, clamp, cfg, seed):
        self.m = copy.deepcopy(net).double().to(DEV).eval()
        self.mu, self.sd, self.qlo, self.qhi = _t(mu), _t(sd), _t(qlo), _t(qhi)
        self.clamp, self.cfg, self.seed = clamp, cfg, seed

    def fwd(self, xb):
        if self.clamp:
            xb = torch.minimum(torch.maximum(xb, self.qlo), self.qhi)
        return self.m((xb - self.mu) / self.sd)

    def raw(self, X):
        X = np.asarray(X, dtype="float64")
        out = np.empty((len(X), 3))
        with torch.no_grad():
            for i in range(0, len(X), B):
                out[i:i + B] = self.fwd(_t(X[i:i + B])).cpu().numpy()
        return out

    predict3 = raw

    def n_clamped(self, X):
        X = np.asarray(X, dtype="float64")
        out = np.empty(len(X), "int16")
        with torch.no_grad():
            for i in range(0, len(X), B):
                xb = _t(X[i:i + B])
                out[i:i + B] = ((xb < self.qlo) | (xb > self.qhi)).sum(1).cpu().numpy()
        return out

    def save(self, path):
        torch.save(dict(state=self.m.state_dict(), mu=self.mu.cpu().numpy(), sd=self.sd.cpu().numpy(),
                        qlo=self.qlo.cpu().numpy(), qhi=self.qhi.cpu().numpy(), clamp=self.clamp, cfg=self.cfg,
                        seed=self.seed), path)

    @staticmethod
    def load(path, n_in):
        z = torch.load(path, weights_only=False)
        net = MTNet(n_in, z["cfg"]["w"], z["cfg"]["drop"], z["cfg"]["head"], np.ones(3)).double()
        net.load_state_dict(z["state"])
        return MT(net, z["mu"], z["sd"], z["qlo"], z["qhi"], z["clamp"], z["cfg"], z["seed"])


class Ens:
    """Mean of members; member SD (ddof 1). One GPU transfer per batch."""

    def __init__(self, members):
        self.members = members
        q = members[0]
        for m in members[1:]:
            assert torch.equal(m.qlo, q.qlo) and torch.equal(m.qhi, q.qhi), "members do not share the clamp range"

    def predict3(self, X):
        X = np.asarray(X, dtype="float64")
        mu, sd = np.empty((len(X), 3)), np.empty((len(X), 3))
        with torch.no_grad():
            for i in range(0, len(X), B):
                xb = _t(X[i:i + B])
                P_ = torch.stack([m.fwd(xb) for m in self.members])
                mu[i:i + B] = P_.mean(0).cpu().numpy()
                sd[i:i + B] = P_.std(0).cpu().numpy() if len(self.members) > 1 else 0.0
        return mu, sd

    def n_clamped(self, X):
        return self.members[0].n_clamped(X)


class View:
    """Column j of a 3-output model, sklearn-like .predict (for map_checks, perm_sensor)."""

    def __init__(self, m, j):
        self.m, self.j = m, j

    def predict(self, X):
        r = self.m.predict3(X)
        return (r[0] if isinstance(r, tuple) else r)[:, self.j]


def inv_softplus(x):
    return float(np.log(np.expm1(x)))


def fit(X, Y, cfg, seed, clamp=True, max_ep=M.MAX_EP):
    torch.manual_seed(seed)
    strat = pd.qcut(Y[:, 0], 10, labels=False, duplicates="drop")
    tr, va = train_test_split(np.arange(len(Y)), test_size=0.15, stratify=strat, random_state=seed)
    qlo, qhi = np.percentile(X, [0.5, 99.5], axis=0)
    Xc = np.clip(X, qlo, qhi) if clamp else X
    mu, sd = Xc[tr].mean(0), Xc[tr].std(0) + 1e-9
    st = Y[tr].std(0)
    delta = cfg["head"] == "delta"
    y1 = Y[tr, 1] - Y[tr, 0] if delta else Y[tr, 1]
    assert y1.mean() > 0
    hs = [st[0], y1.std(), 1.0]
    net = MTNet(X.shape[1], cfg["w"], cfg["drop"], cfg["head"], hs).to(DEV)
    cb = Y[tr, 2].mean()
    with torch.no_grad():
        net.f[-1].bias.copy_(torch.tensor([inv_softplus(Y[tr, 0].mean() / st[0]), inv_softplus(y1.mean() / hs[1]),
                                           float(np.log(cb / (1 - cb)))]))
    Xt = torch.tensor((Xc - mu) / sd, dtype=torch.float32, device=DEV)
    Yt = torch.tensor(Y, dtype=torch.float32, device=DEV)
    w = torch.tensor(1.0 / st ** 2, dtype=torch.float32, device=DEV)

    def loss_fn(p, y):
        return (((p - y) ** 2).mean(0) * w).sum()

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    tr_t, va_t = torch.tensor(tr, device=DEV), torch.tensor(va, device=DEV)
    best, best_state, best_ep, wait, hist = np.inf, None, -1, 0, []
    g = torch.Generator(device=DEV).manual_seed(seed)
    for ep in range(max_ep):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t), device=DEV, generator=g)]
        tl = 0.0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
            tl += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xt[va_t]), Yt[va_t]).item()
        sch.step(vl)
        hist.append((ep, tl / len(tr), vl, opt.param_groups[0]["lr"]))
        if vl < best - 1e-4:
            best, best_ep, wait = vl, ep, 0
            best_state = {n: t.detach().clone() for n, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= M.PATIENCE:
                break
    net.load_state_dict(best_state)
    return MT(net, mu, sd, qlo, qhi, clamp, cfg, seed), hist, best_ep


def kfold_mt(X, Y, fold, cfg, seed, clamp=True, folds=range(5), max_ep=M.MAX_EP):
    oof, trr2, eps, ran, models, hist = np.full(Y.shape, np.nan), [], [], [], {}, []
    for k in folds:
        trn, te = fold != k, fold == k
        m, h, be = fit(X[trn], Y[trn], cfg, seed, clamp, max_ep)
        oof[te] = m.raw(X[te])
        ptr = m.raw(X[trn])
        trr2.append([r2_score(Y[trn, j], ptr[:, j]) for j in range(3)])
        eps.append(be + 1)
        ran.append(len(h))
        models[k] = m
        hist += [dict(fold=k, epoch=e, train_loss=a, val_loss=b, lr=c) for e, a, b, c in h]
    return dict(oof=oof, train_r2=np.mean(trr2, 0), best_ep=eps, ran=ran, models=models, hist=hist)


def score3(Y, P_):
    return [r2_score(Y[:, j], P_[:, j]) for j in range(3)]


# ------------------------------------------------------------------------------------------- data

def load():
    d, base = M.load()
    Y = d[YC].to_numpy("float64")
    return d, base, d[base].to_numpy("float64"), Y, d.fold.to_numpy()


def old_mlp(d):
    out = {}
    for t in T3:
        for s in M.SEEDS:
            q = pd.read_parquet(M.OUTM / f"pred_MLP_{t}_s{s}.parquet")
            assert (q[KEY + ["fold"]].to_numpy() == d[KEY + ["fold"]].to_numpy()).all(), "MLP row order / folds differ"
            out[(t, s)] = q.pred.to_numpy()
    return out


def label_check(d):
    f = float((d.p98 < d["mean"]).mean())
    print(f"label px with p98 < mean: {f:.3%} (delta head dropped if > 0.5 %)", flush=True)
    return f


# ------------------------------------------------------------------------------------------- quick

def quick():
    import regression_2026 as RG
    d, base, X, Y, fold = load()
    rf, k = M.rf_preds(d)
    label_check(d)
    te = fold == 0
    cfg = GRID[0]
    runs = []
    for s in [0, 1]:
        t0 = time.time()
        r = kfold_mt(X, Y, fold, cfg, s, folds=[0], max_ep=30)
        runs.append(r)
        print(f"seed {s}: fold 0 R2 " + ", ".join(f"{t} {v:.3f}" for t, v in zip(T3, score3(Y[te], r["oof"][te])))
              + f" | RF " + ", ".join(f"{t} {r2_score(Y[te, j], rf[(t, 0)][te]):.3f}" for j, t in enumerate(T3))
              + f" | best ep {r['best_ep']} | {time.time() - t0:.0f} s", flush=True)
    p = runs[0]["oof"][te]
    assert (p[:, :2] > 0).all() and ((p[:, 2] > 0) & (p[:, 2] < 1)).all(), "outputs outside bounds"
    assert (p[:, 1] >= p[:, 0]).all(), "delta head: p98 < mean"
    assert not np.allclose(p, runs[1]["oof"][te]), "member seeds give the same preds"
    e = Ens([runs[0]["models"][0], runs[1]["models"][0]])
    mu, sd = e.predict3(X[te])
    assert np.allclose(mu, (runs[0]["oof"][te] + runs[1]["oof"][te]) / 2, atol=1e-12), "ensemble != member mean"
    print(f"bounds, p98 >= mean, seeds differ, ensemble = member mean: ok; test rows clamped "
          f"{(e.n_clamped(X[te]) > 0).mean():.1%}; median member SD " + ", ".join(
              f"{t} {v:.3f}" for t, v in zip(T3, np.median(sd, 0))), flush=True)
    g_rf, _ = M.g_fold(d, k, k.cover_hat.to_numpy())
    err = float(np.abs(g_rf - k.g_hat).max())
    assert err < 1e-3, err
    print(f"g reproduction: max |d| {err:.2e}", flush=True)
    d2, base2, bands, RR = RG.train_table()
    assert base2 == base
    m0, m1 = runs[0]["models"][0], runs[1]["models"][0]
    ens5 = Ens([m0, m1, m0, m1, m0])
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(RR.STACK_TIF) as s:
        h, w = s.height, s.width
        a = s.read(bands, window=Window(0, h // 2 - 512, w, 1024)).reshape(len(bands), -1).T
    Xt = a[np.isfinite(a).all(1)][:1_000_000]
    t1 = time.time()
    ens5.predict3(Xt)
    ncl = ens5.n_clamped(Xt)
    dt = time.time() - t1
    assert (ncl > 0).any(), "clamp not active on the tile"
    n_valid = 64_392_072
    print(f"tile: {len(Xt):,} px, 5 members (cfg {cfg['w']}) {dt:.1f} s; rows clamped {(ncl > 0).mean():.1%}; "
          f"about {dt * n_valid / len(Xt) / 60:.0f} min per map pass (x 2 for 1024-512)", flush=True)
    print("quick passed", flush=True)


# ------------------------------------------------------------------------------------------- kfold

def fm3(model, seed, d, Y, P_):
    return sum((M.fm_rows(model, t, seed, d, Y[:, j], P_[:, j]) for j, t in enumerate(T3)), [])


def kfold():
    t0 = time.time()
    OUTE.mkdir(exist_ok=True)
    d, base, X, Y, fold = load()
    rf, k = M.rf_preds(d)
    mlp = old_mlp(d)
    frac_lab = label_check(d)
    grid = [c for c in GRID if not (c["head"] == "delta" and frac_lab > 0.005)]

    # 1. grid, seed 0
    rows, res = [], {}
    for i, cfg in enumerate(grid):
        t1 = time.time()
        r = kfold_mt(X, Y, fold, cfg, 0)
        res[i] = r
        r2 = score3(Y, r["oof"])
        rows.append(dict(cfg=i, w=str(cfg["w"]), drop=cfg["drop"], head=cfg["head"],
                         **{f"r2_{t}": v for t, v in zip(T3, r2)}, r2_avg=np.mean(r2),
                         best_ep_mean=np.mean(r["best_ep"]), ran_max=max(r["ran"]), s=time.time() - t1))
        print(f"[{(time.time() - t0) / 60:5.1f} min] cfg {i} {cfg}: pooled OOF R2 {np.round(r2, 4)}, avg "
              f"{np.mean(r2):.4f}, best ep {r['best_ep']}", flush=True)
    cf = pd.DataFrame(rows)
    cf.to_csv(OUTE / "configs.csv", index=False)
    b = int(cf.loc[cf.r2_avg.idxmax(), "cfg"])
    cfg = grid[b]
    json.dump(dict(best=b, cfg=cfg), open(OUTE / "best_config.json", "w"))
    print(f"best cfg {b} {cfg}", flush=True)

    # 2. members
    mem = {0: res[b]}
    for s in MSEEDS[1:]:
        mem[s] = kfold_mt(X, Y, fold, cfg, s)
        print(f"[{(time.time() - t0) / 60:5.1f} min] member {s}: R2 {np.round(score3(Y, mem[s]['oof']), 4)}",
              flush=True)
    pd.DataFrame(mem[0]["hist"]).to_csv(OUTE / "history.csv", index=False)
    P_ = np.stack([mem[s]["oof"] for s in MSEEDS])
    ens_oof, ens_sd = P_.mean(0), P_.std(0, ddof=1)
    for j, t in enumerate(T3):
        d[KEY + ["group", "fold"]].assign(y=Y[:, j], pred=ens_oof[:, j], sd=ens_sd[:, j]).to_parquet(
            OUTE / f"pred_MTE_{t}.parquet")
        for s in MSEEDS:
            d[KEY + ["group", "fold"]].assign(y=Y[:, j], pred=P_[s][:, j]).to_parquet(
                OUTE / f"pred_MTE_member_{t}_s{s}.parquet")
    ens_f = {kk: Ens([mem[s]["models"][kk] for s in MSEEDS]) for kk in range(5)}
    chk = np.full(Y.shape, np.nan)
    for kk, e in ens_f.items():
        chk[fold == kk] = e.predict3(X[fold == kk])[0]
    assert np.allclose(chk, ens_oof, atol=1e-10), "Ens != member mean"

    # 3. ablation
    nc = kfold_mt(X, Y, fold, cfg, 0, clamp=False)
    for j, t in enumerate(T3):
        d[KEY + ["group", "fold"]].assign(y=Y[:, j], pred=nc["oof"][:, j]).to_parquet(
            OUTE / f"pred_MTnoclamp_{t}.parquet")
    print(f"[{(time.time() - t0) / 60:5.1f} min] no-clamp ablation: R2 {np.round(score3(Y, nc['oof']), 4)}",
          flush=True)

    # 4. metrics
    fm = []
    for s in M.SEEDS:
        fm += fm3("RF", s, d, Y, np.stack([rf[(t, s)] for t in T3], 1))
        fm += fm3("MLP", s, d, Y, np.stack([mlp[(t, s)] for t in T3], 1))
    fm += fm3("MT1", 0, d, Y, mem[0]["oof"]) + fm3("MTnoclamp", 0, d, Y, nc["oof"]) + fm3("MTE", 0, d, Y, ens_oof)
    for s in MSEEDS:
        fm += fm3("member", s, d, Y, P_[s])
    fm = pd.DataFrame(fm)
    fm.to_csv(OUTE / "fold_metrics.csv", index=False)
    srf = summarise(fm[(fm.model == "RF") & (fm.group == "pooled")], ["target"], "r2")
    for t, v in M.RF_R2.items():
        assert abs(srf.loc[t].m - v) < 0.0006, f"RF {t} pooled R2 {srf.loc[t].m:.4f} != {v}"
    print("RF pooled K-fold R2 reproduced", flush=True)

    # 6. residual
    g_rf, g_e = M.g_fold(d, k, ens_oof[:, 2])
    err = float(np.abs(g_rf - k.g_hat).max())
    assert err < 1e-3, f"g not reproduced: max |d| {err}"
    _, g_m = M.g_fold(d, k, mlp[("cover", 0)])
    rt = k.res_true.to_numpy()
    rr = M.fm_rows("RF", "resid", 0, d, rt, k.res_direct.to_numpy())
    rr += M.fm_rows("MLP", "resid", 0, d, rt, mlp[("p98", 0)] - g_m)
    rr += M.fm_rows("MTE", "resid", 0, d, rt, ens_oof[:, 1] - g_e)
    for s in MSEEDS:
        _, g_s = M.g_fold(d, k, P_[s][:, 2])
        rr += M.fm_rows("member", "resid", s, d, rt, P_[s][:, 1] - g_s)
    pd.DataFrame(rr).to_csv(OUTE / "resid_fold_metrics.csv", index=False)

    # 7. class check
    name, cs = next(iter(json.load(open(C.OUT / "cut_sets.json")).items()))
    yt = np.digitize(d["mean"], cs["cuts"])
    cc = []
    src = [("RF", s, rf[("mean", s)]) for s in M.SEEDS] + [("MLP", s, mlp[("mean", s)]) for s in M.SEEDS]
    for model, s, pm in src + [("MTE", 0, ens_oof[:, 0])]:
        yp = np.digitize(pm, cs["cuts"])
        for g in GROUPS:
            i = np.ones(len(d), bool) if g == "pooled" else (d.group == g).to_numpy()
            Fs = precision_recall_fscore_support(yt[i], yp[i], labels=range(len(cs["cuts"]) + 1), zero_division=0)[2]
            cc.append(dict(model=model, seed=s, group=g, cut_set=name, macro_f1=Fs.mean(), worst_f1=Fs.min(),
                           f1w=f1_score(yt[i], yp[i], average="weighted", zero_division=0)))
    pd.DataFrame(cc).to_csv(OUTE / "class_check.csv", index=False)

    # 8. importance
    grp = {g: [base.index(c) for c in cols] for g, cols in R.groups(base).items()}
    imp = []
    for j, t in enumerate(T3):
        rng = np.random.default_rng(0)
        imp += [dict(target=t, **x) for x in M.perm_sensor([View(ens_f[kk], j) for kk in range(5)], X, Y[:, j],
                                                           fold, grp, rng)]
    pd.DataFrame(imp).to_csv(OUTE / "importance.csv", index=False)
    print(f"[{(time.time() - t0) / 60:5.1f} min] importance done", flush=True)

    # 9. effectiveness
    ridge = pd.read_csv(M.OUTM / "fold_metrics.csv")
    ridge = ridge[(ridge.model == "ridge") & (ridge.group == "pooled")].groupby("target").r2.mean()
    ncl_te = np.zeros(len(d), int)
    for kk in range(5):
        ncl_te[fold == kk] = ens_f[kk].n_clamped(X[fold == kk])
    rf_m, rf_p = rf[("mean", 0)], rf[("p98", 0)]
    eff = []
    for model, runs in (("MT1", [mem[0]]), ("MTnoclamp", [nc]), ("MTE members", list(mem.values()))):
        eps = sum((r["best_ep"] for r in runs), [])
        ran = sum((r["ran"] for r in runs), [])
        for j, t in enumerate(T3):
            kf = [r2_score(Y[:, j], r["oof"][:, j]) for r in runs]
            trr = float(np.mean([r["train_r2"][j] for r in runs]))
            eff.append(dict(model=model, target=t, r2_kfold=np.mean(kf), member_sd=np.std(kf, ddof=1) if len(kf) > 1
                            else 0.0, r2_train=trr, train_minus_kfold=trr - np.mean(kf), ridge_r2=ridge[t],
                            best_ep_mean=np.mean(eps), best_ep_max=max(eps), fits=len(ran),
                            fits_at_max_ep=int(sum(n >= M.MAX_EP for n in ran))))
    pd.DataFrame(eff).to_csv(OUTE / "effectiveness.csv", index=False)
    e_r2 = score3(Y, ens_oof)
    misc = dict(label_p98_lt_mean=frac_lab, oof_p98_lt_mean_RF=float((rf_p < rf_m).mean()),
                oof_p98_lt_mean_MLP=float((mlp[("p98", 0)] < mlp[("mean", 0)]).mean()),
                oof_p98_lt_mean_MTE=float((ens_oof[:, 1] < ens_oof[:, 0]).mean()),
                test_rows_clamped=float((ncl_te > 0).mean()), test_feats_clamped_mean=float(ncl_te.mean()),
                **{f"MTE_r2_{t}": v for t, v in zip(T3, e_r2)},
                **{f"MTE_beats_ridge_{t}": bool(v > ridge[t]) for t, v in zip(T3, e_r2)})
    json.dump(misc, open(OUTE / "misc.json", "w"), indent=1)

    # 10. uncertainty check
    sk = []
    for j, t in enumerate(T3):
        err_ = np.abs(ens_oof[:, j] - Y[:, j])
        rho = spearmanr(ens_sd[:, j], err_)[0]
        q = pd.qcut(ens_sd[:, j], 5, labels=False, duplicates="drop")
        sk.append(dict(target=t, spearman=rho, sd_median=np.median(ens_sd[:, j]),
                       **{f"abs_err_sdq{i + 1}": err_[q == i].mean() for i in range(int(q.max()) + 1)}))
    pd.DataFrame(sk).to_csv(OUTE / "sd_skill.csv", index=False)
    report()
    print(f"kfold done {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------------------------------- maps

def predict_multi(ens, bands, RR, outs, block_rows=512):
    """FS.predict_raster + predict_year.predict_stack, multi-output: a px is predicted only where every feature is
    finite; one ensemble pass per block -> 7 single-band rasters; *.partial.tif then rename; nodata -9999."""
    import rasterio
    from rasterio.windows import Window
    t0 = time.time()
    with rasterio.open(RR.STACK_TIF) as s, contextlib.ExitStack() as es:
        prof = dict(driver="GTiff", width=s.width, height=s.height, count=1, dtype="float32", crs=s.crs,
                    transform=s.transform, nodata=P.NODATA, tiled=True, compress="deflate", BIGTIFF="IF_SAFER")
        part = {k: v.with_name(v.stem + ".partial.tif") for k, v in outs.items()}
        dst = {k: es.enter_context(rasterio.open(p, "w", **(prof | ({"dtype": "int16"} if k == "ncl" else {}))))
               for k, p in part.items()}
        nb = (s.height + block_rows - 1) // block_rows
        for bi, r0 in enumerate(range(0, s.height, block_rows)):
            n = min(block_rows, s.height - r0)
            win = Window(0, r0, s.width, n)
            X = s.read(bands, window=win).reshape(len(bands), -1).T
            good = np.isfinite(X).all(1)
            vals = {}
            if good.any():
                mu, sd = ens.predict3(X[good])
                vals = {**{t: mu[:, j] for j, t in enumerate(T3)}, **{t + "sd": sd[:, j] for j, t in enumerate(T3)},
                        "ncl": ens.n_clamped(X[good])}
            for kk, ds in dst.items():
                flat = np.full(X.shape[0], P.NODATA, ds.dtypes[0])
                if good.any():
                    flat[good] = vals[kk].astype(ds.dtypes[0])
                ds.write(flat.reshape(n, -1), 1, window=win)
            if bi % 10 == 0:
                print(f"[predict {(time.time() - t0) / 60:5.1f} min] block {bi + 1}/{nb}, {good.sum():,} px",
                      flush=True)
    for k, p in part.items():
        p.replace(outs[k])
        print(f"written: {outs[k]}", flush=True)


def maps():
    import warnings
    import regression_2026 as RG
    warnings.filterwarnings("ignore", "X does not have valid feature names")
    t0 = time.time()
    d, base, bands, RR = RG.train_table()
    d = d.merge(pd.read_parquet(P.OUTP / "rows_p98.parquet")[KEY + ["p98"]], on=KEY, how="left", validate="1:1")
    assert d.p98.notna().all()
    X, Y = d[base].to_numpy("float64"), d[YC].to_numpy("float64")
    cfg = json.load(open(OUTE / "best_config.json"))["cfg"]
    cfg["w"] = tuple(cfg["w"])
    pts = [RG.MAPD / f"mte_ALL_s{s}.pt" for s in MSEEDS]
    if all(p.exists() for p in pts):
        members = [MT.load(p, len(base)) for p in pts]
        print("members loaded (exist)", flush=True)
    else:
        members = []
        for s, p in zip(MSEEDS, pts):
            m, h, be = fit(X, Y, cfg, s)
            m.save(p)
            members.append(m)
            print(f"[{(time.time() - t0) / 60:5.1f} min] member {s} {cfg}: best ep {be + 1}, ran {len(h)} -> {p}",
                  flush=True)
    ens = Ens(members)
    outs = {**{t: RG.MAPD / f"{MAPS[t]}_MTE.tif" for t in T3}, **{t + "sd": RG.MAPD / f"{MAPS[t]}_MTEsd.tif" for t in T3},
            "ncl": RG.MAPD / "n_clamped_MTE.tif"}
    if all(p.exists() for p in outs.values()):
        print("skip rasters (exist)", flush=True)
    else:
        predict_multi(ens, bands, RR, outs)
    print(f"[{(time.time() - t0) / 60:5.1f} min] rasters done; checks", flush=True)
    rf_valid = M.read_nan(RG.MAPD / "height_mean_ALL.tif")[0]
    rf_valid = np.isfinite(rf_valid)
    assert rf_valid.sum() == 64_392_072, rf_valid.sum()
    for j, t in enumerate(T3):
        RG.map_checks(f"{t}_MTE", outs[t], View(ens, j), d, base, YC[j], RR)
        for kk in (t, t + "sd"):
            a = M.read_nan(outs[kk])[0]
            assert (np.isfinite(a) == rf_valid).all(), f"{kk}: valid mask != RF mask"
            del a
    import rasterio
    with rasterio.open(outs["ncl"]) as s:
        assert ((s.read(1) != P.NODATA) == rf_valid).all(), "n_clamped mask != RF mask"
    print(f"[{(time.time() - t0) / 60:5.1f} min] map checks passed; valid mask = RF mask (64,392,072 px)", flush=True)

    res, resw, prof, gy = M.resid_maps(outs["p98"], outs["cover"])
    P.write(res, RG.MAPD / "p98_resid_MTE.tif", prof)
    P.write(resw, RG.MAPD / f"p98_resid_w{P.WIN}_MTE.tif", prof)
    gline = (f"MTE g(cover) on the map: g(0) {gy[0]:.2f}, g(0.25) {gy[250]:.2f}, g(0.5) {gy[500]:.2f}, "
             f"g(0.75) {gy[750]:.2f}, g(1) {gy[-1]:.2f} m")
    del res, resw

    prods = {"mean": "height_mean", "p98": "height_p98", "cover": "cover137", "resid": "p98_resid",
             f"resid_w{P.WIN}": f"p98_resid_w{P.WIN}"}
    rec = []
    for ref in ["ALL", "MLP"]:
        rng = np.random.default_rng(0)
        for name, stem in prods.items():
            a, _ = M.read_nan(RG.MAPD / f"{stem}_{ref}.tif")
            b, _ = M.read_nan(RG.MAPD / f"{stem}_MTE.tif")
            m = np.isfinite(a) & np.isfinite(b)
            o = RG.map_metrics(a, b, m)
            nr = [RG.map_metrics(a, RG.block_shuffle(b, rng), m)["r"] for _ in range(20)]
            rec.append(dict(vs="RF" if ref == "ALL" else "MLP", product=name, n=o["n"], r=o["r"], r_null=np.mean(nr),
                            r_null_sd=np.std(nr), lift=o["r"] - np.mean(nr), rmsd=o["rmsd"],
                            mean_mte_minus_ref=-o["mean_diff"]))
            print(f"[{(time.time() - t0) / 60:5.1f} min] MTE vs {ref} {name}: r {o['r']:.3f}, null {np.mean(nr):.3f}",
                  flush=True)
            del a, b
    mc = pd.DataFrame(rec).set_index(["vs", "product"])
    mc.to_csv(OUTE / "map_compare.csv")
    L = [f"Maps: {RG.MAPD}\\{{height_mean,height_p98,cover137}}_MTE.tif, *_MTEsd.tif, n_clamped_MTE.tif, "
         f"p98_resid_MTE.tif, p98_resid_w{P.WIN}_MTE.tif",
         "Checks (outputs/regression/map_checks_<t>_MTE.txt): map = ensemble at the label px (atol 1e-5), nodata -9999,",
         "no NaN, none outside the mainland; valid mask = RF map mask, 64,392,072 px (asserted, all 7 rasters).",
         gline, "", "MTE vs RF and MTE vs MLP map agreement (all valid px); null = 8 x 8 px block shuffle of the MTE "
                    "map, 20 reps", mc.round(3).to_string()]
    (OUTE / "map_compare.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    map_range()
    print(f"maps done {(time.time() - t0) / 60:.1f} min", flush=True)


def map_range():
    import rasterio
    import regression_2026 as RG
    d, _ = M.load()
    rec, prs = [], []
    stems = {"RF": "ALL", "MLP": "MLP", "MTE": "MTE"}
    for model, sfx in stems.items():
        mp = {}
        for t, yc in zip(T3, YC):
            a, _ = M.read_nan(RG.MAPD / f"{MAPS[t]}_{sfx}.tif")
            v = a[np.isfinite(a)]
            rec.append(dict(target=t, model=model, at_zero=np.mean(v == 0), lt_005=np.mean(v < 0.05),
                            label_lt_005=np.mean(d[yc] < 0.05), below_label_min=np.mean(v < d[yc].min()),
                            above_label_max=np.mean(v > d[yc].max()), map_max=v.max(), label_max=d[yc].max()))
            if t != "cover":
                mp[t] = a
            del a, v
        ok = np.isfinite(mp["mean"]) & np.isfinite(mp["p98"])
        prs.append(dict(model=model, map_p98_lt_mean=float((mp["p98"][ok] < mp["mean"][ok]).mean())))
        del mp
    r = pd.DataFrame(rec).set_index(["target", "model"])
    r.to_csv(OUTE / "map_range.csv")
    pr = pd.DataFrame(prs).set_index("model")
    with rasterio.open(RG.MAPD / "n_clamped_MTE.tif") as s:
        ncl = s.read(1)
    valid = ncl != P.NODATA
    nv = ncl[valid]
    bins = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 49)]
    nd = pd.Series({f"{a}-{b}" if a != b else str(a): np.mean((nv >= a) & (nv <= b)) for a, b in bins},
                   name="frac of valid px")
    q = [1, 5, 25, 50, 75, 95, 99]
    sdp = {}
    for t in T3:
        a, _ = M.read_nan(RG.MAPD / f"{MAPS[t]}_MTEsd.tif")
        sdp[t] = np.nanpercentile(a, q)
        del a
    sdp = pd.DataFrame(sdp, index=[f"p{x}" for x in q]).T
    zr = []
    for t in T3:
        old, _ = M.read_nan(RG.MAPD / f"{MAPS[t]}_MLP.tif")
        z = (old == 0) & valid
        new, _ = M.read_nan(RG.MAPD / f"{MAPS[t]}_MTE.tif")
        rfm, _ = M.read_nan(RG.MAPD / f"{MAPS[t]}_ALL.tif")
        zr.append(dict(target=t, mlp_zero_px=int(z.sum()), ncl_ge1_there=np.mean(ncl[z] >= 1),
                       ncl_ge1_all=np.mean(nv >= 1), ncl_median_there=np.median(ncl[z]),
                       **{f"MTE p{x}": v for x, v in zip([5, 50, 95], np.percentile(new[z], [5, 50, 95]))},
                       **{f"RF p{x}": v for x, v in zip([5, 50, 95], np.percentile(rfm[z], [5, 50, 95]))}))
        del old, new, rfm, z
    zr = pd.DataFrame(zr).set_index("target")
    zr.to_csv(OUTE / "mlp_zeros.csv")
    L = ["Map px outside the label range (fraction of valid px). RF and MTE have no clip; MLP clips raw < 0 to 0.",
         r.round(4).to_string(), "", "map px with p98 < mean", pr.round(4).to_string(), "",
         "n_clamped_MTE: features outside the training p0.5 / p99.5 per px", nd.round(4).to_string(), "",
         "MTE member SD percentiles (m; cover fraction)", sdp.round(3).to_string(), "",
         "Old MLP zeros (px where *_MLP.tif = 0): clamp share and MTE / RF values there", zr.round(3).to_string()]
    (OUTE / "map_range.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    report()


# ------------------------------------------------------------------------------------------- report

def gains(fm, val):
    f0 = fm[fm.seed == 0]
    msd = summarise(fm[fm.model == "member"], ["target", "group"], val).seed_sd
    out = {}
    for b in ["RF", "MLP"]:
        g = paired(f0, b, "MTE", "model", ["target", "group"], val=val)
        g["seed_sd"] = msd.reindex(g.index).to_numpy()
        g["real"] = (g.m.abs() > 2 * g.fold_sd) & (g.m.abs() > g.seed_sd)
        out[b] = g
    return out


def report():
    fm = pd.read_csv(OUTE / "fold_metrics.csv")
    cf = pd.read_csv(OUTE / "configs.csv")
    best = json.load(open(OUTE / "best_config.json"))
    eff = pd.read_csv(OUTE / "effectiveness.csv")
    misc = json.load(open(OUTE / "misc.json"))
    rr = pd.read_csv(OUTE / "resid_fold_metrics.csv")
    cc = pd.read_csv(OUTE / "class_check.csv")
    imp = pd.read_csv(OUTE / "importance.csv")
    sk = pd.read_csv(OUTE / "sd_skill.csv").set_index("target")
    order = pd.MultiIndex.from_product([T3, GROUPS], names=["target", "group"])

    head = pd.DataFrame(index=order)
    G = {}
    for val in ["r2", "rmse"]:
        sm = summarise(fm, ["model", "target", "group"], val)
        G[val] = gains(fm, val)
        for mdl in ["RF", "MLP", "MTE"]:
            head[f"{mdl} {val}"] = sm.loc[mdl].reindex(order).m.round(3)
        for b in ["RF", "MLP"]:
            g = G[val][b].reindex(order)
            head[f"MTE-{b} {val}"] = [f"{v:+.3f}{'*' if r else ''}" for v, r in zip(g.m, g.real)]
    gp = {b: G["r2"][b].xs("pooled", level="group") for b in ["RF", "MLP"]}

    rs = summarise(rr, ["model", "group"], "r2").m.unstack("model")[["RF", "MLP", "MTE"]].reindex(GROUPS)
    rmsd = summarise(rr[rr.model == "member"], ["group"], "r2").seed_sd
    rg = {}
    for b in ["RF", "MLP"]:
        g = paired(rr, b, "MTE", "model", ["group"]).reindex(GROUPS)
        g["seed_sd"] = rmsd.reindex(GROUPS).to_numpy()
        g["real"] = (g.m.abs() > 2 * g.fold_sd) & (g.m.abs() > g.seed_sd)
        rg[b] = g
    ccs = cc.groupby(["group", "model"])[["macro_f1", "worst_f1", "f1w"]].mean().unstack("model").reindex(GROUPS)
    cut_set = cc.cut_set.iloc[0]

    def verdict(g):
        return (f"MTE better" if g.m > 0 else "MTE worse") if g.real else "tie (noise)"

    rdg = [f"| {'Question':30s} | Answer", f"| {'-' * 30} | {'-' * 70}"]
    for b in ["RF", "MLP"]:
        rdg.append(f"| {'Better than ' + b + '?':30s} | " + "; ".join(
            f"{t} {verdict(gp[b].loc[t])} ({gp[b].loc[t].m:+.3f})" for t in T3))
    rdg.append(f"| {'p98 (head ' + best['cfg']['head'] + ')':30s} | MTE R2 {misc['MTE_r2_p98']:.3f}; vs RF "
               f"{gp['RF'].loc['p98'].m:+.3f}, vs MLP {gp['MLP'].loc['p98'].m:+.3f}")
    rdg.append(f"| {'p98 residual':30s} | R2 RF {rs.loc['pooled', 'RF']:.3f} / MLP {rs.loc['pooled', 'MLP']:.3f} / "
               f"MTE {rs.loc['pooled', 'MTE']:.3f}; MTE-RF {rg['RF'].loc['pooled'].m:+.3f} "
               f"({'real' if rg['RF'].loc['pooled'].real else 'noise'})")
    rdg.append(f"| {'Classes (' + cut_set + ')':30s} | macro F1 pooled RF {ccs.loc['pooled', ('macro_f1', 'RF')]:.3f}"
               f" / MLP {ccs.loc['pooled', ('macro_f1', 'MLP')]:.3f} / MTE {ccs.loc['pooled', ('macro_f1', 'MTE')]:.3f}")
    rdg.append(f"| {'p98 >= mean (OOF)?':30s} | px with p98 < mean: RF {misc['oof_p98_lt_mean_RF']:.2%}, MLP "
               f"{misc['oof_p98_lt_mean_MLP']:.2%}, MTE {misc['oof_p98_lt_mean_MTE']:.2%} "
               f"(labels {misc['label_p98_lt_mean']:.2%})")
    rdg.append(f"| {'SD flags errors?':30s} | Spearman(SD, |err|): " + ", ".join(
        f"{t} {sk.loc[t].spearman:.2f}" for t in T3))
    mr = OUTE / "map_range.csv"
    if mr.exists():
        z = pd.read_csv(mr).set_index(["target", "model"])
        rdg.append(f"| {'Map px clipped / near 0':30s} | at 0: MLP " + ", ".join(
            f"{t} {z.loc[(t, 'MLP')].at_zero:.1%}" for t in T3) + "; MTE < 0.05: " + ", ".join(
            f"{t} {z.loc[(t, 'MTE')].lt_005:.1%} (labels {z.loc[(t, 'MTE')].label_lt_005:.1%})" for t in T3))
        rdg.append(f"| {'Map heights in range?':30s} | max MTE / MLP / label: " + "; ".join(
            f"{t} {z.loc[(t, 'MTE')].map_max:.1f} / {z.loc[(t, 'MLP')].map_max:.1f} / {z.loc[(t, 'MTE')].label_max:.1f}"
            for t in ["mean", "p98"]))
    ef_m = eff[eff.model == "MTE members"]
    rdg.append(f"| {'Training effective?':30s} | beats ridge: " + ", ".join(
        f"{t} {misc['MTE_beats_ridge_' + t]}" for t in T3) + f"; fits at max epoch {int(ef_m.fits_at_max_ep.iloc[0])}"
        f" of {int(ef_m.fits.iloc[0])}")

    sm = summarise(fm[fm.group == "pooled"], ["model", "target"], "r2").m.unstack("target")[T3]
    abl = sm.reindex(["RF", "MLP", "MTnoclamp", "MT1", "MTE"]).round(3)

    rf_imp = []
    for t in ["mean", "cover"]:
        ii = pd.read_csv(R.OUTD / f"imp_ALL_{t}.csv")
        rf_imp += [dict(model="RF", target=t, sensor=n, dR2=v) for n, v in
                   ii[ii.kind == "perm_dR2"].groupby("name").value.mean().items()]
    mi = pd.read_csv(M.OUTM / "importance.csv").groupby(["target", "sensor"]).dR2.mean().reset_index().assign(
        model="MLP")
    im = pd.concat([pd.DataFrame(rf_imp), mi, imp.groupby(["target", "sensor"]).dR2.mean().reset_index().assign(
        model="MTE")]).set_index(["target", "model", "sensor"]).dR2.unstack("sensor").round(3)

    L = ["REPORT — multi-task bounded MLP, 5-seed ensemble (MTE) vs RF and single-task MLP, ALL stack",
         "(mlp_mte_2026.py; plan PLAN_mlp_mte.md)",
         "Labels 39,023 px (Dampier 2026 + Yawuru kept); folds_cover_height.json; 49 features (NISAR + S1 + S2).",
         f"MTE: one net, 3 bounded heads; best config {best['cfg']} (chosen on the test folds: optimistic);",
         "inputs clamped to the training p0.5 / p99.5; 5 members (seeds 0-4), ensemble = member mean.",
         "Values: fold mean over seeds (RF, MLP 3 seeds; MTE 1 ensemble). Gains paired per fold on seed 0;",
         "* = real: |gain| > 2 x fold SD and > SD of the 5 members' fold-mean value.",
         "Random K-fold under spatial autocorrelation: optimistic.", "",
         "1. HEADLINE — K-fold R2 and RMSE (m; cover as fraction)", head.to_string(), "",
         "2. READING", *rdg, "",
         "3. ABLATION — pooled K-fold R2 (MT1 = member seed 0; MTnoclamp = MT1 without the input clamp)",
         abl.to_string(), "",
         "configs (seed 0, 1 member; pooled OOF R2)", cf.round(4).to_string(index=False), "",
         "4. EFFECTIVENESS", eff.round(3).to_string(index=False),
         f"test rows with >= 1 clamped feature: {misc['test_rows_clamped']:.1%} (mean {misc['test_feats_clamped_mean']:.2f}"
         f" features per row); label px with p98 < mean {misc['label_p98_lt_mean']:.3%}", "",
         "uncertainty: Spearman(member SD, |OOF error|), pooled; mean |error| per SD quintile",
         sk.round(3).to_string(), "",
         "5. p98 RESIDUAL r = p98 - g(cover) (g = E[p98 | cover], training-fold labels); K-fold R2 vs LiDAR r",
         pd.DataFrame({**{f"{m} R2": rs[m].round(3) for m in ["RF", "MLP", "MTE"]},
                       **{f"MTE-{b}": [f"{v:+.3f}{'*' if r else ''}" for v, r in zip(rg[b].m, rg[b].real)]
                          for b in ["RF", "MLP"]}}).to_string(), "",
         f"6. CLASS CHECK — px-mean preds -> classes, cut set {cut_set}; RF / MLP mean of 3 seeds",
         ccs.round(3).to_string(), "",
         "7. IMPORTANCE — permutation R2 drop by sensor (fold mean; MLP / MTE seed 0; RF p98 has impurity only)",
         im.to_string()]
    mc = OUTE / "map_compare.txt"
    L += ["", "8. MAPS", mc.read_text(encoding="utf-8") if mc.exists() else "not run yet (mlp_mte_2026.py maps)"]
    mrt = OUTE / "map_range.txt"
    if mrt.exists():
        L += ["9. MAP RANGE", mrt.read_text(encoding="utf-8")]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    print(f"\nwrote {REPORT}", flush=True)


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
