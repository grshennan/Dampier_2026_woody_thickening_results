"""Step 7b (env pytorch): MLP on the class_defs sets, same rows, features and 5 folds as the XGB runs.

Inputs (from export_features.py, env general): outputs/features_2026.parquet, outputs/folds_<set>.json.
Model: features -> h1 -> h2 -> n_classes, ReLU, dropout; AdamW lr 1e-3; cross-entropy with balanced class weights
(n / (k * n_c), from the training fold; as XGB's balanced weights); batch 512; early stopping on the weighted
validation loss (stratified 15 % of the training fold, patience 20, max 300 epochs; best epoch restored).
Inputs standardised on the training part only.
Search: 12 configs (widths x dropout x weight decay), scored by K-fold F1w -- the folds also choose the config,
the same optimism as XGB (classify_2026.py divergence 1). Best config: seeds 42 (search), 1, 2 -> noise band.
Overfit check: train-part F1w vs K-fold F1w. Importances: permutation drop in K-fold F1w (3 repeats).
Stratified K-fold only; no leave-site-out.
Outputs: outputs/mlp_<set>/ configs.csv, metrics.csv, confusion_kfold.csv, importance.csv, history.csv,
seeds.csv, predictions.parquet.

    C:\\Users\\glens\\miniconda3\\Scripts\\conda.exe run -n pytorch --no-capture-output python -u mlp_2026.py --sets nfi --quick
    C:\\Users\\glens\\miniconda3\\Scripts\\conda.exe run -n pytorch --no-capture-output python -u mlp_2026.py --sets nfi gmm20_k4
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent / "outputs"
META = ["site", "row", "col", "mean"]
GRID = [dict(w=w, drop=d, wd=wd) for w, d, wd in itertools.product([(32, 16), (64, 32), (128, 64)], [0.2, 0.4],
                                                                     [1e-4, 1e-3])]
# --grid wide (user 2026-11-02): the default grid's best was its widest net (128-64); 128-64 repeated as the link
GRID_WIDE = [dict(w=w, drop=d, wd=wd) for w, d, wd in itertools.product([(128, 64), (256, 128), (512, 256), (1024, 512)],
                                                                          [0.2, 0.4], [1e-4, 1e-3])]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def net(n_in, n_out, w, drop):
    return torch.nn.Sequential(torch.nn.Linear(n_in, w[0]), torch.nn.ReLU(), torch.nn.Dropout(drop),
                               torch.nn.Linear(w[0], w[1]), torch.nn.ReLU(), torch.nn.Dropout(drop),
                               torch.nn.Linear(w[1], n_out))


def fit(X, y, k, cfg, seed, max_ep, patience=20):
    torch.manual_seed(seed)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y, dtype=torch.long, device=DEV)
    cw = torch.tensor(len(tr) / (k * np.bincount(y[tr], minlength=k).clip(1)), dtype=torch.float32, device=DEV)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cw)
    m = net(X.shape[1], k, cfg["w"], cfg["drop"]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=cfg["wd"])
    tr_t, va_t = torch.tensor(tr, device=DEV), torch.tensor(va, device=DEV)
    best, best_state, wait, hist = np.inf, None, 0, []
    g = torch.Generator(device=DEV).manual_seed(seed)
    for ep in range(max_ep):
        m.train()
        perm = tr_t[torch.randperm(len(tr_t), device=DEV, generator=g)]
        tl = 0.0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = loss_fn(m(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tl += loss.item() * len(b)
        m.eval()
        with torch.no_grad():
            vl = loss_fn(m(Xt[va_t]), yt[va_t]).item()
        hist.append((ep, tl / len(tr), vl))
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {n: t.detach().clone() for n, t in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, mu, sd, hist


def predict(model, mu, sd, X):
    with torch.no_grad():
        return model(torch.tensor((X - mu) / sd, dtype=torch.float32, device=DEV)).argmax(1).cpu().numpy()


def kfold(X, y, k, folds, cfg, seed, max_ep):
    oof, tr_f1, models, hist = np.full(len(y), -1), [], [], []
    for f, te in enumerate(folds):
        trn = np.setdiff1d(np.arange(len(y)), te)
        m, mu, sd, h = fit(X[trn], y[trn], k, cfg, seed, max_ep)
        oof[te] = predict(m, mu, sd, X[te])
        tr_f1.append(f1_score(y[trn], predict(m, mu, sd, X[trn]), average="macro"))
        models.append((m, mu, sd))
        hist += [dict(fold=f, epoch=e, train_loss=a, val_loss=b) for e, a, b in h]
    assert (oof >= 0).all()
    return oof, float(np.mean(tr_f1)), models, hist


def metrics(y, p, k, names):
    pr, rc, f1, n = precision_recall_fscore_support(y, p, labels=range(k), zero_division=0)
    return {"oa": accuracy_score(y, p), "balanced_acc": balanced_accuracy_score(y, p), "kappa": cohen_kappa_score(y, p),
            "f1_weighted": f1_score(y, p, average="weighted"), "f1_macro": f1_score(y, p, average="macro"),
            "f1_min": float(f1.min()),
            **{f"{s}_{c}": float(v[i]) for s, v in (("precision", pr), ("recall", rc), ("f1", f1), ("n", n))
               for i, c in enumerate(names)}}


def run_set(name, df, feats, a):
    spec = json.load(open(OUT / f"folds_{name}.json"))
    cuts, folds = spec["cuts"], [np.array(t) for t in spec["test_folds"]]
    X, y = df[feats].to_numpy("float64"), np.digitize(df["mean"], cuts)
    k = len(cuts) + 1
    names = [f"<{cuts[0]:g}"] + [f"{lo:g}-{hi:g}" for lo, hi in zip(cuts, cuts[1:])] + [f">={cuts[-1]:g}"]
    out = OUT / (f"mlp_{name}" + ("_wide" if a.grid == "wide" else "") + ("_quick" if a.quick else ""))
    out.mkdir(exist_ok=True)
    g0 = GRID_WIDE if a.grid == "wide" else GRID
    grid, max_ep = (g0[:2], 30) if a.quick else (g0, 300)
    print(f"\n{name}: cuts {cuts}, n {len(y):,}, classes {np.bincount(y).tolist()}, {len(grid)} configs, {DEV}",
          flush=True)
    rows, res = [], {}
    for i, cfg in enumerate(grid):
        t0 = time.time()
        oof, trf1, models, hist = kfold(X, y, k, folds, cfg, 42, max_ep)
        f1k = f1_score(y, oof, average="macro")
        rows.append(dict(cfg=i, w=str(cfg["w"]), drop=cfg["drop"], wd=cfg["wd"], f1m_kfold=f1k, f1m_train=trf1,
                         epochs=np.mean([max(h["epoch"] for h in hist if h["fold"] == f) + 1 for f in range(5)]),
                         s=time.time() - t0))
        res[i] = (oof, models, hist)
        print(f"  cfg {i} {cfg}: macro F1 K-fold {f1k:.4f}, train {trf1:.4f} ({time.time() - t0:.0f} s)", flush=True)
    cfgs = pd.DataFrame(rows)
    cfgs.to_csv(out / "configs.csv", index=False)
    b = int(cfgs.loc[cfgs["f1m_kfold"].idxmax(), "cfg"])
    oof, models, hist = res[b]
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    met = [{"eval": "kfold", "site": "all", **metrics(y, oof, k, names)}]
    groups = [("site", s) for s in sorted(df["site"].unique())]
    groups += [("source", s) for s in sorted(df["source"].unique())] if "source" in df else []  # never blended
    for c, s in groups:
        i = (df[c] == s).to_numpy()
        met.append({"eval": "kfold", "site": s if c == "site" else f"src_{s}", **metrics(y[i], oof[i], k, names)})
    pd.DataFrame(met).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(confusion_matrix(y, oof, labels=range(k)), index=[f"true {c}" for c in names],
                 columns=[f"pred {c}" for c in names]).to_csv(out / "confusion_kfold.csv")
    df[META].assign(cls=y, pred_kfold=oof).to_parquet(out / "predictions.parquet", index=False)

    # permutation importance on the test folds of the best config
    rng, base, imp = np.random.default_rng(0), f1_score(y, oof, average="macro"), {}
    for j, c in enumerate(feats):
        drops = []
        for _ in range(3):
            p = np.empty(len(y), int)
            for (m, mu, sd), te in zip(models, folds):
                Xp = X[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                p[te] = predict(m, mu, sd, Xp)
            drops.append(base - f1_score(y, p, average="macro"))
        imp[c] = np.mean(drops)
    pd.Series(imp).sort_values(ascending=False).rename("f1macro_drop").to_csv(out / "importance.csv")

    srcs = [("all", np.ones(len(y), bool))] + ([(s, (df["source"] == s).to_numpy())
                                                for s in sorted(df["source"].unique())] if "source" in df else [])
    seeds = [dict(seed=42, scope=s, **metrics(y[i], oof[i], k, names)) for s, i in srcs]
    if not a.quick:
        for sd_ in (1, 2):
            o, _, _, _ = kfold(X, y, k, folds, grid[b], sd_, max_ep)
            seeds += [dict(seed=sd_, scope=s, **metrics(y[i], o[i], k, names)) for s, i in srcs]
    sd_df = pd.DataFrame(seeds)
    sd_df.to_csv(out / "seeds.csv", index=False)
    cols = ["f1_macro", "f1_min", "f1_weighted", "balanced_acc", "oa"]
    print(f"  best cfg {b} {grid[b]}; train macro F1 {cfgs.loc[b, 'f1m_train']:.4f}")
    print(f"  K-fold mean / SD over seeds, per source (report sources, not `all`):\n"
          f"{sd_df.groupby('scope')[cols].agg(['mean', 'std']).round(4).to_string()}")
    print(f"  top importances: {pd.Series(imp).sort_values(ascending=False).head(5).round(4).to_dict()}")
    print(f"  written: {out.resolve()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--quick", action="store_true", help="2 configs, 30 epochs, 1 seed")
    ap.add_argument("--threads", type=int, default=10, help="CPU threads (cap 10)")
    ap.add_argument("--grid", choices=["default", "wide"], default="default", help="wide: 128-64 .. 1024-512")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for name in a.sets:      # folds_<set>.json names its features file (Yawuru sets: features_2026_labels_yawuru_kept)
        fn = json.load(open(OUT / f"folds_{name}.json")).get("features", "features_2026.parquet")
        df = pd.read_parquet(OUT / fn)
        feats = [c for c in df.columns if c not in META + ["source"]]
        assert len(feats) == 39, f"{fn}: {len(feats)} features"
        run_set(name, df, feats, a)


if __name__ == "__main__":
    main()
