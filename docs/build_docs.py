"""Fill the result tables in docs\\*.html from the project CSVs, and check every link.

Pages are hand-written. A table sits between markers:
    <!-- BEGIN table:<name> --> ... <!-- END table:<name> -->
This script replaces only that span. A missing source gives a "pending" table, not an error.

    conda run -n general --no-capture-output python docs\\build_docs.py           # fill tables, then check links
    conda run -n general --no-capture-output python docs\\build_docs.py --check   # check links only

Links: relative for files under the project (work on G:\\ and the Linux mount); file:///E:/... for E:\\ rasters
(Windows only; checked only on Windows).
"""
import argparse
import html
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

DOCS = Path(__file__).resolve().parent
PROJ = DOCS.parent
CD = PROJ / "class_defs" / "outputs"
NS = PROJ / "nisar_2026" / "outputs"

# MLP vs RF regression (class_defs\mlp_regression_2026.py, live 2026-11-03). XGB regression: no run yet.
MLP_REG = CD / "mlp_regression"
XGB_REG = CD / "xgb_regression"  # expected location; "pending" until it exists
MTE_REG = CD / "mlp_mte"  # follow-on E (mlp_mte_2026.py), 2026-11-03
GROUPS = {"pooled": "pooled", "Dampier": "Dampier", "Yawuru kept": "Yawuru kept"}


# ---------------------------------------------------------------- helpers
def rel(p):
    return Path(p).relative_to(PROJ).as_posix()


def src_line(*paths):
    out = []
    for p in paths:
        p = Path(p)
        if p.exists():
            when = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            out.append(f'<a href="../{rel(p)}">{html.escape(rel(p))}</a> ({when})')
        else:
            out.append(f"<code>{html.escape(rel(p))}</code> (not yet written)")
    return '<p class="src">source: ' + "; ".join(out) + "</p>"


def table(head, rows, num_from=1):
    h = "".join(f'<th{" class=n" if i >= num_from else ""}>{html.escape(str(c))}</th>' for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td{" class=n" if i >= num_from else ""}>{c}</td>' for i, c in enumerate(r))
                + "</tr>" for r in rows)
    return f'<div class="wrap"><table><tr>{h}</tr>{b}</table></div>'


def ms(m, s, nd=3):
    return f"{m:.{nd}f} ({s:.{nd}f})"


def pending(what, *paths):
    return (f'<p class="todo"><strong>pending</strong>: {html.escape(what)}</p>' + src_line(*paths))


# ---------------------------------------------------------------- tables
def t_reg_kfold():
    f = CD / "regression" / "score_summary.csv"
    if not f.exists():
        return pending("regression K-fold summary", f)
    d = pd.read_csv(f)
    rows = []
    for g in GROUPS:
        for t in ("mean", "cover"):
            x = d[(d.group == g) & (d.target == t)]
            r = lambda st, v: x[(x["stack"] == st) & (x.val == v)].iloc[0]
            rows.append([g, "px mean (m)" if t == "mean" else "cover 1.37 m",
                         *(ms(r(s, "r2").m, r(s, "r2").fold_sd) for s in ("ALL", "SAR", "MS")),
                         ms(r("ALL", "rmse").m, r("ALL", "rmse").fold_sd), f'{r("ALL", "bias").m:+.3f}'])
    return table(["group", "target", "R² ALL", "R² SAR", "R² MS", "RMSE ALL", "bias ALL"], rows, 2) + src_line(f)


def t_reg_sensor():
    f = CD / "regression" / "score_importance.csv"
    if not f.exists():
        return pending("sensor importance", f)
    d = pd.read_csv(f)
    d = d[d["stack"] == "ALL"]
    rows = [[t, *(ms(*d[(d.target == t) & (d.sensor == s)][["dR2", "fold_sd"]].iloc[0]) for s in ("NISAR", "S1", "S2"))]
            for t in ("mean", "cover")]
    return table(["target", "NISAR", "S1", "S2"], rows) + src_line(f)


def _p98(whats, labels):
    f = CD / "p98_cover" / "kfold_fold_metrics.csv"
    if not f.exists():
        return pending("p98 K-fold metrics", f)
    d = pd.read_csv(f)
    rows = []
    for w in whats:
        for g in GROUPS:
            x = d[(d.what == w) & (d.group == g)]
            rows.append([labels[w], g, *(ms(x[c].mean(), x[c].std()) for c in ("r2", "rmse", "bias", "pearson"))])
    return table(["what", "group", "R²", "RMSE (m)", "bias (m)", "Pearson r"], rows, 2) + src_line(f)


def t_p98_kfold():
    return _p98(["direct", "transfer", "transfer_mean", "oracle"],
                {"direct": "direct RF p98", "transfer": "f(mean^, cover^)", "transfer_mean": "f(mean^) only",
                 "oracle": "f(true mean, true cover)"})


def t_p98_resid():
    return _p98(["res direct", "res transfer", "win r direct", "win r transfer"],
                {"res direct": "residual, direct p98", "res transfer": "residual, transfer p98",
                 "win r direct": "5x5 window r(p98, cover), direct", "win r transfer": "5x5 window r, transfer"})


def t_cls_xgb_mlp():
    fx = NS / "classify_cd_gmm50ag4_y" / "metrics.csv"
    fm = CD / "mlp_gmm50ag4_y_wide" / "seeds.csv"
    if not (fx.exists() and fm.exists()):
        return pending("XGB vs MLP classification", fx, fm)
    x = pd.read_csv(fx)
    x = x[x["eval"] == "kfold"].set_index("site")
    f1 = [c for c in x if c.startswith("f1_") and c != "f1_weighted"]
    m = pd.read_csv(fm).groupby("scope")[["f1_macro", "f1_min", "f1_weighted"]].mean()
    rows = []
    for src, lab in (("Dampier", "Dampier"), ("Extra", "Yawuru kept")):
        xx = x.loc[f"src_{src}"]
        rows.append([lab, "XGB", f"{xx[f1].mean():.3f}", f"{xx[f1].min():.3f}", f"{xx.f1_weighted:.3f}"])
        mm = m.loc[src]
        rows.append([lab, "MLP 256-128 (3 seeds)", f"{mm.f1_macro:.3f}", f"{mm.f1_min:.3f}", f"{mm.f1_weighted:.3f}"])
    return table(["source", "model", "macro F1", "worst-class F1", "F1w"], rows, 2) + src_line(fx, fm)


def t_cls_stage15():
    f = NS / "diagnosis" / "stage15.csv"
    if not f.exists():
        return pending("Stage 1.5 runs", f)
    d = pd.read_csv(f)
    rows = [[f"<code>{html.escape(r.run)}</code>", f"{r.oa:.3f}", f"{r.balanced_acc:.3f}", f"{r.kappa:.3f}",
             f"{r.f1_weighted:.3f}"] for r in d.itertuples()]
    return table(["run", "OA", "balanced", "kappa", "F1w"], rows) + src_line(f)


def _rf_baseline():
    """RF pooled / per-group K-fold R² and RMSE for mean, cover, p98 from the finished RF outputs."""
    rows = {}
    s = pd.read_csv(CD / "regression" / "score_summary.csv")
    for g in GROUPS:
        for t in ("mean", "cover"):
            x = s[(s.group == g) & (s.target == t) & (s["stack"] == "ALL")].set_index("val").m
            rows[(t, g)] = (x["r2"], x["rmse"])
    p = pd.read_csv(CD / "p98_cover" / "kfold_fold_metrics.csv")
    for g in GROUPS:
        x = p[(p.what == "direct") & (p.group == g)]
        rows[("p98", g)] = (x.r2.mean(), x.rmse.mean())
    return rows


def t_models_reg():
    fm = MLP_REG / "fold_metrics.csv"
    fx = XGB_REG / "fold_metrics.csv"
    base = [CD / "regression" / "score_summary.csv", CD / "p98_cover" / "kfold_fold_metrics.csv"]
    res = {}  # (model, target, group) -> (r2, rmse)
    if fm.exists():
        d = pd.read_csv(fm)
        for k, x in d.groupby(["model", "target", "group"]):
            res[k] = (x.r2.mean(), x.rmse.mean())
    else:
        for (t, g), v in _rf_baseline().items():
            res[("RF", t, g)] = v
    if fx.exists():
        d = pd.read_csv(fx)
        for k, x in d.groupby(["model", "target", "group"]):
            res[k] = (x.r2.mean(), x.rmse.mean())
    fe = MTE_REG / "fold_metrics.csv"
    if fe.exists():
        d = pd.read_csv(fe)
        for k, x in d[d.model == "MTE"].groupby(["model", "target", "group"]):
            res[k] = (x.r2.mean(), x.rmse.mean())
    models = ["RF", "XGB", "MLP", "MTE", "ridge"]
    rows = []
    for t in ("mean", "p98", "cover"):
        for g in GROUPS:
            row = [t, g]
            for mdl in models:
                v = res.get((mdl, t, g))
                row.append(f"{v[0]:.3f} / {v[1]:.3f}" if v else '<span class="todo">pending</span>')
            rows.append(row)
    note = ('<p class="src">Cells: K-fold R² / RMSE (m; cover as fraction), mean over folds (and seeds). '
            "Same rows (39,023 px) and folds (<code>folds_cover_height.json</code>) for every model.</p>")
    return table(["target", "group", *models], rows, 2) + note + src_line(fm, fx, fe, *base)


def t_models_resid():
    f = MLP_REG / "resid_fold_metrics.csv"
    if not f.exists():
        return pending("p98 residual skill, RF vs MLP", f)
    d = pd.read_csv(f)
    fe = MTE_REG / "resid_fold_metrics.csv"
    if fe.exists():
        e = pd.read_csv(fe)
        d = pd.concat([d, e[e.model == "MTE"]])
    rows = []
    for (mdl, g), x in d.groupby(["model", "group"]):
        rows.append([mdl, g, ms(x.r2.mean(), x.r2.std()), ms(x.rmse.mean(), x.rmse.std())])
    return table(["model", "group", "R²", "RMSE (m)"], rows, 2) + src_line(f, *([fe] if fe.exists() else []))


def t_models_eff():
    f = MLP_REG / "effectiveness.csv"
    if not f.exists():
        return pending("MLP best configuration per target", f)
    d = pd.read_csv(f)
    cols = [c for c in ("target", "w", "drop", "r2_kfold", "seed_sd", "r2_train", "ridge_r2", "best_ep_mean",
                        "folds_at_max_ep", "clipped_frac") if c in d]
    rows = [[(f"{v:.3f}" if isinstance(v, float) else html.escape(str(v))) for v in r] for r in d[cols].itertuples(False)]
    return table(cols, rows, 3) + src_line(f)


TABLES = {"reg_kfold": t_reg_kfold, "reg_sensor": t_reg_sensor, "p98_kfold": t_p98_kfold, "p98_resid": t_p98_resid,
          "cls_xgb_mlp": t_cls_xgb_mlp, "cls_stage15": t_cls_stage15, "models_reg": t_models_reg,
          "models_resid": t_models_resid, "models_eff": t_models_eff}
MARK = re.compile(r"(<!-- BEGIN table:(\w+) -->)(.*?)(<!-- END table:\2 -->)", re.S)


def fill():
    n = 0
    for page in sorted(DOCS.glob("*.html")):
        txt = page.read_text(encoding="utf-8")

        def sub(m):
            nonlocal n
            name = m.group(2)
            if name not in TABLES:
                raise KeyError(f"{page.name}: unknown table {name}")
            n += 1
            return f"{m.group(1)}\n{TABLES[name]()}\n{m.group(4)}"

        new = MARK.sub(sub, txt)
        if new != txt:
            page.write_text(new, encoding="utf-8")
    print(f"filled {n} tables")


LINK = re.compile(r'(?:href|src)="([^"]+)"')


def check():
    bad = 0
    for page in sorted(DOCS.rglob("*.html")):
        for u in LINK.findall(page.read_text(encoding="utf-8")):
            if u.startswith(("http://", "https://", "mailto:", "#")) or "stylesheet" in u:
                continue
            u = unquote(u.split("#")[0])
            if u.startswith("file:///"):
                if not sys.platform.startswith("win"):
                    continue
                p = Path(u[len("file:///"):])
            else:
                p = (page.parent / u)
            if not p.exists():
                bad += 1
                print(f"BROKEN {page.relative_to(DOCS)}: {u}")
    print(f"link check: {bad} broken")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="check links only")
    a = ap.parse_args()
    if not a.check:
        fill()
    sys.exit(1 if check() else 0)
