"""Part B step B1 (PLAN_cover_height.md): rows table + folds for the cover x height regressions.

Outputs (outputs/cover_height/):
  rows.parquet  43,134 rows: source, site, row, col, group (Dampier / Yawuru kept / Yawuru burnt), s2_flag, fire,
                mean, count, cover_1.37, mhi_1.37, mlo_1.37. Row order = folds index.
  ../folds_cover_height.json  StratifiedKFold(5, shuffle, 42) on site x within-site cover quintile, over all rows.
Same strata as fire_gap.folds_for. Main run (B3) uses group != Yawuru burnt; B3b uses all rows, same folds.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import common as C

H = 1.37
FG = C.OUT / "fire_gap"
OUTD = C.OUT / "cover_height"
BURNT = ["YA25_1", "YA25_4", "YJ25_2", "YJ25_3", "YJ25_4", "YJ25_5"]  # Gate A, user 2026-11-02
S2_FLAG = ["20250407_govcowwell", "YA25_3"]  # S2-only NBR drop 2026-04-29..05-02, kept
KEY = ["site", "row", "col"]


def build():
    t = pd.read_parquet(FG / "cells.parquet")
    f = pd.read_parquet(FG / "fire_cells.parquet")[KEY + ["fire"]]
    assert not t.duplicated(KEY).any() and not f.duplicated(KEY).any()
    t = t.merge(f, on=KEY, how="left", validate="1:1")
    assert t.fire.notna().all()
    t["group"] = np.where(t.source == "Dampier 2026", "Dampier",
                          np.where(t.site.isin(BURNT), "Yawuru burnt", "Yawuru kept"))
    t["s2_flag"] = t.site.isin(S2_FLAG)
    # decomposition: mean = c*mhi + (1-c)*mlo (nan component only where its weight is 0)
    c, hi, lo = t[C.col(H)], t[C.mhi(H)].fillna(0), t[C.mlo(H)].fillna(0)
    assert np.allclose(c * hi + (1 - c) * lo, t["mean"], atol=1e-4), "decomposition"
    assert t.loc[c > 0, C.mhi(H)].notna().all() and t.loc[c < 1, C.mlo(H)].notna().all()
    return t.reset_index(drop=True)


def folds(t):
    q = t.groupby("site")[C.col(H)].rank(method="first", pct=True).mul(5).clip(upper=4.999).astype(int)
    strata = t.site + "_" + q.astype(str)
    out = [te.tolist() for _, te in StratifiedKFold(5, shuffle=True, random_state=42).split(t, strata)]
    assert sorted(sum(out, [])) == list(range(len(t)))
    return out


def main():
    OUTD.mkdir(exist_ok=True)
    t = build()
    n = t.group.value_counts()
    print(n.to_string())
    assert (n["Dampier"], n["Yawuru kept"], n["Yawuru burnt"]) == (34715, 4308, 4111), "group counts"
    t.to_parquet(OUTD / "rows.parquet")
    fo = folds(t)
    json.dump({"h": H, "n": len(t), "test_folds": fo}, open(C.OUT / "folds_cover_height.json", "w"))
    fold = np.empty(len(t), int)
    for k, te in enumerate(fo):
        fold[te] = k
    c = t[C.col(H)]
    s = t.assign(fold=fold, hi=c >= 0.05, lo=c <= 0.95)
    print("\nn per group: all / mhi rows (c >= 0.05) / mlo rows (c <= 0.95)")
    print(s.groupby("group")[["hi", "lo"]].agg(["size", "sum"]).iloc[:, [0, 1, 3]]
          .set_axis(["all", "mhi", "mlo"], axis=1).to_string())
    print("\nfold sizes per group")
    print(pd.crosstab(s.group, s.fold).to_string())
    print("\nn per site (s2_flag, fire gap)")
    print(t.groupby(["group", "site"]).agg(n=("row", "size"), s2_flag=("s2_flag", "first"),
                                           gap=("fire", lambda x: (x == "gap").sum())).to_string())
    print(f"\nwrote {OUTD / 'rows.parquet'}, {C.OUT / 'folds_cover_height.json'}")


if __name__ == "__main__":
    main()
