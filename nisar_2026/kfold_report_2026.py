"""Print the K-fold report of one classification run: OA / balanced / kappa / weighted F1, per-class
precision, recall and F1, and the feature importances of that run (top 10, GCOV / GSLC / S2 group shares
and per-track shares) beside every other classification run in outputs\\importance_record.csv.

    conda run -n general --no-capture-output python kfold_report_2026.py outputs\\classify 5class
    (args: the run's output folder, its run name in the record)
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feature_stack_2026 import (DCMP_BANDS, GCOV_DCMP_FEATS, GCOV_FEATS, GSLC_BS_FEATS, GSLC_G0_FEATS,  # noqa: E402
                                IMPORTANCE_RECORD)


def group(f):
    """GCOV backscatter, GSLC backscatter (--sar gslc: sigma0; gslc_gamma: gamma0), GSLC decomposition, GCOV
    no-phase decomposition (gcov_nophase), S2."""
    base = f.split("_", 2)[-1] if f[:5] in ("147_D", "168_A") else f
    return ("S1" if f.startswith("S1_") else "GCOV" if base in GCOV_FEATS else "GSLC_bs" if base in GSLC_BS_FEATS
            else "GSLC_g0" if base in GSLC_G0_FEATS else "GCOV_dcmp" if base in GCOV_DCMP_FEATS
            else "GSLC" if base in DCMP_BANDS else "S2")


def main(folder, run):
    allm = pd.read_csv(Path(folder) / "metrics.csv")
    src = sorted(s for s in allm.site.unique() if s.startswith("src_"))    # classify_2026 --extra-labels
    for scope in src or ["all"]:
        met = allm.query("site == @scope").set_index("eval")
        k = met.loc["kfold"]
        classes = [c[len("precision_"):] for c in met.columns if c.startswith("precision_")]
        print(f"{scope}: K-fold (random stratified 5-fold): OA {k.oa:.3f}, balanced {k.balanced_acc:.3f}, "
              f"kappa {k.kappa:.3f}, weighted F1 {k.f1_weighted:.3f} (naive OA {met.loc['naive', 'oa']:.3f})")
        print(pd.DataFrame({m: [k[f"{m}_{c}"] for c in classes] for m in ("precision", "recall", "f1", "n")},
                           index=classes).round(3).to_string())
    rec = pd.read_csv(IMPORTANCE_RECORD).query("model == 'XGBClassifier'")
    rec = rec.assign(key=rec["run"] + " / " + rec["gslc_grid"] + " / n" + rec["n_iter"].astype(str),
                     group=rec["feature"].map(group),
                     track=rec["feature"].str[:5].where(rec["feature"].str[:5].isin(["147_D", "168_A"]), "S2"))
    rec = rec.drop_duplicates(["key", "feature"], keep="last")
    last = rec[rec["run"] == run].sort_values("written")["key"].iloc[-1]       # the newest version of this run
    this = rec[rec["key"] == last]
    print(f"\nTop 10 features ({this['importance_type'].iloc[0]}, run {this['key'].iloc[0]}):")
    print(this.nsmallest(10, "rank")[["rank", "feature", "group", "importance"]].round(4).to_string(index=False))
    for col in ("group", "track"):
        print(f"\nImportance share by {col} (all classification runs in the record):")
        print(rec.pivot_table(index=col, columns="key", values="importance", aggfunc="sum").round(3).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
