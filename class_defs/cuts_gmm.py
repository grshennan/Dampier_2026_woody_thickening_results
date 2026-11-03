"""Step 3: GMM sets -- 1-D GaussianMixture on (a) 50 cm p98 CHM heights inside the label px and (b) the label
px `mean` (20 m). k = 4 and k = argmin BIC (k = 1..10). Settings as structure_2d/find_classes_2d.py
(n_init 5, random_state 42, reg_covar 1e-4; 1-D, so covariance type does not matter).

Cut = height where the argmax posterior changes (grid step 1 mm over the data range). A component that never
wins gives no cut (reported). 50 cm cuts are canopy heights: they are translated to `mean` with the 20 % cover
fit at h = cut (translation.cover_fit; cover from common.cover_table). CI: 20 bootstrap resamples of the GMM
fit (cut heights only; the translated value carries the cover-fit SE instead).

Outputs: outputs/cut_sets.json (all six sets, incl. old and nfi), outputs/gmm_bic.csv, outputs/fig_gmm.png.

    C:\\Users\\glens\\miniconda3\\Scripts\\conda.exe run -n general --no-capture-output python -u cuts_gmm.py
    ... cuts_gmm.py --test     # self-test on synthetic data
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "10")

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from common import OLD_CUTS, OUT, L, chm_path, col, cover_table, labels, load_chm, ref_grid, translation  # noqa: E402

K_MAX, N_50CM, N_BOOT, SEED = 10, 1_000_000, 20, 42


def gmm(x, k, seed=SEED):
    return GaussianMixture(k, n_init=5, random_state=seed, reg_covar=1e-4).fit(x.reshape(-1, 1))


def cuts_of(g, lo, hi):
    """Change points of the argmax posterior on a 1 mm grid; also the winner sequence."""
    grid = np.arange(lo, hi, 0.001)
    order = np.argsort(g.means_.ravel())
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    win = rank[g.predict(grid.reshape(-1, 1))]
    ch = np.nonzero(np.diff(win))[0]
    seq = [int(win[0])] + [int(win[i + 1]) for i in ch]
    return [round(float(grid[i] + 0.0005), 3) for i in ch], seq


def fit_set(x, rng):
    """BIC over k, then cuts for k = 4 and k = argmin BIC, each with a bootstrap CI."""
    bic = [gmm(x, k).bic(x.reshape(-1, 1)) for k in range(1, K_MAX + 1)]
    k_bic = int(np.argmin(bic)) + 1
    lo, hi = float(x.min()), float(x.max())
    res = {}
    for name, k in (("k4", 4), ("bic", k_bic)):
        g = gmm(x, k)
        cuts, seq = cuts_of(g, lo, hi)
        boot = []
        for b in range(N_BOOT):
            gb = gmm(x[rng.integers(0, len(x), len(x))], k, seed=SEED + 1 + b)
            boot.append(cuts_of(gb, lo, hi)[0])
        same = [c for c in boot if len(c) == len(cuts)]
        ci = np.percentile(same, [2.5, 97.5], axis=0).round(3).tolist() if same else None
        res[name] = dict(k=k, cuts=cuts, winners=seq, monotonic=seq == sorted(seq),
                         ci=ci, boot_same_n=len(same), means=np.sort(g.means_.ravel()).round(3).tolist(),
                         model=g)
        print(f"  {name}: k {k} cuts {cuts} winners {seq} CI {ci} ({len(same)}/{N_BOOT} same count)", flush=True)
    return bic, res


def sample_50cm(lab, rng):
    """All valid 50 cm cells inside the label px of each site, then one pooled random sample of N_50CM."""
    tf = ref_grid().rio.transform()
    parts = []
    for s in L.SITES:
        chm = load_chm(chm_path(s))
        v = chm.values.astype("float32")
        xs, ys = np.meshgrid(chm.x.values, chm.y.values)
        c = np.floor((xs - tf.c) / tf.a).astype("int64")
        r = np.floor((ys - tf.f) / tf.e).astype("int64")
        keep = set(zip(*lab.loc[lab.site == s, ["row", "col"]].to_numpy().T))
        key = r * 100_000 + c
        want = np.array([a * 100_000 + b for a, b in keep])
        m = np.isfinite(v) & np.isin(key, want)
        parts.append(v[m])
        print(f"  {s}: {m.sum():,} cells in {len(keep):,} label px", flush=True)
    allv = np.concatenate(parts)
    n_lab = int(lab["count"].sum())
    assert len(allv) == n_lab, f"{len(allv)} cells != label count sum {n_lab}"
    return rng.choice(allv, N_50CM, replace=False).astype("float64")


def translate(cuts):
    """50 cm cut heights -> `mean` cuts by the 20 % cover fit at h = cut."""
    tab = cover_table(cuts)
    out = []
    for h in cuts:
        f = translation.cover_fit(tab.rename(columns={col(h): f"cover_{h}"}), "mean", h)
        out.append(dict(h=h, cut=round(f["threshold"], 3), se=round(f["se"], 4), r2=round(f["r2"], 3),
                        weak=h < 0.5))
    return out


def figure(data, bics, fits, path):
    import sys
    sys.path.insert(0, str(L.HERE.parent / "chm_regression"))
    import viz_style as V
    import matplotlib.pyplot as plt
    V.apply_style()
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for j, (name, x) in enumerate(data.items()):
        a = ax[0, j]
        hi = np.percentile(x, 99.9)
        a.hist(x, bins=200, range=(0, hi), density=True, color="#b9cde4", edgecolor="none")
        g = fits[name]["k4"]["model"]
        t = np.linspace(0, hi, 1000)
        dens = sum(w * norm.pdf(t, m, np.sqrt(v)) for w, m, v in
                   zip(g.weights_, g.means_.ravel(), g.covariances_.ravel()))
        a.plot(t, dens, color=V.TEXT_PRIMARY, lw=1.5, label="GMM k = 4")
        for c in fits[name]["k4"]["cuts"]:
            a.axvline(c, color=V.TEXT_SECONDARY, ls="--", lw=1)
        for c in fits[name]["bic"]["cuts"]:
            a.axvline(c, color=V.TEXT_MUTED, ls=":", lw=1)
        a.set_yscale("log")
        a.set_xlabel("height (m)" if name == "50cm" else "px mean of p98 (m)")
        a.set_ylabel("density (log)")
        a.set_title(f"{name}: k = 4 cuts dashed, BIC k = {fits[name]['bic']['k']} cuts dotted", loc="left")
        b = ax[1, j]
        b.plot(range(1, K_MAX + 1), bics[name], color=V.TEXT_PRIMARY, lw=2, marker="o", ms=8)
        b.set_xlabel("k")
        b.set_ylabel("BIC")
        b.set_title(f"{name}: BIC, min at k = {fits[name]['bic']['k']}", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def self_test():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 50_000), rng.normal(6, 1, 50_000)])
    g = gmm(x, 2)
    cuts, seq = cuts_of(g, x.min(), x.max())
    assert len(cuts) == 1 and abs(cuts[0] - 3.0) < 0.05, cuts   # equal weights + SD: crossing at 3
    print("self-test ok:", cuts, seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    if ap.parse_args().test:
        return self_test()
    rng = np.random.default_rng(SEED)
    lab = labels()
    print("sampling 50 cm cells", flush=True)
    data = {"50cm": sample_50cm(lab, rng), "20m": lab["mean"].to_numpy("float64")}
    bics, fits = {}, {}
    for name, x in data.items():
        print(f"{name}: n {len(x):,}", flush=True)
        bics[name], fits[name] = fit_set(x, rng)
    pd.DataFrame({f"bic_{n}": b for n, b in bics.items()}, index=range(1, K_MAX + 1)).rename_axis("k") \
        .to_csv(OUT / "gmm_bic.csv")

    sets = {"old": dict(cuts=OLD_CUTS, method="published 0.2 / 1.22 / 2.35")}
    sets["nfi"] = json.load(open(OUT / "cuts_nfi.json"))
    for kname in ("k4", "bic"):
        f = fits["50cm"][kname]
        tr = translate(f["cuts"])
        sets[f"gmm50_{kname}"] = dict(cuts=[t["cut"] for t in tr], cuts_50cm=f["cuts"], ci_50cm=f["ci"], k=f["k"],
                                      translation=tr, winners=f["winners"], monotonic=f["monotonic"],
                                      method="GMM on 50 cm heights, cuts by 20 % cover fit")
        f = fits["20m"][kname]
        sets[f"gmm20_{kname}"] = dict(cuts=f["cuts"], ci=f["ci"], k=f["k"], winners=f["winners"],
                                      monotonic=f["monotonic"], method="GMM on px mean")
    for s in sets.values():
        s.pop("set", None)
        s["cuts"] = sorted(s["cuts"])
    with open(OUT / "cut_sets.json", "w") as fh:
        json.dump(sets, fh, indent=1)
    for n, s in sets.items():
        print(f"{n:10s} {s['cuts']}")
    figure(data, bics, fits, OUT / "fig_gmm.png")
    print(f"written: {(OUT / 'cut_sets.json').resolve()}\nwritten: {(OUT / 'fig_gmm.png').resolve()}")


if __name__ == "__main__":
    main()
