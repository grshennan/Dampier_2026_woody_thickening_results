"""Why the 20 % cover cut lies above h at h = 0.258 m and below h elsewhere; a simple rule for the two groups
in the top-left panel of fig_cover_fit.png. Plan: C:\\Users\\glens\\.claude\\plans\\resume-from-nisar-2026-class-defs-checkp-functional-squid.md

Per px: mean = c * mhi + (1 - c) * mlo (c = cover at h; mhi / mlo = mean of the 50 cm cells >= h / < h).
At c = 0.2 the cut is 0.2 * mhi + 0.8 * mlo, so cut > h when mhi is large compared to h.

Outputs (outputs/): cover_split.csv (per h, px at 18-22 % cover), cover_split_rules.csv (line per group per rule),
cover_split_lowend.csv (50 cm cell heights 0-2 m per site), fig_cover_groups.png.

    C:\\Users\\glens\\miniconda3\\Scripts\\conda.exe run -n general --no-capture-output python -u cover_split.py
"""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from common import COVER, OUT, PROJ, L, chm_path, col, cover_table, labels, load_chm, mhi, mlo, ref_grid

sys.path.insert(0, str(PROJ / "chm_regression"))
import viz_style as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

H0 = 0.258
BLUE, ORANGE = V.TARGET_COLORS["mean"], V.TARGET_COLORS["max"]
LOW_BINS = [-np.inf, 0.0, 0.1, 0.258, 0.5, 1.0, 2.0, np.inf]


def heights_cached():
    return sorted(float(m.group(1)) for c in pd.read_parquet(COVER).columns if (m := re.fullmatch(r"cover_(.+)", c)))


def per_h(t, hs):
    rows = []
    for h in hs:
        c = t[col(h)]
        near = (c >= 0.18) & (c <= 0.22)
        cut = t.loc[near, "mean"].median()
        rows.append(dict(h=h, n=int(near.sum()), cut_local=cut, cut_over_h=cut / h,
                         mhi=t.loc[near, mhi(h)].median(), mlo=t.loc[near, mlo(h)].median(),
                         mhi_minus_h=t.loc[near, mhi(h)].median() - h))
    df = pd.DataFrame(rows)
    r = df.cut_over_h.to_numpy()
    i = np.flatnonzero((r[:-1] - 1) * (r[1:] - 1) < 0)
    x = [np.interp(1, r[[j + 1, j]], df.h.to_numpy()[[j + 1, j]]) for j in i]  # r falls with h
    return df, x


def lowend(lab):
    """Share of 50 cm cells (inside label px) per height bin, per site."""
    ref = ref_grid()
    inv = ~ref.rio.transform()
    out = []
    for s in L.SITES:
        chm = load_chm(chm_path(s))
        v = chm.values.astype("float64")
        xs, ys = np.meshgrid(chm.x.values, chm.y.values)
        ok = np.isfinite(v)
        if chm.rio.nodata is not None and not np.isnan(chm.rio.nodata):
            ok &= v != chm.rio.nodata
        cf, rf = inv * (xs[ok], ys[ok])
        key = np.floor(rf).astype("int64") * 100000 + np.floor(cf).astype("int64")
        ls = lab[lab.site == s]
        keep = np.isin(key, ls["row"].to_numpy() * 100000 + ls["col"].to_numpy())
        h = v[ok][keep]
        cnt, _ = np.histogram(h, LOW_BINS)
        names = [f"[{lo:g}, {hi:g})" for lo, hi in zip(LOW_BINS[:-1], LOW_BINS[1:])]
        out.append(pd.Series(cnt / len(h), index=names, name=s))
        print(f"lowend {s}: {len(h):,} cells", flush=True)
    return pd.DataFrame(out)


def gmm_cut(x):
    """Threshold between the 2 components of a GMM on x (where the labels change); BIC for k = 1, 2, 3."""
    x = x.reshape(-1, 1)
    bic = [GaussianMixture(k, random_state=0, n_init=3).fit(x).bic(x) for k in (1, 2, 3)]
    g = GaussianMixture(2, random_state=0, n_init=3).fit(x)
    lo, hi = sorted(g.means_.ravel())
    grid = np.linspace(lo, hi, 2001).reshape(-1, 1)
    p = g.predict(grid)
    return float(grid[np.flatnonzero(p != p[0])[0], 0]), bic


def lines(c, y, g):
    """OLS line per group; pooled R2 over the groups."""
    rows, ss = [], 0.0
    for k in np.unique(g):
        m = g == k
        b, a = np.polyfit(c[m], y[m], 1)
        r = y[m] - (a + b * c[m])
        ss += (r ** 2).sum()
        rows.append(dict(group=k, n=int(m.sum()), slope=b, intercept=a, r2=1 - (r ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum(),
                         resid_sd=r.std()))
    return rows, 1 - ss / ((y - y.mean()) ** 2).sum()


def main():
    hs = heights_cached()
    t = cover_table(hs, cond=True)
    print(f"{len(t):,} px; decomposition mean = c*mhi + (1-c)*mlo asserted at {len(hs)} heights", flush=True)

    df, cross = per_h(t, hs)
    df.to_csv(OUT / "cover_split.csv", index=False)
    print("\nPx at 18-22 % cover, medians (m):")
    print(df.round(3).to_string(index=False))
    print(f"cut = h crossover at h = {', '.join(f'{x:.2f}' for x in cross)} m")

    low = lowend(labels())
    low.loc["all (cell-weighted mean of sites)"] = low.mean()
    low.to_csv(OUT / "cover_split_lowend.csv")
    print("\n50 cm cells in label px, share per height bin (m):")
    print(low.round(3).to_string())

    # two-group rules at h = 0.258
    c, y = t[col(H0)].to_numpy(), t["mean"].to_numpy()
    vh = t[mhi(H0)].to_numpy()
    share = np.where(c > 0, t[col(2.0)].to_numpy() / np.where(c > 0, c, 1), 0)
    use = c >= 0.05                                      # mhi is noisy where few cells are >= h
    t2, bic2 = gmm_cut(np.log(vh[use]))
    t3, bic3 = gmm_cut(share[use])
    rules = {"one line": np.zeros(len(c), int),
             "R1 site": t.site.to_numpy(),
             f"R2 mhi > {np.exp(t2):.2f} m": (vh > np.exp(t2)).astype(int),
             f"R3 cover2/cover0.258 > {t3:.2f}": (share > t3).astype(int)}
    out = []
    for name, g in rules.items():
        rows, r2p = lines(c, y, g)
        for r in rows:
            out.append(dict(rule=name, pooled_r2=r2p, **r))
    rr = pd.DataFrame(out)
    rr.to_csv(OUT / "cover_split_rules.csv", index=False)
    print(f"\nGMM BIC k=1/2/3: log mhi {np.round(bic2, 0)}; share {np.round(bic3, 0)} (lower = better)")
    print("\nLine per group, h = 0.258 m (mean ~ cover):")
    print(rr.round(3).to_string(index=False))

    fig_groups(t, c, y, vh, share, rules, np.exp(t2), t3)


def fig_groups(t, c, y, vh, share, rules, t2, t3):
    V.apply_style()
    fig, axs = plt.subplots(2, 5, figsize=(22, 8.5))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(c))
    keys = list(rules)
    for ax, k, desc in zip(axs[0, :2], keys[2:], [f"mhi(0.258) = mean height of cells >= 0.258 m; cut {t2:.2f} m",
                                                  f"share of vegetated cells >= 2 m; cut {t3:.2f}"]):
        g = rules[k][order]
        ax.scatter(c[order], y[order], s=1.5, c=np.where(g == 1, ORANGE, BLUE), alpha=0.35, linewidths=0,
                   rasterized=True)
        rr = pd.read_csv(OUT / "cover_split_rules.csv").query("rule == @k")
        for _, r in rr.iterrows():
            ax.plot([0, 1], [r.intercept, r.intercept + r.slope], color=V.TEXT_PRIMARY, lw=2)
            ax.annotate(f"{'high' if str(r.group) == '1' else 'low'}: slope {r.slope:.2f}, n {r.n:,}",
                        (1, r.intercept + r.slope), xytext=(-4, 4), textcoords="offset points", ha="right",
                        fontsize=8, color=V.TEXT_SECONDARY)
        ax.set_title(f"{k}\n{desc}", loc="left", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 14)
        ax.set_xlabel("cover at 0.258 m")
        ax.set_ylabel("px mean of 50 cm p98 CHM (m)")
    ax = axs[0, 2]
    ax.hist(np.log10(vh[c >= 0.05]), bins=120, color=BLUE)
    ax.axvline(np.log10(t2), color=V.TEXT_PRIMARY, ls="--", lw=1)
    ax.set_xlabel("log10 mhi(0.258) (m), px with cover >= 0.05")
    ax.set_ylabel("px")
    ax.set_title("mhi distribution with R2 cut", loc="left", fontsize=10)
    ax = axs[0, 3]
    ax.hist(share[c >= 0.05], bins=100, color=BLUE)
    ax.axvline(t3, color=V.TEXT_PRIMARY, ls="--", lw=1)
    ax.set_xlabel("cover at 2 m / cover at 0.258 m")
    ax.set_title("tree/shrub share with R3 cut", loc="left", fontsize=10)
    axs[0, 4].axis("off")
    from matplotlib.lines import Line2D
    axs[0, 4].legend([Line2D([], [], ls="", marker="o", ms=9, color=BLUE),
                      Line2D([], [], ls="", marker="o", ms=9, color=ORANGE),
                      Line2D([], [], color=V.TEXT_PRIMARY, lw=2)],
                     ["low group (rule false)", "high group (rule true)", "OLS line per group"],
                     loc="center", frameon=False)
    sr = pd.read_csv(OUT / "cover_split_rules.csv").query("rule == 'R1 site'").set_index("group")
    for ax, s in zip(axs[1], L.SITES):
        m = (t.site == s).to_numpy()
        ax.scatter(c[m], y[m], s=1.5, color=V.TEXT_MUTED, alpha=0.3, linewidths=0, rasterized=True)
        r = sr.loc[s]
        ax.plot([0, 1], [r.intercept, r.intercept + r.slope], color=V.TEXT_PRIMARY, lw=2)
        ax.set_title(f"R1 site: {s}\nslope {r.slope:.2f}, R2 {r.r2:.2f}, n {int(r.n):,}", loc="left", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 14)
        ax.set_xlabel("cover at 0.258 m")
    axs[1, 0].set_ylabel("px mean of 50 cm p98 CHM (m)")
    fig.suptitle("h = 0.258 m: px mean against cover, split by candidate rules (2026 LiDAR, 5 sites, NISAR 20 m label px)",
                 x=0.01, ha="left")
    fig.tight_layout()
    out = OUT / "fig_cover_groups.png"
    fig.savefig(out, dpi=130)
    print(f"written: {out.resolve()}")


if __name__ == "__main__":
    main()
