# Canopy structure (height × cover) against NISAR, 2026 Dampier LiDAR

Date: 2026-10-21. Scripts: `structure_grids.py` (point clouds → grids), `structure_vs_sar.py`
(comparison, figures). Outputs: `outputs\structure_sar\` (PNGs in `figures\checks\`, `figures\pooled\`,
`figures\by_site\`). Sites: 9 Mile, Biota track, Bunda Bunda,
Bunda Extra, Country Downs, South Dampier (MVT sites excluded).

## Why

Gap fraction and canopy top height are weakly related at these sites (`gap_vs_height.py`: pooled
R² 0.30–0.45 for gap against CHM p98; near 0 at Biota track and Country Downs). L-band responds to
woody volume, which depends on both. So we tested statistics that combine cover and height.

## Method

- **All structure statistics come from the point clouds** (`_denoise.las`, z = height above ground).
  First returns only, not withheld, not class 7 / 18 (the `gap_fraction.py` filter). No CHM raster.
- **No SAR is resampled.** The points are binned onto each SAR grid:
  - `nisar20m`: GCOV grid, EPSG:32751, 20 m (the `feature_stack_2026.tif` grid). EPSG:7851 point
    coordinates are used without reprojection (project convention).
    ⚠ CRS 2026-11-03: relabelling is withdrawn (user). Reprojection is required to match grids. See `G:\python\crs_audit\CHECKPOINT.md`.
  - `albers30m`: the `E:\NISAR\Dampier` per-pass decomposition and coherence grid, EPSG:3577, 30 m.
    Point XY are transformed EPSG:7851 → EPSG:3577 with pyproj.
- **Statistics per cell:**
  - `cover137` = first returns ≥ 1.37 m / all first returns (= 1 − gap).
  - `_all`: p98, p90, mean of all first returns. `mean_all` is the mean first-return height.
  - `_can`: p98, p90, mean of canopy first returns (z ≥ 1.37 m). Nodata if fewer than 20.
  - `c_<stat>` = cover137 × stat.
- **`_all` against `_can`:** an `_all` height already contains the gap. Ground returns pull p90 and
  the mean down. So `c_*_all` scales with about cover² × height ("the gap counted twice"). The
  `_can` heights are independent of cover, so `c_*_can` counts the cover once.
- Percentiles come from a per-cell histogram (0.05 m bins, extra edge at 1.37 m), linear inside a bin.
- Cells with LiDAR fill < 0.9 (share of occupied 1 m sub-cells) are dropped as edge cells.
  20 m: 35,122 cells kept, 1,855 dropped. 30 m: 15,459 kept, 1,246 dropped.
- SAR: (1) the 26 NISAR bands of the 20 m stack (GCOV γ0 + GSLC decomposition, Sep–Oct, Refined Lee
  5×5). (2) The 30 m series: 17 passes per track, Jun–Oct (σ0 dB, Pd / Ps / Pu, DpRBI, DpRSI, alpha,
  H, DoP; Refined Lee 5×5), the per-track mean over dates, and 12 coherence pairs (unfiltered).
- Metrics: Pearson r, linear R², Spearman ρ, ρ². Pooled and per site. Descriptive only, because the
  cells are spatially autocorrelated. The backscatter–height relation is a curve, so ρ² is the
  better guide. Non-linear fits are not made here.

## Checks (all passed)

| Check | Result |
|---|---|
| First returns per site, against `gap_fraction.log` | equal on both grids (e.g. South Dampier 943,902,107); 0.000 % outside the grid |
| Histogram percentiles against `np.percentile` on raw z | max error 0.0043 m |
| Points p98_all against CHM p98 (`nisar_2026` labels, 20 m) | r 0.947–0.992. Points are 0.5–0.8 m lower, because the CHM takes p98 of 50 cm cells that are already p98 |
| Site cover137 against 1 − gap (`gap_summary.csv`, 1.37 m) | difference ≤ 0.008 |
| Alignment: \|ρ\| (mean_all, HV) with the SAR shifted ±1 cell | largest at zero shift on both grids (20 m 0.748 against ≤ 0.716; 30 m 0.787 against ≤ 0.783) |
| **Gate: backscatter against height** (`figures/checks/check_backscatter_vs_height.png`) | HV and HH rise with height and flatten at the top (saturation), on both tracks and grids. HV against mean first-return height ρ 0.71–0.79, which agrees with the earlier 0.73–0.76 |

A bug was found and fixed during the run. Boundless rasterio reads do not apply the nodata mask, so
−9999 entered the 30 m values. The reader now sets nodata to NaN.

## Results

**20 m stack, pooled, against 168 A HV (the best feature for nearly every statistic):**

| Statistic | R² | ρ² |
|---|---|---|
| **cover × p98, canopy** | **0.57** | **0.59** |
| cover × p98, all | 0.55 | 0.58 |
| cover × p90, canopy | 0.54 | 0.58 |
| mean first-return height | 0.49 | 0.56 |
| cover × mean, all | 0.31 | 0.56 |
| cover (1.37 m) | 0.55 | 0.50 |
| p98, all | 0.51 | 0.40 |
| p98, canopy | 0.36 | 0.31 |

**30 m series, pooled, against 168 A HV σ0 on 2026-07-07:** cover × p98 canopy R² 0.60, ρ² 0.66;
mean first-return height 0.51 / 0.62; cover 0.58 / 0.55; p98 all 0.51 / 0.41. The 30 m cells
correlate a little better than the 20 m cells (more looks per cell). *(Superseded 2026-10-22: the
30 m series is deleted and replaced by the 20 m single-date images, 5 sites; see the section
"20 m single-date images and the decomposition against burn age".)*

**Per site, cover × p98 canopy against 168 A HV (20 m), ρ²:** Bunda Bunda 0.83, Bunda Extra 0.73,
South Dampier 0.53, Country Downs 0.14, 9 Mile 0.16, Biota track 0.12 (all 0.59). The other
statistics follow the same site pattern (`r2_by_site.csv`).

**Season (`figures/pooled/season.png`):** on 168 A, ρ² for backscatter is nearly constant from June to October. On
147 D, the first pass (2026-06-24) is weaker, and HH dips on 2026-09-04. Coherence: windowed HH
coherence falls as vegetation rises (ρ down to −0.58, ρ² ≤ 0.34, best on the 12-day 168 A pairs).
HV coherence holds almost no structure information (ρ² ≤ 0.06).

## Findings

1. **The chain is sound.** Backscatter against height gives the expected saturating curve on
   both grids.
2. **Cover × p98 is the best single statistic, but its gain is small.** It improves on top height
   alone by +0.19 ρ² and on cover alone by +0.09. It improves on mean first-return height by only
   +0.03. Most of the SAR signal is in the cover part. Top height alone is the weakest.
3. **`_can` and `_all` give almost the same result for cover × p98**, because p98 is changed by the
   gap only where the gap is > 98 %. For cover × mean, the `_all` form is cover² × height. Its
   linear R² falls to 0.31, but ρ² stays at 0.56, so the problem is the shape, not the ranking.
4. **Most of the pooled correlation is between sites.** Inside 9 Mile, Biota track and Country
   Downs, ρ² is ≤ 0.19 for every main statistic against the four best features. These three sites have a narrow
   range of structure. Bunda Bunda and Bunda Extra carry the relationship. The pooled numbers
   overstate what NISAR resolves within one site.
5. **168 A (ascending) is better than 147 D** for every statistic, in both HV and HH. This agrees with the
   `nisar_2026` feature importances.

## Caveats

- Spatial autocorrelation: no significance tests. The n values are cell counts, not independent samples.
- The 20 m stack and the 30 m series differ in grid, dates, γ0 against σ0, and compositing.
- EPSG:7851 is used on the EPSG:32751 grid without reprojection (datum offset < 2 m).
  ⚠ CRS 2026-11-03: relabelling is withdrawn (user). Reprojection is required to match grids. See `G:\python\crs_audit\CHECKPOINT.md`.
- Coherence is not speckle-filtered (user decision, 2026-10-20).
- The best feature was selected on the same data the ρ² is reported for, so it is slightly optimistic.

## Cover cutoff: 1.37 m, 2 m, 5 m (2026-10-21)

This section includes Bunda Bunda (6 sites) and the 30 m GSLC series. The next section replaces
it: 5 sites, 20 m only, and 5 cutoffs. `cutoff_compare.csv` / `figures/pooled/cutoff_compare.png` now hold the 5-site run.

Question: do cover and cover × height correlate better with backscatter when cover is the share of
first returns at or above 2 m or 5 m, not 1.37 m? Script: `cover_cutoff_vs_sar.py` (log
`cover_cutoff_vs_sar.log`). Outputs: `cutoff_compare.csv`, `figures/pooled/cutoff_compare.png`.

**Method.** `structure_grids.py` now also writes each cutoff-dependent statistic at 2 m and 5 m
(suffix `_h2` / `_h5`: cover, the canopy heights, cover × height). The canopy heights at 2 m / 5 m
use the first returns at or above that cutoff. Every cutoff uses the same cells (fill ≥ 0.9;
cover, cover × height and the references finite): 34,525 cells at 20 m, 15,225 at 30 m. The
canopy-only heights (p98 / p90 / mean, canopy) need 20 returns above the cutoff. They use their own
cells, which at 5 m excludes about 1,500 cells with little vegetation above 5 m (20 m grid).
SAR: backscatter only (no coherence).

**Speckle filter (confirmed).** All the backscatter in this report and in Stage 2 is Refined Lee
5 × 5. The 20 m stack is `feature_stack_2026.tif` with `gslc_grid: 20m_rflee5` (the unfiltered
`feature_stack_2026_nospeckle.tif` was not used). The 17 per-pass files in `GSLC_dcmp_rflee5` have
the tag `speckle=rflee5x5`. The script asserts both. For reference, the unfiltered stack gives ρ²
0.46 (cover) and 0.54 (cover × p98 canopy) against 168 A HV, against 0.49 / 0.59 filtered.
Coherence is not filtered (decision of 2026-10-20).

**Checks.** The rebuilt grids reproduce the 1.37 m values in `structure_sar_stats.csv` (pooled R²
and ρ² to 1e-3, the same n). Cover falls as the cutoff rises in every cell. First-return counts
are unchanged, and the largest percentile error is 0.0067 m.

**Pooled ρ² (R²) against 168 A HV:**

| Statistic | 20 m stack: 1.37 m | 2 m | 5 m | 30 m mean: 1.37 m | 2 m | 5 m |
|---|---|---|---|---|---|---|
| cover | 0.49 (0.55) | 0.50 (0.54) | 0.46 (0.43) | 0.53 (0.57) | 0.56 (0.58) | 0.52 (0.48) |
| **cover × p98, canopy** | **0.59 (0.56)** | 0.58 (0.54) | 0.48 (0.41) | **0.65 (0.61)** | 0.65 (0.59) | 0.55 (0.46) |
| cover × p98, all | 0.58 (0.55) | 0.57 (0.52) | 0.48 (0.39) | 0.65 (0.59) | 0.64 (0.57) | 0.54 (0.44) |
| cover × p90, canopy | 0.57 (0.54) | 0.57 (0.52) | 0.48 (0.40) | 0.65 (0.59) | 0.64 (0.58) | 0.54 (0.46) |
| cover × mean, canopy | 0.55 (0.48) | 0.54 (0.47) | 0.47 (0.39) | 0.62 (0.53) | 0.61 (0.52) | 0.53 (0.44) |
| cover × mean, all | 0.55 (0.31) | 0.55 (0.29) | 0.50 (0.24) | 0.62 (0.34) | 0.62 (0.32) | 0.57 (0.26) |
| p98, canopy | 0.31 (0.36) | 0.30 (0.35) | 0.26 (0.26) | 0.31 (0.35) | 0.31 (0.35) | 0.28 (0.29) |
| mean first-return height (no cutoff) | 0.55 (0.49) | | | 0.63 (0.53) | | |

The 30 m single pass 2026-07-07 gives the same pattern (cover × p98 canopy 0.65 / 0.66 / 0.51).
The full tables, with p90 / mean canopy and the best feature per statistic, are in the log.

**Per site, ρ² against 168 A HV (20 m), cover × p98 canopy at 1.37 / 2 / 5 m:** Bunda Bunda
0.83 / 0.80 / 0.68, Bunda Extra 0.73 / 0.72 / 0.65, South Dampier 0.52 / 0.52 / 0.35, 9 Mile
0.16 / 0.18 / 0.18, Biota track 0.12 / 0.17 / 0.16, Country Downs 0.14 / 0.16 / 0.17. Cover alone
gives the same pattern. At 30 m, Biota track rises more (cover 0.09 / 0.26 / 0.34).

**Findings.**

1. **A higher cutoff does not improve the pooled correlation.** 2 m is the same as 1.37 m (±0.03 ρ²;
   cover alone +0.01 to +0.03). 5 m is worse for every combined statistic (−0.05 to −0.14 ρ²;
   cover × p98 canopy −0.10 to −0.14) and for cover alone (−0.01 to −0.06).
2. **Cover × p98 (canopy) at 1.37 m or 2 m stays the best statistic.** Its gain over cover alone is
   +0.08 to +0.12 ρ² at 1.37 m and 2 m, but only +0.02 to +0.03 at 5 m. So at 5 m the height
   part adds almost nothing.
3. **At 5 m the sites with tall vegetation lose correlation** (South Dampier −0.17, Bunda Bunda
   −0.15). A 5 m cover ignores the 1–5 m shrub layer, which also gives HV backscatter.
4. **The three weak sites gain a little from a higher cutoff** (+0.02 to +0.10 ρ² at 20 m; Biota
   track +0.25 at 30 m), but their ρ² stays ≤ 0.34. A possible cause: at these sites, low
   vegetation between 1.37 and 2 m adds variation that HV does not see. This was not tested.
5. At 5 m the best feature changes from HV to HH (168 A) for cover and the combined statistics.
   Its pooled ρ² is not higher than HV at 1.37 m.

Conclusion: keep cover at 1.37 m (or 2 m: no difference). Do not use 5 m.

## Cover cutoff 0.2 m to 5 m, Bunda Bunda excluded (2026-10-21)

Question: do cover and cover × height correlate better with backscatter when the cover cutoff is
lower (0.2 m, 1 m)? Same script and outputs as the previous section (`cover_cutoff_vs_sar.py`,
`cutoff_compare.csv`, `figures/pooled/cutoff_compare.png`). Two decisions of the user change the scope from this
section on:
- **Bunda Bunda is excluded** from the structure × SAR analysis. The vegetation floats on a freshwater
  spring, so the ground moves, and the water content and species composition are different.
  `EXCLUDE_SAR` in `structure_grids.py`. 5 sites.
- **Only the 20 m UTM stack** (`feature_stack_2026.tif`, EPSG:32751; GCOV γ0 HH / HV and the GSLC
  decomposition on the GCOV grid, Sep–Oct, Refined Lee 5 × 5). The 30 m EPSG:3577 GSLC series is not
  used. The main feature, 168 A HV, is GCOV.

**Method.** `structure_grids.py` now also writes the cutoff-dependent statistics at 0.2 m (`_h02`)
and 1 m (`_h1`): 56 bands. The 5 sites were rebuilt, and the checks passed again (the same
first-return counts, 0.000 % outside, the largest percentile error 0.0072 m). The 1.37 m values of
all 6 sites still reproduce `structure_sar_stats.csv`. The cell rules are the same as in the
previous section: 32,926 cells, the same at every cutoff. The canopy-only heights use their own
cells (32,890 at 0.2 m, 31,650 at 5 m).

**No 0 m cutoff.** Heights below 0 are clipped to 0, so at 0 m every first return counts as
cover: cover is 1 in every cell, and each cover × height is the plain height (a reference that we
already have). A strict z > 0 does not help. In a sample of about 17 million first returns each
from 9 Mile and South Dampier, the ground-class returns lie at ±0.06 m (p1 to p99), 26–30 % of
first returns are below 0 m and 33 % are at 0–0.05 m. Cover above 0 m would show the scatter of
the ground returns about the DEM, not vegetation. Above 0.2 m, only 0.1–0.6 % of returns are
ground class. Thus 0.2 m is the lowest useful cutoff (it is also the low cutoff of
`run_metrics.ps1`).

**Effect of Bunda Bunda (1.37 m, 168 A HV, before it was removed).** Bunda Bunda has 1,610 of
34,744 cells (5 %).

| Statistic | ρ² 6 sites | ρ² 5 sites | Bunda Bunda ρ² | Bunda Bunda offset |
|---|---|---|---|---|
| cover | 0.50 | 0.47 | 0.80 | −1.7 dB |
| cover × p98, canopy | 0.59 | 0.57 | 0.83 | −2.0 dB |
| mean first-return height | 0.56 | 0.54 | 0.85 | −2.5 dB |

R² changes by −0.03 to +0.01. The offset is the mean residual of the Bunda Bunda cells from a
linear fit on the other 5 sites. The residual SD of those sites is 1.35–1.44 dB. Thus Bunda Bunda
has a strong relationship of its own, but its HV is 1.2–1.7 SD lower than at the other sites with
the same structure. Its strong internal relationship increased the pooled ρ² by 0.02–0.03. A
site-specific relationship agrees with the decision to treat it as a special case. All earlier
sections include Bunda Bunda, so their pooled ρ² are about 0.02–0.03 higher than a 5-site run.

**Pooled ρ² (R²) against 168 A HV, 5 sites, 20 m:**

| Statistic | 0.2 m | 1 m | 1.37 m | 2 m | 5 m |
|---|---|---|---|---|---|
| cover | 0.42 (0.48) | 0.45 (0.50) | 0.46 (0.51) | 0.48 (0.52) | 0.44 (0.42) |
| **cover × p98, canopy** | 0.56 (0.56) | 0.56 (0.56) | **0.57 (0.56)** | 0.56 (0.54) | 0.46 (0.40) |
| cover × p98, all | 0.55 (0.55) | 0.56 (0.54) | 0.56 (0.54) | 0.55 (0.52) | 0.45 (0.38) |
| cover × p90, canopy | 0.55 (0.54) | 0.55 (0.54) | 0.55 (0.53) | 0.55 (0.52) | 0.46 (0.39) |
| cover × mean, canopy | 0.53 (0.49) | 0.53 (0.49) | 0.53 (0.49) | 0.52 (0.48) | 0.44 (0.39) |
| cover × mean, all | 0.53 (0.38) | 0.53 (0.37) | 0.53 (0.36) | 0.52 (0.35) | 0.48 (0.28) |
| p98, canopy | 0.31 (0.37) | 0.31 (0.36) | 0.30 (0.36) | 0.30 (0.36) | 0.26 (0.26) |
| mean, canopy | 0.30 (0.33) | 0.27 (0.30) | 0.26 (0.29) | 0.24 (0.27) | 0.19 (0.18) |
| mean first-return height (no cutoff) | | | 0.53 (0.49) | | |
| p98, all (no cutoff) | | | 0.37 (0.44) | | |

For reference, the unfiltered stack gives 0.43 (cover) and 0.52 (cover × p98 canopy) at 1.37 m.

**Per site, ρ² against 168 A HV, at 0.2 / 1 / 1.37 / 2 / 5 m:**

| Site | cover × p98, canopy | cover |
|---|---|---|
| Bunda Extra | 0.70 / 0.72 / 0.72 / 0.71 / 0.64 | 0.66 / 0.67 / 0.67 / 0.66 / 0.63 |
| South Dampier | 0.51 / 0.52 / 0.52 / 0.52 / 0.35 | 0.49 / 0.51 / 0.52 / 0.52 / 0.36 |
| 9 Mile | 0.13 / 0.16 / 0.16 / 0.18 / 0.18 | 0.12 / 0.15 / 0.16 / 0.18 / 0.19 |
| Country Downs | 0.12 / 0.13 / 0.14 / 0.16 / 0.17 | 0.07 / 0.09 / 0.09 / 0.13 / 0.19 |
| Biota track | 0.07 / 0.10 / 0.12 / 0.17 / 0.16 | 0.03 / 0.06 / 0.09 / 0.17 / 0.17 |

**Findings.**

1. **A lower cutoff does not improve the correlation.** The combined statistics are the same at
   0.2, 1, 1.37 and 2 m (within 0.01 ρ²). Cover alone becomes worse as the cutoff goes down: 0.48
   (2 m), 0.46 (1.37 m), 0.45 (1 m), 0.42 (0.2 m).
2. **Cover × p98 (canopy) at 1.37 m is still the best statistic** (ρ² 0.57, R² 0.56). Its gain
   over cover alone increases as the cutoff goes down (+0.08 at 2 m, +0.11 at 1.37 m, +0.14 at
   0.2 m), because cover alone becomes worse, not because the product becomes better. The gain over
   the mean first-return height, which needs no cutoff, is small (+0.04).
3. **The weak sites cause the fall of cover at low cutoffs.** At Biota track, cover ρ² is 0.03 at
   0.2 m and 0.17 at 2 m. At Country Downs it is 0.07 at 0.2 m and 0.19 at 5 m. At Bunda Extra and
   South Dampier, cover is almost constant from 0.2 m to 2 m. Thus at the weak sites, low
   vegetation (below about 2 m) adds variation that HV does not show. This supports the possible
   cause in finding 4 of the previous section. It does not explain why these sites stay weak
   (ρ² ≤ 0.19 at every cutoff).
4. The canopy-only heights stay weak at every cutoff (ρ² ≤ 0.31). The best feature is HV for the
   combined statistics from 0.2 m to 2 m, and HH at 5 m (as before).
5. **Removing Bunda Bunda lowers the pooled ρ² by 0.02–0.03.** It does not change a conclusion.

Conclusion: keep cover at 1.37 m. 2 m is equal for the combined statistics and slightly better for
cover alone (+0.02). Do not use a cutoff below 1.37 m, or 5 m.

## Per-site scatter, cover × p98 canopy (2026-10-21)

Script `site_scatter.py` (log `site_scatter.log`, about 1 min). Figures in
`outputs\structure_sar\figures\by_site\`: `c_p98_can_<site>.png` (one per site) and
`c_p98_can_site_medians.png` (the 5 site medians on the same axes). Statistic: cover × p98 (canopy),
1.37 m cutoff. 5 sites, 20 m stack, cells with fill ≥ 0.9 (`load_grid`). Each panel shows the cells,
the median in 20 equal bins of cover × p98 (bins with ≥ 30 cells) and the 25th–75th percentile band.
The axes are the same in all files (the widest per-site 0.5–99.5 percentile range), so the sites
can be compared directly. The per-site ρ is the same as in `structure_sar_stats.csv` (asserted).

**The 3 SAR parameters.**
- **168 A HV γ0** (GCOV): the best pooled feature; volume scattering.
- **168 A HH γ0** (GCOV): the best feature at 9 Mile and South Dampier.
- **147 D HV γ0** (GCOV, descending track): the other look direction. It shows whether the level of a
  site depends on the look direction.

DpRSI (GSLC decomposition) was the first choice for the third parameter. It was replaced, because it
is exactly 0 in 54 % of the cells at Biota track and 52 % at Bunda Extra (0–9 % at the other sites).
`gslc_decomposition.py` clips |s1| at the scene 95th percentile and divides by it, so
DpRSI = (1 − H)·√(1 − s1n²) is 0 where |s1| is above that percentile. The 0 floor hides the structure
signal at high structure. Pu was not used, because it is almost the same as HV (ρ within 0.03 at
every site).

**ρ (R²) against cover × p98 canopy:**

| Site | n | 168 A HV | 168 A HH | 147 D HV | Cells < 1 m | ρ 168 A HV, cells ≥ 1 m | 168 A / 147 D HV at 2–4 m (dB) |
|---|---|---|---|---|---|---|---|
| Bunda Extra | 3,710 | 0.85 (0.65) | 0.83 (0.62) | 0.77 (0.58) | 12 % | 0.80 | −16.1 / −16.9 |
| South Dampier | 8,658 | 0.72 (0.48) | 0.75 (0.51) | 0.64 (0.38) | 32 % | 0.48 | −17.8 / −18.4 |
| 9 Mile | 4,605 | 0.40 (0.18) | 0.41 (0.18) | 0.33 (0.11) | 9 % | 0.35 | −17.6 / −18.0 |
| Country Downs | 4,952 | 0.38 (0.13) | 0.33 (0.10) | 0.33 (0.10) | 1 % | 0.37 | −16.2 / −16.8 |
| Biota track | 11,194 | 0.35 (0.13) | 0.28 (0.09) | 0.30 (0.10) | 0 % | 0.34 | −15.3 / −16.0 |
| All 5 sites | 33,119 | 0.76 (0.56) | 0.73 (0.52) | 0.72 (0.51) | | | |

"Cells < 1 m": the share of cells with cover × p98 below 1 m. The last column is the median HV of
the cells with cover × p98 from 2 m to 4 m. Pooled n is 33,119, not the 32,926 of the cutoff
section, because only the 1.37 m statistic must be finite here.

**Findings.**

1. **The sites have different HV levels at the same structure.** At cover × p98 2–4 m, the median
   168 A HV is −15.3 dB at Biota track and −17.8 dB at South Dampier: a range of 2.5 dB. The order is
   the same on the descending track (−16.0 to −18.4 dB). Thus the level is not an effect of the look
   direction. It is a property of the site (for example soil moisture, surface roughness, species or
   the stem density). 9 Mile and South Dampier are low; Biota track, Country Downs and Bunda Extra are
   high. Bunda Bunda (excluded) had the same type of offset (−2.0 dB).
2. **The strong sites have the steep low-structure part of the curve.** HV rises fast from 0 to
   about 2 m of cover × p98, then slowly. South Dampier has 32 % of its cells below 1 m. Without them,
   its ρ falls from 0.72 to 0.48. Bunda Extra stays strong without them (0.80), because its
   structure range is large (to about 12 m). Biota track and Country Downs have almost no cells below
   1 m, so all their cells are on the flat part of the curve.
3. **The weak sites are weak in all 3 parameters** (ρ 0.28–0.41). HH is the best at 9 Mile and
   South Dampier; HV is the best at the other 3. On the descending track, ρ is lower than on 168 A at every
   site.
4. **The site offsets lower the pooled relationship.** Within a site, the slope above 2 m is
   similar at 4 sites (it is steeper at Bunda Extra), but the curves are displaced by up to 2.5 dB. A per-site offset (or a
   covariate that explains it) could increase the pooled R².

### Canopy height alone, p98 canopy (added 2026-10-21)

The same figures now have a second row: p98 of the canopy first returns (z ≥ 1.37 m) alone, without
cover, against the same 3 parameters (2 × 3 panels; the file names did not change). p98 canopy is
NaN where a cell has fewer than 20 canopy first returns. This removes 98 cells at Bunda Extra and no
cells at the other sites (pooled n 33,021). The x axis starts at 0 as in the top row; p98 canopy is
always ≥ 1.37 m.

**ρ (R²) against p98 canopy, with the cover × p98 ρ for 168 A HV for comparison:**

| Site | n | 168 A HV | 168 A HH | 147 D HV | Cover × p98, 168 A HV |
|---|---|---|---|---|---|
| Bunda Extra | 3,612 | 0.67 (0.48) | 0.70 (0.51) | 0.61 (0.40) | 0.85 |
| South Dampier | 8,658 | 0.47 (0.28) | 0.48 (0.30) | 0.46 (0.27) | 0.72 |
| 9 Mile | 4,605 | 0.21 (0.05) | 0.24 (0.06) | 0.16 (0.03) | 0.40 |
| Country Downs | 4,952 | 0.20 (0.04) | 0.29 (0.09) | 0.26 (0.07) | 0.38 |
| Biota track | 11,194 | 0.17 (0.04) | 0.27 (0.09) | 0.18 (0.04) | 0.35 |
| All 5 sites | 33,021 | 0.56 (0.37) | 0.60 (0.41) | 0.55 (0.37) | 0.76 |

**Findings.**

1. **Height alone is weaker than cover × height at every site and in all 3 parameters.** For
   168 A HV, ρ falls by 0.17–0.25 at each site, and the pooled ρ² falls from 0.57 to 0.31. Most of
   the signal comes from cover.
2. **HH is the best parameter against height alone at all 5 sites** (pooled ρ 0.60 against 0.56 for
   HV). Against cover × p98, HV is the best at 3 sites. A possible cause: HH has more trunk and
   trunk–ground scattering, which follows height; HV follows the canopy volume, which follows cover.
   This is a hypothesis. It is not tested.
3. **HV becomes flat above a p98 canopy of about 8 m at 4 sites.** Only Bunda Extra continues to
   rise, to about 14 m. At Biota track and Country Downs most cells have p98 canopy from 6 m to 14 m,
   which is on the flat part. At these sites the trees are tall but the canopy is open, so height does
   not show the structure that HV responds to.
4. **The site offsets are the same as in the top row**: 9 Mile and South Dampier are low and Biota
   track is high, at the same height. Height alone does not remove the offsets.

## Site offsets: fire history, greenness and undergrowth (2026-10-21)

Exploratory scripts and logs: `exploratory\site_offsets_2026-10-21\` (one script per question; the
log beside each script has the full tables). These are first results, not a final analysis. The
plan for the final analysis is in `CHECKPOINT_structure_sar.md`.

**Main result: the two dark sites are the two sites that burn most often and burnt most recently.**
9 Mile and South Dampier burnt in 3 of the 4 years 2023–2026, and all their cells burnt in
May–July 2026 (NAFI), 3–5 months before the SAR and LiDAR. Within Country Downs, the cells that
burnt in 2026 are about 1 dB darker at the same structure than the other cells of that site.

### The offset and how it is measured

The **balanced offset** of a site is its median backscatter minus the 5-site mean, in the 0.5 m
bins of cover × p98 canopy (0.5–6 m) where every site has ≥ 30 cells, averaged over those bins (9
bins, 1–5.5 m). Each site has the same weight in each bin. (An offset from the pooled curve gives a
smaller range, 1.5 dB, because the pooled curve follows the sites with the most cells.)

| Site | 168 A HV | 147 D HV | 168 A HH | 147 D HH |
|---|---|---|---|---|
| Biota track | +1.21 | +1.06 | +0.93 | +1.00 |
| Bunda Extra | +0.54 | +0.47 | +0.29 | +0.61 |
| Country Downs | +0.39 | +0.40 | +0.19 | −0.06 |
| 9 Mile | −0.97 | −0.77 | −0.31 | −0.45 |
| South Dampier | −1.18 | −1.16 | −1.10 | −1.09 |
| Range (dB) | 2.39 | 2.22 | 2.03 | 2.09 |

- **Two groups in HV** (the user's observation from the figures): 9 Mile and South Dampier are about
  1.5–3 dB below the other three sites. HH is a more continuous spread, because 9 Mile is dark in HV
  but near the mean in HH. South Dampier is dark in both.
- **The groups do not depend on the structure statistic.** They are the same against cover
  alone, p98 canopy alone and mean first-return height (`shape.log`). For example, 168 A HV against
  p98 canopy is −1.48 (9 Mile) and −1.72 (South Dampier) against +0.49 to +1.46.
- **The offsets of the 4 bands agree** (Pearson r 0.88–0.98 between bands).
- **The offsets are stable over time.** We computed them for each of the 17 single-date
  20 m GCOV files from June to October 2026 (`G:\python\nisar\geotiff_rflee5\`). The standard
  deviation of each site's offset is 0.1–0.3 dB, and the order is the same on every complete date.
  (24 Jun 147 D covers only 74 % of the cells and does not include South Dampier.) The stack bands
  pool 2 dates of one track (mean of linear γ0); the tracks are never pooled.

### Causes that do not explain the offset

- **Location**: the two dark sites are the most southerly (South Dampier, −17.78°) and the most
  northerly (9 Mile, −17.09°), and the most westerly and the most easterly. Spearman ρ of offset
  against longitude or latitude is 0.1 (`incidence_location.log`).
- **Incidence angle** (GCOV metadata, Sep–Oct passes, h = 0 m): it changes by only 1.3° (147 D) and
  2.1° (168 A) across the sites. An incidence effect would be different on the two tracks, which
  look from opposite sides. The offsets are the same on both tracks.
- **A calibration step between SAR frames**: frames divide the scene by latitude, but the most
  northerly and the most southerly sites are both dark.
- **Land system** (DPIRD rangelands, `landsys.log`): 9 Mile is Yeeda; South Dampier, Biota track and
  Country Downs are Wanganut; Bunda Extra is 68 % Wanganut and 32 % Carpentaria coastal flats. The
  soil map (pindan / sand) also does not match (user).
- **Canopy shape at the same structure** (mean / p98 canopy; p90 / p98): no group pattern. 9 Mile
  is taller and sparser than the mean; South Dampier is near the mean.
- **DEM microrelief** (standard deviation about a plane per 20 m cell, 0.5 m DEM): no match.
- **Slope** matches the site order exactly (South Dampier 0.4°, 9 Mile 0.6°, Country Downs 0.8°,
  Bunda Extra 1.1°, Biota track 1.6°). These slopes are too small to change γ0 by 2 dB, and slope
  has no effect within a site (offset range 2.39 → 2.29 dB when it is removed). With 7 statistics
  tested on 5 sites, an exact order by chance is not unlikely.

### Fire history (NAFI)

NAFI fire scar polygons (Sentinel-2 based for northern WA) for 2023–2026: 2023 and 2024 from
firenorth.org.au (the same data as `G:\python\fire_class_forest_stats\NAFI\fire scars\`), 2025 from
`G:\Karajarri\Karajarri Fire Scars\`, and 2026 to date (downloaded 2026-10-21, updated that day) in
`G:\Veg and Soil Mapping\NAFI_firescars\`. Share of each site's cells inside a scar, with the month:

| Site | 2023 | 2024 | 2025 | 2026 | Fires 2023–26 (most cells) |
|---|---|---|---|---|---|
| 9 Mile | 6 % (Sep) | 100 % (Aug) | 100 % (Aug) | 100 % (Jul) | 3 |
| South Dampier | 56 % (May–Jun) | 37 % (Aug) | 82 % (Jun) | 100 % (May, Jul) | 3 |
| Biota track | 47 % (May, Jul) | 48 % (May) | 39 % (May) | 18 % (May) | 1–2 |
| Country Downs | 59 % (Jun, Oct) | 0 | 30 % (May) | 13 % (Jan, May) | 1 |
| Bunda Extra | 20 % (Sep) | 0 | 38 % (Aug) | 0 | 0 (59 % of cells) |

No scar is after September 2026, so no fire happened between the SAR passes and the LiDAR.

**Balanced offset by site and burn age** (168 A HV; 168 A HH in brackets). Each site and burn age is
one group. The 6 common 0.5 m bins of cover × p98 (0.5–5 m) are those where every group has ≥ 30 cells:

| | Burnt < 1 y | 1–2 y | 2–4 y | Not burnt 2023–26 |
|---|---|---|---|---|
| 9 Mile | −1.41 (−0.64) | | | |
| South Dampier | −1.49 (−1.30) | | | |
| Country Downs | **−0.80 (−0.65)** | +0.42 (+0.41) | −0.01 (−0.27) | +0.14 (−0.19) |
| Biota track | +0.77 (+0.74) | +1.11 (+1.14) | +0.66 (+0.30) | |
| Bunda Extra | | +0.32 (+0.37) | | +0.29 (+0.09) |

1. **The two dark sites burnt almost every year, the last time within 5 months of the SAR**, and
   9 Mile burnt in the late dry season (August 2024 and 2025, July 2026). The three bright sites
   burnt less often, mostly in May (early dry season), and most of their cells did not burn in 2026.
2. **Within Country Downs, the cells that burnt in 2026 are about 1 dB darker** in HV than the
   other cells of the site, at the same structure. This is the only within-site comparison with
   enough cells in each burn age.
3. **Biota track is an exception**: its cells that burnt in May 2026 are still bright (+0.77,
   against +1.11 for cells burnt 1–2 years before). A possible cause is that early dry season fires
   are cooler and patchier. This is not tested.
4. **South Dampier became about 0.4 dB darker in HV after its July 2026 fire.** The mean 168 A HV
   offset was −0.71 dB on the June–July dates and −1.08 dB on the August–October dates; on 147 D,
   −0.78 and −1.08 dB. HH did not change. **9 Mile shows no step** at its mapped July 2026 fire.
   It burnt in August 2025, so there was possibly little fuel to burn again.

Possible mechanisms (not tested): a fire removes leaves, fine branches and the grass layer, which
contribute to volume scattering (HV), and it can leave the soil surface drier and smoother.

Limits: the NAFI polygons are coarse. They show all of 9 Mile and South Dampier as burnt, but the
user saw burnt and unburnt patches in each flight area. With 2 dark sites, how often and how
recently a site burnt cannot be separated. Sentinel-2 burn maps for each fire need much manual work,
so they are not made now (user, 2026-10-21).

### Greenness (Sentinel-2)

NDVI, EVI and BSI from the stack (geomedian 2026-09-20 to 10-10, 20 m), at the same structure:

- **Between sites, greenness follows the offsets**: for NDVI and EVI, Spearman ρ 0.9 over the 5
  sites and Pearson r 0.89–0.99 in all 4 bands. South Dampier is the least green (NDVI −0.063
  against the 5-site mean; site median 0.25, other sites 0.33–0.41), Biota track the most (+0.039).
  This agrees with the fire result: recent fires remove or scorch the leaves.
- **Within a site, greenness does not predict the residual** (ρ −0.29 to +0.26; the sign is not
  the same at all sites). Greenness marks the difference between sites. It is not the direct cause.
- BSI is highest at South Dampier (+0.043 at the same structure). The user saw almost no bare soil
  at the sites, so this probably shows ash or burnt ground after the July fire. BSI is not used.

### Undergrowth (first returns and 0.5 m voxels)

Hypothesis (user): undergrowth below 1.37 m, which the canopy statistics ignore, causes the offset.
**Result: undergrowth is lower at the two dark sites, but within a site it does not change the
backscatter. It is a marker of the site difference, not the cause.** The direct measurement of
undergrowth is not continued now (user, 2026-10-21).

Statistics:
- **First returns** (from the existing struct bands, `undergrowth_first.log`): the share of first
  returns at 0.2–1.37 m among the first returns that pass the canopy (the canopy gaps). In cells
  with gap ≥ 0.7 it is 2–9 % (Bunda Extra 0.019, 9 Mile 0.035, Biota track 0.040, South Dampier
  0.046, Country Downs 0.085). It does not follow the offsets (ρ about 0.1 over the 5 sites).
- **Voxels** (`undergrowth_vox.log`): `lasvoxel64` (LAStools 260914) on each site's
  `_denoise.las`, all returns except withheld and class 7 / 18, z < 1.7 m, z shifted by −0.2 m,
  `-step 0.5`. The voxel layers are < 0.2 m (ground), 0.2–0.7, 0.7–1.2 and 1.2–1.7 m, and the point
  count is in the intensity. Output: `outputs\voxels\<site>_vox0.5_z0.2-1.7.laz` (74 MB in total,
  16 min on 1 core). The statistic is the share of 0.5 m columns with a voxel at 0.2–1.2 m, among
  the columns where a return reached 1.2 m or lower. The canopy hides almost nothing: in nearly all
  columns a return reaches below 1.2 m.

| | 9 Mile | South Dampier | Biota track | Bunda Extra | Country Downs |
|---|---|---|---|---|---|
| Voxel undergrowth, all cells | 0.21 | 0.23 | 0.25 | 0.29 | 0.41 |
| Voxel undergrowth, gap ≥ 0.7 | 0.23 | 0.22 | 0.24 | 0.11 | 0.45 |
| At the same structure (against the 5-site mean) | −0.10 | −0.05 | −0.01 | +0.01 | +0.14 |

- Between the groups, it agrees with the hypothesis: the dark pair has the least undergrowth at the
  same structure. Within the bright group it does not: Country Downs has the most undergrowth but
  the smallest offset of the three.
- Within a site, the residual from the pooled curve does not follow undergrowth (ρ −0.31 to +0.17;
  the slope is about 0.1 dB from 0 to 1). A 2 dB group step from a 0.1 difference would need
  about 20 dB per unit. When the undergrowth term is removed, the offset range does not change
  (2.39 → 2.38 dB).
- Low undergrowth at the dark sites agrees with frequent fire.

## Fire history against the site offsets, NAFI (2026-10-21)

Script `fire_vs_sar.py` (log `fire_vs_sar.log`, about 3 min). Outputs `outputs\structure_sar\fire\`
(`fire_cells.parquet`, `fire_sites.csv`, `fire_offsets.csv`, `fire_models.csv`, `fire_steps.csv`)
and `figures\fire\` (`burn_maps.png`, `offsets_by_burn.png`, `step_test.png`,
`c_p98_can_mixed_sites_by_burn.png`). NAFI only (user).

`c_p98_can_mixed_sites_by_burn.png`: median backscatter (168 A HV, 168 A HH, 147 D HV) against cover × p98
canopy for each burn class at Biota track and Country Downs, the two sites with all three classes. Its axes and bins
are those of `by_site\c_p98_can_site_medians.png`. At Country Downs, the burnt-2026 curve is about 1 dB
below the other two along the whole structure range (a level shift, not a different slope), and the
burnt-2025 curve is the highest. At Biota track the three curves are within about 0.5 dB; burnt 2025
is the highest, and in HH the unburnt cells are the lowest.

Fire data per 20 m cell, sampled at the cell centre: the NAFI 250 m rasters 2016–2025
(`G:\python\fire_class_forest_stats\NAFI`: number of fires, early (Jan–Jul) and late (Aug–Dec) fires,
time since burnt) and the NAFI polygons of 2025 and 2026 (month of the fire). Burn class: **burnt
2026** (all 2026 fires at the sites are in January–July, before the SAR), **burnt 2025** (not 2026),
**unburnt 2025–26**. NAFI is coarse: it shows all of 9 Mile and South Dampier as burnt in 2026.
Offsets are balanced offsets (as above), here with site × burn class as the groups (8 common bins).

**Site level (descriptive; 5 sites):**

| Site | Offset 168 A HV | Fires 2016–25 | Late fires 2016–25 | Burnt 2026 | Burnt 2025 (not 2026) |
|---|---|---|---|---|---|
| Biota track | +1.21 | 3 | 1 | 18 % (May) | 37 % |
| Bunda Extra | +0.54 | 2 | 2 | 0 | 38 % |
| Country Downs | +0.39 | 3 | 2 | 13 % (May) | 30 % |
| 9 Mile | −0.97 | 5 | 4 | 100 % (Jul) | (all burnt again in 2026) |
| South Dampier | −1.18 | 4 | 2 | 100 % (May 19 %, Jul 81 %) | (all burnt again in 2026) |

**Offset by site and burn class** (168 A HV; the 4 bands in `fire_offsets.csv`):

| Site | Burnt 2026 | Burnt 2025 | Unburnt 2025–26 |
|---|---|---|---|
| South Dampier | −1.47 | | |
| 9 Mile | −1.32 | | |
| Country Downs | **−0.80** | +0.47 | +0.04 |
| Bunda Extra | | +0.26 | +0.26 |
| Biota track | +0.76 | +1.14 | +0.67 |

Findings:

1. **Within Country Downs, the cells burnt in 2026 are darker than the other cells** by 0.8–1.3 dB in
   HV (against the unburnt and the 2025-burnt cells) and by 0.4–1.2 dB in HH. The sign is the same in
   all 4 bands. Country Downs is the only site with all three classes. Its 2026 fire was in May.
2. **At Biota track, NAFI shows no such effect**: its cells burnt in May 2026 (+0.76) are as bright
   as the unburnt cells (+0.67).
3. **Cells that burnt in 2025 and not in 2026 are the brightest part** of Biota track (+0.47 against
   unburnt) and Country Downs (+0.43). At Bunda Extra (fire in August 2025) they are the same as the
   unburnt cells in 168 A HV, and 0.4 dB darker in 147 D HV. A burn 12–17 months before does not make
   a cell dark. A possible cause is regrowth after the fire (not tested).
4. **The burnt-2026 parts of the bright sites are not as dark as the dark sites.** Country Downs
   burnt-2026 cells are −0.80; 9 Mile and South Dampier are −1.32 and −1.47.
5. **The 2026 fire term removes about a third of the between-site range.** A pooled fit (no site
   term) of the residual on burnt 2026 has β −0.89 dB (168 A HV) and reduces the offset range from
   2.39 to 1.63 dB (147 D HV: 2.22 → 1.49). Within sites, β is −0.32 dB (range 2.12 dB). The number of
   fires 2016–2025 has no effect within sites (β 0.05 dB per fire); at 250 m it hardly changes inside
   a site. After the 2026 term is removed, Biota track is still +0.93 and South Dampier −0.69 dB.
6. **Step test** (17 single dates, `step_test.png`):
   - **9 Mile has no step at its July 2026 fire.** It was already dark in June (−0.75 dB, 168 A HV).
     Its last fire before that was in August 2025.
   - **South Dampier became gradually darker after July**: from −0.38 / −0.48 dB (June–July) to
     −0.70 / −0.74 dB (October), 168 A HV / 147 D HV. HH did not change.
   - The Country Downs burnt-2026 cells (fire in May) are dark on all dates from June.
   - Biota track and Bunda Extra are flat.

**Interpretation.** A fire in the same year makes the backscatter at the same structure about 0.8 dB
lower where this can be tested inside one site (Country Downs). The dark sites are the two sites that
burnt completely in 2026, also burnt in 2025, and have the most fires (and the most late fires,
9 Mile) since 2016. Recent fire explains about a third of the between-site range. The rest can come
from the fire regime (frequent and late fires over many years), which 5 sites cannot separate from
other site properties. The 9 Mile result (dark before its 2026 fire) agrees with a long-term effect.
**Splitting sites:** the burn classes inside Country Downs differ by up to 1.3 dB, and inside Biota
track by up to 0.5 dB. These are large compared with the between-site range, so a burn-class split
(or burn class as a term) is justified for these two sites. Bunda Extra's two classes are the same in
168 A HV.

### Exploratory: Sentinel-2 burn classes at 20 m (not checked)

Separate from the NAFI analysis. Scripts `s2_nbr_series.py` (NBR series and burn metrics; log
`s2_nbr_series.log`) and `s2_fire_vs_sar.py` (the same tests with Sentinel-2 classes; log
`s2_fire_vs_sar.log`). Outputs `outputs\structure_sar\s2_fire\`, `figures\s2_fire\`. **The user
said that good Sentinel-2 burn maps need much manual work; without it they contain artefacts. This
mapping has no manual checking, and its rules were set by comparison with NAFI.** Use it as a pointer
only.

Method: DEA S2A/B/C ARD, 1 April 2025 to 21 October 2026 (139 dates per site, 78–84 % of the
pixel-dates clear), B8A and B12 read straight onto the 20 m NISAR window of each site. NBR minus the
5-site median NBR of the same date. The largest robust drop (median of 3 valid observations before
against 3 after) in May–October of each year; burnt if the drop ≥ 0.1. Two rules came from the
first results: without the regional baseline, the end-of-wet-season fall of NBR looked like fire
(29 % of Bunda Extra "burnt" in March–April 2026), and drops from November to April are not used
(wet-season green-up and cloud residue).

Results:
- **Agreement with NAFI**: 80–100 % of the NAFI burnt-2026 cells are burnt in 2026 in Sentinel-2,
  except Country Downs (52 %). Sentinel-2 also finds 2026 burns in 21–25 % of the NAFI-unburnt cells
  at Biota track and Country Downs (small patches that NAFI's 250 m grid does not map, or artefacts).
- **The 2026 fires at the dark sites were later and more severe**: median fire date 21–28 July at
  9 Mile and South Dampier, against 19–27 May at Biota track and Country Downs; median dNBR of the
  burnt cells 0.25–0.26 against 0.17–0.18.
- **With Sentinel-2 classes, the burnt-2026 cells of Biota track are darker** than its unburnt
  cells (+0.64 against +0.94, 168 A HV), unlike NAFI. Country Downs: −0.39 against +0.14.
- **Bunda Extra's August 2025 burn is darker than its unburnt part** (−0.13 against +0.17), while
  the May–June 2025 burns at Biota track and Country Downs are brighter than their unburnt parts.
- A pooled fit on burnt 2026 reduces the range from 2.39 to 1.80 dB; dNBR 2026 gives the same.

Pointer (to test with checked burn maps): **late, severe fires may darken the backscatter for longer
than early fires, and early fires may be followed by brighter backscatter after a year.** The dark
sites had late fires in 2025 (9 Mile, August) and 2026 (both, July).

## 20 m single-date images and the decomposition against burn age (2026-10-22)

**Data.** All NISAR data are now native 20 m UTM, Refined Lee 5 × 5 (coherence unfiltered); the 30 m
EPSG:3577 series is deleted (`E:\NISAR\README.md`). For Dampier: 17 single-date GSLC images (147 D: 9
dates, 168 A: 8, Jun–Oct 2026) in `E:\NISAR\Dampier\GSLC_dcmp_20m_rflee5\` and 15 coherence pairs in
`GSLC_coh_20m\`, on the stack grid (no resampling). `structure_vs_sar.py` now reads the stack, every
single-date image, every pair, and the per-track date means (`dcmp_mean`: all dates; `dcmp_sepoct`: the
stack's two Sep–Oct dates). Power bands are averaged in linear power. 5 sites, 33,134 cells.

**Checks.** The single-date file has its largest |ρ| at zero shift. The per-site stack rows of the new
`structure_sar_stats.csv` equal the old rows exactly (the old pooled rows had Bunda Bunda; the old file
is `structure_sar_stats_pre_20m.csv`). Sep–Oct mean against the stack's own band: r ≥ 0.999 (Pu),
≥ 0.989 (Pd), 0.84–0.98 (Ps).

**Ps floor.** The decomposition writes a zero power as −60 dB. Ps = 0 is common where the canopy is
dense: on single Sep–Oct dates 24–26 % of the Bunda Extra cells and 7–11 % of Biota track (the stack Ps
has the same floor, 29 % at Bunda Extra). A mean in dB of −60 and −25 is a false −42.5 dB, so date means
are in linear power. In the figures, the floored cells count in the medians but lie below the axis.

**Results, structure.** More dates do not help: the best single-date or date-mean feature against
cover × p98 canopy is the 168 A all-date mean Pu (ρ² 0.56), against 0.57 for the stack. By date
(`figures\pooled\season.png`), HV ρ² is 0.50–0.55 on every date except 147 D 2026-06-24 (0.35; the
granule with the zero-power band, `G:\python\nisar\LAST_RUN.md`); 147 D HH dips on 2026-09-04 (0.38).
Windowed HH coherence holds little structure information at 20 m (ρ² ≤ 0.25 on 168 A, ≤ 0.13 on 147 D);
the pair 2026-09-16 → 09-28 (147 D) is decorrelated over the whole scene (coh_HH_win median 0.12).

**Results, burn age** (`figures\fire\c_p98_can_mixed_sites_by_burn_dcmp_{168_A,147_D}.png`,
`figures\by_site\c_p98_can_site_medians_dcmp_{168_A,147_D}.png`; Sep–Oct power mean). The fire effect is
in all three scattering parts, and Ps goes the other way:
- **Country Downs, both tracks:** cells burnt in 2026 are about 1 dB lower in Pu (volume) and Pd (double
  bounce) than the unburnt cells at the same structure, but 1–2 dB higher in Ps (surface). Cells burnt in
  2025 are the highest in Pu and Pd and the lowest in Ps (to about 4 dB below burnt 2026).
- **Biota track** (no HV effect): Pu is within about 0.5 dB, but Ps separates the classes (168 A: burnt
  2025 about 5 dB below the others; 147 D: both burnt classes 3–6 dB below unburnt). Pd of the unburnt
  cells is about 1 dB lower.
- **Site medians:** the dark pair in HV (9 Mile, South Dampier: burnt most often and most recently) are
  about 2 dB low in Pu and Pd but the highest in Ps (flat at about −19 dB). At the other sites, Ps falls as
  the structure increases.

Reading (not tested): a recent fire removes volume and double-bounce scattering and exposes the ground
surface; a burn of the year before has more volume and less surface than unburnt cells. Ps carries a
burn signal where HV has none (Biota track). Caveats: NAFI is 250 m; Ps is noisy and has the floor; the
burn classes are confounded with position within each site.

## Next

- **Decomposition and burn age, next steps** (the user decides): the step test
  (`fire_vs_sar.step_test`) on Pu / Ps / Pd per date, now that 17 single-date 20 m images exist (before /
  after each 2026 fire); Ps as a fire term in the offset models (Test 3).
- **Fire, other next steps** (the user decides): a burn-class split of Country Downs and Biota track in the
  structure × SAR statistics; the fire season and severity (early / late) as a term; checked 20 m burn
  maps if the Sentinel-2 pointer is worth the manual work.
- Non-linear fits (e.g. a saturating or log model) of HV against cover × p98 and mean first-return height. Try a per-site offset term (per-site scatter, finding 4).
- Within-site analysis of the three weak sites: is the structure range too small, or is the SAR noise too large at 20 m? The cutoff results suggest low vegetation (below about 2 m) is part of the cause. The per-site scatter shows that these sites have almost no cells on the steep part of the curve.
- From 2026-10-21: 5 sites (no Bunda Bunda), 20 m UTM only (from 2026-10-22: the stack plus the 20 m single-date images and pairs).
