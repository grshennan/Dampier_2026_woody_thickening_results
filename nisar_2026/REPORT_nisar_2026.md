# REPORT — Dampier 2026 classification and CHM regression (NISAR + S2, 2026 LiDAR)

Work 2026-10-19 to 2026-10-21. Specification `PLAN.md` (approved 2026-10-19) plus the user decisions
below. All paths are under `G:\python\dampier_veg_timeseries\nisar_2026\` unless given in full.
QA numbers: `outputs\qa_2026.txt` (from `qa_2026.py`).

## 1. What was made

| Product | File | Model |
|---|---|---|
| 5-class map (Low, Non-forest, Low forest, Forest, Tall forest) | `outputs\XGB_DP_NISAR_S2_2026_5class.tif` | XGB, 39 features |
| 4-class map (Low merged into Non-forest) | `outputs\XGB_DP_NISAR_S2_2026_4class.tif` | XGB, 39 features |
| 4-class map, Bunda Extra not in training | `outputs\XGB_DP_NISAR_S2_2026_4class_noBundaExtra.tif` | XGB, 39 features |
| 4-class map, NISAR only | `outputs\XGB_DP_NISAR_2026_4class_NISARonly.tif` | XGB, 26 features |
| CHM mean / p90 / p98 (3 bands, nodata −9999) | `outputs\RF_CHM_2026.tif` | RF, 39 features |

Class maps: uint8, nodata 255. Grid: NISAR GCOV, EPSG:32751, 20 m, 11625 × 9081. Domain: the
Dampier Peninsula mainland (64.4 M valid px; 64.45 M for the NISAR-only map, which does not need S2).
Model folders: `outputs\classify\`, `outputs\classify_4class\`, `outputs\classify_4class_noBundaExtra\`,
`outputs\classify_4class_NISARonly\`, `outputs\regress\`. Figures: `outputs\figures\`.

**Labels** (`labels_2026.py`): mean of the 50 cm p98 CHM per 20 m pixel, 2026 drone LiDAR (5–9 Oct
2026), cuts 0.2 / 1.22 / 2.35 / 5.381 m, `count >= 30`. 34,715 px: Low 377, Non-forest 4,885,
Low forest 8,141, Forest 19,204, Tall forest 2,108. Sites: 9 Mile, Biota track, Bunda Extra,
Country Downs, South Dampier (Bunda Bunda, JPP MVT, Quandong MVT excluded).

**Features** (`feature_stack_2026.py`, `E:\nisar_2026\feature_stack_2026.tif`, 39 bands): per track
(147 D, 168 A; the passes that bracket the flights) GCOV HH, HV, HV−HH, RVI (4) and GSLC dual-pol
decomposition Pd, Ps, Pu, DpRBI, DpRSI, alpha_dp, H, DoP, alpha_cloude (9); S2 geomedian
2026-09-20..10-10, 10 bands + NDVI, EVI, BSI (13).

**Speckle filter** (user decision 2026-10-20): Refined Lee **5 × 5** (`G:\python\GDV\GEDI\rflee.py`)
on GCOV and GSLC C2, **per acquisition, then the passes pooled**; decomposition window 5. The
InSAR coherence pairs are not filtered and are not features here. The unfiltered runs are kept with
the suffix `_nospeckle`.

## 2. Classification results

Accuracy: naive (resubstitution) and random stratified 5-fold (K-fold). The K-fold figure is the
one to use; it is optimistic (section 6).

| Run | Features | K-fold OA | balanced | kappa | F1w | naive OA |
|---|---|---|---|---|---|---|
| 5-class, filtered | 39 | 0.699 | 0.701 | 0.545 | 0.710 | 0.740 |
| 5-class, unfiltered (`_nospeckle`) | 39 | 0.700 | 0.701 | 0.547 | 0.711 | 0.735 |
| 5-class, 30 m GSLC, n_iter 500 (`_gslc30m`) | 39 | 0.704 | 0.705 | 0.552 | 0.715 | 0.745 |
| **4-class, filtered** | 39 | **0.711** | **0.738** | **0.559** | **0.722** | 0.760 |
| 4-class, filtered, no Bunda Extra (30,728 px) | 39 | 0.705 | 0.724 | 0.543 | 0.719 | 0.758 |
| 4-class NISAR only, filtered | 26 | 0.610 | 0.653 | 0.431 | 0.629 | 0.644 |
| 4-class NISAR only, unfiltered | 26 | 0.586 | 0.646 | 0.407 | 0.608 | 0.597 |
| 5-class from the RF regressed mean (section 3) | 39 | 0.758 | 0.568 | 0.587 | 0.751 | 0.998 |

**K-fold precision / recall / F1 per class**

| Class | 5-class filtered | 5-class unfiltered | 4-class filtered | 4-class no BE | NISAR-only filt. | NISAR-only unfilt. |
|---|---|---|---|---|---|---|
| Low | .596/.584/.590 | .650/.581/.613 | — | — | — | — |
| Non-forest | .748/.702/.724 | .769/.703/.734 | .789/.725/.756 | .772/.724/.748 | .736/.676/.705 | .725/.679/.702 |
| Low forest | .553/.696/.616 | .555/.705/.621 | .555/.705/.621 | .558/.706/.623 | .479/.662/.556 | .486/.644/.554 |
| Forest | .866/.686/.766 | .864/.685/.764 | .866/.697/.773 | .869/.695/.772 | .823/.559/.666 | .812/.518/.633 |
| Tall forest | .410/.836/.550 | .402/.829/.541 | .427/.824/.562 | .327/.771/.460 | .269/.716/.391 | .231/.741/.353 |

Balanced sample weights push recall up and precision down for the small classes (Tall forest).

**K-fold OA per site**

| Site | 5-cl filt. | 5-cl unfilt. | 4-cl filt. | 4-cl no BE | NISAR-only filt. | NISAR-only unfilt. |
|---|---|---|---|---|---|---|
| 9 Mile | .608 | .601 | .615 | .616 | .474 | .546 |
| Biota track | .734 | .731 | .745 | .732 | .633 | .550 |
| Bunda Extra | .679 | .687 | .714 | (held out: .690) | .633 | .610 |
| Country Downs | .739 | .740 | .737 | .734 | .623 | .560 |
| South Dampier | .688 | .696 | .702 | .703 | .637 | .657 |

**Feature importances** (XGB gain; shares by group and by track)

| Run | S2 | GCOV | GSLC | 168 A | 147 D | Top features |
|---|---|---|---|---|---|---|
| 5-class filtered | .531 | .182 | .287 | .280 | .189 | EVI .151, NDVI .081, blue .079, 168 A HH .056, 168 A HV .047 |
| 5-class unfiltered | .554 | .156 | .290 | .257 | .189 | EVI .111, NDVI .104, blue .058, SWIR-2 .049, BSI .047 |
| 4-class filtered | .533 | .250 | .217 | .304 | .163 | EVI .242, 168 A HH .123, NDVI .120, 168 A HV .065, BSI .026 |
| 4-class no Bunda Extra | .499 | .248 | .253 | .309 | .191 | EVI .221, 168 A HH .111, NDVI .099, 168 A HV .074, 147 D DpRSI .039 |
| NISAR-only filtered | — | .386 | .614 | .517 | .483 | 147 D Pu .155, 168 A HV .133, 168 A HH .131, 147 D DpRSI .120 |
| NISAR-only unfiltered | — | .464 | .536 | .740 | .260 | 168 A HV .213, 168 A Pd .155, 168 A Pu .101, 168 A HH .099 |

All importances are also in `outputs\importance_record.csv`.

**Best XGB parameters** (`best_params.json`): 5-class filtered max_depth 4, lr .051, gamma .106,
subsample .80, colsample .85, min_child_weight 6. 4-class (with and without Bunda Extra; the same
candidate won) max_depth 5, lr .035, gamma .041, subsample .69, colsample .85, min_child_weight 8.
NISAR-only filtered max_depth 5, lr .024, gamma .365, subsample .56, colsample .72, min_child_weight 5.

## 3. Regression results (RF, 3 targets)

Targets mean, p90, p98 of the 50 cm CHM per 20 m pixel (`max` dropped by the user, 2026-10-21).
34,715 px.

| Target | naive R² / RMSE | K-fold R² | K-fold RMSE | K-fold MAE | K-fold bias |
|---|---|---|---|---|---|
| mean | 1.000 / 0.011 m | 0.771 | 0.749 m | 0.540 m | +0.003 m |
| p90 | 0.999 / 0.088 m | 0.663 | 1.702 m | 1.242 m | −0.002 m |
| p98 | 0.977 / 0.432 m | 0.618 | 1.764 m | 1.302 m | +0.001 m |

The naive scores are almost perfect because the trees fit the training data exactly (mean and p90:
max_depth 30, min_samples_leaf 1, bootstrap False). Do not use them. p98 chose max_depth None,
min_samples_leaf 3, min_samples_split 5, max_features 0.316.

**K-fold per site** (R² / RMSE m)

| Site | mean | p90 | p98 |
|---|---|---|---|
| 9 Mile | 0.43 / 0.72 | 0.30 / 1.78 | 0.19 / 1.65 |
| Biota track | 0.50 / 0.82 | 0.36 / 1.71 | 0.27 / 1.66 |
| Bunda Extra | 0.84 / 0.90 | 0.79 / 1.42 | 0.76 / 1.51 |
| Country Downs | 0.57 / 0.84 | 0.47 / 1.95 | 0.39 / 2.11 |
| South Dampier | 0.71 / 0.49 | 0.54 / 1.61 | 0.48 / 1.84 |

Per-site R² is low where the height range in the site is small (9 Mile, Biota track); read RMSE.
Bunda Extra has the largest bias in all targets (+0.05 / +0.20 / +0.24 m).

**Feature importances** (RF MDI; not comparable in scale with XGB gain)

| Target | S2 | GCOV | GSLC | 168 A | 147 D | Top features |
|---|---|---|---|---|---|---|
| mean | .418 | .232 | .350 | .361 | .221 | EVI .111, NDVI .104, 168 A HV .085, 168 A HH .069, 168 A Pu .065 |
| p90 | .370 | .276 | .354 | .388 | .243 | 168 A HH .114, NDVI .082, EVI .076, 168 A HV .060, 168 A Pd .057 |
| p98 | .357 | .319 | .324 | .463 | .180 | 168 A HH .186, 168 A Pd .088, NDVI .087, EVI .077, 168 A HV .057 |

The SAR share increases from mean to p98 (.58 → .64), mainly 168 A HH. For the classes the
optical indices are the strongest features; for the canopy-top statistics L-band HH is.

**Classes from the regressed mean** (`np.digitize` on the 5-class cuts), K-fold, P/R/F1:
Low .915/.286/.436, Non-forest .834/.591/.692, Low forest .603/.639/.621, Forest .807/.897/.849,
Tall forest .776/.426/.551. OA 0.758 is higher than the XGB 5-class (0.699), but balanced accuracy
is much lower (0.568 against 0.701). The regression pulls predictions toward the centre, so the
end classes lose recall; XGB with balanced weights does the opposite. The map from the regressed
mean has 1.6 % Tall forest against 3.4–3.6 % from XGB (section 4).

## 4. QA of the maps

All checks below passed or are stated as findings (`outputs\qa_2026.txt`).

- **Map equals the in-memory naive prediction at every training pixel**: all four class maps (assert
  in `classify_2026.py`); CHM bands max |diff| ≤ 9e-7 m. Nodata set on every map. All maps on one grid.
- **CHM band order**: mean > p90 at 0 px; p90 > p98 at 0.29 % of px (expected: separate models; not
  clamped). Band ranges (p1 / p50 / p99 / max): mean 0.23 / 1.48 / 5.90 / 13.17 m, p90 0.78 / 4.62 /
  11.35 / 17.97 m, p98 1.49 / 6.10 / 13.10 / 17.38 m.
- **4-class areal fractions** over the mainland (5-class maps with Low merged after prediction):

| Map | Non-forest | Low forest | Forest | Tall forest |
|---|---|---|---|---|
| 5-class filtered | .574 | .201 | .190 | .036 |
| 5-class unfiltered | .511 | .247 | .204 | .038 |
| 4-class filtered | .572 | .204 | .190 | .034 |
| 4-class no Bunda Extra | .558 | .190 | .222 | .030 |
| NISAR-only filtered | .525 | .312 | .140 | .023 |
| NISAR-only unfiltered | .511 | .302 | .151 | .036 |
| classes from RF mean | .416 | .296 | .272 | .016 |

  5-class filtered: Low .187 / Non-forest .387 (unfiltered: .227 / .284).

- **Map agreement** (4 classes), with a block-shuffle null (50 px = 1 km blocks, 20 draws; the null
  keeps both maps' class frequencies and destroys their spatial correspondence):

| Map A | Map B | agreement | null | lift |
|---|---|---|---|---|
| 5-class filtered | 5-class unfiltered | 0.843 | 0.383 | 0.459 |
| NISAR-only filtered | NISAR-only unfiltered | 0.780 | 0.386 | 0.394 |
| 4-class | 5-class (Low merged) | 0.943 | 0.406 | 0.536 |
| 4-class | 4-class no Bunda Extra | 0.914 | 0.401 | 0.514 |
| 4-class | NISAR-only | 0.752 | 0.392 | 0.362 |
| 4-class | classes from RF mean | 0.762 | 0.351 | 0.410 |

- **Held-out site test**: the model trained without Bunda Extra, read at the 3,987 Bunda Extra label
  pixels: OA **0.690**, kappa 0.528. The random K-fold OA of the 4-class model at the same pixels is
  0.714, so for this site the random K-fold is optimistic by only ~0.02 OA. Errors in the held-out
  test go one class down (Tall forest → Forest 389 of 789; Low forest → Forest 279 of 694).
- Visual check of `outputs\figures\fig_maps.png`: the patterns follow the known pindan / coastal
  structure. Some straight edges are visible (for example a north–south line in the west, and
  rectangular patches in the centre). They may be fire scars, swath or S2 tile edges; not diagnosed.

## 5. Findings

1. **S2 carries the classification.** With S2, the speckle filter changes nothing material in
   accuracy (5-class OA 0.700 → 0.699). Without S2, NISAR alone reaches K-fold OA 0.61 (4 classes),
   and the filter helps (+0.024 OA, Forest F1 .633 → .666, Tall forest .353 → .391).
2. **The filter changes the map more than it changes the accuracy.** The filtered and unfiltered
   5-class maps agree on only 84 % of mainland pixels; the filtered map has 6 pp more Non-forest and
   5 pp less Low forest. Both models score the same on the labels, so the labels cannot choose
   between them. Where the changed pixels are has not been examined.
3. **The filter moves the NISAR-only importance from one track to both** (168 A .74 → .52; 147 D
   .26 → .48) and from GCOV to GSLC (.536 → .614). At the site level the effect is mixed: Biota track
   +0.08 and Country Downs +0.06 OA, but 9 Mile −0.07 and South Dampier −0.02.
4. **4 classes are better than 5** (OA 0.711, balanced 0.738), as expected from the small,
   poorly separated Low class (F1 0.59). The direct 4-class model and the 5-class model with Low merged
   after prediction agree on 94 % of pixels.
5. **Bunda Extra (white sand)** neither helps nor hurts the other sites: their K-fold OA changes by
   ≤ 0.014 when it is left out. Leaving it out changes 8.6 % of the map (Forest +3.3 pp). The held-out
   test at Bunda Extra (OA 0.690) shows that the model transfers to this soil type with a small loss.
6. **Regression**: CHM mean K-fold RMSE 0.75 m (R² 0.77); p90 1.70 m, p98 1.76 m. The canopy-top
   statistics depend more on L-band HH than the mean does.

**Univariate Spearman ρ against CHM mean** (unfiltered stack, 34,715 px): EVI .80, NDVI .79,
168 A HV .73, BSI −.73, SWIR-3 −.71, 168 A Pu .71, 168 A HH .69, 147 D HV .66; H, DoP,
alpha_cloude, HV−HH and RVI ≤ .12. A box mean of 1×1 / 3×3 / 5×5 / 7×7 px: 168 A HV .730 / .759 /
.761 / .759; 147 D HV .663 / .741 / .754 / .753; 168 A HH .689 / .749 / .758 / .759; 168 A Pu .707 /
.762 / .766 / .764; 168 A Pd .646 / .726 / .745 / .749; EVI .801 / .762 / .751 / .745. SAR gains up to
5 × 5 and then is flat; EVI loses. This was the reason to test a 5 × 5 speckle filter.

**Per-site K-fold OA, NISAR only (unfiltered) against the NISAR + S2 proxy** (unfiltered 5-class K-fold,
Low merged after prediction): 9 Mile .546 / .602, Biota .550 / .731, Bunda Extra .610 / .710,
Country Downs .560 / .740, South Dampier .657 / .704. The proxy over all sites: OA 0.705, balanced
0.736, kappa 0.552, F1w 0.717 — the direct filtered 4-class model is +0.006 OA on it.

## 6. Caveats

- **Within-year only.** Different sensor, grid and labels from 2019–2025; no comparison with that
  series. The acquisition-year caveat (`woody_thickening_splice_1\REPORT_year_signature.md`) applies to
  any cross-year statement.
- **No independent check.** There is no GEDI after 2025-07-09 (CMR search 2026-10-19), so the only
  accuracy numbers are against the training LiDAR.
- **The random K-fold is optimistic**: random folds under spatial autocorrelation, and the same folds
  chose the parameters. The one held-out-site test (Bunda Extra) shows a small gap (0.714 → 0.690);
  other sites were not tested this way.
- **The 1.22 m cut has a known provenance defect** (the cover fit gives 1.146 m at nominal 2 m;
  `class_vs_regression\REPORT_threshold_translation.md`). Not changed here.
- **The cuts were fitted on a 10 m grid** and are applied at 20 m without re-fitting.
- The class boundaries are cover classes expressed as a mean height, not height classes.
- The naive figures are resubstitution numbers; the RF regression's are close to 1 by construction.

## 7. Deviations from PLAN.md

- RandomizedSearchCV **n_iter 200** (not 500) for every model, own search per run, seed 42;
  **n_jobs 10** (user decision, nisar_2026 only). Only the kept 30 m 5-class run used 500.
- GSLC decomposition from the native GSLC as an exact 2 × 4 block mean onto the 20 m grid (not the
  30 m EPSG:3577 per-pass rasters); 16 looks (8 with one pass), min looks 8.
- **Speckle filter added** (2026-10-20): Refined Lee 5 × 5 per acquisition, then pool; dcmp window 5.
  The PALSAR chain uses 7 × 7; this is a stated deviation.
- **Refined Lee side rule**: `rflee.py` uses Lee et al. 1999 (the half-window nearer the centre
  sub-window mean). The reference `G:\Yawuru\WCM\SAR\filter_dp_grd_vec.py` (= PolSARpro / polsartools)
  picks the half by the gradient sign, which takes the darker side on a flat scene (HH mean −5.5 %,
  HV −1.4 % at 9.7 looks); `rflee.py` gives −0.1 % and lower RMSE at edges. The 2025 PALSAR
  `*_rflee_w5.tif` came from the reference and carry that bias. Measured ENL ≈ 0.75 × nominal looks;
  sigma² = 1 / nominal looks per pixel.
- Regression targets mean, p90, p98 (no `max`).
- XGB has no `num_class`; the stack is a GeoTIFF, not a zarr; S2 10 → 20 m by average; prediction in
  512-row blocks.
- Extra runs: 4-class, 4-class without Bunda Extra, 4-class NISAR only.
- The GDV\GEDI RH98 model steps were removed from the chain by the user (2026-10-21): the 30 m
  EPSG:3577 NISAR grid is dropped, `E:\NISAR` moves to the native 20 m UTM grids, and that model is
  redone later. The filtered 30 m products (`E:\NISAR\<Region>\GSLC_dcmp_rflee5\`,
  `E:\GEDI_CHM\nisar_gcov_2026_30m_rflee5.tif`) exist but no model uses them.

## 8. User decisions and tests passed

User decisions (2026-10-19 to 21): the Low class is "not useful", so the 4-class runs merge it into
Non-forest (< 1.22 m); Bunda Extra has white sandy soil, unlike the red pindan elsewhere, hence the
run without it; every K-fold report includes precision, recall, F1 per class and the feature
importances (`kfold_report_2026.py <output folder> <run name>`); naive + random stratified 5-fold, no
spatial blocking; coherence pairs not filtered.

Tests passed:
- 20 m C2 alignment (147 D 10 Oct frame 097, 400 × 400 px at Biota track): 8 looks in every cell;
  C11 against GCOV HH r = 0.969 at zero shift, 0.19–0.32 at any 1 px shift; C11 − HH −1.08 dB
  (sigma0 against gamma0).
- Filtered stack: C2 16 looks p1 / p50 in both tracks, PSD 1.00000, energy check 3.45e-16, SAR 100 % /
  S2 99.9 % valid in the mainland (64,469,255 px, clip ratio 1.0000), 0 training rows dropped.
- `rflee.py` self-test: equals the reference in the interior at windows 5 / 7 (side "sign");
  flat-scene mean HH 0.999 (side "centre"); per-product mean ratio filtered / raw 0.995–0.999.
- `--quick` runs of classify / regress / figures end to end; `regress --quick --reuse` gives the
  same K-fold RMSE (1.047 against 1.045) and refuses a model with a different feature count.
- All maps equal the in-memory naive prediction at every training pixel (section 4).

## 9. Files

| File | Role |
|---|---|
| `labels_2026.py` | labels on the GCOV grid → `outputs\labels_2026.parquet`, `outputs\labels\chm_targets_<site>.tif` |
| `s2_geomedian_2026.py` | S2 download + geomedian → `G:\python\odc-stac\ds_geomedian_DP_2026_0920_1010.zarr` |
| `feature_stack_2026.py` | 39-band stack `E:\nisar_2026\feature_stack_2026.tif`; accuracy diagnosis modes `--sar gcov_nophase` / `gslc_gamma`, `--rflee-win`, `--dcmp-ws`, `--s1` (`REPORT_accuracy_diagnosis.md` §1; reference stack from 2026-10-29: `E:\nisar_2026\feature_stack_2026_gslcg0_rflee9_ws5.tif`) |
| `classify_2026.py` | XGB (`--merge-low`, `--exclude-site`, `--sar-only`, `--s2-only`, `--no-nisar`, `--seed`, `--no-map`, `--tag`) |
| `diagnosis\` | accuracy diagnosis scripts and chains (`stage*_chain.ps1`, `stage0_summary.py`, `map_agreement.py`, `stage1_radiometry.py`, `stage1_enl.py`, `s1_mosaic_2026.py`); logs `diagnosis\stage*.log` |
| `G:\python\GDV\GEDI\gcov_decomposition.py` | no-phase (GRD) dual-pol decomposition of GCOV σ0 (Stage 1.1) |
| `compare_4classTF.py` | the 4classTF runs against each other, with the block-shuffle null (sections 10, 11) → `outputs\compare_4classTF.txt` |
| `relaunch_4classTF.ps1` | the section 10 chain, detached (not used: blocked; the run was launched by a direct WMI call) |
| `regress_2026.py` | RF, 3 targets, `--reuse` |
| `kfold_report_2026.py` | K-fold P/R/F1 + importance report |
| `figures_2026.py` | figures → `outputs\figures\` |
| `qa_2026.py` | map QA → `outputs\qa_2026.txt` |
| `run_rflee_all.py`, `relaunch_rflee.ps1`, `watch_rflee.py`, `stop_at_max.py` | the filtered-chain driver (complete 2026-10-21 20:23) |
| `run_chain_2026.py`, `queue_*_2026.py`, `watch_2026.py`, `relaunch_2026.ps1`, `pause_2026.py` | the old unfiltered orchestration |
| `G:\python\GDV\GEDI\rflee.py` | the Refined Lee filter |

Kept older versions: `_gslc30m` (30 m GSLC 5-class run and stack), `_nospeckle` (unfiltered runs,
`E:\nisar_2026\feature_stack_2026_nospeckle.tif`, `logs_nospeckle\`).

## 10. 4classTF, GSLC-only SAR (2026-10-27)

**Designation `4classTF`** ("Tall Forest kept"): Non-forest (< 1.22 m, Low merged in), Low forest
(1.22–2.35 m), Forest (2.35–5.381 m), Tall forest (≥ 5.381 m), on the mean of p98 per 20 m pixel. This
name keeps the scheme apart from the 2019–2025 4-class (Bare / Low veg / Low forest / Forest). The
`_4class` 2026 runs in sections 1–8 (`--merge-low`) use the same 4classTF scheme.

**Run `4classTF_GSLC`** (plan `C:\Users\glens\.claude\plans\we-want-to-do-sparkling-pearl.md`): all SAR
from GSLC. The 8 GCOV features (HH, HV, HV−HH, RVI per track) are replaced by the same 4 features from
C11 / C22 of the filtered, pooled C2 (`gHH_dB`, `gHV_dB`, `gHVHH_dB`, `gRVI`; σ0, ellipsoid, not
terrain-flattened). Labels, the 18 decomposition bands, the 13 S2 bands, the search (n_iter 200,
random_state 42) and the folds are the same as the GCOV 4-class run.

- Stack: `E:\nisar_2026\feature_stack_2026_gslc.tif` (`feature_stack_2026.py --sar gslc`).
- Map: `G:\python\dampier_veg_timeseries\nisar_2026\outputs\XGB_DP_NISAR_S2_2026_4classTF_GSLC.tif`.
- Model folder: `outputs\classify_4classTF_GSLC\`.
- Log: `classify_2026_4classTF_GSLC.log`.
- Comparison: `compare_4classTF.py`.

**Checks (all passed).** The 31 shared bands are bit-identical to `feature_stack_2026.tif`. GSLC
against GCOV backscatter: r 0.992–0.993 at zero shift, 0.95–0.96 at a 1-pixel shift. The GSLC values are
1.35–1.49 dB lower, which is σ0 against γ0. The valid pixels are the same, except 114 fewer on 168 A.
The map equals the naive prediction at every training pixel.

**Result: the same as the GCOV run, within noise.**

| K-fold | GCOV (`4class`) | GSLC (`4classTF_GSLC`) |
|---|---|---|
| OA | 0.711 | 0.707 |
| balanced | 0.738 | 0.738 |
| kappa | 0.559 | 0.554 |
| F1w | 0.722 | 0.718 |
| F1 Non-forest / Low forest / Forest / Tall forest | .756 / .621 / .773 / .562 | .753 / .619 / .768 / .555 |

GSLC K-fold P / R / F1:

| Class | P | R | F1 |
|---|---|---|---|
| Non-forest | .784 | .724 | .753 |
| Low forest | .553 | .703 | .619 |
| Forest | .867 | .689 | .768 |
| Tall forest | .415 | .836 | .555 |

- **Per site:** the OA changes by −0.008 to −0.001 (9 Mile .610, Biota track .737, Bunda Extra .710,
  Country Downs .736, South Dampier .701).
- **Importances (gain):** S2 .533, GSLC backscatter .261, GSLC decomposition .206. The track shares
  are the same as the GCOV run (147 D .163, 168 A .304). The top features are EVI .250, NDVI .122,
  168 A gHH .108, 168 A gHV .095 and 147 D DpRSI .037. The GSLC backscatter takes the place of the
  GCOV backscatter (.250) with almost the same weight.
- **Map:** the two maps agree on 93.5 % of the 64.4 M mainland pixels. The GSLC map has 1.8 pp less
  Non-forest and 1.2 pp more Low forest; most of the changed pixels move by one class.
- **Reading:** the GCOV RTC terrain correction and the GCOV processing add nothing over GSLC on this
  flat peninsula. A GSLC-only chain is enough. The caveats in section 6 apply unchanged.
- **Block-shuffle null** (added with section 11): agreement 0.935, null 0.399, lift 0.536.

## 11. 4classTF, one GSLC acquisition per run (2026-10-27)

**Question:** do all NISAR acquisitions cover the area, and how does a classification from one
acquisition compare with the pooled runs? Only 2 tracks exist, 147 D and 168 A, with 2 acquisitions
each. All 4 cover the mainland (147 D 100 %, 168 A 99.97 %). One run per acquisition, 4 runs.

**Method.** The same as section 10, except that the SAR is from one acquisition only: GSLC backscatter
(4 bands) and the decomposition (9 bands) of that acquisition, plus the same 13 S2 bands, 26 features.
The same chain: Refined Lee 5 × 5, decomposition ws 5, on the stack grid (the single-date files in
`E:\NISAR\Dampier\GSLC_dcmp_20m_rflee5\` are on a different extent and are not used). The same labels,
search (n_iter 200, random_state 42) and folds.

- Stacks: `E:\nisar_2026\feature_stack_2026_gslc_<trk>_<mmdd>.tif`
  (`feature_stack_2026.py --sar gslc`, one acquisition).
- Maps: `G:\python\dampier_veg_timeseries\nisar_2026\outputs\XGB_DP_NISAR_S2_2026_4classTF_GSLC_<trk>_<mmdd>.tif`,
  `<trk>_<mmdd>` = `147D_0928`, `147D_1010`, `168A_0929`, `168A_1011`.
- Model folders: `outputs\classify_4classTF_GSLC_<trk>_<mmdd>\`.
- Log: `classify_2026_4classTF_single.log`.
- Comparison: `compare_4classTF.py` → `G:\python\dampier_veg_timeseries\nisar_2026\outputs\compare_4classTF.txt`.

**Checks (all passed).** In each stack the 13 S2 bands are bit-identical to `feature_stack_2026.tif`.
One-acquisition gHH / gHV against the pooled GCOV HH / HV: r 0.959–0.982 at zero shift, 0.92–0.95 at a
1-pixel shift (lower than in section 10, because one acquisition has more speckle than two pooled).

**K-fold (all sites):**

| Run | OA | balanced | kappa | F1w | F1 NF / LF / F / TF |
|---|---|---|---|---|---|
| GCOV (`4class`) | .711 | .738 | .559 | .722 | .756 / .621 / .773 / .562 |
| GSLC, both tracks | .707 | .738 | .554 | .718 | .753 / .619 / .768 / .555 |
| 147D_0928 | .703 | .728 | .546 | .713 | .741 / .603 / .769 / .560 |
| 147D_1010 | .700 | .730 | .544 | .711 | .743 / .607 / .764 / .554 |
| 168A_0929 | .702 | .735 | .548 | .714 | .752 / .614 / .764 / .546 |
| 168A_1011 | .713 | .736 | .561 | .723 | .753 / .620 / .776 / .565 |

**K-fold P / R per class** (NF / LF / F / TF):

| Run | P | R |
|---|---|---|
| 147D_0928 | .763 / .544 / .859 / .426 | .719 / .677 / .697 / .817 |
| 147D_1010 | .773 / .542 / .862 / .416 | .716 / .689 / .687 / .828 |
| 168A_0929 | .778 / .551 / .864 / .406 | .728 / .694 / .684 / .835 |
| 168A_1011 | .781 / .557 / .865 / .432 | .727 / .698 / .704 / .816 |

**K-fold OA per site:**

| Site | GCOV | GSLC | 147D_0928 | 147D_1010 | 168A_0929 | 168A_1011 |
|---|---|---|---|---|---|---|
| 9 Mile | .615 | .610 | .600 | .597 | .602 | .619 |
| Biota track | .745 | .737 | .744 | .737 | .734 | .750 |
| Bunda Extra | .714 | .710 | .706 | .707 | .705 | .711 |
| Country Downs | .737 | .736 | .742 | .735 | .737 | .744 |
| South Dampier | .702 | .701 | .682 | .687 | .694 | .700 |

**Importances (gain share):**

| Run | S2 | GSLC backscatter | GSLC decomposition | Top SAR features |
|---|---|---|---|---|
| 147D_0928 | .747 | .056 | .197 | Pu .039, DpRSI .037, Pd .027 |
| 147D_1010 | .714 | .093 | .193 | DpRSI .053, gHH .038, Pu .037 |
| 168A_0929 | .645 | .231 | .124 | gHH .130, gHV .083, DpRSI .028 |
| 168A_1011 | .678 | .161 | .161 | gHH .081, gHV .055, DpRSI .036 |

EVI (.26–.29) and NDVI (.16–.22) are first and second in every run.

**Class fractions** (64,391,975 px valid in every map):

| Class | GCOV | GSLC | 147D_0928 | 147D_1010 | 168A_0929 | 168A_1011 |
|---|---|---|---|---|---|---|
| Non-forest | .572 | .555 | .502 | .539 | .547 | .541 |
| Low forest | .204 | .216 | .226 | .226 | .213 | .222 |
| Forest | .190 | .194 | .234 | .199 | .203 | .203 |
| Tall forest | .034 | .035 | .038 | .037 | .037 | .035 |

**Map agreement** with the block-shuffle null (50 px = 1 km blocks, 20 draws, null SD 0.002):

| Map A | Map B | agreement | null | lift |
|---|---|---|---|---|
| GCOV | GSLC | 0.935 | 0.399 | 0.536 |
| GCOV | 147D_0928 | 0.834 | 0.379 | 0.454 |
| GCOV | 147D_1010 | 0.865 | 0.393 | 0.470 |
| GCOV | 168A_0929 | 0.895 | 0.396 | 0.498 |
| GCOV | 168A_1011 | 0.895 | 0.394 | 0.500 |
| GSLC | 147D_0928 | 0.840 | 0.374 | 0.464 |
| GSLC | 147D_1010 | 0.870 | 0.387 | 0.481 |
| GSLC | 168A_0929 | 0.901 | 0.390 | 0.510 |
| GSLC | 168A_1011 | 0.902 | 0.388 | 0.513 |
| 147D_0928 | 147D_1010 | 0.864 | 0.369 | 0.493 |
| 147D_0928 | 168A_0929 | 0.839 | 0.371 | 0.466 |
| 147D_0928 | 168A_1011 | 0.851 | 0.370 | 0.480 |
| 147D_1010 | 168A_0929 | 0.848 | 0.384 | 0.462 |
| 147D_1010 | 168A_1011 | 0.866 | 0.383 | 0.482 |
| 168A_0929 | 168A_1011 | 0.884 | 0.385 | 0.498 |

**Reading.**

1. **At the label pixels, one acquisition is as good as four.** K-fold OA is 0.700–0.713 against 0.707
   (GSLC, both tracks) and 0.711 (GCOV). The range (1.3 pp) is the size of the search and fold noise;
   168A_1011 is the highest of all 6 runs. S2 carries the classification (section 5), so the SAR
   source changes little at the 5 sites.
2. **Away from the labels, the maps differ more.** One-acquisition maps agree with the pooled maps on
   only 83–90 % of the mainland (pooled GCOV against pooled GSLC: 93.5 %). Two acquisitions of the same
   track agree on 86.4 % (147 D) and 88.4 % (168 A). So 10–17 % of the mainland changes class with the
   acquisition, but the K-fold metrics do not show it. Pooling the 4 acquisitions makes the map more
   stable.
3. **168 A (ascending) is better than 147 D (descending).** The model uses more of the 168 A SAR
   (.32–.36 of the importance against .25–.29 for 147 D), mostly gHH / gHV, and the 168 A maps are
   closer to the pooled maps (0.895 against 0.834–0.865). In the 147 D runs the backscatter has little
   weight (.06–.09) and the decomposition (Pu, DpRSI, Pd) has more. This agrees with the pooled runs,
   where 168 A has about twice the share of 147 D (section 10). The cause (incidence angle, look
   direction or the acquisition conditions) is not diagnosed.
4. **147D_0928 is the outlier.** It has 6–7 pp less Non-forest and 3–4 pp more Forest than the other
   runs, and the lowest agreement with all other maps (0.834–0.864). Its K-fold OA is normal, so the
   difference is in areas without labels. The cause (for example rain or soil moisture on 2026-09-28)
   is not diagnosed.
5. **Recommendation:** for a map, use the pooled stack (both tracks, 2 acquisitions each). If only one
   acquisition is available, use 168 A. Report the K-fold numbers together with the map agreement,
   because the K-fold numbers do not show the acquisition effect.
