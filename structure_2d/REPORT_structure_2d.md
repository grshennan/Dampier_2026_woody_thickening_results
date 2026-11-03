# REPORT — natural 2-D structure classes (cover × p98 canopy), 2026-10-29 / 30

Plan: `structure_2d\PLAN.md` (approved 2026-10-29). Question (user): the 4- and 5-class cuts come from the NFI rule
"20 % cover at nominal height"; does the structure itself have natural points to split at? Space: cover at 1.37 m
(cover137) × p98 of the canopy returns (p98_can), the pair that had the strongest relation to NISAR in
`lidar_2026\REPORT_structure_sar.md`. Then: can NISAR + S2 separate the classes found?

> **⚠ Class codes in this report are the OLD order (by median cover × p98): 3 = S3 open, tall; 4 = S4 dense,
> shorter.** From 2026-10-30 (user decision) the time series uses a height order with codes 3 and 4 swapped
> (3 = S3 dense, shorter; 4 = S4 open, tall), tag `k4h`. See `REPORT_timeseries_method.md` (top). Files tagged `k4`
> (this report) and `k4h` (the series) differ in codes 3 and 4.

## Summary

1. **There are no natural thresholds.** The cover × height density has one mode; ICL is lowest at k = 1. The
   structure is one continuum. Any class scheme cuts a gradient.
2. **Stable partitions exist for k = 2, 3, 4** (mean ARI ≥ 0.8 in a 7 × 7-cell block bootstrap and in
   leave-one-area-out, rule fixed before the run); k ≥ 5 is not stable. User chose k = 4 (+ a no-canopy class).
3. **The 4classTF cuts are cover classes** in this space (their iso-lines are almost vertical). The k = 4 classes
   split both ways: along cover × p98 (S1 → S2 → S3/S4) and across it (S3 open-tall against S4 dense-shorter,
   equal cover × p98).
4. **NISAR + S2 separate the 2-D k = 4 scheme as well as 4classTF on the same cells**: kappa 0.613 against 0.616,
   with one class more (5 against 4). The merge rule merges no pair, so the sensor-resolvable k is 4 and the final
   scheme is k = 4 + no canopy.
5. **Confusion follows cover × p98**: pairs close along it are confused more (Spearman ρ −0.81, all cells; −0.88,
   Dampier only). S3 / S4 (Δ ln = 0.035) is the most confused pair in the Dampier-only run (20 %) and the second
   in the pooled run (13 %), but it is still separated (correct 0.75): NISAR + S2 see more than the product.
6. **The 2-D classes use SAR more than 4classTF does**: SAR importance share 0.66 against 0.57; 4classTF leans on
   EVI / NDVI (0.28 together).
7. **Yawuru 2025 labels help Dampier.** Dampier-only training gives lower F1 in every class at Dampier than the
   pooled model (kappa 0.472 against 0.563). The one-year LiDAR gap is not visible; more cells in the minority
   classes outweigh it.

## Stage 0 — Yawuru structure grids (LAStools)

`yawuru_struct.py`; 13 areas of the original Yawuru 4-class work (`G:\Yawuru\Drone Data 2025\`), not the July 2024
survey. Recipe: first returns, `-drop_withheld -drop_class 7 18`, z < 0 → 0, lascanopy `-step 20` on the NISAR
grid (origin 393160, 8197760), fill from lasgrid `-step 1 -counter_16bit`. Output
`structure_2d\outputs\yawuru\struct_nisar20m_<area>.tif` (band names of `lidar_2026\structure_grids.BANDS`).

- Method check: LAStools on 9 Mile against the laspy grid of `lidar_2026` (4,605 cells): cover137 median |Δ| 2e-6,
  p98_can 0.0017 m, fill exact.
- Source check: 12 of 13 clouds re-make their original 0.5 m CHM exactly. **YA25_3's CHM came from
  `YF25_3_height.las`** (name match), not the time-matched `YA25_3_height_thin.las` (max |Δ| 2.9 m).
- **YJ25_3: the stored height cloud is a copy of YJ25_4** (same size and extent). Re-made from the raw
  `YJ25_3.las` with the recorded recipe (lasground_new `-wilderness` → lasthin → lasheight `-replace_z` → lasnoise);
  the new CHM equals the original at all 1,343,038 pixels (|Δ| = 0).
- Cells in the cut polygons: 8,530 (305–850 per area).

## Stage 1 — cell table

`cells_2d.py` → `outputs\cells_2d.parquet`: 41,553 cells with fill ≥ 0.9 (Dampier 2026: 33,134 in 5 sites, Bunda Bunda
excluded; Yawuru 2025: 8,419 in 13 areas). Dampier counts and the ρ check (c_p98_can against 168 A HV: 0.757, report
0.76) reproduce `lidar_2026`. **No canopy = n_can < 20 or cover137 < 1 %**: 1,398 cells (1,227 Yawuru). The 1 %
floor is needed because the Yawuru clouds are so dense that 20 canopy returns can be < 0.1 % cover; without it those
cells piled at the logit clip and made a false density mode.

## Stage 2 — classes (no SAR)

`find_classes_2d.py`; space logit(cover137) × ln(p98_can), robust-standardised, 1,000 cells per area (equal
weight). 40,155 canopy cells. Outputs `outputs\classes\`; figures `outputs\classes\figures\fig_space_k.png`,
`fig_k_selection.png`.

| k | BIC | ICL | boot ARI | LOAO ARI mean / min | ARI vs 4classTF (null ≈ 0.01) | Stable |
|---|---|---|---|---|---|---|
| 1 | 55,559 | **55,559** | – | – | – | – |
| 2 | 53,001 | 59,343 | 0.969 | 0.969 / 0.943 | 0.25 | yes |
| 3 | 52,758 | 65,465 | 0.912 | 0.921 / 0.622 | 0.18 | yes |
| 4 | 52,823 | 68,014 | 0.863 | 0.880 / 0.640 | 0.29 | yes |
| 5 | 52,620 | 70,673 | 0.735 | 0.790 / 0.590 | 0.35 | no |
| 6–10 | 52,458–52,606 | 70,336–72,721 | 0.65–0.84 | 0.78–0.87 | 0.28–0.37 | no |

KDE: one mode at Scott ×1 and ×2 (16 at ×0.5, noise). Weakest area left out: Biota track. Cramér's V class × area
0.49–0.55; no class ≥ 80 % one area.

k = 4 classes (S ordered by median cover137 × p98_can):

| Class | Cells | Cover137 | p98_can (m) | cover × p98 | CHM mean p05–p95 (m) | Dampier share (all: 82 %) |
|---|---|---|---|---|---|---|
| S1 short, sparse | 3,203 | 7 % | 3.9 | 0.29 | 0.23–1.30 | 58 % |
| S2 open, mid height | 6,888 | 17 % | 6.9 | 1.19 | 0.65–2.12 | 72 % |
| S3 open, tall | 22,525 | 33 % | 10.5 | 3.45 | 1.66–5.61 | 95 % |
| S4 dense, shorter | 7,539 | 49 % | 7.4 | 3.58 | 2.03–6.08 | 64 % |

S3 is mostly Dampier and S4 more Yawuru: part of the S3 / S4 split can be site and LiDAR year.

## Stage 3 — re-train (NISAR + S2)

`train_2d.py` (transcription of `nisar_2026\classify_2026.py`): XGB multi:softprob, 500 trees, n_iter 200, F1w,
stratified 5-fold (seed 42), balanced weights, 2 cores. Stack `E:\nisar_2026\feature_stack_2026_gslcg0_rflee9_ws5.tif`
(39 features). All metrics are K-fold (out-of-fold); random K-fold is optimistic under spatial autocorrelation.
Report: `report_2d.py` → `report_2d.log`. Noise band (Stage 1.5, nisar_2026): OA 0.006, kappa 0.008.

| Run | Cells | OA | Balanced | Kappa | F1w |
|---|---|---|---|---|---|
| (a) k4 + no canopy, all | 41,553 | 0.739 | 0.760 | 0.613 | 0.747 |
| (a) — Dampier rows | 33,134 | 0.744 | 0.719 | 0.563 | 0.755 |
| (a) — Yawuru rows | 8,419 | 0.720 | 0.686 | 0.634 | 0.705 |
| (b) 4classTF, same cells | 41,553 | 0.745 | 0.774 | 0.616 | 0.754 |
| (b) — Dampier rows | 33,134 | 0.739 | 0.768 | 0.593 | 0.749 |
| (b) — Yawuru rows | 8,419 | 0.769 | 0.674 | 0.657 | 0.775 |
| (c) k4 + no canopy, Dampier only | 33,134 | 0.655 | 0.743 | 0.472 | 0.675 |

OA is not comparable between (a) (5 classes) and (b) (4 classes); compare kappa and balanced accuracy.

Per class, K-fold precision / recall / F1 (all cells unless stated):

| (a) k4 | P | R | F1 | n | | (b) 4classTF | P | R | F1 | n |
|---|---|---|---|---|---|---|---|---|---|---|
| no canopy | 0.868 | 0.900 | 0.883 | 1,398 | | Non-forest | 0.831 | 0.833 | 0.832 | 7,325 |
| S1 | 0.696 | 0.707 | 0.702 | 3,203 | | Low forest | 0.585 | 0.734 | 0.651 | 9,999 |
| S2 | 0.520 | 0.673 | 0.587 | 6,888 | | Forest | 0.892 | 0.714 | 0.793 | 22,086 |
| S3 | 0.896 | 0.742 | 0.812 | 22,525 | | Tall forest | 0.437 | 0.815 | 0.569 | 2,143 |
| S4 | 0.630 | 0.777 | 0.696 | 7,539 | | | | | | |

| F1 at Dampier | no canopy | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| (a) pooled model | 0.697 | 0.709 | 0.580 | 0.829 | 0.626 |
| (c) Dampier-only model | 0.656 | 0.704 | 0.535 | 0.733 | 0.555 |

(a) at Yawuru: no canopy 0.910, S1 0.691, S2 0.604, **S3 0.391 (recall 0.264, n 1,188)**, S4 0.831.

Class pairs, run (a) (mutual K-fold confusion (C_ij + C_ji) / (n_i + n_j); Δ = |Δ ln cover × p98| of the class
medians):

| Pair | Confusion | Correct | Δ |
|---|---|---|---|
| S1 / S2 | 0.152 | 0.684 | 1.39 |
| S3 / S4 | 0.130 | 0.751 | 0.04 |
| S2 / S3 | 0.128 | 0.726 | 1.07 |
| S2 / S4 | 0.081 | 0.727 | 1.11 |
| no canopy / S1 | 0.062 | 0.765 | – |
| other pairs | ≤ 0.006 | | |

Run (c): S3 / S4 0.204, S2 / S3 0.150, S1 / S2 0.144. Merge rule (merge while confusion > correct): no merge in
(a) or (c).

Feature importances (gain; group shares):

| Run | GSLC γ0 backscatter | GSLC decomposition | S2 | 147 D / 168 A | Top 3 |
|---|---|---|---|---|---|
| (a) k4 | 0.314 | 0.343 | 0.343 | 0.207 / 0.450 | 168A g0HH, 168A g0HV, nbart_blue |
| (b) 4classTF | 0.316 | 0.252 | 0.432 | 0.158 / 0.410 | 168A g0HH, EVI, NDVI |
| (c) k4 Dampier | 0.174 | 0.309 | 0.518 | 0.208 / 0.274 | nbart_swir_2, 168A g0HH, 147D DpRBI |

All importances: `structure_2d\outputs\importance_record.csv`.

## Map

`train_2d.py --labels k4 --no-canopy-class --map-only` (the saved model of run (a), no new search) →
`structure_2d\outputs\XGB_DP_NISAR_S2_2026_2D_k4_all_nocan.tif` (uint8, nodata 255, codes 0 no canopy, 1–4 = S1–S4,
**old order**; the `k4h` series swaps 3 and 4).
The map equals the naive prediction at every training cell. Comparison with the 4classTF reference map
`nisar_2026\outputs\XGB_DP_NISAR_S2_2026_4classTF_GSLCg0_f9w5.tif` (`map_compare_2d.py`, 64,392,143 valid pixels):

| 2-D class | Map fraction | Non-forest | Low forest | Forest | Tall forest |
|---|---|---|---|---|---|
| no canopy | 0.174 | 1.000 | 0.000 | 0.000 | 0.000 |
| S1 | 0.137 | 0.989 | 0.010 | 0.001 | 0.000 |
| S2 | 0.301 | 0.711 | 0.284 | 0.005 | 0.000 |
| S3 | 0.139 | 0.040 | 0.281 | 0.543 | 0.136 |
| S4 | 0.249 | 0.173 | 0.302 | 0.461 | 0.064 |

(rows: share of each 4classTF class inside the 2-D class; 4classTF map fractions 0.572 / 0.201 / 0.192 / 0.035.)
ARI 0.181, block-shuffle null 0.0001 ± 0.0004 (24,984 full 1 km blocks), lift 0.181.

- The 4classTF Non-forest (57 % of the map) becomes no canopy + S1 + most of S2: the 2-D scheme splits the open
  country by cover × height, where 4classTF has one class.
- S3 (open, tall) and S4 (dense, shorter) both hold mostly Low forest + Forest: the 2-D scheme splits the treed
  country by its cover / height mix, which 4classTF does not separate.
- **No canopy is 17.4 % of the map but 3.4 % of the training cells (88 % of them Yawuru).** It extrapolates far
  from its labels; treat its extent with care.

### Class areas over the Dampier Peninsula (added 2026-10-30)

`map_figure_2d.py` → `outputs\map_areas_2d.csv`, `outputs\figures\fig_map_2d.png`. The map covers 99.88 % of the
mainland polygon (no data 30.8 km², stack gaps). Clips asserted against the polygon areas.

| Class | Mainland km² | Mainland share | Pindan km² | Pindan share |
|---|---|---|---|---|
| no canopy | 4,487 | 0.174 | 532 | 0.065 |
| S1 short, sparse | 3,518 | 0.137 | 795 | 0.098 |
| S2 open, mid height | 7,742 | 0.301 | 2,685 | 0.329 |
| S3 open, tall | 3,588 | 0.139 | 1,832 | 0.225 |
| S4 dense, shorter | 6,421 | 0.249 | 2,310 | 0.283 |

- S3 is mostly in the north-west; no canopy is mostly in the south-east and on the King Sound side.
- The map shows straight edges (a horizontal edge in the centre, a north–south line in the south). This is the
  same open item as `nisar_2026\CHECKPOINT.md` item 2 (swath or tile edges?); not diagnosed.

## Caveats

- No natural thresholds: the k = 4 classes are a stable partition of a continuum, not discovered types.
- Yawuru LiDAR is 2025, the stack Sep–Oct 2026. Run (c) shows the pooled labels still help at Dampier.
- No canopy is 88 % Yawuru cells; S3 is 95 % Dampier. Class and source are linked (Cramér's V ≈ 0.5).
- Random K-fold is optimistic (project decision; same as `nisar_2026`). No leave-site-out accuracies (user rule).
- The no-canopy product in the pair table is set to 0.01 for Δ; only its order matters.

## Open follow-on items (not started)

- Apply the k = 4 classes to the PALSAR years 2019–2025 (the Yawuru grids make that possible; the year-signature
  caveat in the project `CLAUDE.md` applies).
- Check the no-canopy extent of the map against imagery.
- Time series with this scheme and stack (user decision 2026-10-30): method in `REPORT_timeseries_method.md`.

## Files

| What | Where |
|---|---|
| Yawuru grids | `G:\python\dampier_veg_timeseries\structure_2d\outputs\yawuru\struct_nisar20m_<area>.tif` |
| Cell table | `G:\python\dampier_veg_timeseries\structure_2d\outputs\cells_2d.parquet` |
| Classes, k-selection, figures | `G:\python\dampier_veg_timeseries\structure_2d\outputs\classes\` |
| Training runs | `G:\python\dampier_veg_timeseries\structure_2d\outputs\train\<tag>\` (metrics, confusion, importance, model) |
| Map | `G:\python\dampier_veg_timeseries\structure_2d\outputs\XGB_DP_NISAR_S2_2026_2D_k4_all_nocan.tif` |
| Map figure, class areas | `G:\python\dampier_veg_timeseries\structure_2d\outputs\figures\fig_map_2d.png`, `...\structure_2d\outputs\map_areas_2d.csv` |
| Logs | `stage0*.log`, `stage12.log`, `stage3_quick.log`, `stage3_full.log`, `stage3_map.log`, `report_2d.log` |
