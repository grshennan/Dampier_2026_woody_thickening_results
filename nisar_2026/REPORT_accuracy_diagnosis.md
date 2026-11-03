# REPORT — why is the 2026 4classTF accuracy lower? (accuracy diagnosis)

Plan and context: `G:\python\dampier_veg_timeseries\nisar_2026\PLAN_accuracy_diagnosis.md` (overview) and one
stage file per section below. One section per stage. All runs: 5 label sites (Bunda Bunda, JPP MVT and Quandong
MVT excluded), native NISAR 20 m grid, XGB, n_iter 200, stratified 5-fold K-fold, balanced weights, n_jobs 10.
K-fold numbers are optimistic (random folds under spatial autocorrelation; `REPORT_nisar_2026.md` §6).

## 0. Baseline and noise band (2026-10-28)

Stage file: `PLAN_stage0_baseline.md`. Stack for every run: `E:\nisar_2026\feature_stack_2026_gslc.tif`
(4classTF, `--merge-low`; 34,715 px, no rows dropped).

**Code.** `classify_2026.py` has 3 new options (docstring divergences 8–10): `--seed` (random_state of the folds,
the search and the model), `--s2-only` (13 S2 bands; the rows are the same as for the full stack, so the folds are
the same), `--no-map`. The defaults do not change. New scripts in `nisar_2026\diagnosis\`:
`repro_check.py`, `stage0_summary.py`, `map_agreement.py`, `stage0_chain.ps1` (the detached chain).

- Log: `G:\python\dampier_veg_timeseries\nisar_2026\diagnosis\stage0.log`.
- Summary: `G:\python\dampier_veg_timeseries\nisar_2026\outputs\diagnosis\stage0.csv`.
- Model folders: `outputs\classify_4classTF_GSLC_seedcheck\`, `outputs\classify_4classTF_GSLC_seed<1–5>\`,
  `outputs\classify_4classTF_S2only\`.
- Map: `G:\python\dampier_veg_timeseries\nisar_2026\outputs\XGB_DP_S2_2026_4classTF_S2only.tif`.

### 0.1 Reproduction check — passed

`--seed 42` (`4classTF_GSLC_seedcheck`) gives a `metrics.csv` and a `confusion_kfold.csv` that are exactly equal
to `outputs\classify_4classTF_GSLC\` (`repro_check.py`, no tolerance). The runs are deterministic, so all of the
spread below comes from the seed.

### 0.2 Seed runs (6 values: seed 42 and seeds 1–5)

Seeds 1–5 ran after the check. After seed 3, the user asked to stop the seeds if seed 3 was not significantly
different (test: OA inside mean ± 2 SD of the first 3 runs, 0.704–0.718). Seed 3 was outside (0.732), so seeds 4
and 5 also ran.

| Run | OA | balanced | kappa | F1w | F1 NF | F1 LF | F1 F | F1 TF | max_depth | search score |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 42 | .707 | .738 | .554 | .718 | .753 | .619 | .768 | .555 | 4 | .7408 |
| seed 1 | .712 | .738 | .560 | .723 | .753 | .620 | .775 | .561 | 3 | .7406 |
| seed 2 | .714 | .737 | .563 | .724 | .751 | .621 | .779 | .566 | 3 | .7406 |
| seed 3 | **.732** | .734 | **.582** | **.740** | .754 | .629 | **.800** | **.590** | 8 | .7386 |
| seed 4 | .713 | .737 | .562 | .724 | .753 | .620 | .777 | .563 | 3 | .7401 |
| seed 5 | .711 | .738 | .559 | .722 | .755 | .621 | .773 | .560 | 6 | .7411 |
| **mean** | **.715** | .737 | .563 | .725 | .753 | .622 | .779 | .566 | | |
| SD | .0087 | .0015 | .0096 | .0076 | .0012 | .0035 | .0109 | .0123 | | |
| **noise band (2 SD)** | **.017** | .003 | **.019** | **.015** | .002 | .007 | .022 | .025 | | |

K-fold precision / recall per class, mean (SD) over the 6 runs:

| Class | P | R |
|---|---|---|
| Non-forest | .784 (.006) | .725 (.004) |
| Low forest | .560 (.007) | .698 (.004) |
| Forest | .865 (.002) | .708 (.020) |
| Tall forest | .433 (.021) | .817 (.021) |

K-fold OA per site, mean (SD; min–max): 9 Mile .621 (.011; .610–.639), Biota track .750 (.013; .737–.776),
Bunda Extra .716 (.009; .710–.735), Country Downs .741 (.006; .736–.752), South Dampier .705 (.004; .701–.712).

Importances (gain share, seeds): S2 .505–.547, GSLC backscatter .235–.297, GSLC decomposition .189–.233; 168 A
.296–.332, 147 D .155–.172. EVI and NDVI are first and second in every run (`kfold_report_2026.py`,
`outputs\importance_record.csv`).

**Finding: the search score is not the reported metric.** The search scores F1w **with the balanced sample
weights** (scikit-learn passes `sample_weight` to the scorer as well as to the fit). That score is the same, to
0.0001, as the balanced-weighted F1w of each run's K-fold predictions. Over the 6 seeds it is almost flat
(.7386–.7411, a range of 0.25 pp), while the unweighted OA ranges from .707 to .732 (2.5 pp). So the search cannot
tell these candidates apart, but they differ in OA. Seed 3 chose deeper trees (max_depth 8, lr .025): a lower
balanced score but fewer false Tall forest (TF precision .475 against .41–.43 for the others; Forest recall .746
against .69–.71), so a higher OA. This affects Stage 3 (weights and scoring).

**Reading.**
1. **The noise band is 0.017 OA, 0.019 kappa and 0.015 F1w.** A difference between two runs that is smaller than
   this is noise. Seed 3 sets most of the band: without it, the OA SD is 0.003 (band 0.006). The band is from 6
   runs, so it is itself uncertain; it is the band the later stages use.
2. Balanced accuracy is very stable (band 0.003), because the search optimises a class-balanced score.
3. The earlier 4classTF runs (`REPORT_nisar_2026.md` §10–11: GCOV .711, GSLC .707, one acquisition .700–.713) are
   all inside the band. Their differences are noise.

### 0.3 S2-only run (`4classTF_S2only`, seed 42, with a map)

| Run | Features | OA | balanced | kappa | F1w | F1 NF / LF / F / TF |
|---|---|---|---|---|---|---|
| NISAR + S2, GSLC (6 seeds, mean) | 39 | .715 | .737 | .563 | .725 | .753 / .622 / .779 / .566 |
| NISAR + S2, GSLC (seed 42, `4classTF_GSLC`) | 39 | .707 | .738 | .554 | .718 | .753 / .619 / .768 / .555 |
| NISAR + S2, GCOV (`4class`, seed 42) | 39 | .711 | .738 | .559 | .722 | .756 / .621 / .773 / .562 |
| **S2 only** | 13 | **.696** | .724 | .537 | .707 | .735 / .597 / .763 / .549 |
| NISAR only (`4class_NISARonly`) | 26 | .610 | .653 | .431 | .629 | .705 / .556 / .666 / .391 |

S2-only K-fold P / R: Non-forest .747 / .723, Low forest .542 / .663, Forest .856 / .689, Tall forest .412 / .823.
Per site OA: 9 Mile .613, Biota track .730, Bunda Extra .709, Country Downs .732, South Dampier .671. Best
parameters: max_depth 7, lr .020; search score .7272 (NISAR + S2: .7386–.7411).

**S2-only importances (gain):** NDVI .324, EVI .319, BSI .052, SWIR-3 .043, red .036, green .034, red edge 2 .030,
blue .029, SWIR-2 .028, red edge 1 .028. The two vegetation indices carry 64 % of the gain.

**Gain from each source over the other, at the label pixels** (NISAR + S2 seed mean minus the one-source run, in
noise-band units = gain / 2 SD). The NISAR-only run is `4class_NISARonly` (GCOV stack, seed 42, one run; there is
no GSLC NISAR-only run), so its own seed noise is not measured either:

| Metric | S2 added to NISAR | noise bands | NISAR added to S2 | noise bands |
|---|---|---|---|---|
| OA | +.105 | 6.0 | +.019 | 1.1 |
| kappa | +.132 | 6.9 | +.026 | 1.4 |
| F1w | +.096 | 6.3 | +.018 | 1.2 |

Per metric, NISAR added to S2:

| Metric | gain | noise bands |
|---|---|---|
| OA | +.019 | 1.1 |
| balanced | +.013 | 4.1 |
| kappa | +.026 | 1.4 |
| F1w | +.018 | 1.2 |
| F1 Non-forest | +.018 | 7.3 |
| F1 Low forest | +.025 | 3.6 |
| F1 Forest | +.015 | 0.7 |
| F1 Tall forest | +.017 | 0.7 |

At the same seed (seed 42 against seed 42) the OA gain is only +.011 (0.6 bands). The S2-only run has one seed,
so its own seed noise is not measured; the gain in OA, kappa and F1w is at the edge of the band.

**Map agreement** (64,392,143 px valid in every map; `map_agreement.py`, 1 km blocks, 20 draws, null SD ≤ 0.002).
`NISARonly` = `XGB_DP_NISAR_2026_4class_NISARonly.tif` (GCOV stack); `GSLC` / `GCOV` = the NISAR + S2 maps
`XGB_DP_NISAR_S2_2026_4classTF_GSLC.tif` / `XGB_DP_NISAR_S2_2026_4class.tif`:

| Map A | Map B | agreement | null | lift |
|---|---|---|---|---|
| S2 only | NISAR + S2 GSLC | 0.692 | 0.328 | 0.361 |
| S2 only | NISAR + S2 GCOV | 0.682 | 0.328 | 0.350 |
| NISAR only | NISAR + S2 GSLC | 0.744 | 0.387 | 0.358 |
| NISAR only | NISAR + S2 GCOV (same stack) | 0.752 | 0.392 | 0.362 |
| S2 only | NISAR only | 0.510 | 0.332 | 0.177 |
| NISAR + S2 GCOV | NISAR + S2 GSLC (§10) | 0.935 | 0.399 | 0.536 |

| Class fraction | NISAR + S2 (GSLC) | NISAR + S2 (GCOV) | S2 only | NISAR only |
|---|---|---|---|---|
| Non-forest | .555 | .572 | .371 | .525 |
| Low forest | .216 | .204 | .316 | .312 |
| Forest | .194 | .190 | .267 | .140 |
| Tall forest | .035 | .034 | .047 | .023 |

**Reading.**
1. **At the label pixels, the two sources overlap, and S2 is the stronger single source.** S2 alone is 8.6 pp
   better than NISAR alone. S2 adds 10.5 pp to NISAR (6 noise bands); NISAR adds 1.9 pp to S2 (1.1 bands, at the
   edge of the noise). The NISAR gain is real for the two low classes (F1 Non-forest +1.8 pp = 7 bands, Low forest
   +2.5 pp = 3.6 bands) and for balanced accuracy (4 bands), and it is noise for Forest and Tall forest (0.7 bands
   each). This is a K-fold statement at 5 sites only; it does not make S2 the primary source of the map (item 2).
2. **Away from the labels, both sources shape the map equally.** By lift, the S2-only and the NISAR-only maps are
   the same distance from the combined map (0.361 and 0.358; the higher raw agreement of NISAR only, 0.744
   against 0.692, comes from a class mix nearer to the combined map, which raises its null). The two one-source
   maps disagree strongly with each other (0.510, lift 0.177). The Non-forest area of the combined map (.555) is
   near NISAR only (.525), not S2 only (.371): **NISAR sets the Non-forest extent.** For Forest, the combined map
   (.194) is between the two (.140 and .267). As in `REPORT_nisar_2026.md` §11, the K-fold metrics at the 5 sites do
   not show how much each source changes the map, and no labels exist away from the sites to say which map is
   nearer to the truth.
3. **For the overview question:** the gap to the Yawuru numbers (0.77–0.80) is not explained by noise (the gap is
   about 4 noise bands). The S2-only 2026 OA (.696) is the same as the Yawuru S2-only OA (.692). Yawuru PALSAR-2
   adds about 8–11 pp over S2 (`PLAN_accuracy_diagnosis.md`), and NISAR here adds about 2 pp; NISAR alone (.610) is
   also below Yawuru PALSAR-2 alone (.641). This is reference only (different sites, grid and classes), but it
   agrees with the Stage 1 hypothesis that the NISAR features do not yet carry the structure information that
   they could. The NISAR-only number is from one GCOV run; Stage 1 should give a GSLC NISAR-only run too.

### 0.4 Open points from Stage 0

- **The two one-source maps disagree strongly (to come back to; user, 2026-10-28).** S2 only against NISAR
  only: 0.510, null 0.332, lift 0.177 — about half the lift of either one against the combined map. Where they
  disagree, and which is nearer the truth away from the sites, is not known. Stage 1.5 repeats the test on the
  corrected NISAR (`PLAN_stage2_sites.md`, Stage 1.5 item 6).
- A GSLC NISAR-only run is added to Stage 1 (1.0b) as the before-reprocessing reference (the NISAR-only number
  above is from one GCOV run). The NISAR-only revisit on the new stack is Stage 1.5.

- `stage0_summary.py` prints its Markdown table with `tabulate` (installed by the user 2026-10-28; the script now
  runs to the end).
- The search scorer uses the balanced weights (0.2). Stage 3 should decide whether the search scores by the
  weighted or the unweighted metric.

### 0.5 GSLC NISAR-only baseline (Stage 1 step 1.0b, 2026-10-28)

The GSLC NISAR-only run was made before any reprocessing: `--merge-low --sar-only --stack
E:\nisar_2026\feature_stack_2026_gslc.tif` (26 features).
- Seed 42 with a map (`4classTF_GSLC_SAR`). Map: `G:\python\dampier_veg_timeseries\nisar_2026\outputs\XGB_DP_NISAR_2026_4classTF_GSLC_SAR.tif`.
- Seeds 1 and 2 without a map (user approved).
- Chain: `diagnosis\stage1_0b_chain.ps1`. Log: `diagnosis\stage1_0b.log`. Summary:
  `G:\python\dampier_veg_timeseries\nisar_2026\outputs\diagnosis\stage1_sar_noise.csv`.
- Seed 2 was stopped once by a pause for RAM (user) and was run again from the start.

| Run | OA | balanced | kappa | F1w | F1 NF | F1 LF | F1 F | F1 TF |
|---|---|---|---|---|---|---|---|---|
| seed 42 | .609 | .655 | .431 | .628 | .706 | .557 | .664 | .389 |
| seed 1 | .611 | .656 | .433 | .631 | .707 | .557 | .667 | .393 |
| seed 2 | .606 | .653 | .427 | .625 | .706 | .554 | .660 | .387 |
| **mean** | **.609** | .655 | .430 | .628 | .706 | .556 | .663 | .390 |
| SD | .0028 | .0013 | .0031 | .0027 | .0004 | .0018 | .0038 | .0029 |
| **NISAR-only noise band (2 SD)** | **.006** | .003 | **.006** | **.005** | .001 | .004 | .008 | .006 |
| `4class_NISARonly` (GCOV stack, seed 42) | .610 | .653 | .431 | .629 | .705 | .556 | .666 | .391 |

- K-fold P / R, mean of 3 seeds (SD ≤ 0.005): Non-forest .740 / .675, Low forest .481 / .658, Forest .821 / .556,
  Tall forest .266 / .728.
- OA per site, mean of 3 seeds: 9 Mile .490, Biota track .624, Bunda Extra .627, Country Downs .623, South
  Dampier .637.
- Importances (gain): the GSLC decomposition has .615–.635 and the GSLC backscatter .365–.385; 147 D has
  .47–.49 and 168 A .51–.53. The top features (seed 42) are 147 D Pu .183, 168 A gHV .146, 147 D DpRSI .136,
  168 A gHH .132, 168 A Pu .061 and 168 A DpRSI .054.

**Gain of each source over the other** (seed means; NISAR + S2 = the 6 Stage 0 seeds; S2 only = 1 run):

| Metric | S2 added to NISAR | in Stage 0 bands (in NISAR-only bands) | NISAR added to S2 | in Stage 0 bands |
|---|---|---|---|---|
| OA | +.106 | 6.2 (19) | +.019 | 1.1 |
| kappa | +.133 | 7.0 (21) | +.026 | 1.4 |
| F1w | +.097 | 6.5 (18) | +.018 | 1.2 |

**Map agreement** (1 km blocks, 20 draws):

| Map A | Map B | agreement | null | lift |
|---|---|---|---|---|
| GSLC NISAR only | S2 only | 0.516 | 0.330 | 0.186 |
| GSLC NISAR only | NISAR + S2 GSLC | 0.747 | 0.378 | 0.371 |
| GSLC NISAR only | GCOV NISAR only (`4class_NISARonly`) | 0.927 | 0.387 | 0.541 |
| *(§0.3)* S2 only | NISAR + S2 GSLC | 0.692 | 0.328 | 0.361 |

Class fractions of the GSLC NISAR-only map: Non-forest .499, Low forest .321, Forest .156, Tall forest .024. The
GCOV NISAR-only map has .525 / .312 / .140 / .023.

**Reading.**
1. **The NISAR-only noise band is 0.006 OA**, about 3 times narrower than the Stage 0 NISAR + S2 band (0.017,
   which seed 3 widened). This is the band for the main metric of Stage 1.
2. **GSLC and GCOV NISAR only are the same** at the labels (OA .609 against .610, 0.2 bands) and nearly the same in
   the map (0.927 agreement). The current GSLC stack, with its σ0 backscatter, is therefore no worse than the
   GCOV γ0 stack. Stage 1.1 tests whether the corrected radiometry changes this.
3. **The one-source disagreement does not depend on the NISAR product.** GSLC NISAR only against S2 only gives
   0.516 (lift 0.186), the same as with GCOV (0.510, lift 0.177). Each one-source map is about the same distance
   from the combined map (lift 0.371 for NISAR, 0.361 for S2).
4. Without S2, Forest recall is low (.556) and Tall forest precision is low (.266). NISAR alone cannot separate
   Forest from Tall forest well. Of the 19,204 Forest px, 3,927–4,060 are predicted Tall forest and 4,373–4,420
   are predicted Low forest.

## 1. Stage 1 — NISAR reprocessing (`PLAN_stage1_nisar.md`, started 2026-10-28)

### 1.0 Inventory (read only, 2026-10-28)

**Inputs: all present.** The 4 acquisitions exist as GCOV and as GSLC h5, 2 frames each (147 D frames 097 + 098,
168 A frames 168 + 169):

| Track | Date | GCOV h5 | GSLC h5 |
|---|---|---|---|
| 147 D | 2026-09-28 | `E:\NISAR\Dampier\GCOV\` cycle 031 | `E:\NISAR\Dampier\GSLC\` cycle 031 |
| 147 D | 2026-10-10 | cycle 032 | cycle 032 |
| 168 A | 2026-09-29 | cycle 031 | cycle 031 |
| 168 A | 2026-10-11 | cycle 032 | cycle 032 |

**The stack grid** is EPSG:32751, 20 m, origin (393160, 8197760), 9081 columns × 11625 rows. The GCOV h5
x / y coordinates are cell centres (x0 = 360010), so the cell edges are on multiples of 20 m. The GSLC 20 m grid
(origin 394700, 8191420) is offset from the stack grid by 77 columns and 317 rows exactly. **All grids align.**

**What exists:**

| Product | Path | Grid | Content and tags | Use in 1.1 |
|---|---|---|---|---|
| GCOV per-date mosaics (not filtered) | `G:\python\nisar\geotiff\NISAR_GCOV_<start>_t<trk>_freqA_{HHHH,HVHV,mask,numberOfLooks,rtcGammaToSigmaFactor}.tif` | stack grid | frames joined, `mosaic_frames.py` (first positive value for power, first value for the others) | the factor, looks and mask for 1.1.1 and 1.1.2 come from here; no h5 read needed for them |
| GCOV γ0 Refined Lee 5 × 5 | `G:\python\nisar\geotiff_rflee5\..._{HHHH,HVHV}.tif` | stack grid | `rflee.py` 5 × 5, channels HHHH + HVHV, span HHHH + HVHV, looks = numberOfLooks, side centre | **reuse** for γ0 HH / HV in 1.1.1 (same filter and grid) |
| GSLC C2 σ0 Refined Lee 5 × 5, per date and pooled | `E:\nisar_2026\c2_20m_rflee5_<trk>_<date>.tif` and `..._<date1>_<date2>.tif` | stack grid | C11, C22, C12 re / im, n_looks; β0 × sin θ (ellipsoid); NESZ −28.07 to −28.17 dB (median of the frame metadata); pooled = look-weighted mean of the filtered dates | input to 1.1.2 (γ0) and 1.1.4 |
| GSLC decomposition, ws 5, per date and pooled | `E:\nisar_2026\dcmp_20m_rflee5_ws5_<trk>_<date(s)>.tif` | stack grid | 12 bands (HH / HV σ0 dB, Pd, Ps, Pu dB, DpRBI, DpRSI, alpha_dp, H, DoP, alpha_cloude, n_looks); min looks 8; S_norm over the mainland mask of the stack grid (`feature_stack_2026.py` replaces `C.boundary_mask`) | **reuse** in 1.1.4: the inputs (σ0 C2) do not change; this agrees with 1.1.6 |
| GSLC per-date C2 and decomposition, regional | `E:\NISAR\Dampier\GSLC_dcmp_20m_rflee5\` | GSLC grid (394700, 8191420) | min looks 6, S_norm over the GSLC extent | not used (different S_norm domain) |
| InSAR coherence | `E:\NISAR\Dampier\GSLC_coh_20m\` | GSLC grid | 15 pairs; the pairs for 1.3 are `coh_20260928_20261010_147_D` and `coh_20260929_20261011_168_A` | 1.3 |

**Refined Lee windows.** `G:\python\GDV\GEDI\rflee.py` supports windows 3, 5, 7, 9 and 11 (`WINDOW_PARAMS`), so
1.2 needs no filter change.

**GCOV metadata (147 D frame 097, 2026-09-28).** `rtcGammaToSigmaFactor` p1 / p50 / p99 = 0.559 / 0.755 / 0.832
(−2.5 / −1.2 / −0.8 dB). This agrees with the GSLC σ0 − GCOV γ0 difference of −1.35 to −1.49 dB in report §10.
`noiseEquivalentBackscatter` (324 × 332, its own x / y): HH −29.6 / −27.1 / −25.0 dB, HV −29.9 / −27.4 /
−25.1 dB (p1 / p50 / p99). The NESZ changes by about 4.6 dB across the frame, so 1.1 will interpolate it
to the grid and will not use one value per acquisition.

**Gaps for 1.1:** there is no GCOV σ0 file, no GSLC γ0 file and no GCOV (no-phase) decomposition. The interpolated
NESZ grid does not exist yet.

### 1.1 Correct radiometry and decompositions (2026-10-28)

**Code.**
- `G:\python\GDV\GEDI\gcov_decomposition.py`: a port of `Dual_pol_powers_decomposition_grd_tif.py` (no phase),
  6 bands. Self-test on a 512 × 512 tile, with the script's equations run on float64 inputs: the largest
  |port − script| is 2e-7 for the powers, 5e-6 for DpRSI and 1.6e-4° for alpha_dp. All are within 1e-5 × max.
  DpRSI and alpha_dp are sensitive where |s1|n → 1. Pu + Pd + Ps = s0 to 3e-16. The tile uses both NESZ
  branches (86.8 % valid).
- `feature_stack_2026.py`: modes `gcov_nophase` and `gslc_gamma`, options `--rflee-win` and `--dcmp-ws`, and the
  per-date and pooled products (module docstring). The `gcov` and `gslc` modes do not change.
- `diagnosis\stage1_radiometry.py` (makes the products and does the check); `kfold_report_2026.group()` has the
  new groups `GSLC_g0` and `GCOV_dcmp`.

**NESZ.** This is the GCOV metadata `noiseEquivalentBackscatter` HH / HV. It is interpolated bilinearly from its
1 km grid to the cell centres, and the frames are joined with the first valid value. One value per acquisition is
not used, because the NESZ changes by about 4.6 dB across a frame (§1.0). In the mainland, NESZ HH p1 / p50 / p99
is −29.7 / −27.9 / −25.5 dB (147 D, 2026-09-28). The decomposition compares the NESZ with σ0, as the GSLC chain
does. The metadata gives the units as "DN²" and does not state the normalisation.

**Timing.** The first acquisition (147 D 2026-09-28: γ0, σ0, factor, looks, NESZ) took 2.0 min. All products for
both tracks, with the radiometry check, took 30.6 min. The 1.0b run used the CPU at the same time.

**GSLC γ0: pool, then divide.** The output uses the pooled C2 divided by the pooled GCOV factor (weights: the GSLC
n_looks of each date). To check it, each date was divided before pooling, using the existing per-date filtered C2.
|Δ| p50 / p99 / p99.9 / max:
- 147 D: 0.00002 / 0.0006 / 0.0015 / 0.09–0.11 dB;
- 168 A: 0.00000 / 0.00005 / 0.0002 / 0.02 dB.

So p99.9 < 0.01 dB. The few larger values are at the swath edges, where only one date has a factor.

**Radiometry check (1.1.3), pooled, mainland** (`outputs\diagnosis\stage1_radiometry.csv`):

| Track | Pol | GSLC − GCOV σ0, median | GSLC − GCOV γ0, median | IQR | r at zero shift (largest other) | Pass |
|---|---|---|---|---|---|---|
| 147 D | HH | −0.186 dB | −0.186 dB | 0.23 dB | 0.995 (0.963) | yes |
| 147 D | HV | −0.194 dB | −0.194 dB | 0.24 dB | 0.995 (0.965) | yes |
| 168 A | HH | −0.190 dB | −0.190 dB | 0.24 dB | 0.995 (0.952) | yes |
| 168 A | HV | **−0.202 dB** | **−0.202 dB** | 0.24 dB | 0.995 (0.960) | **no** (limit 0.2 dB) |

The γ0 offset is the same as the σ0 offset, because both products are divided by the same factor. The grids align,
because r is largest at zero shift. The old error, GSLC σ0 against GCOV γ0, was −1.35 to −1.49 dB (report §10). The
residual offset is about 7 times smaller than that.

**Diagnosis of the offset** (unfiltered, per date, every 4th row of the mainland):
- It is also present without the filter: per-pixel median −0.19 to −0.21 dB, ratio of means −0.16 to −0.18 dB. So
  the filter and the pooling do not cause it.
- **It is not a constant.** By GCOV σ0 level, from the darkest to the brightest fifth of pixels, it goes from
  −0.33 to −0.04 dB (HH) and from −0.36 to −0.05 dB (HV). By RTC factor, from the lowest to the highest fifth, it
  goes from −0.22 to −0.14 dB (HH) and from −0.25 to −0.15 dB (HV). The 4 dates give the same pattern to within
  0.03 dB.
- The cause is not known. It changes with the level, so it is not a calibration constant. A simple additive noise
  term at the NESZ level would give a much larger offset in HV than the one measured, so that does not explain it
  either. The candidates are differences between GSLC and GCOV in the area normalisation (ellipsoid against
  terrain) and in the noise handling.

**Decision (user, 2026-10-28): accept the offset and continue.**
- The check was made to find gross errors, and the 0.2 dB limit was missed only by 168 A HV (σ0 and γ0), by
  0.002 dB.
- Each stack takes its backscatter from one product only.
- XGB is not sensitive to a monotonic change in one feature.

**Stacks (1.1.7).**
- `E:\nisar_2026\feature_stack_2026_gcovnp.tif` (33 bands, 11 min).
- `E:\nisar_2026\feature_stack_2026_gslcg0.tif` (39 bands, 11.5 min).
- Checks: in both stacks, the 13 S2 bands are bit-identical to `feature_stack_2026.tif`.
- GSLC γ0 against GCOV γ0 in the stacks: r 0.994–0.995 at zero shift (0.95–0.96 at a 1 px shift). Median
  −0.186 / −0.194 dB (147 D HH / HV) and −0.190 / −0.202 dB (168 A).
- Valid px: the same for 147 D; 114 fewer in GSLC for 168 A (out of 64.45 M).
- The 0.2 dB assert in `compare_new()` is now a warning, which records the accepted offset.
- GCOV decomposition: energy check 3e-16. Every mainland px is in the valid branch (HH σ0 > NESZ after the 5 × 5
  box).

**Runs (1.1.8)** (seed 42, `--merge-low`, n_iter 200, K-fold, `--no-map`; chain `diagnosis\stage1_1_chain.ps1`,
log `diagnosis\stage1_1.log`):

| Run | Features | OA | balanced | kappa | F1w | F1 NF / LF / F / TF |
|---|---|---|---|---|---|---|
| **NISAR only** | | | | | | |
| `4classTF_GCOVnp_SAR` (GCOV γ0 + GCOV decomposition, no phase) | 20 | .608 | .650 | .428 | .627 | .705 / .554 / .664 / .382 |
| `4classTF_GSLCg0_SAR` (GSLC γ0 + GSLC decomposition, phase) | 26 | .601 | .655 | .423 | .621 | .707 / .555 / .651 / .385 |
| *ref.* `4classTF_GSLC_SAR` (GSLC σ0 + GSLC decomposition), 3 seeds | 26 | .609 | .655 | .430 | .628 | .706 / .556 / .663 / .390 |
| *ref.* `4class_NISARonly` (GCOV γ0 + GSLC decomposition) | 26 | .610 | .653 | .431 | .629 | .705 / .556 / .666 / .391 |
| **NISAR + S2** | | | | | | |
| `4classTF_GCOVnp` | 33 | .709 | .739 | .558 | .720 | .756 / .622 / .770 / .556 |
| `4classTF_GSLCg0` | 39 | .717 | .739 | .566 | .727 | .758 / .622 / .779 / .571 |
| *ref.* `4classTF_GSLC`, 6 seeds | 39 | .715 | .737 | .563 | .725 | .753 / .622 / .779 / .566 |
| *ref.* `4class` (GCOV γ0 + GSLC decomposition) | 39 | .711 | .738 | .559 | .722 | .756 / .621 / .773 / .562 |

K-fold P / R:

| Run | Non-forest | Low forest | Forest | Tall forest |
|---|---|---|---|---|
| GCOVnp_SAR | .730 / .681 | .480 / .654 | .822 / .557 | .262 / .705 |
| GSLCg0_SAR | .739 / .677 | .477 / .664 | .823 / .538 | .260 / .740 |
| GCOVnp | .789 / .727 | .556 / .705 | .866 / .693 | .418 / .829 |
| GSLCg0 | .792 / .727 | .560 / .701 | .865 / .710 | .439 / .816 |

OA per site (9 Mile / Biota track / Bunda Extra / Country Downs / South Dampier):
- GCOVnp_SAR: .473 / .627 / .623 / .629 / .637
- GSLCg0_SAR: .484 / .611 / .623 / .609 / .636
- GCOVnp: .617 / .741 / .705 / .741 / .703
- GSLCg0: .622 / .754 / .716 / .742 / .706

Importances (gain share):

| Run | Backscatter (γ0) | Decomposition | S2 | 147 D / 168 A | Top 5 |
|---|---|---|---|---|---|
| GCOVnp_SAR | .424 | .576 | – | .395 / .605 | 168 A HH .163, 168 A HV .130, 147 D n_Pu .127, 147 D n_DpRSI .078, 168 A n_DpRBI .061 |
| GSLCg0_SAR | .361 | .639 | – | .490 / .510 | 147 D Pu .178, 147 D DpRSI .159, 168 A g0HH .142, 168 A g0HV .135, 168 A Pu .068 |
| GCOVnp | .257 | .195 | .548 | .149 / .303 | EVI .238, 168 A HH .130, NDVI .129, 168 A HV .061, BSI .027 |
| GSLCg0 | .231 | .222 | .547 | .166 / .287 | EVI .239, NDVI .136, 168 A g0HH .102, 168 A g0HV .065, BSI .027 |

**Reading.**
1. **The corrections do not change the accuracy beyond the noise.**
   - NISAR only, against the NISAR-only band (0.006): GCOVnp −0.001 (0.2 bands) and GSLCg0 −0.008 (1.3 bands)
     from the GSLC σ0 seed mean. Both are single seeds, so GSLCg0 is at most a marginal loss.
   - NISAR + S2, against the Stage 0 band (0.017): GCOVnp −0.006 and GSLCg0 +0.002. Both are noise.
   - So the two errors in planning (σ0 backscatter in the GSLC stacks, GSLC phase in the GCOV stacks) did not
     cost accuracy. The low NISAR contribution is not caused by the radiometry.
2. **Without phase, the GCOV decomposition is as useful as the GSLC decomposition with phase:** 6 bands per track
   against 9, the same OA. In the no-phase runs, the decomposition still has more than half of the NISAR gain.
3. **Product choice for 1.2.** GCOVnp_SAR is 0.007 above GSLCg0_SAR, which is 1.2 bands on one seed each. With S2,
   the order is reversed (+0.008 for GSLCg0, 0.5 Stage 0 bands). Neither product is better beyond the noise, so 1.2
   runs both products.

### 1.2 Speckle filter and decomposition window (2026-10-28)

Chain `diagnosis\stage1_2_chain.ps1`, log `diagnosis\stage1_2.log`. For each configuration (filter window, decomposition
ws): a new filter and a new decomposition, then a stack `E:\nisar_2026\feature_stack_2026_<gcovnp|gslcg0>_rflee<w>_ws<ws>.tif`
and a NISAR-only run `4classTF_<GCOVnp|GSLCg0>_SAR_f<w>w<ws>` (seed 42, `--no-map`). The 5 × 5 files did not change.
Each stack took about 12 min. Table: `outputs\diagnosis\stage1_2.csv` (`diagnosis\stage0_summary.py`; its SD rows mix
configurations and are not a noise band).

**NISAR-only K-fold** (one seed each; NISAR-only noise band 0.006, §0.5):

| Product | Config | OA | balanced | kappa | F1w | F1 NF / LF / F / TF | ΔOA to (5, 5) |
|---|---|---|---|---|---|---|---|
| GCOV | 5, 5 (1.1) | .608 | .650 | .428 | .627 | .705 / .554 / .664 / .382 | – |
| GCOV | 7, 5 | .602 | .657 | .427 | .623 | .713 / .560 / .652 / .375 | −.006 |
| GCOV | **9, 5** | **.615** | .662 | .438 | .634 | .716 / .561 / .669 / .390 | +.007 (1.1 bands) |
| GCOV | 5, 7 | .614 | .657 | .436 | .633 | .709 / .557 / .670 / .394 | +.006 |
| GCOV | 5, 9 | .615 | .657 | .437 | .634 | .709 / .558 / .671 / .392 | +.007 |
| GSLC | 5, 5 (1.1) | .601 | .655 | .423 | .621 | .707 / .555 / .651 / .385 | – |
| GSLC | 7, 5 | .613 | .664 | .438 | .633 | .716 / .560 / .666 / .395 | +.012 |
| GSLC | **9, 5** | **.639** | .668 | .462 | .655 | .719 / .565 / .701 / .427 | **+.038 (6.3 bands)** |
| GSLC | 5, 7 | .636 | .662 | .459 | .653 | .710 / .562 / .701 / .425 | +.035 |
| GSLC | 5, 9 | .638 | .666 | .462 | .654 | .717 / .565 / .700 / .427 | +.037 |

OA per site (9 Mile / Biota track / Bunda Extra / Country Downs / South Dampier):
- GCOV 9, 5: .487 / .628 / .619 / .634 / .655
- GSLC 9, 5: .512 / .672 / .630 / .660 / .655 (GSLC 5, 5: .484 / .611 / .623 / .609 / .636)

Importances (gain share), NISAR only:

| Run | Backscatter (γ0) | Decomposition | 147 D / 168 A | Top 5 |
|---|---|---|---|---|
| GCOVnp_SAR_f9w5 | .507 | .493 | .422 / .578 | 168 A HH .173, 168 A HV .156, 147 D n_Pu .135, 147 D HV .083, 168 A n_Pu .052 |
| GSLCg0_SAR_f7w5 | .384 | .616 | .486 / .514 | 147 D Pu .201, 168 A g0HH .151, 168 A g0HV .146, 147 D DpRSI .117, 168 A Pu .064 |
| GSLCg0_SAR_f9w5 | .387 | .613 | .480 / .520 | 147 D Pu .176, 168 A g0HV .141, 168 A g0HH .133, 147 D DpRSI .093, 168 A Pu .062 |
| GSLCg0_SAR_f5w7 | .395 | .605 | .477 / .523 | 147 D Pu .183, 168 A g0HH .135, 168 A g0HV .133, 147 D DpRSI .065, 168 A Pu .063 |
| GSLCg0_SAR_f5w9 | .424 | .576 | .477 / .523 | 147 D Pu .166, 168 A g0HH .140, 168 A g0HV .133, 147 D g0HV .066, 168 A Pu .057 |

**ENL** (`diagnosis\stage1_enl.py`, `outputs\diagnosis\stage1_enl.csv`). 20 homogeneous patches of 10 × 10 px
(200 m), selected from S2 only: NDVI < 0.15, CV of SWIR2 < 0.03, mainland, centres ≥ 2 km apart; list with centres
(EPSG:32751) in `outputs\diagnosis\stage1_enl_patches.csv` (NDVI 0.049–0.075). ENL = mean² / variance in each
patch of the filtered, pooled (2 dates) linear intensity: GCOV γ0 HH / HV, GSLC σ0 C11 / C22. In every row it is
the ENL of the ws × ws box mean of that intensity (ws 5, 7 or 9), which is what the decomposition sees. Median over the 20 patches:

| Config | GCOV 147 D HH / HV | GCOV 168 A HH / HV | GSLC 147 D HH / HV | GSLC 168 A HH / HV |
|---|---|---|---|---|
| 5, 5 | 73 / 329 | 66 / 504 | 74 / 356 | 69 / 455 |
| 7, 5 | 81 / 409 | 76 / 717 | 79 / 395 | 78 / 682 |
| 9, 5 | 82 / 494 | 91 / 836 | 81 / 517 | 86 / 749 |
| 5, 7 | 86 / 359 | 72 / 660 | 82 / 330 | 71 / 662 |
| 5, 9 | 104 / 468 | 72 / 710 | 99 / 465 | 68 / 695 |

- HH ENL goes up by only 10–40 % from (5, 5) to (9, 5) or (5, 9), although the number of pixels averaged goes up by
  more than 3 times. So the variance in these patches is mostly real variation of the target, not speckle: the
  ENL here is limited by the patches, and it does not measure the filters well.
- The spread between patches is large (HH p25–p75 about 30–220), so a "bare" patch in S2 is not always a
  homogeneous target in L-band.- HV ENL is 4–10 times HH ENL. On bare ground, HV is near the noise floor, and noise that is averaged over many looks
  and two dates is more uniform than the backscatter. So the HV ENL does not measure the speckle of a target. Not
  investigated further.

**Reading.**
1. **GSLC gains beyond the noise from a larger window: +0.035 to +0.038 OA (about 6 bands) for (9, 5), (5, 7) and
   (5, 9).** The gain is in Forest (F1 .651 → .701) and Tall forest (.385 → .427), at every site. The three
   configurations are equal within the noise. A larger filter or a larger decomposition box gives the same gain, so the
   cause is probably the smoothing of the decomposition inputs (the phase terms of C12 need more looks), not the
   γ0 bands.
2. (7, 5) GSLC gives only +0.012. It is not between (5, 5) and (9, 5) in the same way as the others. All runs have one
   seed, so a single result can be 1–2 bands out. It does not change the reading.
3. **GCOV gains at most +0.007 (about 1 band)**: without phase, the window has little effect.
4. With the larger window, GSLC is +0.024 above GCOV (both 9, 5), 4 bands. In 1.1 (5, 5), GCOV was 0.007 above.
   **So the best NISAR stack is GSLC (9, 5)**: highest OA, kappa and F1w, and it was first in the plan's
   configuration list.
5. Importances: the larger windows do not change the order. The GSLC decomposition keeps about 0.6 of the NISAR gain,
   and 147 D Pu is first in every GSLC run.

**NISAR + S2 for the best configuration of each product** (chain `diagnosis\stage1_4_chain.ps1`, log
`diagnosis\stage1_4.log`; Stage 0 noise band 0.017):

| Run | Features | OA | balanced | kappa | F1w | F1 NF / LF / F / TF | ΔOA to (5, 5) |
|---|---|---|---|---|---|---|---|
| `4classTF_GCOVnp_f9w5` | 33 | .726 | .739 | .577 | .735 | .758 / .627 / .791 / .578 | +.017 (1.0 band) |
| `4classTF_GSLCg0_f9w5` | 39 | .715 | .743 | .565 | .725 | .761 / .625 / .776 / .563 | −.002 (0.1 band) |

`4classTF_GCOVnp_f9w5`: P / R NF .783 / .735, LF .572 / .695, F .866 / .729, TF .453 / .797. OA per site .639 / .763 /
.722 / .747 / .715. Importances: S2 .526, GCOV γ0 .290, GCOV decomposition .184; 147 D .155, 168 A .319; top 5 EVI
.228, NDVI .127, 168 A HH .126, 168 A HV .092, nbart_red .027.

`4classTF_GSLCg0_f9w5`: P / R NF .792 / .734, LF .562 / .705, F .869 / .701, TF .426 / .831. OA per site .628 / .745 /
.713 / .738 / .710. Importances: S2 .497, GSLC γ0 .268, GSLC decomposition .235; 147 D .179, 168 A .323; top 5 EVI
.228, 168 A g0HH .114, NDVI .103, 168 A g0HV .088, BSI .025.

6. **With S2, the window gain is gone.** GSLC (9, 5) + S2 is −0.002 from GSLC (5, 5) + S2. GCOV (9, 5) + S2 is +0.017
   (1 band). GCOV − GSLC at (9, 5) is +0.011, within the band. The Forest / Tall forest information that the larger
   window adds to NISAR is already in S2. So the speckle configuration does not explain the low 2026 accuracy.

### 1.3 InSAR coherence — dropped (user, 2026-10-28)

The user decided that coherence is outside the scope of this work, so no coherence features were made or tested. A
fact found while preparing it: the 147 D GSLC of 2026-09-28 decorrelates **both** of its pairs (09-16/09-28 and
09-28/10-10; coh_HH_win median 0.12; all other Dampier pairs 0.67–0.79). `E:\NISAR\README.md` is corrected. The pair
files in `E:\NISAR\Dampier\GSLC_coh_20m\` are kept.

### 1.4 C-band (Sentinel-1) (2026-10-28 / 29)

**Source (user, 2026-10-28):** the CDSE Sentinel-1 monthly global mosaic (STAC collection
`sentinel-1-global-mosaics`, VV / VH linear, 20 m MGRS tiles), the source of the 2019–2025 series.
- Download: `diagnosis\s1_mosaic_2026.py` (CDSE S3, ≤ 4 MB/s, size-checked) → `E:\nisar_2026\s1_mosaic\` (9 MGRS
  tiles × VV / VH for 2026-09).
- **Deviation 1:** the plan asks for S1 inside the S2 window (2026-09-20..10-10). On 2026-10-28 the October 2026 mosaic
  was not published (STAC: Sep 9 items, Oct 0), so only the **September 2026** mosaic is used. Its month overlaps the S2 window
  for 11 days only (20–30 Sep); the acquisition dates inside the mosaic were not checked.
- **Deviation 2:** the mosaic gives one radiometry only (γ0 by the CDSE product description), so the S1 decomposition
  uses those values, not σ0.
- Products (`feature_stack_2026.py --s1`): `E:\nisar_2026\s1_g0_20m_2026_M09.tif` (the tiles nest in the 20 m grid,
  joined first-valid) and `E:\nisar_2026\s1dcmp_20m_ws5_2026_M09.tif` (`gcov_decomposition.run`, C11 = VV, C22 = VH,
  ws 5, NESZ −20 dB as the script). Features: `S1_VV_dB`, `S1_VH_dB`, `S1_VHVV_dB`, `S1_RVI` and the 6 no-phase bands
  `S1_n_*`. All 10 are 100 % valid in the mainland.
- Stack: `E:\nisar_2026\feature_stack_2026_gslcg0_rflee9_ws5_s1.tif` (the best NISAR stack of 1.2 + 10 S1 bands =
  49 bands, 11 min).

**Runs** (chain `diagnosis\stage1_4_chain.ps1`, log `diagnosis\stage1_4.log`; Stage 0 noise band 0.017):

| Run | Features | OA | balanced | kappa | F1w | F1 NF / LF / F / TF |
|---|---|---|---|---|---|---|
| `4classTF_S1S2` (S1 + S2, `--no-nisar`) | 23 | .694 | .726 | .536 | .706 | .732 / .599 / .761 / .549 |
| *ref.* `4classTF_S2only` | 13 | .696 | .724 | .537 | .707 | .735 / .597 / .763 / .549 |
| `4classTF_GSLCg0_f9w5_S1` (NISAR + S1 + S2) | 49 | .726 | .743 | .578 | .735 | .760 / .631 / .790 / .580 |
| *ref.* `4classTF_GSLCg0_f9w5` (NISAR + S2) | 39 | .715 | .743 | .565 | .725 | .761 / .625 / .776 / .563 |

`4classTF_S1S2`: P / R NF .748 / .717, LF .538 / .675, F .861 / .681, TF .410 / .830. OA per site .608 / .727 / .713 /
.732 / .669. Importances: S2 .774, S1 .226; top 5 EVI .286, NDVI .247, S1_n_alpha_dp .063, S1_n_Ps .036, BSI .035.

**Reading (S1 + S2).** S1 adds nothing to S2: −0.002 OA (0.1 band), and the per-class F1 and per-site OA are the same
as S2 only. The model gives S1 0.23 of the gain, but this is gain that S2 already supplies. At Yawuru, S1 added
2.6 pp to PALSAR-2 + S2 (reference only, plan); here it adds nothing to S2 alone.

`4classTF_GSLCg0_f9w5_S1`: P / R NF .792 / .731, LF .570 / .706, F .870 / .724, TF .451 / .812. OA per site .644 /
.759 / .734 / .738 / .718. Importances: S2 .423, GSLC γ0 .274, GSLC decomposition .208, S1 .095; 147 D .148, 168 A
.334; top 5 NDVI .126, EVI .120, 168 A g0HH .117, 168 A g0HV .103, BSI .071. Best parameters: max_depth 3, lr .158
(the NISAR + S2 runs: max_depth 7, lr about .02); search score .7467.

**Reading (NISAR + S1 + S2).** +0.011 OA over NISAR + S2 on the same stack (0.65 band), and the same as GCOV (9, 5) +
S2 (.726). The per-site gain is 0.000 (Country Downs) to +0.021 (Bunda Extra) and in Forest F1 (+0.014), but it is within the noise. S1 has
0.095 of the gain. With one month of S1 before the flights, C-band does not change the 2026 accuracy. This run used
the new environment (below); with S2 the environment effect is +0.0006, so the comparison stands.

### Environment change during Stage 1 (2026-10-29 08:13)

During a pause for the user's own work, the conda environment `general` was recreated from
`C:\Users\glens\general-slim.yml`:

| Package | Before (`C:\Users\glens\general.yml`, export of 2026-09-02) | After |
|---|---|---|
| numpy | 1.26.4 | 2.5.4 |
| xgboost | 3.0.5 (CUDA build) | 3.4.2 (CPU build) |
| scikit-learn | 1.9.0 | 1.9.1 |
| scipy | 1.17.1 | 1.18.1 |

- Old environment: Stage 0, 1.0b, 1.1, 1.2, and in 1.4 `4classTF_GCOVnp_f9w5`, `4classTF_GSLCg0_f9w5` (no map) and
  `4classTF_S1S2`. New environment: `4classTF_GSLCg0_f9w5_S1` and every run of §1.5 and Stage 1.5.
- The search draws the same candidate list from the same seed (the same parameter sets occur in both), but the CV
  scores change, so the selected candidate can change.
- **Measured effect, same stack, seed 42:** NISAR + S2 (`4classTF_GSLCg0_f9w5`) OA .7149 → .7155, kappa .5650 →
  .5660, F1w .7255 → .7262, same best parameters. NISAR only (`4classTF_GSLCg0_SAR_f9w5` → `4classTF_GSLCg0_f9w5_SAR`)
  OA .639 → .631, different best parameters (max_depth 7 → 5). The NISAR-only change (−0.008) is 1.3 NISAR-only bands
  (§0.5), so it is at the edge of the seed noise.
- Comparisons in this report that cross the change carry this caveat. The Stage 1.5 seed band is measured wholly in
  the new environment, which Stages 2 and 3 will use.
- The numpy 2 change broke `diagnosis\map_agreement.py` (uint8 compared with −1); fixed (cast to int16 first).

### 1.5 Result of Stage 1 (2026-10-29)

**All Stage 1 runs.** Gain = ΔOA to the reference, in noise-band units. References: NISAR only = `4classTF_GSLC_SAR`
(3 seeds, mean .609, band .006, §0.5); NISAR + S2 = `4classTF_GSLC` (6 seeds, mean .715, band .017, §0.2). One seed
(42) unless stated; *new env* = after the environment change.

| Run | OA | kappa | F1w | F1 NF / LF / F / TF | Gain (bands) |
|---|---|---|---|---|---|
| **NISAR only** | | | | | |
| `4classTF_GCOVnp_SAR` (1.1) | .608 | .428 | .627 | .705 / .554 / .664 / .382 | −0.2 |
| `4classTF_GSLCg0_SAR` (1.1) | .601 | .423 | .621 | .707 / .555 / .651 / .385 | −1.3 |
| `…GCOVnp_SAR_f7w5` / `_f9w5` / `_f5w7` / `_f5w9` | .602 / .615 / .614 / .615 | .427 / .438 / .436 / .437 | .623 / .634 / .633 / .634 | §1.2 | −1.2 / +1.0 / +0.8 / +1.0 |
| `…GSLCg0_SAR_f7w5` | .613 | .438 | .633 | .716 / .560 / .666 / .395 | +0.7 |
| **`…GSLCg0_SAR_f9w5`** | **.639** | .462 | .655 | .719 / .565 / .701 / .427 | **+5.0** |
| `…GSLCg0_SAR_f5w7` / `_f5w9` | .636 / .638 | .459 / .462 | .653 / .654 | §1.2 | +4.5 / +4.8 |
| `4classTF_GSLCg0_f9w5_SAR`, 3 seeds, *new env* | .627 | .451 | .645 | .719 / .563 / .684 / .415 | +3.0 |
| **NISAR + S2** | | | | | |
| `4classTF_GCOVnp` / `4classTF_GSLCg0` (1.1) | .709 / .717 | .558 / .566 | .720 / .727 | §1.1 | −0.4 / +0.1 |
| `4classTF_GCOVnp_f9w5` (1.2) | .726 | .577 | .735 | .758 / .627 / .791 / .578 | +0.6 |
| `4classTF_GSLCg0_f9w5` (1.2) | .715 | .565 | .725 | .761 / .625 / .776 / .563 | 0.0 |
| `4classTF_S1S2` (1.4, no NISAR) | .694 | .536 | .706 | .732 / .599 / .761 / .549 | −1.2 |
| `4classTF_GSLCg0_f9w5_S1` (1.4), *new env* | .726 | .578 | .735 | .760 / .631 / .790 / .580 | +0.6 |
| `4classTF_GSLCg0_f9w5`, 6 seeds, *new env* | .717 | .568 | .727 | .760 / .627 / .779 / .565 | +0.1 |

**The new reference stack** (chosen 2026-10-29; Stages 1.5, 2 and 3 use it):
- Path: `E:\nisar_2026\feature_stack_2026_gslcg0_rflee9_ws5.tif` (39 bands, 20 m, EPSG:32751, the NISAR grid).
- Run tag: `4classTF_GSLCg0_f9w5` (`classify_2026.py --merge-low --n-iter 200 --n-jobs 10 --stack <path>`; NISAR only:
  `--sar-only`, tag `4classTF_GSLCg0_f9w5_SAR`).
- Features (39): per track (147 D, 168 A) GSLC γ0 `g0HH_dB`, `g0HV_dB`, `g0HVHH_dB`, `g0RVI` (pooled C2 / pooled GCOV
  factor) and the 9-band GSLC decomposition (Pd, Ps, Pu dB, DpRBI, DpRSI, alpha_dp, H, DoP, alpha_cloude), Refined Lee
  **9 × 9**, decomposition ws 5; and the 13 S2 bands (10 reflectances, NDVI, EVI, BSI).
- Why: the NISAR + S2 candidates (.715–.726) are all within the Stage 0 band. The only gain beyond the noise in Stage 1
  is its NISAR-only gain (+0.038, 5 bands). GCOV (9, 5) + S2 has the highest single OA (.726, 0.6 band), but its
  NISAR-only result is 4 NISAR-only bands lower. S1 is not included: no gain beyond the noise, 10 more features,
  September only, and its decomposition is on γ0.

**Maps for the Stage 1.5 revisit** (seed 42, *new env*):
- `G:\python\dampier_veg_timeseries\nisar_2026\outputs\XGB_DP_NISAR_2026_4classTF_GSLCg0_f9w5_SAR.tif` (NISAR only):
  K-fold OA .631, kappa .455, F1w .648, against the 1.0b GSLC NISAR-only run (seed 42) .609 / .431 / .628: +0.022 OA.
- `G:\python\dampier_veg_timeseries\nisar_2026\outputs\XGB_DP_NISAR_S2_2026_4classTF_GSLCg0_f9w5.tif` (NISAR + S2):
  K-fold OA .716, kappa .566, F1w .726.

**Answer for Stage 1:** the NISAR reprocessing (corrected radiometry, γ0 GSLC backscatter, a no-phase GCOV
decomposition, larger speckle and decomposition windows) and C-band S1 do **not** raise the NISAR + S2 accuracy beyond
the noise. Only the NISAR-only accuracy goes up (+0.02 to +0.04 with the 9 × 9 filter on GSLC). The low 2026
accuracy is not caused by the NISAR processing.

## Stage 1.5 — repeat of Stage 0 on the new reference stack (`PLAN_stage2_sites.md`, 2026-10-29)

Chain `diagnosis\stage1_5_chain.ps1 -Stack E:\nisar_2026\feature_stack_2026_gslcg0_rflee9_ws5.tif -Ref
4classTF_GSLCg0_f9w5`, log `diagnosis\stage1_5.log`, 09:01–13:38, all in the new environment (the map-agreement steps were restarted once
after the numpy 2 fix to `map_agreement.py`). Tables:
`G:\python\dampier_veg_timeseries\nisar_2026\outputs\diagnosis\stage15.csv` (NISAR + S2, 6 seeds) and
`...\stage15_sar_noise.csv` (NISAR only, 3 seeds). Seed 42 is the §1.5 run with a map; the others are `--no-map`.

**S2 only:** the 13 S2 bands of the new stack are bit-identical to `E:\nisar_2026\feature_stack_2026_gslc.tif`
(0 differing px in each band, checked 2026-10-29), so the Stage 0 S2-only run (`4classTF_S2only`) applies. Not re-run.

**NISAR + S2, 6 seeds:**

| Run | OA | balanced | kappa | F1w | F1 NF | F1 LF | F1 F | F1 TF | max_depth |
|---|---|---|---|---|---|---|---|---|---|
| seed 42 | .716 | .742 | .566 | .726 | .762 | .628 | .776 | .560 | 5 |
| seed 1 | .715 | .743 | .565 | .726 | .760 | .626 | .776 | .566 | 6 |
| seed 2 | .720 | .743 | .571 | .730 | .762 | .627 | .783 | .572 | 3 |
| seed 3 | .713 | .742 | .563 | .724 | .758 | .624 | .774 | .560 | 4 |
| seed 4 | .721 | .744 | .572 | .731 | .761 | .629 | .783 | .568 | 3 |
| seed 5 | .718 | .742 | .569 | .728 | .758 | .628 | .780 | .565 | 3 |
| **mean** | **.717** | .743 | .568 | .727 | .760 | .627 | .779 | .565 | |
| SD | .0032 | .0008 | .0038 | .0028 | .0018 | .0019 | .0038 | .0047 | |
| **noise band (2 SD)** | **.006** | .002 | **.008** | **.006** | .004 | .004 | .008 | .009 | |
| *Stage 0 (`4classTF_GSLC`)* mean / band | .715 / .017 | .737 / .003 | .563 / .019 | .725 / .015 | .753 | .622 | .779 | .566 | |

- K-fold P / R, mean (SD) of 6 seeds: Non-forest .788 (.004) / .735 (.004), Low forest .565 (.004) / .705 (.004),
  Forest .869 (.001) / .705 (.006), Tall forest .429 (.007) / .825 (.006).
- OA per site, mean (SD; min–max): 9 Mile .630 (.006; .620–.638), Biota track .749 (.004; .745–.755), Bunda Extra
  .714 (.003; .709–.718), Country Downs .739 (.002; .735–.742), South Dampier .712 (.003; .708–.716).
- Importances (gain share, 6 seeds): S2 .490–.509, GSLC γ0 .262–.321, GSLC decomposition .189–.233; 168 A .322–.362,
  147 D .149–.179. EVI is first in every run; 168 A g0HH is second in 5 of 6 (NDVI in seed 2).

**NISAR only, 3 seeds** (`4classTF_GSLCg0_f9w5_SAR`, `_seed1`, `_seed2`):

| Run | OA | balanced | kappa | F1w | F1 NF | F1 LF | F1 F | F1 TF |
|---|---|---|---|---|---|---|---|---|
| seed 42 | .631 | .670 | .455 | .648 | .719 | .562 | .691 | .421 |
| seed 1 | .631 | .668 | .455 | .648 | .717 | .563 | .691 | .419 |
| seed 2 | .619 | .670 | .444 | .638 | .720 | .563 | .672 | .406 |
| **mean** | **.627** | .670 | .451 | .645 | .719 | .563 | .684 | .415 |
| **NISAR-only band (2 SD)** | **.014** | .002 | .012 | .012 | .003 | .001 | .021 | .017 |
| *1.0b `4classTF_GSLC_SAR`* mean / band | .609 / .006 | .655 | .430 | .628 | .706 | .556 | .663 | .390 |
| *`4class_NISARonly`* (GCOV, seed 42) | .610 | .653 | .431 | .629 | .705 | .556 | .666 | .391 |

- K-fold P / R, mean of 3 seeds: Non-forest .747 / .693, Low forest .493 / .655, Forest .828 / .584, Tall forest
  .288 / .747.
- OA per site, mean: 9 Mile .503, Biota track .651, Bunda Extra .621, Country Downs .648, South Dampier .652 (1.0b:
  .490 / .624 / .627 / .623 / .637).
- Importances: GSLC decomposition .591–.607, GSLC γ0 .393–.409; 147 D .481–.491, 168 A .509–.519. 147 D Pu is first
  in all 3.

**Gain of each source over the other** (seed means; S2 only = 1 run, .696; band = the Stage 1.5 band, .006 OA):

| Metric | S2 added to NISAR | Stage 1.5 bands (NISAR-only bands) | NISAR added to S2 | Stage 1.5 bands | *Stage 0: NISAR added to S2 (Stage 0 bands)* |
|---|---|---|---|---|---|
| OA | +.090 | 14 (6.5) | +.021 | 3.3 | +.019 (1.1) |
| balanced | +.073 | 47 (30) | +.018 | 11.6 | +.013 (4.1) |
| kappa | +.116 | 15 (9.4) | +.030 | 4.0 | +.026 (1.4) |
| F1w | +.083 | 15 (6.7) | +.021 | 3.7 | +.018 (1.2) |
| F1 Non-forest | +.042 | 12 | +.025 | 7.0 | +.018 (7.3) |
| F1 Low forest | +.064 | 17 | +.031 | 8.0 | +.025 (3.6) |
| F1 Forest | +.094 | 13 | +.016 | 2.1 | +.015 (0.7) |
| F1 Tall forest | +.150 | 16 | +.016 | 1.7 | +.017 (0.7) |

At the same seed (42 against 42), NISAR adds +.019 OA to S2 (3.0 bands).

**Map agreement** (`diagnosis\map_agreement.py`, 1 km blocks, 20 draws, null SD ≤ 0.002; 64,392,143 px valid in all
maps except the last row, 64,452,102). `newSAR` = `XGB_DP_NISAR_2026_4classTF_GSLCg0_f9w5_SAR.tif`, `new` =
`XGB_DP_NISAR_S2_2026_4classTF_GSLCg0_f9w5.tif`:

| Map A | Map B | agreement | null | lift | *Before (§0.3, §0.5)* |
|---|---|---|---|---|---|
| S2 only | newSAR | 0.512 | 0.332 | **0.179** | GCOV NISAR only 0.510 / 0.177; GSLC 0.516 / 0.186 |
| S2 only | new | 0.679 | 0.328 | 0.348 | vs GSLC NISAR + S2: 0.692 / 0.361 |
| newSAR | new | 0.757 | 0.393 | 0.366 | GSLC: 0.747 / 0.371 |
| newSAR | GSLC NISAR only (1.0b, `4classTF_GSLC_SAR`) | 0.871 | 0.387 | 0.485 | – |

| Class fraction | new (NISAR + S2) | S2 only | newSAR | 1.0b GSLC NISAR only | *Stage 0 GSLC NISAR + S2* |
|---|---|---|---|---|---|
| Non-forest | .572 | .371 | .530 | .499 | .555 |
| Low forest | .201 | .316 | .301 | .321 | .216 |
| Forest | .192 | .267 | .149 | .156 | .194 |
| Tall forest | .035 | .047 | .020 | .024 | .035 |

**Result of Stage 1.5.**
1. **The accuracy did not change.** NISAR + S2: mean OA .717 against .715 in Stage 0 (+0.002); kappa, F1w and the
   per-class F1 also within the bands. Per site, all 5 means are within 0.01 of Stage 0.
2. **The SD did change: the noise band is now 0.006 OA (Stage 0: 0.017).** The Stage 0 band came mostly from seed 3
   (without it, 0.006). In Stage 1.5 no seed is out of line. **From now on the noise band is 0.006 OA, 0.008 kappa,
   0.006 F1w** (plan). It is from 6 runs, so it is itself uncertain.
3. **The NISAR-only accuracy went up:** .627 against .609 (+0.018; 1.3 of the new NISAR-only band, 3 of the old). The
   new NISAR-only band is wider (0.014; seed 2 is 0.012 below the other two). The gain is in Forest (F1 +.021) and
   Tall forest (+.025) and at 4 of 5 sites (not Bunda Extra). The one old-environment run on this stack (§1.2, seed 42)
   was .639, 0.008 above the new-environment seed 42 (§1.4); one pair only, so the environment effect on NISAR only
   is not measured well.
4. **The gain of each source has the same size as in Stage 0.** NISAR adds +0.021 OA to S2 (Stage 0: +0.019). It is
   now 3.3 bands, not 1.1, **because the band is narrower, not because the gain is larger.** So the NISAR gain over
   S2 is now beyond the noise for OA, kappa, F1w and balanced accuracy; for Forest and Tall forest F1 it is 1.7–2.1
   bands, at the edge. S2 adds +0.090 to NISAR (Stage 0: +0.106; smaller because NISAR only is better).
5. **The one-source map disagreement did not change.** S2 only against the new NISAR only: 0.512, lift 0.179 (Stage 0:
   0.510 / 0.177). Each one-source map is still about the same distance from the combined map (lift 0.348 for S2,
   0.366 for NISAR). The reprocessing changed 13 % of the NISAR-only map (0.871 against the 1.0b map; net change in
   the fractions: Non-forest +.031, Low forest −.020, Forest −.006, Tall forest −.004), but it did not move it towards
   the S2-only map. NISAR still sets
   the Non-forest extent of the combined map (.572; NISAR only .530, S2 only .371).
6. **Open (raised with the user 2026-10-29):** where the two one-source maps disagree, and which one is nearer the
   truth away from the 5 sites. Not resolved by Stage 1 or 1.5.

## 2. Stage 2 — domain and label-site diversity (`PLAN_stage2_sites.md`, 2026-10-29)

Reference: the Stage 1.5 stack and run `4classTF_GSLCg0_f9w5` (NISAR + S2, 39 features). Noise band (Stage 1.5): OA
0.006, balanced accuracy 0.002, kappa 0.008, F1w 0.006. Chain `diagnosis\stage2_chain.ps1` (log `diagnosis\stage2.log`),
then `diagnosis\stage2_cov_chain.ps1` (log `diagnosis\stage2_cov.log`). New `classify_2026.py` options `--site`,
`--subsample`, `--extra-features` (docstring divergences 12–14). All tables are in
`G:\python\dampier_veg_timeseries\nisar_2026\outputs\diagnosis\`.

### 2.1 Site geography (`diagnosis\site_geography.py`, `site_geography.csv`, `site_distance_km.csv`, `yawuru_sites.csv`)

Covariates (`diagnosis\covariates_2026.py`; sources confirmed by the user 2026-10-29), resampled bilinear to the 20 m
grid, in `outputs\diagnosis\covariates_20m.tif` (float32, nodata NaN): mean annual rain 1991–2020 from ANUClimate 2.0
(0.01°, NCI THREDDS), clay and sand % 0–30 cm (depth-weighted) from SLGA v2 (TERN, 3"; needs a TERN API key in
`C:\Users\glens\.netrc`), elevation from Copernicus GLO-30 (a DSM). Raw subsets in `E:\nisar_2026\covariates_src\`.
Soil is finite on 98.4 % of the mainland (gaps at the coast), the others on 100 %.

| Site | Lat, lon | Label px (km²) | NF / LF / F / TF | Rain (mm) | Clay % | Sand % | Elev (m) | Last fires (NAFI) |
|---|---|---|---|---|---|---|---|---|
| 9 Mile | −17.09, 122.57 | 4,894 (1.96) | .13 / .40 / .47 / .00 | 958 | 11.6 | 81.8 | 111 | Aug 2024, Aug 2025, Jul 2026 (100 %) |
| Biota track | −17.46, 122.35 | 11,612 (4.65) | .02 / .08 / .82 / .08 | 929 | 10.3 | 83.7 | 160 | May fires; 18 % in May 2026 |
| Bunda Extra | −17.16, 122.34 | 3,987 (1.60) | .17 / .17 / .46 / .20 | 891 | 9.7 | 84.2 | 11 | 38 % in Aug 2025; none in 2026 |
| Country Downs | −17.27, 122.50 | 5,206 (2.08) | .03 / .20 / .71 / .07 | 982 | 10.4 | 83.4 | 124 | 13 % in 2026 |
| South Dampier | −17.78, 122.30 | 9,016 (3.61) | .40 / .40 / .20 / .00 | 788 | 7.1 | 88.1 | 19 | May and Jul 2026 (100 %) |

Within-site SD: rain 1–3 mm, clay 0.4–1.3 %, sand 0.6–1.6 %, elevation 3–11 m. So each covariate is almost constant
in a site. Fire columns from `lidar_2026\REPORT_structure_sar.md` ("Fire history (NAFI)").

- **The spread of the 2026 sites is about the same as that of the Yawuru sites.** Centroid distance, Dampier 2026
  (5 sites): median 35 km, max 82 km. Yawuru 2025 (13 sites, cut polygons, read only): median 29 km, max 69 km. The
  Dampier sites are 39–90 km from the mean Yawuru centroid.
- Rain changes by 194 mm between the sites (South Dampier 788 mm, Country Downs 982 mm); sand by 6 %.
- **The class mix changes very much between sites**: Forest is 20 % (South Dampier) to 82 % (Biota track); Tall
  forest is 0–20 %; 2 sites have no or almost no Tall forest.

### 2.2 Site separability (`diagnosis\site_signature.py`, `site_signature*.csv`)

XGB with the 4classTF seed-42 best parameters predicts the site of each labelled px (5 classes, balanced weights).
Stratified 5-fold (the plan) and, as in `REPORT_year_signature.md`, 5-fold StratifiedGroupKFold on 1 km blocks (41
blocks; no block in both train and test).

| Subset | Features | Folds | OA (majority prior) | Balanced (prior 0.20) | Macro AUC |
|---|---|---|---|---|---|
| All classes | all 39 | stratified | 0.983 (0.334) | 0.984 | 1.000 |
| All classes | all 39 | block 1 km | 0.906 | 0.907 | 0.988 |
| All classes | S2 | block 1 km | 0.902 | 0.905 | 0.987 |
| All classes | NISAR | block 1 km | 0.727 | 0.719 | 0.933 |
| **Forest only** | all 39 | stratified | 0.975 (0.497) | 0.979 | 0.999 |
| **Forest only** | **all 39** | **block 1 km** | **0.881** | **0.889** | **0.981** |
| Forest only | S2 | block 1 km | 0.876 | 0.880 | 0.981 |
| Forest only | NISAR | block 1 km | 0.688 | 0.691 | 0.914 |

- Per-site AUC (Forest only, all features, block folds): 9 Mile 1.000, South Dampier 0.996, Bunda Extra 0.990,
  Country Downs 0.961, Biota track 0.960.
- The best single band has a macro AUC of 0.80 (Forest only, `147_D_Pd_db`). So the site signature is
  **multivariate**, not one offset.
- **Result: the features identify the site, also inside Forest only and with spatial block folds** (OA 0.88 against
  a prior of 0.50). S2 alone identifies the site almost as well as all features. So at the same label class, each
  site has its own spectral and backscatter signature.

### 2.3 Within-site against pooled (`--site`, tags `4classTF_GSLCg0_f9w5_site<Name>`, `within_vs_pooled.csv`)

K-fold inside one site (n_iter 200, seed 42), against the pooled seed-42 K-fold at the same px
(`diagnosis\within_vs_pooled.py`). Classes with fewer than 50 px: **9 Mile Tall forest 5 px** (the minimum for 5
folds), **South Dampier Tall forest 0 px** (left out of that fit). All other classes have ≥ 150 px.

| Site | OA within / pooled (Δ, bands) | Balanced within / pooled (Δ) | Kappa within / pooled (Δ) | F1 Forest within / pooled |
|---|---|---|---|---|
| 9 Mile | .680 / .631 (**+.049, 8.2**) | .485 / .434 (+.052) | .470 / .384 (+.086) | .764 / .643 |
| Biota track | .589 / .745 (−.156, −26) | .579 / .473 (+.106) | .253 / .310 (−.057) | .707 / .844 |
| Bunda Extra | .748 / .713 (**+.035, 5.9**) | .754 / .739 (+.015) | .642 / .600 (+.042) | .770 / .718 |
| Country Downs | .697 / .739 (−.042, −7.0) | .632 / .576 (+.056) | .429 / .462 (−.034) | .785 / .822 |
| South Dampier | .738 / .711 (**+.027, 4.5**) | .738 / .640 (+.098) | .594 / .529 (+.065) | .685 / .422 |

- Per-class P / R within site: 9 Mile NF .54/.55, LF .64/.61, F .75/.77, TF 0/0; Biota track NF .12/.30, LF .25/.66,
  F .93/.57, TF .26/.78; Bunda Extra NF .87/.77, LF .54/.65, F .82/.72, TF .73/.87; Country Downs NF .22/.33,
  LF .46/.66, F .88/.71, TF .54/.83; South Dampier NF .84/.82, LF .69/.65, F .64/.74.
- Importances within site (gain): S2 share 0.31 (9 Mile), 0.36 (South Dampier), 0.38–0.56 at the other three. At
  9 Mile and South Dampier the first feature is 168 A γ0 HV or HH; at the other three it is EVI or BSI.
- **Caveat:** within a site, the balanced weights come from the class mix of that site. At Biota track (82 % Forest)
  and Country Downs (71 %) they move predictions away from Forest, and OA goes down (Forest recall .57 and .71). The
  pooled weights come from the pooled mix. So OA is not a fair comparison at the Forest-dominated sites. Balanced
  accuracy is fairer, but it also favours the per-site weights.
- **Result:** within-site OA is higher by > 2 bands at 3 of 5 sites (9 Mile, Bunda Extra, South Dampier: the two
  recently burnt sites and the white-sand site). Balanced accuracy is higher at all 5 (+0.015 to +0.106). The plan's
  test ("higher by > 2 SD at most sites") passes, with the weight caveat. One model for the 5 sites costs accuracy
  at the sites that are least like the others.

### 2.4 Learning curve (`--subsample`, `learning_curve.csv`, `learning_curve.png`)

Stratified subsets of the pooled labels, seeds 42 and 1 (the subset changes with the seed). Full set = the Stage 1.5
runs `4classTF_GSLCg0_f9w5` and `_seed1`.

| n | OA (seed 42 / 1) | OA mean | kappa mean | F1w mean |
|---|---|---|---|---|
| 2,000 | .687 / .688 | .687 | .514 | .696 |
| 5,000 | .687 / .686 | .687 | .525 | .698 |
| 10,000 | .686 / .703 | .694 | .535 | .706 |
| 20,000 | .705 / .721 | .713 | .560 | .723 |
| 34,715 (all) | .716 / .715 | .715 | .566 | .726 |

Figure: `G:\python\dampier_veg_timeseries\nisar_2026\outputs\diagnosis\learning_curve.png`. The importances are the
same in every run (EVI first; NDVI and 168 A γ0 HV or HH next; S2 share 0.41–0.50).

- From 2 k to the full set: OA +0.028 (4.7 bands), kappa +0.052.
- **From 20 k to the full set: OA +0.003 (0.4 band), kappa +0.006 (0.7 band): inside the noise.** The two seeds
  differ by up to 0.017 at 10 k and 20 k, because the subset changes too. So the curve is almost flat at the full set.
- **Result: label quantity is not the main limit.** Two times more labels of the same kind would add less than
  about 0.01 OA, not the 0.06–0.08 to the Yawuru numbers. Note: a subset keeps all 5 sites, so this tests the
  density of labels, not the number of sites. More sites are not tested.

### 2.5 Covariates (`--extra-features`, tag `4classTF_GSLCg0_f9w5_cov`, with a map)

NISAR + S2 + northing, easting, rain, clay, sand, elevation (45 features); the rows and folds are the same as the
reference (asserted).

| Run | OA | balanced | kappa | F1w | F1 NF | F1 LF | F1 F | F1 TF |
|---|---|---|---|---|---|---|---|---|
| reference (seed 42) | .716 | .742 | .566 | .726 | .762 | .628 | .776 | .560 |
| + covariates | **.744** | .762 | **.605** | .752 | .792 | .650 | .802 | .595 |
| Δ (bands) | +.028 (4.7) | +.020 (10) | +.039 (4.9) | +.026 (4.3) | | | | |

- P / R: NF .804/.779, LF .594/.718, F .878/.738, TF .469/.813. OA per site: 9 Mile .692 (+.061), Biota track .772
  (+.027), Bunda Extra .729 (+.016), Country Downs .754 (+.015), South Dampier .737 (+.026).
- Importances (gain): covariates 0.099 in total (elev .028, northing .022, easting .016, rain .015, sand .010, clay
  .009; elev is 6th, northing 8th), S2 .460, NISAR .441.
- **Map:** `G:\python\dampier_veg_timeseries\nisar_2026\outputs\XGB_DP_NISAR_S2_2026_4classTF_GSLCg0_f9w5_cov.tif`.
  Agreement with the reference map 0.686 (63,383,860 px valid in both), **null 0.502, lift 0.182**. Mainland
  fractions: Non-forest .831 (reference .568), Low forest .075 (.204), Forest .077 (.194), Tall forest .017 (.034).
- **Artefacts at covariate edges** (`outputs\diagnosis\cov_map_quicklook.png`): a straight north–south edge just east
  of Country Downs and an east–west edge at the 9 Mile northing. Forest stays only in the box that the label sites
  span; outside it, most px go down by 1–2 classes.
- **Result:** the covariates raise K-fold OA at every site, most at 9 Mile. But the map shows that the model uses
  them as site identifiers (the caveat in the plan): the trees cut on easting and northing at the edges of the
  label sites, and they extrapolate badly. **The +0.028 is site identification, not a better model. Do not use the
  covariates in a product.** The K-fold gain does confirm that the pooled model loses accuracy because the sites
  are different.

### 2.6 9 Mile (`diagnosis\nine_mile.py`, `nine_mile_*.csv`)

Pooled seed-42 K-fold confusion at 9 Mile (rows true, OA .631):

| | NF | LF | F | TF |
|---|---|---|---|---|
| Non-forest | 234 | 346 | 58 | 0 |
| Low forest | 101 | 1,633 | 221 | 0 |
| Forest | 17 | **1,052** | 1,222 | 5 |
| Tall forest | 0 | 0 | 5 | 0 |

- **46 % of the 9 Mile Forest is predicted as Low forest** (1,052 of 2,296). Forest F1 at the other sites is .42
  (South Dampier) to .84 (Biota track); 9 Mile .64.
- **Class mix:** the 9 Mile labels have a mean of 2.29 m (SD 0.95), just under the 2.35 m cut. 16 % of its labels are
  within 0.2 m of 2.35 m (other sites 6–13 %), and 25 % within 0.2 m of any cut (South Dampier 27 %, the others
  13–17 %). OA near a cut is .53–.56 at every site. Away from a cut, 9 Mile is still the lowest (.66; others .75–.78).
  So the cuts explain only a part.
- **LiDAR** (200 evenly spaced samples of 50,000 points per file, `*_ground_gndthin_height_denoise.las`): 9 Mile has
  296 pts/m² (the lowest; others 317–356) and a ground fraction of 0.48 (South Dampier 0.54, others 0.33–0.39). The
  density is high at every site, so it is not a likely limit. The high ground fraction agrees with an open, burnt
  understorey.
- **Fire:** 9 Mile burnt in August 2024, August 2025 and July 2026 (100 % of cells each time; late dry season), 2–3
  months before the S2 (20 Sep–10 Oct 2026) and NISAR (28 Sep–11 Oct 2026) data. `REPORT_structure_sar.md`: 9 Mile and
  South Dampier are 1–3 dB darker in HV at the same structure, and at Country Downs the cells burnt in 2026 are about
  1 dB darker than the other cells.
- **With 9 Mile only, the model reaches .680** (+.049; Forest F1 .764), with NISAR γ0 HV / HH as the first features.
  So the features hold the 9 Mile structure, but with a relation that is different from the other sites.
- **Result:** 9 Mile is low because a recent late-season fire makes its Forest look like the Low forest of the other
  sites (dark SAR, burnt understorey), and because many of its labels are near the 2.35 m cut. The LiDAR quality is
  not the cause.

### 2.7 Result of Stage 2

1. **Site difference: supported.** The 5 sites have strong, multivariate signatures at the same class (site OA 0.88
   against 0.50, Forest only, 1 km blocks). A within-site model is better at the 3 least typical sites (+4.5 to +8.2
   bands OA; balanced accuracy higher at all 5). Covariates that only identify the site add +0.028 OA (4.7 bands) in
   K-fold. The differences agree with fire history (9 Mile, South Dampier: burnt in 2026, dark SAR) and soil /
   elevation (Bunda Extra: white sand, 11 m), not with the distance between the sites.
2. **Label quantity: not supported as the main limit.** The learning curve is flat between 20 k and 34.7 k
   (+0.003 OA, 0.4 band).
3. **Geographic spread: not larger than Yawuru.** The 13 Yawuru sites have about the same distance spread. The
   difference is in what the sites are (fire regime, soil, class mix), not how far apart they are.
4. **For the deliverable:** site-to-site differences limit the 2026 K-fold OA, and one model with these features
   does not remove them. Away from the sites, the map is less certain than K-fold shows (this agrees with the
   one-source map disagreement, Stage 1.5). Not tested: more sites, a fire covariate (time since fire), and a
   site-free validation (LOSO, excluded by the plan). (The fire covariate is tested in §2.8.)

### 2.8 Stage 2b — NAFI fire history as a feature (2026-10-29)

**Question.** The site differences agree with fire history (9 Mile and South Dampier burnt in 2026, dark in HV). Does
a fire feature raise K-fold accuracy and keep a sane map, unlike the §2.5 covariates? The user said: fire frequency
(FF) and time since burnt (TSB) are highly correlated, so they are tested **in separate runs, never together**; and
the 2026 fires must be included.

**Data** (`diagnosis\fire_covariates_2026.py`, log `diagnosis\fire_covariates_2026.log`). NAFI fire scars by month,
250 m, 2000–2026, from the NAFI GeoServer WCS (`https://firenorth.org.au/geoserver/ows`, coverages `public__fsm2000`
… `public__fsm2025`, `public__fsm_current`), raw subsets in `E:\nisar_2026\nafi_src\`. Two service problems, both
corrected: GeoServer resamples a subset to the requested bbox (requests are now snapped to the cell edges, and the
0.0025° pixel is asserted), and `fsm_current` is georeferenced half a cell north-east (shift found by test: agreement
with the 2026 scar shapefile 96.30 % → 99.85 %). 2026 fires count only to September (October scars, 1.85 % of the
mainland, can be after the images). Nearest resampling to the 20 m grid, mainland only.

- `G:\python\dampier_veg_timeseries\nisar_2026\outputs\diagnosis\fire_ff_20m.tif`: `nafi_ff_2017_26`, years burnt
  2017–2026 (0–10).
- `G:\python\dampier_veg_timeseries\nisar_2026\outputs\diagnosis\fire_tsb_20m.tif`: `nafi_tsb_months`, months from
  the last fire to Sep 2026 (0 = Sep 2026; 321 = not burnt 2000–2026, 5.7 % of the mainland).
- Checks: FF and the last-fire year 2016–25, rebuilt from the WCS layers, are **100.0000 % identical** to the local
  `G:\python\fire_class_forest_stats\NAFI\FF_2016_25_gda94.tif` and `TSB_2016_25_gda94.tif`; burnt area against
  the scar shapefiles 99.86 % (2025), 99.85 % (2026); both bands finite at all 34,715 label px. Spearman FF vs TSB
  −0.70 (mainland).
- Per site (`outputs\diagnosis\fire_sites_2026.csv`): FF median 9 Mile 6, South Dampier 5, Biota 3, Country Downs 3,
  Bunda Extra 2. TSB: **9 Mile 2 months at every label px**, South Dampier 2–5, Biota 4–60 (median 16), Country
  Downs 4–97 (35), Bunda Extra 13–321 (47). Burnt Jan–Sep 2026: 9 Mile and South Dampier 100 %, Biota 18 %, Country
  Downs 16 %, Bunda Extra 0 % (agrees with `lidar_2026\REPORT_structure_sar.md`).
- Found on the way: the local `TSB_2016_25_gda94.tif` codes "not burnt 2016–25" as **254**, not 255.
  `lidar_2026\fire_vs_sar.py` line 115 maps only 255 to NaN, so it keeps 254 as a TSB value. Its site medians (0–2)
  are not affected; not fixed (other strand).

**Runs** (`diagnosis\stage2_cov_chain.ps1`, the same rows and folds as the reference, asserted; logs
`diagnosis\stage2_fireFF.log`, `diagnosis\stage2_fireTSB.log`). Δ against the reference seed 42, in noise bands
(OA 0.006, kappa 0.008, F1w 0.006, balanced 0.002):

| Run | OA | balanced | kappa | F1w | F1 NF | F1 LF | F1 F | F1 TF |
|---|---|---|---|---|---|---|---|---|
| reference (seed 42) | .716 | .742 | .566 | .726 | .762 | .628 | .776 | .560 |
| + FF (`4classTF_GSLCg0_f9w5_fireFF`) | .721 | .742 | .573 | .731 | .760 | .628 | .784 | .571 |
| Δ (bands) | +.006 (1.0) | −.000 (−0.1) | +.007 (0.8) | +.005 (0.8) | | | | |
| + TSB (`4classTF_GSLCg0_f9w5_fireTSB`) | .714 | .745 | .565 | .725 | .762 | .627 | .774 | .558 |
| Δ (bands) | −.002 (−0.3) | +.003 (1.4) | −.001 (−0.2) | −.002 (−0.3) | | | | |

- P / R, FF: NF .784/.739, LF .568/.703, F .869/.715, TF .441/.811. TSB: NF .787/.739, LF .564/.705, F .870/.697,
  TF .418/.838.
- OA per site (reference / FF / TSB): 9 Mile .631 / .637 / .628; Biota track .745 / .755 / .740; Bunda Extra .713 /
  .718 / .710; Country Downs .739 / .738 / .742; South Dampier .711 / .716 / .712. (Covariates, §2.5: 9 Mile .692.)
- Importances (gain): FF rank 10 of 40, 0.017; TSB rank 13 of 40, 0.014. The top 4 do not change (EVI, 168 A γ0 HV
  and HH, NDVI). (`kfold_report_2026.py` puts the fire band in the "S2" group; that is a label only.)
- **Maps**: `G:\python\dampier_veg_timeseries\nisar_2026\outputs\XGB_DP_NISAR_S2_2026_4classTF_GSLCg0_f9w5_fireFF.tif`
  and `..._fireTSB.tif`. Agreement with the reference map: FF **0.923** (null 0.404, lift 0.518), TSB **0.930** (null
  0.413, lift 0.516). Mainland fractions (reference / FF / TSB): Non-forest .572 / .568 / .593, Low forest .201 /
  .203 / .185, Forest .192 / .194 / .186, Tall forest .035 / .035 / .035.
- **Quick looks** with the Jan–Sep 2026 scars outlined (`outputs\diagnosis\fireFF_map_quicklook.png`,
  `fireTSB_map_quicklook.png`; `diagnosis\cov_map_quicklook.py` now has `--map`, `--label`, `--out`, `--scar`): no
  straight edges, no 250 m blocks, and the changes do not follow the scar outlines. FF: scattered px. TSB: diffuse
  −1 steps east of Biota track and Country Downs.
- **Class step by fire value** (every 4th px, mainland): with TSB, the mean step is −0.069 in land burnt in **2025**
  (41 % of the mainland, 6.7 % of px lower), but +0.009 in land burnt in 2026 (24 %) and ≈ 0 in land not burnt since
  2017. A real fire effect would act most on the most recent fires; this one acts on the TSB values of Bunda Extra and
  parts of Biota / Country Downs (9–20 months). With FF, the step falls with FF: +0.04 at 1–3 fires, −0.05 at 6,
  −0.13 at 7–10 (13 % of px lower). This is the expected direction, but 6 is the 9 Mile value (next point).

**Site-identifier check** (`diagnosis\fire_site_check.py`, `outputs\diagnosis\fire_site_check.csv`; method as §2.2,
XGB from the one fire band):

| Band | Subset | Single-band macro AUC | Site OA, stratified (prior) | Site OA, 1 km blocks | Balanced, blocks (prior 0.20) |
|---|---|---|---|---|---|
| FF | all classes | 0.717 | 0.535 (0.334) | 0.453 | 0.535 |
| FF | Forest only | 0.673 | 0.480 (0.497) | 0.472 | 0.492 |
| TSB | all classes | 0.815 | 0.660 (0.334) | 0.534 | 0.557 |
| TSB | Forest only | 0.787 | 0.826 (0.497) | 0.711 | 0.543 |

- Per-site AUC, 1 km blocks, Forest only: FF identifies **9 Mile at 0.968** (others 0.60–0.70); TSB identifies every
  site at 0.86–0.95. The best single stack band reaches 0.80 (§2.2), so one fire band is as strong a site key as the
  best SAR band. But the stack alone already identifies the site at 0.88 (§2.2), so the fire band adds no new key, and
  the model gains (almost) nothing from it.

**Result.**
1. **Neither fire band raises accuracy beyond the noise.** FF: +0.006 OA (1.0 band), kappa 0.8 band, balanced
   unchanged. TSB: no change. 9 Mile stays at .628–.637 (the covariates gave .692).
2. The maps change little (agreement 0.92–0.93) and show no artefacts. The changes do not follow the scar edges. TSB
   lowers classes in the 2025 scars, not the 2026 scars, so it is not a consistent fire effect.
3. FF and TSB are site keys at these 5 sites (each site has nearly one fire history at 250 m; 9 Mile is one TSB value).
   With 5 sites, a fire effect on the 2026 structure cannot be separated from the site, the same limit as in
   `lidar_2026\REPORT_structure_sar.md`.
4. **For the deliverable: do not add NAFI fire history (250 m) to the feature set.** It adds no accuracy, and what it
   could add cannot be told from site identity. The dark-SAR effect of recent fire at 9 Mile and South Dampier is
   not modelled by a 250 m fire raster. That would need more sites with different fire histories, or a fire-scar map
   at the image resolution.
