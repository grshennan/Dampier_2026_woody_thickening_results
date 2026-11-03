# REPORT — the acquisition-year signature in the feature stack

**Established 2026-08-11** by `B5_year_signature.py` (phases `labels` and
`pixels`; logs `B5_year_signature_labels.log`, `B5_year_signature_pixels.log`;
outputs `outputs/aggregate/year_predictability.json`,
`year_signature_ablation.csv`, `year_signature_single_band.csv`,
`year_signature_fixed_pixels.csv`, `year_signature_band_shifts.csv`).

Migrated out of the deleted `CHECKPOINT.md` on 2026-08-24 because it is the most
consequential finding this directory produced and it constrains **every**
cross-year statement in the project, not just this one.

## The finding

**Acquisition year is recoverable from the 20 predictor bands at 80.7 % accuracy
against a 20 % prior (4.0×), macro one-vs-rest AUC 0.956** — per year 0.997
(2019), 0.947, 0.943, 0.973, 0.920 (2025). Scored with 5-fold `GroupKFold` on
`block_id`; blocking verified honest (`block_id` is 1:1 with `(block_x, block_y)`
so it is spatially consistent across years, zero train/test block overlap in
every fold, years balanced across folds).

**S1 alone beats all 20 bands together**, and every sensor group identifies 2019
on its own:

| features | bands | accuracy | macro AUC | 2019 AUC |
|---|---|---|---|---|
| all | 20 | 0.807 | 0.956 | 0.997 |
| PALSAR | 6 | 0.796 | 0.961 | 0.998 |
| **S1** | **6** | **0.930** | **0.990** | 0.996 |
| S2 | 8 | 0.594 | 0.852 | 0.955 |

No single band exceeds AUC 0.843, so the signature is **multivariate, not one
offset**.

## It is real over fixed ground, and it is whole-scene

Phase `pixels` sampled 200,000 pindan pixels at fixed map coordinates;
199,984 were valid and finite in all five years. Paired median shift against each
pixel's own 2019 value, in within-band SD:

| band | 2021 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| Red-edge 1 | −1.14 | **−1.56** | −0.65 | −1.18 |
| Red | −1.04 | −1.43 | −0.54 | −1.10 |
| NIR 2 | −0.56 | −1.06 | −0.53 | −0.81 |
| s1_Pu | +0.72 | +1.00 | +0.20 | +0.66 |
| VH | +0.59 | +0.80 | +0.17 | +0.52 |
| Surface | +0.94 | +0.44 | +0.07 | +0.64 |

In 2023 only **2.0 %** of the 199,984 pixels sit above their own 2019 Red-edge 1
value (Green 5.7 %, Red 3.4 %) — 98 % of the scene moved together. **2019 is the
extreme year in both directions at once**: brightest in all 8 optical bands,
lowest in every S1 backscatter band.

## It reads as acquisition, not canopy

Red-edge 1 runs −1.14, −1.56, −0.65, −1.18: 2023 furthest from 2019, 2024 most of
the way back, 2025 away again. **Growth cannot oscillate like that**, and a
scene-wide 1.5 SD optical drop across 98 % of pixels is not vegetation change.

**Lead on the cause (plausible, not established).** S1 comes from the Copernicus
`sentinel-1-global-mosaics` collection — a pre-made *monthly* mosaic — via
`../download_s1_july_linux.py` (July 1–31 each year; 9 items for 2021). That
product's composition depends on constellation availability, and **Sentinel-1B
failed in December 2021**, so the 2019/2021 July mosaics and the 2023/2024 ones
are not built from the same constellation. Different revisit and incidence-angle
mixes would produce exactly this multivariate, year-idiosyncratic S1 fingerprint
with no canopy involvement.

## Gap left open

**The canopy-correlation test was never run.** The script computes marginal
shifts only; it never tested whether the per-pixel difference scales with canopy
height, which was the third discriminating criterion in the phase design. The
canopy explanation is argued against by non-monotonicity but not directly
excluded. Short test, data already in hand.

## Consequences

- **Stage A's +0.579 m 2019→2025 median rise cannot be read as canopy growth**,
  and neither can the `SPLICE4_one25` series. Both rest on stacks whose bands
  shift up to 1.5 SD non-monotonically, with 2019 — the baseline for every change
  number — the most extreme year of the five.
- Per the pre-registered rule in `PLAN_aggregate_height_increase.md` §4.2,
  **a pooled-across-years model (`MPOOL`) is not admissible as trend evidence**:
  it can key on the year signature and reproduce each year's own label mean.
  MPOOL was dropped by the user 2026-08-11.
- **Any cross-year work in this project inherits this**, including
  `class_vs_regression`'s cross-year axis. Within-year comparisons are unaffected
  — which is part of why the live work is now 2025-only.

## Stage A context these numbers attach to

Whole-of-pindan aggregate, 14,045,626 px valid in all 5 years, 8,155 km²
(`pindan_aggregate.py`, `outputs/aggregate/pindan_descriptive_statistics.md`):

- **Population effects are nil** — every year's valid footprint over pindan is
  ≥99.99 %; the ≥0.2 m bare floor removes **zero** pixels (minimum across all
  years 0.13 m).
- **Resampling is not driving anything** — 2025 warped nearest vs bilinear:
  mean 6.2079 vs 6.2078 m, sd 1.937 vs 1.921.
- **The rise is not a growth curve.** 2019→2021 +0.405 m median, then
  2021→2023 −0.006, 2023→2024 −0.030, 2024→2025 +0.239. Almost the whole 6-year
  change lands in the first 2-year step.
- **The distribution's shape is unstable between years** — sd 1.34 / 1.49 / 1.27
  / 1.44 / 1.94; skew flips sign twice (+0.20, +0.87, −0.63, +0.64, −0.51);
  p99 10.3 / 11.7 / 8.8 / 11.2 / 11.2 m. A common growth process cannot do that.
- **2025 has a distinct low mode at ~1–3 m** holding ~4.5 % of pixels that no
  other year shows (`frac_ge_2m` 99.4 % in 2024 → 95.4 % in 2025). Unexplained.
- All values fall inside 0–16 m (max 15.47 m, 2025).

## Operational facts worth keeping

- **`predict_single.py` must be run one year per process** (`--years <one year>`).
  Looping four years in one process reached 47.6 GB RSS by the third year and
  triggered `systemd-oomd`, which killed `dbus.service` and took down the whole
  GNOME session (2026-08-11). One year per invocation peaks at 25.3 GB, measured
  twice. **The script still has the multi-year loop; it has not been fixed.**
- **`rf_hybrid_2025_T4_full.pkl` *is* the M25 model** — it was fitted on exactly
  `train_hybrid_2025_T4.parquet` — so `figsDP2025/RF_HYBRID_T4_M25_2025.tif` and
  `RF_HYBRID_T4_2025.tif` are the same model on the same stack.
- M25 medians across the five stacks: 2019 4.37 m (89.1 % valid), 2021 4.42 m
  (97.3 %), 2023 4.88 m (95.1 %), 2024 5.33 m (96.1 %).
