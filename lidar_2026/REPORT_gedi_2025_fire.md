# REPORT — fire effects on 2025 GEDI RH98 against the 2026 drone LiDAR

**Date:** 2026-10-19. **Status:** complete for 2025.
Code: `gedi_fire.py` (its docstring has the rules; all but one were fixed before any result
was seen, and the one added later is marked). Log: `gedi_fire.log`. Outputs:
`outputs/gedi/fire_*`. It follows `REPORT_gedi_2025_wiggle.md`, with the same 154 shots
(headline stratum RH98 3–20 m), the same footprint statistics and the same offsets.
Numbers below are v003, footprint p90, nominal position, unless stated. v002 and the global
offset give the same conclusions (`fire_summary.csv`, `fire_null.csv`, `fire_diff.csv`).

## 1. Result in brief

- **The fire-free group does not give a higher correlation.** Clean shots (no NAFI scar on
  the footprint in 2024 or 2025-01 to 2026-10): n 39, r 0.66, RMSE 2.42 m, against all
  shots r 0.70, RMSE 1.84 m. Against random subsets with the same per-site counts, the clean
  r is ordinary (null median 0.63, 95 % 0.54–0.70; 20 % of subsets are as high or higher).
  The larger clean RMSE comes from three large errors (Bunda Bunda +11.1 m, Country Downs
  +6.0 m, Bunda Extra −4.5 m) that have no fire in either window.
- **Fire screening is not useful for the 2025 comparison.** Removing burnt shots removes
  three quarters of the data (115 of 154) and does not make the agreement better.
- **The fire groups are almost the same as the sites.** All 51 9 Mile shots burnt in both
  windows (2024, and again in 2025 Aug–Dec and in 2026). The clean group is Bunda Extra (21)
  and Country Downs (14), plus Bunda Bunda (2) and Biota track (2). Pooled group differences
  are therefore site differences. Only within-site comparisons can show a fire effect.
- **Within sites, there is no consistent fire effect.**
  - Biota track: footprints burnt in 2024 (`pre_only`, n 24) have p90 0.88 m higher
    relative to RH98 than footprints burnt between GEDI and the LiDAR (`inter_only`, n 16)
    (bootstrap 95 % 0.19–1.62 m). This is the expected direction (regrowth after a 2024 fire
    adds height before the LiDAR). But the two groups are in different parts of the site,
    so a structural difference can cause the same result.
  - Bunda Extra, clean against `inter_only` (21 / 9): −0.86 m (−2.16 to +0.41). Country
    Downs (14 / 8): +0.35 m (−0.65 to +1.51). Neither bias difference is different from
    zero, and they have opposite signs. At Bunda Extra the burnt shots (2025 Aug–Dec and
    2026 fires) have a lower r (0.20 against 0.83; Δr +0.63, 95 % +0.04 to +1.36), but
    from 9 shots.
- **An early dry season fire did not lower the canopy top.** Of the 15 Biota track shots
  with a May 2025 scar (14 `inter_only`, 1 `both`), 13 burnt **1 to 5 days after** the GEDI
  pass of 2025-05-17 (NAFI mapping periods 2025-05-18 to 05-22). So GEDI saw the canopy before the fire and the LiDAR saw it
  17 months after. These shots agree best of all: bias −0.07 m, RMSE 0.61 m, r 0.76 (Biota
  `inter_only`); all 16 shots with a 2025 Jan–Jul scar: bias −0.15 m, RMSE 0.68 m, r 0.89.
  A May fire in this country is low intensity; the trees that set p90 survived it.

## 2. Data

| Item | Value |
|---|---|
| Fire | NAFI monthly burnt area, 250 m (MODIS; Sentinel-2 for WA North), polygons in EPSG:4283, `G:\python\fire_class_forest_stats\NAFI\fire scars\`: 2024, 2025, 2026 to 18 Oct (provisional; WA North mapped to 2026-10-15) |
| Read with | `woody_thickening_splice_1/fire_data.load_scars` (bbox read, reprojected to EPSG:7851) |
| Scars in the shot bbox + 1 km | 2024: 390 (May 289, Jun 19, Aug 45, Sep 30, Nov 7); 2025: 553 (May–Jul 482, Aug–Dec 71); 2026: 884 (Apr–Jul 875) |
| Shots | as `REPORT_gedi_2025_wiggle.md` §2: 154 in the headline stratum, all in `mainland`; `pindan` is 9 Mile only |

`fire_data.py` changed in two lines (2026-10-19): `NAFI_DIR` is now platform-aware
(`NAFI_DIR_WIN`), and `scar_path` globs `{year}*firescar*`, so it also finds the folder
`2026 to Oct18 firescar shape files`. All twelve years 2015–2026 resolve to one shapefile.
Its own `__main__` self-test still uses Linux-only paths (points, pindan, CHMs), so it was
not run on Windows.

## 3. Method

One reference date for all shots, T0 = 2025-01-01 (user choice; not shot by shot):

| Window | Scars | Meaning |
|---|---|---|
| `pre` | calendar 2024 | burnt in the year before GEDI: regrowth at GEDI time |
| `inter` | 2025-01 to 2026-10 | burnt between GEDI and the LiDAR |
| sub-periods of `inter` (reported only) | 2025 Jan–Jul / 2025 Aug–Dec / 2026 Jan–Oct | Jan–Jul overlaps the shot dates; Oct 2026 overlaps the flights |

A footprint is burnt in a window if its 25 m disc intersects a scar. Groups: `clean`,
`pre_only`, `inter_only`, `both`; and `clean_strict`, no scar within 250 m (one NAFI pixel).
The offsets are not searched again per group (that would inflate r). Null for the clean r:
10,000 subsets of the 154 shots with the clean group's per-site counts. Group differences:
bootstrap (2,000), each group resampled by itself. Within-site pairs (≥ 5 shots each) were
added after the first run showed the site confound; they are marked as such in the
docstring.

Checks: the `all` group reproduces `wiggle_summary.csv` exactly (32 rows; v003 nominal p90
n 154, bias +0.57, RMSE 1.84, r 0.704). Self-test (`--self-test`): planted scars give the
planted groups, `clean_strict`, burnt fractions, scar distance and fire-before-shot flag;
the null median equals the all-shot r (0.876 = 0.876); per-site counts are kept.

## 4. Numbers (v003, nominal, footprint p90, RH98 3–20 m)

| Group | n | bias (m) | RMSE (m) | r | R² | sites |
|---|---|---|---|---|---|---|
| all | 154 | +0.57 | 1.84 | 0.70 | 0.39 | all |
| clean | 39 | +0.30 | 2.42 | 0.66 | 0.28 | Bunda Extra 21, Country Downs 14, Bunda Bunda 2, Biota 2 |
| clean_strict | 4 | +2.22 | 5.63 | 0.07 | −0.75 | too few to interpret |
| pre_only | 24 | +0.82 | 1.86 | 0.77 | 0.50 | Biota 24 |
| inter_only | 35 | +0.20 | 1.06 | 0.87 | 0.76 | Biota 16, Bunda Extra 9, Country Downs 8, South Dampier 2 |
| both | 56 | +0.87 | 1.75 | 0.45 | −0.33 | 9 Mile 51, Biota 5 |

Within-site differences (a − b), p90 bias, bootstrap 95 %:

| Site | a / b | n | Δbias (m) | Δr |
|---|---|---|---|---|
| Biota track | pre_only / inter_only | 24 / 16 | +0.88 [+0.19, +1.62] | +0.02 [−0.26, +0.51] |
| Biota track | pre_only / both | 24 / 5 | +0.89 [−0.18, +2.05] | −0.12 |
| Bunda Extra | clean / inter_only | 21 / 9 | −0.86 [−2.16, +0.41] | +0.63 [+0.04, +1.36] |
| Country Downs | clean / inter_only | 14 / 8 | +0.35 [−0.65, +1.51] | −0.14 [−0.49, +0.27] |

The pooled `inter_only` r of 0.87 and the `both` r of 0.45 are mostly site: `both` is 9 Mile,
which already had the lowest r (0.39) in the first report because its height range is narrow.

## 5. Caveats

- **NAFI is 250 m.** A 25 m footprint inside a 250 m burnt pixel need not have burnt, and
  NAFI misses small or patchy fires. The flags are "in burnt country", not "burnt
  footprint". Only 4 shots are more than 250 m from every scar, so a stricter screen leaves
  nothing to test.
- **No severity.** NAFI gives the month only. Most fires here are early dry season (May–Jul),
  which are usually cool; the 2025 Aug–Dec fires at 9 Mile and Bunda Extra are the most
  likely to be hot, and 9 Mile has no unburnt comparison.
- The 2026 scars are provisional (Sentinel-2 periods span months; `month` is NAFI's
  allocated month).
- One reference date: a fire in 2025 Jan–Jul counts as `inter` even when it came before a
  shot. Here that does not matter: of the 16 such shots, 0 burnt in an earlier month and 13
  of the 15 same-month cases burnt after the shot by the mapping period.
- With n 5–24 per within-site group, a fire effect smaller than about 1 m in bias cannot be
  found.

## 6. Next

- Do not screen the 2025 shots for fire; the nominal 154-shot result in
  `REPORT_gedi_2025_wiggle.md` stays the reference.
- **Back in time: done 2026-10-19**, `REPORT_gedi_earlier_years.md`. It also adds a
  within-site slope of the error on the number of burnt periods; for 2025 that is −0.79 m
  per period at Biota track [−1.41, −0.16] and −0.29 [−0.87, +0.30] pooled
  (`fire_trend.csv`).
- Possible later checks (not done, user decision 2026-10-19): night against day shots;
  fire frequency / time since burnt (`FF_2016_25_gda94.tif`, `TSB_2016_25_gda94.tif`);
  footprint heterogeneity; Sentinel-2 NDVI change for non-fire disturbance.
