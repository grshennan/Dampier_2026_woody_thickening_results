# REPORT — 2025 GEDI RH98 against the 2026 drone LiDAR, with a geolocation offset search

**Date:** 2026-10-19. **Status:** complete for 2025. Earlier GEDI years are the next stage.
Code: `gedi_site_shots.py` (shot extraction), `gedi_wiggle.py` (footprint statistics and
offset search; its docstring has the rules, which were fixed before any result was seen).
Logs: `gedi_site_shots.log`, `gedi_wiggle.log`. Outputs: `outputs/gedi/`.

## 1. Result in brief

- **GEDI geolocation error is not the main error here.** The best single offset for all
  shots is small (v002: 5.0 m west, 2.5 m north; v003: 2.5 m west, 2.5 m north). It
  increases r(RH98, footprint p98) by only +0.075 (v002) and +0.035 (v003), and it decreases
  the p90 RMSE by 0.18 m (v002) and 0.15 m (v003). The gain is the same in leave-one-out, so
  it is not overfit. But its position is poorly constrained: the bootstrap 95 % interval
  spans about 20 m in x and 12–18 m in y. No bootstrap resample chooses (0, 0).
- **Per-track offsets do not survive cross-validation.** In-sample they give +0.15 (v002)
  and +0.11 (v003) in r. Leave-one-out gives +0.005 and −0.014. The per-track "best
  offsets" (circles in `wiggle_surface.png`) are spread up to 35 m, which is noise, not
  geolocation. **Use nominal positions, or the one global offset. Do not use per-track or
  per-shot offsets.** The per-shot level (+0.34 r) is an in-sample upper bound only.
- **v003 positions are better than v002.** At the nominal offset, v003 gives r 0.666 against
  0.621 (p98), the p90 RMSE is 1.84 m against 1.94 m, and the search finds half the gain.
  RH98 is identical in the two versions (max difference 0.000 m). Only the positions changed.
- **RH98 matches footprint p90** (smallest bias +0.58 m, smallest RMSE 1.84–1.94 m, best R²).
  Footprint p98 is +2.2 m above RH98 and footprint max is +3.2 m above. This agrees with the
  Stage 6 finding that RH98 corresponds to p90 (CLAUDE.md). Here it is measured directly on
  the 50 cm CHM, not on a predicted raster.
- **The direct comparison is much better than the raster validation.** Footprint p90
  against RH98 gives R² 0.32 (v002) / 0.39 (v003) at nominal and 0.44 / 0.49 at the global
  offset. Stage 6 found p90 R² ≈ 0 for the RF prediction rasters. GEDI RH98 carries real
  shot-to-shot height information at these sites; the low Stage 6 R² comes from the rasters.

## 2. Data

| Item | Value |
|---|---|
| LiDAR | six 2026 sites (MVT sites excluded, see `G:\Dampier\Dampier Lidar 2026\README_analysis_scope.md`); 50 cm p98 CHM, EPSG:7851; flown 2026-10-05 to 2026-10-09 UTC (from point `gps_time`) |
| GEDI | GEDI02_A, 2025 calendar year, 10 granules over the sites (v002 local, v003 by HTTPS byte ranges at ≤ 4 MB/s) |
| QC | `quality_flag == 1` (v003: `l2a_quality_flag_rel2`), `degrade_flag <= 0`, `sensitivity >= 0.95` |
| QC shots in the site AOI | v002 257, v003 258. Check: v002 gives 231 in the unbuffered site boxes, the same as `chm_regression/outputs/gedi/shots_2025_calendar.parquet` |
| Used (disc ≥ 95 % valid CHM at all 625 offsets) | 160; 154 in the headline stratum RH98 3–20 m. 152 of the 154 are full-power beams |
| Shot dates (headline) | 2025-02-13 (2), 05-17 (69), 06-16 (32), 07-09 (51). **Gap to the LiDAR: 15–20 months** |
| Shots per site (headline) | 9 Mile 51, Biota track 47, Bunda Extra 30, Country Downs 22, Bunda Bunda 2, South Dampier 2 |

## 3. Method

Footprint statistic: all 50 cm CHM pixels with centres within 12.5 m of the footprint centre
(~1,963 px); p90, p95, p98, max and mean. Offsets: dx, dy from −30 to +30 m in 2.5 m steps
(625). Search objective: Pearson r between RH98 and footprint p98. r ignores a constant bias,
so the offset cannot trade against the bias. Levels: nominal; one global offset; one offset
per track (granule × beam, ≥ 5 shots; 7 tracks qualify); one offset per shot. The global and
track levels are also run leave-one-out. The global offset has a 1,000-resample bootstrap.

A within-site permutation null was tried first and rejected. It removes the nominal
correlation, so its gain starts from r ≈ 0 and cannot be compared with the real gain (it
gave a null gain of 0.19 against a planted gain of 0.22 in the self-test). Leave-one-out and
the bootstrap replace it.

Self-test (`--self-test`): a planted (+10, −5) m offset is recovered exactly by every level,
and by leave-one-out for every shot. A planted (0, 0) with 1 m noise gives a leave-one-out
gain of 0.000 (global). The strided gather equals a direct distance mask (0 difference).

## 4. Numbers (headline stratum RH98 3–20 m, all sites, n = 154)

Footprint p98: r, and the offset search.

| Version | r nominal | global offset (m) | r global | gain | gain LOO | bootstrap 95 % dx / dy | track gain (LOO) | shot gain (bound) |
|---|---|---|---|---|---|---|---|---|
| v002 | 0.621 | (−5.0, +2.5) | 0.696 | +0.075 | +0.075 | [−25.0, −2.5] / [0.0, +17.5] | +0.152 (+0.005) | +0.340 |
| v003 | 0.666 | (−2.5, +2.5) | 0.701 | +0.035 | +0.035 | [−20.0, +2.5] / [0.0, +12.5] | +0.109 (−0.014) | +0.296 |

Bias (statistic − RH98) / RMSE (m) / R², nominal → global offset.

| Statistic | v002 nominal | v002 global | v003 nominal | v003 global |
|---|---|---|---|---|
| p90 | +0.58 / 1.94 / 0.32 | +0.61 / 1.76 / 0.44 | +0.57 / 1.84 / 0.39 | +0.59 / 1.69 / 0.49 |
| p95 | +1.45 / 2.36 / −0.01 | +1.46 / 2.28 / 0.06 | +1.43 / 2.27 / 0.07 | +1.49 / 2.24 / 0.10 |
| p98 | +2.17 / 2.91 / −0.53 | +2.22 / 2.83 / −0.44 | +2.16 / 2.82 / −0.43 | +2.20 / 2.80 / −0.41 |
| max | +3.24 / 3.77 / −1.57 | +3.35 / 3.81 / −1.62 | +3.24 / 3.75 / −1.54 | +3.27 / 3.74 / −1.52 |
| mean | −4.17 / 4.62 / −2.85 | −4.18 / 4.60 / −2.81 | −4.20 / 4.64 / −2.88 | −4.18 / 4.59 / −2.80 |

Per site, footprint p90, nominal, v002 (v003): bias / RMSE / r.

| Site | n | v002 | v003 |
|---|---|---|---|
| 9 Mile (the only pindan site) | 51 | +1.04 / 1.87 / 0.34 | +0.96 / 1.81 / 0.39 |
| Biota track | 47 | +0.35 / 1.40 / 0.81 | +0.37 / 1.41 / 0.81 |
| Bunda Extra | 30 | −0.17 / 1.97 / 0.71 | −0.10 / 1.65 / 0.76 |
| Country Downs | 22 | +0.61 / 1.73 / 0.82 | +0.57 / 1.58 / 0.85 |

Boundaries: all 154 shots are in `mainland`. `pindan` is 9 Mile only, so the pindan row is
the 9 Mile row. 9 Mile has the lowest r: its height range in the footprints is narrow.
Bunda Bunda and South Dampier have too few shots (2 each) to report.

v002 against v003, paired on `shot_number` (713 of 722 AOI shots pair): for the 257 shots
that pass QC in both, the position shift is median 1.82 m and max 4.93 m (mean dx −1.15 m,
dy +0.00 m). Degraded shots (`degrade_flag` 30/50, which fail QC) move up to 77 m.

## 5. Caveats

- **Time gap of 15–20 months** between the GEDI shots and the LiDAR. Growth, fire or clearing
  in between all add to the error. The +0.6 m p90 bias may contain growth. **Fire is now
  done** (2026-10-19, `REPORT_gedi_2025_fire.md`): fire-free shots do not agree better, the
  fire groups are nearly the sites, and no consistent within-site fire effect was found.
- **The offset is relative.** An error in the drone georeferencing (accuracy not known) has
  the same effect as a GEDI offset. The global offset is GEDI relative to the drone.
- 97 of 257 QC shots are lost at the site edges, because the population is fixed over ±30 m.
- 7 tracks carry their own offset, with 5–60 shots each. The per-track result is weak
  evidence either way at this sample size.
- RH98 floor: there are no headline shots below 3 m, because of the stratum. The
  `all` stratum (n = 160) gives the same conclusions (`wiggle_summary.csv`).

## 6. Next

- **Done 2026-10-19: `REPORT_gedi_earlier_years.md`.** **Go back in time** (user plan): the same comparison with 2024 (21 QC shots in the site
  boxes), 2021 (245) and 2019 (28) GEDI against the same 2026 LiDAR, to look for accuracy that
  gets worse as the gap grows. 2023 has no shots there. `YEAR = 2025` is a constant in both
  scripts; make it a `--year` option. Use v003 (better positions) with v002 for comparison.
  The gap also grows the confound from growth and fire, so a degradation is not by itself a
  GEDI property.
- Re-check CMR for post-2025-07-09 GEDI data before any later run
  (`G:\python\GEDI\PLAN_2026_download.md`, 2026-10-19 section).
