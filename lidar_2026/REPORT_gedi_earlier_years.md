# REPORT — 2024, 2021 and 2019 GEDI RH98 against the 2026 drone LiDAR, with fire groups

**Date:** 2026-10-19. **Status:** complete. It continues `REPORT_gedi_2025_wiggle.md` (method,
2025 numbers) and `REPORT_gedi_2025_fire.md` (fire method, 2025 fire numbers).
Code: the same three scripts, now with `--year`: `gedi_site_shots.py`, `gedi_wiggle.py`,
`gedi_fire.py`. Logs: `gedi_site_shots_<year>.log`, `gedi_wiggle_<year>.log`,
`gedi_fire_<year>.log`. Outputs: `outputs/gedi/shots_<year>_sites_v<ver>.parquet` and
`outputs/gedi/<year>/` (2025 stays in `outputs/gedi/`). Numbers are v003, footprint p90,
nominal position, RH98 3–20 m, unless stated.

## 1. Result in brief

- **Agreement does not get worse with a longer gap.** r(RH98, p90) is 0.70 (2025, gap
  15–20 months, n 154), 0.66 (2024, 22 months, n 15), 0.74 (2021, 5.3 years, n 228) and
  0.85 (2019, 7.4 years, n 19). RMSE is 1.45–1.88 m in every year. The years cover different
  sites (table in §3), so this is not a clean time series, but no loss of skill is visible.
- **The global offset only holds in the large samples.** In 2021 (v003) the offset
  (−5.0, +7.5) m gives +0.029 in r, also in leave-one-out, as 2025 did. In 2024 and 2019
  (15 and 19 shots) the in-sample gain (+0.38, +0.06) becomes negative in leave-one-out
  (−0.26, −0.12): over-fit. Use the nominal positions for those years.
- **Fire: in 2021 every shot burnt between GEDI and the LiDAR; in 2024 every shot did.**
  So those years have no fire-free group. In 2019, 12 of 19 shots (Bunda Extra) had no
  scar in 7 years.
- **More fires do not consistently mean a lower LiDAR canopy.** Within-site slope of
  (p90 − RH98) on the number of burnt periods between GEDI and the LiDAR:

  | Year | Area | n | slope (m per burnt period) | 95 % |
  |---|---|---|---|---|
  | 2025 | Biota track | 47 | −0.79 | [−1.41, −0.16] |
  | 2025 | pooled, site means removed | 154 | −0.29 | [−0.87, +0.30] |
  | 2024 | South Dampier | 15 | +0.50 | [−0.84, +1.93] |
  | 2021 | South Dampier | 66 | −0.57 | [−1.02, −0.17] |
  | 2021 | 9 Mile | 76 | +0.32 | [−0.17, +0.88] |
  | 2021 | pooled, site means removed | 228 | −0.25 | [−0.51, +0.02] |
  | 2019 | Bunda Extra | 16 | +0.39 | [−0.01, +0.97] |

  Two sites have the expected sign (more fire, lower LiDAR p90 relative to RH98) with an
  interval that excludes zero. The pooled 2021 slope is −0.25 m per burnt period and only
  borderline. 9 Mile and 2019 Bunda Extra go the other way.
- **2019 is the opposite of a fire screen gain.** At Bunda Extra the 5 burnt shots (4–5
  burnt periods each) have p90 1.82 m higher relative to RH98 than the 11 unburnt shots
  (bootstrap 95 % 0.11 to 4.06 m), and r is about the same (0.66 against 0.76; Δr interval
  −0.28 to +1.72). The burnt shots are a different part of the site, so this is site
  structure or regrowth, not a fire loss.
- **Conclusion for the back-in-time work:** fire is not a usable screen for these sites.
  It removes all shots in 2021 and 2024, and the remaining effect is site-dependent and
  smaller than the 1.5–1.9 m RMSE. The GEDI-to-LiDAR agreement is about the same over a
  7-year gap as over a 1.5-year gap.

## 2. Data

| Year | Granules (v002 / v003) | QC shots in AOI (v002 / v003) | Full-disc shots, headline (v003) | Shot dates (headline) | Sites (headline, v003) |
|---|---|---|---|---|---|
| 2025 | 10 / 11 | 257 / 258 | 154 | 2025-02-13 to 07-09 | 9 Mile, Biota, Bunda Extra, Country Downs (+2 + 2) |
| 2024 | 10 / 10 | 23 / 24 | 15 | 2024-12-15 | South Dampier |
| 2021 | 21 / 21 | 280 / 361 | 228 | 2021-05-14 to 06-23 | 9 Mile 76, South Dampier 66, Biota 45, Country Downs 41 |
| 2019 | 18 / 19 | 33 / 33 | 19 | 2019-05-02 (17), 09-17 (2) | Bunda Extra 16, 9 Mile 2, Bunda Bunda 1 |

- v002 check: the QC shots in the unbuffered site boxes equal the
  `chm_regression/outputs/gedi/shots_<year>_calendar.parquet` counts in every year (2024 21,
  2021 245, 2019 28).
- **2021: v003 has 81 more QC shots than v002** (Biota track 106 against 25). RH98 is the
  same in both versions for the paired QC shots of 2025; for 2021 the cause of the
  difference (the v003 `l2a_quality_flag_rel2` against the v002 `quality_flag`) was not
  investigated. v002 2021 gives the same conclusions (n 187, r 0.64 for p98).
- 2023 has no QC shots over the sites (from the earlier planning count); not run.

## 3. Agreement per site and year (all shots, no fire split)

bias (p90 − RH98, m) / RMSE (m) / r, n in brackets:

| Site | 2025 | 2024 | 2021 | 2019 |
|---|---|---|---|---|
| 9 Mile (pindan) | +0.96 / 1.81 / 0.39 (51) | – | +0.57 / 1.44 / 0.52 (76) | – |
| Biota track | +0.36 / 1.41 / 0.81 (47) | – | +0.65 / 1.26 / 0.88 (45) | – |
| Bunda Extra | −0.10 / 1.65 / 0.76 (30) | – | – | +0.39 / 1.85 / 0.87 (16) |
| Country Downs | +0.57 / 1.58 / 0.85 (22) | – | −0.85 / 1.99 / 0.67 (41) | – |
| South Dampier | – | −0.43 / 1.45 / 0.66 (15) | +0.06 / 1.95 / 0.56 (66) | – |
| all sites | +0.57 / 1.84 / 0.70 (154) | −0.43 / 1.45 / 0.66 (15) | +0.18 / 1.68 / 0.74 (228) | +0.61 / 1.88 / 0.85 (19) |

The shots are different footprints in different years, so a change in bias at one site
between years is not a height change.

## 4. Fire groups

Windows per GEDI year Y (one date for all shots, T0 = Y-01-01): `pre` = scars in Y−1;
`inter` = scars from Y-01 to 2026-10, split into Y Jan–Jul, Y Aug–Dec and each later year.
`n_inter` = the number of these periods with a scar on the footprint.

| Year | clean | pre_only | inter_only | both | n_inter distribution |
|---|---|---|---|---|---|
| 2025 | 39 | 24 | 35 | 56 | 0: 63, 1: 38, 2: 53 |
| 2024 | 0 | 0 | 4 | 11 | 1: 8, 2: 7 |
| 2021 | 0 | 0 | 146 | 82 | 1: 18, 2: 58, 3: 75, 4: 71, 5: 6 |
| 2019 | 12 | 0 | 5 | 2 | 0: 12, 4: 5, 5: 2 |

2021 by number of burnt periods, mean (p90 − RH98) in m (n):

| Site | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 9 Mile | – | – | +0.34 (38) | +0.83 (35) | +0.34 (3) |
| Biota track | +0.64 (3) | +0.74 (21) | +0.56 (21) | – | – |
| Country Downs | −0.90 (15) | −0.63 (20) | −1.47 (6) | – | – |
| South Dampier | – | +0.93 (17) | −0.04 (10) | −0.24 (36) | −0.98 (3) |

Method additions (in the `gedi_fire.py` docstring): the within-site slope was added with the
earlier years, because no clean group exists in 2021; it is fitted with site means removed
and bootstrapped within site (self-test: a planted −0.5 m per fire with a site offset gives
−0.525 [−0.586, −0.463]). Its random stream is separate, so the 2025 null and difference
numbers are unchanged (checked: 0.0 difference against the first run).

## 5. Caveats

- All caveats of `REPORT_gedi_2025_fire.md` §5 apply (250 m NAFI, no severity, one date).
- 2024 (15 shots, one site, one date) and 2019 (19 shots, 16 at one site) are small. Their
  numbers are reported, not interpreted.
- Different sites in different years: the cross-year comparison of r is between sites as
  much as between gaps.
- The headline stratum removes RH98 < 3 m; low vegetation burnt to the ground is not in it.

## 6. Next

- Nothing outstanding on fire for these sites. If more is wanted, the candidates named in
  `REPORT_gedi_2025_fire.md` §6 still apply (night/day, fire history, heterogeneity,
  Sentinel-2 change).
- The 2021 v002/v003 QC-count difference is not explained (§2).
