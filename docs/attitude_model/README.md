# Post-launch accuracy improvements to the `roman_pointing` attitude model

`roman_pointing` computes Roman's nominal attitude for a target as follows:

* `calcRomanAngles` points the boresight V1 at the target, with the Sun in the V1–V3 plane.
* The observatory is at the ideal Sun–Earth L2 point (`getL2Positions`).
* The Sun and target directions are geometric.
* `getRomanPositionAngle` returns the PA of V3 about V1.

Three simplifications in this model each change Roman's position angle (PA_V3) or Sun angle by up to a few tenths of
a degree. This note describes optional features to account for each of these effects:

* **A. CGI aperture boresight on the target.** Operations place the CGI aperture boresight, 0.386° from V1, on the target, and quote PA_V3
  there. The aperture position comes from the Roman Science Instrument Aperture File ([SIAF](https://roman-docs.stsci.edu/simulation-tools/additional-simulation-tools/pysiaf-for-roman)).
* **B. Roman's actual position.** Roman orbits L2 at up to 840,000 km from it. The position comes from the
  [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) ephemeris.
* **C. Velocity aberration.** Roman moves at about 30 km/s, which displaces the apparent target and Sun directions by
  up to 21″. The attitude is solved from the apparent directions and then expressed in the catalog frame.

All three are implemented in `roman_pointing`, with tests. With the new arguments left at their
defaults, every current output is unchanged. With all three enabled, the model reproduces the Roman SOC observation plans,
the attitude telemetry, and the CGI header pointing values of six commissioning visits to ≤ 0.004° (section 6).

## 1. Summary

| Change | Option | Effect on PA_V3 | Effect on the FOR | Size in the flight-data visits |
|---|---|---|---|---|
| A. Aperture offset | `aperture=getSIAFAperture()` | V2Ref/tan ε − V2Ref·V3Ref: 0 at ε = 90°, ≤ 0.22° at the FOR edges | Sun angle of V1 = target's + V3Ref (−0.234°) | 0.006°–0.020° (ε ≈ 86°–92°) |
| B. Horizons ephemeris | `r_obs_G=getRomanPositions(ts)` | ≤ 0.33° (rms 0.16°), varying over months | Sun angle ≤ 0.30° (rms 0.15°) | 0.28°–0.30° |
| C. Velocity aberration | `v_obs_G=getRomanVelocities(ts)`, then `removeAberration` | ≈ −8.3 min × dPA_V3/dt (Sun) plus ≤ 0.004° (target) | Sun angle ≤ 20.6″ | 0.005°–0.006° |

ε is the Sun angle of the target, and V2Ref, V3Ref are in radians.

* **B is usually the largest.** It varies with Roman's orbit over months and is the same for all targets on a date.
* **A depends on the Sun angle.** It vanishes at ε = 90° and dominates near the FOR edges.
* **C is the smallest,** but at 0.003°–0.009° it is still larger than the 0.001° rounding of the observation plans.

All three change the PA_V3 rate by ≲ 2%, so the analytic rate model in `field_rotation.py` remains valid as it is.

Verification against the Roman SOC observation plans (section 6.3). The residual is the planned PA_V3 (OPUP "CGI
Fixed v3pa@T") minus the model's nominal PA_V3 at the planned attitude epoch; the table gives its largest absolute
value over the three visits with plans.

| Model | Max \|planned − model\| |
|---|---|
| Default (V1, L2, geometric) | 0.30° |
| A only | 0.31° |
| B only | 0.014° |
| A + B | 0.009° |
| A + B + C | 0.0035° |

## 2. Conventions

* **Body frame.** The rows of `B_C_I[:, :, t]` are the body axes b₁, b₂, b₃ (V1, V2, V3) in inertial
  (BarycentricMeanEcliptic) coordinates. The unit vector to an aperture reference point in the body frame is
  u_B = (cos V2 cos V3, sin V2 cos V3, sin V3), and its inertial direction is `B_C_I[:, :, t].T @ u_B`.
* **Catalog and apparent frames.**
  * Directions in a catalog, in the observation plans (OPUP, CPGS) and in the WFI pointing metadata are
    aberration-free: the catalog frame.
  * The attitude quaternion and the CGI L1 header pointing keywords (`RA_BORE`, `RA_APER`, `PA_V3`, `PA_APER`)
    describe the same attitude in the apparent frame of the moving observatory (section 6.2).
  * The two frames differ by up to 21″ in position and 20.6″ × tan |Dec| in PA.
* **Which PA_V3.** CPGS and the plans quote PA_V3 at the aperture, i.e. the local direction of V3 at the target, in
  the catalog frame. With an `aperture`, `getRomanPositionAngle(B_C_I, aperture)` returns this quantity.
  * With `aperture=None` it returns PA_V3 at V1, which differs by meridian convergence (up to 0.83° for |Dec| < 60°
    and 7.7° at |Dec| = 87°) and is useful only for diagnostics.
* **Field of regard.** The Sun-angle limits (54°–126°) apply to V1, because they are set by the illumination of the
  solar panels (normal +V3). They concern the physical, i.e. apparent, Sun direction. `calcRomanAngles` always returns
  the Sun angle of V1.
* **Attitude epoch.** An observation plan fixes PA_V3 for the planned attitude epoch (the start of the visit slew),
  and the attitude is then held inertially fixed. Comparisons with plans must use that epoch (section 6.2).

## 3. Change A: CGI aperture on the target

### 3.1 Effect

The SIAF gives the calibrated aperture reference point:

| `CGI_CEN` (`roman_siaf_inflight_20261021.xml`) | Value |
|---|---|
| V2Ref, V3Ref | −1104.214″, −840.945″ |
| Offset from V1, δ | 0.3855° |
| Direction of offset | 52.7° from −V3 toward −V2 (lower right in the observatory field-of-view diagram) |
| V3IdlYAngle, VIdlParity | −30.13967°, −1 |
| Ideal-frame vertices | ±11.16″ |

SIAF V2 and V3 are the body axes b₂ and b₃. Values below are for 400 random targets in the FOR on 2027-03-01, plus
the examples in [`docs/field_rotation_rate/README.md`](../field_rotation_rate/README.md).

* **Nominal roll and PA_V3.** Moving V1 so that the aperture lands on the target has two components:
  * The V3 component lies along the Sun–V1 great circle and only changes pitch.
  * The V2 component is perpendicular to it. Keeping the Sun in the V1–V3 plane then needs a roll of −V2Ref/tan ε.

  The PA_V3 at the target therefore changes by

    ΔPA ≈ V2Ref/tan ε − V2Ref·V3Ref, with V2Ref and V3Ref in units of radians.

  * The second term is a constant −0.00125°; the residual error of this approximation is < 10⁻⁶°.
  * |ΔPA| is 0 at ε = 90° and at most 0.22° at the FOR edges.
  * The rate changes by ≤ |V2Ref| λ̇☉/sin²ε ≈ 0.008°/day.

  ![Geometry of the nominal-roll change](boresight_offset_geometry.png)

  *Made with `tools/boresight_offset_diagram.py`, for ε = 60°, with the focal-plane offsets exaggerated ×10. The sky*
  *around the target in a gnomonic projection (great circles are straight lines), in the body frame of the final*
  *attitude (+V3 up, +V2 left). The aperture's offset from V1 splits into a V3 part along V1's great circle to the Sun*
  *and a V2 part across it. At nominal roll, V3 points along the great circle to the Sun, so the PA change is the*
  *angle at which the great circles from V1 and from the target meet at the Sun. In the shaded right triangle*
  *T–N–Sun, the side opposite the Sun is |V2Ref| and the hypotenuse is tan ε, so sin ΔPA = |V2Ref|/tan ε.*

* **Field of regard.** V1 is displaced toward the Sun by V3Ref, so the Sun angle of V1 is the target's minus 0.234°.
  The target's FOR becomes 54.23° ≤ ε ≤ 126.23°, which moves the window edges by ≈ 0.24 day near the ecliptic.
* **Roll axis.** A 15° roll about V1 moves the target 362″ off the aperture, far outside the ±11″ EXCAM field, and
  changes the aperture PA by 14.7°–15.0°. A roll about the aperture axis keeps the target fixed and changes the
  aperture PA by exactly the roll.

### 3.2 Implementation

* **SIAF extract and reader.** The full SIAF release is 16 MB, almost all of it WFI SpecPars grids, and the public
  [pysiaf](https://github.com/spacetelescope/pysiaf) repository does not yet hold the commissioning SIAF.
  * `src/roman_pointing/siaf/roman_siaf_inflight_20261021_CGI.xml` is a 4 kB copy of the release header and its CGI
    `SiafEntry` blocks, made with `tools/extract_siaf_cgi.py`, and packaged via `[tool.setuptools.package-data]`.
  * `getSIAFAperture(aper_name="CGI_CEN", siaf_file=SIAF_FILE)` returns `V2Ref`, `V3Ref`, `V3IdlYAngle`,
    `VIdlParity` and u_B. The keys match pysiaf's aperture attributes, so a later switch to
    `pysiaf.Siaf("Roman")[aper_name]` only changes this function.
  * Extracts must come from releases dated 20260928 or later. Earlier releases used a different ideal-frame
    convention (V3IdlYAngle ≈ 240°).
* **`calcRomanAngles(..., aperture=None)`.**
  * The two Sun-alignment rotations are unchanged. The yaw and pitch steps point b₁ at a direction that is
    iterated until the aperture lands on the target: `rhat_point += rhat_target − B_C_Iᵀ u_B`.
  * The iteration converges by ~10³ per pass (1388″ → 2–5″ → 0.01″ → numerical floor). It keeps the Sun in the
    V1–V3 plane; rotating the V1-on-target attitude rigidly onto the aperture would not.
  * `sun_ang`, `yaw` and `pitch` refer to V1.
* **`applyRollAngle(..., aperture=None)`.** Rolls about u_B with Rodrigues' formula, `angutils.calcDCM(u_B, -a)`.
  `calcDCM` returns the DCM in the opposite sense to `rotMat`, so `calcDCM([1, 0, 0], -a)` equals `rotMat(1, a)`
  and the sign convention carries over: a roll of +r lowers the aperture PA by r.
* **`getRomanPositionAngle(..., aperture=None)`.** Measures the PA about the aperture direction, projecting both north
  and b₃ onto the plane perpendicular to it. The projection of b₃ is the local +V3 direction at the aperture, the
  quantity pysiaf's `posangle` returns.
* **`getEXCAMPositionAngle(..., aperture=None)`.** Gives PA_APER = PA_V3 + V3IdlYAngle + 180° = PA_V3 + 149.86° for the 2026-10-21 CGI SIAF.

### 3.3 Tests (`tests/test_aperture_offset.py`)

* `aperture=None` reproduces the main branch to 10⁻⁹° (regression values). An aperture at V1 (u_B = e₁) reproduces
  the default attitude, roll and PA.
* For 25 targets across the FOR:
  * the aperture is on the target (< 10⁻⁹) and b₂·r̂_sun = 0;
  * the Sun angle of V1 minus the target's equals V3Ref to 10⁻³°;
  * ΔPA matches V2Ref/tan ε − V2Ref·V3Ref to 10⁻⁵°, which catches a sign or axis-convention error.
* A roll of r about the aperture keeps the target on it and changes the aperture PA by exactly −r.
* If pysiaf is installed (checked with 0.29.0): `pysiaf.utils.rotations.attitude` and `posangle` confirm the V2/V3
  axis and V3IdlYAngle conventions. For Roman, `pysiaf.Siaf` ignores `filename` unless `basepath` is also given.

## 4. Change B: Roman's position from the JPL Horizons ephemeris

### 4.1 Effect

`getL2Positions` scales the Earth's barycentric position by the collinear L2 distance. Every Sun direction, Sun angle
and nominal attitude follows from this position. Roman's [Horizons](https://ssd.jpl.nasa.gov/horizons/) trajectory (`-211`) is available from 2026-08-30
to 2028-11-08 (as of 2026-11-11; NASA will extend and update it).

* **Position.** Roman is 290,000–840,000 km from the ideal L2 point, so the Sun direction differs by 0.06°–0.31°.
* **Sun angle and PA_V3.** For 400 random targets at random dates, the Sun angle differs by up to 0.30° (rms 0.15°)
  and the nominal PA_V3 by up to 0.33° (rms 0.16°).
* **Rates.** The offset changes by ≤ 0.012°/day, so PA_V3 rates change by ≲ 2% and the roll-offset limits by
  ≲ 0.1° over a 9-day span.
* **FOR edges.** The edges move by up to ~0.3 day near the ecliptic.

For comparison, roman-opup-tools queries Horizons for the Sun as seen from Roman, once per epoch, and converts the
ecliptic vector with a fixed obliquity.

### 4.2 Implementation

The implementation follows the astroquery recipe of the [`roman_pointing_demo` notebook](https://github.com/roman-corgi/roman_pointing/blob/main/Notebooks/roman_pointing_demo.ipynb) ("Using a real Roman Orbit")
and the structure of the [EXOSIMS Roman observatory](https://github.com/dsavransky/EXOSIMS/blob/master/EXOSIMS/Observatory/Roman.py) (`EXOSIMS/Observatory/Roman.py`): a Horizons query, a cubic-spline
interpolant of the table, and its evaluation at the requested times.

* **`queryRomanOrbit(ts, step="1d")`** queries Horizons with astroquery,
  `Horizons(id="-211", location="@0", ...).vectors(refplane="earth")`, for one table spanning `ts`, padded by
  2 days. It takes about 1 s per call.
  * The times are TDB (Barycentric Dynamical Time, the uniform time scale of solar-system ephemerides, about
    UTC + 69 s in 2026), the time scale of Horizons vector tables.
* **`queryRomanOrbitSpan(start, stop, step)`** makes the query for `queryRomanOrbit`.
  * Its results are cached in memory for the Python session (`functools.lru_cache`), so tools that loop over targets
    with the same time span make one query. They are not cached on disk (astroquery's `cache=False`), because the
    Horizons trajectory of Roman is updated after orbit determination.
  * If the span extends beyond the Horizons coverage, it is clipped to the coverage edge reported in the Horizons
    error message ("No ephemeris … prior to/after A.D. …") and queried again.
* **`genRomanOrbitInterpolant(vectors_table)`** is the [EXOSIMS](https://github.com/dsavransky/EXOSIMS/blob/master/EXOSIMS/Observatory/Roman.py) function of that name: a `CubicSpline` over MJD, with
  the unit check. The table times are labeled TDB.
  * The spline is within 0.5 km of hourly Horizons values, and allows any `ts`, not only the daily table grid of the
    notebook.
* **`getRomanPositions(ts, step="1d")`** evaluates the interpolant and returns barycentric BarycentricMeanEcliptic
  positions (AU), the same form as `getL2Positions`, so the result can be passed as `r_obs_G`.
  * The query is for ICRF vectors, which are converted to ecliptic coordinates with astropy, as `getSunPositions`
    does. EXOSIMS and the notebook use Horizons' own ecliptic or equatorial planes instead; the difference is < 10 km.
* **`getRomanVelocities(ts, step="1d")`** returns the derivative of the same interpolant (km/s) and is used by change C.
* **Fallback.** Without a network, both functions warn and return the L2 approximation. Times in `ts` outside the
  Horizons table (beyond the coverage) get the L2 approximation, with a warning, and the other times the Horizons
  orbit. A span that crosses the coverage edge, like a 2-year observability window, therefore uses Horizons where
  it can.

### 4.3 Tests (`tests/test_roman_ephemeris.py`; tests skipped when there is no network access)

* `getRomanPositions` lies 2–9 × 10⁵ km from `getL2Positions` within the coverage.
* `genRomanOrbitInterpolant` reproduces the table at its nodes, and gives the same result for a table in km.
* The daily spline agrees with hourly Horizons values to < 1 km.
* The Sun direction from Roman agrees with Horizons' Sun vector as seen from Roman to < 1″ (0.03″ achieved).
* `getRomanVelocities` agrees with finite differences of `getRomanPositions` to 10⁻³ km/s.
* Outside the coverage, the functions warn and return L2 values. For times on both sides of the coverage edges,
  only the times outside get L2 values.
* A repeated call for the same span uses the session cache.
* `pa_v3_drift` uses the Horizons orbit by default, and `ephemeris="L2"` the L2 point.
* The full model (A + B + C) reproduces the OPUP planned PA_V3 to 0.005°. For the CGI-prime visits this is
  "CGI Fixed v3pa@T", which A + B alone misses by up to 0.009°. For the WFI-prime visits of CAR-193.2
  it is the V3PA at V1.

## 5. Change C: velocity aberration

### 5.1 Effect

Roman's barycentric velocity displaces the apparent direction of every source toward the direction of motion, by up
to v/c = 20.6″. The nominal attitude is defined by two vectors, and both are seen in apparent directions:

* the target, whose apparent image must be in the aperture;
* the Sun, which must be in the V1–V3 plane.

The physical attitude therefore follows from the aberrated target and Sun directions. The plans describe that same
attitude in the catalog frame, by removing the aberration at the aperture.

* **Sun term.** The Sun's apparent longitude lags its geometric longitude by 20.5″. This is the same as evaluating
  the nominal roll 8.3 min earlier, ΔPA ≈ −8.3 min × dPA_V3/dt, i.e. 0.003°–0.009° for typical rates.
* **Target term.** Up to ≈ 20.6″/tan ε. For 120 random targets in the FOR: ≤ 0.004° at ε = 54°–60° and 120°–126°,
  and ≤ 0.0014° at 75°–105°.
* **Frame of the result.** The apparent-frame PA_V3, as in the CGI headers, differs from the catalog-frame PA_V3 by
  up to 21″ × tan |Dec|: 0.010° at 60°, 0.032° at 80° and 0.065° at 85°.
* **Field of regard.** The apparent Sun angle differs from the geometric one by ≤ 20.6″ (0.006°), about 9 minutes of
  FOR-edge timing.

### 5.2 Implementation

* **`applyAberration(rhat, v_obs_G, inverse=False)`** implements the special-relativistic aberration of
  3×n unit vectors. With β = v/c and γ = (1 − β²)^(−1/2):

    r̂′ = [r̂/γ + β + γ/(1+γ) (r̂·β) β] / (1 + r̂·β),

  and `inverse=True` replaces β by −β (apparent → catalog). To first order, r̂′ = r̂ + β − (r̂·β) r̂.
* **`getL2Velocities(ts)`** gives the velocity of the L2 approximation, by central differences. It is used with the
  default ephemeris.
* **`calcRomanAngles(..., v_obs_G=None)`.** If a velocity is given, the target and Sun directions are aberrated
  before the attitude solution. `B_C_I` is then the physical attitude (the frame of the quaternion), and `sun_ang`
  the apparent Sun angle.
* **`removeAberration(B_C_I, v_obs_G, aperture=None)`** rotates each attitude by the smallest rotation that moves the
  apparent direction of the reference point (aperture, or V1) to its catalog direction. `getRomanPositionAngle` of
  the result is the catalog-frame PA_V3 of the plans and CPGS. This is the same convention as the WFI pipeline's
  pointing metadata.

### 5.3 Tests (`tests/test_aberration.py`, offline with L2 velocities)

* The shift has magnitude β sin θ and points toward the velocity, and the inverse round trip is exact (10⁻¹⁴).
* Zero velocity reproduces the default attitude exactly.
* With a velocity, the aperture is on the apparent target (< 10⁻⁹), the apparent Sun is in the V1–V3 plane, and
  `sun_ang` is the apparent Sun angle.
* `removeAberration` puts the aperture on the catalog target and agrees with an independent two-point transport of
  the PA to 10⁻⁶°.
* `pa_v3_drift(aberration=True)` changes the 11 UMi PA_V3 by the expected few 0.001° without changing the rate.

## 6. Verification with flight data

### 6.1 Data

| Visit | Date (2026) | Target | CGI L1 (CDMS) | WFI (`_wcs.asdf`) | Planned attitude epoch |
|---|---|---|---|---|---|
| CAR-55.1 | Oct 18 | LMC field (80.5525°, −69.5134°) | `0115601001001001001` (515) | `0103201001001001001` | 291/04:44:00 |
| CAR-55.4 | Oct 22 | LMC field | `0115601002001001001` (520) | `0103202002001001001` | 295/07:03:46 |
| CAR-102.7 | Oct 25 | n/a | `0110701003002001001` (520) | — | 298/12:04 (planned) |
| CAR-101.1 | Oct 26 | n/a | `0110601001002001001` (520) | — | 299/20:45 (planned) |
| CAR-193.1 | Nov 1 | 11 UMi (229.2745°, +71.8239°) | `0116001001001001001` (521) | `0104001001001001001` | 305/04:57:00 |
| CAR-193.2 | Nov 12 | WFI-prime 2×1 mosaic: (150.0529°, +2.1405°) and (150.3673°, +2.4193°) | `0116001002002001001` (522), parallel darks | `0104902002001001001`, `…001002`, `…002001`, `…002002` (prime) | 315/23:28:23, 316/00:08:37, 316/00:19:00, 316/00:25:09 |

* **CGI:** L1 files from [MAST](https://mast.stsci.edu/) (`roman_cgi`). (L2b files of the same frames carry identical pointing keywords and give identical results.)
* **CAR-193.2** reverses the roles: WFI is prime (CAR-198.2, with WFI_CEN on the target and dithers), and CGI takes
  dark frames in parallel. The CGI headers have no `RA`/`DEC`; their pointing keywords are complete.
* **WFI:** `_wcs.asdf` files from [MAST](https://mast.stsci.edu/search/ui/#/roman) (`roman`), 49 kB each. They contain:
  * `meta.pointing`: `ra_v1`, `dec_v1` and `pa_v3` are `CALCULATED`, i.e. from the attitude telemetry. `quaternion`
    is the raw ACS attitude quaternion `SCF_AC_SDR_QBJ` at mid-exposure (scalar last, ICRS to body).
  * `meta.ephemeris`: the predicted spacecraft state.
  * `meta.wcs_fit_results`: the alignment of the WFI image to Gaia stars.

* **Retrieving the files.** Search MAST by the visit IDs above: the
  `roman_cgi` collection for CGI and the `roman` collection for WFI. Commissioning data may still be in their
  exclusive-access period, which requires an authentication token. The files used:

  | Visit | CGI files (`cgi_<visit ID>_<time>_<level>.fits`) | WFI files |
  |---|---|---|
  | CAR-55.1 | `…_20261018t0532019_l1_` | `r0103201001001001001_0003_wfi01_f129_wcs.asdf` |
  | CAR-55.4 | `…_20261022t0739286_l1_` | `r0103202002001001001_0002_wfi01_f129_wcs.asdf` |
  | CAR-102.7 | `…_20261025t1452128_l1_`, `…_20261025t1452221_l1_` | — |
  | CAR-101.1 | `…_20261026t1828419_l1_`, `…_20261026t1853167_l1_`, `…_20261026t1917119_l1_` | — |
  | CAR-193.1 | `…_20261101t0051382_l1_`, `…_20261101t0101510_l1_`, `…_20261101t0107434_l1_` (first, middle and last of the 27 frames during the WFI parallel), `…_20261101t0145004_l1_`, `…_20261101t0148104_l1_` (after it) | `r0104001001001001001_0004_wfi01_f129_wcs.asdf`, `r0104001001001001001_0013_wfi01_f129_wcs.asdf` |
  | CAR-193.2 | `…_20261112t0057073_l1_` (during WFI exposure 0002 of the first visit) | `r0104902002001001001_000[1-4]_wfi01_f062_wcs.asdf`, `…001002_000[1-4]_wfi01_f062`, `…002001_000[1-4]_wfi01_f158`, `…002002_000[1-4]_wfi01_f158` (16 files) |
* **Plans:** the Roman SOC observation plans (OPUP timelines; internal to the project) give, for each visit, the slew
  and visit-begin times, the planned PA_V3 at the target ("CGI Fixed v3pa@T" or "WFI Fixed v3pa@T", catalog frame),
  V1 with its PA_V3, and the roll from nominal. The planned and actual execution times come from SOC records.
* **WFI_CEN** (V2Ref 1686.548″, V3Ref −1014.664″, 0.547° from V1) comes from the full
  `roman_siaf_inflight_20261021.xml` release. The packaged extract holds only the CGI apertures.

### 6.2 Conventions established

* **Attitude epoch.** 
  * The plan fixes PA_V3 at the start of the visit slew. This is usually the same as the planned execution time.
  * The OPUP roll grows at the Sun's rate from about 0 at that epoch.
  * The attitude is computed once, for t₀, and is not recomputed when the visit executes at another time: the planned PA_V3 is flown, so the roll from nominal is the PA_V3 rate times the time offset (4.4 h early for CAR-193.1, for example).
* **Coordinate frames (change C).**
  * The CGI L1 pointing keywords equal the raw ACS attitude quaternion (`SCF_AC_SDR_QBJ`, carried in the WFI
    metadata) to 0.07″–0.19″ and 5 × 10⁻⁵°.
  * WFI `meta.pointing` is the same quaternion with the aberration removed at the WFI target aperture (WFI_CEN),
    to 0.003″–0.04″ and 10⁻⁵°, and agrees with the OPUP V1 and V3PA to their rounding.
  * The WCS metadata describing the Gaia alignment of the WFI images changes the orientation by only 0.002°, much
    less than the 0.009°–0.017° PA difference between the apparent and catalog frames. The catalog frame,
    used by the observation plans and the WFI `meta.pointing`, is therefore the frame of the true (Gaia) sky.
  * The positions of the target in the apparent (aberrated) and catalog (aberration-free) frames differ by (v/c) sin θ, where θ is the angle between the target and Roman's velocity:
    20.1″–20.7″ for 11 UMi and the LMC field (θ near 90°), but only 5.5″ for the CAR-193.2 field (θ ≈ 15°).
* **Aperture (change A).**
  * The header PA_V3 is the local PA_V3 at the aperture: the attitude defined by the header boresight and aperture
    positions reproduces it to < 10⁻⁴°.
  * The 20261021 SIAF CGI_CEN is in the aberration-corrected frame. SSC's star-matched EXCAM centre, converted with
    the aberration-corrected WFI attitude, gives V2/V3 within 0.01″ of it for CAR-55.1 (the data used for the
    update), and within 0.04″ for CAR-55.4 (after the release). Without the correction both would differ by 20.6″.
  * CDMS 515 headers (CAR-55.1) were computed with the preflight SIAF: `RA_APER` is 204″ from the star-matched centre,
    and `PA_APER` = `PA_V3` + 150°.
* **Ephemeris (change B).** The WFI `meta.ephemeris` (predicted, EME2000) matches `getRomanPositions` to 116 km, and
  `getRomanVelocities` to < 10⁻⁴ km/s.

### 6.3 Results

With all three changes, the model reproduces the planned PA_V3 (OPUP "CGI Fixed v3pa@T") of the three CGI-prime
visits with plans to ≤ 0.0035° at the planned attitude epoch, and the PA_V3 from the attitude telemetry and the CGI
headers of the CGI-prime visits to ≤ 0.004°; the previous baseline model differs from the plans by 0.27°–0.30°. For
these dates, change B removes most of that error (0.28°–0.30°), A removes 0.006°–0.020° (small because the targets are
near ε = 90°), and C removes the remaining 0.005°–0.006°, changing sign between the LMC field and 11 UMi. The model's
roll from nominal grows at the OPUP's rate over each visit, and given the plan's PA_V3, the model predicts the PA_V3 at
V1 in the telemetry during execution to ≤ 0.0007°. For the four WFI-prime visits of CAR-193.2, with WFI_CEN on the
target, the full model matches the flown PA_V3 to +0.0022°, against 0.08° for the previous baseline. The comparisons
with the CGI headers and WFI telemetry were made with `tools/compare_header_pa.py` (usage examples in its docstring);
the full result tables are documented in an internal memo on the CPP Collaboration confluence workspace.

## 7. Interfaces and scope

All three changes together, for the PA_V3 that the Roman Science Support Center CPGS and the plans quote:

```python
aperture = getSIAFAperture()                                         # A
r_obs_G, v_obs_G = getRomanPositions(ts), getRomanVelocities(ts)     # B (and the velocity for C)
sun_ang, yaw, pitch, B_C_I = calcRomanAngles(target, ts, r_obs_G, aperture=aperture, v_obs_G=v_obs_G)   # C
pa_v3 = getRomanPositionAngle(removeAberration(B_C_I, v_obs_G, aperture), aperture)
```

* **`field_rotation.pa_v3_drift(ra, dec, t_mid, dt, aperture="CGI_CEN", ephemeris="horizons", aberration=False)`**
  does the same for a range of times, with A and B by default (`aperture` takes a SIAF aperture name, an aperture
  from `getSIAFAperture()`, or `None` for V1). Its `sun_ang` is always that of V1, so the FOR checks against `SUN_ANG_MIN`/`SUN_ANG_MAX`
  remain correct.
* **`tools/plan_roll_limits.py`** uses `--aperture CGI_CEN --ephemeris horizons` by default; add `--aberration` for C,
  or use `--aperture none --ephemeris L2` for the boresight model. The reported PA_V3 is then the quantity CPGS uses,
  and the roll-offset arithmetic (a roll of +r changes PA by −r) is exact at the aperture. In the worked examples of
  the field-rotation README the defaults change the roll limits by ≤ 0.03° and the reachable PAs by 0.2°–0.6°.
* **The analytic functions** in `field_rotation.py` (`pa_v3_rate`, `for_window`, `fixed_pa_window`, …) remain
  boresight models. The corrections of section 1 can be applied analytically where needed.
* **Planning tools: A and B by default.** These have new arguments `aperture="CGI_CEN"` (`None` for V1) and
  `ephemeris="horizons"` (`"L2"` for the L2 point), with aberration off. They are offline-tested in
  `tests/test_planning_defaults.py`.
  * `roman_excam_planner.py` (`calculate_roman_EXCAMangles`, `compute_angles`). The PA_V3 and PA_APER are measured
    at the aperture, with PA_APER = PA_V3 + V3IdlYAngle + 180°, and the roll is about the aperture. In a test case
    (47 UMa, June 2027) PA_APER changes by 0.22°–0.55°.
  * `roman_observability.py` (`compute_roman_angles`, `compute_keepout`), and through it
    `Reference_Star_Selection_Tool.py`. The Sun angle and pitch are those of V1 with the aperture on the target.
    Over 2 years at 1-day steps, the observable days of five test targets change by ≤ 3.
  * `drive_cpgs.py` (`find_crossing_time_diff`), which compares FOR crossing times with CPGS. Aberration, left off,
    can still move a crossing by up to ~9 min (section 5.1).
  * `tools/zero_pitch_targets.py` (`--aperture`, `--ephemeris`). The Simbad search circles follow the track of the
    aperture at zero V1 pitch. The search now also uses the yaw sampling derived from `--bound`, which a hard-coded
    100-point yaw grid previously overrode.
* **Not changed:** `slewing.py`, because the options change the yaw and pitch differences between targets by
  ≤ 0.25°, a negligible fraction of a slew; and `tools/field_rotation_rate.py`, which validates the analytic V1-at-L2
  model (the aperture term V2Ref/tan ε is not part of it). The notebooks that call `calcRomanAngles` directly are
  also unchanged.
* **Size:** about 600 added and 120 removed lines in `roman_pointing.py`, `field_rotation.py`, the four planning
  tools above, `plan_roll_limits.py` and `pyproject.toml`, many of them docstrings. There are also the 4 kB SIAF extract, `tools/extract_siaf_cgi.py`,
  `tools/compare_header_pa.py`, and 28 test functions in four files. The field-rotation README
  ([`docs/field_rotation_rate/README.md`](../field_rotation_rate/README.md), sections 2 and 7) documents all three
  corrections.
* **Dependencies:** none new. `angutils.calcDCM`, `scipy.interpolate.CubicSpline`, `astroquery.jplhorizons` and
  the standard-library `xml.etree` are used. `asdf` is needed only for `compare_header_pa.py --wfi`.

## 8. Open items

* **Future SIAF updates.** When a new SIAF release arrives, regenerate the extract, update `SIAF_FILE`, and remove the
  previous extract.
  * During science operations, the most recent SIAF will be synced to the public pysiaf repo. It will then be possible to modify `getSIAFAperture` to import the aperture offset and orientation data directly from pysiaf (https://github.com/spacetelescope/pysiaf).       