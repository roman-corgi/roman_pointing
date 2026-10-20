# Rate of Roman field rotation (PA_V3) vs. time of year and ecliptic latitude

PA_V3 is the position angle of the observatory +Z body axis (V3 in Science Center terminology), the body vector
normal to the sunshield. It is equivalent to PA_Z, as returned by `getRomanPositionAngle` in `roman_pointing`.

Analytic model, scheduling guidelines, and validation against `roman_pointing`. The model functions are in
[`roman_pointing.field_rotation`](../../src/roman_pointing/field_rotation.py). Figures and numbers come from two scripts:

* [`tools/field_rotation_rate.py`](../../tools/field_rotation_rate.py): validation, the rate figures, and the
  fixed-PA table (sections 1–3, 5, 6).
* [`tools/plan_roll_limits.py`](../../tools/plan_roll_limits.py): allowed roll offsets for a specific target and visit,
  the roll-limit figures, and the guideline table (section 4).

```
PYTHONPATH=src python tools/field_rotation_rate.py -o docs/field_rotation_rate
PYTHONPATH=src python tools/plan_roll_limits.py --figures -o docs/field_rotation_rate
```

## Summary

For a target at ecliptic latitude β, the nominal-roll position angle PA_V3 changes at

$$
\boxed{\frac{d\,\mathrm{PA_{V3}}}{dt} = -\,\dot\lambda_\odot\,\frac{\sin\beta}{\sin^2\varepsilon}
= -\,\dot\lambda_\odot\,\frac{\sin\beta}{1-\cos^2\beta\,\cos^2(\lambda_\odot-\lambda)}}
\qquad \dot\lambda_\odot \approx 0.9856^\circ/\mathrm{day}
$$

where ε is the Sun–target angle, λ is the target's ecliptic longitude, and λ☉ is the Sun's ecliptic longitude.
PA is measured east of north.

Practical consequences, assuming the 54°–126° Sun-angle field of regard (FOR) and ±15° off-nominal roll:

* **The sign is fixed by the hemisphere.** PA_V3 decreases with time for northern-ecliptic targets and increases for
  southern ones, all year. It never reverses.
* **The rate depends only on β and ε.** Right ascension, declination, and the equatorial-vs-ecliptic north offset
  set the absolute PA_V3 but not its rate.
* **Within the FOR the rate is bounded:**
  * For |β| < 54°: 0.986 |sin β| ≤ |rate| ≤ 1.51 |sin β| deg/day.
    * The minimum is at quadrature (ε = 90°).
    * The maximum is at the FOR edges (ε = 54° or 126°).
  * For |β| ≥ 54° (continuous viewing): 0.986 |sin β| ≤ |rate| ≤ 0.986 / |sin β| deg/day. PA_V3 turns a full 360° per year.
  * The largest rate anywhere in the FOR is **1.22°/day**. It occurs at |β| ≈ 54° at the FOR edge (about 1.26°/day near
    perihelion in early January).
* **How long a single fixed PA_V3 stays reachable with ±15° roll:**

  | Target latitude | Days |
  |---|---|
  | \|β\| ≲ 20° (whole FOR window) | ~73–78 |
  | \|β\| = 30° | 53–57 |
  | \|β\| = 45° | 33–42 |
  | \|β\| ≈ 54° (worst case) | 24.5 |
  | Ecliptic poles | 30 |
* **Maximum requested roll offsets** ([section 4](#4-maximum-roll-offsets-when-cpgs-fixes-the-pa-at-mid-visit)).
  CPGS fixes each visit's sky PA at the planned mid-visit. To keep that PA reachable over a visit of duration D, with the
  start allowed to move by ±U days:

  $$ |A_\mathrm{nom}|,\ |B_\mathrm{nom}| \le 15^\circ - |\mathrm{rate}| \times (D/2 + U) $$

  Each degree of |roll| given up buys 1/|rate| days of start-time slack, about 0.8–1.6 days per degree for
  |β| ≳ 40°. `tools/plan_roll_limits.py` computes the limits for a specific target, start date, duration and start shift.
* **Fixed-PA observations** (e.g., slit spectroscopy; [section 5](#5-fixed-pa-observations-eg-slit-spectroscopy)).
  The roll is adjusted at execution time so that PA_V3 stays fixed; ±15° of roll then covers about 30°/|rate| days.

## 1. Geometry and conventions

In `calcRomanAngles`, the nominal attitude starts with the body +Z axis (b₃) on the Sun. Yaw and pitch then put +X (b₁)
on the target. After the pitch rotation, +Z is the Sun direction projected onto the sky plane at the target.
`getRomanPositionAngle` measures +Z from celestial north about the boresight. Comparing it with astropy's
`position_angle` confirms the sense is the standard **east of north**.

So **PA_V3 is the position angle of the Sun as seen from the target**. EXCAM differs only by a constant:
PA_EXCAM_Y ≈ PA_V3 + 150°. `getEXCAMPositionAngle` uses a nominal 150° offset, which will be replaced by a calibrated
value from the SIAF reference file. Roll about +X (`applyRollAngle`) shifts PA_V3 one-for-one;
+10° of roll gives −10.000° in PA_V3. Neither offset changes the rate, so everything below applies equally to PA_V3,
PA_EXCAM_Y, and any rolled attitude.

## 2. Derivation

Work in ecliptic coordinates and consider the spherical triangle K–T–☉:

* K is the ecliptic north pole.
* T is the target at (λ, β).
* ☉ is the Sun at (λ☉, 0).

Its sides and angles are:

| Element | Value |
|---|---|
| Side K–T | 90° − β |
| Side K–☉ | 90° |
| Side T–☉ | ε (Sun–target angle) |
| Angle at K | Δ ≡ λ☉ − λ |
| Angle at T | θ, the PA of the Sun measured from ecliptic north (toward ecliptic east) |

The standard position-angle formula with β☉ = 0 gives

$$ \tan\theta = \frac{\sin\Delta}{-\sin\beta\cos\Delta}, \qquad
\theta = \operatorname{atan2}(\sin\Delta,\ -\sin\beta\cos\Delta) $$

and the spherical law of cosines gives cos ε = cos β cos Δ.

Differentiate θ with respect to Δ, writing y = sin Δ and x = −sin β cos Δ:

$$ \frac{d\theta}{d\Delta} = \frac{x\,y' - y\,x'}{x^2+y^2}
= \frac{-\sin\beta\,(\cos^2\Delta+\sin^2\Delta)}{\sin^2\Delta + \sin^2\beta\cos^2\Delta}
= \frac{-\sin\beta}{1-\cos^2\beta\cos^2\Delta} = -\frac{\sin\beta}{\sin^2\varepsilon}. $$

For a fixed target, dΔ/dt = λ̇☉. The PA of ecliptic north measured from celestial north, η(λ, β), is constant in time,
so PA_V3 = θ + η:

$$ \frac{d\,\mathrm{PA_{V3}}}{dt} = -\dot\lambda_\odot\,\frac{\sin\beta}{\sin^2\varepsilon}. $$

If the absolute PA is needed: with obliquity ι = 23.44°,
η = −atan2(sin ι cos λ, cos β cos ι − sin β sin ι sin λ). This reproduces `getRomanPositionAngle` to < 0.005°.

**Geometric reading** ([diagram](pa_v3_rate_geometry_diagram.png), panel a):

1. The Sun moves along the ecliptic at λ̇☉. Only the part of its motion across the T–☉ arc rotates that arc.
2. In the triangle, the angle ψ at ☉ satisfies cos ψ = sin β / sin ε. So the cross-arc speed is λ̇☉ sin β / sin ε.
3. Seen from T, a point at angular distance ε moving sideways at speed v turns the arc at v / sin ε.

The two factors of 1/sin ε have different origins. One is the obliquity of the Sun's motion relative to the arc. The
other is the lever arm. Their product gives the 1/sin²ε dependence.

* **Near the ecliptic** (β → 0), the Sun moves almost along the arc, which stays pointed along the ecliptic. The image
  barely rotates.
* **Near the ecliptic pole** (β → 90°), the arc is a meridian of the pole. It turns at exactly the solar rate.

### Day-of-year form (circular orbit)

With a circular Earth orbit:

$$ \lambda_\odot(d) \approx \lambda_0 + 0.98561^\circ\,(d-1), \qquad d = \text{day of year} $$

Here λ₀ is the Sun's mean ecliptic longitude at Jan 1 0h UTC of the year, referred to the J2000 ecliptic and equinox
used by `roman_pointing`: 280.30° in 2026, 280.05° in 2027, and 279.80° in 2028. `sun_longitude_doy` computes it from
the IERS 2003 mean longitude of the Earth (`erfa.fae03`) plus 180°. The rate formula below uses the nominal
0.98565°/day; the 0.004% difference is negligible.

$$ \frac{d\,\mathrm{PA_{V3}}}{dt}(d;\lambda,\beta) = -0.98565^\circ\,\mathrm{day^{-1}}\;
\frac{\sin\beta}{1-\cos^2\beta\,\cos^2\!\big(\lambda_\odot(d)-\lambda\big)} $$

* **Timing by λ.** Target longitude only sets when the pattern occurs. Solar conjunction (Δ = 0) falls on day
  d ≈ 1 + (λ − λ₀)/0.98561 (mod 365.25). The pattern repeats every half year, because the rate is symmetric in
  Δ → 180° − Δ and Δ → Δ + 180°.
* **Eccentricity.** It changes λ̇☉ by ±3.4%: 1.019°/day in early January, 0.953°/day in early July. It also shifts
  λ☉ from its mean value by up to ±1.9° (~2 days), the equation of center.

Eccentricity is the only significant error of the circular model; it does not affect the rate formula in terms of ε.

## 3. Figures

**[pa_v3_rate_geometry_diagram.png](pa_v3_rate_geometry_diagram.png)**:

* (a) Roman at the center of the celestial sphere, using the example β = 45°, Δ = 60°, ε = 69°.
  * Lines of sight run from Roman to the target and to the Sun. ε is the shaded angle between them at Roman.
  * β is measured along the target's meridian, and Δ along the ecliptic.
  * The great circle from the target to the Sun sets +Z on the sky, and θ is +Z measured from ecliptic north.
  * At the Sun, its motion along the ecliptic is shown together with the part that crosses the target–Sun arc.
* (b) |rate| as a function of Sun angle ε and ecliptic latitude, with the FOR marked. Each target moves along a
  horizontal line between ε = |β| and 180° − |β| every half year.

![diagram](pa_v3_rate_geometry_diagram.png)

**[pa_v3_rate_vs_time.png](pa_v3_rate_vs_time.png)**: |rate| through the year vs. days after solar conjunction for
several β, with solid lines in the FOR.

![vs time](pa_v3_rate_vs_time.png)

**[pa_v3_rate_map.png](pa_v3_rate_map.png)**: |rate| vs. days after conjunction and |β| within the FOR. Each contour is
labeled with its rate and the number of days a fixed PA_V3 stays within ±15° at that rate (30°/rate).

![map](pa_v3_rate_map.png)

## 4. Maximum roll offsets when CPGS fixes the PA at mid-visit

### The problem

The observer requests a start date and two roll offsets, A_nom and B_nom. Roll offsets follow the `roman_pointing`
convention: a roll of +A_nom passed to `applyRollAngle` changes PA_V3 (and PA_EXCAM_Y) by −A_nom. For each visit, CPGS
converts the request into an absolute sky PA computed for the planned middle of the visit:

$$ P_A = \mathrm{PA_{nom}}(t_\mathrm{mid}) - A_\mathrm{nom} $$

The spacecraft then holds that PA. The nominal PA keeps rotating, so the roll actually needed at execution time t is

$$ A(t) = A_\mathrm{nom} + \big[\mathrm{PA_{nom}}(t) - \mathrm{PA_{nom}}(t_\mathrm{mid})\big] $$

That needed roll must stay within ±15° over the whole interval in which the visit might execute:

* **The visit itself** reaches D/2 on either side of mid-visit, where D is the visit duration.
* **Any start shift** the schedulers may need adds U days on top of that, for example within a 3 day scheduling buffer.

Long visits, rolls requested near ±15°, and fast-rotating targets push A(t) past the limit. The visit then has only a
narrow range of legal start times.

### Guideline formula

Let H = D/2 + U be the farthest the visit can execute from the planned mid-visit. To first order:

$$ \boxed{|A_\mathrm{nom}|,\ |B_\mathrm{nom}| \;\le\; 15^\circ - |\mathrm{rate}| \times \left(\tfrac{D}{2} + U\right)} $$

Inverting it, for given rolls the start time can move by up to

$$ U_\mathrm{max} = \frac{15^\circ - \max(|A_\mathrm{nom}|, |B_\mathrm{nom}|)}{|\mathrm{rate}|} - \frac{D}{2}. $$

Here |rate| = λ̇☉ |sin β| / sin²ε from section 2, evaluated at the requested date. Use the largest value over the
span for a conservative limit, and add ~3% in December–January. `tools/plan_roll_limits.py` evaluates the exact version.

* **Each degree of |roll| given up buys 1/|rate| days of start-time slack**, shared between early and late. That is
  about 1 day per degree at the ecliptic poles, 0.8 days per degree in the worst case near |β| ≈ 54°, and several
  days per degree near the ecliptic.
* **The two rolls limit different sides.** For a northern-ecliptic target (β > 0), PA_V3 decreases with time, and so
  does the needed roll A(t). Running late pushes a negative roll further negative, and running early pushes a positive
  roll further positive. So for β > 0:
  * the negative roll limits how late the visit can run, and
  * the positive roll limits how early it can run.

  The roles swap for β < 0. If the buffer is one-sided, only one roll needs the reduced limit; the other can go to
  15° − |rate| × D/2.

### Worked example

**Constraints.**

* Target: ε Eridani (RA 53.233°, Dec −9.458°; β = −27.7°).
* Visit duration D = 1.5 days (conservative), planned to start 2027-01-19 00:00 UTC, so CPGS sets the sky PA for the
  planned mid-visit at 2027-01-19 18:00 UTC.
* The start may move up to 3 days early or late (U = 3 days, H = D/2 + U = 3.75 days). The visit can therefore start
  anywhere in the Jan 16–22 scheduling window (Jan 16 00:00 to Jan 22 00:00 UTC) and may end as late as Jan 23 12:00.
* Over that span the Sun angle is 103.5°–110.2° and PA_V3 increases at ≈ 0.52°/day.

**Allowed roll offsets.** The calculator evaluates the constraints with `roman_pointing` and the solar ephemeris:

```
PYTHONPATH=src python tools/plan_roll_limits.py --ra 53.2327 --dec -9.4583 \
    --start 2027-01-19T00:00:00 --duration 1.5 --early 3 --late 3
```

```
Target beta = -27.72 deg, Sun angle 103.5-110.2 deg, dPA_V3/dt at mid-visit = +0.517 deg/day
Allowed requested roll offsets (applyRollAngle sense): -13.02 to +13.09 deg; symmetric limit +/-13.02 deg
```

**Reading the plot** ([roll_limits_example.png](roll_limits_example.png)):

* **Panel (a)** plots the roll needed to hold the CPGS sky PA against the actual execution time. A requested roll is
  allowed only if its line stays within ±15° across the whole possible execution span (gray band).
  * The lines slope upward because PA_V3 increases for β < 0.
  * The dashed lines are the limiting rolls. The −13.02° roll reaches −15° at the start of the span (earliest start),
    and the +13.09° roll reaches +15° at its end (latest end). So the negative roll is limited by running early, and
    the positive roll by running late.
  * The thin lines at ±13° (rolls A and B) are an illustration. They lie just inside the limits, reaching ±15° at
    3.8 days before and 3.9 days after mid-visit, just outside the 3.75-day span.
* **Panel (b)** shows the symmetric limit vs. allowed start shift U for several visit durations. D = 1.5 days with
  U = 3 days gives ±13.02°. Each extra day of U costs about 0.52° of roll, and each extra day of D about 0.26°.
* **One-sided shift.** If the start can only slip later (`--early 0 --late 3`), the negative roll is limited only by
  the visit length, and the allowed range widens to −14.61° to +13.09°.

![roll limits example](roll_limits_example.png)

### Guideline table

Maximum requested |roll| (°) for half-spans H = D/2 + U, at the worst-case date in the FOR and with mid-visit at
quadrature. The last column gives the days of start-time slack gained per degree of |roll| given up.

| \|β\| | max \|rate\|, worst / quad. (°/d) | H = 1 d | H = 2 d | H = 3 d | H = 4 d | slack per 1° (d), worst / quad. |
|---:|---:|---:|---:|---:|---:|---:|
| 5  | 0.13 / 0.09 | 14.9 / 14.9 | 14.7 / 14.8 | 14.6 / 14.7 | 14.5 / 14.7 | 7.6 / 11.6 |
| 10 | 0.26 / 0.17 | 14.7 / 14.8 | 14.5 / 14.7 | 14.2 / 14.5 | 14.0 / 14.3 | 3.8 / 5.8 |
| 20 | 0.52 / 0.34 | 14.5 / 14.7 | 14.0 / 14.3 | 13.5 / 14.0 | 13.0 / 13.6 | 1.9 / 3.0 |
| 30 | 0.75 / 0.49 | 14.3 / 14.5 | 13.5 / 14.0 | 12.8 / 13.5 | 12.1 / 13.0 | 1.3 / 2.0 |
| 40 | 0.97 / 0.63 | 14.0 / 14.4 | 13.1 / 13.7 | 12.2 / 13.1 | 11.2 / 12.5 | 1.0 / 1.6 |
| 45 | 1.07 / 0.70 | 13.9 / 14.3 | 12.9 / 13.6 | 11.9 / 12.9 | 10.8 / 12.2 | 0.9 / 1.4 |
| 50 | 1.15 / 0.76 | 13.9 / 14.2 | 12.7 / 13.5 | 11.6 / 12.7 | 10.5 / 12.0 | 0.9 / 1.3 |
| 54 | 1.22 / 0.80 | 13.8 / 14.2 | 12.6 / 13.4 | 11.3 / 12.6 | 10.1 / 11.8 | 0.8 / 1.3 |
| 60 | 1.14 / 0.85 | 13.9 / 14.1 | 12.7 / 13.3 | 11.6 / 12.4 | 10.4 / 11.6 | 0.9 / 1.2 |
| 70 | 1.05 / 0.93 | 14.0 / 14.1 | 12.9 / 13.1 | 11.9 / 12.2 | 10.8 / 11.3 | 1.0 / 1.1 |
| 90 | 0.99 / 0.99 | 14.0 / 14.0 | 13.0 / 13.0 | 12.0 / 12.0 | 11.1 / 11.1 | 1.0 / 1.0 |

**Rule of thumb for CPP guidance.** To keep the full 2–3 day buffer for a 1–2 day visit (H ≈ 3–4 days):

| Target latitude | Maximum \|roll\| |
|---|---|
| \|β\| ≲ 20° | 13–14.7° |
| \|β\| ≈ 30–50°, near quadrature | 12–13.5° |
| \|β\| ≈ 30–50°, in the worst season | 10.5–12.8° |
| \|β\| ≈ 50–60°, in the worst season | as low as 10° |
| Continuous viewing zone | 11–12°, nearly independent of date |

**[roll_limits_vs_latitude.png](roll_limits_vs_latitude.png)**:

* (a, b) Maximum |roll| vs. |β| for H = 1–5 days, at the worst-case date and at quadrature.
* (c) Slack per degree, 1/|rate|.

![roll limits vs latitude](roll_limits_vs_latitude.png)

**[roll_limits_map.png](roll_limits_map.png)**: maximum |roll| vs. requested mid-visit date (days after solar
conjunction) and |β|, for H = 2 and 4 days.

![roll limits map](roll_limits_map.png)

### Calculator for a specific target

`tools/plan_roll_limits.py` computes the allowed offsets for one target and visit directly with `roman_pointing`,
using the actual solar ephemeris:

```
PYTHONPATH=src python tools/plan_roll_limits.py --ra 108.949 --dec 67.783 \
    --start 2027-03-01T00:00:00 --duration 2 --early 3 --late 3
```

```
Target beta = 45.00 deg, Sun angle 107.5-112.8 deg, dPA_V3/dt at mid-visit = -0.806 deg/day
Allowed requested roll offsets (applyRollAngle sense): -11.83 to +11.72 deg; symmetric limit +/-11.72 deg
```

With `--early 0 --late 3` (start can only slip later), the positive roll is limited only by the visit length: −11.83° to
+14.19°. The calculator warns if any part of the span falls outside the field of regard. It agrees with the analytic
limits to within 0.03°. Add `--plot` (and optionally `--name`) to write a figure like the worked example for the
same inputs; `-o` sets the output directory (default: current directory).

## 5. Fixed-PA observations (e.g., slit spectroscopy)

Here the roll is chosen at execution time so that PA_V3 lands in a required sky range, rather than holding a fixed
offset. Definitions:

* **Constraint range [PA_lo, PA_hi]:** the sky PA_V3 values the program accepts, of width W_c = PA_hi − PA_lo.
* **PA_start, PA_end:** the nominal PA_V3 on the earliest and latest possible execution dates. These differ by ΔPA(T),
  where T is the length of the window in which the visit may fall.
* On any date t, roll can reach [PA_nom(t) − 15°, PA_nom(t) + 15°].

The visit can be done on every date in the window only if the constraint range contains [PA_end + 15°, PA_start − 15°]
(for β > 0; reverse the roles of start and end for β < 0). That needs a width of at least

$$ W_\mathrm{c,min} = \max\left(0,\ \Delta\mathrm{PA}(T) - 2\times 15^\circ\right). $$

If ΔPA(T) ≤ 30°, one fixed PA anywhere in [PA_start − 15°, PA_end + 15°] can be reached throughout the window.

All values below use the circular-orbit rate. To be conservative for windows in December–January, add ~3%.

| \|β\| | rate at quadrature (°/d) | max rate in FOR (°/d) | FOR window (d) | ΔPA across window (°) | fixed-PA window, worst / at quadrature (d) |
|---:|---:|---:|---:|---:|---:|
| 5  | 0.09 | 0.13 | 73  | 7   | 73 / 73 |
| 10 | 0.17 | 0.26 | 74  | 15  | 74 / 74 |
| 20 | 0.34 | 0.52 | 79  | 31  | 77 / 77 |
| 30 | 0.49 | 0.75 | 87  | 50  | 53 / 57 |
| 40 | 0.63 | 0.97 | 102 | 75  | 39 / 45 |
| 45 | 0.70 | 1.07 | 114 | 93  | 33 / 42 |
| 50 | 0.76 | 1.15 | 134 | 120 | 29 / 39 |
| 54 | 0.80 | 1.22 | 183 | 180 | 24 / 37 |
| 60 | 0.85 | 1.14 | all year | 360/yr | 26 / 34 |
| 70 | 0.93 | 1.05 | all year | 360/yr | 28 / 32 |
| 90 | 0.99 | 0.99 | all year | 360/yr | 30 / 30 |

Minimum constraint range width W_c,min (°) for a window of T days, with ±15° roll and no extra margin. A dash means
the window is longer than the FOR window at that latitude.

| \|β\| | T=30 worst | T=60 worst | T=90 worst | T=30 quad. | T=60 quad. | T=90 quad. |
|---:|---:|---:|---:|---:|---:|---:|
| ≤ 20 | 0 | 0 | – | 0 | 0 | – |
| 30 | 0 | 3  | – | 0 | 2  | – |
| 40 | 0 | 13 | 35 | 0 | 10 | 34 |
| 45 | 0 | 19 | 41 | 0 | 14 | 39 |
| 50 | 0 | 25 | 48 | 0 | 17 | 44 |
| 54 | 6 | 40 | 71 | 0 | 19 | 47 |
| 60 | 4 | 37 | 67 | 0 | 22 | 51 |
| 70 | 1 | 32 | 62 | 0 | 26 | 55 |
| 90 | 0 | 29 | 59 | 0 | 29 | 59 |

How to read these:

* **Low-latitude targets (|β| ≲ 20°)** keep essentially one orientation for an entire visibility window. Roll alone
  covers it.
* **Targets near |β| ≈ 54°** are the most demanding for orientation-constrained programs. Their worst-case window sits
  where the Sun angle passes near 54° or 126°.
* **Within the continuous viewing zone**, the worst case shifts to the dates of closest Sun approach (Δ ≈ 0° or 180°).
  At the poles, the image rotates at a uniform ≈ 1°/day.
* **Roll margin.** To hold back part of the roll range (e.g., for attitude-dependent constraints or execution margin),
  replace 2 × 15° by 2 × (15° − margin).

## 6. Validation against `roman_pointing`

`validate()` in `tools/field_rotation_rate.py` compares the model with `calcRomanAngles` + `getRomanPositionAngle` for 2027. It uses seven
synthetic targets with β from −80° to +85° at various longitudes, sampled every 0.5 day, and keeps only in-FOR times.

| Check | Result |
|---|---|
| Absolute PA_V3 (θ + η, ephemeris solar longitude) | ≤ 0.0043° |
| Rate, ephemeris solar longitude (tests the geometry) | ≤ 0.002 °/day |
| Rate, circular-orbit model vs. day of year | ≤ 0.036 °/day (≤ 3.4%, from eccentricity) |
| Solar longitude, circular-orbit model vs. ephemeris | −1.91° to +1.92° (equation of center) |
| PA sense vs. astropy `position_angle` of the Sun | 0.0000° difference (east of north confirmed) |

![validation](pa_v3_rate_validation.png)

## 7. Assumptions and limitations

* **Roll-limit model (section 4).**
  * CPGS computes each visit's sky PA once, for the planned mid-visit. The PA is held for the whole visit and is not
    recomputed if the visit moves within its buffer.
  * Each visit (roll A or roll B) is treated separately with its own planned mid-time.
  * The guideline tables assume a symmetric buffer ±U; the calculator accepts separate early and late shifts.
  * The visit span plus buffer must lie inside the FOR.
  * Any additional attitude margin CPGS or the observatory reserves inside ±15° is not included; subtract it from 15°.

* **Observatory position.** It is taken at the Sun–Earth L2 point, as in `getL2Positions`. Roman's real halo orbit
  probably displaces the Sun direction by a few tenths of a degree with a ~6-month period. That would change the rate by
  ≲ 0.01°/day, which matters only in relative terms for very-low-|β| targets, whose rates are ~0.1°/day. This is an
  order-of-magnitude estimate; it has not been checked against the Horizons Roman ephemeris used in `drive_cpgs.py`.
* **Target and attitude limits.** Stellar parallax and proper motion are negligible. The FOR is taken as 54° ≤ ε ≤ 126°.
  The full ±15° roll is assumed available everywhere in the FOR.
* **Latitude frame.** β is the barycentric mean-ecliptic (J2000) latitude. In astropy, for a `SkyCoord` named `coord`,
  use `coord.transform_to(BarycentricMeanEcliptic).lat`.
