# Rate of change of the Roman image orientation (PA_Z) vs. time of year and ecliptic latitude

Analytic model, scheduling guidelines, and validation against `roman_pointing`.
Figures and numbers come from [`tools/pa_z_rate_analytic.py`](../../tools/pa_z_rate_analytic.py):

```
PYTHONPATH=src python tools/pa_z_rate_analytic.py -o docs/pa_z_rate_analytic
```

## Summary

For a target at ecliptic latitude β, the nominal-roll position angle PA_Z (and PA_EXCAM_Y = PA_Z + 150°)
changes at

$$
\boxed{\frac{d\,\mathrm{PA_Z}}{dt} = -\,\dot\lambda_\odot\,\frac{\sin\beta}{\sin^2\varepsilon}
= -\,\dot\lambda_\odot\,\frac{\sin\beta}{1-\cos^2\beta\,\cos^2(\lambda_\odot-\lambda)}}
\qquad \dot\lambda_\odot \approx 0.9856^\circ/\mathrm{day}
$$

where ε is the Sun–target angle, λ is the target's ecliptic longitude, and λ☉ is the Sun's ecliptic longitude.
PA is measured east of north.

Practical consequences, assuming the 54°–126° Sun-angle field of regard (FOR) and ±15° off-nominal roll:

* **The sign is fixed by the hemisphere.** PA_Z decreases with time for northern-ecliptic targets and increases for
  southern ones, all year. It never reverses.
* **The rate depends only on β and ε.** Right ascension, declination, and the equatorial-vs-ecliptic north offset
  set the absolute PA_Z but not its rate.
* **Within the FOR the rate is bounded:**
  * For |β| < 54°: 0.986 |sin β| ≤ |rate| ≤ 1.51 |sin β| deg/day.
    * The minimum is at quadrature (ε = 90°).
    * The maximum is at the FOR edges (ε = 54° or 126°).
  * For |β| ≥ 54° (continuous viewing): 0.986 |sin β| ≤ |rate| ≤ 0.986 / |sin β| deg/day. PA_Z turns a full 360° per year.
  * The largest rate anywhere in the FOR is **1.22°/day**. It occurs at |β| ≈ 54° at the FOR edge (about 1.26°/day near
    perihelion in early January).
* **How long a single fixed PA_Z stays reachable with ±15° roll:**

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

  $$ |A|,\ |B| \le 15^\circ - |\mathrm{rate}| \times (D/2 + U) $$

  Each degree of |roll| given up buys 1/|rate| days of start-time slack, about 0.8–1.6 days per degree for
  |β| ≳ 40°. The script also includes a calculator for a specific target, start date, duration and buffer.
* **Fixed-PA observations** (e.g., slit spectroscopy; [section 5](#5-fixed-pa-observations-eg-slit-spectroscopy)).
  The roll is adjusted at execution time so that PA_Z stays fixed; ±15° of roll then covers about 30°/|rate| days.

## 1. Geometry and conventions

In `calcRomanAngles`, the nominal attitude starts with the body +Z axis (b₃) on the Sun. Yaw and pitch then put +X (b₁)
on the target. After the pitch rotation, +Z is the Sun direction projected onto the sky plane at the target.
`getRomanPositionAngle` measures +Z from celestial north about the boresight. Comparing it with astropy's
`position_angle` confirms the sense is the standard **east of north**.

So **PA_Z is the position angle of the Sun as seen from the target**. EXCAM differs only by a constant:
`getEXCAMPositionAngle` returns PA_EXCAM_Y = PA_Z + 150°. Roll about +X (`applyRollAngle`) shifts PA_Z one-for-one;
+10° of roll gives −10.000° in PA_Z. Neither offset changes the rate, so everything below applies equally to PA_Z,
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
so PA_Z = θ + η:

$$ \frac{d\,\mathrm{PA_Z}}{dt} = -\dot\lambda_\odot\,\frac{\sin\beta}{\sin^2\varepsilon}. $$

If the absolute PA is needed: with obliquity ι = 23.44°,
η = −atan2(sin ι cos λ, cos β cos ι − sin β sin ι sin λ). This reproduces `getRomanPositionAngle` to < 0.005°.

**Geometric reading** ([diagram](pa_z_rate_geometry_diagram.png), panel a):

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

$$ \lambda_\odot(d) \approx 280.46^\circ + 0.98565^\circ\,(d-1), \qquad d = \text{day of year} $$

$$ \frac{d\,\mathrm{PA_Z}}{dt}(d;\lambda,\beta) = -0.98565^\circ\,\mathrm{day^{-1}}\;
\frac{\sin\beta}{1-\cos^2\beta\,\cos^2\!\big(\lambda_\odot(d)-\lambda\big)} $$

* **Timing by λ.** Target longitude only sets when the pattern occurs. Solar conjunction (Δ = 0) falls on day
  d ≈ 1 + (λ − 280.46°)/0.98565 (mod 365.25). The pattern repeats every half year, because the rate is symmetric in
  Δ → 180° − Δ and Δ → Δ + 180°.
* **Eccentricity.** It changes λ̇☉ by ±3.4%: 1.019°/day in early January, 0.953°/day in early July. It also shifts
  λ☉ by up to ~2° (~2 days). These are the only significant errors of the circular model.

### Closed-form results for one FOR window (|β| < 54°)

The target enters the FOR at ε = 54° and exits at ε = 126° (or the reverse). In terms of Δ, the window runs from
Δ₁ = arccos(cos 54° / cos β) to 180° − Δ₁:

$$ \text{duration} = \frac{180^\circ - 2\Delta_1}{\dot\lambda_\odot}, \qquad
|\Delta\mathrm{PA_Z}|_\text{window} = 2\arctan\!\left(\frac{|\sin\beta|\cos 54^\circ}{\sqrt{\cos^2\beta-\cos^2 54^\circ}}\right). $$

For any interval, ΔPA_Z = θ(Δ(t₂)) − θ(Δ(t₁)) exactly, with θ = atan2(sin Δ, −sin β cos Δ).

## 3. Figures

**[pa_z_rate_geometry_diagram.png](pa_z_rate_geometry_diagram.png)**: the requested diagram.

* (a) 3D geometry with Roman at the center of the celestial sphere, using the example β = 45°, Δ = 60°, ε = 69°.
  * Lines of sight run from Roman to the target and to the Sun. ε is the shaded angle between them at Roman.
  * β is measured along the target's meridian, and Δ along the ecliptic.
  * The great circle from the target to the Sun sets +Z on the sky, and θ is +Z measured from ecliptic north.
  * At the Sun, its motion along the ecliptic is shown together with the part that crosses the target–Sun arc.
* (b) |rate| as a function of Sun angle ε and ecliptic latitude, with the FOR marked. Each target moves along a
  horizontal line between ε = |β| and 180° − |β| every half year.
* (c) The +Z direction every 5 days for 60 days around quadrature at β = 10°, 35°, and 70°, against the ±15° roll wedge.

![diagram](pa_z_rate_geometry_diagram.png)

**[pa_z_rate_vs_time.png](pa_z_rate_vs_time.png)**: |rate| through the year vs. days after solar conjunction for
several β, with solid lines in the FOR. The right panel shows accumulated PA_Z change relative to quadrature, against
the ±15° roll band.

![vs time](pa_z_rate_vs_time.png)

**[pa_z_rate_map.png](pa_z_rate_map.png)**: |rate| vs. days after conjunction and |β| within the FOR. Each contour is
labeled with its rate and the number of days a fixed PA_Z stays within ±15° at that rate (30°/rate).

![map](pa_z_rate_map.png)

**[pa_z_rate_vs_latitude.png](pa_z_rate_vs_latitude.png)**: summary vs. |β|:

* (a) Minimum, mean, and maximum in-FOR rate.
* (b) FOR window length.
* (c) PA_Z rotation across a window.
* (d) Longest window over which a single PA_Z is reachable with ±15° roll.
* (e, f) For fixed-PA observations ([section 5](#5-fixed-pa-observations-eg-slit-spectroscopy)): the minimum
  constraint range width for 14-, 30-, 60-, and 90-day windows. Panel (e) uses the worst placement in the FOR; panel (f)
  centres the window on quadrature.

![vs latitude](pa_z_rate_vs_latitude.png)

## 4. Maximum roll offsets when CPGS fixes the PA at mid-visit

### The problem

The observer requests a start date and two roll offsets, A and B. For each visit, CPGS converts the request into an
absolute sky PA computed for the planned middle of the visit:

$$ P_A = \mathrm{PA_{nom}}(t_\mathrm{mid}) + A $$

The spacecraft then holds that PA. The nominal PA keeps rotating, so the roll actually needed at execution time t is

$$ A_\mathrm{eff}(t) = A - \big[\mathrm{PA_{nom}}(t) - \mathrm{PA_{nom}}(t_\mathrm{mid})\big] $$

That needed roll must stay within ±15° over the whole interval in which the visit might execute:

* **The visit itself** reaches D/2 on either side of mid-visit, where D is the visit duration.
* **Any start shift** the schedulers may need adds U days on top of that, for example within the 2–3 day buffer of
  the visibility windows.

Long visits, rolls requested near ±15°, and fast-rotating targets push A_eff past the limit. The visit then has only a
narrow range of legal start times.

### Guideline formula

Let H = D/2 + U be the farthest the visit can execute from the planned mid-visit. To first order:

$$ \boxed{|A|,\ |B| \;\le\; 15^\circ - |\mathrm{rate}| \times \left(\tfrac{D}{2} + U\right)} $$

Inverting it, for given rolls the start time can move by up to

$$ U_\mathrm{max} = \frac{15^\circ - \max(|A|, |B|)}{|\mathrm{rate}|} - \frac{D}{2}. $$

Here |rate| = λ̇☉ |sin β| / sin²ε from section 2, evaluated at the requested date. Use the largest value over the
span for a conservative limit, and add ~3% in December–January. The script evaluates the exact version with θ(Δ).

* **Each degree of |roll| given up buys 1/|rate| days of start-time slack**, shared between early and late. That is
  about 1 day per degree at the ecliptic poles, 0.8 days per degree in the worst case near |β| ≈ 54°, and several
  days per degree near the ecliptic.
* **The two rolls limit different sides.** In the PA sense, a northern-ecliptic target (β > 0) has PA_Z decreasing.
  Running early pushes a negative offset further negative, and running late pushes a positive offset further positive.
  So for β > 0:
  * the negative-offset roll limits how early the visit can run, and
  * the positive-offset roll limits how late it can run.

  The roles swap for β < 0. If the buffer is one-sided, only one roll needs the reduced limit; the other can go to
  15° − |rate| × D/2.
* **Sign convention.** Offsets here are in the PA sense: +A rotates PA_Z by +A. In `roman_pointing`, a roll of +r passed
  to `applyRollAngle` changes PA_Z by −r. The magnitude limits are the same in either convention.

### Worked example

([roll_limits_example.png](roll_limits_example.png).) Take β = 45°, mid-visit 60 days after solar conjunction
(|rate| ≈ 0.80°/day), D = 2 days, and requested rolls of ±13°.

* **Legal start times.** The needed roll reaches −15° at 2.5 days before mid-visit and +15° at 2.5 days after. The visit
  start can therefore move by only ±1.5 days, against a ±3 day buffer.
* **What fits the full buffer.** Holding the full ±3 day buffer (H = 4 days) requires |roll| ≤ 11.7°.
* **Shorter visit.** A 0.5-day visit with the same ±3 day buffer allows |roll| ≤ 12.4°.

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

The script can compute the allowed offsets for one target and visit directly with `roman_pointing`, using the actual
solar ephemeris:

```
PYTHONPATH=src python tools/pa_z_rate_analytic.py --ra 108.949 --dec 67.783 \
    --start 2027-03-01T00:00:00 --duration 2 --early 3 --late 3
```

```
Target beta = 45.00 deg, Sun angle 107.5-112.8 deg, dPA_Z/dt at mid-visit = -0.806 deg/day
Allowed requested roll offsets (PA sense): -11.72 to +11.83 deg; symmetric limit +/-11.72 deg
```

With `--early 0 --late 3` (start can only slip later), the negative roll is limited only by the visit length: −14.19° to
+11.83°. The calculator warns if any part of the span falls outside the field of regard. It agrees with the analytic
limits to within 0.03°.

## 5. Fixed-PA observations (e.g., slit spectroscopy)

Here the roll is chosen at execution time so that PA_Z lands in a required sky range, rather than holding a fixed
offset. Definitions:

* **Constraint range [PA_lo, PA_hi]:** the sky PA_Z values the program accepts, of width W_c = PA_hi − PA_lo.
* **PA_start, PA_end:** the nominal PA_Z on the earliest and latest possible execution dates. These differ by ΔPA(T),
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

`validate()` in the script compares the model with `calcRomanAngles` + `getRomanPositionAngle` for 2027. It uses seven
synthetic targets with β from −80° to +85° at various longitudes, sampled every 0.5 day, and keeps only in-FOR times.

| Check | Result |
|---|---|
| Absolute PA_Z (θ + η, ephemeris solar longitude) | ≤ 0.0043° |
| Rate, ephemeris solar longitude (tests the geometry) | ≤ 0.002 °/day |
| Rate, circular-orbit model vs. day of year | ≤ 0.037 °/day (≤ 3.4%, from eccentricity) |
| PA sense vs. astropy `position_angle` of the Sun | 0.0000° difference (east of north confirmed) |

![validation](pa_z_rate_validation.png)

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
