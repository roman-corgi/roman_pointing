# Seasonal drift of the EXCAM image orientation (PA_Z) versus ecliptic latitude

Analytical rate of change of the nominal-roll position angle of the Roman observatory +Z axis
(PA_Z, from `getRomanPositionAngle`) and of EXCAM +Y (`getEXCAMPositionAngle`), as a function of
target ecliptic latitude and day of year, and what it implies for PA_Z specifications that must
remain schedulable with the ±15° off-nominal roll allowance over an uncertain execution window.

Figures, tables, and the validation record are produced by
[`tools/pa_rate_vs_latitude.py`](../../tools/pa_rate_vs_latitude.py).

## Key result

For a target at ecliptic longitude λ\* and latitude β, with the Sun at apparent ecliptic
longitude λ☉(t) as seen from the observatory, Δλ = λ☉ − λ\*, and Sun–target angle ε
(cos ε = cos β cos Δλ):

$$
\boxed{\;\frac{d\,\mathrm{PA}_Z}{dt} \;=\; -\,\dot\lambda_\odot\,\frac{\sin\beta}{\sin^2\epsilon}
\;=\; -\,\dot\lambda_\odot\,\frac{\sin\beta}{1-\cos^2\beta\,\cos^2\Delta\lambda}\;}
$$

with λ̇☉ ≈ 0.9856°/day (0.953–1.019°/day through the year), giving the rate in °/day.

- **EXCAM +Y has the same rate** (PA_EXCAM,+Y = PA_Z + 150°), and an applied roll angle adds a
  constant offset, so the seasonal rate is independent of roll.
- **Sign:** PA is measured from celestial north through east. PA_Z **decreases** with time for
  northern-ecliptic targets (β > 0) and **increases** for southern targets.
- **Latitude sets the scale, Sun angle sets the seasonal modulation.** The rate is slowest at
  quadrature (ε = 90°), where it equals λ̇☉ sin|β|, and grows as 1/sin²ε toward the 54°/126°
  Sun-angle limits.
- **The fastest observable drift is 1.22°/day**, at |β| = 54° on the keepout boundary.
  Targets near the ecliptic barely rotate within a season; targets near the ecliptic poles rotate
  at ≈ 1°/day all year.

## Geometry and conventions in `roman_pointing`

`calcRomanAngles` builds the body frame by pointing b₁ (boresight) at the target and keeping b₃
(+Z) in the plane containing the target and the Sun, on the Sun side. So at nominal roll, +Z is
the direction of the Sun's projection onto the sky plane at the target:

$$
\hat b_3 = \frac{\hat s - (\hat s\cdot\hat n)\,\hat n}{\lVert \hat s - (\hat s\cdot\hat n)\,\hat n\rVert},
\qquad \lVert \hat s - (\hat s\cdot\hat n)\,\hat n\rVert = \sin\epsilon ,
$$

where n̂ is the target direction and ŝ the Sun direction from the observatory.
`getRomanPositionAngle` returns the angle of b₃ from projected celestial north about b₁.
`getEXCAMPositionAngle` adds 150°. `applyRollAngle` rotates about b₁, so a roll ρ shifts PA_Z by
a constant (see [Validation](#validation) for the sign).

## Derivation

Work in ecliptic coordinates. Let n̂ = (cos β cos λ\*, cos β sin λ\*, sin β) and, with the Sun in
the ecliptic plane, ŝ = (cos λ☉, sin λ☉, 0). The local unit vectors toward increasing ecliptic
longitude (east) and latitude (ecliptic north) at the target are

$$
\hat e = (-\sin\lambda_*,\ \cos\lambda_*,\ 0), \qquad
\hat q = (-\sin\beta\cos\lambda_*,\ -\sin\beta\sin\lambda_*,\ \cos\beta).
$$

The projected Sun vector p = ŝ − (ŝ·n̂)n̂ has sky-plane components

$$
E = \hat s\cdot\hat e = \sin\Delta\lambda, \qquad N = \hat s\cdot\hat q = -\sin\beta\cos\Delta\lambda,
$$

so its length is r = √(E² + N²) = √(1 − cos²β cos²Δλ) = sin ε. Measured from ecliptic north
through east, the nominal +Z position angle is

$$
\theta = \operatorname{atan2}(E,\,N), \qquad \mathrm{PA}_Z = \theta + \gamma ,
$$

where γ is the (constant, for a fixed target) position angle of ecliptic north relative to
celestial north. γ drops out of the derivative:

$$
\frac{d\theta}{dt} = \frac{N\dot E - E\dot N}{E^2+N^2}
= \frac{-\sin\beta\cos^2\!\Delta\lambda - \sin\beta\sin^2\!\Delta\lambda}{\sin^2\epsilon}\,\dot\lambda_\odot
= -\dot\lambda_\odot\,\frac{\sin\beta}{\sin^2\epsilon}.
$$

**Equal areas in equal times.** The numerator N Ė − E Ṅ is twice the rate at which p sweeps
area in the sky plane, and it is constant: ½ λ̇☉ sin|β|. In polar form the same rate is
½ r² |dθ/dt|. When the projected Sun is close to the target (small r = sin ε), the +Z axis must
rotate faster to sweep the same area. This is the origin of the 1/sin²ε factor.

**General form.** For an arbitrary Sun direction ŝ(t) (eccentric orbit, out-of-ecliptic halo
motion), the same derivation gives

$$
\frac{d\,\mathrm{PA}_Z}{dt} = -\,\frac{\hat n\cdot(\hat s\times\dot{\hat s})}{1-(\hat n\cdot\hat s)^2}
\qquad[\text{rad per unit time}],
$$

which reduces to the boxed result when ŝ × ṡ̂ = λ̇☉ ẑ_ecl.

## Day-of-year dependence

The day-of-year dependence enters only through Δλ(t) = λ☉(t) − λ\* and λ̇☉(t). With the mean
solar motion n☉ = 0.98565°/day and the Sun at λ☉ = 0° near the March equinox (DOY ≈ 80),

$$
\Delta\lambda(\mathrm{DOY}) \approx 0.98565^\circ\,(\mathrm{DOY}-80) - \lambda_*,
\qquad
\frac{d\,\mathrm{PA}_Z}{dt}(\mathrm{DOY}) \approx
\frac{-0.98565\,\sin\beta}{1-\cos^2\beta\,\cos^2[\,0.98565\,(\mathrm{DOY}-80)-\lambda_*\,]}
\ \ ^\circ/\text{day}.
$$

The script's `sun_longitude()` uses the Astronomical Almanac low-precision solar formula, which
includes orbital eccentricity (λ̇☉ is ≈ 3.3% faster near perihelion in early January).

Consequences:

- **The target longitude only shifts the calendar.** The rate curve repeats every half year
  (Δλ → Δλ + 180° leaves it unchanged). The slowest drift is at the two quadratures,
  DOY ≈ 80 + (λ\* ± 90°)/0.9856 (mod 365). The fastest is toward solar conjunction or opposition
  in longitude, DOY ≈ 80 + λ\*/0.9856 and half a year later.
- **Observing seasons.** For |β| < 54° the target is observable during two seasons per year, each
  centered on a quadrature and lasting (180° − 2 arccos[cos 54°/cos β])/0.9856 days (73 days at
  β = 0°). For |β| ≥ 54° it is observable all year.

![Rate and Sun angle through a season](fig1_rate_vs_season.png)

*Figure 1. |dPA_Z/dt| (top) and Sun angle (bottom) versus days from quadrature for
|β| = 0°–90°. Solid segments are observable.*

![Signed rate map](fig2_rate_map.png)

*Figure 2. Signed rate versus target latitude and Sun–target longitude difference (equivalently
day of year, offset by the target's longitude). The hatched region is the Sun-angle keepout.*

![Example targets in 2027](fig3_targets_2027.png)

*Figure 3. Nominal EXCAM +Y position angle and its rate through 2027 for example targets,
computed with the full `roman_pointing` solver (colored). The black dashed lines show the
analytical formula with the day-of-year solar model.*

## Variation with ecliptic latitude

Over the observable part of the year (54° < ε < 126°, and ε can never be smaller than |β|):

$$
\left|\frac{d\,\mathrm{PA}_Z}{dt}\right|_{\min} = \dot\lambda_\odot\sin|\beta| \quad(\text{quadrature}),
\qquad
\left|\frac{d\,\mathrm{PA}_Z}{dt}\right|_{\max} =
\begin{cases}
\dot\lambda_\odot\,\sin|\beta|/\sin^2 54^\circ, & |\beta| \le 54^\circ \ (\text{at the keepout edge})\\[2pt]
\dot\lambda_\odot/\sin|\beta|, & |\beta| \ge 54^\circ \ (\text{at } \Delta\lambda = 0^\circ, 180^\circ)
\end{cases}
$$

- **Ecliptic (β ≈ 0):** the rate is ≈ 0 throughout each season; PA_Z is essentially fixed, but it
  flips by 180° between the two seasons.
- **Intermediate latitudes:** the rate varies within a season by a factor of 1/sin²54° = 1.53
  between quadrature and the season edges.
- **|β| = 54°:** largest observable rate, 0.9856/sin 54° = 1.218°/day (1.245°/day at perihelion
  speed).
- **Ecliptic poles:** the projected Sun path is a circle and PA_Z rotates uniformly at λ̇☉ all
  year (≈ 360°/yr).

## From rate to scheduling margin

Over an execution window [t₁, t₂] within one observing season, the nominal PA_Z changes by
D = |θ(t₂) − θ(t₁)|. This has a closed form, the angle between the projected Sun vectors at the
two ends:

$$
D = \left|\operatorname{atan2}\!\big(\sin\beta\,\sin(\Delta\lambda_2-\Delta\lambda_1),\
\sin\Delta\lambda_1\sin\Delta\lambda_2+\sin^2\beta\cos\Delta\lambda_1\cos\Delta\lambda_2\big)\right| .
$$

For a window of full width W centered on quadrature, it reduces to
D = 2 arctan[sin|β| tan(n☉W/2)].

With roll ρ ∈ [−15°, +15°] available on each date, a single PA_Z specification P is reachable on
every date of the window if and only if

$$
\max_t \mathrm{PA}_{Z,\mathrm{nom}}(t) - 15^\circ \;\le\; P \;\le\; \min_t \mathrm{PA}_{Z,\mathrm{nom}}(t) + 15^\circ
\quad\Longleftrightarrow\quad D \le 30^\circ ,
$$

and the spare roll left at the worst date is 15° − D/2 when P is centered on the drift range.
To keep a roll reserve m (or allow a science PA tolerance to be spent elsewhere), require
D ≤ 30° − 2m. Conversely, if the science case accepts any PA within ±T of a goal, the condition
relaxes to D ≤ 30° + 2T.

Panels (b) and (c) of Figure 4, and the table below, give D and the longest window that satisfies
it for two placements of the window. "Quadrature-centered" is the most favorable placement.
"Worst case" places the window where drift is fastest within one season (against a keepout edge
for |β| < 54°, centered on Δλ = 0° or 180° otherwise) and uses perihelion solar speed, so it
bounds any window of that length that lies inside one observing season.

![Latitude envelope and scheduling margins](fig4_latitude_envelope.png)

*Figure 4. (a) Range of |dPA_Z/dt| over observable dates. (b) PA_Z drift over 7–56-day windows.
(c) Longest window over which a common fixed PA_Z is reachable, for drift budgets of 30° (no
spare roll), 20° (±5° spare), and 10° (±10° spare).*

TABLE_PLACEHOLDER

D is the nominal PA_Z change over the window (quadrature-centered / worst case). "Longest window"
is the longest execution window for which one PA_Z value is reachable on every date with ±15° roll
(D ≤ 30°), or with ±5° roll reserved (D ≤ 20°). For |β| < 54° a window cannot exceed the season
length. Values are also in [`pa_rate_summary.csv`](pa_rate_summary.csv).

### Suggested guidelines for PA_Z specifications

1. **Low latitude (|β| ≲ 15°):** drift is ≲ 0.3°/day even at the season edges. A fixed PA_Z with
   most of the ±15° roll range to spare holds for an entire season. Each season has its own PA_Z
   (they differ by ≈ 180°), so specify the season.
GUIDELINES_PLACEHOLDER
4. **Specific targets:** evaluate the exact D for the actual candidate window with the closed-form
   expression above, or with the solver (`calcRomanAngles` + `getEXCAMPositionAngle`), then apply
   D ≤ 30° − 2m.

## Diagram: dependence on Sun angle and ecliptic latitude

![Geometry diagram](fig5_geometry_diagram.png)

*Figure 5. Top: the Sun's projection on the sky around the target for β = 20°, 54°, and 80°.
Over a year it traces an ellipse with semi-axes 1 (along ecliptic east–west) and sin β (along
ecliptic north–south). Nominal +Z is the ray from the target to the projected Sun, and the
Sun-angle keepout is the hatched disk r < sin 54°. Equal time steps (dots) sweep equal areas,
so the rays fan out fastest where the projected Sun is closest to the target. Bottom: the rate
as a function of Sun angle and |β|. A target moves along a horizontal track through the year,
from ε = |β| to 180° − |β|.*

## Validation

VALIDATION_PLACEHOLDER

## Assumptions and limitations

- Stationary, distant target (parallax and proper motion contribute < 10⁻⁴ °/day for nearby stars).
- Nominal attitude as defined by `calcRomanAngles` (+Z toward the projected Sun). The ±15° roll
  allowance is assumed to be available at all pitch angles and dates.
- Sun-angle limits of 54°–126° (the `compute_keepout` defaults). No other scheduling constraints.
- The repository places the observatory at L2 on the Sun–Earth line. A real halo orbit moves the
  Sun direction by a few tenths of a degree, including out of the ecliptic. This produces small,
  slowly varying PA_Z changes for near-ecliptic targets that the planar formula omits. The general
  vector form above captures them if ŝ(t) is taken from the actual ephemeris.

## Reproducing

```bash
python tools/pa_rate_vs_latitude.py            # figures, tables, and solver validation (offline)
python tools/pa_rate_vs_latitude.py --no-validate   # analytical figures only (skips Figure 3)
```
